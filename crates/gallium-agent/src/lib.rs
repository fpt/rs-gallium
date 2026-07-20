//! gallium-agent: the local-agent core.
//!
//! ReAct loop, tool registry, MCP client/server, and the LLM providers
//! (OpenAI, in-process llama.cpp via the `local` feature, and the native candle
//! `gallium` backend). Also hosts the JSON-RPC **app-server** (`appserver`) that
//! exposes the agent as a whole-turn ACP backend — the role formerly served by
//! `kessel-cli app-server`.
//!
//! This crate is headless: frontends (voice, VM host, etc.) drive it over the
//! app-server protocol rather than linking it in-process.

pub mod appserver;
pub mod github;
mod harmony;
// Shared Gemma native tool-call parsing, used by both local backends.
#[cfg(any(feature = "local", feature = "gallium"))]
pub mod gemma;
mod llm;
#[cfg(feature = "gallium")]
pub mod llm_gallium;
#[cfg(feature = "local")]
pub mod llm_local;
#[cfg(feature = "gallium")]
pub mod protocol;
pub mod mcp;
pub mod mcp_client;
pub mod mcp_client_http;
pub mod mcp_server;
pub mod mcp_server_http;
mod memory;
pub mod model_downloader;
pub mod react;
pub mod situation;
pub mod skill;
mod state_updater;
pub mod tool;

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tool::ToolAccess;

pub use harmony::HarmonyTemplate;
pub use llm::{create_provider, ChatMessage, ChatRole, TokenUsage};
pub use memory::ConversationMemory;
pub use state_updater::{BackchannelDetector, RuleBasedBackchannelDetector};

/// JSON Schema for keyword extraction
fn get_keyword_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Your natural language response to the user"
            },
            "keywords": {
                "type": "array",
                "description": "Important keywords from this conversation for speech recognition context (proper nouns, technical terms, domain-specific words)",
                "items": {
                    "type": "string"
                },
                "maxItems": 10
            }
        },
        "required": ["response", "keywords"],
        "additionalProperties": false
    })
}

/// Parse structured JSON response containing both response text and keywords
fn parse_structured_response(json_str: &str) -> Result<(String, Vec<String>), AgentError> {
    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| AgentError::ParseError(format!("Failed to parse JSON: {}", e)))?;

    let response = parsed["response"]
        .as_str()
        .ok_or_else(|| AgentError::ParseError("Missing 'response' field".to_string()))?
        .to_string();

    let keywords = parsed["keywords"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Ok((response, keywords))
}

/// Configuration for an external MCP server to spawn and connect to.
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
    /// If set, connect over Streamable HTTP to this URL instead of spawning
    /// `command`. (stdio uses command/args; HTTP uses url.)
    pub url: Option<String>,
}

/// Connect each configured MCP server and register its tools into `registry`.
/// A `url` selects the Streamable HTTP transport; otherwise `command`/`args` are
/// spawned (stdio). A server that fails to connect is logged and skipped, so one
/// bad entry does not take down the agent.
///
/// Shared by `agent_new` and the app-server's `thread/start`, so both transports
/// stay reachable from every frontend.
pub(crate) fn register_mcp_servers(registry: &mut tool::ToolRegistry, servers: &[McpServerConfig]) {
    for server_cfg in servers {
        let http_url = server_cfg.url.as_deref().filter(|u| !u.is_empty());
        let result = match http_url {
            Some(url) => mcp_client_http::McpHttpClient::connect(url).map(|c| c.tool_handlers()),
            None => {
                let args_ref: Vec<&str> = server_cfg.args.iter().map(|s| s.as_str()).collect();
                mcp_client::McpClient::connect(&server_cfg.command, &args_ref)
                    .map(|c| c.tool_handlers())
            }
        };
        match result {
            Ok(handlers) => {
                for handler in handlers {
                    registry.register(handler);
                }
            }
            Err(e) => {
                let target = http_url.unwrap_or(server_cfg.command.as_str());
                tracing::warn!("Failed to connect MCP server '{}': {}", target, e);
            }
        }
    }
}

/// Configuration for the agent
pub struct AgentConfig {
    pub model_path: Option<String>,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub use_harmony_template: bool,
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    /// Model context window size in tokens (used for compaction triggering).
    pub context_window: u32,
    pub language: Option<String>,
    pub working_dir: Option<String>,
    pub reasoning_effort: Option<String>,
    /// Local inference backend: "llamacpp" (default) or "gallium". Overridable at
    /// runtime by the `INFERENCE_ENGINE` env var. `None` auto-detects from
    /// `model_path`.
    pub inference_engine: Option<String>,
    pub mcp_servers: Vec<McpServerConfig>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-5.6-luna".to_string(),
            api_key: None,
            use_harmony_template: true,
            temperature: Some(0.7),
            max_tokens: 2048,
            context_window: 128_000,
            language: Some("en".to_string()),
            working_dir: None,
            reasoning_effort: None,
            inference_engine: None,
            mcp_servers: Vec::new(),
        }
    }
}

/// Response from the agent
pub struct AgentResponse {
    pub content: String,
    pub role: String,
    pub is_final: bool,
    pub keywords: Option<Vec<String>>,
    pub reasoning: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub context_percent: f32,
    /// Self-paced cadence hint from `observe()` (seconds until next check), set
    /// when the agent calls the `suggest_next_check` tool. `None` for `step()`.
    pub suggested_next_check_seconds: Option<u32>,
}

/// Error types for the agent
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Main agent struct
pub struct Agent {
    config: AgentConfig,
    client: Box<dyn llm::LlmProvider>,
    memory: Arc<Mutex<ConversationMemory>>,
    backchannel_detector: Box<dyn BackchannelDetector>,
    system_prompt: Arc<Mutex<Option<String>>>,
    tool_registry: tool::ToolRegistry,
    skill_registry: Arc<skill::SkillRegistry>,
    situation: Arc<situation::SituationMessages>,
    last_input_tokens: AtomicU64,
    /// Self-paced cadence hint set by the `suggest_next_check` tool (0 = unset),
    /// read by `observe()`. Shared with the tool handler.
    next_check: Arc<AtomicU64>,
}

/// Top-level constructor.
pub fn agent_new(config: AgentConfig) -> Result<Arc<Agent>, AgentError> {
    // Initialize tracing (only once)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    // Create LLM provider
    let client = create_provider(
        config.model_path.clone(),
        config.base_url.clone(),
        config.model.clone(),
        config.api_key.clone(),
        config.temperature,
        config.max_tokens,
        config.reasoning_effort.clone(),
        config.inference_engine.clone(),
    )
    .map_err(|e| AgentError::ConfigError(e.to_string()))?;

    // Create tool registry with built-in tools
    let working_dir = config
        .working_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    tracing::info!("Tool working directory: {}", working_dir.display());
    let skill_registry = Arc::new(skill::SkillRegistry::new());

    let situation = Arc::new(situation::SituationMessages::default());

    let mut tool_registry = tool::create_default_registry(
        working_dir,
        skill_registry.clone(),
        situation.clone(),
    );

    register_mcp_servers(&mut tool_registry, &config.mcp_servers);

    // Self-pacing hint tool for the ambient loop, sharing the next_check cell.
    let next_check = Arc::new(AtomicU64::new(0));
    tool_registry.register(Box::new(tool::SuggestNextCheckTool::new(next_check.clone())));

    // Register GitHub Projects tools when configured (KESSEL_GH_ORG/PROJECT).
    if let Some(gh) = github::GithubClient::from_env() {
        let gh = Arc::new(gh);
        let gh_session = Arc::new(tool::ToolSession::new());
        tool_registry.register(Box::new(github::GithubListTasksTool::new(gh.clone())));
        tool_registry.register(Box::new(github::GithubCreateDraftTool::new(gh.clone(), gh_session.clone())));
        tool_registry.register(Box::new(github::GithubPromoteDraftTool::new(gh.clone(), gh_session.clone())));
        tool_registry.register(Box::new(github::GithubSetStatusTool::new(gh.clone(), gh_session.clone())));
        tool_registry.register(Box::new(github::GithubLogActivityTool::new(gh, gh_session)));
        tracing::info!("Registered GitHub Projects tools");
    }

    Ok(Arc::new(Agent {
        config,
        client,
        memory: Arc::new(Mutex::new(ConversationMemory::new())),
        backchannel_detector: Box::new(RuleBasedBackchannelDetector::new()),
        system_prompt: Arc::new(Mutex::new(None)),
        tool_registry,
        skill_registry,
        situation,
        last_input_tokens: AtomicU64::new(0),
        next_check,
    }))
}

impl Agent {
    /// Process a user input and return the agent's response
    pub fn step(&self, user_input: String) -> Result<AgentResponse, AgentError> {
        // Clear any stale cadence hint; the turn may set a fresh one via the
        // `suggest_next_check` tool (used by the self-paced ambient `/loop`).
        self.next_check.store(0, Ordering::SeqCst);

        let mut memory = self.memory.lock();

        // Compact if last turn approached context window limit (>= 90%)
        self.maybe_compact(&mut memory);

        // Add user message to memory
        memory.add_message(ChatMessage::user(user_input.clone()));

        // Get conversation context
        let mut messages = memory.get_messages();

        // Prepend custom system prompt if set
        let system_prompt = self.system_prompt.lock().clone();
        if let Some(prompt) = system_prompt {
            messages.insert(0, ChatMessage::system(prompt));
        }

        // Inject skill catalog so LLM knows what skills are available
        if let Some(catalog) = self.skill_registry.catalog() {
            messages.push(ChatMessage::system(catalog));
        }

        // Apply Harmony template if enabled
        let formatted_messages = if self.config.use_harmony_template {
            HarmonyTemplate::format_messages(&messages)
        } else {
            messages.clone()
        };

        // Use ReAct loop if provider supports tools and tools are registered
        let (response_text, keywords, reasoning, usage) = if self.client.supports_tools()
            && !self.tool_registry.is_empty()
        {
            let mut react_messages = formatted_messages;
            let (text, reasoning, usage) = react::run(
                self.client.as_ref(),
                &mut react_messages,
                &self.tool_registry,
                None,
            )?;

            (text, Vec::new(), reasoning, usage)
        } else if self.client.supports_structured_output() {
            let schema = get_keyword_schema();
            let json_response = self
                .client
                .chat_with_schema(&formatted_messages, schema, "conversation_response")
                .map_err(|e| AgentError::NetworkError(e.to_string()))?;
            let (text, keywords) = parse_structured_response(&json_response)?;
            (text, keywords, None, TokenUsage::default())
        } else {
            let response = self
                .client
                .chat(&formatted_messages)
                .map_err(|e| AgentError::NetworkError(e.to_string()))?;
            (response, Vec::new(), None, TokenUsage::default())
        };

        // Track token usage for compaction decisions
        self.last_input_tokens.store(usage.input_tokens, Ordering::Relaxed);

        // Add assistant response to memory
        memory.add_message(ChatMessage::assistant(response_text.clone()));

        let context_percent = if self.config.context_window > 0 {
            (usage.input_tokens as f64 / self.config.context_window as f64 * 100.0) as f32
        } else {
            0.0
        };

        let suggested_next_check_seconds = match self.next_check.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n.min(u32::MAX as u64) as u32),
        };

        Ok(AgentResponse {
            content: response_text,
            role: "assistant".to_string(),
            is_final: true,
            keywords: if keywords.is_empty() { None } else { Some(keywords) },
            reasoning,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            context_percent,
            suggested_next_check_seconds,
        })
    }

    /// Process a backchannel event (audio only, no history pollution)
    pub fn process_backchannel(&self, partial_input: String, pause_ms: u64) -> Option<String> {
        if let Some(backchannel_text) = self
            .backchannel_detector
            .should_backchannel(&partial_input, pause_ms)
        {
            let mut memory = self.memory.lock();
            memory.add_backchannel();
            tracing::debug!("Backchannel triggered: '{}'", backchannel_text);
            return Some(backchannel_text);
        }
        None
    }

    /// Reset the conversation memory
    pub fn reset(&self) {
        let mut memory = self.memory.lock();
        memory.clear();
    }

    /// Get the conversation history as JSON string
    pub fn get_conversation_history(&self) -> String {
        let memory = self.memory.lock();
        serde_json::to_string_pretty(&memory.get_messages()).unwrap_or_default()
    }

    /// Set a custom system prompt for the conversation
    pub fn set_system_prompt(&self, prompt: String) {
        let mut system_prompt = self.system_prompt.lock();
        *system_prompt = Some(prompt);
        tracing::info!("System prompt set");
    }

    /// Register a skill with the agent
    pub fn add_skill(&self, name: String, description: String, prompt: String) {
        self.skill_registry.add(name, description, prompt);
    }

    /// Process user input with only a subset of tools enabled
    pub fn step_with_allowed_tools(
        &self,
        user_input: String,
        allowed_tools: Vec<String>,
    ) -> Result<AgentResponse, AgentError> {
        let mut memory = self.memory.lock();

        self.maybe_compact(&mut memory);

        memory.add_message(ChatMessage::user(user_input.clone()));

        let mut messages = memory.get_messages();

        let system_prompt = self.system_prompt.lock().clone();
        if let Some(prompt) = system_prompt {
            messages.insert(0, ChatMessage::system(prompt));
        }

        if let Some(catalog) = self.skill_registry.catalog() {
            messages.push(ChatMessage::system(catalog));
        }

        let formatted_messages = if self.config.use_harmony_template {
            HarmonyTemplate::format_messages(&messages)
        } else {
            messages.clone()
        };

        let filtered = self.tool_registry.filtered(&allowed_tools);
        let (response_text, keywords, reasoning, usage) = if self.client.supports_tools()
            && !filtered.is_empty()
        {
            let mut react_messages = formatted_messages;
            let (text, reasoning, usage) = react::run(
                self.client.as_ref(),
                &mut react_messages,
                &filtered,
                None,
            )?;
            (text, Vec::new(), reasoning, usage)
        } else if self.client.supports_structured_output() {
            let schema = get_keyword_schema();
            let json_response = self
                .client
                .chat_with_schema(&formatted_messages, schema, "conversation_response")
                .map_err(|e| AgentError::NetworkError(e.to_string()))?;
            let (text, keywords) = parse_structured_response(&json_response)?;
            (text, keywords, None, TokenUsage::default())
        } else {
            let response = self
                .client
                .chat(&formatted_messages)
                .map_err(|e| AgentError::NetworkError(e.to_string()))?;
            (response, Vec::new(), None, TokenUsage::default())
        };

        self.last_input_tokens.store(usage.input_tokens, Ordering::Relaxed);

        memory.add_message(ChatMessage::assistant(response_text.clone()));

        let context_percent = if self.config.context_window > 0 {
            (usage.input_tokens as f64 / self.config.context_window as f64 * 100.0) as f32
        } else {
            0.0
        };

        Ok(AgentResponse {
            content: response_text,
            role: "assistant".to_string(),
            is_final: true,
            keywords: if keywords.is_empty() { None } else { Some(keywords) },
            reasoning,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            context_percent,
            suggested_next_check_seconds: None,
        })
    }

    /// Run a one-shot, **non-persisting** turn for ambient/background observation
    /// (the `/loop` ambient mode). Unlike `step`, this does NOT read or write the
    /// conversation memory — it builds an ephemeral message list so periodic
    /// checks don't pollute the chat. Scope it to read-only tools via
    /// `allowed_tools`.
    pub fn observe(
        &self,
        prompt: String,
        allowed_tools: Vec<String>,
    ) -> Result<AgentResponse, AgentError> {
        self.next_check.store(0, Ordering::SeqCst);

        let mut messages: Vec<ChatMessage> = Vec::new();
        if let Some(prompt_s) = self.system_prompt.lock().clone() {
            messages.push(ChatMessage::system(prompt_s));
        }
        if let Some(catalog) = self.skill_registry.catalog() {
            messages.push(ChatMessage::system(catalog));
        }
        messages.push(ChatMessage::user(prompt));

        let formatted = if self.config.use_harmony_template {
            HarmonyTemplate::format_messages(&messages)
        } else {
            messages
        };

        let filtered = self.tool_registry.filtered(&allowed_tools);
        let (response_text, reasoning, usage) = if self.client.supports_tools() && !filtered.is_empty()
        {
            let mut react_messages = formatted;
            react::run(self.client.as_ref(), &mut react_messages, &filtered, None)?
        } else {
            let response = self
                .client
                .chat(&formatted)
                .map_err(|e| AgentError::NetworkError(e.to_string()))?;
            (response, None, TokenUsage::default())
        };

        let suggested_next_check_seconds = match self.next_check.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n.min(u32::MAX as u64) as u32),
        };

        let context_percent = if self.config.context_window > 0 {
            (usage.input_tokens as f64 / self.config.context_window as f64 * 100.0) as f32
        } else {
            0.0
        };

        Ok(AgentResponse {
            content: response_text,
            role: "assistant".to_string(),
            is_final: true,
            keywords: None,
            reasoning,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            context_percent,
            suggested_next_check_seconds,
        })
    }

    /// Push a situation message (e.g. from a watcher/frontend) onto the stack the
    /// `read_situation_messages` tool exposes to the model.
    pub fn push_situation_message(&self, text: String, source: String, session_id: String) {
        self.situation.push(text, source, session_id);
    }

    /// Compact memory if the last turn's input tokens reached >= 90% of context
    /// window. Targets 50% of context window after compaction to leave room.
    fn maybe_compact(&self, memory: &mut ConversationMemory) {
        let last = self.last_input_tokens.load(Ordering::Relaxed);
        if last == 0 {
            return;
        }
        let threshold = (self.config.context_window as f64 * 0.9) as u64;
        if last >= threshold {
            let target = self.config.context_window as usize / 2;
            let dropped = memory.compact(target);
            if dropped > 0 {
                tracing::info!(
                    "Compacted memory: dropped {} messages (last input: {} tokens, window: {})",
                    dropped,
                    last,
                    self.config.context_window,
                );
            }
        }
    }
}
