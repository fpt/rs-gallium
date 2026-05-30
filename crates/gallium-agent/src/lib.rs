//! gallium-agent library.
//!
//! Exposes a UniFFI interface (`agent_new`, `Agent`) for use from Swift.
//! Also re-exports all submodules for the `gallium-agent` binary.

pub mod agent;
pub mod llm;
pub mod mcp;
pub mod mcp_client;
pub mod memory;
pub mod protocol;
pub mod provider;
pub mod react;
pub mod session;
pub mod skill;
pub mod tool;

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

uniffi::include_scaffolding!("agent");

// ============================================================================
// AgentError — shared across all modules via `use crate::AgentError`
// ============================================================================

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

// ============================================================================
// UniFFI types
// ============================================================================

/// Configuration for an MCP stdio server to spawn and connect to.
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
}

/// Configuration for creating a cloud-backed (OpenAI) agent via UniFFI.
pub struct CloudAgentConfig {
    pub base_url: Option<String>,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    pub context_window: u32,
    pub working_dir: Option<String>,
    pub reasoning_effort: Option<String>,
    pub system_prompt: Option<String>,
    pub mcp_servers: Vec<McpServerConfig>,
}

/// Response from a single agent turn.
pub struct AgentResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub context_percent: f32,
}

// ============================================================================
// UniFFI-exposed Agent (thread-safe wrapper around the internal Agent)
// ============================================================================

pub struct Agent(Mutex<agent::Agent>);

/// Constructor function exposed to Swift via UniFFI.
///
/// Creates an OpenAI-backed agent. For local gallium model inference, use the
/// `gallium-agent` binary directly.
pub fn agent_new(config: CloudAgentConfig) -> Result<Arc<Agent>, AgentError> {
    // Initialize tracing once.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

    let base_url = config.base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    let api_key = config.api_key
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| AgentError::ConfigError("api_key required (or set OPENAI_API_KEY)".to_string()))?;

    let client: Box<dyn llm::LlmProvider> = Box::new(llm::OpenAiProvider::new_with_base_url(
        api_key,
        config.model,
        base_url,
        Some(config.temperature.unwrap_or(0.7)),
        config.max_tokens,
        config.reasoning_effort,
    ));

    let working_dir = config.working_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let skill_registry = Arc::new(skill::SkillRegistry::new());
    skill::load_skills(&skill_registry, &working_dir);

    let mut tool_registry = tool::create_default_registry(working_dir, Arc::clone(&skill_registry));

    for srv in &config.mcp_servers {
        let args: Vec<&str> = srv.args.iter().map(|s| s.as_str()).collect();
        match mcp_client::McpClient::connect(&srv.command, &args) {
            Ok(mc) => {
                for handler in mc.tool_handlers() {
                    tool_registry.register(handler);
                }
            }
            Err(e) => tracing::warn!("Failed to connect MCP server '{}': {}", srv.command, e),
        }
    }

    let agent_config = agent::AgentConfig {
        system_prompt: config.system_prompt,
        max_tokens: config.max_tokens,
        context_window: config.context_window,
    };

    let inner = agent::Agent::new_with_skills(client, tool_registry, agent_config, skill_registry);
    Ok(Arc::new(Agent(Mutex::new(inner))))
}

impl Agent {
    pub fn step(&self, user_input: String) -> Result<AgentResponse, AgentError> {
        let mut inner = self.0.lock();
        let resp = inner.step(user_input)?;
        Ok(AgentResponse {
            content: resp.content,
            reasoning: resp.reasoning,
            input_tokens: resp.input_tokens,
            output_tokens: resp.output_tokens,
            total_tokens: resp.total_tokens,
            context_percent: resp.context_percent,
        })
    }

    pub fn reset(&self) {
        self.0.lock().reset();
    }

    pub fn set_system_prompt(&self, prompt: String) {
        self.0.lock().set_system_prompt(prompt);
    }

    pub fn add_skill(&self, name: String, description: String, prompt: String) {
        self.0.lock().add_skill(name, description, prompt);
    }

    pub fn step_with_allowed_tools(
        &self,
        user_input: String,
        _allowed_tools: Vec<String>,
    ) -> Result<AgentResponse, AgentError> {
        // For now delegate to step(); filtered tool support can be added later.
        self.step(user_input)
    }

    pub fn get_conversation_history(&self) -> String {
        self.0.lock().get_conversation_history()
    }
}
