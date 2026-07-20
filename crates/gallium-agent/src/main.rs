//! gallium-agent CLI — a text-mode ReAct REPL plus the `app-server` mode that
//! exposes the agent over JSON-RPC (the whole-turn ACP backend a client like
//! klein spawns). Replaces `kessel-cli`.
//!
//! Usage:
//!   # OpenAI:
//!   OPENAI_API_KEY=sk-... gallium-agent
//!
//!   # Local model (llama.cpp `local` feature, or native `gallium` backend):
//!   MODEL_PATH=/path/to/model.gguf gallium-agent
//!   INFERENCE_ENGINE=gallium MODEL_PATH=hf:ORG/REPO/file.gguf gallium-agent
//!
//!   # One-shot (piped stdin, for integration tests):
//!   echo "Read configs/default.yaml" | MODEL_PATH=... gallium-agent
//!
//!   # As a whole-turn backend for another agent (e.g. klein):
//!   OPENAI_API_KEY=sk-... gallium-agent app-server

use gallium_agent::tool::ToolAccess;
use gallium_agent::{create_provider, ChatMessage};

use std::io::{self, BufRead, IsTerminal};

/// Environment-derived settings shared by both modes.
struct EnvConfig {
    model_path: Option<String>,
    base_url: String,
    model: String,
    api_key: Option<String>,
    working_dir: String,
    max_tokens: u32,
    max_react_iterations: u32,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    inference_engine: Option<String>,
}

impl EnvConfig {
    fn from_env() -> Self {
        Self {
            model_path: std::env::var("MODEL_PATH").ok(),
            base_url: std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "gpt-5.6-luna".to_string()),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            working_dir: std::env::var("WORKING_DIR")
                .unwrap_or_else(|_| std::env::current_dir().unwrap().to_string_lossy().to_string()),
            max_tokens: std::env::var("MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048),
            // Falls back to the library default rather than restating it, so the
            // two cannot drift apart.
            max_react_iterations: std::env::var("MAX_REACT_ITERATIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(gallium_agent::react::DEFAULT_MAX_ITERATIONS),
            temperature: std::env::var("LLM_TEMPERATURE").ok().and_then(|s| s.parse().ok()),
            reasoning_effort: std::env::var("REASONING_EFFORT").ok(),
            inference_engine: std::env::var("INFERENCE_ENGINE").ok(),
        }
    }
}

fn main() {
    let app_server = std::env::args().nth(1).as_deref() == Some("app-server");

    // In app-server mode stdout carries the JSON-RPC stream, so logs must not
    // touch it. (The default fmt subscriber writes to stdout.)
    let subscriber = tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    );
    if app_server {
        subscriber.with_writer(io::stderr).init();
    } else {
        subscriber.init();
    }

    let config = EnvConfig::from_env();
    if app_server {
        run_app_server(config);
    } else {
        run_repl(config);
    }
}

/// Serve the agent over JSON-RPC on stdio until the client disconnects.
fn run_app_server(config: EnvConfig) {
    gallium_agent::appserver::run_stdio(gallium_agent::appserver::ServerConfig {
        model_path: config.model_path,
        base_url: config.base_url,
        model: config.model,
        api_key: config.api_key,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        reasoning_effort: config.reasoning_effort,
        inference_engine: config.inference_engine,
        max_iterations: Some(config.max_react_iterations),
    });
}

fn run_repl(config: EnvConfig) {
    let EnvConfig {
        model_path,
        base_url,
        model,
        api_key,
        working_dir,
        max_tokens,
        max_react_iterations,
        temperature,
        reasoning_effort,
        inference_engine,
    } = config;

    let client = create_provider(
        model_path.clone(),
        base_url.clone(),
        model.clone(),
        api_key.clone(),
        temperature,
        max_tokens,
        reasoning_effort,
        inference_engine.clone(),
    )
    .expect("Failed to create LLM provider");

    // Create tool registry
    let skill_registry = std::sync::Arc::new(gallium_agent::skill::SkillRegistry::new());
    gallium_agent::skill::load_skills(&skill_registry, std::path::Path::new(&working_dir));
    let situation = std::sync::Arc::new(gallium_agent::situation::SituationMessages::default());
    let mut tool_registry = gallium_agent::tool::create_default_registry(
        std::path::PathBuf::from(&working_dir),
        skill_registry,
        situation,
    );

    // Connect MCP servers from MCP_SERVERS env (comma-separated "command arg1 arg2,...")
    if let Ok(mcp_spec) = std::env::var("MCP_SERVERS") {
        for entry in mcp_spec.split(',') {
            let parts: Vec<&str> = entry.trim().split_whitespace().collect();
            if let Some((cmd, args)) = parts.split_first() {
                match gallium_agent::mcp_client::McpClient::connect(cmd, args) {
                    Ok(client) => {
                        for handler in client.tool_handlers() {
                            tool_registry.register(handler);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to connect MCP server '{}': {}", cmd, e);
                    }
                }
            }
        }
    }

    let provider_name = if model_path.is_some() {
        "Local"
    } else if api_key.is_some() {
        "OpenAI"
    } else {
        "Unknown"
    };

    // Check if stdin is a pipe (one-shot mode) or terminal (interactive)
    let is_interactive = io::stdin().is_terminal();

    if is_interactive {
        eprintln!("=== gallium-agent (ReAct Tool Calling) ===");
        eprintln!("Provider: {} ({})", provider_name, model);
        eprintln!("Working dir: {}", working_dir);
        eprintln!("Tools: {:?}", tool_registry.get_definitions().iter().map(|t| &t.name).collect::<Vec<_>>());
        eprintln!("Type /quit to exit\n");
    }

    let mut messages: Vec<ChatMessage> = vec![
        ChatMessage::system(
            "You are a helpful assistant with access to tools. \
             Use tools when the user asks you to read files, find files, or manage tasks. \
             Be concise in your responses."
                .to_string(),
        ),
    ];

    let stdin = io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/quit" || input == "/exit" {
            break;
        }

        if input == "/reset" {
            messages.truncate(1); // Keep system prompt
            eprintln!("Conversation reset.");
            continue;
        }

        // Add user message
        messages.push(ChatMessage::user(input.clone()));

        if is_interactive {
            eprint!("Thinking...");
        }

        // Run ReAct loop
        let mut react_messages = messages.clone();

        let result = gallium_agent::react::run(
            client.as_ref(),
            &mut react_messages,
            &tool_registry,
            Some(max_react_iterations),
        );

        if is_interactive {
            eprint!("\r            \r"); // Clear "Thinking..."
        }

        match result {
            Ok((response, reasoning, usage)) => {
                if let Some(ref thinking) = reasoning {
                    eprintln!("\x1b[90m💭 {}\x1b[0m", thinking);
                }
                // Prefix so consumers can find the reply (matches the testsuite's
                // "Assistant:" contract).
                println!("Assistant: {}", response);
                if usage.total_tokens > 0 {
                    eprintln!(
                        "\x1b[90m📊 tokens: in={}, out={}, total={}\x1b[0m",
                        usage.input_tokens, usage.output_tokens, usage.total_tokens
                    );
                }

                // Add assistant response to conversation history
                messages.push(ChatMessage::assistant(response));
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }

        if is_interactive {
            println!();
        }
    }

    if is_interactive {
        eprintln!("Goodbye!");
    }
}
