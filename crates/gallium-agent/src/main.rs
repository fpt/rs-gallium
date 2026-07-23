//! gallium CLI — a text-mode ReAct REPL plus the `app-server` mode that exposes
//! the agent over JSON-RPC as a whole-turn backend, using the codex-app-server
//! protocol that clients like rs-kessel and klein-cli call "ACP" (not the
//! agentclientprotocol.com standard). Replaces `kessel-cli`.
//!
//! Usage:
//!   # OpenAI:
//!   OPENAI_API_KEY=sk-... gallium
//!
//!   # Local model (llama.cpp `local` feature, or native `gallium` backend):
//!   MODEL_PATH=/path/to/model.gguf gallium
//!   INFERENCE_ENGINE=gallium MODEL_PATH=hf:ORG/REPO/file.gguf gallium
//!
//!   # One-shot (piped stdin, for integration tests):
//!   echo "Read Cargo.toml" | MODEL_PATH=... gallium
//!
//!   # As a whole-turn backend for another agent (e.g. klein):
//!   OPENAI_API_KEY=sk-... gallium app-server
//!
//!   # Load settings from a TOML config (env vars still override individual fields):
//!   gallium --config configs/gemma4.toml
//!   gallium app-server --config configs/openai.toml

mod config;

use gallium_agent::tool::ToolAccess;
use gallium_agent::{create_provider, ChatMessage};

use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;

/// Settings shared by both modes, resolved from (in order of precedence)
/// environment variables, an optional `--config` file, then built-in defaults.
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
    /// System-prompt text loaded from the config's `systemPromptPath` (REPL only).
    system_prompt: Option<String>,
    /// SKILL.md dirs from the config's `skillPaths`, resolved to absolute/cwd-relative.
    skill_paths: Vec<PathBuf>,
    /// MCP servers declared in the config file (REPL only).
    mcp_servers: Vec<config::McpServerConfig>,
}

impl EnvConfig {
    /// Resolve settings from env vars layered over an optional config file.
    /// `config_dir` is the directory of the config file, used to resolve its
    /// relative `systemPromptPath` / `skillPaths`.
    fn resolve(file: config::FileConfig, config_dir: Option<&std::path::Path>) -> Self {
        let config::FileConfig {
            llm,
            agent,
            mcp_servers,
        } = file;

        let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());

        // A config's `baseURL: ""` for local models must not shadow the default.
        let base_url = env("LLM_BASE_URL")
            .or(llm.base_url.filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());

        // Read the system prompt file eagerly so failures surface at startup.
        let system_prompt = agent.system_prompt_path.and_then(|p| {
            let path = config::resolve_relative(config_dir, &p);
            match std::fs::read_to_string(&path) {
                Ok(text) => Some(text),
                Err(e) => {
                    eprintln!("Warning: systemPromptPath '{}': {}", path.display(), e);
                    None
                }
            }
        });

        let skill_paths = agent
            .skill_paths
            .iter()
            .map(|p| config::resolve_relative(config_dir, p))
            .collect();

        // An env `MODEL_PATH` is a runtime override (cwd-relative, left as-is);
        // a config `modelPath` is resolved relative to the config file's dir.
        let model_path = env("MODEL_PATH")
            .or_else(|| llm.model_path.map(|p| config::resolve_model_path(config_dir, p)));

        Self {
            model_path,
            base_url,
            model: env("LLM_MODEL")
                .or(llm.model)
                .unwrap_or_else(|| "gpt-5.6-luna".to_string()),
            api_key: env("OPENAI_API_KEY").or(llm.api_key.filter(|s| !s.is_empty())),
            working_dir: env("WORKING_DIR").unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            }),
            max_tokens: env("MAX_TOKENS")
                .and_then(|s| s.parse().ok())
                .or(llm.max_tokens)
                .unwrap_or(2048),
            // Falls back to the library default rather than restating it, so the
            // two cannot drift apart.
            max_react_iterations: env("MAX_REACT_ITERATIONS")
                .and_then(|s| s.parse().ok())
                .or(agent.max_turns)
                .unwrap_or(gallium_agent::react::DEFAULT_MAX_ITERATIONS),
            temperature: env("LLM_TEMPERATURE")
                .and_then(|s| s.parse().ok())
                .or(llm.temperature),
            reasoning_effort: env("REASONING_EFFORT").or(llm.reasoning_effort),
            inference_engine: env("INFERENCE_ENGINE").or(llm.inference_engine),
            system_prompt,
            skill_paths,
            mcp_servers,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // The first positional (before any flags) selects the mode.
    let app_server = args.get(1).map(String::as_str) == Some("app-server");

    // Load the optional `--config <path>` TOML, resolving its relative paths
    // against the file's own directory.
    let config_path = config::parse_config_flag(&args).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(2);
    });
    let (file_config, config_dir) = match &config_path {
        Some(path) => {
            let file = config::FileConfig::load(std::path::Path::new(path)).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            let dir = std::path::Path::new(path).parent().map(|p| p.to_path_buf());
            (file, dir)
        }
        None => (config::FileConfig::default(), None),
    };

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

    let config = EnvConfig::resolve(file_config, config_dir.as_deref());
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
        system_prompt,
        skill_paths,
        mcp_servers,
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
    // Additional SKILL.md dirs from the config's `skillPaths`.
    for dir in &skill_paths {
        skill_registry.load_from_dir(dir);
    }
    let situation = std::sync::Arc::new(gallium_agent::situation::SituationMessages::default());
    let mut tool_registry = gallium_agent::tool::create_default_registry(
        std::path::PathBuf::from(&working_dir),
        skill_registry,
        situation,
    );

    // Connect MCP servers declared in the config file (stdio `command` or HTTP `url`).
    for server in &mcp_servers {
        if let Some(url) = &server.url {
            match gallium_agent::mcp_client_http::McpHttpClient::connect(url) {
                Ok(client) => {
                    for handler in client.tool_handlers() {
                        tool_registry.register(handler);
                    }
                }
                Err(e) => eprintln!("Failed to connect MCP server '{}': {}", url, e),
            }
        } else if let Some(cmd) = &server.command {
            let args: Vec<&str> = server.args.iter().map(String::as_str).collect();
            match gallium_agent::mcp_client::McpClient::connect(cmd, &args) {
                Ok(client) => {
                    for handler in client.tool_handlers() {
                        tool_registry.register(handler);
                    }
                }
                Err(e) => eprintln!("Failed to connect MCP server '{}': {}", cmd, e),
            }
        }
    }

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
    // `model` is the cloud model id and is unused by the local providers, so show
    // the loaded path/hf spec instead of a default that was never applied.
    let model_label = model_path.as_deref().unwrap_or(&model);

    // Check if stdin is a pipe (one-shot mode) or terminal (interactive)
    let is_interactive = io::stdin().is_terminal();

    if is_interactive {
        eprintln!("=== gallium (ReAct Tool Calling) ===");
        eprintln!("Provider: {} ({})", provider_name, model_label);
        eprintln!("Working dir: {}", working_dir);
        eprintln!(
            "Tools: {:?}",
            tool_registry
                .get_definitions()
                .iter()
                .map(|t| &t.name)
                .collect::<Vec<_>>()
        );
        eprintln!("Type /quit to exit\n");
    }

    let system_prompt = system_prompt.unwrap_or_else(|| {
        "You are a helpful assistant with access to tools. \
         Use tools when the user asks you to read files, find files, or manage tasks. \
         Be concise in your responses."
            .to_string()
    });
    let mut messages: Vec<ChatMessage> = vec![ChatMessage::system(system_prompt)];

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
