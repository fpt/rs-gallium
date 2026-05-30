//! gallium-agent CLI — interactive ReAct agent backed by a local gallium model or OpenAI.
//!
//! Usage examples:
//!   # Local GPT-OSS (downloads from HuggingFace)
//!   gallium-agent --provider gallium --arch gpt-oss --hf-repo openai/gpt-oss-20b --dtype f16
//!
//!   # OpenAI
//!   gallium-agent --provider openai --openai-model gpt-4o-mini
//!
//! Commands during a session:
//!   /reset    — clear conversation history
//!   /help     — show commands
//!   /quit     — exit

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use gallium_agent::agent::{Agent, AgentConfig};
use gallium_agent::llm::OpenAiProvider;
use gallium_agent::mcp_client::McpClient;
use gallium_agent::protocol::{GemmaProtocol, HarmonyProtocol, QwenProtocol};
use gallium_agent::provider::GalliumProvider;
use gallium_agent::skill::{load_skills, SkillRegistry};
use gallium_agent::tool::create_default_registry;

// ============================================================================
// CLI args
// ============================================================================

#[derive(Debug, Clone, clap::ValueEnum)]
enum ProviderKind {
    Gallium,
    Openai,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum ModelArch {
    GptOss,
    Qwen35,
    Gemma4,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum ModelFormat {
    Safetensors,
    Gguf,
}

#[derive(Parser, Debug)]
#[command(name = "gallium-agent", about = "Interactive ReAct agent — local model or OpenAI")]
struct Args {
    // --- Provider ---
    #[arg(long, default_value = "gallium")]
    provider: ProviderKind,

    // --- Gallium provider ---
    #[arg(long)]
    arch: Option<ModelArch>,

    #[arg(long, default_value = "safetensors")]
    format: ModelFormat,

    #[arg(long)]
    model: Option<PathBuf>,

    #[arg(long)]
    hf_repo: Option<String>,

    #[arg(long)]
    hf_file: Option<String>,

    #[arg(long)]
    hf_tokenizer_repo: Option<String>,

    #[arg(long, default_value = "f16")]
    dtype: String,

    #[arg(long, default_value_t = false)]
    thinking: bool,

    // --- OpenAI provider ---
    #[arg(long, default_value = "gpt-4o-mini")]
    openai_model: String,

    #[arg(long)]
    openai_api_key: Option<String>,

    #[arg(long)]
    reasoning_effort: Option<String>,

    // --- MCP servers ---
    /// MCP server to connect: "command arg1 arg2" (repeat for multiple servers)
    #[arg(long = "mcp")]
    mcp_servers: Vec<String>,

    // --- Common ---
    #[arg(long)]
    system_prompt: Option<String>,

    #[arg(long)]
    working_dir: Option<PathBuf>,

    #[arg(long, default_value = "512")]
    max_tokens: u32,

    #[arg(long, default_value = "0.7")]
    temperature: f32,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    presence_penalty: Option<f32>,

    #[arg(long, default_value = "32000")]
    context_window: u32,

    /// Session ID for persistent conversation (saves to .gallium/sessions/<id>.jsonl).
    #[arg(long)]
    session: Option<String>,

    #[arg(long, short = 'f')]
    file: Option<PathBuf>,
}

// ============================================================================
// HuggingFace download
// ============================================================================

fn download_from_hub(
    repo_id: &str,
    format: &ModelFormat,
    hf_file: Option<&str>,
    hf_tokenizer_repo: Option<&str>,
) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;

    eprintln!("Fetching from HuggingFace: {repo_id}");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    match format {
        ModelFormat::Safetensors => {
            let info = repo.info()?;
            let shards: Vec<_> = info.siblings.iter()
                .map(|s| s.rfilename.clone())
                .filter(|name| name.ends_with(".safetensors"))
                .collect();
            if shards.is_empty() {
                anyhow::bail!("no .safetensors files found in {repo_id}");
            }
            eprintln!("  Downloading config.json and tokenizer.json");
            repo.get("config.json")?;
            let tok_repo = hf_tokenizer_repo.unwrap_or(repo_id);
            api.model(tok_repo.to_string()).get("tokenizer.json")?;
            for shard in &shards {
                eprintln!("  Downloading {shard}");
                repo.get(shard)?;
            }
            let config_local = repo.get("config.json")?;
            Ok(config_local.parent().unwrap().to_path_buf())
        }
        ModelFormat::Gguf => {
            let filename = hf_file.ok_or_else(|| anyhow::anyhow!("--hf-file required with --format gguf"))?;
            let tok_repo = hf_tokenizer_repo.unwrap_or(repo_id);
            eprintln!("  Downloading tokenizer.json from {tok_repo}");
            let tok_local = api.model(tok_repo.to_string()).get("tokenizer.json")?;
            eprintln!("  Downloading {filename}");
            let gguf_local = repo.get(filename)?;
            let gguf_dir = gguf_local.parent().unwrap();
            let tok_dest = gguf_dir.join("tokenizer.json");
            if !tok_dest.exists() {
                std::fs::copy(&tok_local, &tok_dest)?;
            }
            Ok(gguf_local)
        }
    }
}

// ============================================================================
// Model loading
// ============================================================================

fn load_gallium_provider(args: &Args) -> Result<GalliumProvider> {
    use gallium_core::SamplingParams;
    use gallium_models::loader;

    let arch = args.arch.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--arch is required for --provider gallium")
    })?;

    let model_path = match &args.hf_repo {
        Some(repo) => download_from_hub(repo, &args.format, args.hf_file.as_deref(), args.hf_tokenizer_repo.as_deref())?,
        None => args.model.clone().ok_or_else(|| anyhow::anyhow!("--model or --hf-repo is required"))?,
    };

    let device = Device::Cpu;

    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        presence_penalty: args.presence_penalty,
        ..Default::default()
    };

    let (model, tokenizer) = match args.format {
        ModelFormat::Gguf => {
            eprintln!("Loading GGUF model from {:?}...", model_path);
            let (metadata, vb) = gallium_core::load_gguf(&model_path, &device)?;
            let dir = model_path.parent().unwrap_or(std::path::Path::new("."));
            let tok_path = dir.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {:?}: {e}", tok_path))?;
            let model: Box<dyn gallium_core::CausalLM> = match arch {
                ModelArch::GptOss => Box::new(gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device)?),
                ModelArch::Qwen35 => Box::new(gallium_models::qwen35_q::Qwen35Q::load(&metadata, &vb, &device)?),
                ModelArch::Gemma4 => Box::new(gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device)?),
            };
            eprintln!("Model loaded.");
            (model, tokenizer)
        }
        ModelFormat::Safetensors => {
            let dtype = match args.dtype.as_str() {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::BF16,
                other => anyhow::bail!("unsupported dtype: {other}"),
            };
            eprintln!("Loading safetensors model from {:?}...", model_path);
            let safetensors: Vec<PathBuf> = std::fs::read_dir(&model_path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|ext| ext == "safetensors").unwrap_or(false))
                .collect();
            if safetensors.is_empty() {
                anyhow::bail!("no .safetensors files in {:?}", model_path);
            }
            let config_path = model_path.join("config.json");
            let vb = loader::load_safetensors(&safetensors, dtype, &device)?;
            let tok_path = model_path.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
            let model: Box<dyn gallium_core::CausalLM> = match arch {
                ModelArch::GptOss => {
                    let cfg: gallium_models::gpt_oss::GptOssConfig = loader::load_config(&config_path)?;
                    Box::new(gallium_models::gpt_oss::GptOss::load(&cfg, vb, &safetensors, &device)?)
                }
                ModelArch::Qwen35 => {
                    let full: serde_json::Value = loader::load_config(&config_path)?;
                    let text = full.get("text_config").unwrap_or(&full);
                    let cfg: gallium_models::qwen35::Qwen35Config = serde_json::from_value(text.clone())
                        .map_err(|e| anyhow::anyhow!("Qwen35 config error: {e}"))?;
                    Box::new(gallium_models::qwen35::Qwen35::load(&cfg, vb, &device)?)
                }
                ModelArch::Gemma4 => {
                    let full: serde_json::Value = loader::load_config(&config_path)?;
                    let text = full.get("text_config").unwrap_or(&full);
                    let cfg: gallium_models::gemma4::Gemma4Config = serde_json::from_value(text.clone())
                        .map_err(|e| anyhow::anyhow!("Gemma4 config error: {e}"))?;
                    Box::new(gallium_models::gemma4::Gemma4::load(&cfg, vb, &device)?)
                }
            };
            eprintln!("Model loaded.");
            (model, tokenizer)
        }
    };

    let protocol: Box<dyn gallium_agent::protocol::ModelProtocol> = match arch {
        ModelArch::GptOss  => Box::new(HarmonyProtocol),
        ModelArch::Gemma4  => {
            if args.thinking {
                Box::new(GemmaProtocol::with_thinking())
            } else {
                Box::new(GemmaProtocol::new())
            }
        }
        ModelArch::Qwen35  => Box::new(QwenProtocol),
    };

    Ok(GalliumProvider::new(model, tokenizer, params, args.max_tokens as usize, protocol))
}

// ============================================================================
// REPL
// ============================================================================

fn run_batch(mut agent: Agent, file_path: &PathBuf) -> Result<()> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {:?}: {}", file_path, e))?;

    let mut turns: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in content.lines() {
        if line.trim() == "----" {
            turns.push(current.join("\n"));
            current.clear();
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        turns.push(current.join("\n"));
    }

    let mut turn_num = 0;
    for turn in &turns {
        let input = turn.trim().to_string();
        if input.is_empty() { continue; }
        turn_num += 1;
        println!("=== Turn {} ===", turn_num);
        match agent.step(input) {
            Ok(resp) => println!("{}", resp.content),
            Err(e) => eprintln!("[Error] {}", e),
        }
        println!();
    }

    Ok(())
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  /reset         — clear conversation history");
    eprintln!("  /help          — show this message");
    eprintln!("  /skills        — list available skills");
    eprintln!("  /quit          — exit");
}

fn run_repl(mut agent: Agent, session_path: Option<PathBuf>) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => { eprintln!("Read error: {e}"); break; }
        }

        let input = line.trim().to_string();
        if input.is_empty() { continue; }

        match input.as_str() {
            "/quit" | "/exit" | "/q" => break,
            "/reset" => {
                agent.reset();
                if let Some(ref path) = session_path {
                    let _ = std::fs::remove_file(path);
                }
                eprintln!("[Conversation reset]");
                continue;
            }
            "/help" => { print_help(); continue; }
            "/skills" => {
                // Skills are in the skill_registry — for now use lookup_skill tool indirectly.
                eprintln!("[Use the 'lookup_skill' tool in your prompt to see available skills]");
                continue;
            }
            _ if input.starts_with('/') => {
                eprintln!("Unknown command: {}. Type /help for commands.", input);
                continue;
            }
            _ => {}
        }

        match agent.step(input) {
            Ok(resp) => {
                println!("{}", resp.content);

                if let Some(ref reasoning) = resp.reasoning {
                    eprintln!("\n[Reasoning]\n{}", reasoning);
                }

                if resp.total_tokens > 0 || resp.context_percent > 0.0 {
                    eprintln!(
                        "[in={} out={} total={} | ctx={:.0}%]",
                        resp.input_tokens, resp.output_tokens, resp.total_tokens,
                        resp.context_percent
                    );
                }
            }
            Err(e) => eprintln!("[Error] {}", e),
        }
    }

    Ok(())
}

// ============================================================================
// main
// ============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    let working_dir = args.working_dir.clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let provider_desc: String;

    let client: Box<dyn gallium_agent::llm::LlmProvider> = match args.provider {
        ProviderKind::Gallium => {
            let arch_name = args.arch.as_ref().map(|a| format!("{:?}", a)).unwrap_or_else(|| "?".to_string());
            provider_desc = format!("gallium ({}, {})", arch_name.to_lowercase(), args.dtype);
            Box::new(load_gallium_provider(&args)?)
        }
        ProviderKind::Openai => {
            let api_key = args.openai_api_key.clone()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| anyhow::anyhow!(
                    "OpenAI API key required: pass --openai-api-key or set OPENAI_API_KEY"
                ))?;
            provider_desc = format!("openai ({})", args.openai_model);
            Box::new(OpenAiProvider::new(
                api_key,
                args.openai_model.clone(),
                Some(args.temperature),
                args.max_tokens,
                args.reasoning_effort.clone(),
            ))
        }
    };

    // Skill registry: load from disk, share with tool registry.
    let skill_registry = Arc::new(SkillRegistry::new());
    load_skills(&skill_registry, &working_dir);

    let mut tool_registry = create_default_registry(working_dir.clone(), Arc::clone(&skill_registry));

    // Connect MCP servers (space-separated "command arg1 arg2" strings).
    for mcp_spec in &args.mcp_servers {
        let parts: Vec<&str> = mcp_spec.split_whitespace().collect();
        if parts.is_empty() { continue; }
        let (cmd, mcp_args) = (parts[0], &parts[1..]);
        match McpClient::connect(cmd, mcp_args) {
            Ok(mc) => {
                eprintln!("[MCP] Connected to '{}'", cmd);
                for handler in mc.tool_handlers() {
                    tool_registry.register(handler);
                }
            }
            Err(e) => eprintln!("[MCP] Warning: could not connect to '{}': {}", cmd, e),
        }
    }

    // Session persistence: optionally load prior conversation.
    let session_path = args.session.as_deref().map(|id| {
        gallium_agent::session::session_path(&working_dir, id)
    });

    let config = AgentConfig {
        system_prompt: args.system_prompt.clone(),
        max_tokens: args.max_tokens,
        context_window: args.context_window,
    };

    let mut agent = Agent::new_with_skills(client, tool_registry, config, Arc::clone(&skill_registry));

    // Load saved session if requested.
    if let Some(ref path) = session_path {
        match gallium_agent::session::load(path) {
            Ok(mem) if !mem.is_empty() => {
                eprintln!("[Loaded session from {:?}: {} messages]", path, mem.len());
                for msg in mem.get_messages() {
                    agent.memory_push(msg);
                }
            }
            Ok(_) => {}
            Err(e) => eprintln!("[Warning: could not load session: {}]", e),
        }
    }

    eprintln!("gallium-agent | provider: {} | type /help for commands", provider_desc);

    if let Some(ref file_path) = args.file {
        run_batch(agent, file_path)
    } else {
        run_repl(agent, session_path)
    }
}
