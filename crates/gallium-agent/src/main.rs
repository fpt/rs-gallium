//! gallium CLI — a text-mode ReAct REPL plus the `app-server` mode that exposes
//! the agent over JSON-RPC as a whole-turn backend, using the codex-app-server
//! protocol (not the agentclientprotocol.com standard — see `appserver`).
//!
//! Usage:
//!   # OpenAI:
//!   OPENAI_API_KEY=sk-... gallium
//!
//!   # Local model (llama.cpp `local` feature, or native `gallium` backend):
//!   MODEL_PATH=/path/to/model.gguf gallium
//!   INFERENCE_ENGINE=candle MODEL_PATH=hf:ORG/REPO/file.gguf gallium
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

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

/// Renders turn progress to the terminal from the agent's event stream.
///
/// Everything goes to stderr: stdout carries the reply itself, which the
/// testsuite parses, and a one-shot piped run must not have progress chatter
/// interleaved into it.
struct TerminalRenderer;

impl TerminalRenderer {
    /// The line an event should print, or `None` when it prints nothing.
    ///
    /// Split out from `on_event` so the formatting is testable without a
    /// terminal to capture.
    fn render(event: &gallium_agent::AgentEvent<'_>) -> Option<String> {
        use gallium_agent::AgentEvent;
        match event {
            AgentEvent::ToolStarted {
                name, arguments, ..
            } => {
                // Arguments can be a whole file's contents; show the shape, not
                // the payload.
                let summary = arguments
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| format!("{k}={}", summarize_arg(v)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                Some(format!("\x1b[90m⚙  {name}({summary})\x1b[0m"))
            }
            AgentEvent::ToolCompleted { name, result, .. } => {
                let text = result.display_text();
                let first_line = text.lines().next().unwrap_or("").trim();
                Some(if result.is_error {
                    format!("\x1b[31m   ✗ {name}: {first_line}\x1b[0m")
                } else {
                    format!("\x1b[90m   ✓ {first_line}\x1b[0m")
                })
            }
            AgentEvent::Error { message } => Some(format!("\x1b[31m   ✗ {message}\x1b[0m")),
            // Only reachable once a turn can be steered, which the REPL cannot
            // do — it reads one line at a time and has nothing to say while a
            // turn is running. Rendered anyway rather than dropped: the day it
            // grows a way to interject, an unshown reply would be the bug.
            AgentEvent::AgentMessage { text } => Some(format!("\x1b[90m{text}\x1b[0m")),
            // Per *call*, which is the granularity a tuning run needs: a turn
            // total averages a long first prefill together with the short ones
            // that follow, and it is the individual call that a batch size or
            // an offload split moves. Counts are left to the turn's 📊 line —
            // this one is about speed.
            AgentEvent::Usage { usage } => usage.timing.map(|t| {
                format!(
                    "\x1b[90m   ⏱  ttft {:.2}s · prefill {} tok {} · decode {} tok {}\x1b[0m",
                    t.ttft.as_secs_f64(),
                    t.prefill_tokens,
                    gallium_agent::fmt_rate(t.prefill_rate()),
                    t.decode_tokens,
                    gallium_agent::fmt_rate(t.decode_rate()),
                )
            }),
            // The REPL prints the reply and the token line itself, from the
            // values `run_observed` returns.
            AgentEvent::TurnCompleted { .. } => None,
        }
    }
}

impl gallium_agent::AgentObserver for TerminalRenderer {
    fn on_event(&self, event: gallium_agent::AgentEvent<'_>) {
        if let Some(line) = Self::render(&event) {
            eprintln!("{line}");
        }
    }
}

/// One-line rendering of a tool argument, capped so a `write` of a large file
/// does not fill the terminal.
fn summarize_arg(value: &serde_json::Value) -> String {
    const MAX: usize = 60;
    let raw = match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let one_line = raw.replace('\n', "⏎");
    if one_line.chars().count() <= MAX {
        return one_line;
    }
    let cut: String = one_line.chars().take(MAX).collect();
    format!("{cut}… ({} chars)", one_line.chars().count())
}

/// Settings shared by both modes, resolved from (in order of precedence)
/// environment variables, an optional `--config` file, then built-in defaults.
struct EnvConfig {
    model_path: Option<String>,
    /// Multimodal projector for the llama.cpp backend. `None` is text only.
    mmproj_path: Option<String>,
    base_url: String,
    model: String,
    api_key: Option<String>,
    working_dir: String,
    max_tokens: u32,
    /// Explicitly configured window, and nothing else. The provider reports its
    /// own when this is `None`; the fallback guesses only when neither can say.
    context_window: Option<u32>,
    max_react_iterations: u32,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    inference_engine: Option<String>,
    /// Where the native candle backend finds `tokenizer.json`: a local path, or
    /// a HuggingFace repo id. `None` leaves it to derive one from `model_path`.
    tokenizer_path: Option<String>,
    /// GPU layers to offload for the llama.cpp backend. `None` leaves it to
    /// llama.cpp's own default (999, offload everything).
    gpu_layers: Option<u32>,
    /// Move MoE expert tensors to CPU for the llama.cpp backend. `false`
    /// leaves them offloaded same as everything else.
    cpu_moe: bool,
    /// Which model profile reads the model's output. `None` detects it from what
    /// the model file reports, which is the right answer almost always.
    profile: Option<String>,
    /// System-prompt text loaded from the config's `systemPromptPath` (REPL only).
    system_prompt: Option<String>,
    /// SKILL.md dirs from the config's `skillPaths`, resolved to absolute/cwd-relative.
    skill_paths: Vec<PathBuf>,
    /// Per-tier approval rules from the config's `[agent.approvals]`.
    approval_policy: gallium_agent::approval::ApprovalPolicy,
    /// Where the config asked for turn traces, if it did.
    trace_dir: Option<PathBuf>,
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

        let approval_policy = agent.approvals.resolve();

        // Resolved against the config's directory like every other path in the
        // file, so `dir = "traces"` means the same thing from any working
        // directory.
        let trace_dir = agent
            .trace
            .dir
            .filter(|d| !d.trim().is_empty())
            .map(|d| config::resolve_relative(config_dir, &d));

        // An env `MODEL_PATH` is a runtime override (cwd-relative, left as-is);
        // a config `modelPath` is resolved relative to the config file's dir.
        let model_path = env("MODEL_PATH").or_else(|| {
            llm.model_path
                .map(|p| config::resolve_model_path(config_dir, p))
        });

        // The projector travels with the model, so it resolves exactly like one.
        let mmproj_path = env("MMPROJ_PATH").or_else(|| {
            llm.mmproj_path
                .map(|p| config::resolve_model_path(config_dir, p))
        });

        // Same shape, and the same reason the env var wins: it is the runtime
        // override. `GALLIUM_TOKENIZER_REPO` keeps its name — it has only ever
        // meant a repo, and renaming a documented variable to match a new
        // config key would break every script that sets it.
        let tokenizer_path = env("GALLIUM_TOKENIZER_REPO").or_else(|| {
            llm.tokenizer_path
                .map(|p| config::resolve_tokenizer_path(config_dir, p))
        });

        // Left unresolved on purpose. A window the user did not name is one the
        // *model* may know — `LlmProvider::context_window` reads it out of the
        // GGUF or `config.json` — and that answer is only available once the
        // provider is built. `memory::resolve_context_window` settles it there,
        // guessing only when neither can say.
        let context_window = env("CONTEXT_WINDOW")
            .and_then(|s| s.parse().ok())
            .or(llm.context_window);

        Self {
            model_path,
            mmproj_path,
            base_url,
            model: env("LLM_MODEL")
                .or(llm.model)
                .unwrap_or_else(|| "gpt-5.6-luna".to_string()),
            api_key: env("OPENAI_API_KEY").or(llm.api_key.filter(|s| !s.is_empty())),
            tokenizer_path,
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
            context_window,
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
            gpu_layers: env("GALLIUM_GPU_LAYERS")
                .and_then(|s| s.parse().ok())
                .or(llm.gpu_layers),
            profile: env("GALLIUM_PROFILE").or(llm.profile),
            cpu_moe: env("GALLIUM_CPU_MOE")
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(llm.cpu_moe),
            system_prompt,
            skill_paths,
            approval_policy,
            trace_dir,
            mcp_servers,
        }
    }
}

/// The input prompt, in the style of the `pure` shell prompt: one glyph, no
/// decoration, and colour used for exactly one thing — it turns red when the
/// last turn failed.
fn prompt_string(last_turn_failed: bool) -> String {
    let color = if last_turn_failed {
        "\x1b[31m" // red
    } else {
        "\x1b[35m" // magenta
    };
    format!("{color}\u{276f}\x1b[0m ")
}

/// Draw the prompt and flush it by hand: it has no trailing newline, so nothing
/// else will. It goes to stderr because stdout carries the replies that piped
/// consumers parse.
fn draw_prompt(last_turn_failed: bool) {
    let mut err = io::stderr();
    let _ = write!(err, "{}", prompt_string(last_turn_failed));
    let _ = err.flush();
}

/// How a reply is marked.
///
/// On a terminal it echoes the input prompt — same glyph, different colour —
/// because "Assistant:" is a label for a machine, and a person reading their
/// own terminal already knows which half is the reply.
///
/// Piped, it stays `Assistant: `. That prefix is a contract: `runner.sh`,
/// `matrix_runner.sh`, and `extract_response.sh` all grep `^Assistant:`, and so
/// may anything else someone has scripted around this binary.
fn reply_line(text: &str, interactive: bool) -> String {
    if interactive {
        format!("\x1b[32m\u{276f}\x1b[0m {text}")
    } else {
        format!("Assistant: {text}")
    }
}

/// The share of the context window this turn's prompt occupied, as a suffix for
/// the token line — or nothing at all.
///
/// Empty when the window is unknown, which is the whole discipline: gallium's
/// fallbacks exist so compaction has a threshold, and printing a percentage of
/// one would dress a guess up as a measurement. Empty when the model reported no
/// usage either, since `0%` of a real window is a claim of its own.
fn context_gauge(peak_input_tokens: u64, window: Option<u32>) -> String {
    let Some(window) = window.filter(|w| *w > 0) else {
        return String::new();
    };
    if peak_input_tokens == 0 {
        return String::new();
    }
    let percent = (peak_input_tokens as f64 / window as f64 * 100.0).round() as u64;
    format!(" · {percent}% of {window}")
}

/// The turn's throughput, as a suffix for the token line — or nothing at all
/// when the provider does not measure it (a blocking cloud API cannot).
///
/// `ttft` is the *first* call's, not the sum: it is the wait before the turn
/// showed any sign of life, and summing every call's prefill would report a
/// number nobody sat through. Decode is summed, because that genuinely is how
/// long the turn spent generating.
///
/// Both rates are omitted rather than shown as `0.0` when there is nothing to
/// divide — a rate of zero and an absent measurement look identical on a line
/// someone is reading to compare two configurations.
fn speed_summary(usage: &gallium_agent::TokenUsage) -> String {
    let Some(timing) = usage.timing else {
        return String::new();
    };
    let mut out = format!(" · ttft {:.2}s", timing.ttft.as_secs_f64());
    if let Some(rate) = timing.prefill_rate() {
        out.push_str(&format!(" · prefill {rate:.1} tok/s"));
    }
    if let Some(rate) = timing.decode_rate() {
        out.push_str(&format!(" · decode {rate:.1} tok/s"));
    }
    out
}

/// A path as the user would recognize it: relative to the working directory
/// when it is inside it, absolute otherwise (a global skills dir, say).
fn display_path(path: &std::path::Path, working_dir: &str) -> String {
    path.strip_prefix(working_dir)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Bytes, rounded, for a line the user skims. The size is worth showing: a
/// large context file is a real bite out of a local model's window.
fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    }
}

/// What a Ctrl-C means, which depends on what the REPL is doing when it arrives.
#[derive(Debug, PartialEq, Eq)]
enum Interrupt {
    /// Nothing is running. Ctrl-C at the prompt quits, as it always has — a
    /// Ctrl-C that no longer quits is its own kind of surprise.
    Quit,
    /// A turn is running and has not been asked to stop yet.
    StopTurn,
    /// A turn was asked to stop and is still going. Cancellation is prompt but
    /// not instantaneous — an OpenAI round trip runs to completion — so an
    /// impatient second press needs a way out.
    QuitImpatiently,
}

/// The cancel button, for the one turn that is running right now.
///
/// The REPL is blocked in `read_line` or inside `run_turn`, never in a position
/// to poll for a keypress, so the signal handler has to decide on its own what
/// a Ctrl-C means. This is the state it decides from: whether a turn is armed,
/// and whether that turn has already been asked to stop.
#[derive(Clone, Default)]
struct TurnSlot(std::sync::Arc<parking_lot::Mutex<Option<gallium_agent::CancellationToken>>>);

impl TurnSlot {
    /// Arm the button for a turn that is about to run.
    fn enter(&self, token: gallium_agent::CancellationToken) {
        *self.0.lock() = Some(token);
    }

    /// Disarm it once the turn is over, so the next Ctrl-C quits rather than
    /// cancelling a turn that already finished.
    ///
    /// Returns whether the turn had been asked to stop. A turn can be asked and
    /// still finish: the request only takes effect at the next checkpoint, and
    /// one that lands after the last one is simply too late. That is worth
    /// reporting rather than swallowing — the handler has already said
    /// "stopping…", and a full reply appearing underneath it needs explaining.
    fn leave(&self) -> bool {
        self.0
            .lock()
            .take()
            .is_some_and(|token| token.is_cancelled())
    }

    /// Decide what this Ctrl-C means, and cancel the turn if that is what it
    /// means. The token doubles as the record of having asked once, so a second
    /// press is distinguishable from the first without a separate flag.
    fn interrupt(&self) -> Interrupt {
        match self.0.lock().as_ref() {
            None => Interrupt::Quit,
            Some(token) if token.is_cancelled() => Interrupt::QuitImpatiently,
            Some(token) => {
                token.cancel();
                Interrupt::StopTurn
            }
        }
    }
}

/// The exit status a shell expects from a process killed by SIGINT (128 + 2).
/// Handling the signal means reporting it ourselves on platforms that cannot
/// simply re-raise it, or a script wrapping this binary sees a clean exit where
/// it used to see an interrupt.
const EXIT_INTERRUPTED: i32 = 130;

/// Die the way an unhandled Ctrl-C did, from inside the handler.
///
/// Deliberately **not** `std::process::exit`. That runs C++ static destructors,
/// and llama.cpp's Metal backend frees its device from one — while the main
/// thread is still blocked in `read` and ggml's own residency-set thread is
/// still running. It asserts and then hangs, on a loaded model with nothing
/// else wrong:
///
/// ```text
/// ggml-metal-device.m:622: GGML_ASSERT([rsets->data count] == 0) failed
///   ggml_metal_device_free … __cxa_finalize_ranges … exit
/// ```
///
/// Exiting normally through `main` (`/quit`, Ctrl-D) is unaffected and stays as
/// it was — the difference is the teardown running on the handler's thread
/// under a live process. The default SIGINT disposition never had the problem
/// because it runs no teardown at all, so restore it and re-raise: identical
/// death, the same 130 the shell already saw, and still a *signalled* exit, so
/// a loop wrapping this binary breaks out of it the way it used to.
fn die_from_interrupt() -> ! {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
        libc::raise(libc::SIGINT);
    }
    // On unix the signal has already landed. This is the whole story on
    // Windows, where ctrlc installs a console handler rather than a SIGINT one.
    std::process::exit(EXIT_INTERRUPTED);
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
    // With no `--config`, fall back to `~/.config/gallium/config.toml`. Without
    // it, `gallium` is only configured in whichever directory happens to hold a
    // TOML, and the same command means something different one directory over.
    let config_path = config_path
        .map(PathBuf::from)
        .or_else(config::default_config_path);
    let (file_config, config_dir) = match &config_path {
        Some(path) => {
            let file = config::FileConfig::load(path).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            let dir = path.parent().map(|p| p.to_path_buf());
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
        run_repl(config, config_path);
    }
}

/// Serve the agent over JSON-RPC on stdio until the client disconnects.
fn run_app_server(config: EnvConfig) {
    gallium_agent::appserver::run_stdio(gallium_agent::appserver::ServerConfig {
        model_path: config.model_path,
        mmproj_path: config.mmproj_path,
        base_url: config.base_url,
        model: config.model,
        api_key: config.api_key,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        reasoning_effort: config.reasoning_effort,
        inference_engine: config.inference_engine,
        tokenizer_path: config.tokenizer_path,
        gpu_layers: config.gpu_layers,
        cpu_moe: config.cpu_moe,
        profile: config.profile,
        max_iterations: Some(config.max_react_iterations),
        context_window: config.context_window,
        skill_paths: config.skill_paths,
        trace_dir: config.trace_dir,
    });
}

/// `config_path` is only for the banner: a config picked up from the home
/// directory rather than named on the command line is otherwise invisible, and
/// settings arriving from a file nobody mentioned is the confusing case.
fn run_repl(config: EnvConfig, config_path: Option<PathBuf>) {
    let EnvConfig {
        model_path,
        mmproj_path,
        base_url,
        model,
        api_key,
        working_dir,
        max_tokens,
        context_window,
        max_react_iterations,
        temperature,
        reasoning_effort,
        inference_engine,
        tokenizer_path,
        gpu_layers,
        cpu_moe,
        profile,
        system_prompt,
        skill_paths,
        approval_policy,
        trace_dir,
        mcp_servers,
    } = config;

    let client = create_provider(
        model_path.clone(),
        mmproj_path.clone(),
        base_url.clone(),
        model.clone(),
        api_key.clone(),
        temperature,
        max_tokens,
        reasoning_effort,
        inference_engine.clone(),
        tokenizer_path.clone(),
        gpu_layers,
        cpu_moe,
        profile,
    )
    .expect("Failed to create LLM provider");

    // Now that the provider exists it can be asked what window it runs in — the
    // model's own metadata beats any guess made from the config alone.
    let context_window = gallium_agent::resolve_context_window(
        context_window,
        client.context_window(),
        if model_path.is_some() {
            gallium_agent::LOCAL_CONTEXT_WINDOW
        } else {
            gallium_agent::DEFAULT_CONTEXT_WINDOW
        },
    );

    // Create tool registry
    let skill_registry = std::sync::Arc::new(gallium_agent::skill::SkillRegistry::new());
    let mut skill_sources =
        gallium_agent::skill::load_skills(&skill_registry, std::path::Path::new(&working_dir));
    // Additional SKILL.md dirs from the config's `skillPaths`.
    for dir in &skill_paths {
        match skill_registry.load_from_dir(dir) {
            0 => {}
            count => skill_sources.push(gallium_agent::skill::SkillSource {
                dir: dir.clone(),
                count,
            }),
        }
    }

    // What the project says about itself: AGENTS.md, else CLAUDE.md.
    let context_file =
        gallium_agent::project::find_context_file(std::path::Path::new(&working_dir));
    // One broker for the session, carrying the configured policy. The REPL has
    // a terminal, so a tier whose rule is `ask` becomes a prompt here rather
    // than a refusal.
    let broker = std::sync::Arc::new(gallium_agent::approval::ApprovalBroker::new(
        approval_policy,
    ));
    let session = std::sync::Arc::new(gallium_agent::tool::ToolSession::with_broker(
        std::path::PathBuf::from(&working_dir),
        std::sync::Arc::clone(&broker),
    ));
    let mut tool_registry = gallium_agent::tool::create_default_registry_with_session(
        std::path::PathBuf::from(&working_dir),
        std::sync::Arc::clone(&skill_registry),
        session,
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

    // Off unless asked for: a trace holds every byte the model saw, including
    // whatever the tools read out of the workspace.
    let trace = gallium_agent::TraceSession::from_env(
        trace_dir,
        gallium_agent::TraceMeta::new(
            gallium_agent::TraceMeta::engine_label(model_path.as_deref(), inference_engine.clone()),
            model_path.clone().unwrap_or_else(|| model.clone()),
            working_dir.clone(),
            broker.policy(),
        ),
        Some(std::sync::Arc::clone(&broker)),
    );

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
        match &config_path {
            Some(path) => eprintln!("Config: {}", path.display()),
            None => eprintln!("Config: none (no --config, no ~/.config/gallium/config.toml)"),
        }
        // What will happen without asking is not something to leave anyone
        // guessing at, least of all when a config they did not write set it.
        eprintln!("Approvals: {}", broker.policy());
        // Said out loud for the same reason: every turn is about to be written
        // to disk in full, and that should not be a surprise.
        if let Some(trace) = &trace {
            eprintln!("Traces: {}", display_path(trace.dir(), &working_dir));
        }
        // What the agent read before the first turn. Both are silent when absent,
        // and a missing skill dir or an unread CLAUDE.md looks exactly like a
        // model ignoring them — so say which it was.
        match &context_file {
            Some(ctx) => eprintln!(
                "Context: {} ({})",
                display_path(&ctx.path, &working_dir),
                human_size(ctx.content.len())
            ),
            None => eprintln!("Context: none (no AGENTS.md or CLAUDE.md here)"),
        }
        if skill_sources.is_empty() {
            eprintln!("Skills: none");
        } else {
            let where_from: Vec<String> = skill_sources
                .iter()
                .map(|s| format!("{} from {}", s.count, display_path(&s.dir, &working_dir)))
                .collect();
            eprintln!("Skills: {}", where_from.join(", "));
        }
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
    // A second system message rather than an append to the first: the operator's
    // prompt and the project's instructions come from different people, and a
    // model that has to weigh them should be able to see the seam.
    if let Some(ctx) = &context_file {
        messages.push(ChatMessage::system(ctx.as_system_message()));
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    // Ctrl-C, on a terminal only. Piped stdin keeps the default SIGINT
    // disposition, so the testsuite and any script driving this binary behave
    // exactly as before.
    //
    // The handler runs on a thread of ctrlc's own — not in async-signal context —
    // so it is free to take a lock, print, and exit.
    let interrupts = TurnSlot::default();
    if is_interactive {
        let slot = interrupts.clone();
        if let Err(e) = ctrlc::set_handler(move || match slot.interrupt() {
            // Say so at once: the turn stops at its next checkpoint, which for a
            // cloud round trip is not for a while, and a Ctrl-C that appears to
            // do nothing invites a second one.
            Interrupt::StopTurn => eprintln!("\n\x1b[90m⏹ stopping…\x1b[0m"),
            Interrupt::Quit | Interrupt::QuitImpatiently => {
                eprintln!();
                die_from_interrupt();
            }
        }) {
            // Not fatal: without a handler Ctrl-C keeps killing the process,
            // which is the behavior this replaces. Worth one line, though —
            // silently losing it would look like cancellation being ignored.
            eprintln!("Warning: Ctrl-C will kill the process (no handler: {e})");
        }
    }

    // Peak prompt size of the previous turn, which decides whether this one
    // needs history compacted first. Same policy the app-server applies.
    let mut last_input_tokens: u64 = 0;
    // Colors the next prompt, the one thing the `pure` prompt says with color.
    let mut last_turn_failed = false;

    // Read line by line rather than iterating `.lines()`, so a prompt can be
    // drawn before each read. Piped input gets no prompt at all, which keeps
    // the testsuite's captured output exactly as it was.
    let mut line = String::new();
    loop {
        if is_interactive {
            draw_prompt(last_turn_failed);
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF: the pipe ended, or the user pressed Ctrl-D.
            Ok(_) => {}
            Err(_) => break,
        }
        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/quit" || input == "/exit" {
            break;
        }

        if input == "/reset" {
            messages.truncate(1); // Keep system prompt
            last_input_tokens = 0;
            last_turn_failed = false;
            eprintln!("Conversation reset.");
            continue;
        }

        // `@image:<path>` markers become attachments; everything else is text.
        // Paths resolve against the agent's working dir, not the process cwd.
        //
        // A bad attachment ends this turn and not the session: the user is at a
        // prompt and can retype the path, and a piped run gets the reason on
        // stderr next to the turn that did not happen.
        let input =
            match gallium_agent::input::parse_line(&input, std::path::Path::new(&working_dir)) {
                Ok(input) => input,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    last_turn_failed = true;
                    continue;
                }
            };

        if is_interactive {
            eprintln!("\x1b[90mThinking...\x1b[0m");
            if !input.media.is_empty() {
                // Named per kind: "2 attached" would not say whether the clip
                // you meant to include actually made it.
                let kinds: Vec<&str> = input.media.iter().map(|m| m.kind()).collect();
                eprintln!("\x1b[90m🖼  attached: {}\x1b[0m", kinds.join(", "));
            }
        }

        let renderer = TerminalRenderer;
        let observer: Option<&dyn gallium_agent::AgentObserver> = if is_interactive {
            Some(&renderer)
        } else {
            None
        };
        // A fresh token per turn, armed for the Ctrl-C handler: cancelling one
        // turn must not leave the next one born cancelled. Piped input gets no
        // context at all, since with no handler installed nothing could set it —
        // an honest `None` rather than a token nobody can reach.
        let turn_context = is_interactive
            .then(|| gallium_agent::TurnContext::new(gallium_agent::CancellationToken::new()));
        if let Some(ctx) = &turn_context {
            interrupts.enter(ctx.cancellation.clone());
        }

        let setup = gallium_agent::TurnSetup {
            provider: client.as_ref(),
            tools: &tool_registry,
            skills: Some(&skill_registry),
            max_iterations: Some(max_react_iterations),
            context_window: context_window.effective,
            observer,
            context: turn_context.as_ref(),
            trace: trace.as_ref(),
            // The REPL has no id for a turn, so the session numbers them.
            turn_id: None,
        };

        let result =
            gallium_agent::run_turn(&setup, &mut messages, last_input_tokens, input.clone());
        // Disarm before printing: from here on a Ctrl-C should quit, not cancel
        // a turn that has already returned.
        let stop_arrived_too_late = interrupts.leave();

        match result {
            Ok(outcome) => {
                if outcome.compacted > 0 {
                    eprintln!(
                        "\x1b[90m🗜  compacted history: dropped {} messages (last turn peaked at {} tokens, window {})\x1b[0m",
                        outcome.compacted, last_input_tokens, context_window.effective
                    );
                }
                if let Some(ref thinking) = outcome.reasoning {
                    eprintln!("\x1b[90m💭 {}\x1b[0m", thinking);
                }
                println!("{}", reply_line(&outcome.text, is_interactive));
                if outcome.usage.total_tokens > 0 {
                    eprintln!(
                        "\x1b[90m📊 tokens: in={}, out={}, total={}{}{}\x1b[0m",
                        outcome.usage.input_tokens,
                        outcome.usage.output_tokens,
                        outcome.usage.total_tokens,
                        context_gauge(outcome.usage.peak_input_tokens, context_window.known),
                        speed_summary(&outcome.usage),
                    );
                }
                last_input_tokens = outcome.usage.peak_input_tokens;
                last_turn_failed = false;
                // The handler said "stopping…" and then a whole reply arrived
                // anyway. Say why, rather than leaving the two contradicting
                // each other: the turn was past its last checkpoint.
                if stop_arrived_too_late {
                    eprintln!("\x1b[90m⏹ too late to stop — the turn had already finished\x1b[0m");
                }
            }
            // Stopping on request is not a failure, so it is not reported as
            // one and the prompt does not turn red. `run_turn` has already
            // rolled history back to before the prompt, so the conversation is
            // exactly as it was and the next turn is unaffected.
            Err(gallium_agent::AgentError::Cancelled) => {
                eprintln!("\x1b[90m⏹ stopped\x1b[0m");
                last_turn_failed = false;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                last_turn_failed = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use gallium_agent::tool::ToolResult;
    use gallium_agent::AgentEvent;

    #[test]
    fn the_gauge_reports_the_share_of_a_known_window() {
        assert_eq!(context_gauge(3382, Some(12800)), " · 26% of 12800");
        assert_eq!(context_gauge(12800, Some(12800)), " · 100% of 12800");
    }

    /// The discipline the whole feature rests on: no window, no gauge. Gallium's
    /// fallbacks exist to give compaction a threshold, and a percentage of one
    /// would read as a measurement.
    #[test]
    fn the_gauge_says_nothing_when_the_window_is_a_guess() {
        assert_eq!(context_gauge(3382, None), "");
    }

    /// `0%` of a real window is a claim too — that the prompt was empty — and a
    /// provider that reports no usage has not said that.
    #[test]
    fn the_gauge_says_nothing_when_nothing_was_measured() {
        assert_eq!(context_gauge(0, Some(12800)), "");
        // Compaction disabled: no threshold to be a share of.
        assert_eq!(context_gauge(3382, Some(0)), "");
    }

    /// At the prompt with nothing running, Ctrl-C quits — the one behavior that
    /// has to survive taking over the signal.
    #[test]
    fn ctrl_c_with_no_turn_running_quits() {
        let slot = TurnSlot::default();
        assert_eq!(slot.interrupt(), Interrupt::Quit);
    }

    /// The first Ctrl-C during a turn cancels the token the turn itself is
    /// holding — a clone shares the flag, so the running turn sees it.
    #[test]
    fn ctrl_c_during_a_turn_stops_that_turn() {
        let slot = TurnSlot::default();
        let held_by_the_turn = gallium_agent::CancellationToken::new();
        slot.enter(held_by_the_turn.clone());

        assert_eq!(slot.interrupt(), Interrupt::StopTurn);
        assert!(held_by_the_turn.is_cancelled());
    }

    /// A turn that will not stop promptly — a cloud round trip has no
    /// interruption point — must not trap the user in it.
    #[test]
    fn a_second_ctrl_c_during_the_same_turn_quits() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());

        assert_eq!(slot.interrupt(), Interrupt::StopTurn);
        assert_eq!(slot.interrupt(), Interrupt::QuitImpatiently);
    }

    /// Once the turn is over the button is disarmed, so the next Ctrl-C quits
    /// rather than cancelling a turn that already returned.
    #[test]
    fn ctrl_c_after_a_turn_ends_quits_again() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());
        assert_eq!(slot.interrupt(), Interrupt::StopTurn);

        slot.leave();

        assert_eq!(slot.interrupt(), Interrupt::Quit);
    }

    /// An ordinary turn nobody interrupted has nothing to report.
    #[test]
    fn leaving_an_uninterrupted_turn_reports_nothing() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());

        assert!(!slot.leave());
    }

    /// A turn asked to stop that finished anyway — the request landed after its
    /// last checkpoint. The REPL needs to know, because the handler has already
    /// printed "stopping…" and a full reply is about to appear under it.
    ///
    /// This is also the narrow completion window: between `run_turn` returning
    /// and the slot being disarmed, an interrupt still reads as `StopTurn`. It
    /// cannot stop anything by then, so the only question is whether the REPL
    /// notices — and it does.
    #[test]
    fn leaving_a_turn_that_was_asked_to_stop_reports_it() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());
        assert_eq!(slot.interrupt(), Interrupt::StopTurn);

        assert!(slot.leave());
    }

    /// Reporting is not the same as staying armed: whatever `leave` returns, the
    /// next Ctrl-C has to quit.
    #[test]
    fn leaving_disarms_whether_or_not_it_reports() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());
        slot.interrupt();

        assert!(slot.leave());
        assert_eq!(slot.interrupt(), Interrupt::Quit);
    }

    /// Each turn gets its own token, so cancelling one does not leave the next
    /// one born cancelled.
    #[test]
    fn a_cancelled_turn_does_not_poison_the_next_one() {
        let slot = TurnSlot::default();
        slot.enter(gallium_agent::CancellationToken::new());
        slot.interrupt();
        slot.leave();

        let next_turn = gallium_agent::CancellationToken::new();
        slot.enter(next_turn.clone());

        assert!(!next_turn.is_cancelled());
        assert_eq!(slot.interrupt(), Interrupt::StopTurn);
    }

    /// The handler runs on ctrlc's thread while the REPL waits on the turn, so
    /// the state it decides from has to be safely shared.
    #[test]
    fn the_slot_is_usable_from_the_handler_thread() {
        let slot = TurnSlot::default();
        let held_by_the_turn = gallium_agent::CancellationToken::new();
        slot.enter(held_by_the_turn.clone());

        let handler = slot.clone();
        std::thread::spawn(move || handler.interrupt())
            .join()
            .unwrap();

        assert!(held_by_the_turn.is_cancelled());
    }

    /// The prompt is one glyph and a space. Anything more — a path, a model
    /// name, a token count — is something the banner already said once.
    #[test]
    fn the_prompt_is_a_single_glyph() {
        let prompt = prompt_string(false);
        assert!(prompt.ends_with("\u{276f}\x1b[0m "), "{prompt:?}");
        assert_eq!(prompt.chars().filter(|c| !c.is_ascii()).count(), 1);
    }

    /// The reply echoes the prompt on a terminal, and keeps the machine-readable
    /// prefix when piped — the testsuite greps `^Assistant:`, and breaking that
    /// silently would look like every test failing at once.
    #[test]
    fn a_reply_is_marked_for_whoever_is_reading_it() {
        let interactive = reply_line("hello", true);
        assert!(interactive.contains('\u{276f}'), "{interactive:?}");
        assert!(interactive.ends_with(" hello"));
        assert!(!interactive.contains("Assistant:"));

        assert_eq!(reply_line("hello", false), "Assistant: hello");
    }

    /// Colour says one thing: whether the last turn failed. A red prompt is how
    /// the user notices an error they scrolled past.
    #[test]
    fn the_prompt_turns_red_after_a_failed_turn() {
        assert!(prompt_string(false).starts_with("\x1b[35m"));
        assert!(prompt_string(true).starts_with("\x1b[31m"));
    }

    #[test]
    fn a_tool_call_renders_its_arguments_not_their_payload() {
        let args = serde_json::json!({ "file_path": "src/main.rs" });
        let line = TerminalRenderer::render(&AgentEvent::ToolStarted {
            call_id: "c1",
            name: "read",
            arguments: &args,
        })
        .expect("tool starts are rendered");
        assert!(line.contains("read(file_path=src/main.rs)"), "got: {line}");
    }

    #[test]
    fn a_large_argument_is_summarized_rather_than_dumped() {
        let args = serde_json::json!({ "content": "x".repeat(5000) });
        let line = TerminalRenderer::render(&AgentEvent::ToolStarted {
            call_id: "c1",
            name: "write",
            arguments: &args,
        })
        .unwrap();
        assert!(line.contains("(5000 chars)"), "got: {line}");
        assert!(
            line.len() < 200,
            "a 5000-char write must not fill the terminal"
        );
    }

    #[test]
    fn a_completion_renders_the_display_form_not_the_model_text() {
        let result = ToolResult::text("the entire file body".to_string())
            .displaying("Read 3 lines from a.txt".to_string());
        let line = TerminalRenderer::render(&AgentEvent::ToolCompleted {
            call_id: "c1",
            name: "read",
            arguments: &serde_json::Value::Null,
            result: &result,
        })
        .unwrap();
        assert!(line.contains("Read 3 lines from a.txt"), "got: {line}");
        assert!(!line.contains("entire file body"));
    }

    #[test]
    fn a_failed_tool_renders_distinctly_from_a_successful_one() {
        let ok = ToolResult::text("fine".to_string());
        let bad = ToolResult::error("nope".to_string());
        let ok_line = TerminalRenderer::render(&AgentEvent::ToolCompleted {
            call_id: "c1",
            name: "read",
            arguments: &serde_json::Value::Null,
            result: &ok,
        })
        .unwrap();
        let bad_line = TerminalRenderer::render(&AgentEvent::ToolCompleted {
            call_id: "c2",
            name: "read",
            arguments: &serde_json::Value::Null,
            result: &bad,
        })
        .unwrap();
        assert!(ok_line.contains('✓') && !ok_line.contains('✗'));
        assert!(bad_line.contains('✗'), "got: {bad_line}");
    }

    #[test]
    fn events_the_repl_prints_itself_render_nothing() {
        // An untimed provider's usage: the counts go on the turn's 📊 line, and
        // there is no speed to report, so the call prints nothing at all.
        let usage = gallium_agent::TokenUsage::single(10, 2, 12);
        assert!(TerminalRenderer::render(&AgentEvent::Usage { usage: &usage }).is_none());
        assert!(TerminalRenderer::render(&AgentEvent::TurnCompleted { text: "hi" }).is_none());
    }

    #[test]
    fn a_timed_call_reports_its_own_speed() {
        let usage = gallium_agent::TokenUsage::timed(
            200,
            41,
            241,
            std::time::Duration::from_millis(500),
            std::time::Duration::from_secs(4),
        );
        let line = TerminalRenderer::render(&AgentEvent::Usage { usage: &usage }).unwrap();
        assert!(line.contains("ttft 0.50s"), "got: {line}");
        assert!(line.contains("prefill 200 tok 400.0 tok/s"), "got: {line}");
        // 40, not 41: the first token was prefill's output.
        assert!(line.contains("decode 40 tok 10.0 tok/s"), "got: {line}");
    }

    #[test]
    fn the_turn_line_reports_the_first_calls_wait_and_the_summed_rates() {
        let mut usage = gallium_agent::TokenUsage::timed(
            100,
            11,
            111,
            std::time::Duration::from_millis(500),
            std::time::Duration::from_secs(1),
        );
        usage.add(&gallium_agent::TokenUsage::timed(
            300,
            21,
            321,
            std::time::Duration::from_millis(1500),
            std::time::Duration::from_secs(1),
        ));
        let line = speed_summary(&usage);
        // The wait, not the sum of the waits.
        assert!(line.contains("ttft 0.50s"), "got: {line}");
        assert!(line.contains("prefill 200.0 tok/s"), "got: {line}");
        assert!(line.contains("decode 15.0 tok/s"), "got: {line}");
    }

    #[test]
    fn a_provider_that_does_not_time_itself_adds_nothing_to_the_turn_line() {
        assert_eq!(
            speed_summary(&gallium_agent::TokenUsage::single(100, 20, 120)),
            ""
        );
    }

    #[test]
    fn newlines_in_an_argument_do_not_break_the_single_line_layout() {
        assert_eq!(summarize_arg(&serde_json::json!("a\nb")), "a⏎b");
    }
}
