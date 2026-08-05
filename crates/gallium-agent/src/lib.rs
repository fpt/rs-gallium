//! gallium-agent: the local-agent core.
//!
//! ReAct loop, tool registry, MCP client/server, and the LLM providers
//! (OpenAI, in-process llama.cpp via the `local` feature, and the native candle
//! `gallium` backend). Also hosts the JSON-RPC **app-server** (`appserver`) that
//! exposes the agent as a whole-turn backend over the codex-app-server protocol
//! — what rs-kessel and klein-cli call "ACP", *not* the agentclientprotocol.com
//! standard.
//!
//! This crate is headless: frontends (voice, VM host, etc.) drive it over the
//! app-server protocol rather than linking it in-process.

pub mod approval;
pub mod appserver;
pub mod cancel;
pub mod event;
pub mod github;
pub mod input;
// Shared Gemma native tool-call parsing, used by both local backends.
#[cfg(any(feature = "local", feature = "candle"))]
pub mod gemma;
mod llm;
#[cfg(feature = "candle")]
pub mod llm_candle;
#[cfg(feature = "local")]
pub mod llm_local;
// No feature gate: it depends on nothing, and a client's CI needs it present in
// whatever build it was handed.
pub mod llm_scripted;
pub mod mcp;
pub mod mcp_client;
pub mod mcp_client_http;
pub mod mcp_server;
pub mod mcp_server_http;
mod memory;
pub mod model_downloader;
pub mod project;
#[cfg(feature = "candle")]
pub mod protocol;
pub mod react;
pub mod runtime;
pub mod skill;
pub mod tool;
pub mod trace;

pub use approval::{
    ApprovalBroker, ApprovalDecision, ApprovalOutcome, ApprovalPolicy, ApprovalRecord,
    ApprovalRequest, ApprovalRule, ApprovalSink, RiskLevel,
};
pub use cancel::{CancellationToken, SteerInbox, TurnContext};
pub use event::{AgentEvent, AgentObserver};
pub use input::UserInput;
pub use llm::{
    create_provider, ChatMessage, ChatRole, ImageContent, TokenUsage, LOCAL_CONTEXT_WINDOW,
};
pub use memory::{
    compact_messages, compaction_target, estimate_messages_tokens, resolve_context_window,
    ContextWindow, DEFAULT_CONTEXT_WINDOW,
};
pub use runtime::{run_turn, TurnOutcome, TurnSetup};
pub use trace::{TraceMeta, TraceSession, TurnEnding, TurnTrace};

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

/// Error types for the agent
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    /// The turn's input could not be taken as given — an attachment that would
    /// not load, a marker with no path. The user's to fix, not the agent's to
    /// work around, and never to be swallowed: an image that quietly failed to
    /// attach is indistinguishable from a model that cannot see it.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    /// The turn was stopped on request. Not a failure: the caller asked for it,
    /// and a frontend should say "stopped", not "something went wrong".
    #[error("Cancelled")]
    Cancelled,
}
