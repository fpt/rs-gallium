//! Exposes the kessel agent as a whole-turn backend over JSON-RPC on stdio.
//!
//! A driving client (klein's `internal/codex` runner) hands kessel an entire
//! conversation turn and takes back the final text; kessel runs its own ReAct
//! loop, tools, and MCP connections inside that turn. This is the same shape
//! codex's app-server presents, and deliberately the same wire protocol — see
//! `server.rs` for the subset implemented and why.

pub mod rpc;
pub mod server;
pub mod tools;

#[cfg(test)]
mod e2e_tests;

use std::io::BufReader;
use std::sync::Arc;

pub use server::{AppServer, ServerConfig};

/// Serve the agent on stdin/stdout until the client closes the connection.
///
/// Callers must ensure nothing else writes to stdout — logs belong on stderr,
/// or they will corrupt the JSON-RPC stream.
pub fn run_stdio(config: ServerConfig) {
    let conn = rpc::Connection::new(Box::new(std::io::stdout()));
    let handler = Arc::new(AppServer::new(config));
    tracing::info!("kessel app-server listening on stdio");
    rpc::serve(BufReader::new(std::io::stdin()), conn, handler);
    tracing::info!("kessel app-server: client disconnected");
}
