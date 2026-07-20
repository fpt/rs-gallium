//! MCP server over HTTP with SSE (Streamable HTTP transport).
//!
//! Accepts JSON-RPC 2.0 requests via POST and returns responses as
//! `text/event-stream` (Server-Sent Events). Tracks sessions via
//! the `Mcp-Session-Id` header.

use crate::mcp::*;
use crate::mcp_server::McpServer;
use crate::tool::ToolAccess;

/// An MCP server that serves tools over HTTP with SSE responses.
pub struct McpHttpServer<'a> {
    handler: McpServer<'a>,
    session_id: String,
}

impl<'a> McpHttpServer<'a> {
    pub fn new(tools: &'a dyn ToolAccess) -> Self {
        Self {
            handler: McpServer::new(tools),
            session_id: generate_session_id(),
        }
    }

    pub fn with_info(mut self, name: &str, version: &str) -> Self {
        self.handler = self.handler.with_info(name, version);
        self
    }

    /// The session ID assigned to this server instance.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Run the HTTP server, blocking until the server is shut down.
    ///
    /// Binds to `addr` (e.g. `"127.0.0.1:0"` for a random port).
    pub fn run(&self, addr: &str) -> Result<(), McpHttpError> {
        let server = tiny_http::Server::http(addr)
            .map_err(|e| McpHttpError::Bind(format!("{}", e)))?;

        tracing::info!(
            "MCP HTTP server listening on http://{}",
            server.server_addr()
        );

        for request in server.incoming_requests() {
            self.handle_http_request(request);
        }

        Ok(())
    }

    /// Run the HTTP server in a background thread, returning the bound address.
    ///
    /// Use `addr` `"127.0.0.1:0"` for a random port. The returned address
    /// includes the actual port assigned by the OS.
    pub fn run_background(
        tools: &'static (dyn ToolAccess + Send + Sync),
        addr: &str,
    ) -> Result<(std::net::SocketAddr, McpHttpHandle), McpHttpError> {
        let server = tiny_http::Server::http(addr)
            .map_err(|e| McpHttpError::Bind(format!("{}", e)))?;

        let bound_addr = match server.server_addr() {
            tiny_http::ListenAddr::IP(addr) => addr,
            _ => {
                return Err(McpHttpError::Bind(
                    "Unix socket not supported".to_string(),
                ))
            }
        };

        let session_id = generate_session_id();
        let session_id_clone = session_id.clone();

        let join = std::thread::spawn(move || {
            let handler = McpServer::new(tools);
            for request in server.incoming_requests() {
                dispatch_http_request(&handler, &session_id_clone, request);
            }
        });

        Ok((
            bound_addr,
            McpHttpHandle {
                _join: join,
                session_id,
            },
        ))
    }

    fn handle_http_request(&self, request: tiny_http::Request) {
        dispatch_http_request(&self.handler, &self.session_id, request);
    }
}

/// Process a single HTTP request. Takes `request` by value since
/// `tiny_http::Request::respond` consumes self.
fn dispatch_http_request(
    handler: &McpServer<'_>,
    session_id: &str,
    mut request: tiny_http::Request,
) {
    // Only accept POST
    if *request.method() != tiny_http::Method::Post {
        let resp = tiny_http::Response::from_string("Method Not Allowed")
            .with_status_code(405)
            .with_header(
                tiny_http::Header::from_bytes("Allow", "POST").unwrap(),
            );
        let _ = request.respond(resp);
        return;
    }

    // Read body
    let mut body = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body) {
        tracing::error!("Failed to read request body: {}", e);
        let _ = request.respond(
            tiny_http::Response::from_string("Bad Request")
                .with_status_code(400),
        );
        return;
    }

    // Parse JSON-RPC
    let rpc_request: JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(e) => {
            let error_resp = JsonRpcResponse::error(
                serde_json::Value::Null,
                PARSE_ERROR,
                format!("Parse error: {}", e),
            );
            let sse = format_sse_event(&error_resp);
            let _ = request.respond(
                tiny_http::Response::from_string(sse)
                    .with_header(content_type_sse())
                    .with_header(session_id_header(session_id)),
            );
            return;
        }
    };

    // Dispatch
    match handler.process(&rpc_request) {
        Some(rpc_response) => {
            let sse = format_sse_event(&rpc_response);
            let _ = request.respond(
                tiny_http::Response::from_string(sse)
                    .with_header(content_type_sse())
                    .with_header(no_cache_header())
                    .with_header(session_id_header(session_id)),
            );
        }
        None => {
            // Notification — 202 Accepted
            let _ = request.respond(
                tiny_http::Response::empty(202)
                    .with_header(session_id_header(session_id)),
            );
        }
    }
}

/// Handle returned by `run_background`.
pub struct McpHttpHandle {
    _join: std::thread::JoinHandle<()>,
    pub session_id: String,
}

// ============================================================================
// Helpers
// ============================================================================

fn content_type_sse() -> tiny_http::Header {
    tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap()
}

fn no_cache_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes("Cache-Control", "no-cache").unwrap()
}

fn session_id_header(id: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(MCP_SESSION_ID_HEADER, id).unwrap()
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum McpHttpError {
    Bind(String),
    Io(std::io::Error),
}

impl std::fmt::Display for McpHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpHttpError::Bind(e) => write!(f, "Failed to bind: {}", e),
            McpHttpError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for McpHttpError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{TaskTool, ToolRegistry};

    // Leak a ToolRegistry so it has 'static lifetime for run_background
    fn make_static_registry() -> &'static ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(TaskTool::new()));
        Box::leak(Box::new(reg))
    }

    #[test]
    fn test_http_server_initialize() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}/", addr);

        let req = serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"}
            }
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&req).unwrap())
            .unwrap();

        assert_eq!(resp.content_type(), "text/event-stream");

        // Check session ID header
        assert!(resp.header(MCP_SESSION_ID_HEADER).is_some());

        let body = resp.into_string().unwrap();
        let rpc_resp = parse_sse_response(&body).unwrap();
        assert!(rpc_resp.error.is_none());

        let result: InitializeResult =
            serde_json::from_value(rpc_resp.result.unwrap()).unwrap();
        assert_eq!(result.protocol_version, PROTOCOL_VERSION);
    }

    #[test]
    fn test_http_server_tools_list() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}/", addr);

        let req = serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list"
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&req).unwrap())
            .unwrap();

        let body = resp.into_string().unwrap();
        let rpc_resp = parse_sse_response(&body).unwrap();
        let result: ToolsListResult =
            serde_json::from_value(rpc_resp.result.unwrap()).unwrap();
        assert_eq!(result.tools.len(), 1);
        assert_eq!(result.tools[0].name, "tasks");
    }

    #[test]
    fn test_http_server_tools_call() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}/", addr);

        let req = serde_json::json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "tasks", "arguments": {"action": "list"}}
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&req).unwrap())
            .unwrap();

        let body = resp.into_string().unwrap();
        let rpc_resp = parse_sse_response(&body).unwrap();
        let result: ToolsCallResult =
            serde_json::from_value(rpc_resp.result.unwrap()).unwrap();
        assert!(result.is_error.is_none());
    }

    #[test]
    fn test_http_server_notification_202() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}/", addr);

        let notif = serde_json::json!({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&serde_json::to_string(&notif).unwrap())
            .unwrap();

        assert_eq!(resp.status(), 202);
    }

    #[test]
    fn test_http_server_method_not_allowed() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}/", addr);

        let result = ureq::get(&url).call();
        match result {
            Err(ureq::Error::Status(405, _)) => {} // expected
            other => panic!("Expected 405, got: {:?}", other),
        }
    }
}
