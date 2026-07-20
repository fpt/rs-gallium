//! MCP client over HTTP with SSE (Streamable HTTP transport).
//!
//! Sends JSON-RPC 2.0 requests via HTTP POST and parses SSE responses.
//! Wraps discovered tools as `ToolHandler` for `ToolRegistry` integration.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::mcp::*;
use crate::tool::ToolHandler;
use crate::AgentError;

/// An MCP client that connects to an MCP server over HTTP (Streamable HTTP transport).
///
/// Use `McpHttpClient::connect(url)` to connect and perform the MCP handshake.
/// Then call `tool_handlers()` to get `ToolHandler` wrappers for each remote tool.
pub struct McpHttpClient {
    url: String,
    session_id: Mutex<Option<String>>,
    next_id: AtomicU64,
    tools: Mutex<Vec<ToolInfo>>,
}

impl McpHttpClient {
    /// Connect to an MCP server at the given URL and perform the initialization handshake.
    pub fn connect(url: &str) -> Result<Arc<Self>, AgentError> {
        let client = Arc::new(Self {
            url: url.trim_end_matches('/').to_string(),
            session_id: Mutex::new(None),
            next_id: AtomicU64::new(1),
            tools: Mutex::new(Vec::new()),
        });

        client.do_initialize()?;
        client.do_discover_tools()?;

        Ok(client)
    }

    /// Send a JSON-RPC request and wait for the response.
    fn send_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, AgentError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let body = serde_json::to_string(&request)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize error: {}", e)))?;

        let mut req = ureq::post(&self.url)
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream, application/json");

        // Include session ID if we have one
        if let Some(ref sid) = *self.session_id.lock() {
            req = req.set(MCP_SESSION_ID_HEADER, sid);
        }

        let response = req.send_string(&body).map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let err_body = resp
                    .into_string()
                    .unwrap_or_else(|_| "Unable to read error body".to_string());
                AgentError::NetworkError(format!("HTTP {} from MCP server: {}", code, err_body))
            }
            ureq::Error::Transport(t) => {
                AgentError::NetworkError(format!("HTTP transport error: {}", t))
            }
        })?;

        // Capture session ID from response
        if let Some(sid) = response.header(MCP_SESSION_ID_HEADER) {
            *self.session_id.lock() = Some(sid.to_string());
        }

        let content_type = response.content_type().to_string();
        let body = response.into_string().map_err(|e| {
            AgentError::NetworkError(format!("Failed to read MCP server response: {}", e))
        })?;

        // Parse response based on content type
        let rpc_response = if content_type.contains("text/event-stream") {
            parse_sse_response(&body).map_err(|e| {
                AgentError::ParseError(format!("Failed to parse SSE response: {}", e))
            })?
        } else {
            serde_json::from_str(&body).map_err(|e| {
                AgentError::ParseError(format!("Failed to parse JSON response: {}", e))
            })?
        };

        if let Some(err) = rpc_response.error {
            return Err(AgentError::InternalError(format!(
                "MCP error ({}): {}",
                err.code, err.message
            )));
        }

        rpc_response
            .result
            .ok_or_else(|| AgentError::InternalError("Empty result from MCP server".to_string()))
    }

    /// Send a notification (no response expected).
    fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), AgentError> {
        let notification = JsonRpcRequest::notification(method, params);
        let body = serde_json::to_string(&notification)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize error: {}", e)))?;

        let mut req = ureq::post(&self.url)
            .set("Content-Type", "application/json");

        if let Some(ref sid) = *self.session_id.lock() {
            req = req.set(MCP_SESSION_ID_HEADER, sid);
        }

        // Notifications may return 202 or 200 — we don't care about the body
        req.send_string(&body).map_err(|e| match e {
            ureq::Error::Status(code, _) => {
                AgentError::NetworkError(format!("HTTP {} sending notification", code))
            }
            ureq::Error::Transport(t) => {
                AgentError::NetworkError(format!("HTTP transport error: {}", t))
            }
        })?;

        Ok(())
    }

    fn do_initialize(&self) -> Result<(), AgentError> {
        let params = serde_json::to_value(InitializeParams {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities {},
            client_info: Implementation {
                name: "kessel-cli".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        })
        .map_err(|e| AgentError::InternalError(format!("JSON error: {}", e)))?;

        let result = self.send_request("initialize", Some(params))?;
        let init_result: InitializeResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid initialize result: {}", e)))?;

        tracing::info!(
            "MCP HTTP server: {} v{}",
            init_result.server_info.name,
            init_result.server_info.version
        );

        self.send_notification("notifications/initialized", None)?;

        Ok(())
    }

    fn do_discover_tools(&self) -> Result<(), AgentError> {
        let result = self.send_request("tools/list", None)?;
        let list_result: ToolsListResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid tools/list result: {}", e)))?;

        tracing::info!("Discovered {} MCP tools (HTTP)", list_result.tools.len());
        // Tool descriptions are written for models, not humans — keep them out of
        // the normal log; available with RUST_LOG=debug.
        for tool in &list_result.tools {
            tracing::debug!("  - {}: {}", tool.name, tool.description);
        }

        *self.tools.lock() = list_result.tools;

        Ok(())
    }

    /// Call a tool on the remote MCP server.
    pub fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, AgentError> {
        let params = serde_json::to_value(ToolsCallParams {
            name: name.to_string(),
            arguments,
        })
        .map_err(|e| AgentError::InternalError(format!("JSON error: {}", e)))?;

        let result = self.send_request("tools/call", Some(params))?;
        let call_result: ToolsCallResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid tools/call result: {}", e)))?;

        let text = call_result
            .content
            .iter()
            .filter_map(|c| match c {
                ToolContent::Text { text } => Some(text.as_str()),
            })
            .collect::<Vec<_>>()
            .join("\n");

        if call_result.is_error == Some(true) {
            return Err(AgentError::InternalError(text));
        }

        Ok(text)
    }

    /// Get the list of tools discovered from the remote server.
    pub fn tool_infos(&self) -> Vec<ToolInfo> {
        self.tools.lock().clone()
    }

    /// Create `ToolHandler` wrappers for all remote tools.
    pub fn tool_handlers(self: &Arc<Self>) -> Vec<Box<dyn ToolHandler>> {
        let tools = self.tools.lock();
        tools
            .iter()
            .map(|info| {
                Box::new(McpHttpRemoteTool {
                    client: Arc::clone(self),
                    info: info.clone(),
                }) as Box<dyn ToolHandler>
            })
            .collect()
    }
}

/// A `ToolHandler` that delegates calls to a remote MCP server via HTTP.
pub struct McpHttpRemoteTool {
    client: Arc<McpHttpClient>,
    info: ToolInfo,
}

impl ToolHandler for McpHttpRemoteTool {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.info.input_schema.clone()
    }

    fn call(&self, args: serde_json::Value) -> Result<crate::tool::ToolResult, AgentError> {
        self.client.call_tool(&self.info.name, args).map(crate::tool::ToolResult::text)
    }
}

// McpHttpRemoteTool is Send + Sync because McpHttpClient uses Mutex internally
unsafe impl Send for McpHttpRemoteTool {}
unsafe impl Sync for McpHttpRemoteTool {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp_server_http::McpHttpServer;
    use crate::tool::{TaskTool, ToolRegistry};

    fn make_static_registry() -> &'static ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(TaskTool::new()));
        Box::leak(Box::new(reg))
    }

    #[test]
    fn test_http_client_connect_and_discover() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}", addr);

        let client = McpHttpClient::connect(&url).unwrap();
        let tools = client.tool_infos();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "tasks");
    }

    #[test]
    fn test_http_client_call_tool() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}", addr);

        let client = McpHttpClient::connect(&url).unwrap();

        // Create a task
        let result = client
            .call_tool("tasks", serde_json::json!({"action": "create", "subject": "HTTP test"}))
            .unwrap();
        assert!(result.contains("HTTP test"));

        // List tasks
        let result = client
            .call_tool("tasks", serde_json::json!({"action": "list"}))
            .unwrap();
        assert!(result.contains("HTTP test"));
    }

    #[test]
    fn test_http_client_tool_handlers() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}", addr);

        let client = McpHttpClient::connect(&url).unwrap();
        let handlers = client.tool_handlers();
        assert_eq!(handlers.len(), 1);
        assert_eq!(handlers[0].name(), "tasks");

        // Call through ToolHandler interface
        let result = handlers[0]
            .call(serde_json::json!({"action": "list"}))
            .unwrap()
            .text;
        assert!(result.contains("No tasks"));
    }

    #[test]
    fn test_http_client_tool_error() {
        let registry = make_static_registry();
        let (addr, _handle) =
            McpHttpServer::run_background(registry, "127.0.0.1:0").unwrap();
        let url = format!("http://{}", addr);

        let client = McpHttpClient::connect(&url).unwrap();
        let result = client.call_tool("nonexistent", serde_json::json!({}));
        assert!(result.is_err());
    }
}
