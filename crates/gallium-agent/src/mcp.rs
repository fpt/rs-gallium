//! Core MCP (Model Context Protocol) and JSON-RPC 2.0 types.
//!
//! Implements the wire format for MCP protocol version 2024-11-05.

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: &str = "2024-11-05";
pub const JSONRPC_VERSION: &str = "2.0";

// ============================================================================
// JSON-RPC 2.0 types
// ============================================================================

/// A JSON-RPC 2.0 request or notification.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// Absent for notifications.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// Standard JSON-RPC 2.0 error codes
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(serde_json::Value::Number(id.into())),
            method: method.to_string(),
            params,
        }
    }

    pub fn notification(method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: None,
            method: method.to_string(),
            params,
        }
    }

    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ============================================================================
// MCP protocol types
// ============================================================================

/// Parameters for the `initialize` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: Implementation,
}

/// Client capabilities (currently empty — reserved for future use).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {}

/// Name and version of a client or server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
}

/// Result of the `initialize` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: Implementation,
}

/// Server capabilities advertised during initialization.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
}

/// Capability flags for the tools feature.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged", skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// A tool definition as returned by `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Result of the `tools/list` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<ToolInfo>,
}

/// Parameters for the `tools/call` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Result of the `tools/call` request.
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolsCallResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Content block in a tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
}

// ============================================================================
// SSE helpers
// ============================================================================

/// Format a JSON-RPC response as a Server-Sent Events message.
pub fn format_sse_event(response: &JsonRpcResponse) -> String {
    let json = serde_json::to_string(response).unwrap();
    format!("event: message\ndata: {}\n\n", json)
}

/// Parse the first JSON-RPC response from an SSE body.
///
/// Handles both `text/event-stream` (extracts `data:` lines) and plain JSON.
pub fn parse_sse_response(body: &str) -> Result<JsonRpcResponse, serde_json::Error> {
    // Try plain JSON first
    if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(body.trim()) {
        return Ok(resp);
    }

    // Parse SSE: collect data lines from the first complete event
    let data = extract_sse_data(body);
    serde_json::from_str(&data)
}

/// Extract all `data:` content from an SSE body, joining multi-line data.
pub fn extract_sse_data(body: &str) -> String {
    let mut data_parts = Vec::new();
    for line in body.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            data_parts.push(d);
        } else if let Some(d) = line.strip_prefix("data:") {
            data_parts.push(d);
        }
    }
    data_parts.join("\n")
}

/// Generate a session ID for HTTP transport.
///
/// The timestamp alone is not enough: the system clock's resolution is coarser
/// than a nanosecond (notably on macOS), so two sessions opened in quick
/// succession can read the same instant and collide. A per-process counter makes
/// each id unique regardless of how the clock ticks; the pid keeps ids distinct
/// across processes.
pub fn generate_session_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("mcp-{:x}-{:x}-{:x}", std::process::id(), ts, seq)
}

/// Header name for MCP session tracking.
pub const MCP_SESSION_ID_HEADER: &str = "Mcp-Session-Id";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn test_notification_has_no_id() {
        let notif = JsonRpcRequest::notification("notifications/initialized", None);
        assert!(notif.is_notification());
        let json = serde_json::to_string(&notif).unwrap();
        assert!(!json.contains("\"id\""));
    }

    #[test]
    fn test_response_success() {
        let resp = JsonRpcResponse::success(
            serde_json::Value::Number(1.into()),
            serde_json::json!({"tools": []}),
        );
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_response_error() {
        let resp = JsonRpcResponse::error(
            serde_json::Value::Number(1.into()),
            METHOD_NOT_FOUND,
            "Unknown method".to_string(),
        );
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, METHOD_NOT_FOUND);
    }

    #[test]
    fn test_tool_info_roundtrip() {
        let info = ToolInfo {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: ToolInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "read");
        assert_eq!(parsed.input_schema["type"], "object");
    }

    #[test]
    fn test_tool_content_tagged() {
        let content = ToolContent::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"hello\""));
    }

    #[test]
    fn test_format_sse_event() {
        let resp = JsonRpcResponse::success(
            serde_json::Value::Number(1.into()),
            serde_json::json!({"tools": []}),
        );
        let sse = format_sse_event(&resp);
        assert!(sse.starts_with("event: message\n"));
        assert!(sse.contains("data: "));
        assert!(sse.ends_with("\n\n"));

        // The data line should be valid JSON-RPC
        let data = extract_sse_data(&sse);
        let parsed: JsonRpcResponse = serde_json::from_str(&data).unwrap();
        assert!(parsed.error.is_none());
    }

    #[test]
    fn test_parse_sse_response() {
        // SSE format
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
        let resp = parse_sse_response(sse).unwrap();
        assert_eq!(resp.id, serde_json::Value::Number(1.into()));
        assert!(resp.error.is_none());

        // Plain JSON (fallback)
        let json = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"ok\":true}}";
        let resp = parse_sse_response(json).unwrap();
        assert_eq!(resp.id, serde_json::Value::Number(2.into()));
    }

    #[test]
    fn test_generate_session_id() {
        let id1 = generate_session_id();
        let id2 = generate_session_id();
        assert!(id1.starts_with("mcp-"));
        assert_ne!(id1, id2);
    }

    /// Ids must stay unique in a tight burst, where the clock may not advance
    /// between calls — the case that used to collide.
    #[test]
    fn test_generate_session_id_unique_in_burst() {
        let ids: std::collections::HashSet<String> =
            (0..1000).map(|_| generate_session_id()).collect();
        assert_eq!(ids.len(), 1000, "session ids collided within a burst");
    }

    #[test]
    fn test_tools_call_result() {
        let result = ToolsCallResult {
            content: vec![ToolContent::Text {
                text: "line 1\nline 2".to_string(),
            }],
            is_error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("isError"));

        let error_result = ToolsCallResult {
            content: vec![ToolContent::Text {
                text: "oops".to_string(),
            }],
            is_error: Some(true),
        };
        let json = serde_json::to_string(&error_result).unwrap();
        assert!(json.contains("\"isError\":true"));
    }
}
