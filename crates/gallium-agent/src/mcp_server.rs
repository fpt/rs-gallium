//! MCP server — exposes a `ToolAccess` as an MCP tool server over stdio.
//!
//! Reads newline-delimited JSON-RPC 2.0 messages from a reader (stdin by default),
//! dispatches to the tool registry, and writes responses to a writer (stdout).

use std::io::{self, BufRead, Write};

use crate::mcp::*;
use crate::tool::ToolAccess;

/// An MCP server that serves tools over a line-delimited JSON-RPC 2.0 transport.
pub struct McpServer<'a> {
    tools: &'a dyn ToolAccess,
    server_name: String,
    server_version: String,
}

impl<'a> McpServer<'a> {
    pub fn new(tools: &'a dyn ToolAccess) -> Self {
        Self {
            tools,
            server_name: "kessel-cli".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn with_info(mut self, name: &str, version: &str) -> Self {
        self.server_name = name.to_string();
        self.server_version = version.to_string();
        self
    }

    /// Process a single JSON-RPC request, returning the response.
    /// Returns `None` for notifications.
    pub fn process(&self, request: &JsonRpcRequest) -> Option<JsonRpcResponse> {
        if request.is_notification() {
            tracing::debug!("MCP notification: {}", request.method);
            return None;
        }
        let id = request.id.clone().unwrap_or(serde_json::Value::Null);
        Some(self.handle_request(id, &request.method, request.params.clone()))
    }

    /// Run the server, reading from stdin and writing to stdout.
    /// Blocks until stdin is closed.
    pub fn run(&self) -> io::Result<()> {
        self.run_with(io::stdin().lock(), io::stdout().lock())
    }

    /// Run the server with custom reader/writer (useful for testing).
    pub fn run_with<R: BufRead, W: Write>(&self, reader: R, mut writer: W) -> io::Result<()> {
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(req) => req,
                Err(e) => {
                    let resp = JsonRpcResponse::error(
                        serde_json::Value::Null,
                        PARSE_ERROR,
                        format!("Parse error: {}", e),
                    );
                    writeln!(writer, "{}", serde_json::to_string(&resp).unwrap())?;
                    writer.flush()?;
                    continue;
                }
            };

            // Notifications don't get responses
            if request.is_notification() {
                tracing::debug!("MCP notification: {}", request.method);
                continue;
            }

            let id = request.id.clone().unwrap_or(serde_json::Value::Null);
            let response = self.handle_request(id, &request.method, request.params);

            writeln!(writer, "{}", serde_json::to_string(&response).unwrap())?;
            writer.flush()?;
        }

        Ok(())
    }

    fn handle_request(
        &self,
        id: serde_json::Value,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        match method {
            "initialize" => self.handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, params),
            _ => JsonRpcResponse::error(
                id,
                METHOD_NOT_FOUND,
                format!("Unknown method: {}", method),
            ),
        }
    }

    fn handle_initialize(&self, id: serde_json::Value) -> JsonRpcResponse {
        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: Some(false),
                }),
            },
            server_info: Implementation {
                name: self.server_name.clone(),
                version: self.server_version.clone(),
            },
        };

        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
    }

    fn handle_tools_list(&self, id: serde_json::Value) -> JsonRpcResponse {
        let defs = self.tools.get_definitions();
        let tools: Vec<ToolInfo> = defs
            .into_iter()
            .map(|d| ToolInfo {
                name: d.name,
                description: d.description,
                input_schema: d.parameters,
            })
            .collect();

        let result = ToolsListResult { tools };
        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
    }

    fn handle_tools_call(
        &self,
        id: serde_json::Value,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Missing params".to_string(),
                )
            }
        };

        let call_params: ToolsCallParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Invalid params: {}", e),
                )
            }
        };

        match self.tools.call(&call_params.name, call_params.arguments) {
            Ok(tool_result) => {
                let result = ToolsCallResult {
                    content: vec![ToolContent::Text { text: tool_result.text }],
                    is_error: None,
                };
                JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
            }
            Err(e) => {
                let result = ToolsCallResult {
                    content: vec![ToolContent::Text {
                        text: e.to_string(),
                    }],
                    is_error: Some(true),
                };
                JsonRpcResponse::success(id, serde_json::to_value(result).unwrap())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{TaskTool, ToolRegistry};

    fn make_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(TaskTool::new()));
        reg
    }

    fn run_server_line(registry: &ToolRegistry, input: &str) -> String {
        let server = McpServer::new(registry);
        let reader = std::io::Cursor::new(format!("{}\n", input));
        let mut output = Vec::new();
        server.run_with(reader, &mut output).unwrap();
        String::from_utf8(output).unwrap().trim().to_string()
    }

    #[test]
    fn test_initialize() {
        let reg = make_registry();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"}
            }
        });
        let output = run_server_line(&reg, &serde_json::to_string(&req).unwrap());
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_none());
        let result: InitializeResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.protocol_version, "2024-11-05");
        assert!(result.capabilities.tools.is_some());
    }

    #[test]
    fn test_tools_list() {
        let reg = make_registry();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });
        let output = run_server_line(&reg, &serde_json::to_string(&req).unwrap());
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_none());
        let result: ToolsListResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.tools.len(), 1);
        assert_eq!(result.tools[0].name, "tasks");
    }

    #[test]
    fn test_tools_call_success() {
        let reg = make_registry();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "tasks",
                "arguments": {"action": "list"}
            }
        });
        let output = run_server_line(&reg, &serde_json::to_string(&req).unwrap());
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_none());
        let result: ToolsCallResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert!(result.is_error.is_none());
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn test_tools_call_unknown_tool() {
        let reg = make_registry();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "nonexistent",
                "arguments": {}
            }
        });
        let output = run_server_line(&reg, &serde_json::to_string(&req).unwrap());
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_none()); // Tool errors return in result, not JSON-RPC error
        let result: ToolsCallResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn test_unknown_method() {
        let reg = make_registry();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "prompts/list"
        });
        let output = run_server_line(&reg, &serde_json::to_string(&req).unwrap());
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn test_notification_no_response() {
        let reg = make_registry();
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let output = run_server_line(&reg, &serde_json::to_string(&notif).unwrap());
        assert!(output.is_empty());
    }

    #[test]
    fn test_parse_error() {
        let reg = make_registry();
        let output = run_server_line(&reg, "not json at all");
        let resp: JsonRpcResponse = serde_json::from_str(&output).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, PARSE_ERROR);
    }

    #[test]
    fn test_multi_line_session() {
        let reg = make_registry();
        let server = McpServer::new(&reg);

        // Simulate a full session: initialize -> initialized -> tools/list -> tools/call
        let lines = vec![
            serde_json::json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "test", "version": "0.1"}}
            }),
            serde_json::json!({
                "jsonrpc": "2.0", "method": "notifications/initialized"
            }),
            serde_json::json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/list"
            }),
            serde_json::json!({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "tasks", "arguments": {"action": "create",
                           "subject": "Test task", "description": "A test"}}
            }),
        ];

        let input = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");

        let reader = std::io::Cursor::new(input);
        let mut output = Vec::new();
        server.run_with(reader, &mut output).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        let responses: Vec<&str> = output_str.trim().split('\n').collect();

        // 3 responses (initialize, tools/list, tools/call — notification has none)
        assert_eq!(responses.len(), 3);

        // Last response should confirm task creation
        let last: JsonRpcResponse = serde_json::from_str(responses[2]).unwrap();
        let result: ToolsCallResult =
            serde_json::from_value(last.result.unwrap()).unwrap();
        match &result.content[0] {
            ToolContent::Text { text } => assert!(text.contains("Test task")),
        }
    }
}
