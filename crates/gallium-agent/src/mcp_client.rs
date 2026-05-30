//! MCP client — connects to external MCP server subprocesses via stdio and wraps
//! their tools as `ToolHandler` implementations.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::mcp::*;
use crate::tool::ToolHandler;
use crate::AgentError;

struct Transport {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

pub struct McpClient {
    transport: Mutex<Transport>,
    child: Mutex<Option<Child>>,
    next_id: AtomicU64,
    tools: Mutex<Vec<ToolInfo>>,
}

impl McpClient {
    pub fn connect(command: &str, args: &[&str]) -> Result<Arc<Self>, AgentError> {
        Self::connect_with_env(command, args, &[])
    }

    pub fn connect_with_env(
        command: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Arc<Self>, AgentError> {
        tracing::info!("Spawning MCP server: {} {:?}", command, args);

        let mut cmd = Command::new(command);
        cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            AgentError::InternalError(format!("Failed to spawn MCP server '{}': {}", command, e))
        })?;

        let stdin = child.stdin.take()
            .ok_or_else(|| AgentError::InternalError("Failed to capture MCP server stdin".to_string()))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| AgentError::InternalError("Failed to capture MCP server stdout".to_string()))?;

        let client = Arc::new(Self {
            transport: Mutex::new(Transport { stdin, reader: BufReader::new(stdout) }),
            child: Mutex::new(Some(child)),
            next_id: AtomicU64::new(1),
            tools: Mutex::new(Vec::new()),
        });

        client.do_initialize()?;
        client.do_discover_tools()?;

        Ok(client)
    }

    fn send_request(&self, method: &str, params: Option<serde_json::Value>) -> Result<serde_json::Value, AgentError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = JsonRpcRequest::new(id, method, params);
        let request_json = serde_json::to_string(&request)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize: {}", e)))?;

        let mut transport = self.transport.lock();
        writeln!(transport.stdin, "{}", request_json)
            .map_err(|e| AgentError::InternalError(format!("Write to MCP server: {}", e)))?;
        transport.stdin.flush()
            .map_err(|e| AgentError::InternalError(format!("Flush to MCP server: {}", e)))?;

        let mut line = String::new();
        transport.reader.read_line(&mut line)
            .map_err(|e| AgentError::InternalError(format!("Read from MCP server: {}", e)))?;

        if line.is_empty() {
            return Err(AgentError::InternalError("MCP server closed stdout unexpectedly".to_string()));
        }

        let response: JsonRpcResponse = serde_json::from_str(line.trim())
            .map_err(|e| AgentError::ParseError(format!("Invalid JSON-RPC response: {}", e)))?;

        if let Some(err) = response.error {
            return Err(AgentError::InternalError(format!("MCP error ({}): {}", err.code, err.message)));
        }

        response.result.ok_or_else(|| AgentError::InternalError("Empty result from MCP server".to_string()))
    }

    fn send_notification(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), AgentError> {
        let notification = JsonRpcRequest::notification(method, params);
        let json = serde_json::to_string(&notification)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize: {}", e)))?;
        let mut transport = self.transport.lock();
        writeln!(transport.stdin, "{}", json)
            .map_err(|e| AgentError::InternalError(format!("Write to MCP server: {}", e)))?;
        transport.stdin.flush()
            .map_err(|e| AgentError::InternalError(format!("Flush to MCP server: {}", e)))?;
        Ok(())
    }

    fn do_initialize(&self) -> Result<(), AgentError> {
        let params = serde_json::to_value(InitializeParams {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities {},
            client_info: Implementation {
                name: "gallium-agent".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }).map_err(|e| AgentError::InternalError(format!("JSON: {}", e)))?;

        let result = self.send_request("initialize", Some(params))?;
        let init_result: InitializeResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid initialize result: {}", e)))?;

        tracing::info!("MCP server: {} v{}", init_result.server_info.name, init_result.server_info.version);
        self.send_notification("notifications/initialized", None)?;
        Ok(())
    }

    fn do_discover_tools(&self) -> Result<(), AgentError> {
        let result = self.send_request("tools/list", None)?;
        let list_result: ToolsListResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid tools/list result: {}", e)))?;

        tracing::info!("Discovered {} MCP tool(s)", list_result.tools.len());
        for t in &list_result.tools {
            tracing::info!("  MCP tool: {}", t.name);
        }

        *self.tools.lock() = list_result.tools;
        Ok(())
    }

    pub fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String, AgentError> {
        let params = serde_json::to_value(ToolsCallParams { name: name.to_string(), arguments })
            .map_err(|e| AgentError::InternalError(format!("JSON: {}", e)))?;

        let result = self.send_request("tools/call", Some(params))?;
        let call_result: ToolsCallResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid tools/call result: {}", e)))?;

        let text = call_result.content.iter()
            .filter_map(|c| match c { ToolContent::Text { text } => Some(text.as_str()) })
            .collect::<Vec<_>>()
            .join("\n");

        if call_result.is_error == Some(true) {
            return Err(AgentError::InternalError(text));
        }
        Ok(text)
    }

    pub fn tool_handlers(self: &Arc<Self>) -> Vec<Box<dyn ToolHandler>> {
        let tools = self.tools.lock();
        tools.iter()
            .map(|info| Box::new(McpRemoteTool { client: Arc::clone(self), info: info.clone() }) as Box<dyn ToolHandler>)
            .collect()
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub struct McpRemoteTool {
    client: Arc<McpClient>,
    info: ToolInfo,
}

impl ToolHandler for McpRemoteTool {
    fn name(&self) -> &str { &self.info.name }
    fn description(&self) -> &str { &self.info.description }
    fn parameters_schema(&self) -> serde_json::Value { self.info.input_schema.clone() }
    fn call(&self, args: serde_json::Value) -> Result<crate::tool::ToolResult, AgentError> {
        self.client.call_tool(&self.info.name, args).map(crate::tool::ToolResult::text)
    }
}

unsafe impl Send for McpRemoteTool {}
unsafe impl Sync for McpRemoteTool {}
