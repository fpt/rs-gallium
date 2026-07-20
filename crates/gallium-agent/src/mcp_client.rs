//! MCP client — connects to external MCP server subprocesses and wraps their tools
//! as `ToolHandler` implementations for seamless integration with `ToolRegistry`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::mcp::*;
use crate::tool::ToolHandler;
use crate::AgentError;

/// Serializes request/response pairs over stdin/stdout of the child process.
struct Transport {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

/// An MCP client that connects to an MCP server running as a subprocess.
///
/// Use `McpClient::connect()` to spawn the server and perform the MCP handshake.
/// Then call `tool_handlers()` to get `ToolHandler` wrappers for each remote tool.
pub struct McpClient {
    transport: Mutex<Transport>,
    child: Mutex<Option<Child>>,
    next_id: AtomicU64,
    server_info: Mutex<Option<Implementation>>,
    tools: Mutex<Vec<ToolInfo>>,
}

impl McpClient {
    /// Spawn a subprocess MCP server and perform the initialization handshake.
    ///
    /// Returns an `Arc<McpClient>` ready for creating tool handlers.
    pub fn connect(command: &str, args: &[&str]) -> Result<Arc<Self>, AgentError> {
        Self::connect_with_env(command, args, &[])
    }

    /// Spawn with extra environment variables.
    pub fn connect_with_env(
        command: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Arc<Self>, AgentError> {
        tracing::info!("Spawning MCP server: {} {:?}", command, args);

        let mut cmd = build_command(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            AgentError::InternalError(format!("Failed to spawn MCP server '{}': {}", command, e))
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            AgentError::InternalError("Failed to capture MCP server stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            AgentError::InternalError("Failed to capture MCP server stdout".to_string())
        })?;

        let client = Arc::new(Self {
            transport: Mutex::new(Transport {
                stdin,
                reader: BufReader::new(stdout),
            }),
            child: Mutex::new(Some(child)),
            next_id: AtomicU64::new(1),
            server_info: Mutex::new(None),
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
        let request_json = serde_json::to_string(&request)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize error: {}", e)))?;

        let mut transport = self.transport.lock();

        // Write request
        writeln!(transport.stdin, "{}", request_json)
            .map_err(|e| AgentError::InternalError(format!("Write to MCP server failed: {}", e)))?;
        transport
            .stdin
            .flush()
            .map_err(|e| AgentError::InternalError(format!("Flush to MCP server failed: {}", e)))?;

        // Read until the response matching our request id arrives. Servers may
        // interleave notifications (no `id`), server-initiated requests, or plain
        // log lines on stdout; skip anything that isn't our response.
        loop {
            let mut line = String::new();
            let n = transport.reader.read_line(&mut line).map_err(|e| {
                AgentError::InternalError(format!("Read from MCP server failed: {}", e))
            })?;
            if n == 0 {
                return Err(AgentError::InternalError(
                    "MCP server closed stdout unexpectedly".to_string(),
                ));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Non-JSON output (e.g. server logging to stdout) — ignore.
            let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
                continue;
            };
            // Only our response carries our id; notifications/other ids are skipped.
            if value.get("id").and_then(serde_json::Value::as_u64) != Some(id) {
                continue;
            }

            if let Some(err) = value.get("error") {
                let code = err.get("code").and_then(serde_json::Value::as_i64).unwrap_or(0);
                let message = err
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error");
                return Err(AgentError::InternalError(format!(
                    "MCP error ({}): {}",
                    code, message
                )));
            }

            return value
                .get("result")
                .cloned()
                .ok_or_else(|| AgentError::InternalError("Empty result from MCP server".to_string()));
        }
    }

    /// Send a notification (no response expected).
    fn send_notification(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), AgentError> {
        let notification = JsonRpcRequest::notification(method, params);
        let json = serde_json::to_string(&notification)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize error: {}", e)))?;

        let mut transport = self.transport.lock();
        writeln!(transport.stdin, "{}", json)
            .map_err(|e| AgentError::InternalError(format!("Write to MCP server failed: {}", e)))?;
        transport
            .stdin
            .flush()
            .map_err(|e| AgentError::InternalError(format!("Flush to MCP server failed: {}", e)))?;

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
            "MCP server: {} v{}",
            init_result.server_info.name,
            init_result.server_info.version
        );
        *self.server_info.lock() = Some(init_result.server_info);

        self.send_notification("notifications/initialized", None)?;

        Ok(())
    }

    fn do_discover_tools(&self) -> Result<(), AgentError> {
        let result = self.send_request("tools/list", None)?;
        let list_result: ToolsListResult = serde_json::from_value(result)
            .map_err(|e| AgentError::ParseError(format!("Invalid tools/list result: {}", e)))?;

        tracing::info!("Discovered {} MCP tools", list_result.tools.len());
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
    ///
    /// These can be registered directly in a `ToolRegistry`.
    pub fn tool_handlers(self: &Arc<Self>) -> Vec<Box<dyn ToolHandler>> {
        let tools = self.tools.lock();
        tools
            .iter()
            .map(|info| {
                Box::new(McpRemoteTool {
                    client: Arc::clone(self),
                    info: info.clone(),
                }) as Box<dyn ToolHandler>
            })
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

/// A `ToolHandler` that delegates calls to a remote MCP server via `McpClient`.
pub struct McpRemoteTool {
    client: Arc<McpClient>,
    info: ToolInfo,
}

impl ToolHandler for McpRemoteTool {
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

// McpRemoteTool is Send + Sync because McpClient uses Mutex internally
unsafe impl Send for McpRemoteTool {}
unsafe impl Sync for McpRemoteTool {}

/// Build a `Command` for an MCP server program. On Windows, many MCP servers are
/// launched via `npx`/`pnpm`/etc., which are `.cmd`/`.bat` shims that
/// `Command::new` can't find (it doesn't apply PATHEXT). Resolve the program
/// against PATH + PATHEXT so those work; Rust (>=1.77) runs `.cmd`/`.bat` safely.
fn build_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        if let Some(resolved) = resolve_windows_exe(command) {
            return Command::new(resolved);
        }
    }
    Command::new(command)
}

/// Resolve a bare command name to a concrete executable path using PATH and
/// PATHEXT (Windows). Returns None if it's already a path or can't be resolved
/// (caller falls back to spawning the name as-is).
#[cfg(windows)]
fn resolve_windows_exe(command: &str) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};

    let p = Path::new(command);
    // Already a usable path (has an extension and exists): use as-is.
    if p.extension().is_some() && p.is_file() {
        return Some(p.to_path_buf());
    }

    let exts: Vec<String> = std::env::var("PATHEXT")
        .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
        .split(';')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    // If the command names a directory, search only there; else search PATH.
    let dirs: Vec<PathBuf> = if p.components().count() > 1 {
        vec![p.parent().map(Path::to_path_buf).unwrap_or_default()]
    } else {
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).collect()
    };
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or(command);

    for dir in dirs {
        for ext in &exts {
            let candidate = dir.join(format!("{name}{ext}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_remote_tool_name() {
        // Just verify McpRemoteTool works with ToolInfo
        let info = ToolInfo {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };

        // We can't create a real McpClient without a subprocess,
        // but we can verify the ToolInfo clone works
        let cloned = info.clone();
        assert_eq!(cloned.name, "test_tool");
        assert_eq!(cloned.description, "A test tool");
    }
}
