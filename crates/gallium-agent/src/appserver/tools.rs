//! Client-provided tools (`dynamicTools`) and approval routing.
//!
//! The client registers its own tools on `thread/start`. Each becomes a
//! `ToolHandler` in the thread's registry whose `call()` sends an
//! `item/tool/call` request back over the connection and blocks for the answer —
//! the mirror image of `McpRemoteTool`, which wraps a tool living in a
//! subprocess we spawned.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::appserver::rpc::Connection;
use crate::tool::{ApprovalDecision, ApprovalSink, ToolHandler, ToolResult};
use crate::AgentError;

/// A tool the client declared in `thread/start`'s `dynamicTools`.
#[derive(Debug, Clone, Deserialize)]
pub struct DynamicToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the tool's arguments.
    #[serde(default = "empty_object", rename = "inputSchema")]
    pub input_schema: Value,
}

fn empty_object() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// A `ToolHandler` that dispatches back to the client over JSON-RPC.
///
/// Tools are registered once per thread, but each call must report the turn it
/// belongs to — so the live turn id is shared with the thread rather than
/// captured at registration.
pub struct RemoteTool {
    conn: Arc<Connection>,
    spec: DynamicToolSpec,
    thread_id: String,
    current_turn: Arc<Mutex<String>>,
}

impl RemoteTool {
    pub fn new(
        conn: Arc<Connection>,
        spec: DynamicToolSpec,
        thread_id: String,
        current_turn: Arc<Mutex<String>>,
    ) -> Self {
        Self { conn, spec, thread_id, current_turn }
    }
}

impl ToolHandler for RemoteTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn parameters_schema(&self) -> Value {
        self.spec.input_schema.clone()
    }

    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let call_id = format!("call_{}", uuid_like());
        let params = json!({
            "threadId": self.thread_id,
            "turnId": self.current_turn.lock().clone(),
            "callId": call_id,
            "tool": self.spec.name,
            "arguments": args,
        });

        let response = self.conn.request("item/tool/call", params)?;
        parse_tool_response(&response, &self.spec.name)
    }
}

/// Read a `DynamicToolCallResponse` back into a `ToolResult`.
///
/// `success: false` is the client reporting that *its* tool failed, which is a
/// normal ReAct outcome (feed the message back to the model), not a transport
/// error — so it comes back as `Ok` text, matching how `execute_tool_call`
/// already folds tool errors into the conversation.
fn parse_tool_response(response: &Value, tool: &str) -> Result<ToolResult, AgentError> {
    let text = response
        .get("contentItems")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    let success = response.get("success").and_then(Value::as_bool).unwrap_or(false);
    if !success {
        let detail = if text.is_empty() { "no detail provided" } else { &text };
        return Ok(ToolResult::text(format!("Error executing tool '{tool}': {detail}")));
    }
    Ok(ToolResult::text(text))
}

/// Approves every mutation without asking, for `approvalPolicy: "never"`.
///
/// The client has told us it does not want to be consulted (a headless surface),
/// so round-tripping each write would only add latency and noise.
pub struct AutoApproveSink;

impl ApprovalSink for AutoApproveSink {
    fn request(&self, action: &str, target: &str) -> Result<ApprovalDecision, AgentError> {
        tracing::debug!("auto-approving {} '{}' (approvalPolicy=never)", action, target);
        Ok(ApprovalDecision::Allow)
    }
}

/// Routes kessel's mutation approvals to the client instead of the terminal.
///
/// Under the app-server there is no TTY, so `ToolSession`'s built-in prompt
/// would fail closed on every `write`/`edit`/`bash`. Instead we raise the same
/// question over JSON-RPC and let the driving client decide.
pub struct RemoteApprovalSink {
    conn: Arc<Connection>,
    thread_id: String,
}

impl RemoteApprovalSink {
    pub fn new(conn: Arc<Connection>, thread_id: String) -> Self {
        Self { conn, thread_id }
    }
}

impl ApprovalSink for RemoteApprovalSink {
    fn request(&self, action: &str, target: &str) -> Result<ApprovalDecision, AgentError> {
        // `run command` maps to the command-execution approval; everything else
        // (write file, edit file, GitHub mutations) is a file-change approval.
        let (method, params) = if action == "run command" {
            (
                "item/commandExecution/requestApproval",
                json!({ "threadId": self.thread_id, "command": target }),
            )
        } else {
            (
                "item/fileChange/requestApproval",
                json!({ "threadId": self.thread_id, "reason": format!("{action} '{target}'") }),
            )
        };

        let response = self.conn.request(method, params)?;
        let decision = response.get("decision").and_then(Value::as_str).unwrap_or("decline");
        Ok(match decision {
            "accept" => ApprovalDecision::Allow,
            "accept_for_session" => ApprovalDecision::AllowAll,
            _ => ApprovalDecision::Deny,
        })
    }
}

/// A short unique-enough id for correlating tool calls within one connection.
/// Not a real UUID — it only has to be distinct among concurrent in-flight calls.
fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{nanos:08x}{n:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_successful_tool_response() {
        let response = json!({
            "success": true,
            "contentItems": [
                { "type": "inputText", "text": "line one" },
                { "type": "inputText", "text": "line two" },
            ],
        });
        let result = parse_tool_response(&response, "memory").unwrap();
        assert_eq!(result.text, "line one\nline two");
    }

    #[test]
    fn failed_tool_call_becomes_error_text_not_transport_error() {
        let response = json!({
            "success": false,
            "contentItems": [{ "type": "inputText", "text": "file not found" }],
        });
        let result = parse_tool_response(&response, "memory").unwrap();
        assert_eq!(result.text, "Error executing tool 'memory': file not found");
    }

    #[test]
    fn failed_tool_call_without_detail_still_reports_the_tool() {
        let response = json!({ "success": false, "contentItems": [] });
        let result = parse_tool_response(&response, "schedule").unwrap();
        assert!(result.text.contains("schedule"), "got: {}", result.text);
        assert!(result.text.contains("no detail provided"));
    }

    #[test]
    fn missing_success_field_is_treated_as_failure() {
        let response = json!({ "contentItems": [{ "text": "hi" }] });
        let result = parse_tool_response(&response, "t").unwrap();
        assert!(result.text.starts_with("Error executing tool 't'"));
    }

    #[test]
    fn spec_defaults_input_schema_when_absent() {
        let spec: DynamicToolSpec = serde_json::from_value(json!({ "name": "memory" })).unwrap();
        assert_eq!(spec.name, "memory");
        assert_eq!(spec.input_schema["type"], "object");
    }
}
