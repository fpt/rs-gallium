//! Client-provided tools (`dynamicTools`) and approval routing.
//!
//! The client registers its own tools on `thread/start`. Each becomes a
//! `Tool` in the thread's registry whose `call()` sends an
//! `item/tool/call` request back over the connection and blocks for the answer —
//! the mirror image of `McpRemoteTool`, which wraps a tool living in a
//! subprocess we spawned.

use std::sync::Arc;

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::approval::{ApprovalDecision, ApprovalRequest, ApprovalSink};
use crate::appserver::rpc::Connection;
use crate::cancel::{wait_cancellable, CancellationToken, TurnContext};
use crate::tool::{Tool, ToolAnnotations, ToolResult, ToolSource};
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

/// A `Tool` that dispatches back to the client over JSON-RPC.
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
        Self {
            conn,
            spec,
            thread_id,
            current_turn,
        }
    }

    /// The `item/tool/call` payload. The turn id is read at call time rather
    /// than at registration, since the tool outlives any one turn.
    fn params(&self, args: Value) -> Value {
        json!({
            "threadId": self.thread_id,
            "turnId": self.current_turn.lock().clone(),
            "callId": format!("call_{}", uuid_like()),
            "tool": self.spec.name,
            "arguments": args,
        })
    }
}

impl Tool for RemoteTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn parameters_schema(&self) -> Value {
        self.spec.input_schema.clone()
    }

    /// The call runs inside the client, which tells us nothing about what it
    /// touches — `dynamicTools` has no hint field — so it is external and
    /// assumed to write.
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::EXTERNAL
    }

    fn source(&self) -> ToolSource {
        ToolSource::Dynamic
    }

    fn call(&self, args: Value) -> Result<ToolResult, AgentError> {
        let response = self.conn.request("item/tool/call", self.params(args))?;
        parse_tool_response(&response, &self.spec.name)
    }

    /// The client is answering at its own pace and cannot be interrupted, so
    /// cancellation stops the turn's wait rather than the call. The reply, if it
    /// ever comes, lands in a dropped channel.
    fn call_with(&self, ctx: &TurnContext, args: Value) -> Result<ToolResult, AgentError> {
        let conn = Arc::clone(&self.conn);
        let params = self.params(args);
        let response = wait_cancellable(ctx, move || conn.request("item/tool/call", params))??;
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

    let success = response
        .get("success")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !success {
        let detail = if text.is_empty() {
            "no detail provided"
        } else {
            &text
        };
        return Ok(ToolResult::error(format!(
            "Error executing tool '{tool}': {detail}"
        )));
    }
    Ok(ToolResult::text(text))
}

/// Approves every mutation without asking, for `approvalPolicy: "never"`.
///
/// The client has told us it does not want to be consulted (a headless surface),
/// so round-tripping each write would only add latency and noise.
pub struct AutoApproveSink;

impl ApprovalSink for AutoApproveSink {
    fn request(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, AgentError> {
        tracing::debug!(
            "auto-approving {} '{}' [{}] (approvalPolicy=never)",
            request.action,
            request.target,
            request.risk.label()
        );
        Ok(ApprovalDecision::AllowOnce)
    }
}

/// Routes gallium's mutation approvals to the client instead of the terminal.
///
/// Under the app-server there is no TTY, so `ToolSession`'s built-in prompt
/// would fail closed on every `write`/`edit`/`bash`. Instead we raise the same
/// question over JSON-RPC and let the driving client decide.
pub struct RemoteApprovalSink {
    conn: Arc<Connection>,
    thread_id: String,
    /// The running turn's stop switch, or `None` between turns — set by
    /// `run_turn` on the same cell `RemoteTool` reads its turn id from. This is
    /// how a `cancel` decision reaches the turn: see [`decode_decision`].
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
}

impl RemoteApprovalSink {
    pub fn new(
        conn: Arc<Connection>,
        thread_id: String,
        current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    ) -> Self {
        Self {
            conn,
            thread_id,
            current_cancel,
        }
    }
}

/// One decision from the client, in the protocol's own spelling.
///
/// Codex's `FileChangeApprovalDecision` and `CommandExecutionApprovalDecision`
/// are both `#[serde(rename_all = "camelCase")]`, so the session grant is
/// `acceptForSession` — gallium matched `accept_for_session` and every such
/// answer fell through to a refusal. The bug survived because the fallthrough
/// was silent and because `../klein-cli` sends only `accept` / `decline`
/// (`internal/agentserver/dynamictools.go`), so nothing exercised it.
///
/// Hence `None` for an unrecognized answer rather than a fourth variant: the
/// caller still refuses, but it can say *why* it refused. A decision gallium
/// does not understand is a client and a server that disagree about the
/// protocol, which is worth a line in the log — the alternative is this same
/// bug again, mute.
///
/// `Cancel` is not an [`ApprovalDecision`]: that enum answers "may this action
/// proceed", and cancelling answers it the same way `Decline` does. What makes
/// it different is the second half — *and stop the turn* — which is a property
/// of the turn, not of the call, and already has a channel of its own.
enum Decision {
    AllowOnce,
    AllowForSession,
    Decline,
    /// Refuse, and interrupt the turn rather than letting it try something else.
    Cancel,
}

fn decode_decision(decision: &str) -> Option<Decision> {
    match decision {
        "accept" => Some(Decision::AllowOnce),
        "acceptForSession" => Some(Decision::AllowForSession),
        "decline" => Some(Decision::Decline),
        "cancel" => Some(Decision::Cancel),
        _ => None,
    }
}

impl ApprovalSink for RemoteApprovalSink {
    fn request(&self, request: &ApprovalRequest) -> Result<ApprovalDecision, AgentError> {
        let (action, target) = (request.action, request.target);
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
        let answer = response.get("decision").and_then(Value::as_str);
        let decision = answer.and_then(decode_decision).unwrap_or_else(|| {
            // Refusing is the safe reading of an answer we cannot parse, but it
            // is indistinguishable from a deliberate refusal at the tool, so say
            // so here — this is the one place that knows the difference.
            tracing::warn!(
                "client answered {} with an unrecognized decision {:?}; refusing '{}'",
                method,
                answer.unwrap_or("<missing>"),
                target
            );
            Decision::Decline
        });

        Ok(match decision {
            Decision::AllowOnce => ApprovalDecision::AllowOnce,
            Decision::AllowForSession => ApprovalDecision::AllowForSession,
            Decision::Decline => ApprovalDecision::Deny,
            Decision::Cancel => {
                // Fire the stop switch *before* returning the refusal: the tool
                // gets its answer either way, and the ReAct loop checks the
                // token at its next boundary. Cancelling after returning would
                // race that boundary and could let one more model call through.
                //
                // A `cancel` between turns has nothing to stop. It still refuses
                // the action, which is the half of the decision that was about
                // to happen anyway.
                match self.current_cancel.lock().as_ref() {
                    Some(cancel) => {
                        tracing::info!(
                            "client cancelled at approval for {} '{}'; stopping the turn",
                            action,
                            target
                        );
                        cancel.cancel();
                    }
                    None => tracing::warn!(
                        "client cancelled at approval for {} '{}' with no turn running",
                        action,
                        target
                    ),
                }
                ApprovalDecision::Deny
            }
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

    /// The protocol's spellings, which are codex's `camelCase` and not the
    /// `snake_case` gallium used to match. `acceptForSession` reaching the
    /// wildcard arm is the bug this test exists for.
    #[test]
    fn decodes_the_protocol_spelling_of_every_decision() {
        assert!(matches!(
            decode_decision("accept"),
            Some(Decision::AllowOnce)
        ));
        assert!(matches!(
            decode_decision("acceptForSession"),
            Some(Decision::AllowForSession)
        ));
        assert!(matches!(
            decode_decision("decline"),
            Some(Decision::Decline)
        ));
        assert!(matches!(decode_decision("cancel"), Some(Decision::Cancel)));
    }

    /// Including the old misspelling: a client sending `accept_for_session` is
    /// not speaking the protocol, and the honest answer is the one that gets
    /// logged rather than the one that silently means something else.
    #[test]
    fn an_unknown_decision_is_not_decoded() {
        for answer in ["accept_for_session", "approve", "yes", ""] {
            assert!(
                decode_decision(answer).is_none(),
                "{answer:?} should not decode"
            );
        }
    }

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
        assert_eq!(result.model_text(), "line one\nline two");
    }

    #[test]
    fn failed_tool_call_becomes_error_text_not_transport_error() {
        let response = json!({
            "success": false,
            "contentItems": [{ "type": "inputText", "text": "file not found" }],
        });
        let result = parse_tool_response(&response, "memory").unwrap();
        assert_eq!(
            result.model_text(),
            "Error executing tool 'memory': file not found"
        );
    }

    #[test]
    fn failed_tool_call_without_detail_still_reports_the_tool() {
        let response = json!({ "success": false, "contentItems": [] });
        let result = parse_tool_response(&response, "schedule").unwrap();
        assert!(
            result.model_text().contains("schedule"),
            "got: {}",
            result.model_text()
        );
        assert!(result.model_text().contains("no detail provided"));
    }

    #[test]
    fn missing_success_field_is_treated_as_failure() {
        let response = json!({ "contentItems": [{ "text": "hi" }] });
        let result = parse_tool_response(&response, "t").unwrap();
        assert!(result.model_text().starts_with("Error executing tool 't'"));
    }

    #[test]
    fn spec_defaults_input_schema_when_absent() {
        let spec: DynamicToolSpec = serde_json::from_value(json!({ "name": "memory" })).unwrap();
        assert_eq!(spec.name, "memory");
        assert_eq!(spec.input_schema["type"], "object");
    }
}
