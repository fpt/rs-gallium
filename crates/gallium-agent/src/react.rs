use crate::llm::{ChatMessage, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo};
use crate::tool::{ToolAccess, ToolResult};
use crate::AgentError;

/// Tool-calling rounds allowed in one turn before the loop gives up.
///
/// Each iteration is one LLM call, so this bounds both cost and how long a turn
/// can run. 10 was too tight: a multi-file task spends iterations on
/// read/grep/edit and would hit the ceiling mid-way and abort, losing the work.
/// 30 matches klein's `max_iterations` default.
///
/// Callers that want a different bound pass `Some(n)` to [`run`]; the Rust CLI
/// and app-server expose it as `MAX_REACT_ITERATIONS`.
pub const DEFAULT_MAX_ITERATIONS: u32 = 30;

/// Something that happened partway through a turn, reported as it occurs.
///
/// A caller driving kessel remotely (the app-server) relays these so its client
/// sees progress instead of silence during a long multi-tool turn.
pub enum ReactEvent<'a> {
    /// The model asked to call a tool.
    ToolCall { name: &'a str, arguments: &'a serde_json::Value },
    /// A tool returned. `text` is the result fed back to the model.
    ToolResult { name: &'a str, text: &'a str },
}

/// Notified as a turn progresses. Called on the turn's own thread, so an
/// implementation that blocks stalls the turn.
pub trait ReactObserver {
    fn on_event(&self, event: ReactEvent<'_>);
}

/// Run a ReAct (Reason+Act) loop: call LLM with tools, execute tool calls, repeat until text response.
///
/// Returns the final text response, optional reasoning, and accumulated token usage.
pub fn run(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    run_observed(client, messages, tools, max_iterations, None)
}

/// As [`run`], reporting each step to `observer` as it happens.
pub fn run_observed(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
    observer: Option<&dyn ReactObserver>,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    let max_iter = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);
    let tool_defs = tools.get_definitions();
    let mut total_usage = TokenUsage::default();

    let emit = |event: ReactEvent<'_>| {
        if let Some(obs) = observer {
            obs.on_event(event);
        }
    };

    for iteration in 0..max_iter {
        tracing::info!("ReAct iteration {}/{}", iteration + 1, max_iter);

        let response = client
            .chat_with_tools(messages, &tool_defs)
            .map_err(|e| AgentError::NetworkError(e.to_string()))?;

        match response {
            LlmResponse::Text { content, reasoning, usage } => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                }
                tracing::info!(
                    "ReAct complete: text response after {} iterations (tokens: in={}, out={}, total={})",
                    iteration + 1, total_usage.input_tokens, total_usage.output_tokens, total_usage.total_tokens
                );
                return Ok((content, reasoning, total_usage));
            }
            LlmResponse::ToolCalls(calls, usage) => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                }
                tracing::info!(
                    "ReAct iteration {}: {} tool call(s)",
                    iteration + 1,
                    calls.len()
                );

                // Record the assistant's tool calls in message history
                messages.push(ChatMessage::assistant_tool_calls(calls.clone()));

                // Execute each tool call and add results
                for call in &calls {
                    emit(ReactEvent::ToolCall { name: &call.name, arguments: &call.arguments });

                    let result = execute_tool_call(tools, call);

                    tracing::info!(
                        "Tool '{}' ({}): {} chars result, {} images",
                        call.name,
                        call.id,
                        result.text.len(),
                        result.images.len(),
                    );
                    emit(ReactEvent::ToolResult { name: &call.name, text: &result.text });

                    if result.images.is_empty() {
                        messages.push(ChatMessage::tool_result(
                            call.id.clone(),
                            call.name.clone(),
                            result.text,
                        ));
                    } else {
                        messages.push(ChatMessage::tool_result_with_images(
                            call.id.clone(),
                            call.name.clone(),
                            result.text,
                            result.images,
                        ));
                    }
                }
            }
        }
    }

    Err(AgentError::InternalError(format!(
        "ReAct loop exceeded maximum iterations ({})",
        max_iter
    )))
}

/// Execute a single tool call, returning the result (or error message)
fn execute_tool_call(tools: &dyn ToolAccess, call: &ToolCallInfo) -> ToolResult {
    match tools.call(&call.name, call.arguments.clone()) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("Tool '{}' error: {}", call.name, e);
            ToolResult::text(format!("Error executing tool '{}': {}", call.name, e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatRole, ToolDefinition};
    use crate::tool::ToolRegistry;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock LLM provider for testing the ReAct loop
    struct MockProvider {
        responses: Vec<LlmResponse>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl LlmProvider for MockProvider {
        fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok("mock".to_string())
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> anyhow::Result<LlmResponse> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                // We need to clone the response — reconstruct it
                let resp = &self.responses[idx];
                match resp {
                    LlmResponse::Text { content, reasoning, usage } => Ok(LlmResponse::Text {
                        content: content.clone(),
                        reasoning: reasoning.clone(),
                        usage: usage.clone(),
                    }),
                    LlmResponse::ToolCalls(calls, usage) => {
                        Ok(LlmResponse::ToolCalls(calls.clone(), usage.clone()))
                    }
                }
            } else {
                Ok(LlmResponse::Text {
                    content: "fallback".to_string(),
                    reasoning: None,
                    usage: None,
                })
            }
        }
    }

    #[test]
    fn test_react_direct_text_response() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "Hello!".to_string(),
            reasoning: None,
            usage: None,
        }]);
        let mut messages = vec![ChatMessage::user("Hi".to_string())];
        let tools = ToolRegistry::new();

        let (text, reasoning, usage) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "Hello!");
        assert!(reasoning.is_none());
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_react_tool_then_text() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_1".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::Text {
                content: "There are no tasks.".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);

        let mut messages = vec![ChatMessage::user("List tasks".to_string())];

        // Create registry with task tool
        use crate::tool::TaskTool;
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(TaskTool::new()));

        let (text, _, _) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "There are no tasks.");

        // Messages should contain: user, assistant(tool_calls), tool_result
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].role, ChatRole::Tool);
    }

    /// Mock tool that returns a ToolResult with images
    struct MockImageTool;

    impl crate::tool::ToolHandler for MockImageTool {
        fn name(&self) -> &str {
            "capture_screen"
        }
        fn description(&self) -> &str {
            "Mock screen capture"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        fn call(&self, _args: serde_json::Value) -> Result<ToolResult, crate::AgentError> {
            Ok(ToolResult::with_images(
                "Window: Chrome, Size: 1920x1080".to_string(),
                vec![crate::llm::ImageContent {
                    base64: "iVBORw0KGgoAAAANS".to_string(),
                    media_type: "image/png".to_string(),
                }],
            ))
        }
    }

    #[test]
    fn test_react_tool_with_images_stores_in_messages() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_img".to_string(),
                name: "capture_screen".to_string(),
                arguments: serde_json::json!({"process_name": "Chrome"}),
            }], None),
            LlmResponse::Text {
                content: "I can see a Chrome window.".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);

        let mut messages = vec![ChatMessage::user("capture Chrome".to_string())];
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(MockImageTool));

        let (text, _, _) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "I can see a Chrome window.");

        // Messages: user, assistant(tool_calls), tool_result_with_images
        assert_eq!(messages.len(), 3);

        // The tool result message should have images
        let tool_msg = &messages[2];
        assert_eq!(tool_msg.role, ChatRole::Tool);
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_img"));
        assert_eq!(tool_msg.tool_name.as_deref(), Some("capture_screen"));
        assert_eq!(tool_msg.content, "Window: Chrome, Size: 1920x1080");
        assert_eq!(tool_msg.images.len(), 1, "Tool result should carry 1 image");
        assert_eq!(tool_msg.images[0].media_type, "image/png");
        assert_eq!(tool_msg.images[0].base64, "iVBORw0KGgoAAAANS");
    }

    #[test]
    fn test_react_tool_without_images_has_empty_images() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_1".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::Text {
                content: "done".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);

        let mut messages = vec![ChatMessage::user("list".to_string())];
        use crate::tool::TaskTool;
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(TaskTool::new()));

        run(&provider, &mut messages, &tools, Some(5)).unwrap();

        let tool_msg = &messages[2];
        assert_eq!(tool_msg.role, ChatRole::Tool);
        assert!(tool_msg.images.is_empty(), "Plain tool result should have no images");
    }

    #[test]
    fn test_react_max_iterations() {
        // Provider always returns tool calls — should hit max iterations
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_1".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_2".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_3".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
        ]);

        let mut messages = vec![ChatMessage::user("Loop forever".to_string())];

        use crate::tool::TaskTool;
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(TaskTool::new()));

        let result = run(&provider, &mut messages, &tools, Some(2));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("maximum iterations"));
    }

    /// `None` must fall through to DEFAULT_MAX_ITERATIONS. The frontends (Swift,
    /// C#) always pass None, so this is the bound they actually run under —
    /// worth pinning behaviorally rather than trusting the constant.
    #[test]
    fn test_react_none_uses_the_default_iteration_cap() {
        // A provider that never stops asking for tools, so the cap is what ends it.
        let provider = MockProvider::new(
            (0..DEFAULT_MAX_ITERATIONS + 5)
                .map(|i| {
                    LlmResponse::ToolCalls(
                        vec![ToolCallInfo {
                            id: format!("call_{i}"),
                            name: "tasks".to_string(),
                            arguments: serde_json::json!({"action": "list"}),
                        }],
                        None,
                    )
                })
                .collect(),
        );

        let mut messages = vec![ChatMessage::user("Loop forever".to_string())];
        use crate::tool::TaskTool;
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(TaskTool::new()));

        let err = run(&provider, &mut messages, &tools, None).unwrap_err().to_string();
        assert!(
            err.contains(&format!("({})", DEFAULT_MAX_ITERATIONS)),
            "expected the cap to be DEFAULT_MAX_ITERATIONS, got: {err}"
        );
        assert_eq!(
            provider.call_count.load(Ordering::SeqCst),
            DEFAULT_MAX_ITERATIONS as usize,
            "the loop should run exactly DEFAULT_MAX_ITERATIONS times"
        );
    }
}
