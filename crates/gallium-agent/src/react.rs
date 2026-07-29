use crate::cancel::TurnContext;
use crate::event::{self, AgentEvent, AgentObserver};
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

/// Run a ReAct (Reason+Act) loop: call LLM with tools, execute tool calls, repeat until text response.
///
/// Returns the final text response, optional reasoning, and accumulated token usage.
pub fn run(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    run_observed(
        client,
        messages,
        tools,
        max_iterations,
        None,
        &TurnContext::detached(),
    )
}

/// As [`run`], reporting each step to `observer` as it happens, and stopping
/// when `ctx` is cancelled.
///
/// Cancellation is checked at three points: before each model call, after one
/// returns, and before each tool the response asked for. Between them the turn
/// is inside a provider or a tool, which stop themselves — generation between
/// tokens, `bash` between polls of its child.
pub fn run_observed(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
    observer: Option<&dyn AgentObserver>,
    ctx: &TurnContext,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    let max_iter = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);
    let tool_defs = tools.get_definitions();
    let mut total_usage = TokenUsage::default();

    let emit = |event: AgentEvent<'_>| event::emit(observer, event);

    // The prompt and the catalog as the model first sees them. Recorded before
    // the loop rather than per iteration: later prompts are this list plus the
    // tool transcript the trace already holds.
    if let Some(trace) = &ctx.trace {
        trace.record_prompt(messages, &tool_defs);
    }

    for iteration in 0..max_iter {
        ctx.check()?;
        tracing::info!("ReAct iteration {}/{}", iteration + 1, max_iter);

        let asked_at = std::time::Instant::now();
        let response =
            match client.chat_with_tools_cancellable(messages, &tool_defs, &ctx.cancellation) {
                Ok(response) => response,
                Err(e) => {
                    // A provider that stopped because the turn was cancelled is
                    // not a network failure, and must not be reported as one.
                    if let Some(AgentError::Cancelled) = e.downcast_ref::<AgentError>() {
                        return Err(AgentError::Cancelled);
                    }
                    let error = AgentError::NetworkError(e.to_string());
                    emit(AgentEvent::Error {
                        message: &error.to_string(),
                    });
                    return Err(error);
                }
            };

        if let Some(trace) = &ctx.trace {
            trace.record_response(iteration + 1, &response, asked_at.elapsed());
        }

        // A provider with no interruption point runs to completion even after
        // the turn is cancelled. Its answer is discarded here rather than being
        // fed back to a model whose turn is over.
        ctx.check()?;

        match response {
            LlmResponse::Text {
                content,
                reasoning,
                usage,
            } => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                    emit(AgentEvent::Usage { usage: u });
                }
                tracing::info!(
                    "ReAct complete: text response after {} iterations (tokens: in={}, out={}, total={})",
                    iteration + 1, total_usage.input_tokens, total_usage.output_tokens, total_usage.total_tokens
                );
                emit(AgentEvent::TurnCompleted { text: &content });
                return Ok((content, reasoning, total_usage));
            }
            LlmResponse::ToolCalls(calls, usage) => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                    emit(AgentEvent::Usage { usage: u });
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
                    ctx.check()?;
                    emit(AgentEvent::ToolStarted {
                        call_id: &call.id,
                        name: &call.name,
                        arguments: &call.arguments,
                    });

                    let called_at = std::time::Instant::now();
                    let result = execute_tool_call(tools, ctx, call)?;
                    if let Some(trace) = &ctx.trace {
                        trace.record_tool_call(call, &result, called_at.elapsed());
                    }

                    tracing::info!(
                        "Tool '{}' ({}): {} chars result, error={}",
                        call.name,
                        call.id,
                        result.model_text().len(),
                        result.is_error,
                    );
                    emit(AgentEvent::ToolCompleted {
                        call_id: &call.id,
                        name: &call.name,
                        result: &result,
                    });

                    let (text, images) = result.into_text_and_images();
                    if images.is_empty() {
                        messages.push(ChatMessage::tool_result(
                            call.id.clone(),
                            call.name.clone(),
                            text,
                        ));
                    } else {
                        messages.push(ChatMessage::tool_result_with_images(
                            call.id.clone(),
                            call.name.clone(),
                            text,
                            images,
                        ));
                    }
                }
            }
        }
    }

    let error = AgentError::InternalError(format!(
        "ReAct loop exceeded maximum iterations ({})",
        max_iter
    ));
    emit(AgentEvent::Error {
        message: &error.to_string(),
    });
    Err(error)
}

/// Execute a single tool call.
///
/// A tool that fails is a normal ReAct outcome: the message goes back to the
/// model as the call's result and the loop carries on. Cancellation is the one
/// exception — it ends the turn, so it propagates instead of being narrated to
/// a model that will never read it.
fn execute_tool_call(
    tools: &dyn ToolAccess,
    ctx: &TurnContext,
    call: &ToolCallInfo,
) -> Result<ToolResult, AgentError> {
    match tools.call_with(ctx, &call.name, call.arguments.clone()) {
        Ok(result) => Ok(result),
        Err(AgentError::Cancelled) => Err(AgentError::Cancelled),
        Err(e) => {
            tracing::warn!("Tool '{}' error: {}", call.name, e);
            Ok(ToolResult::error(format!(
                "Error executing tool '{}': {}",
                call.name, e
            )))
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
                    LlmResponse::Text {
                        content,
                        reasoning,
                        usage,
                    } => Ok(LlmResponse::Text {
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
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                None,
            ),
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

    impl crate::tool::Tool for MockImageTool {
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
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_img".to_string(),
                    name: "capture_screen".to_string(),
                    arguments: serde_json::json!({"process_name": "Chrome"}),
                }],
                None,
            ),
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
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                None,
            ),
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
        assert!(
            tool_msg.images.is_empty(),
            "Plain tool result should have no images"
        );
    }

    #[test]
    fn test_react_max_iterations() {
        // Provider always returns tool calls — should hit max iterations
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                None,
            ),
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_2".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                None,
            ),
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "call_3".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                None,
            ),
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

        let err = run(&provider, &mut messages, &tools, None)
            .unwrap_err()
            .to_string();
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

    // ------------------------------------------------------------------
    // The event stream
    // ------------------------------------------------------------------

    /// Records events as one line each, so a test can assert on the sequence.
    #[derive(Default)]
    struct Recorder {
        lines: std::sync::Mutex<Vec<String>>,
    }

    impl AgentObserver for Recorder {
        fn on_event(&self, event: AgentEvent<'_>) {
            let line = match event {
                AgentEvent::ToolStarted { call_id, name, .. } => {
                    format!("started {name} ({call_id})")
                }
                AgentEvent::ToolCompleted {
                    call_id,
                    name,
                    result,
                } => format!(
                    "completed {name} ({call_id}) error={} text={}",
                    result.is_error,
                    result.display_text()
                ),
                AgentEvent::Usage { usage } => format!("usage in={}", usage.input_tokens),
                AgentEvent::TurnCompleted { text } => format!("turn {text}"),
                AgentEvent::Error { message } => format!("error {message}"),
            };
            self.lines.lock().unwrap().push(line);
        }
    }

    #[test]
    fn a_tool_using_turn_emits_start_completion_and_turn_events() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                Some(TokenUsage::single(11, 2, 13)),
            ),
            LlmResponse::Text {
                content: "done".to_string(),
                reasoning: None,
                usage: Some(TokenUsage::single(20, 3, 23)),
            },
        ]);

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let (text, _reasoning, _usage) = run_observed(
            &provider,
            &mut messages,
            &registry,
            Some(5),
            Some(&recorder),
            &TurnContext::detached(),
        )
        .unwrap();
        assert_eq!(text, "done");

        let lines = recorder.lines.lock().unwrap().clone();
        assert_eq!(
            lines,
            vec![
                "usage in=11".to_string(),
                "started echo (c1)".to_string(),
                "completed echo (c1) error=false text=echoed".to_string(),
                "usage in=20".to_string(),
                "turn done".to_string(),
            ]
        );
    }

    #[test]
    fn a_failing_tool_is_reported_as_an_error_not_as_output() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "nonexistent".to_string(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ),
            LlmResponse::Text {
                content: "gave up".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);

        let registry = ToolRegistry::new();
        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        run_observed(
            &provider,
            &mut messages,
            &registry,
            Some(5),
            Some(&recorder),
            &TurnContext::detached(),
        )
        .unwrap();

        let lines = recorder.lines.lock().unwrap().clone();
        let completed = lines
            .iter()
            .find(|l| l.starts_with("completed"))
            .expect("a completion event");
        assert!(
            completed.contains("error=true"),
            "an unknown tool must surface as an error: {completed}"
        );
    }

    /// Always asks for a tool, so the loop can only end by exhausting its
    /// budget. `MockProvider` falls back to a text reply once its script runs
    /// out, which would terminate the turn normally.
    struct NeverFinishesProvider;

    impl LlmProvider for NeverFinishesProvider {
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
            Ok(LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ))
        }
    }

    #[test]
    fn exhausting_the_iteration_budget_emits_an_error_event() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(
            &NeverFinishesProvider,
            &mut messages,
            &registry,
            Some(2),
            Some(&recorder),
            &TurnContext::detached(),
        );
        assert!(result.is_err(), "the budget must actually run out");

        let lines = recorder.lines.lock().unwrap().clone();
        assert!(
            lines.iter().any(|l| l.starts_with("error ")),
            "a failed turn must reach the frontend: {lines:?}"
        );
    }

    /// A turn cancelled while a tool was running must not go back to the model
    /// with the result: the loop stops at the next boundary rather than paying
    /// for another iteration nobody is waiting for.
    #[test]
    fn a_turn_cancelled_during_a_tool_call_does_not_reach_the_model_again() {
        struct CancelsItself(TurnContext);

        impl crate::tool::Tool for CancelsItself {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "cancels the turn it is running in"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn call(&self, _args: serde_json::Value) -> Result<ToolResult, AgentError> {
                self.0.cancellation.cancel();
                Ok(ToolResult::text("done, but the user hit stop".to_string()))
            }
        }

        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(CancelsItself(ctx.clone())));

        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ),
            LlmResponse::Text {
                content: "should never be asked for".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx);

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(
            provider.call_count.load(Ordering::SeqCst),
            1,
            "the loop must not call the model again after the turn was cancelled"
        );
    }

    /// Cancelling before the turn starts should cost nothing at all.
    #[test]
    fn a_turn_cancelled_before_it_starts_never_calls_the_model() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "unused".to_string(),
            reasoning: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        ctx.cancellation.cancel();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx);

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 0);
    }

    /// Minimal tool so the loop has something real to execute.
    struct EchoTool;

    impl crate::tool::Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }
        fn call(&self, _args: serde_json::Value) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::text("echoed".to_string()))
        }
    }
}
