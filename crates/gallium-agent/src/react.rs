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
///
/// `ctx.steer` is read at two of those points: before each model call, and
/// again when the model returns text, where pending steering means the turn
/// carries on instead of ending. Together they cover every way a turn can be
/// waiting when the user speaks.
pub fn run_observed(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
    observer: Option<&dyn AgentObserver>,
    ctx: &TurnContext,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    let outcome = react_loop(client, messages, tools, max_iterations, observer, ctx);

    // The backstop. A turn that ends by choosing to stop reading closes the
    // inbox itself, atomically, in `SteerInbox::finish`; the endings that
    // announce themselves first — a provider failure, running out of iterations
    // — close it before they announce. What is left for here is the exits that
    // return immediately, where the gap is a handful of instructions rather than
    // a notification's worth of lock and I/O.
    //
    // The invariant this maintains is not "accepted means delivered" — a turn
    // can always fail after accepting — but the narrower one that matters: a
    // steer accepted by a turn that goes on to *complete* is a steer the model
    // saw. Anything accepted and then dropped comes with a `turn/completed`
    // whose status is `failed` or `interrupted`, naming that turn, which a
    // client can act on.
    if outcome.is_err() {
        ctx.steer.close();
    }
    outcome
}

fn react_loop(
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

    // `max_iter` bounds the *model's* looping: how many rounds of asking for
    // tools it gets before the turn gives up. A round the user asked for by
    // steering is not the model looping, so it is not charged — otherwise a
    // steer arriving on the last iteration turns an answer that was already
    // produced into a failed turn, and `runtime::run_turn` rolls the whole
    // turn's history back over it.
    //
    // Uncharged rounds cannot run away: each one needs a fresh `turn/steer`
    // from the client, and the drain that consumes it is what allows the next.
    let mut charged = 0u32;
    // Every model call, charged or not. What the trace and the logs count, so a
    // steered turn's records still number its calls in order.
    let mut calls = 0u32;

    while charged < max_iter {
        ctx.check()?;

        // Whatever the user said while the turn was running goes in before the
        // model is asked again — the point of steering is that the next model
        // call sees it, and this is the last moment at which that is still true.
        take_steering(ctx, messages);

        calls += 1;
        tracing::info!("ReAct iteration {}/{}", charged + 1, max_iter);

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
                    // Before the `emit`, for the reason given at the loop's
                    // exhaustion exit: the turn has stopped reading, and saying
                    // so after a notification has gone out is saying so late.
                    ctx.steer.close();
                    let error = AgentError::NetworkError(e.to_string());
                    emit(AgentEvent::Error {
                        message: &error.to_string(),
                    });
                    return Err(error);
                }
            };

        if let Some(trace) = &ctx.trace {
            trace.record_response(calls, &response, asked_at.elapsed());
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

                // A steer that lands while the model is composing its answer
                // would otherwise arrive one instant too late and be dropped —
                // the turn is over, and the loop's other drain point is never
                // reached again. So the answer becomes an intermediate message
                // and the turn carries on, which is also what the user meant:
                // they were still talking when the model stopped.
                //
                // `finish` rather than `has_pending`: this is the moment the
                // turn stops reading, and asking and then deciding as two steps
                // would let a steer land in between — accepted, acknowledged to
                // the client, and never read by anyone.
                if !ctx.steer.finish() {
                    tracing::info!(
                        "ReAct call {}: text response, but the turn was steered — continuing",
                        calls
                    );
                    emit(AgentEvent::AgentMessage { text: &content });
                    messages.push(ChatMessage::assistant(content).with_reasoning(reasoning));
                    continue;
                }

                tracing::info!(
                    "ReAct complete: text response after {} call(s) (tokens: in={}, out={}, total={})",
                    calls, total_usage.input_tokens, total_usage.output_tokens, total_usage.total_tokens
                );
                emit(AgentEvent::TurnCompleted { text: &content });
                return Ok((content, reasoning, total_usage));
            }
            // `tool_calls` rather than `calls`: the loop's own `calls` counts
            // model calls, and shadowing it here would read as the same thing.
            LlmResponse::ToolCalls {
                calls: tool_calls,
                usage,
                reasoning,
            } => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                    emit(AgentEvent::Usage { usage: u });
                }
                // The model asked for work, so this round is the model's own and
                // counts against the budget. The steered continuation above is
                // the one path that reaches the next iteration without paying.
                charged += 1;
                tracing::info!(
                    "ReAct iteration {}/{}: {} tool call(s)",
                    charged,
                    max_iter,
                    tool_calls.len()
                );

                // Record the assistant's tool calls in message history.
                messages.push(
                    ChatMessage::assistant_tool_calls(tool_calls.clone()).with_reasoning(reasoning),
                );

                // Execute each tool call and add results
                for call in &tool_calls {
                    ctx.check()?;
                    emit(AgentEvent::ToolStarted {
                        call_id: &call.id,
                        name: &call.name,
                        arguments: &call.arguments,
                    });

                    let called_at = std::time::Instant::now();
                    let result = match execute_tool_call(tools, ctx, call) {
                        Ok(result) => result,
                        // Cancelled while this call was running. It still ran,
                        // and it may already have been approved, so it is
                        // recorded before the cancellation propagates —
                        // otherwise an interrupted turn's trace shows the model
                        // asking for a tool and nothing happening.
                        //
                        // Recording is also what drains the approval journal.
                        // The broker is session-scoped and the journal outlives
                        // the turn, so leaving on this path would hand the
                        // decision to the next turn's first tool call.
                        Err(e) => {
                            if let Some(trace) = &ctx.trace {
                                trace.record_cancelled_tool_call(call, called_at.elapsed());
                            }
                            return Err(e);
                        }
                    };
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
                        arguments: &call.arguments,
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

    // Out of budget, so the loop will not read the inbox again — and it says so
    // here rather than on the way out of `run_observed`. What sits in between is
    // an `emit`, which on the app-server takes the connection's lock and writes
    // a notification; a steer arriving inside that window would be accepted by a
    // turn that had already stopped reading.
    ctx.steer.close();

    let error = AgentError::InternalError(format!(
        "ReAct loop exceeded maximum iterations ({})",
        max_iter
    ));
    emit(AgentEvent::Error {
        message: &error.to_string(),
    });
    Err(error)
}

/// Move anything the user said mid-turn out of the inbox and into the prompt.
///
/// Drains rather than peeks, so a message is delivered to the model exactly
/// once even though the loop reads the inbox at two different boundaries.
fn take_steering(ctx: &TurnContext, messages: &mut Vec<ChatMessage>) {
    let steered = ctx.steer.drain();
    if steered.is_empty() {
        return;
    }
    tracing::info!("turn steered: {} message(s) added mid-turn", steered.len());
    messages.extend(steered);
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
                    LlmResponse::ToolCalls {
                        calls,
                        usage,
                        reasoning,
                    } => Ok(LlmResponse::ToolCalls {
                        calls: calls.clone(),
                        usage: usage.clone(),
                        reasoning: reasoning.clone(),
                    }),
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
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
            },
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
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_img".to_string(),
                    name: "capture_screen".to_string(),
                    arguments: serde_json::json!({"process_name": "Chrome"}),
                }],
                usage: None,
                reasoning: None,
            },
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
        assert_eq!(
            tool_msg.images().count(),
            1,
            "Tool result should carry 1 image"
        );
        assert_eq!(tool_msg.images().next().unwrap().media_type, "image/png");
        assert_eq!(
            tool_msg.images().next().unwrap().base64,
            "iVBORw0KGgoAAAANS"
        );
    }

    #[test]
    fn test_react_tool_without_images_has_empty_images() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
            },
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
            tool_msg.media.is_empty(),
            "Plain tool result should have no images"
        );
    }

    #[test]
    fn test_react_max_iterations() {
        // Provider always returns tool calls — should hit max iterations
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_1".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
            },
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_2".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
            },
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_3".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
            },
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
                .map(|i| LlmResponse::ToolCalls {
                    calls: vec![ToolCallInfo {
                        id: format!("call_{i}"),
                        name: "tasks".to_string(),
                        arguments: serde_json::json!({"action": "list"}),
                    }],
                    usage: None,
                    reasoning: None,
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
                    ..
                } => format!(
                    "completed {name} ({call_id}) error={} text={}",
                    result.is_error,
                    result.display_text()
                ),
                AgentEvent::Usage { usage } => format!("usage in={}", usage.input_tokens),
                AgentEvent::AgentMessage { text } => format!("message {text}"),
                AgentEvent::TurnCompleted { text } => format!("turn {text}"),
                AgentEvent::Error { message } => format!("error {message}"),
            };
            self.lines.lock().unwrap().push(line);
        }
    }

    #[test]
    fn a_tool_using_turn_emits_start_completion_and_turn_events() {
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: Some(TokenUsage::single(11, 2, 13)),
                reasoning: None,
            },
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
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "nonexistent".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: None,
                reasoning: None,
            },
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
            Ok(LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: None,
                reasoning: None,
            })
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
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: None,
                reasoning: None,
            },
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

    // ------------------------------------------------------------------
    // Steering
    // ------------------------------------------------------------------

    /// The ordinary case: the user speaks while a tool is running, and the
    /// model's next call has to see it. A tool that steers the turn it is
    /// running in puts the message in at exactly that moment.
    #[test]
    fn a_turn_steered_during_a_tool_call_carries_the_message_into_the_next_prompt() {
        struct SteersItself(TurnContext);

        impl crate::tool::Tool for SteersItself {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "steers the turn it is running in"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn call(&self, _args: serde_json::Value) -> Result<ToolResult, AgentError> {
                assert!(
                    self.0.steer.push("actually, use tabs".to_string()),
                    "the turn is mid-tool-call, so it is still reading"
                );
                Ok(ToolResult::text("echoed".to_string()))
            }
        }

        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(SteersItself(ctx.clone())));

        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    arguments: serde_json::json!({}),
                }],
                usage: None,
                reasoning: None,
            },
            LlmResponse::Text {
                content: "done, with tabs".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let (text, _reasoning, _usage) =
            run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx).unwrap();

        assert_eq!(text, "done, with tabs");
        assert_eq!(
            provider.call_count.load(Ordering::SeqCst),
            2,
            "steering carries the turn on rather than starting a new one"
        );

        // `messages` is what the loop hands the model, so its shape is the
        // assertion that the steer actually reached it — and where it landed.
        let tool_at = messages
            .iter()
            .position(|m| m.role == ChatRole::Tool)
            .expect("the tool result");
        let steer_at = messages
            .iter()
            .position(|m| m.role == ChatRole::User && m.content == "actually, use tabs")
            .expect("the steered message must reach the prompt");
        assert!(
            steer_at > tool_at,
            "a steer belongs after the work it interrupted, not before it"
        );
        assert!(!ctx.steer.has_pending(), "delivered once, then gone");
    }

    /// Answers, and steers the turn as it does so on the first call — the user
    /// still typing while the model composed its reply.
    struct SteersWhileAnswering {
        inbox: crate::cancel::SteerInbox,
        call_count: AtomicUsize,
    }

    impl LlmProvider for SteersWhileAnswering {
        fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok("unused".to_string())
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
            if idx == 0 {
                // The user was still typing while this answer was produced.
                assert!(
                    self.inbox.push("wait — in Python".to_string()),
                    "the loop has not reached its decision to stop reading yet"
                );
            }
            Ok(LlmResponse::Text {
                content: if idx == 0 {
                    "here it is in Go"
                } else {
                    "here it is in Python"
                }
                .to_string(),
                reasoning: None,
                usage: None,
            })
        }
    }

    /// The awkward case: the steer lands while the model is composing its
    /// answer, so it arrives an instant after the turn would have ended. The
    /// turn has to carry on instead — otherwise the message is silently lost.
    #[test]
    fn a_steer_that_lands_as_the_model_answers_continues_the_turn() {
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let provider = SteersWhileAnswering {
            inbox: ctx.steer.clone(),
            call_count: AtomicUsize::new(0),
        };
        let registry = ToolRegistry::new();
        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("write it".to_string())];

        let (text, _reasoning, _usage) = run_observed(
            &provider,
            &mut messages,
            &registry,
            Some(5),
            Some(&recorder),
            &ctx,
        )
        .unwrap();

        assert_eq!(
            text, "here it is in Python",
            "the turn ends with the answer that took the steer into account"
        );
        assert_eq!(
            recorder.lines.lock().unwrap().clone(),
            vec![
                // The superseded answer is still shown: it was produced, and a
                // client that never saw it would show the user a gap.
                "message here it is in Go".to_string(),
                "turn here it is in Python".to_string(),
            ]
        );
        assert!(
            messages
                .iter()
                .any(|m| m.role == ChatRole::Assistant && m.content == "here it is in Go"),
            "the superseded answer stays in the transcript the model reasons from"
        );
    }

    /// `max_iterations` bounds the model's own looping. A round the *user* asked
    /// for by steering is not that, and charging it would mean a steer arriving
    /// on the last iteration turns an answer that was already produced into a
    /// failed turn — which `runtime::run_turn` then rolls the history back over.
    #[test]
    fn a_steered_continuation_is_not_charged_against_the_iteration_budget() {
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let provider = SteersWhileAnswering {
            inbox: ctx.steer.clone(),
            call_count: AtomicUsize::new(0),
        };
        let registry = ToolRegistry::new();
        let mut messages = vec![ChatMessage::user("write it".to_string())];

        // One iteration of budget: exactly enough for the model to answer once.
        let (text, _reasoning, _usage) =
            run_observed(&provider, &mut messages, &registry, Some(1), None, &ctx).unwrap();

        assert_eq!(
            text, "here it is in Python",
            "the steered turn must reach its second answer, not exhaust the budget"
        );
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 2);
    }

    /// Once the loop has decided to stop reading, the inbox says so — which is
    /// what lets the app-server refuse a late steer instead of acknowledging
    /// text nobody will ever be given.
    #[test]
    fn a_finished_turn_stops_accepting_steering() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "done".to_string(),
            reasoning: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let mut messages = vec![ChatMessage::user("go".to_string())];

        run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx).unwrap();

        assert!(!ctx.steer.push("too late".to_string()));
    }

    /// Running out of iterations is an ending too, and the widest window in
    /// which a steer could be accepted by a turn that has stopped reading: the
    /// loop exits, then formats an error and *emits* it, which on the app-server
    /// means taking the connection's lock and writing a notification.
    ///
    /// The observer here steers from inside that window — the one moment the
    /// race is reachable deterministically — and must be refused.
    #[test]
    fn a_steer_is_refused_from_inside_the_out_of_iterations_ending() {
        /// Tries to steer the turn while its ending is being announced.
        struct SteersDuringTheEnding {
            inbox: crate::cancel::SteerInbox,
            accepted: std::sync::Mutex<Vec<bool>>,
        }

        impl AgentObserver for SteersDuringTheEnding {
            fn on_event(&self, event: AgentEvent<'_>) {
                if let AgentEvent::Error { .. } = event {
                    self.accepted
                        .lock()
                        .unwrap()
                        .push(self.inbox.push("too late".to_string()));
                }
            }
        }

        // Always asks for a tool, so the turn can only end by running out.
        let provider = MockProvider::new(vec![LlmResponse::ToolCalls {
            calls: vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "echo".to_string(),
                arguments: serde_json::json!({"text": "hi"}),
            }],
            usage: None,
            reasoning: None,
        }]);
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let observer = SteersDuringTheEnding {
            inbox: ctx.steer.clone(),
            accepted: std::sync::Mutex::new(Vec::new()),
        };
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(
            &provider,
            &mut messages,
            &registry,
            Some(1),
            Some(&observer),
            &ctx,
        );

        assert!(
            result.is_err(),
            "one iteration of budget, always tool calls"
        );
        assert_eq!(
            observer.accepted.lock().unwrap().clone(),
            vec![false],
            "the turn had already stopped reading when it announced its ending"
        );
    }

    /// The same for an ending the turn did not choose. A cancelled turn's
    /// history is rolled back, so there is nowhere for a steer to land.
    #[test]
    fn a_cancelled_turn_also_stops_accepting_steering() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "unreached".to_string(),
            reasoning: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        ctx.cancellation.cancel();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx);

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert!(!ctx.steer.push("too late".to_string()));
    }

    /// Nothing pushed, nothing changed: the inbox costs an empty check per
    /// boundary and no more.
    #[test]
    fn a_turn_nobody_steers_runs_exactly_as_before() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "done".to_string(),
            reasoning: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let (text, _reasoning, _usage) = run(&provider, &mut messages, &registry, Some(5)).unwrap();

        assert_eq!(text, "done");
        assert_eq!(provider.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(messages.len(), 1, "only the prompt we put in");
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
