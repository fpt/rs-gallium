use crate::cancel::TurnContext;
use crate::event::{self, AgentEvent, AgentObserver};
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo};
use crate::memory;
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
    // No window, so no mid-turn compaction: a caller with no context budget to
    // enforce is one that has not said what the budget is, and guessing one
    // would drop history on a turn nobody asked to bound.
    run_observed(
        client,
        messages,
        tools,
        max_iterations,
        None,
        &TurnContext::detached(),
        0,
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
    context_window: u32,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    // Filled in by the loop if mid-turn compaction drops anything, and put back
    // here if the turn then fails. The caller recovers a failed turn's history
    // by truncating its own additions, which cannot restore messages taken out
    // of the *middle* — so the loop that removed them owns putting them back.
    let mut pre_compaction = None;
    let outcome = react_loop(
        client,
        messages,
        tools,
        max_iterations,
        observer,
        ctx,
        ContextBudget {
            window: context_window,
            restore: &mut pre_compaction,
        },
    );

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
        if let Some(original) = pre_compaction {
            *messages = original;
        }
    }
    outcome
}

/// Compact the running transcript if it has grown past the trigger, taking a
/// snapshot the first time it drops anything.
///
/// The snapshot is why this is a function and not three lines in the loop:
/// compaction removes exchanges from the *middle* of the history, and a caller
/// recovering a failed turn by truncating its own additions cannot put those
/// back. Taken lazily, so a turn that never compacts never clones.
fn compact_within_turn(
    messages: &mut Vec<ChatMessage>,
    last_input_tokens: u64,
    budget: &mut ContextBudget<'_>,
) {
    let Some(target) = memory::compaction_target(
        last_input_tokens,
        memory::estimate_messages_tokens(messages),
        budget.window,
    ) else {
        return;
    };

    let snapshot = messages.clone();
    let dropped = memory::compact_active_turn(messages, target);
    if dropped == 0 {
        // Nothing came out, so there is nothing to put back. Arming the restore
        // here would replace the history with an identical copy on failure —
        // harmless, but it would also claim a compaction that did not happen.
        return;
    }

    tracing::info!(
        "Context compacted mid-turn: dropped {dropped} message(s) to reach {target} tokens \
         (window {}). The next prompt is no longer a prefix of the KV cache, so it is \
         re-evaluated in full.",
        budget.window
    );

    // Only the first snapshot is kept: the earliest state is the one a failed
    // turn has to be restored to, and a later compaction would otherwise
    // overwrite it with an already-shortened history.
    budget.restore.get_or_insert(snapshot);
}

/// The window a running turn has to stay inside, and the history to put back if
/// it does not finish.
///
/// One struct rather than two parameters because they are one decision: the
/// window is what makes mid-turn compaction happen, and the snapshot is what
/// makes it safe. A caller that had the first without the second would drop
/// history that a failed turn could not recover.
struct ContextBudget<'a> {
    /// `0` disables compaction, the same convention
    /// [`memory::compaction_target`] uses.
    window: u32,
    /// The history as it stood before the first mid-turn drop. `None` until one
    /// happens, so a turn that never compacts never pays for the clone.
    restore: &'a mut Option<Vec<ChatMessage>>,
}

fn react_loop(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
    observer: Option<&dyn AgentObserver>,
    ctx: &TurnContext,
    mut budget: ContextBudget<'_>,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    let max_iter = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);
    let mut total_usage = TokenUsage::default();

    let emit = |event: AgentEvent<'_>| event::emit(observer, event);

    // The prompt and the catalog as the model first sees them. Recorded before
    // the loop rather than per iteration: later prompts are this list plus the
    // tool transcript the trace already holds.
    //
    // *As it first sees them* is now a narrower claim than it was: `ToolSearch`
    // can reveal a deferred tool mid-turn, so the projection grows. The initial
    // one is still what belongs here — a trace records the turn as it was set
    // up, and each reveal is a tool call the trace already holds, so the growth
    // is reconstructible from what is recorded. What is not reconstructible is
    // the starting point, once it stops being recorded.
    if let Some(trace) = &ctx.trace {
        trace.record_prompt(messages, &tools.get_definitions());
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
    // What the last model call reported its prompt cost, which is what
    // compaction measures against — an estimate is only a stand-in until a
    // provider has said. `0` before the first call, where the estimate stands
    // alone.
    let mut last_input_tokens = 0u64;

    while charged < max_iter {
        ctx.check()?;

        // Whatever the user said while the turn was running goes in before the
        // model is asked again — the point of steering is that the next model
        // call sees it, and this is the last moment at which that is still true.
        take_steering(ctx, messages);

        // Bound the transcript *inside* the turn, not only at its start.
        //
        // `runtime::run_turn` compacts once, before the loop, on the premise
        // that a turn begins roughly where the last one ended. An agentic turn
        // does not: this loop appends an assistant message and a tool result per
        // iteration, so a turn that starts comfortably inside the window can
        // leave it without any turn boundary in between. Measured on a klein
        // session: a fresh thread — no prior usage, so the turn-start check saw
        // nothing to do — reached 25 050 tokens by its sixth tool call against a
        // 24 576-token context, and the turn died on a prompt that would not
        // fit. Compaction had been available the whole time and was never asked.
        //
        // The same policy as the turn-start check, deliberately: one rule for
        // when history is too long, whichever boundary notices. It fires at a
        // fraction of the window rather than when the next prompt would
        // overflow, which leaves room for the model's own output — a prompt
        // squeezed in with nothing left to generate into has not been saved.
        //
        // It costs the KV cache on the iteration it fires: compaction rewrites
        // the front of the transcript, so the next prompt is no longer a prefix
        // of what the slot holds and the whole thing is re-evaluated. That is
        // the trade — a slow iteration against a failed turn.
        compact_within_turn(messages, last_input_tokens, &mut budget);

        calls += 1;
        tracing::info!("ReAct iteration {}/{}", charged + 1, max_iter);

        // Recomputed per iteration rather than hoisted, because the projection
        // is no longer fixed for the turn: a `ToolSearch` call in the previous
        // iteration reveals tools, and revealing them is worth nothing unless
        // the very next model call carries their schemas. Cheap — the registry
        // stores each descriptor rather than rebuilding it, so this is a clone
        // of a handful of small values, against a model call.
        let tool_defs = tools.get_definitions();

        let asked_at = std::time::Instant::now();
        // Deltas stream straight through to the observer as the provider decodes
        // them; providers that don't stream simply never call this closure.
        let mut on_delta = |text: &str| event::emit(observer, AgentEvent::MessageDelta { text });
        let response = match client.chat_with_tools_streaming(
            messages,
            &tool_defs,
            &ctx.cancellation,
            &mut on_delta,
        ) {
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
                // Recorded above via `trace.record_response`; the loop itself
                // has no use for the pre-parse decode.
                raw: _,
            } => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                    last_input_tokens = u.input_tokens;
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
                raw: _,
            } => {
                if let Some(ref u) = usage {
                    total_usage.add(u);
                    last_input_tokens = u.input_tokens;
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
    use std::sync::{Arc, Mutex};

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
                        ..
                    } => Ok(LlmResponse::Text {
                        content: content.clone(),
                        reasoning: reasoning.clone(),
                        raw: None,
                        usage: usage.clone(),
                    }),
                    LlmResponse::ToolCalls {
                        calls,
                        usage,
                        reasoning,
                        ..
                    } => Ok(LlmResponse::ToolCalls {
                        calls: calls.clone(),
                        usage: usage.clone(),
                        reasoning: reasoning.clone(),
                        raw: None,
                    }),
                }
            } else {
                Ok(LlmResponse::Text {
                    content: "fallback".to_string(),
                    reasoning: None,
                    raw: None,
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
            raw: None,
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
                raw: None,
            },
            LlmResponse::Text {
                content: "There are no tasks.".to_string(),
                reasoning: None,
                raw: None,
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

    /// A `ToolSearch` in one iteration must reach the model in the *next* one.
    ///
    /// This is what the per-iteration `get_definitions()` buys. Hoisted above
    /// the loop — as it was — the reveal changes the registry and the model
    /// never learns of it, so the tool it just asked for stays as invisible as
    /// before and the turn is one wasted call further along. The test watches
    /// what each model call was actually offered, because that is the only place
    /// the difference shows.
    #[test]
    fn a_tool_revealed_mid_turn_reaches_the_next_model_call() {
        /// Records the tool names offered on every call, in order.
        struct OfferWatcher {
            seen: Mutex<Vec<Vec<String>>>,
            calls: AtomicUsize,
        }

        impl LlmProvider for OfferWatcher {
            fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
                Ok("unused".to_string())
            }

            fn supports_tools(&self) -> bool {
                true
            }

            fn chat_with_tools(
                &self,
                _messages: &[ChatMessage],
                tools: &[ToolDefinition],
            ) -> anyhow::Result<LlmResponse> {
                self.seen
                    .lock()
                    .unwrap()
                    .push(tools.iter().map(|t| t.name.clone()).collect());
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(LlmResponse::ToolCalls {
                        calls: vec![ToolCallInfo {
                            id: "call_1".to_string(),
                            name: "ToolSearch".to_string(),
                            arguments: serde_json::json!({ "query": "directory tree" }),
                        }],
                        usage: None,
                        reasoning: None,
                        raw: None,
                    });
                }
                Ok(LlmResponse::Text {
                    content: "done".to_string(),
                    reasoning: None,
                    raw: None,
                    usage: None,
                })
            }
        }

        struct TreeDir;
        impl crate::tool::Tool for TreeDir {
            fn name(&self) -> &str {
                "tree_dir"
            }
            fn description(&self) -> &str {
                "walk a directory tree"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn call(&self, _args: serde_json::Value) -> Result<ToolResult, crate::AgentError> {
                Ok(ToolResult::text("walked".to_string()))
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(TreeDir));
        tools.visibility().hide("tree_dir", "walk a directory tree");
        let visibility = Arc::clone(tools.visibility());
        tools.register(Box::new(crate::tool_search::ToolSearchTool::new(
            visibility,
        )));

        let provider = OfferWatcher {
            seen: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        };
        let mut messages = vec![ChatMessage::user("walk the tree".to_string())];
        let (text, _, _) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "done");

        let seen = provider.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "expected two model calls: {seen:?}");
        assert!(
            !seen[0].contains(&"tree_dir".to_string()),
            "deferred at the start: {:?}",
            seen[0]
        );
        assert!(
            seen[1].contains(&"tree_dir".to_string()),
            "revealed by the search, so the next call must carry it: {:?}",
            seen[1]
        );
    }

    /// The klein turn that produced this: a **fresh thread**, so the turn-start
    /// compaction saw no prior usage and did nothing, and six tool calls later
    /// the prompt did not fit the context.
    ///
    /// The loop now applies the same policy at each boundary. The check is on
    /// what the model was actually sent — the task still there, the oldest tool
    /// output gone — because the history is the only place the difference shows.
    #[test]
    fn a_turn_that_outgrows_its_window_compacts_instead_of_failing() {
        /// Reports a prompt cost above the compaction trigger, and records the
        /// history it was handed on every call.
        struct BulkyProvider {
            seen: Mutex<Vec<Vec<String>>>,
            calls: AtomicUsize,
        }

        impl LlmProvider for BulkyProvider {
            fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
                Ok("unused".to_string())
            }

            fn supports_tools(&self) -> bool {
                true
            }

            fn chat_with_tools(
                &self,
                messages: &[ChatMessage],
                _tools: &[ToolDefinition],
            ) -> anyhow::Result<LlmResponse> {
                self.seen
                    .lock()
                    .unwrap()
                    .push(messages.iter().map(|m| m.content.clone()).collect());
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                // 950 of a 1000-token window: past the 90% trigger.
                let usage = Some(TokenUsage {
                    input_tokens: 950,
                    ..Default::default()
                });
                if n < 2 {
                    return Ok(LlmResponse::ToolCalls {
                        calls: vec![ToolCallInfo {
                            id: format!("call_{n}"),
                            name: "Bulk".to_string(),
                            arguments: serde_json::json!({}),
                        }],
                        usage,
                        reasoning: None,
                        raw: None,
                    });
                }
                Ok(LlmResponse::Text {
                    content: "done".to_string(),
                    reasoning: None,
                    raw: None,
                    usage,
                })
            }
        }

        struct BulkTool;
        impl crate::tool::Tool for BulkTool {
            fn name(&self) -> &str {
                "Bulk"
            }
            fn description(&self) -> &str {
                "returns a lot"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn call(&self, _args: serde_json::Value) -> Result<ToolResult, crate::AgentError> {
                Ok(ToolResult::text("y".repeat(4000)))
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(BulkTool));

        let provider = BulkyProvider {
            seen: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        };
        let mut messages = vec![ChatMessage::user("why did CI fail?".to_string())];
        let ctx = TurnContext::detached();
        let (text, _, _) =
            run_observed(&provider, &mut messages, &tools, Some(10), None, &ctx, 1000).unwrap();
        assert_eq!(text, "done");

        let seen = provider.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 3, "the turn ran to completion: {seen:?}");
        assert!(
            seen[2].iter().any(|c| c == "why did CI fail?"),
            "the task must still be in front of the model: {:?}",
            seen[2].iter().map(|c| c.len()).collect::<Vec<_>>()
        );
        assert!(
            seen[2].iter().filter(|c| c.len() == 4000).count() < 2,
            "the oldest bulky tool result should have been compacted away: {:?}",
            seen[2].iter().map(|c| c.len()).collect::<Vec<_>>()
        );
    }

    /// A turn that compacts and *then* fails must leave the caller's history as
    /// it found it.
    ///
    /// `runtime::run_turn` recovers a failed turn by truncating the turn's own
    /// additions, which cannot restore messages taken out of the middle — so the
    /// loop that removed them puts them back on the way out. Without this a
    /// failed turn silently costs the user history they never asked to lose.
    #[test]
    fn a_turn_that_compacts_then_fails_puts_the_history_back() {
        struct FailsAfterOneCall {
            calls: AtomicUsize,
        }

        impl LlmProvider for FailsAfterOneCall {
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
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(LlmResponse::ToolCalls {
                        calls: vec![ToolCallInfo {
                            id: "call_0".to_string(),
                            name: "Tasks".to_string(),
                            arguments: serde_json::json!({ "action": "list" }),
                        }],
                        usage: Some(TokenUsage {
                            input_tokens: 950,
                            ..Default::default()
                        }),
                        reasoning: None,
                        raw: None,
                    });
                }
                anyhow::bail!("the provider fell over")
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(crate::tool::TaskTool::new()));

        // Two prior turns, the older one bulky enough that compaction reaches it.
        let mut messages = vec![
            ChatMessage::user("an old question".to_string()),
            ChatMessage::assistant("z".repeat(4000)),
            ChatMessage::user("the current task".to_string()),
        ];
        let before = messages.clone();

        let provider = FailsAfterOneCall {
            calls: AtomicUsize::new(0),
        };
        let ctx = TurnContext::detached();
        let result = run_observed(&provider, &mut messages, &tools, Some(10), None, &ctx, 1000);
        assert!(result.is_err(), "the provider was supposed to fail");

        assert_eq!(
            messages[..before.len()]
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>(),
            before.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
            "the pre-turn history has to come back — the caller can only truncate \
             what the turn appended, not restore what it dropped from the middle"
        );
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
                raw: None,
            },
            LlmResponse::Text {
                content: "I can see a Chrome window.".to_string(),
                reasoning: None,
                raw: None,
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
                raw: None,
            },
            LlmResponse::Text {
                content: "done".to_string(),
                reasoning: None,
                raw: None,
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
                raw: None,
            },
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_2".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
                raw: None,
            },
            LlmResponse::ToolCalls {
                calls: vec![ToolCallInfo {
                    id: "call_3".to_string(),
                    name: "tasks".to_string(),
                    arguments: serde_json::json!({"action": "list"}),
                }],
                usage: None,
                reasoning: None,
                raw: None,
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
                    raw: None,
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
                AgentEvent::MessageDelta { text } => format!("delta {text}"),
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
                raw: None,
            },
            LlmResponse::Text {
                content: "done".to_string(),
                reasoning: None,
                raw: None,
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
            0,
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

    /// A provider that streams (overrides `chat_with_tools_streaming`): the
    /// deltas reach the observer as `MessageDelta` in order, before the turn's
    /// final text.
    #[test]
    fn streamed_deltas_reach_the_observer_before_the_final_text() {
        struct StreamingMock;
        impl LlmProvider for StreamingMock {
            fn chat(&self, _: &[ChatMessage]) -> anyhow::Result<String> {
                Ok(String::new())
            }
            fn supports_tools(&self) -> bool {
                true
            }
            fn chat_with_tools_streaming(
                &self,
                _messages: &[ChatMessage],
                _tools: &[ToolDefinition],
                _cancel: &crate::cancel::CancellationToken,
                on_delta: &mut dyn FnMut(&str),
            ) -> anyhow::Result<LlmResponse> {
                for frag in ["Paris ", "is the ", "capital."] {
                    on_delta(frag);
                }
                Ok(LlmResponse::Text {
                    content: "Paris is the capital.".to_string(),
                    reasoning: None,
                    raw: None,
                    usage: None,
                })
            }
        }

        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("capital of France?".to_string())];
        let (text, _, _) = run_observed(
            &StreamingMock,
            &mut messages,
            &ToolRegistry::new(),
            Some(3),
            Some(&recorder),
            &TurnContext::detached(),
            0,
        )
        .unwrap();
        assert_eq!(text, "Paris is the capital.");
        assert_eq!(
            recorder.lines.lock().unwrap().clone(),
            vec![
                "delta Paris ".to_string(),
                "delta is the ".to_string(),
                "delta capital.".to_string(),
                "turn Paris is the capital.".to_string(),
            ]
        );
    }

    /// A non-streaming provider (the default) emits no deltas — react must not
    /// invent them.
    #[test]
    fn a_non_streaming_provider_emits_no_deltas() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "hi".to_string(),
            reasoning: None,
            raw: None,
            usage: None,
        }]);
        let recorder = Recorder::default();
        let mut messages = vec![ChatMessage::user("hi".to_string())];
        run_observed(
            &provider,
            &mut messages,
            &ToolRegistry::new(),
            Some(3),
            Some(&recorder),
            &TurnContext::detached(),
            0,
        )
        .unwrap();
        let lines = recorder.lines.lock().unwrap().clone();
        assert!(
            !lines.iter().any(|l| l.starts_with("delta ")),
            "no deltas expected, got {lines:?}"
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
                raw: None,
            },
            LlmResponse::Text {
                content: "gave up".to_string(),
                reasoning: None,
                raw: None,
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
            0,
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
                raw: None,
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
            0,
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
                raw: None,
            },
            LlmResponse::Text {
                content: "should never be asked for".to_string(),
                reasoning: None,
                raw: None,
                usage: None,
            },
        ]);
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx, 0);

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
            raw: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        ctx.cancellation.cancel();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx, 0);

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
                raw: None,
            },
            LlmResponse::Text {
                content: "done, with tabs".to_string(),
                reasoning: None,
                raw: None,
                usage: None,
            },
        ]);
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let (text, _reasoning, _usage) =
            run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx, 0).unwrap();

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
                raw: None,
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
            0,
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
            run_observed(&provider, &mut messages, &registry, Some(1), None, &ctx, 0).unwrap();

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
            raw: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        let mut messages = vec![ChatMessage::user("go".to_string())];

        run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx, 0).unwrap();

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
            raw: None,
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
            0,
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
            raw: None,
            usage: None,
        }]);
        let registry = ToolRegistry::new();
        let ctx = TurnContext::new(crate::cancel::CancellationToken::new());
        ctx.cancellation.cancel();
        let mut messages = vec![ChatMessage::user("go".to_string())];

        let result = run_observed(&provider, &mut messages, &registry, Some(5), None, &ctx, 0);

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
            raw: None,
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
