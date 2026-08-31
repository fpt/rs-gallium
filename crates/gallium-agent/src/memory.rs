use crate::llm::{ChatMessage, ChatRole};

// ============================================================================
// Compaction policy
//
// One policy, applied in one place: `runtime::run_turn` triggers on
// `compaction_target` and sizes history with `estimate_message_tokens`.
// ============================================================================

/// Context window assumed when nothing configures one.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

/// The window a conversation runs against, and whether anyone can vouch for it.
///
/// The two are not the same question. Compaction needs a number no matter what,
/// so it takes a fallback when nothing better is available. A context gauge
/// shown to a user does not: a share of a made-up denominator reads as fact, and
/// removing exactly that was `fpt/voice-agent#18`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextWindow {
    /// What compaction measures against. Always a number, and `0` when
    /// compaction is deliberately switched off.
    pub effective: u32,
    /// The same number, when it came from somewhere real and is a window
    /// anything could be a share of. `None` means no gauge should be drawn.
    ///
    /// **Never `Some(0)`.** Everything downstream divides by this — the REPL's
    /// percentage, and any client handed `modelContextWindow` over the wire —
    /// so zero is excluded here rather than guarded for at each of them.
    pub known: Option<u32>,
}

/// Settle the window from what the user said and what the model reports.
///
/// `configured` wins: someone who sets `contextWindow` is describing their own
/// setup — a llama.cpp server started with a smaller `n_ctx`, or a deliberately
/// earlier compaction — and knows something the model file does not. Failing
/// that, `reported` is the model's own metadata. Failing both, the fallback
/// keeps compaction working and the gauge dark.
///
/// Zero means different things on the two inputs, so it is handled twice.
/// Configured zero is the sentinel that switches compaction off, and is honored
/// as such — but it is not a window, so nothing is displayed against it.
/// Reported zero is a model file saying nothing useful, and is discarded before
/// it can switch compaction off by accident.
pub fn resolve_context_window(
    configured: Option<u32>,
    reported: Option<u32>,
    fallback: u32,
) -> ContextWindow {
    let reported = reported.filter(|w| *w > 0);
    let settled = configured.or(reported);
    ContextWindow {
        effective: settled.unwrap_or(fallback),
        known: settled.filter(|w| *w > 0),
    }
}

/// Fraction of the context window the previous turn's prompt must reach before
/// history is compacted.
const COMPACTION_TRIGGER: f64 = 0.9;

/// Fraction of the context window to compact down to, leaving room for the
/// turn that is about to run.
const COMPACTION_TARGET: f64 = 0.5;

/// Estimated token cost of one message (~4 chars/token, plus per-message
/// framing overhead).
pub fn estimate_message_tokens(message: &ChatMessage) -> usize {
    message.content.len() / 4 + 10
}

/// Estimated token cost of a whole history.
pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// The budget to compact history down to, or `None` when the conversation is
/// not yet close enough to the window to bother.
///
/// `last_input_tokens` is the previous turn's peak prompt *as reported by the
/// provider* — ground truth when we have it, and `0` before the first turn
/// completes. A provider that reports no usage would otherwise leave compaction
/// blind, so `estimated_tokens` (our own count of the history) is taken as a
/// floor. Both local backends do report now; the floor stays because a provider
/// is allowed not to, and silently never compacting is the failure this policy
/// exists to prevent.
pub fn compaction_target(
    last_input_tokens: u64,
    estimated_tokens: usize,
    context_window: u32,
) -> Option<usize> {
    if context_window == 0 {
        return None;
    }
    let observed = last_input_tokens.max(estimated_tokens as u64);
    let threshold = (context_window as f64 * COMPACTION_TRIGGER) as u64;
    (observed >= threshold).then_some((context_window as f64 * COMPACTION_TARGET) as usize)
}

/// Drop oldest history until the estimate is under `target_tokens`, a whole
/// exchange at a time. Returns the number of messages dropped.
///
/// This runs over a history that has been through the ReAct loop, so an exchange
/// is not just a user/assistant pair: it is a user message plus the assistant
/// replies, tool calls, and `Tool` results that answered it. Dropping those individually would leave two
/// kinds of wreckage behind:
///
/// - a `Tool` result whose assistant tool-call is gone, which providers reject
///   outright;
/// - an assistant reply whose user message is gone — an answer to a question no
///   longer in the history. It costs context to say nothing, and a chat template
///   that expects a user-first or strictly alternating history (several GGUFs
///   embed one, and `llm_local` renders it verbatim) can fail on it.
///
/// So each pass removes a message and everything up to the next user message,
/// which leaves the retained history starting at a user turn.
pub fn compact_messages(messages: &mut Vec<ChatMessage>, target_tokens: usize) -> usize {
    let mut dropped = 0;
    while estimate_messages_tokens(messages) > target_tokens {
        let Some(start) = messages.iter().position(|m| m.role != ChatRole::System) else {
            break; // Only the system prompt is left; it is not ours to drop.
        };
        // Everything up to the next user message answered the same prompt. Stop
        // at a system message too — those are never ours to drop.
        let mut end = start + 1;
        while end < messages.len()
            && messages[end].role != ChatRole::User
            && messages[end].role != ChatRole::System
        {
            end += 1;
        }
        messages.drain(start..end);
        dropped += end - start;
    }
    dropped
}

/// Compact a history that a turn is **still running in**, keeping the task.
///
/// [`compact_messages`] is right between turns and wrong inside one. It drops a
/// message and everything up to the next user message, and the current turn's
/// prompt *is* a user message with no later one behind it — so the first pass
/// that reaches it takes the prompt and the whole turn with it, and the model is
/// left reasoning about a task it can no longer read.
///
/// The distinction only shows up on a thread with no history to give: with prior
/// turns available both functions drop those first and agree. On a **fresh**
/// thread every message belongs to the running turn, so "drop whole exchanges,
/// oldest first" has exactly one exchange to choose from. That is the case that
/// motivated this — a first turn reaching 25 050 tokens against a 24 576-token
/// context by its sixth tool call.
///
/// So this pins the newest user message and works around it:
///
/// 1. whole exchanges *before* the pin — prior turns, the cheapest thing to
///    lose, and exactly what [`compact_messages`] would have done;
/// 2. then tool exchanges *after* it, oldest first — an assistant message and
///    the tool results answering it, dropped together so no result is orphaned
///    from the call that produced it.
///
/// Phase 2 is the model losing its own working notes mid-task: it keeps what it
/// was asked and its most recent findings, and forgets the oldest ones. Worth
/// saying plainly because it is a real loss — a turn may redo work it already
/// did. It buys the alternative being a turn that cannot continue at all.
///
/// The pin itself is never dropped. A prompt larger than the target on its own
/// therefore ends compaction over budget rather than empty, and the caller finds
/// out from the model call, not from a history with no task in it.
pub fn compact_active_turn(messages: &mut Vec<ChatMessage>, target_tokens: usize) -> usize {
    // The newest user message is the task in hand. Newest rather than first
    // because `turn/steer` appends one mid-turn, and a steered turn's task is
    // what the user just said.
    if !messages.iter().any(|m| m.role == ChatRole::User) {
        // No user message at all — nothing to protect, so the between-turns rule
        // is already the right one.
        return compact_messages(messages, target_tokens);
    }

    let mut dropped = 0;

    // Phase 1: whole exchanges before the pin.
    while estimate_messages_tokens(messages) > target_tokens {
        let Some(start) = messages.iter().position(|m| m.role != ChatRole::System) else {
            break;
        };
        let pin_now = messages
            .iter()
            .rposition(|m| m.role == ChatRole::User)
            .unwrap_or(start);
        if start >= pin_now {
            break; // Reached the task; the rest is phase 2's to decide.
        }
        let mut end = start + 1;
        while end < messages.len()
            && messages[end].role != ChatRole::User
            && messages[end].role != ChatRole::System
        {
            end += 1;
        }
        messages.drain(start..end);
        dropped += end - start;
    }

    // Phase 2: tool exchanges after the pin, oldest first.
    while estimate_messages_tokens(messages) > target_tokens {
        let Some(pin) = messages.iter().rposition(|m| m.role == ChatRole::User) else {
            break;
        };
        let start = pin + 1;
        if start >= messages.len() {
            break; // Only the task is left, and it is not ours to drop.
        }
        // One assistant message and every non-assistant message answering it, so
        // a `Tool` result never outlives the call it belongs to.
        let mut end = start + 1;
        while end < messages.len() && messages[end].role == ChatRole::Tool {
            end += 1;
        }
        messages.drain(start..end);
        dropped += end - start;
    }

    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCallInfo;

    #[test]
    fn compaction_target_holds_off_until_the_window_is_nearly_full() {
        // 89% of the window: not yet.
        assert_eq!(compaction_target(890, 0, 1000), None);
        // 90% is the trigger, and the target is half the window.
        assert_eq!(compaction_target(900, 0, 1000), Some(500));
        assert_eq!(compaction_target(1200, 0, 1000), Some(500));
    }

    #[test]
    fn an_explicit_window_wins_over_what_the_model_reports() {
        let w = resolve_context_window(Some(4096), Some(32768), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.effective, 4096);
        assert_eq!(w.known, Some(4096), "the user's number is a known one");
    }

    #[test]
    fn a_model_that_reports_its_window_beats_the_fallback() {
        let w = resolve_context_window(None, Some(32768), 8192);
        assert_eq!(
            w.effective, 32768,
            "compacting at the fallback would trim a conversation the model could still hold"
        );
        assert_eq!(w.known, Some(32768));
    }

    /// The case the gauge exists to get right: nobody knows, so compaction still
    /// has a policy and the client is told nothing to draw.
    #[test]
    fn a_window_nobody_can_vouch_for_is_usable_but_not_reportable() {
        let w = resolve_context_window(None, None, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.effective, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.known, None);
    }

    /// Zero disables compaction. It is not a window, and everything downstream
    /// divides by `known` — so switching compaction off must not hand a client a
    /// denominator of zero.
    #[test]
    fn compaction_switched_off_is_not_a_window() {
        let w = resolve_context_window(Some(0), Some(128_000), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.effective, 0, "the sentinel still switches compaction off");
        assert_eq!(
            w.known, None,
            "nothing can be shown as a share of zero, whatever the model reports"
        );
    }

    /// The other direction. A model file claiming a window of zero has said
    /// nothing, and must not be mistaken for the disable sentinel — that would
    /// switch off compaction on the strength of bad metadata.
    #[test]
    fn a_model_reporting_zero_is_a_model_that_said_nothing() {
        let w = resolve_context_window(None, Some(0), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.effective, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(w.known, None);
    }

    #[test]
    fn compaction_target_is_none_without_a_measurement() {
        // Nothing reported and nothing in history yet.
        assert_eq!(compaction_target(0, 0, 1000), None);
        // Compaction explicitly disabled.
        assert_eq!(compaction_target(999_999, 999_999, 0), None);
    }

    #[test]
    fn compaction_target_falls_back_to_the_estimate_when_usage_is_unreported() {
        // The native candle backend reports 0 usage forever; the estimated
        // history size must still be able to trigger compaction.
        assert_eq!(compaction_target(0, 950, 1000), Some(500));
        assert_eq!(compaction_target(0, 100, 1000), None);
        // A reported count below the estimate does not mask it.
        assert_eq!(compaction_target(10, 950, 1000), Some(500));
    }

    #[test]
    fn compact_messages_drops_oldest_and_keeps_system() {
        let mut messages = vec![ChatMessage::system("sys".to_string())];
        for i in 0..10 {
            messages.push(ChatMessage::user(format!(
                "Message {} {}",
                i,
                "x".repeat(380)
            )));
        }

        let dropped = compact_messages(&mut messages, 500);
        assert!(dropped > 0);
        assert_eq!(messages[0].role, ChatRole::System, "system must survive");
        assert!(
            messages.last().unwrap().content.starts_with("Message 9"),
            "the newest message must survive"
        );
        assert!(estimate_messages_tokens(&messages) <= 500);
    }

    #[test]
    fn compact_messages_drops_whole_exchanges() {
        // Two complete exchanges; only the newer one can fit the target.
        let mut messages = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user(format!("first question {}", "x".repeat(4000))),
            ChatMessage::assistant("first answer".to_string()),
            ChatMessage::user("second question".to_string()),
            ChatMessage::assistant("second answer".to_string()),
        ];

        let dropped = compact_messages(&mut messages, 100);

        // The stale answer must not outlive the question it answered.
        assert_eq!(
            dropped, 2,
            "the whole first exchange goes, not just the user"
        );
        assert_eq!(
            messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["sys", "second question", "second answer"],
        );
    }

    #[test]
    fn compacted_history_resumes_at_a_user_turn() {
        // A tool-using exchange followed by a plain one, compacted hard enough
        // that only part can survive.
        let mut messages = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user("q1".to_string()),
            ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
            }]),
            ChatMessage::tool_result("c1".to_string(), "read".to_string(), "y".repeat(4000)),
            ChatMessage::assistant("a1".to_string()),
            ChatMessage::user("q2".to_string()),
            ChatMessage::assistant("a2".to_string()),
        ];

        compact_messages(&mut messages, 100);

        // Whatever survives, the first non-system message is a user turn — no
        // orphaned assistant reply, no orphaned tool call.
        let first = messages
            .iter()
            .find(|m| m.role != ChatRole::System)
            .expect("some history survives");
        assert_eq!(
            first.role,
            ChatRole::User,
            "history must resume at a user turn, got {:?}",
            first.role
        );
    }

    #[test]
    fn compact_messages_never_orphans_a_tool_result() {
        // An assistant tool-call message plus its results, then a fresh exchange.
        let mut messages = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user("x".repeat(4000)),
            ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
            }]),
            ChatMessage::tool_result("c1".to_string(), "read".to_string(), "y".repeat(4000)),
            ChatMessage::user("recent".to_string()),
            ChatMessage::assistant("reply".to_string()),
        ];

        compact_messages(&mut messages, 100);

        // Whatever survived, no `tool` message may lead the non-system history
        // or follow anything but the assistant call that requested it.
        for (i, m) in messages.iter().enumerate() {
            if m.role == ChatRole::Tool {
                let prev = messages.get(i - 1).expect("a tool result cannot lead");
                assert!(
                    prev.role == ChatRole::Tool || prev.tool_calls.is_some(),
                    "orphaned tool result at {i}: previous is {:?}",
                    prev.role
                );
            }
        }
    }

    #[test]
    fn compact_messages_stops_when_only_the_system_prompt_is_left() {
        let mut messages = vec![
            ChatMessage::system("x".repeat(4000)),
            ChatMessage::user("y".repeat(4000)),
        ];
        // Unsatisfiable target: the system prompt alone busts it.
        let dropped = compact_messages(&mut messages, 10);
        assert_eq!(dropped, 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::System);
    }
}

#[cfg(test)]
mod active_turn_tests {
    use super::*;
    use crate::llm::ToolCallInfo;

    fn asst_call(id: &str) -> ChatMessage {
        let mut m = ChatMessage::assistant(String::new());
        m.tool_calls = Some(vec![ToolCallInfo {
            id: id.to_string(),
            name: "Read".to_string(),
            arguments: serde_json::json!({}),
        }]);
        m
    }

    fn tool_out(id: &str, bulk: usize) -> ChatMessage {
        ChatMessage::tool_result(id.to_string(), "Read".to_string(), "x".repeat(bulk))
    }

    /// The failure this exists for: a **fresh thread**, where every message
    /// belongs to the running turn, so there are no prior exchanges to give up.
    ///
    /// `compact_messages` drops a message and everything up to the next user
    /// message — with the task at the front and nothing after it, that is the
    /// whole turn, prompt included. Here the prompt survives and the oldest tool
    /// output goes instead.
    #[test]
    fn a_first_turn_keeps_its_task_and_loses_its_oldest_tool_output() {
        let mut messages = vec![
            ChatMessage::user("why did CI fail?".to_string()),
            asst_call("call_1"),
            tool_out("call_1", 4000),
            asst_call("call_2"),
            tool_out("call_2", 4000),
        ];

        let dropped = compact_active_turn(&mut messages, 500);

        assert!(dropped > 0, "something had to go");
        assert_eq!(
            messages[0].content, "why did CI fail?",
            "the task must survive — the model is still working on it"
        );
        assert!(
            !messages
                .iter()
                .any(|m| m.content.len() == 4000 && m.tool_call_id.as_deref() == Some("call_1")),
            "the oldest tool result is the first thing to lose"
        );
    }

    /// What `compact_messages` would have done to the same history, and why it
    /// could not be reused: it empties the turn.
    #[test]
    fn the_between_turns_rule_would_have_dropped_the_task() {
        let mut messages = vec![
            ChatMessage::user("why did CI fail?".to_string()),
            asst_call("call_1"),
            tool_out("call_1", 4000),
        ];

        compact_messages(&mut messages, 500);

        assert!(
            !messages.iter().any(|m| m.role == ChatRole::User),
            "this is the behavior compact_active_turn exists to avoid"
        );
    }

    /// A `Tool` result whose assistant tool-call is gone is rejected outright by
    /// providers, so phase 2 drops the pair together — the same rule phase 1
    /// follows for whole exchanges.
    #[test]
    fn a_tool_result_never_outlives_the_call_that_produced_it() {
        let mut messages = vec![
            ChatMessage::user("task".to_string()),
            asst_call("call_1"),
            tool_out("call_1", 6000),
            asst_call("call_2"),
            tool_out("call_2", 100),
        ];

        compact_active_turn(&mut messages, 200);

        for m in &messages {
            if let Some(id) = &m.tool_call_id {
                assert!(
                    messages.iter().any(|c| c
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| calls.iter().any(|call| &call.id == id))),
                    "orphaned tool result {id}"
                );
            }
        }
    }

    /// Prior turns are still the cheapest thing to lose, so they go first and
    /// the running turn is left whole when they are enough.
    #[test]
    fn prior_turns_go_before_the_running_one() {
        let mut messages = vec![
            ChatMessage::user("an old question".to_string()),
            ChatMessage::assistant("x".repeat(4000)),
            ChatMessage::user("the current task".to_string()),
            asst_call("call_1"),
            tool_out("call_1", 100),
        ];

        compact_active_turn(&mut messages, 200);

        assert_eq!(messages[0].content, "the current task");
        assert!(
            messages
                .iter()
                .any(|m| m.tool_call_id.as_deref() == Some("call_1")),
            "the running turn's own work survives while older turns can be given up"
        );
    }

    /// A prompt bigger than the target on its own ends compaction still over
    /// budget rather than with an empty history. The caller learns that from the
    /// model call; a history with no task in it would be a worse answer than a
    /// long one.
    #[test]
    fn a_task_larger_than_the_target_is_still_never_dropped() {
        let mut messages = vec![ChatMessage::user("x".repeat(8000))];

        compact_active_turn(&mut messages, 100);

        assert_eq!(messages.len(), 1, "the task is not ours to drop");
    }

    /// The system prompt is never ours to drop either, in both phases.
    #[test]
    fn the_system_prompt_survives_both_phases() {
        let mut messages = vec![
            ChatMessage::system("you are an agent".to_string()),
            ChatMessage::user("task".to_string()),
            asst_call("call_1"),
            tool_out("call_1", 8000),
        ];

        compact_active_turn(&mut messages, 100);

        assert_eq!(messages[0].role, ChatRole::System);
    }
}
