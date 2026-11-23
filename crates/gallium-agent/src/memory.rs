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
///
/// The target this returns is spent by [`compact_messages`] and
/// [`compact_active_turn`] against `estimate_messages_tokens`, the same crude
/// per-message estimate as `estimated_tokens` above — so it has to be
/// expressed in that estimate's units, not the real ones `last_input_tokens`
/// is in. Those units diverge by more than rounding error: the estimate covers
/// only what is in `messages`, while the real prompt also carries the tool
/// catalog (rendered separately, never a `ChatMessage`) and whatever a chat
/// template adds — klein's dynamic-tools catalog alone runs to thousands of
/// tokens per CLAUDE.md's "Deferred tools" note, dwarfing a short
/// conversation's own message text. Left unadjusted, a real prompt that is
/// mostly tool schema always looks "under target" to the estimate, even while
/// genuinely over the window — which is what let a real session grow past
/// `maxCtx` with compaction never dropping anything (no message content was
/// ever *the* problem, so the estimate never noticed there was a problem at
/// all). So the target is shifted down by `overhead`, the gap this same call
/// already reveals between the real total and the estimate of the same
/// history, before it is handed to a function that only sees the estimate.
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
    if observed < threshold {
        return None;
    }
    let overhead = last_input_tokens.saturating_sub(estimated_tokens as u64);
    let real_target = (context_window as f64 * COMPACTION_TARGET) as u64;
    Some(real_target.saturating_sub(overhead) as usize)
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
        // 90% is the trigger. `estimated_tokens: 0` makes every one of these
        // 900/1200 real tokens "overhead" the estimate can't see, so the
        // target collapses to 0 rather than the naive half-window — see
        // `compaction_target_discounts_overhead_the_estimate_cannot_see`.
        assert_eq!(compaction_target(900, 0, 1000), Some(0));
        assert_eq!(compaction_target(1200, 0, 1000), Some(0));
    }

    /// The bug a real klein session hit: a large dynamic-tools catalog (never
    /// a `ChatMessage`, so invisible to `estimate_messages_tokens`) made the
    /// real prompt blow past the window while the message-only estimate
    /// stayed low — so `compact_active_turn`'s own `estimate_messages_tokens`
    /// check never saw a reason to drop anything, and the turn died on a
    /// context the un-adjusted target believed it had already fit into.
    #[test]
    fn compaction_target_discounts_overhead_the_estimate_cannot_see() {
        // 9000 real tokens, but the messages alone only estimate to 3000 — a
        // 6000-token tool catalog the estimate can't see is the rest.
        let target = compaction_target(9000, 3000, 10000).unwrap();
        // Naive: window(10000) * 0.5 = 5000. That's still over the estimate's
        // own 3000, so `compact_active_turn` would (wrongly) call the turn
        // already compacted enough. Corrected: 5000 - 6000 overhead,
        // saturating at 0 — force it to drop until the estimate is as small
        // as it can go, since the overhead alone already exceeds the budget.
        assert_eq!(target, 0);
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

/// Property tests for the compaction policy: random well-formed histories and
/// budgets, checked against the invariants the doc comments above state.
/// Numbered so a failure names the rule it broke — `R` resolve_context_window,
/// `T` compaction_target, `C` compact_messages, `A` compact_active_turn.
#[cfg(test)]
mod pbt {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::*;
    use crate::llm::ToolCallInfo;

    /// A window as configured, reported, or fallen back to: zero (the disable
    /// sentinel, or a model that said nothing), the sizes real models report,
    /// and the edges of the type.
    fn window() -> impl Strategy<Value = u32> {
        prop_oneof![
            1 => Just(0u32),
            1 => Just(1u32),
            3 => 1u32..200_000,
            1 => Just(u32::MAX),
            1 => any::<u32>(),
        ]
    }

    /// `(last_input_tokens, estimated_tokens, context_window)`, with the token
    /// counts drawn around the window so both sides of the 90% trigger and of
    /// "the estimate under-counts the real prompt" (tool-catalog overhead,
    /// f26a37f) come up often.
    fn usage() -> impl Strategy<Value = (u64, usize, u32)> {
        window().prop_flat_map(|w| {
            let w64 = u64::from(w);
            let around = move || {
                prop_oneof![
                    1 => Just(0u64),
                    1 => Just(w64 * 9 / 10),
                    1 => Just((w64 * 9 / 10).saturating_sub(1)),
                    4 => 0..=w64.saturating_mul(2).max(1),
                ]
            };
            (around(), around().prop_map(|e| e as usize), Just(w))
        })
    }

    /// How long a message body is — the only thing `estimate_message_tokens`
    /// reads. Mostly short, sometimes a file dump.
    fn body_len() -> impl Strategy<Value = usize> {
        prop_oneof![
            3 => 0usize..40,
            2 => 40usize..400,
            1 => 400usize..4000,
        ]
    }

    /// What follows a user message inside one turn, in the order `react.rs`
    /// appends it.
    #[derive(Debug, Clone)]
    enum Step {
        /// One assistant message making these calls, then one result each.
        Tools(Vec<usize>),
        /// A `turn/steer` message: a user message landing mid-turn.
        Steer(usize),
    }

    fn step() -> impl Strategy<Value = Step> {
        prop_oneof![
            5 => prop::collection::vec(body_len(), 1..=3).prop_map(Step::Tools),
            1 => body_len().prop_map(Step::Steer),
        ]
    }

    #[derive(Debug, Clone)]
    struct Turn {
        prompt: usize,
        steps: Vec<Step>,
        /// The closing text answer; `None` is a turn still running.
        answer: Option<usize>,
        /// A system message after this turn — never ours to drop, and a stop
        /// for `compact_messages`'s exchange scan.
        system_after: bool,
    }

    prop_compose! {
        fn turn()(
            prompt in body_len(),
            steps in prop::collection::vec(step(), 0..=4),
            answer in prop::option::weighted(0.7, body_len()),
            system_after in prop::bool::weighted(0.1),
        ) -> Turn {
            Turn { prompt, steps, answer, system_after }
        }
    }

    /// A history as the ReAct loop builds it: an optional system prompt, then
    /// turns. Every body is prefixed with a unique tag so a message can be
    /// followed through compaction by its content.
    fn history() -> impl Strategy<Value = Vec<ChatMessage>> {
        (any::<bool>(), prop::collection::vec(turn(), 1..=5)).prop_map(|(system, turns)| {
            let mut n = 0usize;
            let mut text = |len: usize| {
                n += 1;
                format!("#{n}:{}", "x".repeat(len))
            };
            let mut out = Vec::new();
            if system {
                out.push(ChatMessage::system(text(20)));
            }
            for t in turns {
                out.push(ChatMessage::user(text(t.prompt)));
                for step in t.steps {
                    match step {
                        Step::Tools(results) => {
                            let ids: Vec<String> = results.iter().map(|_| text(0)).collect();
                            out.push(ChatMessage::assistant_tool_calls(
                                ids.iter()
                                    .map(|id| ToolCallInfo {
                                        id: id.clone(),
                                        name: "Read".to_string(),
                                        arguments: serde_json::json!({}),
                                    })
                                    .collect(),
                            ));
                            for (id, len) in ids.into_iter().zip(results) {
                                out.push(ChatMessage::tool_result(
                                    id,
                                    "Read".to_string(),
                                    text(len),
                                ));
                            }
                        }
                        Step::Steer(len) => out.push(ChatMessage::user(text(len))),
                    }
                }
                if let Some(len) = t.answer {
                    out.push(ChatMessage::assistant(text(len)));
                }
                if t.system_after {
                    out.push(ChatMessage::system(text(10)));
                }
            }
            out
        })
    }

    /// A history and a budget for it: none, exactly what it already costs (the
    /// `>` versus `>=` edge), one under, or a fraction.
    fn history_and_target() -> impl Strategy<Value = (Vec<ChatMessage>, usize)> {
        (history(), 0u8..6, 0.0f64..1.2).prop_map(|(h, pick, frac)| {
            let est = estimate_messages_tokens(&h);
            let target = match pick {
                0 => 0,
                1 => est,
                2 => est.saturating_sub(1),
                _ => (est as f64 * frac) as usize,
            };
            (h, target)
        })
    }

    /// `after` is `before` with messages removed and none added, moved or edited.
    fn is_subsequence(after: &[ChatMessage], before: &[ChatMessage]) -> bool {
        let mut rest = before.iter();
        after
            .iter()
            .all(|a| rest.any(|b| b.role == a.role && b.content == a.content))
    }

    /// The pairing providers insist on: every `Tool` result sits in the run of
    /// results directly after the assistant message that made its call, and
    /// every call keeps its result.
    fn pairing_violation(messages: &[ChatMessage]) -> Option<String> {
        let mut open: HashSet<&str> = HashSet::new();
        for (i, m) in messages.iter().enumerate() {
            if m.role == ChatRole::Tool {
                let id = m.tool_call_id.as_deref().unwrap_or("");
                if !open.remove(id) {
                    return Some(format!("message {i}: result {id} has no call before it"));
                }
                continue;
            }
            if let Some(id) = open.iter().next() {
                return Some(format!("message {i}: call {id} lost its result"));
            }
            if let Some(calls) = &m.tool_calls {
                open = calls.iter().map(|c| c.id.as_str()).collect();
            }
        }
        open.iter()
            .next()
            .map(|id| format!("call {id} lost its result"))
    }

    fn systems(messages: &[ChatMessage]) -> Vec<&str> {
        messages
            .iter()
            .filter(|m| m.role == ChatRole::System)
            .map(|m| m.content.as_str())
            .collect()
    }

    fn newest_user(messages: &[ChatMessage]) -> Option<&str> {
        messages
            .iter()
            .rev()
            .find(|m| m.role == ChatRole::User)
            .map(|m| m.content.as_str())
    }

    /// **Pins current behaviour, not a decided rule.** Found by A1: phase 2
    /// drops whatever follows the task, a system message included, though
    /// phase 1 and `compact_messages` never drop one and
    /// `runtime::tests::a_successful_turn_that_compacts_mid_turn_still_lifts_the_right_message`
    /// relies on "`compact_active_turn` never drops system messages". Not
    /// reachable through `run_turn` today — the skill catalog goes before the
    /// first non-system message — so whether phase 2 should stop at a system
    /// message is open.
    #[test]
    fn a_system_message_after_the_task_is_dropped_in_phase_2() {
        let mut messages = vec![
            ChatMessage::user("task".to_string()),
            ChatMessage::system("late system".to_string()),
        ];
        assert_eq!(compact_active_turn(&mut messages, 0), 1);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::User);
    }

    proptest! {
        /// R1–R4: the doc comment's rules for settling the window. `known` is
        /// never `Some(0)` and, when present, is `effective`; a configured value
        /// wins (and zero there disables compaction without being a window); a
        /// reported zero is the same as nothing reported.
        #[test]
        fn resolve_context_window_follows_its_rules(
            configured in prop::option::of(window()),
            reported in prop::option::of(window()),
            fallback in window(),
        ) {
            let w = resolve_context_window(configured, reported, fallback);
            prop_assert_ne!(w.known, Some(0), "R1");
            if let Some(k) = w.known {
                prop_assert_eq!(k, w.effective, "R2");
            }
            match (configured, reported) {
                (Some(c), _) => {
                    prop_assert_eq!(w.effective, c, "R3");
                    prop_assert_eq!(w.known, (c > 0).then_some(c), "R3");
                }
                (None, Some(r)) if r > 0 => {
                    prop_assert_eq!((w.effective, w.known), (r, Some(r)), "R4");
                }
                (None, _) => {
                    prop_assert_eq!((w.effective, w.known), (fallback, None), "R4");
                }
            }
        }

        /// T1: compaction fires exactly when the window is on and the larger of
        /// the reported and estimated prompt has reached 90% of it.
        /// T2: the target is in the estimate's units — a history the estimate
        /// puts at the target, plus the overhead the estimate cannot see, fits
        /// half the window (unless the overhead alone does not, and the target
        /// is zero). The invariant f26a37f restored.
        /// T3: the target never exceeds half the window.
        #[test]
        fn compaction_target_fires_at_90_and_aims_at_half_in_estimate_units(
            (last, est, w) in usage(),
        ) {
            let w64 = u64::from(w);
            let fired = w > 0 && last.max(est as u64) >= w64 * 9 / 10;
            let target = compaction_target(last, est, w);
            prop_assert_eq!(target.is_some(), fired, "T1");
            if let Some(t) = target {
                let half = w64 / 2;
                let overhead = last.saturating_sub(est as u64);
                prop_assert!(t == 0 || t as u64 + overhead <= half,
                    "T2: target {t} + overhead {overhead} > half window {half}");
                prop_assert!(t as u64 <= half, "T3");
            }
        }

        /// T4: more overhead never raises the target.
        #[test]
        fn compaction_target_shrinks_as_overhead_grows(
            (last, est, w) in usage(),
            more in 0u64..100_000,
        ) {
            if let (Some(a), Some(b)) = (
                compaction_target(last, est, w),
                compaction_target(last.saturating_add(more), est, w),
            ) {
                prop_assert!(b <= a, "T4");
            }
        }

        /// C1: compaction only removes — what is left is the input in order with
        ///     messages taken out, every system message among them, and the
        ///     return value is how many went.
        /// C2: no tool result is separated from its call, either way round.
        /// C3: the retained conversation resumes at a user turn.
        /// C4: it stops under target, or when only system messages are left.
        /// C5: a history already within budget is left alone.
        #[test]
        fn compact_messages_drops_whole_exchanges_only((before, target) in history_and_target()) {
            let mut after = before.clone();
            let dropped = compact_messages(&mut after, target);
            prop_assert!(is_subsequence(&after, &before), "C1");
            prop_assert_eq!(systems(&after), systems(&before), "C1");
            prop_assert_eq!(dropped, before.len() - after.len(), "C1");
            prop_assert_eq!(pairing_violation(&after), None, "C2");
            if let Some(first) = after.iter().find(|m| m.role != ChatRole::System) {
                prop_assert_eq!(&first.role, &ChatRole::User, "C3");
            }
            prop_assert!(
                estimate_messages_tokens(&after) <= target
                    || after.iter().all(|m| m.role == ChatRole::System),
                "C4"
            );
            if estimate_messages_tokens(&before) <= target {
                prop_assert_eq!(dropped, 0, "C5");
            }
        }

        /// A1: as C1 — only removal, count returned, and every system message
        ///     before the task kept. (One *after* the task is not: see
        ///     `a_system_message_after_the_task_is_dropped_in_phase_2`.)
        /// A2: as C2 — calls and results stay paired.
        /// A3: the newest user message — the task in hand — survives, and is
        ///     still the newest.
        /// A4: it stops under target, or when nothing but system messages and
        ///     the task remain before the task and nothing remains after it.
        /// A5: a history already within budget is left alone.
        /// A6: the running turn's tool exchanges go only once every earlier
        ///     exchange already has.
        #[test]
        fn compact_active_turn_keeps_the_task((before, target) in history_and_target()) {
            let mut after = before.clone();
            let dropped = compact_active_turn(&mut after, target);
            prop_assert!(is_subsequence(&after, &before), "A1");
            prop_assert_eq!(dropped, before.len() - after.len(), "A1");
            prop_assert_eq!(pairing_violation(&after), None, "A2");

            let task = newest_user(&before).expect("every generated history has a user turn");
            prop_assert_eq!(newest_user(&after), Some(task), "A3");

            let pin = after.iter().position(|m| m.content == task).unwrap();
            let pin_before = before.iter().position(|m| m.content == task).unwrap();
            prop_assert_eq!(systems(&after[..pin]), systems(&before[..pin_before]), "A1");
            let settled = after[..pin].iter().all(|m| m.role == ChatRole::System);
            prop_assert!(
                estimate_messages_tokens(&after) <= target || (settled && pin + 1 == after.len()),
                "A4"
            );
            if estimate_messages_tokens(&before) <= target {
                prop_assert_eq!(dropped, 0, "A5");
            }
            let lost_after_pin = before.len() - pin_before != after.len() - pin;
            if lost_after_pin {
                prop_assert!(settled, "A6: dropped from the running turn while earlier exchanges remained");
            }
        }
    }
}
