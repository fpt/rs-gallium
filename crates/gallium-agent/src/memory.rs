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
    /// What compaction measures against. Always a number.
    pub effective: u32,
    /// The same number, when it came from somewhere real. `None` means the
    /// fallback was used and no gauge should be drawn.
    pub known: Option<u32>,
}

/// Settle the window from what the user said and what the model reports.
///
/// `configured` wins: someone who sets `contextWindow` is describing their own
/// setup — a llama.cpp server started with a smaller `n_ctx`, or a deliberately
/// earlier compaction — and knows something the model file does not. Failing
/// that, `reported` is the model's own metadata. Failing both, the fallback
/// keeps compaction working and the gauge dark.
pub fn resolve_context_window(
    configured: Option<u32>,
    reported: Option<u32>,
    fallback: u32,
) -> ContextWindow {
    let known = configured.or(reported);
    ContextWindow {
        effective: known.unwrap_or(fallback),
        known,
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
