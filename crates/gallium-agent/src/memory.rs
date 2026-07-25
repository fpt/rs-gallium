use crate::llm::{ChatMessage, ChatRole};

/// Backchannel marker in message history
const BACKCHANNEL_MARKER: &str = "⟂";

// ============================================================================
// Compaction policy
//
// Shared by every frontend so they cannot drift apart: `Agent` (which holds a
// `ConversationMemory`) and the app-server (which holds a bare
// `Vec<ChatMessage>` per thread) both trigger on `compaction_target` and size
// history with `estimate_message_tokens`.
// ============================================================================

/// Context window assumed when nothing configures one.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;

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
/// completes. The native candle backend never reports usage at all, so
/// `estimated_tokens` (our own count of the history) is taken as a floor:
/// without it compaction would silently never fire on that engine, which is the
/// same failure this policy exists to prevent.
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

/// Drop oldest non-system messages from a plain history until the estimate is
/// under `target_tokens`. Returns the number of messages dropped.
///
/// Unlike [`ConversationMemory::compact`], this runs over a history that has
/// been through the ReAct loop, so it contains assistant tool-call messages and
/// their `Tool` results. Those are dropped as a unit: a `tool` message is only
/// valid immediately after the assistant message that requested it, and
/// providers reject an orphan.
pub fn compact_messages(messages: &mut Vec<ChatMessage>, target_tokens: usize) -> usize {
    let mut dropped = 0;
    while estimate_messages_tokens(messages) > target_tokens {
        let Some(i) = messages.iter().position(|m| m.role != ChatRole::System) else {
            break; // Only the system prompt is left; it is not ours to drop.
        };
        messages.remove(i);
        dropped += 1;
        // Take the tool results that belonged to it, so none is left orphaned.
        while messages.get(i).is_some_and(|m| m.role == ChatRole::Tool) {
            messages.remove(i);
            dropped += 1;
        }
    }
    dropped
}

/// Message with backchannel flag
#[derive(Debug, Clone)]
struct MessageEntry {
    message: ChatMessage,
    is_backchannel: bool,
}

/// Conversation memory manager.
#[derive(Debug, Clone)]
pub struct ConversationMemory {
    messages: Vec<MessageEntry>,
    max_messages: usize,
}

impl ConversationMemory {
    /// Create a new conversation memory
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    /// Create a new conversation memory with specified capacity
    pub fn with_capacity(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages,
        }
    }

    /// Add a regular message to the conversation
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(MessageEntry {
            message,
            is_backchannel: false,
        });
        self.trim_messages();
    }

    /// Add a backchannel marker to the conversation
    /// This is for tempo tracking only — doesn't pollute context
    pub fn add_backchannel(&mut self) {
        self.messages.push(MessageEntry {
            message: ChatMessage::assistant(BACKCHANNEL_MARKER.to_string()),
            is_backchannel: true,
        });
        self.trim_messages();
    }

    /// Trim messages to max capacity
    fn trim_messages(&mut self) {
        if self.messages.len() > self.max_messages {
            // Keep system messages at the beginning
            let system_messages: Vec<_> = self
                .messages
                .iter()
                .filter(|e| e.message.role == ChatRole::System)
                .cloned()
                .collect();

            // Calculate how many non-system messages to keep
            let non_system_to_keep = self.max_messages.saturating_sub(system_messages.len());

            // Get all non-system messages
            let all_non_system: Vec<_> = self
                .messages
                .iter()
                .filter(|e| e.message.role != ChatRole::System)
                .cloned()
                .collect();

            // Keep the last N non-system messages
            let total_non_system = all_non_system.len();
            let non_system_messages: Vec<_> = if total_non_system > non_system_to_keep {
                all_non_system
                    .into_iter()
                    .skip(total_non_system - non_system_to_keep)
                    .collect()
            } else {
                all_non_system
            };

            self.messages = system_messages;
            self.messages.extend(non_system_messages);
        }
    }

    /// Get all messages (excluding backchannel markers by default)
    pub fn get_messages(&self) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .filter(|e| !e.is_backchannel)
            .map(|e| e.message.clone())
            .collect()
    }

    /// Get all messages including backchannel markers
    pub fn get_messages_with_backchannels(&self) -> Vec<ChatMessage> {
        self.messages.iter().map(|e| e.message.clone()).collect()
    }

    /// Get the last N messages (excluding backchannel markers)
    pub fn get_last_messages(&self, n: usize) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .filter(|e| !e.is_backchannel)
            .map(|e| e.message.clone())
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Estimate total token count of non-backchannel messages.
    pub fn estimate_tokens(&self) -> usize {
        self.messages
            .iter()
            .filter(|e| !e.is_backchannel)
            .map(|e| estimate_message_tokens(&e.message))
            .sum()
    }

    /// Drop oldest non-system messages until estimated tokens < `target_tokens`.
    /// Returns the number of messages dropped.
    pub fn compact(&mut self, target_tokens: usize) -> usize {
        let mut dropped = 0;
        while self.estimate_tokens() > target_tokens {
            // Find the first non-system, non-backchannel message
            let pos = self
                .messages
                .iter()
                .position(|e| !e.is_backchannel && e.message.role != ChatRole::System);
            match pos {
                Some(i) => {
                    self.messages.remove(i);
                    dropped += 1;
                }
                None => break, // Only system/backchannel messages left
            }
        }
        dropped
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get the number of messages (excluding backchannel markers)
    pub fn len(&self) -> usize {
        self.messages.iter().filter(|e| !e.is_backchannel).count()
    }

    /// Get total number of messages including backchannel markers
    pub fn total_len(&self) -> usize {
        self.messages.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCallInfo;

    #[test]
    fn test_add_message() {
        let mut memory = ConversationMemory::new();
        memory.add_message(ChatMessage::user("Hello".to_string()));
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut memory = ConversationMemory::new();
        memory.add_message(ChatMessage::user("Hello".to_string()));
        memory.clear();
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_compact_drops_oldest_non_system() {
        let mut memory = ConversationMemory::new();
        memory.add_message(ChatMessage::system("System prompt".to_string()));
        // Each 400-char message ≈ 110 estimated tokens (400/4 + 10)
        for i in 0..10 {
            let msg = format!("Message {} {}", i, "x".repeat(380));
            memory.add_message(ChatMessage::user(msg));
        }

        let before = memory.len();
        assert_eq!(before, 11); // 1 system + 10 user

        // Compact to ~500 tokens — should keep system + a few user messages
        let dropped = memory.compact(500);
        assert!(dropped > 0);

        let messages = memory.get_messages();
        // System message must survive
        assert_eq!(messages[0].role, ChatRole::System);
        // Remaining messages should be the newest ones
        let last = &messages[messages.len() - 1];
        assert!(last.content.starts_with("Message 9"));
    }

    #[test]
    fn test_compact_preserves_all_when_under_target() {
        let mut memory = ConversationMemory::new();
        memory.add_message(ChatMessage::user("short".to_string()));
        let dropped = memory.compact(10000);
        assert_eq!(dropped, 0);
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_estimate_tokens() {
        let mut memory = ConversationMemory::new();
        memory.add_message(ChatMessage::user("x".repeat(400).to_string()));
        // 400 chars / 4 + 10 overhead = 110
        assert_eq!(memory.estimate_tokens(), 110);
    }

    // ------------------------------------------------------------------
    // Shared compaction policy
    // ------------------------------------------------------------------

    #[test]
    fn compaction_target_holds_off_until_the_window_is_nearly_full() {
        // 89% of the window: not yet.
        assert_eq!(compaction_target(890, 0, 1000), None);
        // 90% is the trigger, and the target is half the window.
        assert_eq!(compaction_target(900, 0, 1000), Some(500));
        assert_eq!(compaction_target(1200, 0, 1000), Some(500));
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

    #[test]
    fn test_max_messages() {
        let mut memory = ConversationMemory::with_capacity(3);

        // Add system message
        memory.add_message(ChatMessage::system("System prompt".to_string()));

        // Add more than capacity
        for i in 0..5 {
            memory.add_message(ChatMessage::user(format!("Message {}", i)));
        }

        // After adding 6 messages total (1 system + 5 user) with capacity 3:
        // - System messages are kept
        // - Last 2 non-system messages are kept (capacity - system_count)
        // - Total: 1 system + 2 user = 3 messages
        assert_eq!(memory.total_len(), 3);
        assert_eq!(memory.len(), 3); // All are non-backchannel

        let messages = memory.get_messages();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].content, "Message 3"); // Second-to-last
        assert_eq!(messages[2].content, "Message 4"); // Last
    }
}
