//! Conversation memory manager.
//!
//! Copied verbatim from voice-agent/crates/lib/src/memory.rs.

use crate::llm::{ChatMessage, ChatRole};

const BACKCHANNEL_MARKER: &str = "⟂";

#[derive(Debug, Clone)]
struct MessageEntry {
    message: ChatMessage,
    is_backchannel: bool,
}

/// Manages conversation history with token-based compaction.
#[derive(Debug, Clone)]
pub struct ConversationMemory {
    messages: Vec<MessageEntry>,
    max_messages: usize,
}

impl ConversationMemory {
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    pub fn with_capacity(max_messages: usize) -> Self {
        Self { messages: Vec::new(), max_messages }
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(MessageEntry { message, is_backchannel: false });
        self.trim_messages();
    }

    pub fn add_backchannel(&mut self) {
        self.messages.push(MessageEntry {
            message: ChatMessage::assistant(BACKCHANNEL_MARKER.to_string()),
            is_backchannel: true,
        });
        self.trim_messages();
    }

    fn trim_messages(&mut self) {
        if self.messages.len() > self.max_messages {
            let system_messages: Vec<_> = self.messages.iter()
                .filter(|e| e.message.role == ChatRole::System).cloned().collect();
            let non_system_to_keep = self.max_messages.saturating_sub(system_messages.len());
            let all_non_system: Vec<_> = self.messages.iter()
                .filter(|e| e.message.role != ChatRole::System).cloned().collect();
            let total = all_non_system.len();
            let non_system: Vec<_> = if total > non_system_to_keep {
                all_non_system.into_iter().skip(total - non_system_to_keep).collect()
            } else {
                all_non_system
            };
            self.messages = system_messages;
            self.messages.extend(non_system);
        }
    }

    pub fn get_messages(&self) -> Vec<ChatMessage> {
        self.messages.iter().filter(|e| !e.is_backchannel).map(|e| e.message.clone()).collect()
    }

    pub fn get_last_messages(&self, n: usize) -> Vec<ChatMessage> {
        self.messages.iter()
            .filter(|e| !e.is_backchannel)
            .map(|e| e.message.clone())
            .rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
    }

    /// Estimate total token count (~4 chars/token + 10 per message overhead).
    pub fn estimate_tokens(&self) -> usize {
        self.messages.iter()
            .filter(|e| !e.is_backchannel)
            .map(|e| e.message.content.len() / 4 + 10)
            .sum()
    }

    /// Drop oldest non-system messages until estimated tokens < target.
    pub fn compact(&mut self, target_tokens: usize) -> usize {
        let mut dropped = 0;
        while self.estimate_tokens() > target_tokens {
            let pos = self.messages.iter().position(|e| {
                !e.is_backchannel && e.message.role != ChatRole::System
            });
            match pos {
                Some(i) => { self.messages.remove(i); dropped += 1; }
                None => break,
            }
        }
        dropped
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn len(&self) -> usize {
        self.messages.iter().filter(|e| !e.is_backchannel).count()
    }

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

    #[test]
    fn test_add_and_get() {
        let mut m = ConversationMemory::new();
        m.add_message(ChatMessage::user("Hello".to_string()));
        assert_eq!(m.len(), 1);
        assert_eq!(m.get_messages()[0].content, "Hello");
    }

    #[test]
    fn test_compact_preserves_system_and_newest() {
        let mut m = ConversationMemory::new();
        m.add_message(ChatMessage::system("System".to_string()));
        for i in 0..10 {
            m.add_message(ChatMessage::user(format!("msg{} {}", i, "x".repeat(380))));
        }
        let dropped = m.compact(500);
        assert!(dropped > 0);
        let msgs = m.get_messages();
        assert_eq!(msgs[0].role, ChatRole::System);
        assert!(msgs.last().unwrap().content.starts_with("msg9"));
    }

    #[test]
    fn test_max_messages_trims_oldest_non_system() {
        let mut m = ConversationMemory::with_capacity(3);
        m.add_message(ChatMessage::system("sys".to_string()));
        for i in 0..5 {
            m.add_message(ChatMessage::user(format!("msg{}", i)));
        }
        assert_eq!(m.messages.len(), 3);
        let msgs = m.get_messages();
        assert_eq!(msgs[0].role, ChatRole::System);
        assert_eq!(msgs[1].content, "msg3");
        assert_eq!(msgs[2].content, "msg4");
    }

    #[test]
    fn test_clear() {
        let mut m = ConversationMemory::new();
        m.add_message(ChatMessage::user("hi".to_string()));
        m.clear();
        assert!(m.is_empty());
    }
}
