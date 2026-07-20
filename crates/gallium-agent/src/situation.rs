//! Volatile situation messages — windowed store for system events.
//!
//! Events from Claude Code hooks, session JSONL, and MCP tools are pushed here
//! as one-line records. Messages auto-expire after a configurable TTL (default 10min).
//! The ReAct loop reads them via the `read_situation_messages` tool.
//!
//! Each message carries a `session_id` (working directory path) so multiple
//! Claude Code instances can be distinguished. Filtering supports partial,
//! case-insensitive matching on any part of the path or its basename.

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use crate::tool::ToolHandler;
use crate::AgentError;

/// A single volatile situation message.
#[derive(Debug, Clone)]
pub struct SituationMessage {
    pub text: String,
    pub source: String,
    pub session_id: String,
    pub timestamp: SystemTime,
    created: Instant,
}

/// Volatile message store with time-based expiry.
///
/// Messages auto-expire after `ttl`. No max-count compaction.
/// Thread-safe via internal `Mutex`.
pub struct SituationMessages {
    messages: Mutex<Vec<SituationMessage>>,
    ttl: Duration,
}

impl SituationMessages {
    pub fn new(ttl: Duration) -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            ttl,
        }
    }

    /// Push a new message and prune expired ones.
    pub fn push(&self, text: String, source: String, session_id: String) {
        let now = Instant::now();
        let mut msgs = self.messages.lock().unwrap();
        msgs.retain(|m| now.duration_since(m.created) < self.ttl);
        msgs.push(SituationMessage {
            text,
            source,
            session_id,
            timestamp: SystemTime::now(),
            created: now,
        });
    }

    /// Return all non-expired messages (oldest first).
    pub fn read_all(&self) -> Vec<SituationMessage> {
        let now = Instant::now();
        let mut msgs = self.messages.lock().unwrap();
        msgs.retain(|m| now.duration_since(m.created) < self.ttl);
        msgs.clone()
    }

    /// Return non-expired messages whose session_id contains `query`
    /// (case-insensitive, matches against both the full path and its basename).
    pub fn read_by_session(&self, query: &str) -> Vec<SituationMessage> {
        let now = Instant::now();
        let mut msgs = self.messages.lock().unwrap();
        msgs.retain(|m| now.duration_since(m.created) < self.ttl);
        let q = query.to_lowercase();
        msgs.iter()
            .filter(|m| {
                let id_lower = m.session_id.to_lowercase();
                let basename = session_basename(&id_lower);
                id_lower.contains(&q) || basename.contains(&q)
            })
            .cloned()
            .collect()
    }

    /// Count of non-expired messages.
    pub fn count(&self) -> usize {
        let now = Instant::now();
        let msgs = self.messages.lock().unwrap();
        msgs.iter()
            .filter(|m| now.duration_since(m.created) < self.ttl)
            .count()
    }

    /// Distinct session IDs among non-expired messages, ordered by most recent first.
    pub fn session_ids(&self) -> Vec<String> {
        let now = Instant::now();
        let msgs = self.messages.lock().unwrap();
        // Walk backwards (newest first) to get recency order.
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::new();
        for m in msgs.iter().rev().filter(|m| now.duration_since(m.created) < self.ttl) {
            if seen.insert(m.session_id.clone()) {
                ids.push(m.session_id.clone());
            }
        }
        ids
    }

    /// Timestamp of the most recent non-expired message.
    pub fn last_timestamp(&self) -> Option<SystemTime> {
        let now = Instant::now();
        let msgs = self.messages.lock().unwrap();
        msgs.iter()
            .rev()
            .find(|m| now.duration_since(m.created) < self.ttl)
            .map(|m| m.timestamp)
    }
}

impl Default for SituationMessages {
    fn default() -> Self {
        Self::new(Duration::from_secs(600)) // 10 minutes
    }
}

/// Extract the last path component for display.
fn session_basename(session_id: &str) -> &str {
    std::path::Path::new(session_id)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(session_id)
}

// ============================================================================
// ReadSituationMessagesTool
// ============================================================================

/// Tool that lets the LLM read current situation messages.
pub struct ReadSituationMessagesTool {
    messages: std::sync::Arc<SituationMessages>,
}

impl ReadSituationMessagesTool {
    pub fn new(messages: std::sync::Arc<SituationMessages>) -> Self {
        Self { messages }
    }

    fn format_time(t: SystemTime) -> String {
        let dur = t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
        let secs = dur.as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{:02}:{:02}:{:02}", h, m, s)
    }

    fn format_messages(msgs: &[SituationMessage], show_session: bool) -> String {
        if msgs.is_empty() {
            return "No recent situation messages.".to_string();
        }
        let mut output = format!("{} situation message(s):\n", msgs.len());
        for msg in msgs {
            let time_str = Self::format_time(msg.timestamp);
            if show_session {
                let basename = session_basename(&msg.session_id);
                output.push_str(&format!(
                    "[{}] ({}) [{}] {}\n",
                    time_str, msg.source, basename, msg.text
                ));
            } else {
                output.push_str(&format!("[{}] ({}) {}\n", time_str, msg.source, msg.text));
            }
        }
        output
    }
}

impl ToolHandler for ReadSituationMessagesTool {
    fn name(&self) -> &str {
        "read_situation_messages"
    }

    fn description(&self) -> &str {
        "Read recent situation messages from Claude Code sessions (10-min window). \
         Each message is prefixed with [Claude Code <project>]. \
         Filter by session name, paginate with offset/limit (default 50)."
    }

    fn dynamic_state(&self) -> Option<String> {
        let count = self.messages.count();
        if count == 0 {
            return None;
        }
        let sessions = self.messages.session_ids(); // newest first
        let last_str = self
            .messages
            .last_timestamp()
            .map(|t| Self::format_time(t))
            .unwrap_or_else(|| "?".to_string());

        let session_info = if sessions.len() <= 1 {
            let label = sessions.first()
                .map(|s| session_basename(s))
                .unwrap_or("?");
            format!(", Claude Code on {}", label)
        } else {
            let names: Vec<&str> = sessions.iter().map(|s| session_basename(s)).collect();
            format!(", Claude Code sessions: {}", names.join(", "))
        };

        Some(format!(
            "{} message{}, last at {}{}",
            count,
            if count == 1 { "" } else { "s" },
            last_str,
            session_info,
        ))
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Filter by partial match on session path or project name (case-insensitive). E.g. \"kessel-cli\", \"go-gennai-cli\", \"claude\"."
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N messages (0-based, default: 0). Messages are oldest-first."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 50)"
                }
            },
            "required": []
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<crate::tool::ToolResult, AgentError> {
        let query = args.get("session").and_then(|v| v.as_str());
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        let all_msgs = if let Some(q) = query {
            self.messages.read_by_session(q)
        } else {
            self.messages.read_all()
        };

        let total = all_msgs.len();
        let start = offset.min(total);
        let end = (start + limit).min(total);
        let msgs = &all_msgs[start..end];

        let sessions = self.messages.session_ids();
        let show_session = sessions.len() > 1 && query.is_none();
        let mut output = Self::format_messages(msgs, show_session);

        if end < total {
            output.push_str(&format!(
                "\n... (showing {}-{} of {}. Use offset={} to see more.)\n",
                start + 1, end, total, end
            ));
        }

        Ok(crate::tool::ToolResult::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_push_and_read() {
        let store = SituationMessages::new(Duration::from_secs(60));
        store.push("[hook] Write: main.rs".into(), "hook".into(), "/project/a".into());
        store.push("[session] assistant: Edit x2".into(), "session".into(), "/project/a".into());
        assert_eq!(store.count(), 2);
        let all = store.read_all();
        assert_eq!(all.len(), 2);
        assert!(all[0].text.contains("main.rs"));
    }

    #[test]
    fn test_ttl_expiry() {
        let store = SituationMessages::new(Duration::from_millis(50));
        store.push("old".into(), "hook".into(), "/p".into());
        thread::sleep(Duration::from_millis(100));
        assert_eq!(store.count(), 0);
        assert!(store.read_all().is_empty());
    }

    #[test]
    fn test_push_prunes_expired() {
        let store = SituationMessages::new(Duration::from_millis(50));
        store.push("old".into(), "hook".into(), "/p".into());
        thread::sleep(Duration::from_millis(100));
        store.push("new".into(), "session".into(), "/p".into());
        assert_eq!(store.count(), 1);
        assert_eq!(store.read_all()[0].text, "new");
    }

    #[test]
    fn test_last_timestamp() {
        let store = SituationMessages::new(Duration::from_secs(60));
        assert!(store.last_timestamp().is_none());
        store.push("first".into(), "hook".into(), "/p".into());
        assert!(store.last_timestamp().is_some());
    }

    #[test]
    fn test_read_by_session_partial_match() {
        let store = SituationMessages::new(Duration::from_secs(60));
        store.push("a1".into(), "hook".into(), "/home/user/kessel-cli".into());
        store.push("b1".into(), "hook".into(), "/home/user/go-gennai-cli".into());
        store.push("a2".into(), "hook".into(), "/home/user/kessel-cli".into());

        // Partial match on basename
        let a_msgs = store.read_by_session("kessel-cli");
        assert_eq!(a_msgs.len(), 2);
        assert_eq!(a_msgs[0].text, "a1");
        assert_eq!(a_msgs[1].text, "a2");

        // Partial match on substring
        let a_msgs = store.read_by_session("kessel");
        assert_eq!(a_msgs.len(), 2);

        // Case-insensitive
        let b_msgs = store.read_by_session("GENNAI");
        assert_eq!(b_msgs.len(), 1);
        assert_eq!(b_msgs[0].text, "b1");

        // No match
        assert!(store.read_by_session("nonexistent").is_empty());
    }

    #[test]
    fn test_session_ids_newest_first() {
        let store = SituationMessages::new(Duration::from_secs(60));
        store.push("a1".into(), "hook".into(), "/project/alpha".into());
        store.push("b1".into(), "hook".into(), "/project/beta".into());
        store.push("a2".into(), "hook".into(), "/project/alpha".into());

        let ids = store.session_ids();
        assert_eq!(ids.len(), 2);
        // alpha has the most recent message, so it comes first
        assert_eq!(ids[0], "/project/alpha");
        assert_eq!(ids[1], "/project/beta");
    }

    #[test]
    fn test_session_ids_order_changes_with_activity() {
        let store = SituationMessages::new(Duration::from_secs(60));
        store.push("a1".into(), "hook".into(), "/project/alpha".into());
        store.push("b1".into(), "hook".into(), "/project/beta".into());

        // beta is newest
        let ids = store.session_ids();
        assert_eq!(ids[0], "/project/beta");

        // now alpha gets a new message
        store.push("a2".into(), "hook".into(), "/project/alpha".into());
        let ids = store.session_ids();
        assert_eq!(ids[0], "/project/alpha");
    }

    #[test]
    fn test_default_ttl_is_10_min() {
        let store = SituationMessages::default();
        assert_eq!(store.ttl, Duration::from_secs(600));
    }

    #[test]
    fn test_tool_call_empty() {
        let store = Arc::new(SituationMessages::default());
        let tool = ReadSituationMessagesTool::new(store);
        let result = tool.call(serde_json::json!({})).unwrap().text;
        assert!(result.contains("No recent"));
    }

    #[test]
    fn test_tool_call_with_messages() {
        let store = Arc::new(SituationMessages::default());
        store.push("[hook] Write: foo.rs".into(), "hook".into(), "/p".into());
        let tool = ReadSituationMessagesTool::new(store);
        let result = tool.call(serde_json::json!({})).unwrap().text;
        assert!(result.contains("1 situation message"));
        assert!(result.contains("foo.rs"));
    }

    #[test]
    fn test_tool_call_filter_by_partial_session() {
        let store = Arc::new(SituationMessages::default());
        store.push("a-event".into(), "hook".into(), "/home/user/kessel-cli".into());
        store.push("b-event".into(), "hook".into(), "/home/user/go-gennai-cli".into());
        let tool = ReadSituationMessagesTool::new(store);

        // Filter by partial match
        let result = tool.call(serde_json::json!({"session": "kessel"})).unwrap().text;
        assert!(result.contains("a-event"));
        assert!(!result.contains("b-event"));

        let result = tool.call(serde_json::json!({"session": "gennai"})).unwrap().text;
        assert!(!result.contains("a-event"));
        assert!(result.contains("b-event"));

        // No filter shows all with session labels
        let result = tool.call(serde_json::json!({})).unwrap().text;
        assert!(result.contains("a-event"));
        assert!(result.contains("b-event"));
        assert!(result.contains("[kessel-cli]"));
        assert!(result.contains("[go-gennai-cli]"));
    }

    #[test]
    fn test_dynamic_state() {
        let store = Arc::new(SituationMessages::default());
        let tool = ReadSituationMessagesTool::new(Arc::clone(&store));
        assert!(tool.dynamic_state().is_none());

        store.push("test".into(), "hook".into(), "/project/myapp".into());
        let state = tool.dynamic_state().unwrap();
        assert!(state.contains("1 message,"));
        assert!(state.contains("Claude Code on myapp"));

        // full_description combines static + dynamic
        let full = crate::tool::full_description(&tool);
        assert!(full.starts_with("Read recent situation messages"));
        assert!(full.contains("[1 message,"));
    }

    #[test]
    fn test_dynamic_state_multi_session_newest_first() {
        let store = Arc::new(SituationMessages::default());
        let tool = ReadSituationMessagesTool::new(Arc::clone(&store));

        store.push("a".into(), "hook".into(), "/project/alpha".into());
        store.push("b".into(), "hook".into(), "/project/beta".into());

        let state = tool.dynamic_state().unwrap();
        assert!(state.contains("2 messages"));
        // beta is newer, so it comes first
        assert!(state.contains("Claude Code sessions: beta, alpha"));
    }
}
