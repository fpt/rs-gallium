//! Session persistence: save and restore conversation memory as newline-delimited JSON.
//!
//! Sessions are stored in `.gallium/sessions/<id>.jsonl` inside the working directory.
//! Each line is a serialized `ChatMessage`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::llm::ChatMessage;
use crate::memory::ConversationMemory;
use crate::AgentError;

/// Resolve the session file path for a given working dir and session ID.
pub fn session_path(working_dir: &Path, session_id: &str) -> PathBuf {
    // Sanitize: keep only alphanumeric, dash, underscore
    let safe: String = session_id.chars().map(|c| {
        if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }
    }).collect();
    working_dir.join(".gallium").join("sessions").join(format!("{}.jsonl", safe))
}

/// Save a conversation memory to disk.
pub fn save(memory: &ConversationMemory, path: &Path) -> Result<(), AgentError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AgentError::InternalError(format!("Cannot create session dir: {}", e))
        })?;
    }
    let mut file = std::fs::File::create(path).map_err(|e| {
        AgentError::InternalError(format!("Cannot create session file {:?}: {}", path, e))
    })?;
    for msg in memory.get_messages() {
        let line = serde_json::to_string(&msg).map_err(|e| {
            AgentError::InternalError(format!("JSON serialize: {}", e))
        })?;
        writeln!(file, "{}", line).map_err(|e| {
            AgentError::InternalError(format!("Write session: {}", e))
        })?;
    }
    tracing::info!("Session saved to {:?} ({} messages)", path, memory.len());
    Ok(())
}

/// Load a conversation memory from disk.
/// Returns an empty memory if the file does not exist.
pub fn load(path: &Path) -> Result<ConversationMemory, AgentError> {
    if !path.exists() {
        return Ok(ConversationMemory::new());
    }
    let file = std::fs::File::open(path).map_err(|e| {
        AgentError::InternalError(format!("Cannot open session file {:?}: {}", path, e))
    })?;
    let mut memory = ConversationMemory::new();
    for (lineno, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| AgentError::InternalError(format!("Read session: {}", e)))?;
        let line = line.trim();
        if line.is_empty() { continue; }
        let msg: ChatMessage = serde_json::from_str(line).map_err(|e| {
            AgentError::ParseError(format!("Session line {}: {}", lineno + 1, e))
        })?;
        memory.add_message(msg);
    }
    tracing::info!("Session loaded from {:?} ({} messages)", path, memory.len());
    Ok(memory)
}

/// Append a single message to a session file.
pub fn append(path: &Path, msg: &ChatMessage) -> Result<(), AgentError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AgentError::InternalError(format!("Cannot create session dir: {}", e))
        })?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)
        .map_err(|e| AgentError::InternalError(format!("Cannot open session file: {}", e)))?;
    let line = serde_json::to_string(msg).map_err(|e| {
        AgentError::InternalError(format!("JSON serialize: {}", e))
    })?;
    writeln!(file, "{}", line).map_err(|e| {
        AgentError::InternalError(format!("Write session: {}", e))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;
    use tempfile::TempDir;

    #[test]
    fn test_session_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.jsonl");

        let mut mem = ConversationMemory::new();
        mem.add_message(ChatMessage::user("hello".to_string()));
        mem.add_message(ChatMessage::assistant("hi".to_string()));

        save(&mem, &path).unwrap();
        let loaded = load(&path).unwrap();
        let msgs = loaded.get_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[1].content, "hi");
    }

    #[test]
    fn test_load_missing_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.jsonl");
        let mem = load(&path).unwrap();
        assert!(mem.is_empty());
    }
}
