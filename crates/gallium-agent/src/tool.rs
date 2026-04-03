//! Tool system: ToolHandler trait, ToolRegistry, built-in tools.
//!
//! Adapted from voice-agent/crates/lib/src/tool.rs.
//! Removed: SkillLookupTool, ReadSituationMessagesTool (voice-specific).
//! Kept: ReadTool, GlobTool, TaskTool, all registry infrastructure.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::llm::ToolDefinition;
use crate::AgentError;

/// Maximum characters in a tool result before truncation (~2k tokens).
const MAX_OUTPUT_CHARS: usize = 8000;

/// Result of a tool call.
#[derive(Debug)]
pub struct ToolResult {
    pub text: String,
    pub images: Vec<crate::llm::ImageContent>,
}

impl ToolResult {
    pub fn text(s: String) -> Self {
        Self { text: s, images: vec![] }
    }

    fn truncate(&mut self) {
        if self.text.len() > MAX_OUTPUT_CHARS {
            let total = self.text.len();
            let end = self.text.floor_char_boundary(MAX_OUTPUT_CHARS);
            self.text.truncate(end);
            self.text.push_str(&format!(
                "\n\n... (truncated: showing {}/{} chars. Use offset/limit or filter to narrow results.)",
                end, total
            ));
        }
    }
}

impl From<String> for ToolResult {
    fn from(s: String) -> Self {
        Self::text(s)
    }
}

/// Trait for tool implementations.
pub trait ToolHandler: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError>;

    fn dynamic_state(&self) -> Option<String> { None }
}

fn full_description(tool: &dyn ToolHandler) -> String {
    match tool.dynamic_state() {
        Some(state) => format!("{} [{}]", tool.description(), state),
        None => tool.description().to_string(),
    }
}

/// Trait for accessing tools (implemented by ToolRegistry and FilteredToolRegistry).
pub trait ToolAccess {
    fn get_definitions(&self) -> Vec<ToolDefinition>;
    fn call(&self, name: &str, args: serde_json::Value) -> Result<ToolResult, AgentError>;
    fn is_empty(&self) -> bool;
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn ToolHandler>) {
        tracing::info!("Registered tool: {}", tool.name());
        self.tools.push(tool);
    }

    pub fn filtered(&self, allowed: &[String]) -> FilteredToolRegistry<'_> {
        FilteredToolRegistry { tools: &self.tools, allowed: allowed.to_vec() }
    }
}

impl ToolAccess for ToolRegistry {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| ToolDefinition {
            name: t.name().to_string(),
            description: full_description(t.as_ref()),
            parameters: t.parameters_schema(),
        }).collect()
    }

    fn call(&self, name: &str, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let tool = self.tools.iter().find(|t| t.name() == name)
            .ok_or_else(|| AgentError::InternalError(format!("Unknown tool: {}", name)))?;
        tracing::info!("Calling tool: {} with args: {}", name, args);
        let mut result = tool.call(args)?;
        result.truncate();
        Ok(result)
    }

    fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

pub struct FilteredToolRegistry<'a> {
    tools: &'a [Box<dyn ToolHandler>],
    allowed: Vec<String>,
}

impl<'a> ToolAccess for FilteredToolRegistry<'a> {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter()
            .filter(|t| self.allowed.iter().any(|a| a == t.name()))
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: full_description(t.as_ref()),
                parameters: t.parameters_schema(),
            }).collect()
    }

    fn call(&self, name: &str, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        if !self.allowed.iter().any(|a| a == name) {
            return Err(AgentError::InternalError(format!("Tool not allowed: {}", name)));
        }
        let tool = self.tools.iter().find(|t| t.name() == name)
            .ok_or_else(|| AgentError::InternalError(format!("Unknown tool: {}", name)))?;
        tracing::info!("Calling tool: {} with args: {}", name, args);
        let mut result = tool.call(args)?;
        result.truncate();
        Ok(result)
    }

    fn is_empty(&self) -> bool {
        !self.tools.iter().any(|t| self.allowed.iter().any(|a| a == t.name()))
    }
}

/// Create default registry with built-in tools.
pub fn create_default_registry(working_dir: PathBuf) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(ReadTool::new(working_dir.clone())));
    registry.register(Box::new(GlobTool::new(working_dir.clone())));
    registry.register(Box::new(WriteTool::new(working_dir.clone())));
    registry.register(Box::new(EditTool::new(working_dir.clone())));
    registry.register(Box::new(TaskTool::new()));
    registry
}

// ============================================================================
// ReadTool
// ============================================================================

pub struct ReadTool { working_dir: PathBuf }

impl ReadTool {
    pub fn new(working_dir: PathBuf) -> Self { Self { working_dir } }

    fn resolve(&self, file_path: &str) -> PathBuf {
        let p = Path::new(file_path);
        if p.is_absolute() { p.to_path_buf() } else { self.working_dir.join(p) }
    }
}

impl ToolHandler for ReadTool {
    fn name(&self) -> &str { "read" }

    fn description(&self) -> &str {
        "Read a file's contents with line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Path to the file (absolute or relative to working directory)" },
                "offset": { "type": "integer", "description": "Line to start reading from (1-based, default: 1)" },
                "limit": { "type": "integer", "description": "Max lines to read (default: 2000)" }
            },
            "required": ["file_path"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let file_path = args["file_path"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing file_path".to_string()))?;
        let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = args["limit"].as_u64().unwrap_or(2000) as usize;

        let resolved = self.resolve(file_path);
        let content = std::fs::read_to_string(&resolved).map_err(|e| {
            AgentError::InternalError(format!("Failed to read {}: {}", resolved.display(), e))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let start = (offset - 1).min(total);
        let end = (start + limit).min(total);

        let mut output = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            output.push_str(&format!("{:>6}\t{}\n", start + i + 1, line));
        }
        if end < total {
            output.push_str(&format!("\n... ({} more lines, {} total)\n", total - end, total));
        }

        Ok(ToolResult::text(output))
    }
}

// ============================================================================
// GlobTool
// ============================================================================

pub struct GlobTool { working_dir: PathBuf }

impl GlobTool {
    pub fn new(working_dir: PathBuf) -> Self { Self { working_dir } }
}

impl ToolHandler for GlobTool {
    fn name(&self) -> &str { "glob" }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g. \"**/*.rs\"). Returns matching paths."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern (e.g. \"**/*.rs\")" },
                "path": { "type": "string", "description": "Base directory (default: working directory)" },
                "limit": { "type": "integer", "description": "Max results (default: 100)" }
            },
            "required": ["pattern"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let pattern = args["pattern"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing pattern".to_string()))?;
        let limit = args["limit"].as_u64().unwrap_or(100) as usize;
        let base = args["path"].as_str()
            .map(|p| {
                let path = Path::new(p);
                if path.is_absolute() { path.to_path_buf() } else { self.working_dir.join(p) }
            })
            .unwrap_or_else(|| self.working_dir.clone());

        let full_pattern = base.join(pattern);
        let full_str = full_pattern.to_string_lossy();

        let mut matches: Vec<String> = Vec::new();
        let mut total = 0usize;
        let entries = glob::glob(&full_str).map_err(|e| {
            AgentError::InternalError(format!("Invalid glob '{}': {}", full_str, e))
        })?;

        for entry in entries {
            match entry {
                Ok(path) => {
                    total += 1;
                    if matches.len() < limit {
                        let display = path.strip_prefix(&self.working_dir)
                            .unwrap_or(&path).to_string_lossy().to_string();
                        matches.push(display);
                    }
                }
                Err(e) => tracing::warn!("Glob error: {}", e),
            }
        }
        matches.sort();

        if matches.is_empty() {
            Ok(ToolResult::text(format!("No files found matching '{}'", pattern)))
        } else if total > matches.len() {
            Ok(ToolResult::text(format!(
                "{}\n\n... (showing {}/{} files)",
                matches.join("\n"), matches.len(), total
            )))
        } else {
            Ok(ToolResult::text(format!("{}\n\n({} files found)", matches.join("\n"), total)))
        }
    }
}

// ============================================================================
// WriteTool
// ============================================================================

pub struct WriteTool { working_dir: PathBuf }

impl WriteTool {
    pub fn new(working_dir: PathBuf) -> Self { Self { working_dir } }

    fn resolve(&self, file_path: &str) -> PathBuf {
        let p = Path::new(file_path);
        if p.is_absolute() { p.to_path_buf() } else { self.working_dir.join(p) }
    }
}

impl ToolHandler for WriteTool {
    fn name(&self) -> &str { "write" }

    fn description(&self) -> &str {
        "Write content to a file, creating it (or overwriting it) and any missing parent directories."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "Path to write (absolute or relative to working directory)" },
                "content":   { "type": "string", "description": "Full content to write" }
            },
            "required": ["file_path", "content"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let file_path = args["file_path"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing file_path".to_string()))?;
        let content = args["content"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing content".to_string()))?;

        let resolved = self.resolve(file_path);
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                AgentError::InternalError(format!("Cannot create directories for {}: {}", resolved.display(), e))
            })?;
        }
        std::fs::write(&resolved, content).map_err(|e| {
            AgentError::InternalError(format!("Cannot write {}: {}", resolved.display(), e))
        })?;

        Ok(ToolResult::text(format!("Wrote {} bytes to {}", content.len(), file_path)))
    }
}

// ============================================================================
// EditTool
// ============================================================================

pub struct EditTool { working_dir: PathBuf }

impl EditTool {
    pub fn new(working_dir: PathBuf) -> Self { Self { working_dir } }

    fn resolve(&self, file_path: &str) -> PathBuf {
        let p = Path::new(file_path);
        if p.is_absolute() { p.to_path_buf() } else { self.working_dir.join(p) }
    }
}

impl ToolHandler for EditTool {
    fn name(&self) -> &str { "edit" }

    fn description(&self) -> &str {
        "Replace an exact string in a file. Fails if old_string is not found or appears more than once."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path":  { "type": "string", "description": "Path to the file (absolute or relative to working directory)" },
                "old_string": { "type": "string", "description": "Exact text to replace (must appear exactly once)" },
                "new_string": { "type": "string", "description": "Replacement text" }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let file_path = args["file_path"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing file_path".to_string()))?;
        let old_string = args["old_string"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing old_string".to_string()))?;
        let new_string = args["new_string"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing new_string".to_string()))?;

        let resolved = self.resolve(file_path);
        let content = std::fs::read_to_string(&resolved).map_err(|e| {
            AgentError::InternalError(format!("Cannot read {}: {}", resolved.display(), e))
        })?;

        let count = content.matches(old_string).count();
        match count {
            0 => Err(AgentError::InternalError(format!(
                "old_string not found in {}. Check exact whitespace and indentation.",
                file_path
            ))),
            1 => {
                let new_content = content.replacen(old_string, new_string, 1);
                std::fs::write(&resolved, &new_content).map_err(|e| {
                    AgentError::InternalError(format!("Cannot write {}: {}", resolved.display(), e))
                })?;
                Ok(ToolResult::text(format!("Edited {}", file_path)))
            }
            n => Err(AgentError::InternalError(format!(
                "old_string appears {} times in {}. Provide more context to make it unique.",
                n, file_path
            ))),
        }
    }
}

// ============================================================================
// TaskTool
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TaskItem {
    id: u32,
    subject: String,
    description: String,
    status: String,
}

pub struct TaskTool {
    tasks: Mutex<Vec<TaskItem>>,
    next_id: Mutex<u32>,
}

impl TaskTool {
    pub fn new() -> Self {
        Self { tasks: Mutex::new(Vec::new()), next_id: Mutex::new(1) }
    }
}

impl ToolHandler for TaskTool {
    fn name(&self) -> &str { "tasks" }

    fn description(&self) -> &str {
        "Manage an in-memory task list. Actions: create, update, list."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["create", "update", "list"] },
                "subject": { "type": "string", "description": "Task title (for create)" },
                "description": { "type": "string", "description": "Task description (for create)" },
                "task_id": { "type": "integer", "description": "Task ID (for update)" },
                "status": { "type": "string", "enum": ["pending", "in_progress", "completed"], "description": "New status (for update)" }
            },
            "required": ["action"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        let action = args["action"].as_str()
            .ok_or_else(|| AgentError::ParseError("Missing action".to_string()))?;

        match action {
            "create" => {
                let subject = args["subject"].as_str().unwrap_or("Untitled").to_string();
                let description = args["description"].as_str().unwrap_or("").to_string();
                let mut tasks = self.tasks.lock().map_err(|e| AgentError::InternalError(e.to_string()))?;
                let mut next_id = self.next_id.lock().map_err(|e| AgentError::InternalError(e.to_string()))?;
                let id = *next_id; *next_id += 1;
                tasks.push(TaskItem { id, subject: subject.clone(), description, status: "pending".to_string() });
                Ok(ToolResult::text(format!("Created task #{}: {}", id, subject)))
            }
            "update" => {
                let task_id = args["task_id"].as_u64()
                    .ok_or_else(|| AgentError::ParseError("Missing task_id".to_string()))? as u32;
                let status = args["status"].as_str()
                    .ok_or_else(|| AgentError::ParseError("Missing status".to_string()))?;
                let mut tasks = self.tasks.lock().map_err(|e| AgentError::InternalError(e.to_string()))?;
                let task = tasks.iter_mut().find(|t| t.id == task_id)
                    .ok_or_else(|| AgentError::InternalError(format!("Task #{} not found", task_id)))?;
                task.status = status.to_string();
                Ok(ToolResult::text(format!("Updated task #{} '{}' → {}", task_id, task.subject, status)))
            }
            "list" => {
                let tasks = self.tasks.lock().map_err(|e| AgentError::InternalError(e.to_string()))?;
                if tasks.is_empty() { return Ok(ToolResult::text("No tasks.".to_string())); }
                let mut out = String::from("Tasks:\n");
                for t in tasks.iter() {
                    let icon = match t.status.as_str() { "completed" => "[x]", "in_progress" => "[~]", _ => "[ ]" };
                    out.push_str(&format!("  #{} {} {} - {}\n", t.id, icon, t.subject, t.status));
                    if !t.description.is_empty() { out.push_str(&format!("       {}\n", t.description)); }
                }
                Ok(ToolResult::text(out))
            }
            _ => Err(AgentError::ParseError(format!("Unknown action: {}. Use create/update/list.", action))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_tool() {
        let dir = std::env::temp_dir();
        let mut file = NamedTempFile::new_in(&dir).unwrap();
        writeln!(file, "line one").unwrap();
        writeln!(file, "line two").unwrap();
        let tool = ReadTool::new(dir);
        let result = tool.call(serde_json::json!({
            "file_path": file.path().to_string_lossy().to_string()
        })).unwrap().text;
        assert!(result.contains("line one"));
        assert!(result.contains("1\t"));
    }

    #[test]
    fn test_glob_tool() {
        let dir = std::env::temp_dir().join("gallium_agent_glob_test");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("test.txt"), "hello").unwrap();
        std::fs::write(dir.join("test.rs"), "fn main(){}").unwrap();
        let tool = GlobTool::new(dir.clone());
        let result = tool.call(serde_json::json!({"pattern": "*.txt"})).unwrap().text;
        assert!(result.contains("test.txt"));
        assert!(!result.contains("test.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_task_lifecycle() {
        let tool = TaskTool::new();
        let r = tool.call(serde_json::json!({"action": "create", "subject": "Fix bug"})).unwrap().text;
        assert!(r.contains("#1"));
        let r = tool.call(serde_json::json!({"action": "list"})).unwrap().text;
        assert!(r.contains("Fix bug"));
        let r = tool.call(serde_json::json!({"action": "update", "task_id": 1, "status": "completed"})).unwrap().text;
        assert!(r.contains("completed"));
    }

    #[test]
    fn test_default_registry_has_five_tools() {
        let dir = std::env::temp_dir();
        let reg = create_default_registry(dir);
        let defs = reg.get_definitions();
        assert_eq!(defs.len(), 5);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"read"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"write"));
        assert!(names.contains(&"edit"));
        assert!(names.contains(&"tasks"));
    }

    #[test]
    fn test_write_tool() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteTool::new(dir.path().to_path_buf());
        let result = tool.call(serde_json::json!({
            "file_path": "hello.txt",
            "content": "hello world"
        })).unwrap();
        assert!(result.text.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(), "hello world");
    }

    #[test]
    fn test_edit_tool() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("code.txt");
        std::fs::write(&path, "hello world\n").unwrap();
        let tool = EditTool::new(dir.path().to_path_buf());

        // successful edit
        let result = tool.call(serde_json::json!({
            "file_path": "code.txt",
            "old_string": "hello world",
            "new_string": "goodbye world"
        })).unwrap();
        assert!(result.text.contains("Edited"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "goodbye world\n");

        // not found
        let err = tool.call(serde_json::json!({
            "file_path": "code.txt",
            "old_string": "hello world",
            "new_string": "x"
        })).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }
}
