use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::tool::ToolHandler;
use crate::AgentError;

/// A skill is a named prompt template that the agent can look up and apply.
pub struct Skill {
    pub name: String,
    pub description: String,
    pub prompt: String,
}

/// Thread-safe registry of skills.
pub struct SkillRegistry {
    skills: RwLock<HashMap<String, Skill>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new skill.
    pub fn add(&self, name: String, description: String, prompt: String) {
        let mut skills = self.skills.write().unwrap();
        tracing::info!("Registered skill: {}", name);
        skills.insert(
            name.clone(),
            Skill {
                name,
                description,
                prompt,
            },
        );
    }

    /// List all skills as "name: description" lines.
    pub fn list(&self) -> String {
        let skills = self.skills.read().unwrap();
        if skills.is_empty() {
            return "No skills registered.".to_string();
        }
        let mut lines: Vec<String> = skills
            .values()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect();
        lines.sort();
        lines.join("\n")
    }

    /// Get a skill's full prompt by name.
    pub fn get(&self, name: &str) -> Option<String> {
        let skills = self.skills.read().unwrap();
        skills.get(name).map(|s| s.prompt.clone())
    }

    /// Build a catalog string for injection into system prompt.
    /// Returns None if no skills registered.
    pub fn catalog(&self) -> Option<String> {
        let skills = self.skills.read().unwrap();
        if skills.is_empty() {
            return None;
        }
        let mut lines: Vec<String> = skills
            .values()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect();
        lines.sort();
        Some(format!(
            "Available skills (use lookup_skill tool to get full instructions):\n{}",
            lines.join("\n")
        ))
    }

    /// Load all SKILL.md files from a directory, skipping on parse errors.
    pub fn load_from_dir(&self, dir: &Path) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_md = path.extension().map(|e| e == "md").unwrap_or(false);
            if !is_md {
                continue;
            }
            if let Err(e) = self.load_skill_file(&path) {
                tracing::warn!("Skipping skill file {:?}: {}", path, e);
            }
        }
    }

    fn load_skill_file(&self, path: &Path) -> Result<(), String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;

        let (name, description, prompt) = parse_skill_md(&content)
            .ok_or_else(|| "missing or invalid frontmatter".to_string())?;

        self.add(name, description, prompt);
        Ok(())
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a SKILL.md string. Returns (name, description, prompt body) or None.
///
/// Expected format:
/// ```text
/// ---
/// name: my-skill
/// description: Short description
/// ---
/// Prompt body...
/// ```
fn parse_skill_md(content: &str) -> Option<(String, String, String)> {
    let content = content.trim_start();
    if !content.starts_with("---") {
        return None;
    }

    let after_fence = content.strip_prefix("---")?.trim_start_matches('\n');
    let end = after_fence.find("\n---")?;
    let frontmatter = &after_fence[..end];
    let body = after_fence[end..]
        .strip_prefix("\n---")?
        .trim_start_matches('\n');

    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(v.trim().to_string());
        }
    }

    Some((name?, description.unwrap_or_default(), body.to_string()))
}

/// Load skills from project-local and user-global dirs into registry.
pub fn load_skills(registry: &SkillRegistry, working_dir: &Path) {
    // User-global: ~/.config/gallium/skills/
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let global = Path::new(&home)
            .join(".config")
            .join("gallium")
            .join("skills");
        registry.load_from_dir(&global);
    }
    // Project-local overrides global: <working_dir>/.gallium/skills/
    let local = working_dir.join(".gallium").join("skills");
    registry.load_from_dir(&local);
}

// ============================================================================
// SkillLookupTool
// ============================================================================

/// Tool that lets the LLM look up skills from the registry.
pub struct SkillLookupTool {
    registry: std::sync::Arc<SkillRegistry>,
}

impl SkillLookupTool {
    pub fn new(registry: std::sync::Arc<SkillRegistry>) -> Self {
        Self { registry }
    }
}

impl ToolHandler for SkillLookupTool {
    fn name(&self) -> &str {
        "lookup_skill"
    }

    fn description(&self) -> &str {
        "Look up available skills. Use action 'list' to see all skills with descriptions, or action 'get' with a skill name to retrieve the full prompt instructions."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get"],
                    "description": "Action to perform: 'list' all skills or 'get' a specific skill"
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (required when action is 'get')"
                }
            },
            "required": ["action"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<crate::tool::ToolResult, AgentError> {
        let action = args["action"]
            .as_str()
            .ok_or_else(|| AgentError::ParseError("Missing 'action' field".to_string()))?;

        match action {
            "list" => Ok(crate::tool::ToolResult::text(self.registry.list())),
            "get" => {
                let name = args["name"]
                    .as_str()
                    .ok_or_else(|| AgentError::ParseError("Missing 'name' field for 'get' action".to_string()))?;
                match self.registry.get(name) {
                    Some(prompt) => Ok(crate::tool::ToolResult::text(format!("## Skill: {}\n\n{}", name, prompt))),
                    None => Ok(crate::tool::ToolResult::text(format!("Skill '{}' not found. Use action 'list' to see available skills.", name))),
                }
            }
            _ => Err(AgentError::ParseError(format!("Unknown action: {}", action))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_skill_registry() {
        let registry = SkillRegistry::new();
        assert_eq!(registry.list(), "No skills registered.");
        assert!(registry.catalog().is_none());

        registry.add(
            "test-skill".to_string(),
            "A test skill".to_string(),
            "Do the test thing.".to_string(),
        );

        assert!(registry.list().contains("test-skill"));
        assert!(registry.list().contains("A test skill"));
        assert_eq!(registry.get("test-skill"), Some("Do the test thing.".to_string()));
        assert_eq!(registry.get("nonexistent"), None);
        assert!(registry.catalog().unwrap().contains("lookup_skill"));
    }

    #[test]
    fn test_skill_lookup_tool() {
        let registry = Arc::new(SkillRegistry::new());
        registry.add(
            "greeting".to_string(),
            "Greet the user".to_string(),
            "Say hello warmly.".to_string(),
        );

        let tool = SkillLookupTool::new(registry);

        // List
        let result = tool.call(serde_json::json!({"action": "list"})).unwrap().text;
        assert!(result.contains("greeting"));

        // Get existing
        let result = tool.call(serde_json::json!({"action": "get", "name": "greeting"})).unwrap().text;
        assert!(result.contains("Say hello warmly."));

        // Get nonexistent
        let result = tool.call(serde_json::json!({"action": "get", "name": "nope"})).unwrap().text;
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_parse_skill_md() {
        let md = "---\nname: code\ndescription: Write code\n---\nDo the code thing.\n";
        let (name, desc, prompt) = parse_skill_md(md).unwrap();
        assert_eq!(name, "code");
        assert_eq!(desc, "Write code");
        assert_eq!(prompt.trim(), "Do the code thing.");
    }

    #[test]
    fn test_parse_skill_md_no_frontmatter() {
        assert!(parse_skill_md("Just some text").is_none());
    }
}
