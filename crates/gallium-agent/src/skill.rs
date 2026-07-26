use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::tool::{Tool, ToolAnnotations};
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

    /// Load skills from a directory, skipping on parse errors. Returns how many
    /// were loaded, so a caller can report where its skills came from.
    ///
    /// Two layouts, because two conventions exist and both turn up in the same
    /// checkout:
    ///
    /// - `<dir>/*.md` — one file per skill, which is what gallium's own
    ///   `.gallium/skills` uses.
    /// - `<dir>/<name>/SKILL.md` — a directory per skill, which is what
    ///   `.claude/skills` and `.agents/skills` use, and what lets a skill ship
    ///   supporting files next to its prompt.
    ///
    /// Only one level deep: a skill directory holds a `SKILL.md`, not more
    /// skills.
    pub fn load_from_dir(&self, dir: &Path) -> usize {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return 0,
        };
        let mut loaded = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let candidate = if path.is_dir() {
                let nested = path.join("SKILL.md");
                if !nested.is_file() {
                    continue;
                }
                nested
            } else if path.extension().map(|e| e == "md").unwrap_or(false) {
                path
            } else {
                continue;
            };

            match self.load_skill_file(&candidate) {
                Ok(()) => loaded += 1,
                Err(e) => tracing::warn!("Skipping skill file {:?}: {}", candidate, e),
            }
        }
        loaded
    }

    fn load_skill_file(&self, path: &Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {}", e))?;

        let (name, description, prompt) =
            parse_skill_md(&content).ok_or_else(|| "missing or invalid frontmatter".to_string())?;

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

/// A directory skills were actually loaded from, and how many came out of it.
pub struct SkillSource {
    pub dir: std::path::PathBuf,
    pub count: usize,
}

/// Load skills from every standard location, nearest last.
///
/// Skills are keyed by name, so a later directory overrides an earlier one of
/// the same name. The order is least specific to most: the user's global
/// directory, then the conventions a project may already carry for other
/// agents, then gallium's own — which wins, because a skill someone wrote
/// specifically for this tool should beat a shared one it collides with.
///
/// Returns only the directories that yielded something, so a frontend can tell
/// the user which of these actually exist.
pub fn load_skills(registry: &SkillRegistry, working_dir: &Path) -> Vec<SkillSource> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        dirs.push(
            Path::new(&home)
                .join(".config")
                .join("gallium")
                .join("skills"),
        );
    }
    dirs.push(working_dir.join(".claude").join("skills"));
    dirs.push(working_dir.join(".agents").join("skills"));
    dirs.push(working_dir.join(".gallium").join("skills"));

    dirs.into_iter()
        .filter_map(|dir| match registry.load_from_dir(&dir) {
            0 => None,
            count => Some(SkillSource { dir, count }),
        })
        .collect()
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

impl Tool for SkillLookupTool {
    fn name(&self) -> &str {
        "lookup_skill"
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
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
                let name = args["name"].as_str().ok_or_else(|| {
                    AgentError::ParseError("Missing 'name' field for 'get' action".to_string())
                })?;
                match self.registry.get(name) {
                    Some(prompt) => Ok(crate::tool::ToolResult::text(format!(
                        "## Skill: {}\n\n{}",
                        name, prompt
                    ))),
                    None => Ok(crate::tool::ToolResult::text(format!(
                        "Skill '{}' not found. Use action 'list' to see available skills.",
                        name
                    ))),
                }
            }
            _ => Err(AgentError::ParseError(format!(
                "Unknown action: {}",
                action
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn skill_md(name: &str) -> String {
        format!("---\nname: {name}\ndescription: does {name}\n---\nBody for {name}.\n")
    }

    /// `.claude/skills` and `.agents/skills` put each skill in its own
    /// directory with a `SKILL.md` inside, so a loader that only reads `*.md`
    /// at the top level finds nothing there and says nothing about it.
    #[test]
    fn both_the_flat_and_the_directory_layout_load() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("flat.md"), skill_md("flat")).unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("SKILL.md"), skill_md("nested")).unwrap();
        // A directory with no SKILL.md is not a skill, and not an error either.
        std::fs::create_dir(dir.path().join("assets")).unwrap();

        let registry = SkillRegistry::new();
        assert_eq!(registry.load_from_dir(dir.path()), 2);
        assert!(registry.get("flat").is_some());
        assert!(registry.get("nested").is_some());
    }

    #[test]
    fn a_directory_that_does_not_exist_loads_nothing_and_does_not_fail() {
        let registry = SkillRegistry::new();
        assert_eq!(registry.load_from_dir(Path::new("/nonexistent/skills")), 0);
    }

    /// The nearest definition wins, so a project can override a global skill.
    #[test]
    fn a_later_directory_overrides_a_skill_of_the_same_name() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        std::fs::write(first.path().join("dup.md"), skill_md("dup")).unwrap();
        std::fs::write(
            second.path().join("dup.md"),
            "---\nname: dup\ndescription: the closer one\n---\nCloser body.\n",
        )
        .unwrap();

        let registry = SkillRegistry::new();
        registry.load_from_dir(first.path());
        registry.load_from_dir(second.path());
        assert!(registry.get("dup").unwrap().contains("Closer body"));
    }

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
        assert_eq!(
            registry.get("test-skill"),
            Some("Do the test thing.".to_string())
        );
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
        let result = tool
            .call(serde_json::json!({"action": "list"}))
            .unwrap()
            .model_text()
            .to_string();
        assert!(result.contains("greeting"));

        // Get existing
        let result = tool
            .call(serde_json::json!({"action": "get", "name": "greeting"}))
            .unwrap()
            .model_text()
            .to_string();
        assert!(result.contains("Say hello warmly."));

        // Get nonexistent
        let result = tool
            .call(serde_json::json!({"action": "get", "name": "nope"}))
            .unwrap()
            .model_text()
            .to_string();
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
