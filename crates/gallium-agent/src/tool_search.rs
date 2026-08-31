//! `ToolSearch` — the discovery half of deferred tools.
//!
//! A client may register a tool without advertising it (`thread/start`'s
//! `dynamicTools`, `"advertised": false`), which keeps its schema out of every
//! prompt for the life of the thread. That is only safe if the model has a way
//! back to it, and this is that way: the tool the model calls to find what was
//! held back and put it on the list.
//!
//! It is the piece a client cannot supply. Nothing in the protocol lets a client
//! tell a running thread "now show these", and adding a way would be new wire
//! surface; the registry is already here, so the reveal happens on this side.

use std::sync::Arc;

use crate::tool::{Tool, ToolAnnotations, ToolResult, ToolVisibility};
use crate::AgentError;

/// Lets the model list and un-hide tools that are registered but not advertised.
///
/// Holds the [`ToolVisibility`] mask rather than the registry, which is what
/// keeps this from being a cycle: `ToolSearch` is itself registered in the
/// registry whose tools it reveals, and `Tool::call` takes `&self`.
pub struct ToolSearchTool {
    visibility: Arc<ToolVisibility>,
}

impl ToolSearchTool {
    pub fn new(visibility: Arc<ToolVisibility>) -> Self {
        Self { visibility }
    }

    /// Case-insensitive match of every whitespace-separated term against the
    /// name and description together.
    ///
    /// All terms must appear — "github issue" should not return every tool that
    /// mentions GitHub. Deliberately not ranked: a model that asked for the
    /// wrong thing is better served by a short list it can read than by a
    /// confident first result, and the list is bounded by how much was deferred.
    fn matches(query: &str, name: &str, description: &str) -> bool {
        let haystack = format!("{} {}", name, description).to_lowercase();
        query
            .split_whitespace()
            .all(|term| haystack.contains(&term.to_lowercase()))
    }

    fn render(entries: &[(String, String)]) -> String {
        entries
            .iter()
            .map(|(name, description)| format!("- {}: {}", name, description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Tool for ToolSearchTool {
    fn name(&self) -> &str {
        "ToolSearch"
    }

    fn annotations(&self) -> ToolAnnotations {
        // Reveals nothing outside this process and changes nothing in the
        // world: what it mutates is which schemas the *next* model call carries.
        ToolAnnotations::READ_ONLY
    }

    fn description(&self) -> &str {
        "Find tools that are available but not listed in your current tool set. \
         Call this when no listed tool does what you need, with a few words \
         describing the capability (for example 'github issues' or 'go \
         documentation'). Matching tools are added to your tool set and can be \
         called normally from your next message onward. Call with an empty query \
         to see everything available."
    }

    /// How many tools are still held back, folded into the description each time
    /// the catalog is read. A model deciding whether to search is better off
    /// knowing there are 24 tools it has not seen than guessing, and once the
    /// count reaches zero the line says so rather than inviting another search.
    fn dynamic_state(&self) -> Option<String> {
        Some(match self.visibility.hidden_count() {
            0 => "no unlisted tools remain".to_string(),
            1 => "1 unlisted tool available".to_string(),
            n => format!("{} unlisted tools available", n),
        })
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words describing the capability you need. Empty lists every unlisted tool."
                }
            },
            "required": ["query"]
        })
    }

    fn call(&self, args: serde_json::Value) -> Result<ToolResult, AgentError> {
        // A missing `query` is read as the empty one rather than refused: "show
        // me what there is" is the request a model makes when it does not yet
        // know what to ask for, and failing the call teaches it not to look.
        let query = args["query"].as_str().unwrap_or("").trim().to_string();
        let hidden = self.visibility.hidden_tools();

        if hidden.is_empty() {
            return Ok(ToolResult::text(
                "Every available tool is already listed in your tool set.".to_string(),
            ));
        }

        if query.is_empty() {
            return Ok(ToolResult::text(format!(
                "{} tool(s) are available but not yet listed. Search for one by \
                 capability to add it to your tool set:\n{}",
                hidden.len(),
                Self::render(&hidden)
            )));
        }

        let found: Vec<(String, String)> = hidden
            .iter()
            .filter(|(name, description)| Self::matches(&query, name, description))
            .cloned()
            .collect();

        if found.is_empty() {
            return Ok(ToolResult::text(format!(
                "No unlisted tool matches '{}'. The {} unlisted tool(s) are:\n{}",
                query,
                hidden.len(),
                Self::render(&hidden)
            )));
        }

        for (name, _) in &found {
            self.visibility.reveal(name);
        }

        Ok(ToolResult::text(format!(
            "Added {} tool(s) to your tool set. They are callable from your next \
             message, with the arguments their schemas describe:\n{}",
            found.len(),
            Self::render(&found)
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn masked() -> Arc<ToolVisibility> {
        let visibility = Arc::new(ToolVisibility::new());
        visibility.hide("tree_dir", "walk a Go package tree");
        visibility.hide("read_godoc", "read Go package documentation");
        visibility.hide("list_github_issues", "list issues on a GitHub repository");
        visibility
    }

    fn search(visibility: &Arc<ToolVisibility>, query: &str) -> String {
        ToolSearchTool::new(Arc::clone(visibility))
            .call(serde_json::json!({ "query": query }))
            .unwrap()
            .model_text()
            .to_string()
    }

    /// The reveal is the whole job: after a search the tool stops being hidden,
    /// which is what puts its schema in the next model call.
    #[test]
    fn a_match_is_revealed_and_the_others_stay_hidden() {
        let visibility = masked();
        let found = search(&visibility, "github issues");

        assert!(found.contains("list_github_issues"), "{found}");
        assert!(visibility.is_advertised("list_github_issues"));
        assert!(!visibility.is_advertised("tree_dir"));
        assert_eq!(visibility.hidden_count(), 2);
    }

    /// Every term has to appear, or "github issues" returns each tool that
    /// merely mentions GitHub and the model is back to reading a full catalog.
    #[test]
    fn all_terms_must_match_not_any() {
        let visibility = masked();
        let found = search(&visibility, "go documentation");

        assert!(found.contains("read_godoc"), "{found}");
        assert!(
            !found.contains("tree_dir"),
            "'tree_dir' matches 'go' but not 'documentation': {found}"
        );
    }

    /// A model that does not yet know what to ask for asks for everything. That
    /// lists without revealing — names and descriptions are what the mask keeps,
    /// and handing over every schema on a bare look is the cost being deferred.
    #[test]
    fn an_empty_query_lists_without_revealing() {
        let visibility = masked();
        let listed = search(&visibility, "");

        for name in ["tree_dir", "read_godoc", "list_github_issues"] {
            assert!(listed.contains(name), "{listed}");
        }
        assert_eq!(visibility.hidden_count(), 3, "listing is not revealing");
    }

    /// A miss says what there is instead of just saying no: the model guessed a
    /// vocabulary, and the useful answer is the vocabulary that exists.
    #[test]
    fn a_miss_shows_the_alternatives_and_reveals_nothing() {
        let visibility = masked();
        let missed = search(&visibility, "send an email");

        assert!(missed.contains("No unlisted tool matches"), "{missed}");
        assert!(missed.contains("tree_dir"), "{missed}");
        assert_eq!(visibility.hidden_count(), 3);
    }

    /// Once everything has been revealed the tool says so, rather than inviting
    /// a search that can only come back empty.
    #[test]
    fn an_exhausted_mask_says_there_is_nothing_left() {
        let visibility = Arc::new(ToolVisibility::new());
        let tool = ToolSearchTool::new(Arc::clone(&visibility));

        assert_eq!(
            tool.dynamic_state().as_deref(),
            Some("no unlisted tools remain")
        );
        let answer = tool
            .call(serde_json::json!({ "query": "anything" }))
            .unwrap()
            .model_text()
            .to_string();
        assert!(answer.contains("already listed"), "{answer}");
    }

    /// The count rides on `dynamic_state`, so it is recomputed each time the
    /// catalog is read rather than frozen into the stored descriptor — a model
    /// deciding whether to search should see what is left, not what there was.
    #[test]
    fn the_count_tracks_what_is_still_hidden() {
        let visibility = masked();
        let tool = ToolSearchTool::new(Arc::clone(&visibility));

        assert_eq!(
            tool.dynamic_state().as_deref(),
            Some("3 unlisted tools available")
        );
        visibility.reveal("tree_dir");
        visibility.reveal("read_godoc");
        assert_eq!(
            tool.dynamic_state().as_deref(),
            Some("1 unlisted tool available")
        );
    }
}
