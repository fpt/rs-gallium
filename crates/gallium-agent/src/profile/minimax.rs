//! MiniMax-M2.7: `<minimax:tool_call><invoke name="…">`.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::ModelProfile;

/// MiniMax-M2.7 (issue #105). Reasons before answering, and its template
/// pre-fills `<think>\n` into the *prompt* rather than generating it — so the
/// output carries only the closing tag, which the generic strip handles.
pub struct MiniMaxM2;

impl ModelProfile for MiniMaxM2 {
    fn name(&self) -> &'static str {
        "minimax-m2"
    }

    /// Exactly `minimax-m2`, not every `minimax-*`: llama.cpp also knows
    /// `minimax-m3`, whose wire format nobody here has looked at. An
    /// unrecognized sibling is better served by the generic fallback than by a
    /// parser claiming to understand it.
    fn matches_arch(&self, arch: &str) -> bool {
        arch == "minimax-m2"
    }

    fn parse_native_tool_calls(&self, text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        // `tools` is load-bearing here and only here: the format renders a
        // string argument raw and everything else tojson-encoded, so `"42"` and
        // `42` are byte-identical on the wire and only the schema separates them.
        wire::minimax::parse_calls(text, tools)
    }

    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains(wire::minimax::WRAPPER_OPEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolDefinition;

    #[test]
    fn m3_is_not_claimed_as_m2() {
        assert!(MiniMaxM2.matches_arch("minimax-m2"));
        assert!(!MiniMaxM2.matches_arch("minimax-m3"));
    }

    #[test]
    fn argument_types_come_from_the_tool_schema() {
        let tools = [ToolDefinition {
            name: "grep".to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "limit": {"type": "integer"}}
            }),
        }];
        let calls = MiniMaxM2.tool_calls(
            "<minimax:tool_call>\n<invoke name=\"grep\">\n\
             <parameter name=\"pattern\">50</parameter>\n\
             <parameter name=\"limit\">50</parameter>\n\
             </invoke>\n</minimax:tool_call>",
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "50");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    /// The reason a family profile tries its native format *before* the JSON
    /// scan. This payload is a legitimate `Write` whose content happens to be
    /// JSON with a `name` key — the balanced-span scan would return a call to
    /// "impostor", which is what the old json-first cascade did (and `Generic`
    /// still does, deliberately: see its doc).
    #[test]
    fn a_json_argument_carrying_a_name_key_is_not_mistaken_for_the_call() {
        let calls = MiniMaxM2.tool_calls(
            "<minimax:tool_call>\n<invoke name=\"Write\">\n\
             <parameter name=\"content\">{\"name\": \"impostor\", \"arguments\": {}}</parameter>\n\
             </invoke>\n</minimax:tool_call>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(
            calls[0].arguments["content"],
            "{\"name\": \"impostor\", \"arguments\": {}}"
        );
    }

    /// The template pre-fills `<think>\n` into the prompt, so only the closing
    /// tag is generated (6f80ba8).
    #[test]
    fn a_bare_closing_think_tag_still_marks_the_reasoning() {
        let raw = "Let me summarize.\n</think>\n\n## Summary\nDone.";
        assert_eq!(MiniMaxM2.clean_reply(raw), "## Summary\nDone.");
    }
}
