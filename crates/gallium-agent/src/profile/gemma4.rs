//! Gemma 4: `<|tool_call>call:NAME{…}<tool_call|>` and the thought channel.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::ModelProfile;

/// Note `crate::gemma`'s `normalise_tool_name` / `normalise_path_args` are
/// **not** applied here. They are opt-in, and only the candle path
/// (`protocol.rs`) opts in — llama.cpp keeps names verbatim so a mixed-case MCP
/// tool is never folded. Unifying the two is a behavior change on one engine
/// whichever way it goes, so it waits for its own change rather than riding along
/// with a refactor.
///
/// Gemma 4 (E2B/E4B, 12B, 26B-A4B), whose template declares tools as
/// `<|tool>declaration:…<tool|>` and which answers with
/// `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` — `<|"|>` being its
/// quote token. See docs/GEMMA4.md.
pub struct Gemma4;

impl ModelProfile for Gemma4 {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    /// The exact Gemma **4** architecture names, not `gemma` by prefix: Gemma 2
    /// and 3 are different formats with no `<|tool_call>` at all, and claiming
    /// them would hand their output a parser built for another generation.
    /// `gemma4-assistant` is llama.cpp's separate arch for the draft/assistant
    /// variant, same wire format.
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(arch, "gemma4" | "gemma4-assistant")
    }

    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        wire::gemma_calls(text)
    }

    /// Also trims the trailing turn markers, which matters on one engine only:
    /// the candle backend decodes with special tokens kept as text, so a reply
    /// arrives ending in `<turn|>` / `<eos>` and would show them to the user.
    /// On llama.cpp those are EOG tokens that stop generation and never reach
    /// the string, so the trim is a no-op there rather than a second behavior.
    fn clean_reply(&self, text: &str) -> String {
        let s = crate::gemma::strip_thinking_blocks(text);
        let s = wire::think::strip_think_blocks(&s);
        // `<end_of_turn>` is Gemma 2's spelling, kept for a GGUF converted
        // from that generation's template.
        wire::strip_trailing_markers(s.trim(), &["<turn|>", "<eos>", "<end_of_turn>"]).to_string()
    }

    /// Stop as soon as the model closes a tool call, or claims a tool
    /// *response*: the first means the call is complete and gallium should run
    /// it, the second means the model has started writing the result itself,
    /// which it must not be allowed to finish.
    ///
    /// The two take different tests. A closing `<tool_call|>` is only a boundary
    /// at the very end of what has been sampled — it appears mid-text in a reply
    /// that merely quotes the syntax — while `<|tool_response>` anywhere is
    /// already the failure.
    fn stops_generation(&self, text: &str) -> bool {
        text.ends_with("<tool_call|>") || text.contains("<|tool_response>")
    }

    /// Both are single tokens in every Gemma 4 vocabulary this has been
    /// checked against (`<tool_call|>` id 49, `<|tool_response>` id 50 — see
    /// `protocol.rs`'s `GemmaProtocol` doc comment) so the id-comparison path
    /// (ADR 0003 step 5) is expected to apply here, with `stops_generation`
    /// above as the fallback should a converted GGUF ever split one.
    fn stop_markers(&self) -> &[&'static str] {
        &["<tool_call|>", "<|tool_response>"]
    }

    /// Three literals because a Gemma 4 GGUF's template may spell its tool
    /// section any of these ways depending on how it was converted.
    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains("<|tool_call>")
            || template.contains("<|tool>")
            || template.contains("declaration:")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 2 and 3 are real llama.cpp architectures with a different format.
    /// Matching the family by substring would have claimed them.
    #[test]
    fn only_gemma_4_is_claimed() {
        for arch in ["gemma4", "gemma4-assistant"] {
            assert!(Gemma4.matches_arch(arch), "{arch}");
        }
        for arch in ["gemma", "gemma2", "gemma3", "gemma3n", "gemma-embedding"] {
            assert!(!Gemma4.matches_arch(arch), "{arch}");
        }
    }

    #[test]
    fn parses_its_native_call_including_hyphenated_mcp_names() {
        let calls = Gemma4.tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc");
        assert_eq!(calls[0].arguments["query"], "mcp-go");
    }

    /// The thought channel reached a user in a real gemma4-12b session. Both of
    /// Gemma's wrapper shapes have to come off.
    #[test]
    fn both_thinking_wrappers_are_stripped() {
        assert_eq!(
            Gemma4.clean_reply("<|channel>thought\nchecking git log<channel|>Here it is."),
            "Here it is."
        );
        assert_eq!(
            Gemma4.clean_reply("<|think|>reasoning<|/think|>The answer."),
            "The answer."
        );
    }

    #[test]
    fn generation_stops_at_a_closed_call_or_any_claimed_response() {
        assert!(Gemma4.stops_generation("<|tool_call>call:read{}<tool_call|>"));
        assert!(Gemma4.stops_generation("<|tool_response>response:read{...}"));
        // Mid-call: the arguments are still being written.
        assert!(!Gemma4.stops_generation("<|tool_call>call:read{path:<|\"|>a"));
        assert!(!Gemma4.stops_generation("The answer is 42."));
    }

    /// The two names `stop_markers` returns must be exactly the two literals
    /// `stops_generation` above tests against — an engine that resolves them
    /// to token ids is replacing that predicate, not answering a different
    /// question.
    #[test]
    fn stop_markers_match_stops_generation() {
        assert_eq!(Gemma4.stop_markers(), &["<tool_call|>", "<|tool_response>"]);
    }
}

#[cfg(test)]
mod port_tests {
    use super::*;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    /// The candle path decodes with special tokens kept as text, so a reply
    /// arrives with its turn marker attached. Transcribed from `protocol.rs`'s
    /// `strip_gemma_specials`, which is the spec for this.
    #[test]
    fn trailing_turn_markers_do_not_reach_the_user() {
        assert_eq!(
            Gemma4.clean_reply("The answer is 42.<turn|>"),
            "The answer is 42."
        );
        assert_eq!(Gemma4.clean_reply("Done.<turn|><eos>"), "Done.");
        assert_eq!(Gemma4.clean_reply("Done.<end_of_turn>"), "Done.");
        // A no-op on llama.cpp, where these never reach the string.
        assert_eq!(Gemma4.clean_reply("The answer is 42."), "The answer is 42.");
    }

    /// Mixed-case MCP names still survive when offered, which is the property
    /// `crate::gemma`'s "verbatim" contract exists to protect.
    #[test]
    fn a_hyphenated_mcp_name_survives() {
        let tools = [tool("search-godoc")];
        let calls = Gemma4.tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
            &tools,
        );
        assert_eq!(calls[0].name, "search-godoc");
    }
}
