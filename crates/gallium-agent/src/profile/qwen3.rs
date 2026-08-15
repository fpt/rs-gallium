//! Qwen 3.x: ChatML, `<think>`, and two shapes of tool call.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::{ModelProfile, ReasoningEffort, ReasoningParams};

/// Qwen 3.6 and its `qwen3*` siblings. A reasoning model on ChatML
/// (`<|im_start|>role`), which emits `<think>…</think>` before answering — the
/// generic reply cleaning is exactly right for it.
///
/// **Two shapes, and which one arrives depends on what gallium asked for.**
/// `unsloth/Qwen3.5-9B-GGUF` (`arch = "qwen35"`) has `<function=` /
/// `<parameter=` in its embedded template and no `"name"` anywhere, so the
/// XML-parameter form is what that template renders. But gallium does not use it
/// on the llama.cpp path: `template_formats_tools_natively` is false for this
/// family, so tools arrive as gallium's JSON-prose instructions and the model
/// follows them — a live run answers `{"name": "Read", "arguments": {…}}`, read
/// by [`wire::json`].
///
/// So both shapes are real and neither parser is dead. The XML one
/// ([`wire::qwen_xml`], ported out of `protocol.rs` where it served candle alone)
/// is what the candle backend needs, since that engine renders the model's own
/// prompt format and gets XML back. Step 2 claimed this family needed no native
/// parser at all: wrong about candle, and the reason it looked right on llama.cpp
/// is the template check, not the family.
pub struct Qwen3;

/// Closes a `<tool_call>` block. A single token in Qwen 3.5's vocabulary.
const TOOL_CALL_CLOSE: &str = "</tool_call>";

impl ModelProfile for Qwen3 {
    fn name(&self) -> &'static str {
        "qwen3"
    }

    /// Every `qwen3`-prefixed architecture: `qwen3`, `qwen3moe`, `qwen3next`,
    /// `qwen3vl`, `qwen3vlmoe`, and the `qwen35`/`qwen35moe` family Qwen 3.6
    /// ships as. They differ in weights, not on the wire, which is all a profile
    /// speaks to. `qwen2*` is excluded: that generation has no `<think>` block,
    /// so stripping one is at best a no-op and at worst a claim about a model
    /// nobody checked.
    fn matches_arch(&self, arch: &str) -> bool {
        arch.starts_with("qwen3")
    }

    /// The XML-parameter form, `<tool_call><function=NAME><parameter=K>V…`.
    /// Values are strings — the format carries no type information.
    ///
    /// **Which replies this catches is narrower than it looks.** Asked for
    /// gallium's JSON protocol, Qwen 3.5 complies: a live
    /// `unsloth/Qwen3.5-9B-GGUF` run answers a read request with
    /// `{"name": "Read", "arguments": {…}}`, which the fallback has always read.
    /// So on the llama.cpp path this is inert today — `template_formats_tools_natively`
    /// is false for the family, so tools are never rendered through the template
    /// that would elicit XML. It matters where the prompt *is* the model's own
    /// format: the candle backend renders that and gets XML back, and it would
    /// matter here too if native rendering were ever turned on.
    ///
    /// The `<tool_call>`-wrapped *JSON* shape is deliberately not handled here:
    /// [`wire::json`] already reads it out of the middle of the tags, so there is
    /// one JSON path rather than two.
    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        wire::qwen_xml::parse_calls(text)
    }

    /// Stop once the call is closed, so the model cannot run on and invent a
    /// result. `</tool_call>` is a single USER_DEFINED token in the Qwen 3.5
    /// vocabulary (id 248059), so the id path applies.
    fn stop_markers(&self) -> &[&'static str] {
        &[TOOL_CALL_CLOSE]
    }

    /// The string fallback for the marker above, used only when it does not
    /// resolve to one id. Unlike LFM2's CONTROL marker this one is USER_DEFINED,
    /// so it *does* survive into the decoded text and a check here can actually
    /// fire.
    ///
    /// `ends_with`, not `contains`: a reply quoting the tag while explaining the
    /// format has not finished a call.
    fn stops_generation(&self, text: &str) -> bool {
        text.trim_end().ends_with(TOOL_CALL_CLOSE)
    }

    /// ChatML's turn marker reaches the text on candle; see
    /// [`wire::strip_trailing_markers`].
    fn clean_reply(&self, text: &str) -> String {
        let s = wire::think::strip_think_blocks(text);
        wire::strip_trailing_markers(s.trim(), &["<|im_end|>"]).to_string()
    }

    /// Qwen 3.6's own GGUF template reads only a boolean `enable_thinking`
    /// (on unless explicitly set `false`) — no effort granularity beyond
    /// that. `Low` is the only level that turns it off.
    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        ReasoningParams {
            thinking: Some(effort != ReasoningEffort::Low),
            effort_text: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_qwen3_generation_matches_and_qwen2_does_not() {
        for arch in [
            "qwen3",
            "qwen3moe",
            "qwen3next",
            "qwen3vl",
            "qwen3vlmoe",
            "qwen35",
            "qwen35moe",
        ] {
            assert!(Qwen3.matches_arch(arch), "{arch}");
        }
        for arch in ["qwen", "qwen2", "qwen2moe", "qwen2vl"] {
            assert!(!Qwen3.matches_arch(arch), "{arch}");
        }
    }

    /// How Qwen tool calls actually reach gallium: the model wraps a JSON object
    /// in `<tool_call>` tags, and the balanced-span scan reads the object out of
    /// the middle. Documented by a test because it looks accidental — and is the
    /// reason this profile claims no native format.
    #[test]
    fn a_tool_call_tag_wrapped_json_object_is_read_by_the_prose_protocol() {
        let calls = Qwen3.tool_calls(
            "<think>The user wants a file.</think>\n\
             <tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"a.txt\"}}\n</tool_call>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
    }

    #[test]
    fn reasoning_is_not_part_of_the_reply() {
        assert_eq!(
            Qwen3.clean_reply("<think>Let me check.</think>\nThe answer is 42."),
            "The answer is 42."
        );
    }

    #[test]
    fn only_low_turns_thinking_off() {
        assert_eq!(
            Qwen3.reasoning_params(ReasoningEffort::Low).thinking,
            Some(false)
        );
        for effort in [
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            let params = Qwen3.reasoning_params(effort);
            assert_eq!(params.thinking, Some(true));
            assert_eq!(params.effort_text, None);
        }
    }
}

#[cfg(test)]
mod marker_tests {
    use super::*;

    /// Candle keeps ChatML's turn marker in the decoded text; llama.cpp stops on
    /// it as EOG and never does.
    #[test]
    fn a_trailing_turn_marker_does_not_reach_the_user() {
        assert_eq!(
            Qwen3.clean_reply("The answer is 42.<|im_end|>"),
            "The answer is 42."
        );
        assert_eq!(
            Qwen3.clean_reply("<think>hm</think>\nThe answer is 42.<|im_end|>"),
            "The answer is 42."
        );
        assert_eq!(Qwen3.clean_reply("The answer is 42."), "The answer is 42.");
    }
}

#[cfg(test)]
mod xml_tests {
    use super::*;

    /// The form the candle backend gets back, and the one this profile could not
    /// read before the parser was wired in.
    #[test]
    fn the_native_xml_form_parses() {
        let calls = Qwen3.tool_calls(
            "<tool_call>\n<function=Read>\n<parameter=file_path>a.txt</parameter>\n</function>\n</tool_call>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
    }

    /// What a live Qwen 3.5 actually emits on llama.cpp, where gallium asks for
    /// its own JSON protocol. Unchanged by wiring the XML parser: there is no
    /// `<function=` to find, so the fallback runs exactly as before.
    #[test]
    fn the_json_reply_a_live_model_gives_is_unaffected() {
        let calls = Qwen3.tool_calls(
            r#"{"name": "Read", "arguments": {"file_path": "codeword.txt"}}"#,
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "codeword.txt");
    }

    /// Reasoning is stripped before either shape is looked for, so a model
    /// describing the syntax has not called anything.
    #[test]
    fn xml_inside_reasoning_is_not_a_call() {
        let calls = Qwen3.tool_calls(
            "<think>I could write <function=Bash><parameter=command>rm -rf /</parameter>\
             </function> but I won't.</think>\nNothing to do.",
            &[],
        );
        assert!(calls.is_empty(), "{calls:?}");
    }
}

#[cfg(test)]
mod stop_marker_tests {
    use super::*;

    /// The marker and its string fallback must be the same literal — an engine
    /// resolving one to a token id is replacing the other, not answering a
    /// different question. Same invariant `gemma4.rs` pins.
    #[test]
    fn stop_markers_match_stops_generation() {
        assert_eq!(Qwen3.stop_markers(), &[TOOL_CALL_CLOSE]);
        assert!(Qwen3.stops_generation("<tool_call>\n<function=Read>…</function>\n</tool_call>"));
        assert!(Qwen3.stops_generation("…</tool_call>\n"));
    }

    /// `ends_with`, so a reply that merely quotes the tag while explaining the
    /// format is not a finished call.
    #[test]
    fn quoting_the_tag_mid_reply_is_not_a_boundary() {
        assert!(!Qwen3.stops_generation("You close a call with </tool_call> at the end."));
        assert!(!Qwen3.stops_generation("The answer is 42."));
    }
}
