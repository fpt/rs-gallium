//! DeepSeek-V4: the DSML tool-call protocol.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::{ModelProfile, ReasoningEffort, ReasoningParams};

/// DeepSeek-V4-Flash (issue #116), which writes calls as
/// `<｜DSML｜tool_calls><｜DSML｜invoke name="…">` with a fullwidth-pipe
/// delimiter and a `string="true|false"` attribute naming each parameter's type.
pub struct DeepSeekV4;

impl ModelProfile for DeepSeekV4 {
    fn name(&self) -> &'static str {
        "deepseek-v4"
    }

    /// Exactly `deepseek4`. llama.cpp also knows `deepseek`, `deepseek2`,
    /// `deepseek2-ocr` and `deepseek32`, none of which emit DSML — matching the
    /// family by prefix would hand a V2 checkpoint a V4 parser.
    fn matches_arch(&self, arch: &str) -> bool {
        arch == "deepseek4"
    }

    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        // No `tools` needed, unlike MiniMax's otherwise-identical shape: each
        // parameter names its own type on the wire.
        wire::dsml::parse_calls(text)
    }

    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains(wire::dsml::WRAPPER_OPEN)
    }

    /// `configs/deepseek-v4-flash.toml`'s UD-IQ1_S quant dropped its DSML
    /// block and narrated a markdown bash fence instead on 4 of 4 runs at
    /// temperature 0.6 (see the test above). The prose fallback is why that
    /// run was still readable, not why this exists — the fallback stays for
    /// whatever this preamble doesn't catch, the same way it stays for every
    /// other family.
    fn agent_preamble_suffix(&self) -> Option<&'static str> {
        Some(
            "Always write tool calls using your own DSML format \
             (<｜DSML｜tool_calls>…</｜DSML｜tool_calls>). Never narrate a tool call as \
             a markdown code fence or as plain prose describing what you would run.",
        )
    }

    /// DeepSeek-V4's own GGUF template reads two variables: `thinking`
    /// (bool — reasoning happens at all only when true) and
    /// `reasoning_effort` (string, only `"high"`/`"max"` have defined
    /// instruction text; the template appends nothing extra for any other
    /// value, including absence). `Low` turns thinking off entirely — the
    /// fastest option for a family with an explicit on/off switch, unlike
    /// GPT-OSS/Qwen/Gemma4 which have no such switch to turn off.
    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        match effort {
            ReasoningEffort::Low => ReasoningParams {
                thinking: Some(false),
                effort_text: None,
                preserve_thinking: None,
            },
            ReasoningEffort::Medium => ReasoningParams {
                thinking: Some(true),
                effort_text: None,
                preserve_thinking: None,
            },
            ReasoningEffort::High => ReasoningParams {
                thinking: Some(true),
                effort_text: Some("high"),
                preserve_thinking: None,
            },
            ReasoningEffort::XHigh | ReasoningEffort::Max => ReasoningParams {
                thinking: Some(true),
                effort_text: Some("max"),
                preserve_thinking: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_v4_is_claimed() {
        assert!(DeepSeekV4.matches_arch("deepseek4"));
        for arch in ["deepseek", "deepseek2", "deepseek2-ocr", "deepseek32"] {
            assert!(!DeepSeekV4.matches_arch(arch), "{arch}");
        }
    }

    #[test]
    fn each_parameter_names_its_own_type() {
        let calls = DeepSeekV4.tool_calls(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"grep\">\n\
             <｜DSML｜parameter name=\"pattern\" string=\"true\">50</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"limit\" string=\"false\">50</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "50");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    /// `configs/deepseek-v4-flash.toml` records the UD-IQ1_S quant dropping its
    /// DSML block and narrating a markdown bash fence instead on 4 of 4 runs at
    /// temperature 0.6. When it *does* comply with gallium's prose instructions
    /// instead of its training, that reply still has to be read — which is why a
    /// named profile keeps the fallback rather than being strict.
    #[test]
    fn a_reply_in_the_prose_protocol_is_still_read() {
        let calls = DeepSeekV4.tool_calls(
            r#"{"name": "Read", "arguments": {"file_path": "a.txt"}}"#,
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    /// The preamble names the exact wrapper `wire::dsml::parse_calls` reads,
    /// so a future rename of the format can't drift the two apart silently.
    #[test]
    fn the_preamble_names_the_wire_format_it_asks_for() {
        let preamble = DeepSeekV4.agent_preamble().expect("has a preamble");
        assert!(preamble.contains(wire::dsml::WRAPPER_OPEN));
    }

    #[test]
    fn low_turns_thinking_off_and_only_high_and_max_have_effort_text() {
        assert_eq!(
            DeepSeekV4.reasoning_params(ReasoningEffort::Low).thinking,
            Some(false)
        );
        let medium = DeepSeekV4.reasoning_params(ReasoningEffort::Medium);
        assert_eq!(medium.thinking, Some(true));
        assert_eq!(medium.effort_text, None);
        assert_eq!(
            DeepSeekV4
                .reasoning_params(ReasoningEffort::High)
                .effort_text,
            Some("high")
        );
        for effort in [ReasoningEffort::XHigh, ReasoningEffort::Max] {
            assert_eq!(DeepSeekV4.reasoning_params(effort).effort_text, Some("max"));
        }
    }
}
