//! GPT-OSS: the Harmony protocol.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::{ModelProfile, ReasoningEffort, ReasoningParams};

/// GPT-OSS (120b; also 20b's wire format — see [`GptOss20b`] for why the
/// preamble splits off), which writes tool calls and its reply in
/// [Harmony](https://github.com/openai/harmony): named channels, and
/// `to=functions.NAME<|channel|>commentary<|constrain|>json<|message|>{…}<|call|>`.
pub struct GptOss;

impl ModelProfile for GptOss {
    fn name(&self) -> &'static str {
        "gpt-oss"
    }

    /// llama.cpp calls this architecture "gpt-oss" (LLM_ARCH_OPENAI_MOE);
    /// safetensors `model_type` spells it with an underscore, which is what the
    /// candle backend will hand over.
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(arch, "gpt-oss" | "gpt_oss" | "gptoss")
    }

    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        wire::harmony_calls(text)
    }

    /// Harmony names its answer precisely — `<|channel|>final<|message|>` to the
    /// next `<|end|>`/`<|return|>` — so when a final channel is present it is
    /// authoritative and nothing else needs consulting. A reply that never
    /// opened one (the model stopped inside `analysis`) falls through to the
    /// generic strip, which leaves it as-is rather than guessing at a boundary
    /// Harmony did not draw.
    fn clean_reply(&self, text: &str) -> String {
        match crate::harmony::extract_final(text) {
            Some(final_text) => final_text,
            None => wire::think::strip_think_blocks(text).trim().to_string(),
        }
    }

    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains(HARMONY_CHANNEL)
    }

    /// First family opted into `BASE_AGENT_PREAMBLE` — see
    /// `docs/adr/0003-model-profiles.md` and the eval-improve skill for how
    /// this is meant to be measured before another family gets a suffix of
    /// its own: run `testsuite/runner.sh` against `configs/gpt-oss*.toml`
    /// with and without this line and compare turn count, tool-call count,
    /// and pass/fail per case, rather than assuming a plan reminder helps.
    ///
    /// Verified on 120b (7/7 pass at both settings, no material call-count
    /// change — see gpt-oss-120b.toml's own comment). The **same suffix
    /// text does not carry to 20b** — `verify-preamble` against that
    /// checkpoint found it took `coding` from 3 model calls to 19 and turned
    /// a passing `refactoring` into a failing one, which is why 20b gets its
    /// own profile ([`GptOss20b`]) rather than sharing this one: the
    /// preamble's effect is per-checkpoint, not per-family, exactly the
    /// "each family earns its own suffix" rule this method's own evidence
    /// bar already states.
    fn agent_preamble_suffix(&self) -> Option<&'static str> {
        Some("For multi-step work, maintain a concise plan and revise it as new evidence arrives.")
    }

    /// GPT-OSS's own GGUF template reads `reasoning_effort` as a free string
    /// and injects it verbatim as `"Reasoning: " + reasoning_effort`
    /// (defaulting to `"medium"` when the key is absent) — no boolean
    /// toggle, since Harmony always reasons. The Harmony spec defines
    /// nothing above `"high"`, so XHigh/Max clamp to it rather than sending
    /// a string the model was never tuned to recognize.
    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        let effort_text = match effort {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => "high",
        };
        ReasoningParams {
            thinking: None,
            effort_text: Some(effort_text),
            preserve_thinking: None,
        }
    }
}

/// Harmony's channel delimiter. Note the pipes on **both** sides — Gemma 4's
/// `<|channel>thought`/`<channel|>` markers are a different format that this
/// literal deliberately does not match.
const HARMONY_CHANNEL: &str = "<|channel|>";

/// GPT-OSS 20b: the same Harmony wire format as [`GptOss`] — tool parsing,
/// reply cleaning, and reasoning-effort mapping all delegate to it unchanged
/// — but opted **out** of the agent preamble, per `GptOss::agent_preamble_suffix`'s
/// own doc comment on why the two checkpoints don't share one answer.
///
/// A GGUF's metadata cannot tell a 20b checkpoint from a 120b one — both
/// report the same `general.architecture` and the same Harmony chat
/// template — so this profile can only be reached by explicit name
/// (`[llm] profile = "gpt-oss-20b"`, already set in `configs/gpt-oss-20b.toml`);
/// `matches_arch`/`matches_template` both answer `false` so detection never
/// picks it for a 120b model that happens to load through an unrecognized
/// arch string.
pub struct GptOss20b;

impl ModelProfile for GptOss20b {
    fn name(&self) -> &'static str {
        "gpt-oss-20b"
    }

    fn matches_arch(&self, _arch: &str) -> bool {
        false
    }

    fn matches_template(&self, _template: &str) -> bool {
        false
    }

    fn parse_native_tool_calls(&self, text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        GptOss.parse_native_tool_calls(text, tools)
    }

    fn clean_reply(&self, text: &str) -> String {
        GptOss.clean_reply(text)
    }

    fn template_formats_tools_natively(&self, template: &str) -> bool {
        GptOss.template_formats_tools_natively(template)
    }

    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        GptOss.reasoning_params(effort)
    }

    // agent_preamble_suffix: default `None` — the whole point of this profile.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_from_either_spelling_of_the_architecture() {
        for arch in ["gpt-oss", "gpt_oss", "gptoss"] {
            assert!(GptOss.matches_arch(arch), "{arch}");
        }
        assert!(!GptOss.matches_arch("seed_oss"));
    }

    #[test]
    fn parses_a_harmony_call_through_the_functions_namespace() {
        let calls = GptOss.tool_calls(
            "<|start|>assistant to=functions.Glob<|channel|>commentary <|constrain|>json\
             <|message|>{\"pattern\":\"crates/*\"}<|call|>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Glob");
        assert_eq!(calls[0].arguments["pattern"], "crates/*");
    }

    /// Regression cover for 26d0f80 from the profile side: `to=` inside argument
    /// *content* is not a call boundary. The bug it guards is now unreachable
    /// from any other family, since only this profile runs the Harmony parser.
    #[test]
    fn a_stray_to_marker_inside_arguments_is_not_a_second_call() {
        let calls = GptOss.tool_calls(
            "<|start|>assistant to=functions.Write<|channel|>commentary <|constrain|>json\
             <|message|>{\"content\":\"set to=foo in the docs\",\"file_path\":\"a.txt\"}<|call|>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "set to=foo in the docs");
    }

    #[test]
    fn the_final_channel_is_the_reply() {
        let raw = "<|channel|>analysis<|message|>Thinking.<|end|>\
                   <|start|>assistant<|channel|>final<|message|>42.<|end|>";
        assert_eq!(GptOss.clean_reply(raw), "42.");
    }

    /// No `final` channel opened — the model stopped inside its reasoning. There
    /// is no answer to extract, so nothing is invented; Gemma's
    /// "everything after the last marker" heuristic would have cut here and
    /// returned reasoning as if it were the reply.
    #[test]
    fn a_reply_with_no_final_channel_is_not_guessed_at() {
        let raw = "<|channel|>analysis<|message|>Still working through it.";
        assert_eq!(GptOss.clean_reply(raw), raw);
    }

    #[test]
    fn generation_is_not_stopped_by_another_familys_marker() {
        assert!(!GptOss.stops_generation("<|tool_call>call:read{}<tool_call|>"));
    }

    /// The composed preamble carries gallium's shared agent contract plus
    /// this family's own suffix, not one or the other — a regression this
    /// pins because `ModelProfile::agent_preamble`'s default composition is
    /// exactly the thing a profile could accidentally bypass by overriding
    /// the wrong method.
    #[test]
    fn the_preamble_carries_both_the_base_contract_and_the_suffix() {
        let preamble = GptOss.agent_preamble().expect("has a preamble");
        assert!(preamble.contains(super::super::BASE_AGENT_PREAMBLE));
        assert!(preamble.contains("maintain a concise plan"));
    }

    #[test]
    fn reasoning_effort_is_a_free_string_clamped_above_high() {
        let params = GptOss.reasoning_params(ReasoningEffort::Medium);
        assert_eq!(params.thinking, None);
        assert_eq!(params.effort_text, Some("medium"));

        for effort in [
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            assert_eq!(GptOss.reasoning_params(effort).effort_text, Some("high"));
        }
    }
}
