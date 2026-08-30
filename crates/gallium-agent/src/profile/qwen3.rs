//! Qwen 3.x: ChatML, `<think>`, and two shapes of tool call.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::{ModelProfile, ReasoningEffort, ReasoningParams};

/// Qwen 3.6 and its `qwen3*` siblings. A reasoning model on ChatML
/// (`<|im_start|>role`), which emits `<think>…</think>` before answering — the
/// generic reply cleaning is exactly right for it.
///
/// **Two shapes, both real, and now both live on llama.cpp too.**
/// `unsloth/Qwen3.5-9B-GGUF` (`arch = "qwen35"`) has `<function=` /
/// `<parameter=` in its embedded template and no `"name"` anywhere, so the
/// XML-parameter form is what that template renders — and now that
/// `template_formats_tools_natively` matches on `<function=`, `build_prompt`
/// renders it via `llm_local.rs`'s `render_native` rather than asking for
/// gallium's JSON-prose protocol instead. [`wire::qwen_xml`] (ported out of
/// `protocol.rs`, where it served the candle backend alone — that engine
/// always rendered the model's own prompt format and always got XML back)
/// now reads what llama.cpp gets back too.
///
/// The JSON-prose fallback ([`wire::json`]) is not dead: a `qwen3*` GGUF
/// whose template declares no tool support at all still gets it, and it is
/// what a native-template render failure (a template that raises on some
/// input `render_native` doesn't expect) falls back to.
///
/// Turning native rendering on was a live-model finding, not a preemptive
/// choice: Qwen3.8-27B, asked for JSON prose instead of the format its own
/// template (and, presumably, training) declares, was observed producing a
/// hybrid of the two — an unclosed `<tool_call>` JSON object trailing into a
/// duplicated `<tool_call>`/`</function>` fragment. See `Qwen3::template_formats_tools_natively`'s
/// own doc for the fix this is.
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
    ///
    /// Plus `qwen4exp`, which breaks the prefix: Qwen3.8-Flash-Next reports
    /// `model_type: qwen4_exp` and llama.cpp registers it as `qwen4exp`
    /// ([PR #27742], unmerged, so nothing can load it here yet). On the wire it
    /// is this family exactly — its published `chat_template.jinja` is
    /// **byte-identical** to Qwen3.8-27B's, which is what a profile is about.
    ///
    /// Matched exactly rather than by a second `qwen4` prefix, for the reason
    /// `qwen2*` is excluded: a sibling generation nobody has looked at should
    /// not inherit this family's parsers by accident. Detection already lands
    /// here without this line, through the template rescue in [`super::detect`],
    /// since `<function=` appears in no other profile's literals — but it lands
    /// with a log line saying the architecture was not recognized, which reads
    /// as a model nobody has checked.
    ///
    /// [PR #27742]: https://github.com/ggml-org/llama.cpp/pull/27742
    fn matches_arch(&self, arch: &str) -> bool {
        arch.starts_with("qwen3") || arch == "qwen4exp"
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
    ///
    /// **This is also what caps a turn at one tool call**, and it is kept
    /// deliberately rather than by omission. [`wire::qwen_xml`] now reads every
    /// `<function=>` block, because the family's own template renders one
    /// `<tool_call>` per call and the format is plural — but generation halts
    /// at the first close tag, so a second block is never sampled.
    ///
    /// Lifting it is a measurement, not an edit: the guard exists because a
    /// model that runs past its own call invents the tool's result, and whether
    /// this family does that is a question about a specific model on specific
    /// hardware. Until someone runs it, one call per iteration is the safe
    /// answer — the loop simply takes another iteration, which costs a
    /// round trip and never a wrong result.
    fn stop_markers(&self) -> &[&'static str] {
        &[TOOL_CALL_CLOSE]
    }

    /// `<function=` is the literal that identifies this family's native XML
    /// tool-call format — see [`wire::qwen_xml`]'s module doc for the shape.
    /// Deliberately not gated any further (no `<tool_call>` co-check): the
    /// XML parser itself accepts a bare `<function=` block with no wrapper,
    /// so the two must agree on what counts as "native" here.
    ///
    /// Turned on after a live Qwen3.8-27B run produced a hybrid of this
    /// format and the JSON-prose one gallium had been asking for instead —
    /// an unclosed `<tool_call>` JSON object trailing into a duplicated
    /// `<tool_call>`/`</function>` fragment, consistent with a model reverting
    /// mid-generation to the format its own template (and, presumably,
    /// training) actually declares. Rendering that native format instead, via
    /// `llm_local.rs`'s existing `render_native` (already shared by
    /// GPT-OSS/Gemma4/MiniMax/DeepSeek — no new plumbing needed here), is the
    /// fix this tries: give the model the one format instead of the one it
    /// keeps drifting toward anyway.
    ///
    /// A qwen35-family template with no tool support at all (rare, but the
    /// reason this checks the template rather than assuming) simply never
    /// matches, and `build_prompt` falls back to JSON-prose exactly as before.
    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains("<function=")
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

    /// The default think handling, plus the same trailing `<|im_end|>` trim as
    /// [`Qwen3::clean_reply`] — without it the marker, held back while forming
    /// (it starts with `<`), would be released whole by the end-of-generation
    /// flush. This family's template pre-fills `<think>\n` when thinking is on,
    /// so the raw text often opens mid-reasoning with no `<think>` in it (#233)
    /// — the engine prepends that dangling opener before calling this; see
    /// `streaming::prompt_prefills_thinking`.
    fn stream_reply(&self, raw: &str) -> Option<String> {
        let s = wire::think::stream_visible(raw);
        Some(wire::strip_trailing_markers(&s, &["<|im_end|>"]).to_string())
    }

    /// Yes — the opposite of Gemma 4 and LFM2.5, and deliberately.
    ///
    /// This family's template preserves by default:
    /// `{%- if preserve_thinking is undefined or preserve_thinking is true or
    /// loop.index0 > ns.last_query_index %}`. The first clause short-circuits
    /// whenever nobody sets the variable, so every prior assistant turn keeps
    /// its `<think>` block. That is Qwen's choice in the file their model ships
    /// with, and this states it rather than changing it.
    ///
    /// It is not free: at this family's default `xhigh` effort the think blocks
    /// are long, and a ten-turn conversation replays all of them. Whether
    /// gallium should overrule a vendor here to buy back context is a real
    /// question and a separate one — the value of writing the answer down is
    /// that the question is now visible instead of buried in a jinja
    /// conditional.
    ///
    /// Pinning it also matters more for this family than for the others,
    /// because the difference is load-bearing: the same conversation gets a
    /// materially different prompt on Qwen than on Gemma, and until now nothing
    /// in gallium said so.
    fn preserve_prior_reasoning(&self) -> Option<bool> {
        Some(true)
    }

    /// Gallium's five-point scale projected onto this family's four states.
    ///
    /// | [`ReasoningEffort`] | this family |
    /// |---|---|
    /// | `Low` | thinking off |
    /// | `Medium` | `low` |
    /// | `High` | `medium` |
    /// | `XHigh`, `Max` | `xhigh` |
    ///
    /// **The clamp is the point, not a detail.** Qwen3.8's template accepts
    /// exactly `xhigh`, `medium` and `low`, and `raise_exception`s on anything
    /// else — so the obvious mapping, each variant to its own lowercase name,
    /// would emit `high` and `max` into a template that refuses them. That
    /// failure is not visible: `build_prompt` catches the raise and asks the
    /// model for JSON prose instead, which is a different wire protocol
    /// arriving with no error anyone sees. `configs/default.toml` ships
    /// `reasoningEffort = "high"`, so this is the value a copied config
    /// actually produces.
    ///
    /// The shift in the middle (gallium's `High` → this family's `medium`) is
    /// what four states cost when the caller has five: the projection stays
    /// monotone, which is the property a scale has to keep, and every level is
    /// reachable. Nothing here can be reached by a value that raises.
    ///
    /// `Low` stays thinking-off rather than becoming the template's own `low`
    /// effort, which would read more naturally and would delete gallium's only
    /// way to turn reasoning off on this family — a real lever on a model whose
    /// template defaults to `xhigh` and whose decode is the expensive half of a
    /// turn. The template's `low` is `Medium` instead.
    ///
    /// Qwen 3.6's older template has no `reasoning_effort` variable at all, so
    /// it reads the `thinking` half and ignores the rest; minijinja drops
    /// unused context keys.
    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        let effort_text = match effort {
            ReasoningEffort::Low => None,
            ReasoningEffort::Medium => Some("low"),
            ReasoningEffort::High => Some("medium"),
            ReasoningEffort::XHigh | ReasoningEffort::Max => Some("xhigh"),
        };
        ReasoningParams {
            thinking: Some(effort != ReasoningEffort::Low),
            effort_text,
            preserve_thinking: None,
        }
    }

    /// Verified via the `verify-preamble` skill against `qwen3.8` — see that
    /// run's summary for the before/after testsuite comparison this line is
    /// based on, the same evidence bar `BASE_AGENT_PREAMBLE`'s doc comment
    /// asks for.
    fn agent_preamble_suffix(&self) -> Option<&'static str> {
        Some("Prefer the smallest change consistent with the existing design.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #233's crash shape: the template pre-filled `<think>\n`, the engine
    /// prepends the opener, and `stream_reply` holds the whole reasoning —
    /// where `clean_reply` on the same prefixes showed it as prose and then
    /// collapsed it to `""`, which is the non-monotonicity that panicked.
    #[test]
    fn stream_reply_holds_prefilled_reasoning_until_it_closes() {
        // What the engine hands stream_reply: "<think>" + the model's output.
        let mid_thought = "<think>Okay, the user wants the capital of France, that is";
        assert_eq!(Qwen3.stream_reply(mid_thought).as_deref(), Some(""));
        let closed = format!("{mid_thought} Paris.\n</think>\n\nParis.<|im_end|>");
        assert_eq!(Qwen3.stream_reply(&closed).as_deref(), Some("Paris."));
    }

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
            // Qwen3.8-Flash-Next, which breaks the prefix but not the wire
            // format — same chat template, byte for byte.
            "qwen4exp",
        ] {
            assert!(Qwen3.matches_arch(arch), "{arch}");
        }
        // `qwen4exp` is matched exactly, so a sibling generation nobody has
        // looked at does not inherit this family's parsers by prefix.
        for arch in ["qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen4", "qwen4moe"] {
            assert!(!Qwen3.matches_arch(arch), "{arch}");
        }
    }

    /// A `<tool_call>`-wrapped JSON object — what a model still answers with when
    /// asked for gallium's JSON-prose protocol (the fallback for a template
    /// `template_formats_tools_natively` doesn't match, or a native render that
    /// failed) — reaches gallium via the balanced-span scan, not
    /// `parse_native_tool_calls`. Documented by a test because it looks
    /// accidental: `wire::qwen_xml`'s own module doc explicitly leaves this shape
    /// to `wire::json` rather than reading it out of the same tags itself.
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

    /// `Low` is off and nothing else is — pinned separately from the effort
    /// projection below, because the thinking toggle is the half Qwen 3.6's
    /// older template reads and it must keep working there whatever the newer
    /// one grows.
    ///
    /// It stays off rather than becoming the template's own `low` effort: that
    /// reads more naturally and would delete gallium's only way to turn
    /// reasoning off on this family. The template's `low` is `Medium`.
    #[test]
    fn only_low_turns_thinking_off() {
        let low = Qwen3.reasoning_params(ReasoningEffort::Low);
        assert_eq!(low.thinking, Some(false));
        assert_eq!(low.effort_text, None, "off is off, not off-at-some-effort");

        for effort in [
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            assert_eq!(Qwen3.reasoning_params(effort).thinking, Some(true));
        }
    }

    /// See `profile::tests::agent_preamble_is_named_by_exactly_the_families_that_have_one`
    /// for the pin on *which* families have one; this checks the composition
    /// itself carries both layers, the same regression `gpt_oss.rs`'s
    /// equivalent test guards.
    #[test]
    fn the_preamble_carries_both_the_base_contract_and_the_suffix() {
        let preamble = Qwen3.agent_preamble().expect("has a preamble");
        assert!(preamble.contains(super::super::BASE_AGENT_PREAMBLE));
        assert!(preamble.contains("smallest change"));
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
mod native_template_tests {
    use super::*;

    /// The literal transcribed from a live Qwen3.8-27B GGUF's own embedded
    /// template (the "If you choose to call a function..." instruction block).
    #[test]
    fn a_template_with_function_syntax_is_native() {
        assert!(Qwen3.template_formats_tools_natively(
            "If you choose to call a function ONLY reply in the following format \
             with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
             <parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n\
             </tool_call>"
        ));
    }

    /// A template with no tool syntax at all falls back to JSON-prose, same as
    /// every other family without an override.
    #[test]
    fn a_template_without_function_syntax_is_not_native() {
        assert!(!Qwen3.template_formats_tools_natively(
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>"
        ));
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

#[cfg(test)]
mod reasoning_tests {
    use super::*;

    /// The values Qwen3.8's template accepts, transcribed from its own guard:
    /// `if resolved_reasoning_effort not in ('xhigh', 'medium', 'low')`.
    /// Anything else raises, and a raise is a silent protocol switch.
    const ACCEPTED: &[&str] = &["xhigh", "medium", "low"];

    #[test]
    fn no_effort_maps_to_a_value_the_template_refuses() {
        for effort in [
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            if let Some(text) = Qwen3.reasoning_params(effort).effort_text {
                assert!(
                    ACCEPTED.contains(&text),
                    "{effort:?} maps to {text:?}, which this family's template raises on"
                );
            }
        }
    }

    /// Monotone, and every one of the template's levels reachable — the two
    /// properties that make this a projection of a scale rather than an
    /// arbitrary table.
    #[test]
    fn the_projection_is_monotone_and_onto() {
        let rank = |p: ReasoningParams| match (p.thinking, p.effort_text) {
            (Some(false), _) => 0,
            (_, Some("low")) => 1,
            (_, Some("medium")) => 2,
            (_, Some("xhigh")) => 3,
            other => panic!("unexpected params {other:?}"),
        };

        let ladder = [
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ]
        .map(|e| rank(Qwen3.reasoning_params(e)));

        assert!(
            ladder.windows(2).all(|w| w[0] <= w[1]),
            "asking for more reasoning must never ask for less: {ladder:?}"
        );
        assert_eq!(*ladder.iter().min().unwrap(), 0, "thinking-off unreachable");
        assert_eq!(*ladder.iter().max().unwrap(), 3, "xhigh unreachable");
        for level in 0..=3 {
            assert!(ladder.contains(&level), "no effort reaches level {level}");
        }
    }
}
