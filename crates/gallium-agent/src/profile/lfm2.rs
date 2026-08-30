//! LFM2.5: a reasoning model on the prose tool protocol.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::ModelProfile;

/// LFM2.5 (`lfm2`, `lfm2moe`). Emits `<think>…</think>` before its answer, which
/// the generic reply cleaning handles. It **does** have a native tool format, but
/// claiming it buys nothing measurable (below), so this profile stays on the
/// prose protocol and is its identity plus that finding.
///
/// **Measured, not assumed** (LFM2.5-8B-A1B Q4_K_M, `arch = "lfm2moe"`, 7-case
/// testsuite run). Its template does declare tools — `{% if tools %}` injects
/// `List of tools: [<schemas>]` — and renders a call as
/// `<|tool_call_start|>[name(arg='v')]<|tool_call_end|>`. Claiming that as a
/// native format changes **nothing measurable**: the same 5 cases pass and the
/// same 2 fail either way, so this profile stays on the prose protocol, which is
/// the simpler of two equal options.
///
/// Two things are worth knowing before revisiting it.
///
/// All four `<|tool_*|>` markers are **control** tokens, and the sampler decodes
/// with `special=false`, so they never reach the parser — a native call arrives
/// as a bare `[Read(file_path="a.txt")]`. That is why
/// [`super::wire::python`] exists, and why every profile keeps it in
/// `fallback_calls`: dropping it would break this model outright.
///
/// And the format is Python-call syntax with no escaping, so an argument
/// containing `)` cannot survive it — `Write(content='…fmt.Println("hi")…')`
/// truncates at the first paren and the trailing text parses as a *second*,
/// invented call. Source code always contains parens, which caps how much
/// claiming the native format here could ever be worth.
///
/// The `coding` / `refactoring` failures are neither of those, and are not about
/// template rendering at all: this family drops the `{"name": …, "arguments": …}`
/// envelope and writes the call some other way, which
/// [`super::wire::json`] then has to recognize. **Which** other way turns out to
/// depend on the machine — same GGUF, same Q4_K_M blob, same fixed sampler seed,
/// and the two accelerators diverge:
///
/// | case | CUDA (RTX 4070) | Metal (M3) |
/// |---|---|---|
/// | `coding` | `{"file_path": …, "content": …}` — no name at all (#194) | `{"Write": {…}}` — name as the key |
/// | `refactoring` | (not observed) | `{"MultiEdit": [ {…} ]}` — name as the key, value the *unwrapped* array |
///
/// All three shapes are read today: the first by `args_match_unique_tool`, the
/// second and third by `keyed_by_tool_name` / `arguments_for`. `refactoring`
/// passes on Metal as of the third one.
///
/// `coding` still fails, and no wire change fixes it: the model over-escapes the
/// **newlines** in its code payload — `\\n` where `\n` was meant, so the `.go`
/// file gets a literal backslash-n and `go build` says
/// `invalid character U+005C '\'`. Quotes come through intact; it is the
/// newlines. Accepting the shape was necessary and is not sufficient (#118).
pub struct Lfm2;

/// The call markers. CONTROL tokens, so whether they reach a parser is an
/// *engine* question: llama.cpp decodes with `special=false` and drops them,
/// candle keeps them.
const CALL_START: &str = "<|tool_call_start|>";
const CALL_END: &str = "<|tool_call_end|>";

impl ModelProfile for Lfm2 {
    /// No, which is this family's own template's default:
    /// `{%- set preserve_thinking = preserve_thinking | default(false) -%}`,
    /// with the per-turn gate `loop.index0 > ns.last_user_index` deciding the
    /// rest.
    ///
    /// Stated rather than inherited for the reason in
    /// [`ModelProfile::preserve_prior_reasoning`]: the families disagree about
    /// this, and a difference that only exists in three separate jinja files is
    /// one nobody can review. Behaviour is unchanged.
    ///
    /// Worth knowing that this template was, until #182, never parsed at all —
    /// minijinja rejected its `{% generation %}` markers and every LFM2 prompt
    /// came from the manual ChatML fallback. So this default has only recently
    /// started applying to anything.
    fn preserve_prior_reasoning(&self) -> Option<bool> {
        Some(false)
    }

    fn name(&self) -> &'static str {
        "lfm2"
    }

    fn matches_arch(&self, arch: &str) -> bool {
        arch.starts_with("lfm2")
    }

    /// Yes: `{%- if tools -%}` puts a `List of tools: [<schema>, …]` block in the
    /// system message and `render_tool_calls` emits
    /// `<|tool_call_start|>[name(arg='v')]<|tool_call_end|>`, which is the format
    /// this model was fine-tuned on. So the llama.cpp backend renders through it
    /// rather than asking for gallium's JSON prose.
    ///
    /// This is the experiment `lfm2.rs` has been asking for since #182: the
    /// earlier "claiming the native format changes nothing" measurement was taken
    /// while the template could not parse at all (its `{% generation %}` markers
    /// broke minijinja), so both arms were the *same* prose fallback and the
    /// comparison could not have shown anything.
    ///
    /// Matching on `<|tool_call_start|>` rather than on `List of tools:`: the
    /// marker is this family's own token, while the phrase is ordinary words that
    /// another family's template could carry — the same distinction that made the
    /// arch pass outrank the template pass in `profile::detect`.
    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains(CALL_START)
    }

    /// Handles the case where the markers **did** survive, which is the candle
    /// backend's.
    ///
    /// This is not the same as claiming a native format — the payload inside the
    /// markers is the Python-ish call list [`wire::python`] already reads, and
    /// with the markers gone (llama.cpp) this returns nothing and the fallback
    /// reads the bare list exactly as before. What it fixes is narrow and would
    /// otherwise be silent: `wire::python` gates on the *whole reply* being a
    /// bracketed list, so a reply still wrapped in markers fails that gate and
    /// the call vanishes. Bounding to the region first restores it.
    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        match wire::tags::wrapper_body(text, CALL_START, CALL_END) {
            Some(inner) => wire::python::parse_calls(inner.trim()),
            None => Vec::new(),
        }
    }

    /// Stop once the call is closed, so the model cannot carry on past it and
    /// narrate a result it has not been given — the same reason Gemma 4 stops at
    /// its own closing marker.
    ///
    /// **This family can only be served by the id path.** `<|tool_call_end|>` is
    /// a CONTROL token, and the sampler decodes with `special=false`, so it never
    /// reaches the decoded text at all — which is why `stops_generation` is *not*
    /// overridden below: a string check for this marker could never fire, and
    /// writing one would look like cover that isn't there. If the marker fails to
    /// resolve to a single id the engine falls back to that predicate, which
    /// returns `false`, and behavior is exactly what it was before this method
    /// existed.
    fn stop_markers(&self) -> &[&'static str] {
        &[CALL_END]
    }

    /// Both call markers are CONTROL tokens the llama.cpp decode drops, and
    /// their text is the only thing that separates a native call from prose:
    /// without them a reply like `Let me look. [Glob(pattern='…')]` reaches the
    /// parser as prose ending in a bracketed list, which the python wire's
    /// whole-reply gate refuses — the turn ends as a text response with an
    /// unread call in it. Restored, [`Lfm2::parse_native_tool_calls`] bounds to
    /// the wrapped region on llama.cpp exactly as it does on candle, where the
    /// markers survive decoding on their own.
    fn restore_markers(&self) -> &[&'static str] {
        &[CALL_START, CALL_END]
    }

    /// ChatML's turn marker reaches the text on candle; see
    /// [`wire::strip_trailing_markers`].
    fn clean_reply(&self, text: &str) -> String {
        let s = wire::think::strip_think_blocks(text);
        wire::strip_trailing_markers(s.trim(), &["<|im_end|>"]).to_string()
    }

    /// Same shape as Qwen3's: the default think handling plus the trailing
    /// `<|im_end|>` trim its `clean_reply` applies, so the end-of-generation
    /// flush cannot release the marker whole. This family emits its own
    /// `<think>` opener, so no engine-side prefix is involved.
    fn stream_reply(&self, raw: &str) -> Option<String> {
        let s = wire::think::stream_visible(raw);
        Some(wire::strip_trailing_markers(&s, &["<|im_end|>"]).to_string())
    }

    // Deliberately no `agent_preamble_suffix` override — tried twice and
    // reverted twice, not left unconsidered.
    //
    // The second attempt was aimed at the escaping failure documented above: one
    // sentence saying to escape a newline as `\n` once, and that `\\n` writes a
    // backslash and an `n`. It made both cases **worse**, which is worth more
    // than the ineffective first attempt. `coding` went from mixed escaping to
    // uniformly double-escaped — the model read an instruction about escaping and
    // escaped *more* — and `refactoring` flipped PASS -> FAIL: it began sending
    // `edits` as a *string* holding a Python-ish list rather than an array, so
    // the edit applied nothing. Telling this model about a wire detail moves it
    // away from the format, not toward it.
    //
    // A third attempt, after this family moved onto its native tool format,
    // aimed at the one failure left in `refactoring` — the model rewrites
    // `counter.go` without its `import "fmt"`, so it no longer compiles. The
    // suffix said to build or run what you write and fix what the compiler
    // reports. `coding` held at 3/3; `refactoring` went to **0/3**, and not by
    // writing a worse file — by not writing one at all. The file came back
    // unmodified while the model observed and explained, which is the
    // over-explore-before-acting failure the trait docs name.
    //
    // Three suffixes now, three regressions, and the texts have nothing in
    // common — so the thing that costs this family is being opted into
    // `BASE_AGENT_PREAMBLE` at all (a suffix is the only way in; see the trait
    // docs). Its "inspect relevant state before changing it" and "verify the
    // result" clauses are exactly what a reasoning model needs least. Treat
    // `None` here as measured, and do not spend a fourth round on wording.
    //
    // The first attempt, for the record: A line telling the model to use the exact call
    // syntax instead of `{"Write": {…}}` (this struct's own documented
    // `coding`/`refactoring` failure) was tried via `verify-preamble` against
    // `lfm2`: `coding`/`refactoring` still fail identically with it present
    // (3 runs each condition, no pass/fail flip either way) — the model still
    // answers with a bare `{` and stops. Unlike Gemma4's reverted suffix this
    // wasn't harmful, just ineffective: it doesn't earn a place per this
    // trait's own bar (an *observed, measured* correction, not a guess), so
    // it's left unset rather than kept on the strength of what it was
    // supposed to do.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_lfm2_architectures_match() {
        for arch in ["lfm2", "lfm2moe"] {
            assert!(Lfm2.matches_arch(arch), "{arch}");
        }
        assert!(!Lfm2.matches_arch("llama"));
    }

    #[test]
    fn a_reasoning_block_is_not_part_of_the_reply() {
        assert_eq!(
            Lfm2.clean_reply("<think>Working it out.</think>\nThe answer."),
            "The answer."
        );
    }

    /// The reply that ended a real turn unread: prose, then the native call —
    /// whose control markers `special=false` decoding had dropped. Bare, the
    /// python wire's whole-reply gate refuses it (correctly — that gate is what
    /// keeps `name(...)` in documentation from becoming a phantom call). With
    /// the markers restored (`restore_markers`, which `llm_local` resolves to
    /// token ids and puts back), the native parser bounds to the wrapped region
    /// and reads the call, prose and all — the same reply candle sees natively.
    #[test]
    fn a_native_call_after_prose_needs_its_markers() {
        let bare = "I'll investigate the MCP implementation.\n\
                    [Glob(pattern='**/*tool_manager.go')]";
        assert!(
            Lfm2.tool_calls(bare, &[]).is_empty(),
            "without markers this is prose ending in a bracketed list"
        );

        let restored = format!(
            "I'll investigate the MCP implementation.\n\
             {CALL_START}[Glob(pattern='**/*tool_manager.go')]{CALL_END}"
        );
        let calls = Lfm2.tool_calls(&restored, &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Glob");
        assert_eq!(calls[0].arguments["pattern"], "**/*tool_manager.go");
    }

    #[test]
    fn the_markers_the_engine_restores_are_the_call_wrapper() {
        assert_eq!(Lfm2.restore_markers(), &[CALL_START, CALL_END]);
    }

    #[test]
    fn a_call_in_the_prose_protocol_parses_after_the_reasoning() {
        let calls = Lfm2.tool_calls(
            "<think>I should use {Read} here.</think>\n\
             {\"name\": \"Read\", \"arguments\": {\"file_path\": \"a.txt\"}}",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn no_agent_preamble() {
        assert!(Lfm2.agent_preamble().is_none());
    }
}

/// Cover for the engine difference: the same model's reply reaches these parsers
/// with markers on candle and without them on llama.cpp.
#[cfg(test)]
mod marker_tests {
    use super::*;

    /// The candle shape. Before the region was bounded this returned **nothing**:
    /// `wire::python` requires the whole reply to be a bracketed list, and a
    /// marker-wrapped one is not — so the call disappeared silently.
    #[test]
    fn a_marker_wrapped_call_parses() {
        let calls = Lfm2.tool_calls(
            "<|tool_call_start|>[Read(file_path=\"a.txt\")]<|tool_call_end|>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
    }

    /// The llama.cpp shape, where the markers were already dropped by
    /// `special=false`. Unchanged: the fallback reads the bare list.
    #[test]
    fn a_bare_call_list_still_parses() {
        let calls = Lfm2.tool_calls("[Read(file_path=\"a.txt\")]", &[]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn a_trailing_turn_marker_does_not_reach_the_user() {
        assert_eq!(
            Lfm2.clean_reply("The answer is 42.<|im_end|>"),
            "The answer is 42."
        );
        assert_eq!(Lfm2.clean_reply("The answer is 42."), "The answer is 42.");
    }
}

#[cfg(test)]
mod stop_marker_tests {
    use super::*;

    /// The marker is the same literal the parser closes a region on, so the two
    /// cannot drift apart.
    #[test]
    fn the_stop_marker_is_the_regions_closing_tag() {
        assert_eq!(Lfm2.stop_markers(), &[CALL_END]);
    }

    /// Deliberately no string fallback: `<|tool_call_end|>` is a CONTROL token
    /// that `special=false` drops, so a decoded-text check could never fire. This
    /// pins the *absence* — if someone adds one later, it will look like cover
    /// that does not exist.
    #[test]
    fn there_is_no_decoded_text_fallback_because_the_marker_never_reaches_text() {
        assert!(!Lfm2.stops_generation("[Read(file_path=\"a.txt\")]<|tool_call_end|>"));
        assert!(!Lfm2.stops_generation("The answer is 42."));
    }
}
