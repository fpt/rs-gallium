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
/// template rendering at all: the model answers a write request with
/// `{"Write": {"file_path": …, "content": …}}`, a shape
/// [`super::wire::json`] does not accept (it looks for `name`/`arguments`, or
/// `function`, or `tool_calls`), so the call is read as a text reply and printed
/// to the user. Its `content` also carries `\\n` where `\n` was meant, so
/// accepting the shape is likely necessary but not sufficient.
pub struct Lfm2;

/// The call markers. CONTROL tokens, so whether they reach a parser is an
/// *engine* question: llama.cpp decodes with `special=false` and drops them,
/// candle keeps them.
const CALL_START: &str = "<|tool_call_start|>";
const CALL_END: &str = "<|tool_call_end|>";

impl ModelProfile for Lfm2 {
    fn name(&self) -> &'static str {
        "lfm2"
    }

    fn matches_arch(&self, arch: &str) -> bool {
        arch.starts_with("lfm2")
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

    /// ChatML's turn marker reaches the text on candle; see
    /// [`wire::strip_trailing_markers`].
    fn clean_reply(&self, text: &str) -> String {
        let s = wire::think::strip_think_blocks(text);
        wire::strip_trailing_markers(s.trim(), &["<|im_end|>"]).to_string()
    }

    // Deliberately no `agent_preamble_suffix` override — tried and reverted,
    // not left unconsidered. A line telling the model to use the exact call
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
