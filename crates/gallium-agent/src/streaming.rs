//! Turning a growing raw generation into a stream of answer fragments that are
//! safe to show a user — reasoning and tool-call syntax removed.
//!
//! Both local backends feed this the same way: after each decoded token, hand
//! it `profile.stream_reply(raw_so_far)` — the family's own prefix-monotonic
//! statement of what may stream (see that method's contract; `None` means the
//! protocol has not decided yet) — and forward whatever it returns to
//! `AgentEvent::MessageDelta`. This filter additionally **freezes** — stops
//! streaming for the rest of that model call — the instant the visible text
//! starts to look like a tool call the wire layer hasn't parsed yet, and holds
//! back a trailing `<`/`[` run that might be a marker forming. The turn's final
//! message (`profile.clean_reply`, authoritative) always supersedes what the
//! fragments accumulate to, so freezing or lagging costs only the progressive
//! render of one call.

/// Whether the rendered prompt ends with a *dangling* `<think>` opener — a
/// template that pre-fills the start of the model's reasoning into the prompt
/// (Qwen3.8 with thinking on, MiniMax-M2.7 always), so the model's own output
/// opens mid-thought and carries only the closing `</think>`.
///
/// `ModelProfile::stream_reply` sees only the model's own output and cannot
/// tell such reasoning from an answer until the closer lands — which is how
/// #233 streamed 54 characters of Qwen3 reasoning and then froze on the
/// collapse. Only the engine knows what its prompt ended with, so the engine
/// checks here and, when true, prepends `"<think>"` to the raw text it hands
/// `stream_reply` — making the pre-filled case read exactly like the
/// model-emitted one. A prompt whose pre-filled block is already *closed*
/// (thinking off: `<think>\n\n</think>\n\n`) ends with `</think>` and is
/// correctly left alone.
pub(crate) fn prompt_prefills_thinking(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<think>")
}

/// Substrings that mean the model is emitting *protocol*, not answer — a
/// tool-call opener from any family, or Harmony's non-final channels.
const FREEZE_MARKERS: &[&str] = &[
    "<tool_call",
    "</tool_call",
    "<|tool_call",
    "<|tool_calls",
    "<|tool>",
    "<｜tool",
    "tool▁calls",
    "[TOOL_CALLS",
    "<|tool_call_start|>",
    "<|python",
    "<function=",
    "<|channel|>analysis",
    "<|channel|>commentary",
];

/// How far back from the end of the visible text to look for a `<` or `[` that
/// might be a marker forming — long enough to cover the longest opener above.
const LOOKBACK: usize = 24;

/// LFM2's `[Name(arg='v')]` calls reach this with no wrapping marker on the
/// llama.cpp path (its `<|tool_call_start|>` is a control token dropped at
/// `special=false`), so `FREEZE_MARKERS` can't catch them — recognise the shape
/// instead: `[` then an upper-case identifier then `(`.
fn has_python_call_opener(s: &str) -> bool {
    let b = s.as_bytes();
    let mut from = 0;
    while let Some(rel) = s[from..].find('[') {
        let after = from + rel + 1; // `[` is ASCII, so this is a char boundary
        let rest = &b[after..];
        let name = rest
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric() || **c == b'_')
            .count();
        if name > 0 && rest[0].is_ascii_uppercase() && rest.get(name) == Some(&b'(') {
            return true;
        }
        from = after;
    }
    false
}

/// Incremental filter over the visible answer. One per model call.
#[derive(Default)]
pub(crate) struct StreamingReply {
    /// Bytes of the visible text already handed out.
    emitted: usize,
    /// The exact text handed out so far. `profile.clean_reply` is *not*
    /// contractually monotonic: a Qwen3 `<think>` block that streamed as prose
    /// collapses to `""` the instant `</think>` lands, and a Harmony `final`
    /// channel supersedes visible `analysis` text. Each call checks the new
    /// `visible` still begins with this before extending it — a shrink or a
    /// rewritten prefix freezes the stream instead of panicking or emitting
    /// across the seam.
    emitted_text: String,
    /// Once protocol syntax has appeared this call, stop for the rest of it.
    pub(crate) frozen: bool,
}

impl StreamingReply {
    /// `visible` is what `profile.stream_reply(raw_so_far)` returned (a `None`
    /// from there never reaches here — it means hold everything). Returns the
    /// newly safe-to-show suffix, or `None` to hold. `done` (generation
    /// finished) releases the trailing hold-back — nothing more can turn the
    /// tail into a marker.
    pub(crate) fn advance<'a>(&mut self, visible: &'a str, done: bool) -> Option<&'a str> {
        if self.frozen {
            return None;
        }
        // The runtime backstop for `stream_reply`'s monotonicity contract: a
        // profile whose visible text shrinks or rewrites what was already
        // emitted has broken it (or the model produced a shape the family has
        // not accounted for — a second Gemma channel close, a bare `</think>`
        // no one pre-filled). The already-streamed bytes cannot be recalled and
        // the turn's final message is authoritative, so stop here rather than
        // panic on `clamp(min > max)` below or emit across a rewritten prefix.
        if !visible.starts_with(&self.emitted_text) {
            tracing::debug!(
                emitted = self.emitted_text.len(),
                visible = visible.len(),
                "stream_reply output stopped extending what was emitted — stream frozen for this call"
            );
            self.frozen = true;
            return None;
        }
        if FREEZE_MARKERS.iter().any(|m| visible.contains(m)) || has_python_call_opener(visible) {
            self.frozen = true;
            return None;
        }
        // Stream everything except a trailing run that might be a marker in
        // progress: from the last `<` or `[` within `LOOKBACK` bytes of the end
        // to the end. Plain prose has neither and streams straight through, so a
        // one-line answer is not stuck behind a fixed-size guard. Walked as
        // `char_indices` rather than sliced, so a multi-byte char straddling the
        // window edge cannot land a byte index mid-character.
        let hold_from = if done {
            visible.len()
        } else {
            let cutoff = visible.len().saturating_sub(LOOKBACK);
            visible
                .char_indices()
                .rev()
                .take_while(|(i, _)| *i >= cutoff)
                .find(|(_, c)| *c == '<' || *c == '[')
                .map_or(visible.len(), |(i, _)| i)
        };
        let mut end = hold_from.clamp(self.emitted, visible.len());
        while end > self.emitted && !visible.is_char_boundary(end) {
            end -= 1;
        }
        if end <= self.emitted {
            return None;
        }
        let chunk = &visible[self.emitted..end];
        self.emitted = end;
        self.emitted_text.push_str(chunk);
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dangling opener at the prompt's tail is a pre-filled thought; a
    /// pre-filled block that is already *closed* (thinking off) is not.
    #[test]
    fn a_prompt_prefills_thinking_only_when_its_opener_dangles() {
        // Qwen3.8's template, thinking on / off (fixtures/chat_templates).
        assert!(prompt_prefills_thinking("<|im_start|>assistant\n<think>\n"));
        assert!(!prompt_prefills_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        assert!(!prompt_prefills_thinking("<start_of_turn>model\n"));
    }

    /// Feed `visible` one growing prefix at a time (as the decode loop would,
    /// after `clean_reply`), collecting every emitted fragment; the last step is
    /// the `done` flush.
    fn run(steps: &[&str]) -> (String, bool) {
        let mut s = StreamingReply::default();
        let mut out = String::new();
        let last = steps.len().saturating_sub(1);
        for (i, step) in steps.iter().enumerate() {
            if let Some(chunk) = s.advance(step, i == last) {
                out.push_str(chunk);
            }
        }
        (out, s.frozen)
    }

    /// Plain answer text — no `<` or `[` — streams straight through and the
    /// `done` flush delivers the rest, so a one-line reply is not stuck behind
    /// a fixed guard.
    #[test]
    fn plain_text_streams_in_full() {
        let full = "The capital of France is Paris, a city on the Seine.";
        let mut steps: Vec<&str> = (0..full.len())
            .filter(|i| full.is_char_boundary(*i))
            .map(|i| &full[..i])
            .collect();
        steps.push(full);
        let (streamed, frozen) = run(&steps);
        assert!(!frozen);
        assert_eq!(streamed, full);
    }

    /// `clean_reply` removes an open or closed `<think>` block, so the filter
    /// sees `""` until the answer proper begins, then streams it.
    #[test]
    fn nothing_streams_until_visible_text_exists() {
        let answer = "Hi there, the answer is that Paris is the capital.";
        let (streamed, frozen) = run(&["", "", "", answer, answer]);
        assert!(!frozen);
        assert_eq!(streamed, answer);
    }

    /// A `<` that could still be a marker forming is held back until it either
    /// resolves into prose or trips the freeze.
    #[test]
    fn a_lone_bracket_is_held_then_released_as_prose() {
        let (streamed, frozen) = run(&[
            "2 ",
            "2 <",
            "2 < 3 is true, obviously",
            "2 < 3 is true, obviously",
        ]);
        assert!(!frozen);
        assert_eq!(streamed, "2 < 3 is true, obviously");
    }

    #[test]
    fn a_tool_call_opener_freezes_the_stream() {
        let (streamed, frozen) = run(&[
            "Let me look that up for you now, one moment please. ",
            "Let me look that up for you now, one moment please. <tool_call>{\"name\"",
            "Let me look that up for you now, one moment please. <tool_call>{\"name\":\"read\"}",
        ]);
        assert!(frozen);
        assert!(
            "Let me look that up for you now, one moment please. ".starts_with(streamed.trim_end())
        );
        assert!(!streamed.contains("tool_call"));
    }

    /// Harmony's analysis channel is protocol, not answer: it freezes too.
    #[test]
    fn a_harmony_analysis_channel_freezes_the_stream() {
        let (_streamed, frozen) =
            run(&["<|channel|>analysis<|message|>The user wants the capital"]);
        assert!(frozen);
    }

    /// LFM2's bare `[Name(args)]` call — no wrapping marker on the llama.cpp
    /// path — is recognised by shape and freezes the stream.
    #[test]
    fn a_python_style_call_freezes_the_stream() {
        assert!(has_python_call_opener("Sure. [Read(file_path='a.txt')]"));
        assert!(has_python_call_opener("[MultiEdit(edits=[])]"));
        assert!(!has_python_call_opener("see item [3] in the list"));
        assert!(!has_python_call_opener("array[i] = f(x)"));

        let (streamed, frozen) = run(&[
            "Sure, reading it now. ",
            "Sure, reading it now. [Read(file_path=",
            "Sure, reading it now. [Read(file_path='a.txt')]",
        ]);
        assert!(frozen);
        assert!(!streamed.contains("Read("));
    }

    #[test]
    fn a_delta_never_splits_a_utf8_char() {
        let full = "café ".repeat(20);
        let steps: Vec<&str> = (0..=full.len())
            .filter(|i| full.is_char_boundary(*i))
            .map(|i| &full[..i])
            .collect();
        let (streamed, _) = run(&steps);
        assert!(std::str::from_utf8(streamed.as_bytes()).is_ok());
        assert!(full.starts_with(&streamed));
    }

    /// Regression for the app-server crash on Qwen3.8 (#233): `clean_reply` is
    /// not monotonic. A `<think>` block streams as prose (nothing in it is a
    /// freeze marker), then `strip_think_blocks` collapses the whole visible
    /// text to `""` the instant `</think>` lands. `advance` used to panic in
    /// `hold_from.clamp(self.emitted, visible.len())` — `clamp(54, 0)`. It must
    /// freeze instead, and the `done` flush must not panic either.
    #[test]
    fn a_visible_string_that_shrinks_freezes_instead_of_panicking() {
        let mut s = StreamingReply::default();
        let reasoning = "Okay, the user wants the capital of France, that is Paris";
        assert_eq!(s.advance(reasoning, false), Some(reasoning));
        // `</think>` lands; the visible text collapses.
        assert_eq!(s.advance("", false), None);
        assert!(s.frozen);
        // The real answer arrives after `</think>` — still frozen, no panic.
        assert_eq!(s.advance("Paris.", true), None);
    }

    /// A rewrite that is *not* shorter (the emitted prefix no longer matches)
    /// also freezes rather than emitting across the seam.
    #[test]
    fn a_rewritten_prefix_freezes() {
        let mut s = StreamingReply::default();
        assert_eq!(
            s.advance("thinking about it...", false),
            Some("thinking about it...")
        );
        assert_eq!(
            s.advance("The answer is 42, and then some more text.", false),
            None
        );
        assert!(s.frozen);
    }

    /// Regression: the forming-marker lookback walks `char_indices`, not a byte
    /// slice — text whose last `LOOKBACK` bytes land mid-character used to panic.
    #[test]
    fn a_long_multibyte_tail_does_not_panic() {
        let full = "日本語のテキストをたくさん書いてみるところ".repeat(4);
        let steps: Vec<&str> = full
            .char_indices()
            .map(|(i, _)| &full[..i])
            .chain(std::iter::once(full.as_str()))
            .collect();
        let (streamed, frozen) = run(&steps);
        assert!(!frozen);
        assert_eq!(streamed, full);
    }
}
