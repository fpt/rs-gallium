//! Reasoning wrappers: `<think>…</think>` and the opener-less variant.

use std::sync::OnceLock;

/// Remove well-formed `<think>...</think>` blocks (case-insensitive). An
/// **unclosed** `<think>` — the model hit EOS before finishing this thought —
/// is dropped too, from the opener to the end of the string: there is nothing
/// after it but more of the same reasoning, and showing it as if it were the
/// answer is worse than showing nothing. This is the same call
/// `crate::gemma::strip_thinking_blocks` already makes for Gemma 4's own
/// `<|think|>` marker ("safest to discard the tail"); this function only
/// needed to catch up to it.
///
/// Some chat templates — MiniMax-M2.7's among them, see `configs/minimax-m2.toml`
/// — pre-fill `<think>\n` into the *prompt* rather than generating it, so the
/// model's own output carries only the closing `</think>`. Without an opening
/// tag to pair it with, the reasoning before it would otherwise pass straight
/// through untouched, so a `</think>` found before any `<think>` (or with none
/// at all) is treated the same way: everything up to and including it is the
/// model's thinking. A generation that stops before *that* closer ever lands
/// carries no tag at all — the opener is invisible, sitting in the prompt —
/// which is what `streaming::restore_thinking_opener` is for: it puts the
/// opener back before this function ever sees the text, so the case above
/// applies here too instead of the raw reasoning reading as a plain answer.
pub fn strip_think_blocks(text: &str) -> String {
    // Matched directly against the original string rather than a
    // `to_lowercase()`'d copy: lowercasing can change a string's byte length
    // (e.g. Turkish İ), which would desync offsets found in the lowercase
    // copy from the original they're sliced out of — a panic or a wrong cut
    // waiting on the right non-ASCII reasoning text. `regex`'s `(?i)` matches
    // case-insensitively while still returning offsets into the string it
    // was run on, so there is no second copy to fall out of sync with.
    static OPEN: OnceLock<regex::Regex> = OnceLock::new();
    static CLOSE: OnceLock<regex::Regex> = OnceLock::new();
    let open_re = OPEN.get_or_init(|| regex::Regex::new(r"(?i)<think>").unwrap());
    let close_re = CLOSE.get_or_init(|| regex::Regex::new(r"(?i)</think>").unwrap());

    let mut s = text.to_string();

    if let Some(close_m) = close_re.find(&s) {
        let has_earlier_open = open_re
            .find(&s)
            .is_some_and(|open_m| open_m.start() < close_m.start());
        if !has_earlier_open {
            s.replace_range(0..close_m.end(), "");
        }
    }

    while let Some(open_m) = open_re.find(&s) {
        let Some(close_m) = close_re.find(&s[open_m.start()..]) else {
            // Unclosed: nothing after the opener is the answer, so none of it
            // is kept.
            s.truncate(open_m.start());
            break;
        };
        let end = open_m.start() + close_m.end();
        s.replace_range(open_m.start()..end, "");
    }
    s
}

/// The reasoning [`strip_think_blocks`] removes, or `None` when there is none.
///
/// The exact inverse, deliberately sharing its rules — including the
/// opener-less variant, where a template pre-filled `<think>\n` into the prompt
/// and the model's own output carries only the closing tag, and the unclosed
/// variant, where the model hit EOS mid-thought and never produced a closer at
/// all. Two scans that disagreed about where reasoning ends would put part of
/// the answer in the think block, or part of the thinking in the answer — an
/// unclosed opener that `strip_think_blocks` drops but this function didn't
/// capture would silently lose that reasoning rather than misplace it, which
/// is just as much a disagreement.
///
/// Several blocks are joined by a blank line: a model that reasons, answers,
/// and reasons again produced one turn's thinking, and the caller wants it as
/// one string. Whitespace-only reasoning is `None`, since an empty
/// `reasoning_content` and no reasoning at all should reach a template the
/// same way.
pub fn think_content(text: &str) -> Option<String> {
    let open_re = regex::Regex::new(r"(?i)<think>").ok()?;
    let close_re = regex::Regex::new(r"(?i)</think>").ok()?;

    let mut blocks: Vec<String> = Vec::new();
    let mut s = text.to_string();

    // A `</think>` with no `<think>` before it: everything up to it is thinking.
    if let Some(close_m) = close_re.find(&s) {
        let has_earlier_open = open_re
            .find(&s)
            .is_some_and(|open_m| open_m.start() < close_m.start());
        if !has_earlier_open {
            blocks.push(s[..close_m.start()].to_string());
            s.replace_range(0..close_m.end(), "");
        }
    }

    while let Some(open_m) = open_re.find(&s) {
        let Some(close_m) = close_re.find(&s[open_m.start()..]) else {
            // Unclosed: everything after the opener is reasoning, same as
            // strip_think_blocks drops.
            blocks.push(s[open_m.end()..].to_string());
            break;
        };
        let end = open_m.start() + close_m.end();
        blocks.push(s[open_m.end()..open_m.start() + close_m.start()].to_string());
        s.replace_range(open_m.start()..end, "");
    }

    let joined = blocks
        .iter()
        .map(|b| b.trim())
        .filter(|b| !b.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!joined.is_empty()).then_some(joined)
}

/// The visible answer-so-far while generation is still in progress —
/// [`strip_think_blocks`]'s incremental counterpart, and the default body of
/// `ModelProfile::stream_reply`.
///
/// Now the same function in every case, not a different one: closed
/// `<think>…</think>` blocks, the opener-less prefix before a bare `</think>`,
/// and an *unclosed* trailing opener are all removed exactly as the batch
/// cleaner removes them — an unclosed block used to be `clean_reply`'s one
/// deliberate difference (shown as prose, then collapsed to `""` when
/// `</think>` landed a few tokens later — the #233 leak), but showing an
/// in-progress thought as if it were the answer is wrong on a stream for the
/// same reason it is wrong on the final message, so both now hold it back.
/// `trim_start` is the only work left for this function to do on its own.
///
/// Prefix-monotonic by construction: text before an opener never changes once
/// the opener exists, and closing a block only appends what follows it. The
/// case this cannot decide from text alone is reasoning whose opener lives in
/// the *prompt* (a template that pre-fills `<think>\n`, Qwen3.8's and
/// MiniMax-M2.7's both do) — prose then would be reasoning, indistinguishable
/// from an answer until the bare `</think>` lands. Only the engine knows what
/// its rendered prompt ended with, so it prepends the dangling opener before
/// calling `stream_reply` — see `streaming::prompt_prefills_thinking` — and,
/// since #233's non-streaming counterpart, before calling `clean_reply` too
/// (`streaming::restore_thinking_opener`).
pub fn stream_visible(text: &str) -> String {
    strip_think_blocks(text).trim_start().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse property, asserted rather than assumed: what one takes out
    /// is what the other hands back.
    #[test]
    fn think_content_is_what_strip_removes() {
        let text = "<think>weighing it up</think>the answer";
        assert_eq!(think_content(text).as_deref(), Some("weighing it up"));
        assert_eq!(strip_think_blocks(text), "the answer");
    }

    /// The opener-less variant, where the template pre-filled `<think>` into
    /// the prompt. Both functions have to agree that everything before the
    /// closing tag is reasoning.
    #[test]
    fn an_openerless_close_tag_is_all_reasoning() {
        let text = "still weighing it up</think>the answer";
        assert_eq!(think_content(text).as_deref(), Some("still weighing it up"));
        assert_eq!(strip_think_blocks(text), "the answer");
    }

    #[test]
    fn several_blocks_join() {
        assert_eq!(
            think_content("<think>one</think>a<think>two</think>b").as_deref(),
            Some("one\n\ntwo")
        );
    }

    /// An unclosed block is still reasoning — the model hit EOS before
    /// answering — and `strip_think_blocks` now drops it, so `think_content`
    /// must report it rather than silently lose it (they are exact inverses).
    #[test]
    fn an_unclosed_block_is_reasoning_too() {
        assert_eq!(
            think_content("<think>still going"),
            Some("still going".to_string())
        );
        assert_eq!(strip_think_blocks("<think>still going"), "");
    }

    /// A self-emitted opener with text and nothing after it: same rule,
    /// starting mid-string.
    #[test]
    fn an_unclosed_block_after_other_text_drops_only_the_tail() {
        assert_eq!(strip_think_blocks("Sure — <think>but wait"), "Sure — ");
        assert_eq!(
            think_content("Sure — <think>but wait").as_deref(),
            Some("but wait")
        );
    }

    /// Nothing to report reads the same as an empty report — a template
    /// branching on `reasoning_content is string` should see neither.
    #[test]
    fn no_reasoning_and_empty_reasoning_are_both_none() {
        assert_eq!(think_content("just an answer"), None);
        assert_eq!(think_content("<think>   </think>answer"), None);
    }

    /// An unclosed `<think>` is held back, same as the batch cleaner now
    /// drops it (#233's original streaming fix; no longer a divergence).
    #[test]
    fn stream_visible_holds_an_unclosed_think_block() {
        assert_eq!(stream_visible("<think>Okay, the user wants"), "");
        assert_eq!(stream_visible("Sure — <think>but wait"), "Sure — ");
        // Once it closes, what follows is answer, exactly as the batch strip.
        assert_eq!(
            stream_visible("<think>Okay.</think>\n\nParis."),
            strip_think_blocks("<think>Okay.</think>\n\nParis.").trim_start()
        );
    }

    /// Prefix monotonicity, checked through the real pipeline: every growing
    /// prefix of a reply that reasons, answers, and reasons again, fed through
    /// `stream_visible` into `StreamingReply` — the stream never freezes (which
    /// is what a shrink or rewrite would trigger) and delivers exactly what the
    /// batch cleaner keeps. A forming `<think>` mid-answer transiently shows as
    /// `…<thi`, but always at the tail, where `StreamingReply`'s lookback holds
    /// it until it resolves.
    #[test]
    fn stream_visible_through_the_stream_filter_never_freezes_or_leaks() {
        let full = "<think>one</think>The answer <think>two</think>continues here.";
        let mut s = crate::streaming::StreamingReply::default();
        let mut out = String::new();
        for (i, _) in full.char_indices().skip(1) {
            if let Some(chunk) = s.advance(&stream_visible(&full[..i]), false) {
                out.push_str(chunk);
            }
        }
        if let Some(chunk) = s.advance(&stream_visible(full), true) {
            out.push_str(chunk);
        }
        assert!(!s.frozen, "monotonic input must never trip the freeze");
        assert!(
            !out.contains("one") && !out.contains("two"),
            "leaked: {out:?}"
        );
        assert_eq!(out, "The answer continues here.");
    }
}
