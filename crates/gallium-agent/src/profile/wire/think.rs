//! Reasoning wrappers: `<think>…</think>` and the opener-less variant.

use std::sync::OnceLock;

/// Remove well-formed `<think>...</think>` blocks (case-insensitive). An unclosed
/// `<think>` (model still reasoning, no answer yet) is left as-is.
///
/// Some chat templates — MiniMax-M2.7's among them, see `configs/minimax-m2.toml`
/// — pre-fill `<think>\n` into the *prompt* rather than generating it, so the
/// model's own output carries only the closing `</think>`. Without an opening
/// tag to pair it with, the reasoning before it would otherwise pass straight
/// through untouched, so a `</think>` found before any `<think>` (or with none
/// at all) is treated the same way: everything up to and including it is the
/// model's thinking.
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
/// and the model's own output carries only the closing tag. Two scans that
/// disagreed about where reasoning ends would put part of the answer in the
/// think block, or part of the thinking in the answer.
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

    /// An unclosed block is a model still reasoning; `strip_think_blocks`
    /// leaves it in the text, so nothing has been decided to be reasoning yet.
    #[test]
    fn an_unclosed_block_is_not_reasoning_yet() {
        assert_eq!(think_content("<think>still going"), None);
    }

    /// Nothing to report reads the same as an empty report — a template
    /// branching on `reasoning_content is string` should see neither.
    #[test]
    fn no_reasoning_and_empty_reasoning_are_both_none() {
        assert_eq!(think_content("just an answer"), None);
        assert_eq!(think_content("<think>   </think>answer"), None);
    }
}
