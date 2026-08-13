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
