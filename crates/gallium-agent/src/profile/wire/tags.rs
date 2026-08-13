//! The `<TAG name="…">value</TAG>` scanner shared by the two `<invoke>`-shaped
//! wire formats (MiniMax-M2.7 and DeepSeek-V4's DSML).
//!
//! Neither format escapes anything, so the whole difficulty is deciding where a
//! value ends when the value itself may contain the closing tag verbatim. See
//! [`value_boundaries`].

/// Split `text` into `(name, value)` pairs for a repeated
/// `<TAG name="...">value</CLOSE>` run — `open_prefix` is `<TAG name="` (the
/// `">` that ends the opening tag is assumed literal), `close` is `</CLOSE>`.
/// A value runs from the end of its opening tag to the start of the *next*
/// opening tag (or end of `text`) — that span, the "window", is a hard
/// boundary the value cannot have leaked past, so finding `close` *within*
/// it (last occurrence, in case the value itself repeats `close` — see
/// below) is always the real closing tag, never a truncation point.
///
/// Only a plain `rfind` inside that bounded window is safe this way; the
/// unbounded version (searching all the way to literal end-of-string for the
/// last element) is exactly the bug this function exists to avoid — see
/// `super::minimax::parse_calls`'s doc comment for why its caller pre-trims
/// `text` to a real boundary before calling this at all.
pub fn value_boundaries<'a>(
    text: &'a str,
    open_prefix: &str,
    close: &str,
) -> Vec<(&'a str, &'a str)> {
    // Every opening tag's (name, byte offset right after its closing `">`).
    let mut opens: Vec<(&str, usize)> = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(open_prefix) {
        let name_start = search_from + rel + open_prefix.len();
        let Some(name_end_rel) = text[name_start..].find('"') else {
            break;
        };
        let name_end = name_start + name_end_rel;
        let Some(gt_rel) = text[name_end..].find('>') else {
            break;
        };
        let value_start = name_end + gt_rel + 1;
        opens.push((&text[name_start..name_end], value_start));
        search_from = value_start;
    }

    opens
        .iter()
        .enumerate()
        .map(|(i, &(name, value_start))| {
            let boundary = opens
                .get(i + 1)
                .map(|&(_, next_start)| {
                    // next_start is just past the next tag's own opening `">`;
                    // walk back to where that tag's `<` began.
                    text[..next_start].rfind(open_prefix).unwrap_or(next_start)
                })
                .unwrap_or(text.len());
            let window = &text[value_start..boundary];
            let value = window.rfind(close).map_or(window, |pos| &window[..pos]);
            (name, value)
        })
        .collect()
}

/// Narrow `text` to what sits inside a `open`…`close` wrapper, or `None` when
/// the wrapper never opens.
///
/// The closing tag is found with `rfind`, not `find`, for the same reason
/// [`value_boundaries`] bounds its own search: an argument value inside the
/// wrapper can legally contain the wrapper's closing tag verbatim, so the
/// *last* occurrence is the real one. A missing close runs to end-of-text —
/// a truncated generation still yields the calls it managed to emit.
pub fn wrapper_body<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.find(open)?;
    let inner_start = start + open.len();
    let inner_end = text[inner_start..]
        .rfind(close)
        .map(|rel| inner_start + rel)
        .unwrap_or(text.len());
    Some(&text[inner_start..inner_end])
}
