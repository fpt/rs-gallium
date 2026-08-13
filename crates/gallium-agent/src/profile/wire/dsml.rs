//! DeepSeek-V4's native "DSML" tool-call format.
//!
//! Note every delimiter here uses U+FF5C (fullwidth vertical bar, "｜"), not the
//! ASCII "|" the other formats delimit with — a `<|…|>`-shaped substring search
//! will never match DSML.

use serde_json::Value;

use crate::llm::ToolCallInfo;

use super::tags::{value_boundaries, wrapper_body};

/// The tag that opens a call block. Public because a chat template *declaring*
/// this format and a model *emitting* it write the same literal, and the two
/// checks must not be able to drift apart.
pub const WRAPPER_OPEN: &str = "<｜DSML｜tool_calls>";
const WRAPPER_CLOSE: &str = "</｜DSML｜tool_calls>";
const INVOKE_OPEN: &str = "<｜DSML｜invoke name=\"";
const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const PARAM_OPEN: &str = "<｜DSML｜parameter name=\"";
const PARAM_CLOSE: &str = "</｜DSML｜parameter>";

/// Parse DeepSeek-V4's native "DSML" tool-call format:
/// `<｜DSML｜tool_calls><｜DSML｜invoke name="...">
/// <｜DSML｜parameter name="..." string="true|false">value</｜DSML｜parameter>
/// ...</｜DSML｜invoke>...</｜DSML｜tool_calls>` — one or more `<｜DSML｜invoke>`
/// blocks in one wrapper, same shape as MiniMax's `<invoke>` format
/// (`super::minimax`) but with DeepSeek's fullwidth-pipe delimiter instead of a
/// `minimax:` namespace, and its own `string` attribute naming each parameter's
/// type directly — no tool schema needed to disambiguate `"42"` from `42` the
/// way MiniMax's wire format does.
///
/// Trims to the wrapper before scanning invokes, and bounds each invoke's
/// parameter scan to its own body, for the same no-escaping reason
/// `super::minimax::parse_calls` does: the format has no way to escape a
/// literal `</｜DSML｜invoke>` or `</｜DSML｜parameter>` inside an argument value
/// (source code, a nested tool-call transcript), so an unbounded search could
/// latch onto one inside the value instead of the real terminator.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    let Some(wrapped) = wrapper_body(text, WRAPPER_OPEN, WRAPPER_CLOSE) else {
        return Vec::new();
    };

    let mut calls = Vec::new();
    for (name, body) in value_boundaries(wrapped, INVOKE_OPEN, INVOKE_CLOSE) {
        let mut args = serde_json::Map::new();
        for (key, is_string, raw) in parameter_boundaries(body, PARAM_OPEN, PARAM_CLOSE) {
            let value = if is_string {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            args.insert(key.to_string(), value);
        }
        calls.push(ToolCallInfo {
            id: String::new(),
            name: name.to_string(),
            arguments: Value::Object(args),
        });
    }
    calls
}

/// Like [`value_boundaries`], but for DSML's `<｜DSML｜parameter name="..."
/// string="true|false">value</｜DSML｜parameter>` tag, which carries a second
/// attribute after `name` that `value_boundaries` has nowhere to return.
/// `is_string` defaults to `true` (a missing, unrecognized, or malformed
/// `string=` attribute is treated as a string — and parsing continues to
/// the next parameter rather than aborting the scan — the same
/// lossless-default reasoning `super::minimax::parse_calls` uses for an unknown
/// parameter's schema type) — only a literal `"false"` decodes the value
/// as JSON.
fn parameter_boundaries<'a>(
    text: &'a str,
    open_prefix: &str,
    close: &str,
) -> Vec<(&'a str, bool, &'a str)> {
    let mut opens: Vec<(&str, bool, usize)> = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(open_prefix) {
        let name_start = search_from + rel + open_prefix.len();
        let Some(name_end_rel) = text[name_start..].find('"') else {
            break;
        };
        let name_end = name_start + name_end_rel;
        // Bound the `string=` lookup to this tag's own attribute list (up to
        // its own closing `>`) rather than searching the rest of `text`
        // unbounded — otherwise a tag missing the attribute would silently
        // borrow a *later* tag's `string="..."`, or the literal text
        // `string="` inside an earlier value, instead of defaulting.
        let Some(tag_close_rel) = text[name_end..].find('>') else {
            break;
        };
        let tag_close = name_end + tag_close_rel;
        let attrs = &text[name_end..tag_close];
        let is_string = attrs
            .find("string=\"")
            .and_then(|sa_rel| {
                let val_start = sa_rel + "string=\"".len();
                attrs[val_start..]
                    .find('"')
                    .map(|val_end_rel| &attrs[val_start..val_start + val_end_rel] != "false")
            })
            // No (or malformed) `string=` attribute: default to string, the
            // same lossless-default reasoning as an unknown MiniMax
            // parameter — don't abort the rest of the scan over it.
            .unwrap_or(true);
        let value_start = tag_close + 1;
        opens.push((&text[name_start..name_end], is_string, value_start));
        search_from = value_start;
    }

    opens
        .iter()
        .enumerate()
        .map(|(i, &(name, is_string, value_start))| {
            let boundary = opens
                .get(i + 1)
                .map(|&(_, _, next_start)| {
                    text[..next_start].rfind(open_prefix).unwrap_or(next_start)
                })
                .unwrap_or(text.len());
            let window = &text[value_start..boundary];
            let value = window.rfind(close).map_or(window, |pos| &window[..pos]);
            (name, is_string, value)
        })
        .collect()
}
