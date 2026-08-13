//! MiniMax-M2.7's native tool-call format.

use serde_json::Value;

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::tags::{value_boundaries, wrapper_body};

/// The tag that opens a call block. Public because a chat template *declaring*
/// this format and a model *emitting* it write the same literal, and the two
/// checks must not be able to drift apart.
pub const WRAPPER_OPEN: &str = "<minimax:tool_call>";
const WRAPPER_CLOSE: &str = "</minimax:tool_call>";
const INVOKE_OPEN: &str = "<invoke name=\"";
const INVOKE_CLOSE: &str = "</invoke>";
const PARAM_OPEN: &str = "<parameter name=\"";
const PARAM_CLOSE: &str = "</parameter>";

/// Parse MiniMax-M2.7's native tool-call format:
/// `<minimax:tool_call><invoke name="...">
/// <parameter name="...">value</parameter>...</invoke></minimax:tool_call>`,
/// possibly with several `<invoke>` blocks in one wrapper. The model's own
/// template (`render_native`) renders a string-typed argument raw/unquoted
/// and everything else `tojson`-encoded, so the wire format alone can't say
/// which a given `<parameter>` value is — `"42"` and the integer `42` render
/// identically. `tools`' JSON Schema resolves that per parameter name; an
/// unknown tool or parameter defaults to string, since treating an actual
/// string as a string is lossless while the reverse is not.
///
/// The wire format has no escaping at all, so a raw string argument (source
/// code, HTML, another tool-call transcript) can legally contain the literal
/// text `</parameter>` or `</invoke>` — matching those tags with a plain
/// non-greedy regex would truncate the value at the first occurrence inside
/// it rather than the real boundary. `value_boundaries` instead finds every
/// *opening* tag first (those are well-formed — models don't have a reason
/// to fabricate one mid-string) and bounds each value's search window to
/// where the *next* opening tag starts (or a real end-of-content boundary —
/// see below), so a stray closing tag earlier in the value can never be
/// mistaken for the terminator: whatever `close` sits last inside that
/// window is the real one. The failure mode this fixes is silently cutting
/// off a real `MultiEdit`-style payload, which is the one issue #105's
/// discussion specifically flagged (a review comment on the PR that
/// introduced this).
///
/// "Real end-of-content boundary" is why this narrows `text` to inside the
/// `<minimax:tool_call>…</minimax:tool_call>` wrapper (`wrapper_body`) before
/// ever calling `value_boundaries`: without that, the *last* invoke's search
/// window would run all the way to the literal end of the model's raw
/// completion — including the wrapper's own `</minimax:tool_call>` and
/// anything the model wrote after it — and `rfind` inside an unbounded
/// window is exactly the original bug again, just moved to a different
/// layer.
pub fn parse_calls(text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
    let Some(wrapped) = wrapper_body(text, WRAPPER_OPEN, WRAPPER_CLOSE) else {
        return Vec::new();
    };

    let mut calls = Vec::new();
    for (name, body) in value_boundaries(wrapped, INVOKE_OPEN, INVOKE_CLOSE) {
        let schema = tools.iter().find(|t| t.name == name).map(|t| &t.parameters);

        let mut args = serde_json::Map::new();
        for (key, raw) in value_boundaries(body, PARAM_OPEN, PARAM_CLOSE) {
            let is_string_type = schema
                .and_then(|s| s.get("properties"))
                .and_then(|p| p.get(key))
                .and_then(|p| p.get("type"))
                .and_then(|t| t.as_str())
                .map(|t| t == "string")
                .unwrap_or(true);
            let value = if is_string_type {
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
