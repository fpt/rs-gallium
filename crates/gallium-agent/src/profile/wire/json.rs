//! The JSON-prose protocol: what gallium *asks* a model for when its own
//! template declares no native tool format (`llm_local`'s `tool_instructions`).
//!
//! This is the one format no model is trained on, so it is also the one every
//! model may fall back to — a profile whose native parser finds nothing should
//! still try this before giving up.

use serde_json::Value;

use crate::llm::ToolCallInfo;

/// Leniently extract tool calls from a reply that follows gallium's own JSON
/// protocol. Accepts the whole reply as JSON, or the first balanced
/// `{...}`/`[...]` block (handles models that wrap JSON in prose or ``` fences).
/// Returns empty if none found.
///
/// **`text` must already have its reasoning removed** — a `<think>` block is
/// prose full of braces, and the balanced-span scan below will happily latch
/// onto a `{…}` inside one. Stripping is the caller's job because it has to
/// happen exactly once per reply (see [`super::think::strip_think_blocks`]: a
/// second pass over text with two bare `</think>` tags cuts again) and because
/// which reasoning shape a model emits is a property of its family.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    let mut candidates: Vec<String> = Vec::new();
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        candidates.push(trimmed.to_string());
    }
    if let Some(block) = first_balanced_json(text) {
        if candidates.first().map(|c| c != &block).unwrap_or(true) {
            candidates.push(block);
        }
    }

    for candidate in candidates {
        if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
            let calls = extract_calls(&val);
            if !calls.is_empty() {
                return calls;
            }
        }
    }
    Vec::new()
}

/// Pull ToolCallInfo out of a parsed JSON value in any of the shapes a model
/// might emit: a bare object, an array of objects, `{"tool_calls": [...]}`,
/// and either `{name, arguments}` or `{function: {name, arguments}}`.
fn extract_calls(val: &Value) -> Vec<ToolCallInfo> {
    fn one(v: &Value) -> Option<ToolCallInfo> {
        let obj = v.as_object()?;
        let (name, raw_args) = if let Some(f) = obj.get("function").and_then(|f| f.as_object()) {
            (
                f.get("name")?.as_str()?.to_string(),
                f.get("arguments").cloned(),
            )
        } else {
            (
                obj.get("name")?.as_str()?.to_string(),
                obj.get("arguments").cloned(),
            )
        };
        let arguments = match raw_args {
            // OpenAI serializes arguments as a JSON string; accept that too.
            Some(Value::String(s)) => {
                serde_json::from_str(&s).unwrap_or(Value::Object(Default::default()))
            }
            Some(v) => v,
            None => Value::Object(Default::default()),
        };
        Some(ToolCallInfo {
            id: String::new(),
            name,
            arguments,
        })
    }

    match val {
        Value::Array(arr) => arr.iter().filter_map(one).collect(),
        Value::Object(o) if o.contains_key("tool_calls") => o
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(one).collect())
            .unwrap_or_default(),
        Value::Object(_) => one(val).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Find the first balanced `{...}` or `[...]` span in `text`, respecting JSON
/// string literals (so braces inside strings don't unbalance it). Returns the
/// substring including the brackets, or None.
fn first_balanced_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(text[start..=i].to_string());
            }
        }
    }
    None
}
