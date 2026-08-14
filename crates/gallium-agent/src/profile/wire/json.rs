//! The JSON-prose protocol: what gallium *asks* a model for when its own
//! template declares no native tool format (`llm_local`'s `tool_instructions`).
//!
//! This is the one format no model is trained on, so it is also the one every
//! model may fall back to — a profile whose native parser finds nothing should
//! still try this before giving up.

use serde_json::Value;

use crate::llm::{ToolCallInfo, ToolDefinition};

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
pub fn parse_calls(text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
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
            let calls = extract_calls(&val, tools);
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
fn extract_calls(val: &Value, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
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
        Value::Object(_) => {
            let calls = one(val).into_iter().collect::<Vec<_>>();
            if calls.is_empty() {
                keyed_by_tool_name(val, tools)
            } else {
                calls
            }
        }
        _ => Vec::new(),
    }
}

/// The shape where the model uses the **tool's name as the key** and its
/// arguments as the value: `{"Write": {"file_path": …, "content": …}}`, or
/// several as sibling keys.
///
/// Nothing asks a model for this — `tool_instructions` requests
/// `{"name": …, "arguments": …}` — but small local models produce it anyway, and
/// until now it was read as a *text reply and printed to the user*, which is the
/// worst way to fail: the model called a tool and the user saw JSON.
///
/// **Every key must name an offered tool**, which is what makes the shape safe
/// to accept at all. Without that gate any single-key JSON object in a reply
/// would become a tool call — a model answering "what does this config mean?"
/// with `{"llm": {...}}` would invent one. Matching is `ToolRegistry`'s: exact,
/// else ignoring case and underscores, so the gate is never stricter than the
/// resolution that follows it.
///
/// A value that is not an object is not arguments, so `{"Read": "a.txt"}` is
/// left alone rather than guessed at.
fn keyed_by_tool_name(val: &Value, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
    let obj = match val.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => return Vec::new(),
    };
    let known = |key: &str| {
        tools
            .iter()
            .any(|t| t.name == key || normalized(&t.name) == normalized(key))
    };
    if !obj.iter().all(|(k, v)| known(k) && v.is_object()) {
        return Vec::new();
    }
    obj.iter()
        .map(|(name, args)| ToolCallInfo {
            id: String::new(),
            name: name.clone(),
            arguments: args.clone(),
        })
        .collect()
}

/// `ToolRegistry::normalized`, duplicated rather than shared: that one is a
/// private detail of the registry, and this gate only needs to agree with it,
/// not depend on it.
fn normalized(name: &str) -> String {
    name.chars()
        .filter(|c| *c != '_')
        .flat_map(char::to_lowercase)
        .collect()
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

#[cfg(test)]
mod keyed_tests {
    use super::*;

    fn tools(names: &[&str]) -> Vec<ToolDefinition> {
        names
            .iter()
            .map(|n| ToolDefinition {
                name: n.to_string(),
                description: String::new(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            })
            .collect()
    }

    /// The shape LFM2.5 sends for a file write. Before this it was returned as a
    /// text reply and the user saw the JSON.
    #[test]
    fn a_tool_name_key_is_accepted_when_it_names_an_offered_tool() {
        let t = tools(&["Write", "Read"]);
        let calls = parse_calls(
            r#"{"Write": {"file_path": "hello.go", "content": "package main"}}"#,
            &t,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["file_path"], "hello.go");
        assert_eq!(calls[0].arguments["content"], "package main");
    }

    /// Several as sibling keys — the `refactoring` shape, when its JSON is valid.
    #[test]
    fn sibling_tool_name_keys_all_become_calls() {
        let t = tools(&["Read", "Edit"]);
        let calls = parse_calls(
            r#"{"Read": {"file_path": "a.go"}, "Edit": {"file_path": "a.go", "old_string": "x"}}"#,
            &t,
        );
        assert_eq!(calls.len(), 2);
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.contains(&"Read") && names.contains(&"Edit"),
            "{names:?}"
        );
    }

    /// The gate. Without it any single-key object in a reply becomes a call — a
    /// model explaining a config with `{"llm": {...}}` would invent one.
    #[test]
    fn a_key_naming_no_offered_tool_is_not_a_call() {
        let t = tools(&["Write", "Read"]);
        assert!(parse_calls(r#"{"llm": {"modelPath": "x.gguf"}}"#, &t).is_empty());
        // Same object, no tools offered at all.
        assert!(parse_calls(r#"{"Write": {"file_path": "a"}}"#, &[]).is_empty());
    }

    /// Matched the way `ToolRegistry` resolves, so the gate is never stricter
    /// than what would have run: a model writing `write_file` reaches `WriteFile`.
    #[test]
    fn case_and_underscores_are_ignored_like_the_registry_does() {
        let t = tools(&["MultiEdit"]);
        let calls = parse_calls(r#"{"multi_edit": {"file_path": "a"}}"#, &t);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "multi_edit", "name passes through verbatim");
    }

    /// A value that is not an object is not an argument set.
    #[test]
    fn a_non_object_value_is_not_arguments() {
        let t = tools(&["Read"]);
        assert!(parse_calls(r#"{"Read": "a.txt"}"#, &t).is_empty());
    }

    /// The canonical shape still wins: `{"name": …}` is checked first, so a tool
    /// literally called `name` cannot turn a normal call into a keyed one.
    #[test]
    fn the_canonical_shape_is_still_preferred() {
        let t = tools(&["Read", "name"]);
        let calls = parse_calls(r#"{"name": "Read", "arguments": {"file_path": "a"}}"#, &t);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    /// The two shapes a live LFM2.5 was observed sending that this does **not**
    /// reach, pinned so the limit is not rediscovered: an argument object with no
    /// name anywhere, and a `{"ToolName": …}` whose JSON is truncated.
    #[test]
    fn the_two_observed_shapes_this_does_not_fix() {
        let t = tools(&["Write", "Read", "Edit"]);
        // `coding`: arguments only, no key to gate on.
        assert!(parse_calls(
            r#"{"file_path": "hello.go", "content": "package main"}"#,
            &t
        )
        .is_empty());
        // `refactoring`: right shape, unclosed object — parsing fails first.
        assert!(parse_calls(
            r#"{"Read": {"file_path": "a.go"}, "Edit": {"file_path": "a.go"}"#,
            &t
        )
        .is_empty());
    }
}
