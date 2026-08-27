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

/// The two name-less shapes a small local model reaches for when asked for
/// `{"name": …, "arguments": …}` — read as a *text reply and printed to the
/// user* until they were accepted here, which is the worst way to fail: the
/// model called a tool and the user saw JSON.
///
/// 1. **Tool name as the key**, arguments as the value:
///    `{"Write": {"file_path": …, "content": …}}`, or several as sibling keys.
///    Safe because **every key must name an offered tool** — without that gate a
///    model answering "what does this config mean?" with `{"llm": {...}}` would
///    invent a call. Matching is `ToolRegistry`'s: exact, else ignoring case and
///    underscores. A value that is not an object is not arguments, so
///    `{"Read": "a.txt"}` is left alone.
///
/// 2. **The argument object with no name at all**:
///    `{"file_path": "hello.go", "content": "package main"}`. This is what LFM2.5
///    sends for a write (#118). Safe because it is bound only when the key set
///    fits **exactly one** offered tool's schema — every `required` parameter
///    present, and no key outside that tool's `properties`. Ambiguity (two tools
///    fit) or a foreign key means no call, same as shape 1's gate.
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
    // Shape 1: every key is a tool name, every value an argument object.
    if obj.iter().all(|(k, v)| known(k) && v.is_object()) {
        return obj
            .iter()
            .map(|(name, args)| ToolCallInfo {
                id: String::new(),
                name: name.clone(),
                arguments: args.clone(),
            })
            .collect();
    }
    // Shape 2: the whole object is one tool's arguments, name dropped.
    args_match_unique_tool(obj, tools)
}

/// Bind a name-less argument object to the single offered tool whose parameter
/// schema it fits: every key is one of that tool's `properties`, and every
/// `required` parameter is present. Returns a call only when **exactly one**
/// tool qualifies — two matches, or none, means the model's intent is not
/// recoverable and it stays a text reply.
///
/// Schema-driven so a client's `dynamicTools` get the same treatment as the
/// built-ins. A tool that declares no `properties` object cannot be judged and
/// never matches here.
fn args_match_unique_tool(
    obj: &serde_json::Map<String, Value>,
    tools: &[ToolDefinition],
) -> Vec<ToolCallInfo> {
    let fits = |t: &ToolDefinition| -> bool {
        let schema = match t.parameters.as_object() {
            Some(s) => s,
            None => return false,
        };
        let props = match schema.get("properties").and_then(|p| p.as_object()) {
            Some(p) if !p.is_empty() => p,
            _ => return false,
        };
        let required = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|r| r.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        obj.keys().all(|k| props.contains_key(k)) && required.iter().all(|r| obj.contains_key(*r))
    };

    let mut matched = tools.iter().filter(|t| fits(t));
    match (matched.next(), matched.next()) {
        (Some(t), None) => vec![ToolCallInfo {
            id: String::new(),
            name: t.name.clone(),
            arguments: Value::Object(obj.clone()),
        }],
        _ => Vec::new(),
    }
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

    /// The built-ins' real parameter schemas, for the shape-2 gate — which needs
    /// `properties` and `required` to judge a name-less object against.
    fn schema_tools() -> Vec<ToolDefinition> {
        let mk = |name: &str, props: &[&str], required: &[&str]| ToolDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": props.iter().map(|p| (p.to_string(), serde_json::json!({"type": "string"}))).collect::<serde_json::Map<_, _>>(),
                "required": required,
            }),
        };
        vec![
            mk("Read", &["file_path", "limit", "offset"], &["file_path"]),
            mk(
                "Write",
                &["file_path", "content"],
                &["file_path", "content"],
            ),
            mk(
                "Edit",
                &["file_path", "old_string", "new_string", "replace_all"],
                &["file_path", "old_string", "new_string"],
            ),
            mk("Glob", &["pattern", "path", "limit"], &["pattern"]),
        ]
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

    /// Shape 2 (#118): a name-less argument object binds to the one tool whose
    /// schema it fits. This is what LFM2.5 sends for `coding` — before this it
    /// was a text reply and the user saw the JSON.
    #[test]
    fn a_name_less_argument_object_binds_to_the_only_tool_whose_schema_fits() {
        let t = schema_tools();

        let w = parse_calls(
            r#"{"file_path": "hello.go", "content": "package main"}"#,
            &t,
        );
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].name, "Write");
        assert_eq!(w[0].arguments["content"], "package main");

        let e = parse_calls(
            r#"{"file_path": "a.go", "old_string": "x", "new_string": "y"}"#,
            &t,
        );
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].name, "Edit");

        // Optional params along for the ride still resolve.
        let e2 = parse_calls(
            r#"{"file_path": "a.go", "old_string": "x", "new_string": "y", "replace_all": "true"}"#,
            &t,
        );
        assert_eq!(e2.len(), 1);
        assert_eq!(e2[0].name, "Edit");
    }

    /// The shape-2 gate. A key no offered tool declares, or a key set that fits
    /// two tools, or fits none — no call.
    #[test]
    fn a_name_less_object_that_is_ambiguous_or_foreign_is_not_a_call() {
        let t = schema_tools();
        // `content` is Write's alone, but `pattern` belongs to no tool that also
        // takes `content` — fits nothing.
        assert!(parse_calls(r#"{"content": "x", "pattern": "*.go"}"#, &t).is_empty());
        // A foreign key: no tool has `mode`.
        assert!(parse_calls(r#"{"file_path": "a", "mode": "rw"}"#, &t).is_empty());
        // No tools with judgeable schemas at all.
        assert!(parse_calls(
            r#"{"file_path": "a", "content": "b"}"#,
            &tools(&["Write", "Read"])
        )
        .is_empty());
    }

    /// `{"file_path": …}` alone fits only `Read` among the built-ins (Write and
    /// Edit need more required params), so it resolves — the uniqueness gate is
    /// what keeps this from being a guess.
    #[test]
    fn a_lone_file_path_resolves_to_read() {
        let calls = parse_calls(r#"{"file_path": "notes.md"}"#, &schema_tools());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    /// Still not reached: a `{"ToolName": …}` whose JSON is truncated —
    /// `serde_json::from_str` fails before any shape check runs (#118, needs a
    /// lenient parse pass).
    #[test]
    fn a_truncated_keyed_object_is_still_not_fixed() {
        let t = schema_tools();
        assert!(parse_calls(
            r#"{"Read": {"file_path": "a.go"}, "Edit": {"file_path": "a.go""#,
            &t
        )
        .is_empty());
    }
}
