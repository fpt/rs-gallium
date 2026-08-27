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
    // Each candidate carries whether it is the *whole* reply. Shape 2 in
    // [`keyed_by_tool_name`] needs that: an object with no tool name anywhere in
    // it is only recoverable when the model wrote nothing else, or a JSON example
    // quoted inside an explanation becomes a call (see that function).
    let mut candidates: Vec<(String, bool)> = Vec::new();
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        candidates.push((trimmed.to_string(), true));
    }
    if let Some(block) = first_balanced_json(text) {
        if candidates.first().map(|(c, _)| c != &block).unwrap_or(true) {
            let sole = is_sole_content(text, &block);
            candidates.push((block, sole));
        }
    }

    for (candidate, sole) in candidates {
        if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
            let calls = extract_calls(&val, tools, sole);
            if !calls.is_empty() {
                return calls;
            }
        }
    }
    Vec::new()
}

/// Whether `block` is the entire reply, discounting a code fence around it:
/// models habitually wrap their JSON in ```` ```json ````, and that is still a
/// reply that says nothing but the call. Anything else left over — a sentence
/// before it, a second block after — means the JSON was quoted *inside* prose.
fn is_sole_content(text: &str, block: &str) -> bool {
    strip_code_fence(text.trim()).trim() == block
}

/// Remove a leading ```` ```lang ```` line and the matching trailing ```` ``` ````.
/// Returns `s` unchanged when it is not fenced.
fn strip_code_fence(s: &str) -> &str {
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    let Some(rest) = rest.strip_suffix("```") else {
        return s;
    };
    // Whatever precedes the first newline is the language tag, not content.
    match rest.find('\n') {
        Some(i) => &rest[i + 1..],
        None => rest,
    }
}

/// Pull ToolCallInfo out of a parsed JSON value in any of the shapes a model
/// might emit: a bare object, an array of objects, `{"tool_calls": [...]}`,
/// and either `{name, arguments}` or `{function: {name, arguments}}`.
///
/// `sole` says whether this value was the model's whole reply; it gates the
/// name-less shape in [`keyed_by_tool_name`] and nothing else.
fn extract_calls(val: &Value, tools: &[ToolDefinition], sole: bool) -> Vec<ToolCallInfo> {
    fn one(v: &Value, tools: &[ToolDefinition]) -> Option<ToolCallInfo> {
        let obj = v.as_object()?;
        let (name, raw_args) = if let Some(f) = obj.get("function").and_then(|f| f.as_object()) {
            (
                f.get("name")?.as_str()?.to_string(),
                f.get("arguments").cloned(),
            )
        } else {
            let name = obj.get("name")?.as_str()?.to_string();
            let raw_args = obj.get("arguments").cloned();
            // A bare `name` key is the call's name only when something says the
            // object is *about* a call: an `arguments` sibling, or a value that
            // names a tool we offer. Otherwise `name` is an ordinary parameter —
            // `LookupSkill{action, name}` is a built-in — and taking it as the
            // call's name both invented a call to a tool nobody offers and hid
            // shape 2 below, which would have bound the object correctly.
            if raw_args.is_none() && find_tool(&name, tools).is_none() {
                return None;
            }
            (name, raw_args)
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
        Value::Array(arr) => arr.iter().filter_map(|v| one(v, tools)).collect(),
        Value::Object(o) if o.contains_key("tool_calls") => o
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| one(v, tools)).collect())
            .unwrap_or_default(),
        Value::Object(_) => {
            let calls = one(val, tools).into_iter().collect::<Vec<_>>();
            if calls.is_empty() {
                keyed_by_tool_name(val, tools, sole)
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
///    underscores. What counts as the arguments is [`arguments_for`].
///
/// 2. **The argument object with no name at all**:
///    `{"file_path": "hello.go", "content": "package main"}`. This is what LFM2.5
///    sends for a write (#118). Two gates, and both are load-bearing:
///
///    - the key set must fit **exactly one** offered tool's schema — every
///      `required` parameter present, no key outside that tool's `properties`,
///      and that tool must declare at least one required parameter
///      ([`args_match_unique_tool`]);
///    - it must be the model's **whole reply** (`sole`). Shape 1 needs no such
///      gate because its keys are tool *names*, a token a model does not write
///      unless it is calling something. Shape 2's keys are ordinary words —
///      `path`, `limit`, `command` — so a JSON example quoted inside an
///      explanation ("pass it `{"file_path": "notes.md"}`") would otherwise be
///      extracted by [`first_balanced_json`] and executed as a `Read`.
///
///    A whole reply that is exactly `{"command": …}` still binds to `Bash`,
///    which is the right reading of that reply. Note that the approval broker is
///    *not* the backstop it looks like here: `BashTool` whitelists the commands
///    it considers safe and only asks about the rest, so a mis-bound
///    `{"command": "ls -la"}` would have run. That is the `sole` gate's job, not
///    the broker's.
fn keyed_by_tool_name(val: &Value, tools: &[ToolDefinition], sole: bool) -> Vec<ToolCallInfo> {
    let obj = match val.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => return Vec::new(),
    };
    // Shape 1: every key names a tool, and every value is that tool's arguments.
    let shape_1: Option<Vec<ToolCallInfo>> = obj
        .iter()
        .map(|(key, value)| {
            let tool = find_tool(key, tools)?;
            Some(ToolCallInfo {
                id: String::new(),
                name: key.clone(),
                arguments: arguments_for(tool, value)?,
            })
        })
        .collect();
    if let Some(calls) = shape_1 {
        return calls;
    }
    // Shape 2: the whole object is one tool's arguments, name dropped.
    if sole {
        args_match_unique_tool(obj, tools)
    } else {
        Vec::new()
    }
}

/// The arguments a shape-1 value carries for `tool`.
///
/// An object is already the arguments. An **array** is the one other thing a
/// live model has been seen sending: LFM2.5 answers a multi-file edit with
/// `{"MultiEdit": [ {…} ]}` — the tool's single parameter, unwrapped. That is
/// recoverable *because the tool name is already known*, which is what makes it
/// unlike shape 2: the array binds to the tool's one required parameter only
/// when that parameter is declared an array, so nothing is guessed. A tool with
/// two required parameters, or a scalar one, is left alone.
///
/// Anything else is not arguments, so `{"Read": "a.txt"}` is still left alone
/// rather than guessed at — no model has been observed sending it, and this
/// module's bar is an observed shape, not a plausible one. Returning `None` here
/// fails shape 1 for the whole object, which is deliberate: a reply mixing a
/// real call with a value nobody can read is not a batch to half-execute.
fn arguments_for(tool: &ToolDefinition, value: &Value) -> Option<Value> {
    if value.is_object() {
        return Some(value.clone());
    }
    let array = value.as_array()?;
    let param = sole_array_parameter(tool)?;
    let mut args = serde_json::Map::new();
    args.insert(param, Value::Array(array.clone()));
    Some(Value::Object(args))
}

/// The name of `tool`'s single required parameter, when it has exactly one and
/// declares it an array.
fn sole_array_parameter(tool: &ToolDefinition) -> Option<String> {
    let schema = tool.parameters.as_object()?;
    let [Value::String(name)] = schema.get("required")?.as_array()?.as_slice() else {
        return None;
    };
    let props = schema.get("properties")?.as_object()?;
    if props.get(name)?.get("type")?.as_str()? == "array" {
        Some(name.clone())
    } else {
        None
    }
}

/// `ToolRegistry`'s name resolution, as the gates here need it: exact, else
/// ignoring case and underscores, so this is never stricter than the resolution
/// that follows it.
fn find_tool<'a>(name: &str, tools: &'a [ToolDefinition]) -> Option<&'a ToolDefinition> {
    tools
        .iter()
        .find(|t| t.name == name || normalized(&t.name) == normalized(name))
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
///
/// **A tool with no `required` parameters never matches either**, because it
/// would match nearly everything: `LS{ignore, limit, path}` requires nothing, so
/// a bare `{"limit": 10}` fitted it and only it, and a config fragment in a reply
/// became a directory listing. A tool that demands nothing cannot be identified
/// by what a caller supplied.
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
            Some(p) => p,
            None => return false,
        };
        let required = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|r| r.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        if required.is_empty() {
            return false;
        }
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

    /// The built-ins' real parameter schemas, as `create_default_registry`
    /// reports them — the gates here turn on `properties` and `required`, so a
    /// subset of the catalog would test a uniqueness that does not ship.
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
            mk("LS", &["ignore", "limit", "path"], &[]),
            mk(
                "Grep",
                &[
                    "pattern",
                    "path",
                    "glob",
                    "output_mode",
                    "case_insensitive",
                    "limit",
                ],
                &["pattern"],
            ),
            mk("Bash", &["command", "timeout_ms"], &["command"]),
            mk(
                "Tasks",
                &["action", "task_id", "subject", "description", "status"],
                &["action"],
            ),
            mk("LookupSkill", &["action", "name"], &["action"]),
            // The one built-in whose single required parameter is an array,
            // which is what `arguments_for` binds a shape-1 array value to.
            ToolDefinition {
                name: "MultiEdit".to_string(),
                description: String::new(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"edits": {"type": "array", "items": {"type": "object"}}},
                    "required": ["edits"],
                }),
            },
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
        // Transcribed from a live LFM2.5 reply, unclosed outer object and all —
        // the value of this pin is its provenance, so it stays byte-for-byte.
        assert!(parse_calls(
            r#"{"Read": {"file_path": "a.go"}, "Edit": {"file_path": "a.go"}"#,
            &t
        )
        .is_empty());
    }

    /// The reason shape 2 is gated on being the whole reply: `first_balanced_json`
    /// digs a block out of prose, and shape 2's keys are ordinary words, so an
    /// explanation that quotes an argument object would otherwise *execute* it.
    #[test]
    fn a_json_example_quoted_in_prose_is_not_a_call() {
        let t = schema_tools();
        assert!(parse_calls(
            r#"To open it, pass {"file_path": "notes.md"} to the reader."#,
            &t
        )
        .is_empty());
        // Trailing prose is prose too.
        assert!(parse_calls(r#"{"file_path": "notes.md"} — that one."#, &t).is_empty());
        // The same object alone is still recovered, which is the point of the gate
        // rather than of a ban.
        let calls = parse_calls(r#"{"file_path": "notes.md"}"#, &t);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    /// A fence is not prose: a model wrapping its whole reply in ```json is still
    /// saying nothing but the call.
    #[test]
    fn a_fenced_whole_reply_still_binds() {
        let t = schema_tools();
        let calls = parse_calls(
            "```json\n{\"file_path\": \"hello.go\", \"content\": \"package main\"}\n```",
            &t,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        // Unlabelled fences too.
        let bare = parse_calls("```\n{\"file_path\": \"notes.md\"}\n```", &t);
        assert_eq!(bare.len(), 1);
        assert_eq!(bare[0].name, "Read");
        // Shape 1 never needed the gate, fenced or not.
        let keyed = parse_calls("```json\n{\"Read\": {\"file_path\": \"a\"}}\n```", &t);
        assert_eq!(keyed.len(), 1);
        assert_eq!(keyed[0].name, "Read");
    }

    /// `LS` requires nothing, so before the `required`-non-empty rule any subset
    /// of its properties fitted it and only it — a bare `{"limit": 10}` in a reply
    /// became a directory listing.
    #[test]
    fn a_tool_that_requires_nothing_is_never_the_unique_fit() {
        let t = schema_tools();
        assert!(parse_calls(r#"{"limit": "10"}"#, &t).is_empty());
        assert!(parse_calls(r#"{"path": "src"}"#, &t).is_empty());
        assert!(parse_calls(r#"{"ignore": "target"}"#, &t).is_empty());
    }

    /// A `name` *parameter* is not the call's name. `LookupSkill{action, name}` is
    /// a built-in, so reading its `name` as the tool called both invented a call
    /// to a tool nobody offers and hid the shape-2 bind that was right.
    #[test]
    fn a_name_parameter_is_not_the_calls_name() {
        let t = schema_tools();
        let calls = parse_calls(r#"{"action": "get", "name": "sweep-edit"}"#, &t);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "LookupSkill");
        assert_eq!(calls[0].arguments["name"], "sweep-edit");

        // The envelope gallium actually asks for is untouched: an `arguments`
        // sibling means `name` is the name, whatever it names.
        let env = parse_calls(r#"{"name": "Read", "arguments": {"file_path": "a"}}"#, &t);
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].name, "Read");
        let unknown = parse_calls(r#"{"name": "mcp__thing", "arguments": {}}"#, &t);
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].name, "mcp__thing");

        // And a lone `name` that does name an offered tool is still a call.
        let lone = parse_calls(r#"{"name": "Read"}"#, &t);
        assert_eq!(lone.len(), 1);
        assert_eq!(lone[0].name, "Read");
    }

    /// The shape a live LFM2.5 sends for `refactoring` on llama.cpp: the tool
    /// name keys the object, but the value is the *unwrapped* array its one
    /// required parameter takes.
    #[test]
    fn a_shape_one_array_value_binds_to_the_tools_one_array_parameter() {
        let t = schema_tools();
        let calls = parse_calls(
            r#"{"MultiEdit": [{"file_path": "counter.go", "old_string": "x", "new_string": "y"}]}"#,
            &t,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "MultiEdit");
        assert_eq!(calls[0].arguments["edits"][0]["file_path"], "counter.go");

        // Not extended to scalars: `Read` takes one required parameter too, but
        // no model has been seen sending this and the module's bar is observation.
        assert!(parse_calls(r#"{"Read": "a.txt"}"#, &t).is_empty());
        // A tool whose required set is not a single array parameter is left alone.
        assert!(parse_calls(r#"{"Write": ["a.txt", "hi"]}"#, &t).is_empty());
    }
}
