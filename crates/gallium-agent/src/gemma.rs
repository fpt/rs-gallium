//! Shared Gemma 4 native tool-call parsing.
//!
//! Both local backends speak Gemma's native tool wire format — `llm_local`
//! (llama.cpp, via the GGUF's embedded chat template) and
//! `protocol::GemmaProtocol` (gallium, hand-written template). The two used to
//! carry independent parsers for it; this module owns the format knowledge so
//! they parse it identically.
//!
//! Wire format:
//! `<|tool_call>call:NAME{key:<|"|>strval<|"|>, key2:42}<tool_call|>`
//! where `<|"|>` delimits string values (so a value may contain commas/braces).
//!
//! Names are returned verbatim. The alias helpers ([`normalise_tool_name`],
//! [`normalise_path_args`]) are opt-in: gallium applies them (its small Gemma
//! models hallucinate names like `write_file`); the llama.cpp path keeps names
//! exact so mixed-case MCP tool names still match.

use serde_json::{Map, Value};

/// The Gemma string-value delimiter token.
const STR_DELIM: &str = "<|\"|>";

/// One parsed native tool call, with the name exactly as the model emitted it.
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaCall {
    pub name: String,
    pub arguments: Value,
}

/// Parse every `call:NAME{...}` native tool call in `text`, in order. The
/// `<|tool_call>` marker is optional — matching the `call:` form is enough, and
/// both engines' real outputs contain it.
pub fn parse_native_tool_calls(text: &str) -> Vec<GemmaCall> {
    use std::sync::OnceLock;
    // Match only the *opening* `call:NAME{` (names use the MCP charset
    // letters/digits/._-). The body is then scanned by [`scan_call_body`], which
    // is `<|"|>`-string-aware: a string argument value may itself contain `{`/`}`
    // (any code the model writes as an arg), so a `[^{}]*` body capture would
    // silently fail to match the whole call and drop it — the model's turn then
    // leaks raw `<|tool_call>` markup as a "text" answer.
    static OPEN_RE: OnceLock<regex::Regex> = OnceLock::new();
    let open_re =
        OPEN_RE.get_or_init(|| regex::Regex::new(r"call:\s*([A-Za-z0-9_.\-]+)\s*\{").unwrap());

    let mut calls = Vec::new();
    let mut from = 0;
    while let Some(cap) = open_re.captures(&text[from..]) {
        let whole = cap.get(0).unwrap();
        let name = cap[1].to_string();
        let body_start = from + whole.end();
        match scan_call_body(text, body_start) {
            Some((body, next)) => {
                calls.push(GemmaCall {
                    name,
                    arguments: parse_kv_args(&body),
                });
                from = next;
            }
            None => {
                // No matching close brace: consume the remainder as the body so a
                // truncated final call is still recovered, then stop.
                calls.push(GemmaCall {
                    name,
                    arguments: parse_kv_args(&text[body_start..]),
                });
                break;
            }
        }
    }
    calls
}

/// Scan the body of a `call:NAME{ ... }` starting just after the opening `{`
/// (at byte offset `start`). Returns the body text (between the braces) and the
/// offset just past the matching `}`. Brace depth is tracked only **outside**
/// string values, so braces inside a string argument value are ignored.
/// Returns `None` if no matching close brace is found.
///
/// Both string syntaxes [`parse_kv_args`] accepts are recognised here, or the
/// two disagree about where the call ends: `call:Write{content:"a } b"}` would
/// close on the brace *inside* the value and silently truncate it.
///
/// An ordinary quote only opens a string where one may *start* — at a key, or
/// just past that key's `:` — matching how [`parse_kv_args`] decides the same
/// thing. A quote anywhere else is an apostrophe in a bare value
/// (`command:echo it's fine`), and treating that as an opening quote would
/// swallow the call's closing brace.
///
/// Tracking the key as a string too is what keeps a `:` *inside* a quoted key
/// from being read as the separator: `{"a:":"x } y"}` would otherwise open its
/// value string one quote early, leaving the real value's `}` counted.
fn scan_call_body(text: &str, start: usize) -> Option<(String, usize)> {
    /// Where in a `key:value` pair the scanner is. Only the two `*Start` states
    /// let an ordinary quote open a string, which is what confines quoting to
    /// the positions [`parse_kv_args`] also reads it in.
    #[derive(PartialEq, Clone, Copy)]
    enum At {
        KeyStart,
        KeyInside,
        ValueStart,
        ValueInside,
    }
    use At::*;

    let mut depth = 1usize;
    let mut in_str = false; // inside `<|"|>` … `<|"|>`
    let mut quote: Option<char> = None; // inside an ordinary "…" / '…'
    let mut at = KeyStart;
    let mut idx = start;
    while idx < text.len() {
        let rest = &text[idx..];
        if quote.is_none() {
            if let Some(after) = rest.strip_prefix(STR_DELIM) {
                in_str = !in_str;
                at = ValueInside;
                idx = text.len() - after.len();
                continue;
            }
        }
        // `idx` is always on a char boundary: `start` is right after `{`, and we
        // only advance by whole chars or by the (ASCII) `STR_DELIM` length.
        let ch = rest.chars().next().unwrap();
        idx += ch.len_utf8();

        if in_str {
            continue;
        }
        if let Some(q) = quote {
            if ch == '\\' {
                // Skip the escaped character whole, so `\"` does not close.
                if let Some(esc) = text[idx..].chars().next() {
                    idx += esc.len_utf8();
                }
            } else if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '{' => {
                depth += 1;
                at = KeyStart;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((text[start..idx - ch.len_utf8()].to_string(), idx));
                }
                at = ValueInside;
            }
            ',' => at = KeyStart,
            // Only the separator ends the key. A colon *inside* a bare value —
            // `url:http://x` — must not re-arm the value position, or the next
            // apostrophe would open a string.
            ':' if at == KeyStart || at == KeyInside => at = ValueStart,
            '"' | '\'' if at == KeyStart => {
                quote = Some(ch);
                at = KeyInside;
            }
            '"' | '\'' if at == ValueStart => {
                quote = Some(ch);
                at = ValueInside;
            }
            c if c.is_whitespace() => {}
            _ => {
                at = match at {
                    KeyStart | KeyInside => KeyInside,
                    ValueStart | ValueInside => ValueInside,
                }
            }
        }
    }
    None
}

/// Parse a `key:<|"|>strval<|"|>, key2:scalar, ...` body into a JSON object.
/// String values keep everything between the `<|"|>` delimiters (commas
/// included); bare values are coerced by [`parse_scalar`].
///
/// Ordinary `"` / `'` quotes are accepted wherever `<|"|>` is, on keys and on
/// values, because models mix the two: a Gemma emitting `call:LS{path:"."}`
/// used to yield the *three-character* path `"."`, which every path-taking tool
/// then failed to find. Quoting is a syntax the parser has to strip, and
/// leaving it in produces an argument that looks plausible in a log and cannot
/// possibly work.
pub fn parse_kv_args(inner: &str) -> Value {
    let mut map = Map::new();
    let mut s = inner;

    loop {
        s = s.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
        if s.is_empty() {
            break;
        }

        // A quoted key is scanned as a string, so a `:` *inside* it is part of
        // the key rather than the separator. Reaching for `find(':')` first read
        // `"a:b":1` as the key `"a` and the value `b":1` — quoted keys are a
        // syntax this accepts, so it has to accept the ones containing a colon.
        let (key, after_key) = match scan_quoted(s) {
            Some((key, rest)) => (key, rest),
            None => match s.find(':') {
                Some(p) => (s[..p].trim().to_string(), &s[p..]),
                None => break,
            },
        };
        // Whatever the key's shape, its value begins past the separator.
        s = match after_key.trim_start().strip_prefix(':') {
            Some(rest) => rest.trim_start(),
            None => break,
        };
        if key.is_empty() {
            break;
        }

        if let Some(rest) = s.strip_prefix(STR_DELIM) {
            // String value enclosed in <|"|>...<|"|>.
            match rest.find(STR_DELIM) {
                Some(end) => {
                    map.insert(key, Value::String(rest[..end].to_string()));
                    s = &rest[end + STR_DELIM.len()..];
                }
                None => {
                    // Malformed: consume the remainder as the value.
                    map.insert(key, Value::String(rest.to_string()));
                    break;
                }
            }
        } else if let Some((value, rest)) = scan_quoted(s) {
            map.insert(key, Value::String(value));
            s = rest;
        } else {
            // Bare value: read until the next comma or the end.
            let end = s.find(',').unwrap_or(s.len());
            map.insert(key, parse_scalar(s[..end].trim()));
            s = &s[end..];
        }
    }

    Value::Object(map)
}

/// If `s` opens with an ordinary `"` or `'`, return the quoted value (escapes
/// resolved) and the remainder just past the closing quote. `None` when `s` is
/// not quoted at all, so the caller falls back to bare-value parsing.
///
/// An unterminated quote takes the rest of the body as the value, matching what
/// the `<|"|>` branch does with the same malformation: whatever the model meant,
/// it did not mean to include the opening quote in the string.
fn scan_quoted(s: &str) -> Option<(String, &str)> {
    let quote = match s.chars().next() {
        Some(c @ ('"' | '\'')) => c,
        _ => return None,
    };
    let body = &s[quote.len_utf8()..];

    let mut out = String::new();
    let mut chars = body.char_indices();
    while let Some((i, c)) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some((_, esc)) => out.push_str(&unescape(esc)),
                // Trailing backslash: keep it and stop.
                None => {
                    out.push('\\');
                    return Some((out, ""));
                }
            }
        } else if c == quote {
            return Some((out, &body[i + c.len_utf8()..]));
        } else {
            out.push(c);
        }
    }
    Some((out, ""))
}

/// Resolve the character after a backslash. Only the escapes quoting itself
/// *requires* are resolved — the quote characters, and the backslash that
/// escapes them. Everything else keeps both characters.
///
/// Deliberately not `\n` / `\t` / `\r`: this is a path- and command-carrying
/// format, and those are the opening letters of `\temp`, `\node_modules`,
/// `\repos`, so resolving them turns `"C:\temp\notes.txt"` into
/// `C:<TAB>emp<LF>otes.txt` — which renders back as the original path in a log
/// and cannot possibly work. It also matches the `<|"|>` syntax, which takes its
/// value verbatim; a model that means a real newline has that syntax available.
fn unescape(c: char) -> String {
    match c {
        '\\' => "\\".to_string(),
        '"' => "\"".to_string(),
        '\'' => "'".to_string(),
        other => format!("\\{other}"),
    }
}

/// Coerce a bare (non-string) Gemma value: bool / null / integer / float, else
/// keep it as a string.
pub fn parse_scalar(s: &str) -> Value {
    match s {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ => {
            if let Ok(n) = s.parse::<i64>() {
                Value::from(n)
            } else if let Ok(f) = s.parse::<f64>() {
                Value::from(f)
            } else {
                Value::String(s.to_string())
            }
        }
    }
}

/// Fold common tool-name aliases a Gemma model may hallucinate onto the
/// registered names (e.g. `write_file` → `Write`). Opt-in per caller.
///
/// Only whole invented names are folded here. Plain case and underscore drift —
/// `read` for `Read`, `multi_edit` for `MultiEdit` — is handled for every
/// backend by the registry's own lookup, so it does not need a row each.
///
/// Note `ls` is NOT an alias: gallium registers a real `LS` tool, so an `ls`
/// call must route to it verbatim (folding it onto `Glob` used to send a bogus
/// `file_path` arg to a tool that wants `pattern`, wedging the ReAct loop).
pub fn normalise_tool_name(name: &str) -> String {
    match name {
        "write_file" | "create_file" | "file_write" | "write_to_file" | "writefile"
        | "write_tool" | "writetool" | "write_content" | "create" => "Write".to_string(),
        "read_file" | "file_read" | "readfile" | "open_file" | "read_tool" => "Read".to_string(),
        "list_files" | "list_file" | "find_files" | "glob_tool" => "Glob".to_string(),
        "edit_file" | "file_edit" | "update_file" | "patch_file" | "edit_tool" => {
            "Edit".to_string()
        }
        _ => name.to_string(),
    }
}

/// Fold the short `file` / `path` argument aliases onto `file_path` — but only
/// for the file tools whose canonical parameter IS `file_path`. Other tools
/// (`LS`, `Glob`, MCP tools, …) legitimately take `path`-named params that must
/// pass through untouched.
///
/// Matched case-insensitively for the same reason the registry is: the name may
/// arrive as the model wrote it, not as it was advertised.
pub fn normalise_path_args(tool: &str, args: &mut Value) {
    if !matches!(
        tool.to_lowercase().replace('_', "").as_str(),
        "read" | "write" | "edit" | "multiedit"
    ) {
        return;
    }
    if let Some(map) = args.as_object_mut() {
        if let Some(v) = map.remove("file") {
            map.entry("file_path".to_string()).or_insert(v);
        }
        if let Some(v) = map.remove("path") {
            map.entry("file_path".to_string()).or_insert(v);
        }
    }
}

/// Remove Gemma 4 thinking blocks from a message body.
///
/// Shared by both local backends, because both need it for different reasons
/// and one of them had it and the other did not. The native candle path strips
/// history so thinking never re-enters a prompt (the model card requires that);
/// the llama.cpp path strips the reply so a `<|channel>thought` wrapper does not
/// reach the user, which it did.
///
/// Both forms the model may emit:
///   - `<|think|>…<|/think|>` paired wrappers
///   - `<|channel>…<channel|>` (retain only the text after the last channel close)
///
/// Applied to assistant history *and* to the freshly parsed response stored in
/// memory, so thinking content never re-enters a subsequent prompt.
pub fn strip_thinking_blocks(s: &str) -> String {
    // 1. Drop everything up to and including the last `<channel|>` (Gemma channel close).
    let after_channel = match s.rfind("<channel|>") {
        Some(pos) => &s[pos + "<channel|>".len()..],
        None => s,
    };

    // 2. Remove paired `<|think|>…<|/think|>` blocks (non-greedy, iterative).
    let mut out = String::with_capacity(after_channel.len());
    let mut rest = after_channel;
    while let Some(start) = rest.find("<|think|>") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + "<|think|>".len()..];
        match after_open.find("<|/think|>") {
            Some(end) => {
                rest = &after_open[end + "<|/think|>".len()..];
            }
            None => {
                // Unclosed think block — drop everything from here (the model didn't
                // finish thinking before hitting EOS; safest to discard the tail).
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_native_call() {
        let calls = parse_native_tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc"); // verbatim, not normalised
        assert_eq!(calls[0].arguments["query"], "mcp-go");
    }

    #[test]
    fn parses_mixed_string_and_scalar_args() {
        let calls = parse_native_tool_calls(
            "<|tool_call>call:grep{pattern:<|\"|>foo<|\"|>, limit:50}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "foo");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    #[test]
    fn string_value_may_contain_commas() {
        let v = parse_kv_args("msg:<|\"|>a, b, c<|\"|>, n:3");
        assert_eq!(v["msg"], "a, b, c");
        assert_eq!(v["n"], 3);
    }

    #[test]
    fn parses_multiple_calls() {
        let calls = parse_native_tool_calls(
            "call:read{file_path:<|\"|>a.rs<|\"|>} call:glob{pattern:<|\"|>*.rs<|\"|>}",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[1].name, "glob");
    }

    #[test]
    fn string_value_may_contain_braces() {
        // Regression: a string arg holding content with `{`/`}` (source code,
        // JSON, a shell command, …) must not defeat the body scan — it silently
        // dropped the call and leaked raw `<|tool_call>` markup as a text answer
        // for gemma-4-26B.
        let content = "{\n  \"level\": 1,\n  \"spawn\": { \"x\": 5 }\n}\n";
        let raw = format!(
            "<|channel>thought<channel|><|tool_call>call:write{{file_path:<|\"|>data.json<|\"|>,content:<|\"|>{content}<|\"|>}}<tool_call|>"
        );
        let calls = parse_native_tool_calls(&raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
        assert_eq!(calls[0].arguments["file_path"], "data.json");
        assert_eq!(calls[0].arguments["content"], content);
    }

    #[test]
    fn multiple_calls_with_braced_string_values() {
        let raw = "call:write{file_path:<|\"|>run.sh<|\"|>,content:<|\"|>for i in {1..3}; do echo $i; done<|\"|>} \
                   call:read{file_path:<|\"|>run.sh<|\"|>}";
        let calls = parse_native_tool_calls(raw);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "write");
        assert_eq!(
            calls[0].arguments["content"],
            "for i in {1..3}; do echo $i; done"
        );
        // Verbatim: the parser reports what the model wrote, and the registry
        // is what resolves `read` onto the registered `Read`.
        assert_eq!(calls[1].name, "read");
        assert_eq!(calls[1].arguments["file_path"], "run.sh");
    }

    /// Regression: `call:LS{path:"."}` — ordinary quotes instead of `<|"|>` —
    /// used to yield the literal three-character path `"."`, so LS looked for a
    /// directory named `"."` and every turn that started by listing the cwd
    /// died on the first tool call.
    #[test]
    fn ordinary_quotes_are_stripped_from_values() {
        let calls = parse_native_tool_calls("<|tool_call>call:LS{path:\".\"}<tool_call|>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "LS");
        assert_eq!(calls[0].arguments["path"], ".");
    }

    #[test]
    fn single_quotes_are_stripped_too() {
        let v = parse_kv_args("pattern:'*.rs', limit:5");
        assert_eq!(v["pattern"], "*.rs");
        assert_eq!(v["limit"], 5);
    }

    #[test]
    fn quoted_keys_are_stripped() {
        // A model that reaches for JSON syntax inside the gemma body.
        let v = parse_kv_args("\"path\": \".\", \"limit\": 5");
        assert_eq!(v["path"], ".");
        assert_eq!(v["limit"], 5);
    }

    #[test]
    fn quoted_value_may_contain_commas_and_an_escaped_quote() {
        let v = parse_kv_args("msg:\"a, b \\\" c\", n:3");
        assert_eq!(v["msg"], "a, b \" c");
        assert_eq!(v["n"], 3);
    }

    /// Every escape but the quotes keeps both characters, so a Windows path
    /// survives — including the `\t` / `\n` prefixes that a C-style unescape
    /// would turn into a tab and a newline.
    #[test]
    fn only_quote_escapes_are_resolved() {
        let v = parse_kv_args("file_path:\"C:\\temp\\notes.txt\"");
        assert_eq!(v["file_path"], "C:\\temp\\notes.txt");
        let v = parse_kv_args("file_path:\"C:\\Users\\x\"");
        assert_eq!(v["file_path"], "C:\\Users\\x");
    }

    /// A brace inside an ordinary-quoted value must not end the call: the body
    /// scanner used to close on it, so `content` arrived truncated to `a ` and
    /// the rest of the model's argument was lost.
    #[test]
    fn braces_inside_an_ordinary_quoted_value_are_ignored() {
        let calls = parse_native_tool_calls("call:Write{content:\"a } b\"}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "a } b");

        let calls = parse_native_tool_calls("call:Write{content:\"if (x) { y\"}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "if (x) { y");
    }

    /// …but an apostrophe in a *bare* value is not an opening quote, or it would
    /// swallow the call's own closing brace.
    #[test]
    fn an_apostrophe_in_a_bare_value_does_not_open_a_string() {
        let calls = parse_native_tool_calls("call:Bash{command:echo it's fine}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], "echo it's fine");
    }

    /// A colon inside a *bare* value is not the key separator either, so it must
    /// not re-arm the value position and let the next quote open a string.
    #[test]
    fn a_colon_in_a_bare_value_is_not_a_separator() {
        let calls = parse_native_tool_calls("call:Fetch{url:http://x/it's, n:1}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["url"], "http://x/it's");
        assert_eq!(calls[0].arguments["n"], 1);
    }

    /// A quoted key is a string, so a `:` inside it belongs to the key rather
    /// than separating it from the value. Reaching for the first raw colon read
    /// `"a:b":1` as the key `"a` and the value `b":1`.
    #[test]
    fn a_quoted_key_may_contain_a_colon() {
        let v = parse_kv_args("\"a:b\":1");
        assert_eq!(v["a:b"], 1);
    }

    /// The body scanner has to agree: with the key's colon read as the
    /// separator, the key's *closing* quote opened the value string, and the
    /// real value's `}` was counted — truncating the call.
    #[test]
    fn a_quoted_key_ending_in_a_colon_does_not_truncate_the_call() {
        let calls = parse_native_tool_calls("call:Tool{\"a:\":\"x } y\"}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["a:"], "x } y");

        let calls = parse_native_tool_calls("call:Tool{\"a:b\":\"x } y\", n:1}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["a:b"], "x } y");
        assert_eq!(calls[0].arguments["n"], 1);
    }

    /// An escaped quote inside a quoted value does not close it for the body
    /// scanner either, so the scanner and [`parse_kv_args`] agree on the end.
    #[test]
    fn escaped_quote_does_not_end_the_call_body() {
        let calls = parse_native_tool_calls("call:Write{content:\"say \\\" now\", n:1}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "say \" now");
        assert_eq!(calls[0].arguments["n"], 1);
    }

    /// A quote that never closes takes the rest of the body — the same recovery
    /// the `<|"|>` branch makes, and never leaves the opening quote in the value.
    #[test]
    fn unterminated_quote_consumes_the_remainder() {
        let v = parse_kv_args("path:\"/tmp/x");
        assert_eq!(v["path"], "/tmp/x");
    }

    /// Bare values are still bare: quote stripping must not touch them.
    #[test]
    fn bare_values_are_unaffected() {
        let v = parse_kv_args("path:., limit:5, ok:true");
        assert_eq!(v["path"], ".");
        assert_eq!(v["limit"], 5);
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn plain_prose_is_not_a_call() {
        assert!(parse_native_tool_calls("Sure, I'll call the search tool for you.").is_empty());
    }

    #[test]
    fn name_and_path_aliases_fold() {
        assert_eq!(normalise_tool_name("write_file"), "Write");
        assert_eq!(normalise_tool_name("search-godoc"), "search-godoc");
        let mut args = serde_json::json!({"file": "x.rs"});
        normalise_path_args("Read", &mut args);
        assert_eq!(args["file_path"], "x.rs");
        assert!(args.get("file").is_none());
    }

    /// The model may write the name in whatever case it likes; the arg folding
    /// has to recognise the tool either way.
    #[test]
    fn path_args_fold_whatever_case_the_name_arrives_in() {
        for name in ["read", "Read", "multi_edit", "MultiEdit"] {
            let mut args = serde_json::json!({"file": "x.rs"});
            normalise_path_args(name, &mut args);
            assert_eq!(args["file_path"], "x.rs", "{name}");
        }
    }

    #[test]
    fn ls_is_a_real_tool_and_keeps_its_path_arg() {
        // gallium registers a real `ls` tool taking `path` — neither the name
        // nor the arg may be folded (this wedged the 26B file_read loop).
        assert_eq!(normalise_tool_name("ls"), "ls");
        let mut args = serde_json::json!({"path": "."});
        normalise_path_args("ls", &mut args);
        assert_eq!(args["path"], ".");
        assert!(args.get("file_path").is_none());
    }
}
