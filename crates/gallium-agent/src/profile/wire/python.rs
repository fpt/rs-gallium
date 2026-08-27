//! Python/Llama-style calls: `[name(arg=val, ...)]` or a bare `name(arg=val)`.
//!
//! Nothing gallium prompts for asks for this; some instruction-tuned models
//! reach for it anyway because their fine-tuning data used it, and it is
//! LFM2.5's own native format — its `<|tool_*|>` markers are control tokens that
//! llama.cpp decodes away, so a native call arrives here as a bare
//! `[Write(file_path='a.go', content='…')]`.
//!
//! **Parsed structurally, not scanned.** The format has no escaping rules of its
//! own and its arguments carry code, so a regex over `name\(([^)]*)\)` ended at
//! the first `)` in the payload — `Write(content='fmt.Println("hi")')` lost its
//! tail — and then matched every `funcName()` in the truncated remainder as a
//! further call. Both failures were silent: a phantom call to a tool nobody
//! offers, and a file written with half its content. So the scan here tracks
//! quotes, their backslash escapes, and bracket nesting, and a call that does
//! not parse whole yields *nothing* rather than something partial.

use serde_json::Value;

use crate::llm::ToolCallInfo;

/// Parse Python/Llama-style tool calls, but only when the whole reply is a call
/// list — a bare `name(...)` found inside prose matches sentences like "use the
/// read() function", so this gate is what keeps the format from inventing calls
/// out of documentation.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    let t = text.trim();
    if let Some(inner) = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return parse_list(inner);
    }
    parse_one(t).into_iter().collect()
}

/// The members of a `[...]` list, all or nothing: a list whose second member is
/// truncated is a reply that got cut off, and executing the first half of a
/// batch the model meant as a whole is worse than reporting nothing.
fn parse_list(inner: &str) -> Vec<ToolCallInfo> {
    let Some(parts) = split_top_level(inner, ',') else {
        return Vec::new();
    };
    let mut calls = Vec::new();
    for part in parts {
        if part.trim().is_empty() {
            continue; // A trailing comma, or an empty list.
        }
        match parse_one(part) {
            Some(call) => calls.push(call),
            None => return Vec::new(),
        }
    }
    calls
}

/// One `name(k=v, ...)` call, where the closing paren must be the last
/// character: the arguments between the parens have to balance, which is what
/// proves the paren found is the one that closes the call rather than one inside
/// a code payload.
fn parse_one(s: &str) -> Option<ToolCallInfo> {
    let s = s.trim();
    let open = s.find('(')?;
    let name = s[..open].trim();
    if !is_identifier(name) {
        return None;
    }
    let inner = s.strip_suffix(')')?.get(open + 1..)?;
    Some(ToolCallInfo {
        id: String::new(),
        name: name.to_string(),
        arguments: Value::Object(parse_args(inner)?),
    })
}

/// `k=v` pairs, split on top-level commas.
///
/// A part with no top-level `=` fails the **whole call**. It is a positional
/// argument, and there is no way to name it — dropping it silently, which the
/// regex parser did, is how a `Write` arrives with no `content` and truncates
/// the file it was supposed to fill.
fn parse_args(inner: &str) -> Option<serde_json::Map<String, Value>> {
    let mut args = serde_json::Map::new();
    for part in split_top_level(inner, ',')? {
        if part.trim().is_empty() {
            continue;
        }
        let (key, value) = split_once_top_level(part, '=')?;
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        args.insert(key.to_string(), parse_py_value(value.trim()));
    }
    Some(args)
}

/// ASCII identifier, as the format's own names are: a leading letter or
/// underscore, then letters, digits, or underscores.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Tracks quote and bracket state so a scan can tell structure from content.
#[derive(Default)]
struct Depth {
    brackets: Vec<char>,
    quote: Option<char>,
    escaped: bool,
}

impl Depth {
    /// True when nothing is open, so the next character is structure rather than
    /// part of an argument's value.
    fn top(&self) -> bool {
        self.quote.is_none() && self.brackets.is_empty()
    }

    fn feed(&mut self, c: char) {
        if let Some(quote) = self.quote {
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return;
        }
        match c {
            '\'' | '"' => self.quote = Some(c),
            '(' => self.brackets.push(')'),
            '[' => self.brackets.push(']'),
            '{' => self.brackets.push('}'),
            ')' | ']' | '}' => {
                if self.brackets.last() == Some(&c) {
                    self.brackets.pop();
                } else {
                    // A closer with no opener: not a nesting level, so leave a
                    // marker that cannot be popped and let the caller reject.
                    self.brackets.push('\0');
                }
            }
            _ => {}
        }
    }
}

/// Split `s` on every top-level `sep` — outside every quote and bracket.
///
/// `None` when a quote or bracket is left open, or a stray closer appeared: an
/// unterminated argument means the reply was truncated mid-call, and half a call
/// is not a call.
fn split_top_level(s: &str, sep: char) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut depth = Depth::default();
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if c == sep && depth.top() {
            parts.push(&s[start..i]);
            start = i + c.len_utf8();
            continue;
        }
        depth.feed(c);
    }
    if !depth.top() {
        return None;
    }
    parts.push(&s[start..]);
    Some(parts)
}

/// Split at the **first** top-level `sep`, keeping the rest intact — an argument
/// value may well contain the separator (`old_string='x = 1'`).
fn split_once_top_level(s: &str, sep: char) -> Option<(&str, &str)> {
    let mut depth = Depth::default();
    for (i, c) in s.char_indices() {
        if c == sep && depth.top() {
            return Some((&s[..i], &s[i + c.len_utf8()..]));
        }
        depth.feed(c);
    }
    None
}

/// Parse a Python-literal-ish value into JSON.
fn parse_py_value(v: &str) -> Value {
    let v = v.trim();
    if let Some(body) = quoted_body(v) {
        return Value::String(unescape(body));
    }
    match v {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    if let Ok(n) = v.parse::<i64>() {
        return Value::from(n);
    }
    if let Ok(f) = v.parse::<f64>() {
        return Value::from(f);
    }
    serde_json::from_str::<Value>(v).unwrap_or_else(|_| Value::String(v.to_string()))
}

/// The body of a quoted literal, if `v` is exactly one — the closing quote has
/// to be the one that matches the opener, so `'a' + 'b'` is not a string and is
/// left for the JSON/raw fallbacks below.
fn quoted_body(v: &str) -> Option<&str> {
    let quote = v.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = v.strip_prefix(quote)?.strip_suffix(quote)?;
    let mut escaped = false;
    for c in body.chars() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == quote {
            return None;
        }
    }
    Some(body)
}

/// Decode the string escapes a model actually emits.
///
/// This is the format's own rule, not a repair: `\n` in a quoted literal *is* a
/// newline here, which is why a code payload survives this format when the same
/// payload over-escaped in JSON (`\\n`, meaning a literal backslash and an `n`)
/// cannot be recovered without contradicting JSON.
///
/// An escape we do not know keeps both characters, so a Go regex's `\d` comes
/// through as written.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some(q @ ('\\' | '\'' | '"')) => out.push(q),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(text: &str) -> Vec<ToolCallInfo> {
        parse_calls(text)
    }

    #[test]
    fn a_bracketed_list_and_a_bare_call_both_parse() {
        let list = one(r#"[Read(file_path="a.txt")]"#);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Read");
        assert_eq!(list[0].arguments["file_path"], "a.txt");

        let bare = one("Read(file_path='a.txt')");
        assert_eq!(bare.len(), 1);
        assert_eq!(bare[0].name, "Read");

        let two = one("[Read(file_path='a'), Glob(pattern='*.go', limit=5)]");
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].arguments["limit"], 5);
    }

    /// The failure this parser exists for: source code in an argument. The regex
    /// scan ended at the first `)` and read the rest as more calls.
    #[test]
    fn a_code_payload_keeps_its_parens_and_invents_no_calls() {
        let calls = one(
            "[Write(file_path='hello.go', content='package main\\n\\nfunc main() {\\n\\tfmt.Println(\"hi\")\\n}')]",
        );
        assert_eq!(calls.len(), 1, "one call, not one per funcName() inside it");
        assert_eq!(calls[0].name, "Write");
        assert_eq!(
            calls[0].arguments["content"],
            "package main\n\nfunc main() {\n\tfmt.Println(\"hi\")\n}"
        );
    }

    #[test]
    fn quotes_inside_a_value_survive() {
        // The other quote character, unescaped.
        let calls = one(r#"[Write(file_path='a.go', content='say "hi"')]"#);
        assert_eq!(calls[0].arguments["content"], r#"say "hi""#);
        // The same quote character, escaped.
        let esc = one(r#"[Write(file_path='a.go', content='don\'t')]"#);
        assert_eq!(esc[0].arguments["content"], "don't");
        // A comma and an equals sign inside a value are content, not structure.
        let sep = one(r#"[Edit(file_path='a', old_string='x = 1, y = 2', new_string='z')]"#);
        assert_eq!(sep[0].arguments["old_string"], "x = 1, y = 2");
        assert_eq!(sep[0].arguments["new_string"], "z");
    }

    #[test]
    fn an_unknown_escape_keeps_both_characters() {
        let calls = one(r#"[Grep(pattern='\d+')]"#);
        assert_eq!(calls[0].arguments["pattern"], r"\d+");
        // And a correctly escaped backslash-n stays literal, as the model meant.
        let go = one(r#"[Write(file_path='a.go', content='fmt.Print("x\\n")')]"#);
        assert_eq!(go[0].arguments["content"], r#"fmt.Print("x\n")"#);
    }

    #[test]
    fn json_object_and_array_values_parse() {
        let calls = one(r#"[MultiEdit(edits=[{"file_path": "a", "old_string": "x"}])]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["edits"][0]["file_path"], "a");
    }

    #[test]
    fn a_truncated_call_yields_nothing() {
        // Cut mid-value: the quote never closes.
        assert!(one("[Write(file_path='a.go', content='package main").is_empty());
        // Cut after the arguments: no closing paren.
        assert!(one("[Write(file_path='a.go'").is_empty());
        // One good call and one truncated: all or nothing.
        assert!(one("[Read(file_path='a'), Write(content='x").is_empty());
    }

    /// A positional argument cannot be named, and a `Write` whose `content` was
    /// silently dropped truncates the file it was meant to fill.
    #[test]
    fn a_positional_argument_rejects_the_call() {
        assert!(one("[Write('a.go', content='x')]").is_empty());
        assert!(one("[Read('a.txt')]").is_empty());
    }

    #[test]
    fn prose_is_not_a_call() {
        assert!(one("Use the read() function to load it.").is_empty());
        assert!(one("I called Read(file_path='a') for you.").is_empty());
        assert!(one("[]").is_empty());
        assert!(one("[not a call]").is_empty());
        assert!(one("1(x=2)").is_empty());
    }

    #[test]
    fn an_empty_argument_list_is_a_call_with_no_arguments() {
        let calls = one("[LS()]");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "LS");
        assert!(calls[0].arguments.as_object().unwrap().is_empty());
    }
}
