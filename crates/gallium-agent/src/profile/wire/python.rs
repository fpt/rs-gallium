//! Python/Llama-style calls: `[name(arg=val, ...)]` or a bare `name(arg=val)`.
//!
//! Nothing gallium prompts for asks for this; some instruction-tuned models
//! reach for it anyway because their fine-tuning data used it.

use serde_json::Value;

use crate::llm::ToolCallInfo;

/// Parse Python/Llama-style tool calls, but only when the whole reply looks
/// like a call list — a bare `name(...)` regex over prose matches sentences
/// like "use the read() function", so the gate is what keeps this format from
/// inventing calls out of documentation.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    let t = text.trim();
    if !((t.starts_with('[') && t.ends_with(']')) || is_single_call(t)) {
        return Vec::new();
    }
    parse_call_list(t)
}

/// True if the whole string is a single `name(args)` call.
fn is_single_call(s: &str) -> bool {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^[A-Za-z_]\w*\s*\(.*\)$").unwrap());
    re.is_match(s.trim())
}

/// Parse `[name(k=v, ...), ...]` or `name(k=v)`. Values are parsed as quoted
/// strings, numbers, booleans, or JSON.
fn parse_call_list(text: &str) -> Vec<ToolCallInfo> {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"([A-Za-z_]\w*)\s*\(([^)]*)\)").unwrap());

    let mut calls = Vec::new();
    for cap in re.captures_iter(text) {
        let name = cap[1].to_string();
        let mut args = serde_json::Map::new();
        for part in split_top_commas(&cap[2]) {
            if let Some((k, v)) = part.split_once('=') {
                args.insert(k.trim().to_string(), parse_py_value(v.trim()));
            }
        }
        calls.push(ToolCallInfo {
            id: String::new(),
            name,
            arguments: Value::Object(args),
        });
    }
    calls
}

/// Split argument text on top-level commas, ignoring commas inside quotes.
fn split_top_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in s.chars() {
        match c {
            '"' | '\'' if quote.is_none() => {
                quote = Some(c);
                cur.push(c);
            }
            c if Some(c) == quote => {
                quote = None;
                cur.push(c);
            }
            ',' if quote.is_none() => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse a Python-literal-ish value into JSON.
fn parse_py_value(v: &str) -> Value {
    let v = v.trim();
    if v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')))
    {
        return Value::String(v[1..v.len() - 1].to_string());
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
