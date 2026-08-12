//! Shared Harmony (GPT-OSS) native tool-call and reply parsing.
//!
//! Both local backends can speak GPT-OSS's Harmony wire format — `llm_local`
//! (llama.cpp, via the GGUF's embedded chat template, once
//! `render_native`/`parse_tool_calls` recognize it) and
//! `protocol::HarmonyProtocol` (candle, hand-written prompt). This module
//! owns the format knowledge so they parse it identically.
//!
//! A turn is a sequence of `<|start|>ROLE ...<|channel|>CHANNEL<|message|>
//! CONTENT<|end|>` segments; a tool call is addressed `to=functions.NAME`
//! and closed with `<|call|>` instead of `<|end|>`. Three channels matter:
//! `analysis` (chain-of-thought, dropped before the user sees it),
//! `commentary` (tool calls), and `final` (the user-facing reply).

use serde_json::Value;

/// One parsed Harmony tool call, name exactly as the model emitted it (minus
/// an optional `functions.` namespace prefix — see [`parse_tool_calls`]).
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonyCall {
    pub name: String,
    pub arguments: Value,
}

/// Parse every tool call in one Harmony completion, in order.
///
/// A call looks like `to=functions.NAME<|channel|>commentary ...<|message|>
/// {...}<|call|>` — that's what the model emits once it's been told about
/// the "functions" namespace (see `llm_local::render_native`). The
/// `functions.` prefix is optional here too, so a model that was never told
/// about one (e.g. gallium's old JSON-prose tool-call fallback, before a
/// template's own native protocol was recognized) still gets picked up
/// rather than leaking as raw text.
///
/// Each call's search window is bounded by the position of the *next* `to=`
/// marker (or end of text) — never open-ended — so a `<|call|>` token
/// embedded in an earlier argument's own value (legal: Harmony has no
/// escaping, same as every other wire-format-without-escaping this codebase
/// parses) can't be mistaken for *this* call's terminator. The real
/// terminator is the last `<|call|>` inside that bounded window.
pub fn parse_tool_calls(text: &str) -> Vec<HarmonyCall> {
    let mut markers = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find("to=") {
        markers.push(from + rel);
        from += rel + "to=".len();
    }

    let mut calls = Vec::new();
    for (i, &start) in markers.iter().enumerate() {
        let window_end = markers.get(i + 1).copied().unwrap_or(text.len());
        let window = &text[start..window_end];
        let after_to = &window["to=".len()..];

        let Some(name_end) = after_to.find(|c: char| c.is_whitespace() || c == '<') else {
            continue;
        };
        let raw_name = &after_to[..name_end];
        let name = raw_name.strip_prefix("functions.").unwrap_or(raw_name);
        if name.is_empty() {
            continue;
        }

        let Some(msg_rel) = window.find("<|message|>") else {
            continue;
        };
        let value_window = &window[msg_rel + "<|message|>".len()..];
        let value_end = value_window.rfind("<|call|>").unwrap_or(value_window.len());
        let json = value_window[..value_end].trim();

        if let Ok(arguments) = serde_json::from_str::<Value>(json) {
            calls.push(HarmonyCall {
                name: name.to_string(),
                arguments,
            });
        }
    }
    calls
}

/// Extract the `final`-channel content from a Harmony completion — the
/// user-facing reply, with the `analysis` (chain-of-thought) and
/// `commentary` (tool-call scaffolding) channels stripped out. `None` if no
/// `final` channel segment is present (the completion is entirely a tool
/// call, or generation was cut off before reaching one — the caller decides
/// what to show in that case, this function doesn't guess).
pub fn extract_final(text: &str) -> Option<String> {
    const MARKER: &str = "<|channel|>final<|message|>";
    let start = text.find(MARKER)?;
    let rest = &text[start + MARKER.len()..];
    let end = ["<|end|>", "<|return|>"]
        .iter()
        .filter_map(|t| rest.find(t))
        .min()
        .unwrap_or(rest.len());
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_call_with_specials_visible() {
        // The exact shape llama.cpp emits with skip_special=false — the
        // scenario that leaked raw Harmony markup as a "final answer"
        // before this module existed.
        let calls = parse_tool_calls(
            "<|start|>assistant to=functions.Read<|channel|>commentary <|constrain|>json<|message|>{\"file_path\":\"Cargo.toml\",\"limit\":200}<|call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "Cargo.toml");
        assert_eq!(calls[0].arguments["limit"], 200);
    }

    #[test]
    fn parses_call_without_a_functions_namespace() {
        // What the model actually emitted when gallium's old JSON-prose
        // fallback never told it about a "functions" namespace: it still
        // used Harmony's own `to=NAME` syntax, just without the prefix.
        let calls = parse_tool_calls(
            "<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>commentary to=Read <|constrain|>json<|message|>{\"file_path\":\"Cargo.toml\",\"limit\":200}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "Cargo.toml");
    }

    #[test]
    fn parses_multiple_calls_and_a_stray_call_token_does_not_truncate() {
        let calls = parse_tool_calls(
            "<|start|>assistant to=functions.Write<|channel|>commentary json<|message|>\
             {\"content\":\"see <|call|> in the docs\",\"file_path\":\"a.txt\"}<|call|>\
             <|start|>assistant to=functions.Glob<|channel|>commentary json<|message|>\
             {\"pattern\":\"*.rs\"}<|call|>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["content"], "see <|call|> in the docs");
        assert_eq!(calls[1].name, "Glob");
        assert_eq!(calls[1].arguments["pattern"], "*.rs");
    }

    #[test]
    fn no_to_marker_yields_no_calls() {
        assert!(parse_tool_calls("Sure, I can help with that.").is_empty());
    }

    #[test]
    fn extracts_final_channel_dropping_analysis() {
        let raw = "<|channel|>analysis<|message|>Let me think about this.<|end|>\
                   <|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>";
        assert_eq!(extract_final(raw), Some("The answer is 42.".to_string()));
    }

    #[test]
    fn extracts_final_channel_terminated_by_return() {
        let raw = "<|channel|>final<|message|>Done.<|return|>";
        assert_eq!(extract_final(raw), Some("Done.".to_string()));
    }

    #[test]
    fn no_final_channel_yields_none() {
        assert_eq!(
            extract_final("<|channel|>analysis<|message|>still thinking"),
            None
        );
    }
}
