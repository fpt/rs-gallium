//! Qwen's native XML-parameter tool-call format.
//!
//! ```text
//! <tool_call>
//! <function=write>
//! <parameter=file_path>
//! hello.go
//! </parameter>
//! <parameter=content>
//! package main…
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Ported from `protocol.rs`'s `QwenProtocol::parse_tool_call`, which was the
//! only place that knew this format and served the candle backend alone. Every
//! value is a **string**: the format carries no type information, and unlike
//! MiniMax's otherwise-similar shape there is no schema lookup here either —
//! the original did not consult one, and inventing type coercion during a port
//! would be a behavior change dressed as a move.
//!
//! `<tool_call>` may also wrap a plain JSON object rather than a `<function=>`
//! block; that shape is left to [`super::json`], which already reads it out of
//! the middle by balanced-span scan. This module returns nothing for it rather
//! than duplicating a second JSON path.

use serde_json::Value;

use crate::llm::ToolCallInfo;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// True if `text` carries a `<function=…>` block, the half of this format that
/// is not plain JSON.
pub fn is_present(text: &str) -> bool {
    text.contains(FUNCTION_OPEN)
}

/// Parse the XML-parameter form. Returns empty when absent, or when the
/// `<tool_call>` wrapper holds JSON instead (see the module note).
///
/// One call, not many: the format nests parameters inside a single
/// `<function=>`, and the original parser stopped at the first. Whether a real
/// Qwen emits several `<function=>` blocks in one wrapper is unverified — no
/// Qwen model is cached on the machine this was written on — so this reads
/// exactly as much as the code it replaces did, and a second block would be
/// ignored the same way it always was.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    // The function block may sit inside a <tool_call> wrapper or stand alone —
    // both shapes were accepted before, since a model that opens the wrapper
    // does not always close it.
    let after_wrapper = match text.find(TOOL_CALL_OPEN) {
        Some(i) => &text[i + TOOL_CALL_OPEN.len()..],
        None => text,
    };
    let Some(f) = after_wrapper.find(FUNCTION_OPEN) else {
        return Vec::new();
    };
    let func_content = &after_wrapper[f + FUNCTION_OPEN.len()..];

    let Some(func_end) = func_content.find('>') else {
        return Vec::new();
    };
    let name = func_content[..func_end].trim();
    if name.is_empty() {
        return Vec::new();
    }

    let mut args = serde_json::Map::new();
    let mut search = &func_content[func_end + 1..];
    while let Some(p_start) = search.find(PARAM_OPEN) {
        let p_rest = &search[p_start + PARAM_OPEN.len()..];
        let Some(p_name_end) = p_rest.find('>') else {
            break;
        };
        let p_name = p_rest[..p_name_end].to_string();
        let val_start = &p_rest[p_name_end + 1..];
        let Some(val_end) = val_start.find(PARAM_CLOSE) else {
            break;
        };
        let val = val_start[..val_end].trim().to_string();
        args.insert(p_name, Value::String(val));
        search = &val_start[val_end + PARAM_CLOSE.len()..];
    }

    vec![ToolCallInfo {
        id: String::new(),
        name: name.to_string(),
        arguments: Value::Object(args),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Transcribed from `protocol.rs`'s `QwenProtocol` doc comment — the shape
    /// that parser was written against.
    #[test]
    fn parses_the_xml_parameter_form() {
        let calls = parse_calls(
            "<tool_call>\n<function=write>\n<parameter=file_path>\nhello.go\n</parameter>\n\
             <parameter=content>\npackage main\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
        assert_eq!(calls[0].arguments["file_path"], "hello.go");
        assert_eq!(calls[0].arguments["content"], "package main");
    }

    /// A `<function=>` block with no wrapper was accepted before and still is:
    /// a model that opens `<tool_call>` does not reliably close it, and the
    /// wrapper is not what identifies the format.
    #[test]
    fn a_bare_function_block_parses_without_the_wrapper() {
        let calls =
            parse_calls("<function=Read>\n<parameter=file_path>a.txt</parameter>\n</function>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
    }

    /// Values are strings, verbatim, including ones that look numeric — the
    /// format says nothing about type and neither did the parser this replaces.
    #[test]
    fn every_value_is_a_string() {
        let calls = parse_calls("<function=grep>\n<parameter=limit>50</parameter>\n</function>");
        assert_eq!(calls[0].arguments["limit"], "50");
    }

    /// Multi-line code survives: the value runs to its own `</parameter>`, and
    /// parens are not a boundary here the way they are in `super::python`.
    #[test]
    fn multiline_code_survives_intact() {
        let calls = parse_calls(
            "<function=Write>\n<parameter=content>\nfunc main() {\n\tfmt.Println(\"hi\")\n}\n</parameter>\n</function>",
        );
        assert_eq!(
            calls[0].arguments["content"],
            "func main() {\n\tfmt.Println(\"hi\")\n}"
        );
    }

    /// The JSON-in-wrapper shape is `super::json`'s, not this module's.
    #[test]
    fn json_inside_the_wrapper_is_left_to_the_json_parser() {
        let calls =
            parse_calls("<tool_call>\n{\"name\": \"Read\", \"arguments\": {}}\n</tool_call>");
        assert!(calls.is_empty());
    }

    #[test]
    fn prose_is_not_a_call() {
        assert!(parse_calls("I could use a function= somewhere in prose.").is_empty());
        assert!(parse_calls("The answer is 42.").is_empty());
    }
}
