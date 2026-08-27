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

const FUNCTION_OPEN: &str = "<function=";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
const FUNCTION_CLOSE: &str = "</function>";

/// True if `text` carries a `<function=…>` block, the half of this format that
/// is not plain JSON.
pub fn is_present(text: &str) -> bool {
    text.contains(FUNCTION_OPEN)
}

/// Parse the XML-parameter form. Returns empty when absent, or when the
/// `<tool_call>` wrapper holds JSON instead (see the module note).
///
/// **Every block, not just the first.** The ported parser stopped at one and
/// said so, because whether a real Qwen emits several was unverified. It is
/// verified now, from the family's own chat template
/// (`Qwen/Qwen3.8-27B/chat_template.jinja`, byte-identical to Flash-Next's),
/// which loops over `message.tool_calls` and renders each as its own complete
/// `<tool_call>\n<function=NAME>…</function>\n</tool_call>` block:
///
/// ```jinja
/// {%- for tool_call in message.tool_calls %}
///     ...
///     {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
/// ```
///
/// So the format is plural and the model has been trained on transcripts where
/// it is. Reading one block and dropping the rest would execute part of what
/// the model asked for and report success.
///
/// Note this only pays off once generation stops running past the first block —
/// `Qwen3::stop_markers` halts at `</tool_call>` today, deliberately, and the
/// parser being ready first is what lets that be changed on its own evidence.
/// The replay path benefits immediately regardless: history rendered from an
/// assistant turn with several calls parses back to several calls.
pub fn parse_calls(text: &str) -> Vec<ToolCallInfo> {
    // The wrapper is not what identifies the format — a model that opens
    // `<tool_call>` does not always close it, and a bare `<function=` block was
    // accepted before and still is. So scan for the function blocks themselves.
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(f) = rest.find(FUNCTION_OPEN) {
        let func_content = &rest[f + FUNCTION_OPEN.len()..];
        let Some(func_end) = func_content.find('>') else {
            break;
        };
        let name = func_content[..func_end].trim();
        if name.is_empty() {
            break;
        }

        // This block's parameters end where the block does. Bounded by
        // `</function>` so a malformed block cannot swallow the next call's
        // parameters into its own arguments — which is how one wrong call
        // becomes two wrong calls.
        let body = &func_content[func_end + 1..];
        let (body, after) = match body.find(FUNCTION_CLOSE) {
            Some(end) => (&body[..end], &body[end + FUNCTION_CLOSE.len()..]),
            None => (body, ""),
        };

        let mut args = serde_json::Map::new();
        let mut search = body;
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

        calls.push(ToolCallInfo {
            id: String::new(),
            name: name.to_string(),
            arguments: Value::Object(args),
        });
        rest = after;
    }

    calls
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

    /// The plural shape the family's own template renders: one `<tool_call>`
    /// wrapper per call, each with its own `<function=>` block.
    #[test]
    fn several_blocks_are_several_calls() {
        let calls = parse_calls(
            "<tool_call>\n<function=Read>\n<parameter=file_path>\na.txt\n</parameter>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=Read>\n<parameter=file_path>\nb.txt\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
        assert_eq!(calls[1].arguments["file_path"], "b.txt");
    }

    /// A block's parameters stop at its own `</function>`. Without that bound
    /// the second call's arguments would be read into the first, which turns
    /// one malformed call into two wrong ones.
    #[test]
    fn a_blocks_parameters_do_not_leak_into_the_next() {
        let calls = parse_calls(
            "<function=Read>\n<parameter=file_path>a.txt</parameter>\n</function>\n\
             <function=Write>\n<parameter=content>hi</parameter>\n</function>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert!(
            calls[0].arguments.get("content").is_none(),
            "the second call's parameter leaked into the first: {:?}",
            calls[0].arguments
        );
        assert_eq!(calls[1].name, "Write");
        assert_eq!(calls[1].arguments["content"], "hi");
    }

    /// An unterminated block is still the call the model was making — the
    /// close tag is a bound, not a requirement, since generation can be cut
    /// short by a stop marker or a token budget.
    #[test]
    fn an_unclosed_block_still_parses() {
        let calls = parse_calls("<function=Read>\n<parameter=file_path>a.txt</parameter>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
    }

    #[test]
    fn prose_is_not_a_call() {
        assert!(parse_calls("I could use a function= somewhere in prose.").is_empty());
        assert!(parse_calls("The answer is 42.").is_empty());
    }
}
