//! The fallback profile: every wire format at once, for a model nothing is
//! known about.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::ModelProfile;

/// What an unrecognized GGUF gets: try every format gallium can read, in an
/// order chosen so the ones that name their own boundaries are consulted before
/// the ones that guess.
///
/// This permissiveness is deliberate and is also the reason profiles exist. A
/// model whose family *is* known should be reading only its own formats — see
/// [ADR 0003](../../../docs/adr/0003-model-profiles.md) — because a parser
/// running over another family's output is how tool calls get invented and
/// arguments get truncated. Here there is nothing better available: the
/// alternative to guessing is refusing every model gallium has not been taught,
/// which is the wrong trade for a backend whose job is to serve any GGUF.
///
/// This is also the one profile that overrides
/// [`ModelProfile::parse_tool_calls`] wholesale rather than naming a native
/// format, and it keeps the JSON scan **first** — the order gallium has always
/// used. The family profiles deliberately invert that (native before JSON, see
/// that method's doc), but changing it here would change behavior for exactly
/// the models nobody has run: an unrecognized GGUF should keep behaving as it
/// did, and improvements belong where there is a known model to verify them
/// against.
pub struct Generic;

impl ModelProfile for Generic {
    fn name(&self) -> &'static str {
        "generic"
    }

    /// Never detected, on either signal. This is what detection falls back
    /// *to*, and it recognizes every family's template literal — so inheriting
    /// the default `matches_template` would let it claim a model on the template
    /// pass instead of leaving it to the family that owns the format.
    fn matches_arch(&self, _arch: &str) -> bool {
        false
    }

    fn matches_template(&self, _template: &str) -> bool {
        false
    }

    fn parse_tool_calls(&self, text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        // Once, before any format is tried: a reasoning block is prose, and
        // every scan below would otherwise be willing to read a tool call the
        // model was only thinking about out loud. Once and not per-branch
        // because `strip_think_blocks` is not idempotent — two bare `</think>`
        // tags lose a second span on a second pass.
        let cleaned = wire::think::strip_think_blocks(text);
        let text = cleaned.as_str();

        // Gallium's own JSON protocol first: it is what the prompt actually
        // asked for, so a model that complied should not have its reply
        // interpreted by some other family's rules.
        let calls = wire::json::parse_calls(text);
        if !calls.is_empty() {
            return calls;
        }

        // MiniMax-M2.7. Unlike Gemma's self-describing `<|"|>`-quoted strings,
        // this wire format doesn't mark which parameters are strings, so it
        // needs the tool schema to decode.
        let calls = wire::minimax::parse_calls(text, tools);
        if !calls.is_empty() {
            return calls;
        }

        // DeepSeek-V4's DSML. Unlike MiniMax, each parameter names its own type,
        // so this doesn't need the tool schema at all.
        let calls = wire::dsml::parse_calls(text);
        if !calls.is_empty() {
            return calls;
        }

        // GPT-OSS's Harmony.
        let calls = wire::harmony_calls(text);
        if !calls.is_empty() {
            return calls;
        }

        // Python/Llama-style `[name(arg=val)]`, which self-gates on the whole
        // reply looking like a call list — a bare `name(...)` match over prose
        // would read documentation as a tool call.
        let calls = wire::python::parse_calls(text);
        if !calls.is_empty() {
            return calls;
        }

        // Gemma 4's native envelope, last: it is the most lenient of the six,
        // so anything it would claim that another format also claims should go
        // to the other format.
        wire::gemma_calls(text)
    }

    /// The reply with the model's thinking taken out of it.
    ///
    /// Three shapes, because this profile serves every GGUF rather than one
    /// family: GPT-OSS's Harmony `<|channel|>analysis<|message|>…<|end|>` /
    /// `<|channel|>final<|message|>…`, Gemma 4's `<|channel>thought … <channel|>`
    /// and `<|think|>…<|/think|>`, and the `<think>…</think>` that reasoning
    /// models like LFM2.5 emit (MiniMax-M2.7's variant of the last has no
    /// opening tag in the output at all — see
    /// [`wire::think::strip_think_blocks`]). Each is distinctive enough that
    /// running the others over a model that emits none of them is a no-op.
    ///
    /// Harmony first and exclusively: its `final` channel is the one shape here
    /// that names its own boundaries precisely (`<|channel|>final<|message|>` to
    /// the next `<|end|>`/`<|return|>`), so when present it's authoritative —
    /// running Gemma's "everything after the last `<channel|>`" heuristic over
    /// Harmony's *different* `<|channel|>` marker would silently produce the
    /// wrong slice instead of just being a no-op.
    ///
    /// Otherwise Gemma-channel before `<think>`: channel-stripping keeps only
    /// what follows the last `<channel|>`, so running it after `<think>`-removal
    /// would have to reason about markers already removed.
    fn clean_reply(&self, text: &str) -> String {
        if let Some(final_text) = crate::harmony::extract_final(text) {
            return final_text;
        }
        let s = crate::gemma::strip_thinking_blocks(text);
        wire::think::strip_think_blocks(&s).trim().to_string()
    }

    /// Stop at Gemma-4 tool boundaries: once the model closes a tool call
    /// (`<tool_call|>`) or emits a tool-response marker, stop so we can run the
    /// tool instead of letting it hallucinate a result. These literals are
    /// gemma-specific, so this is a no-op for other local models.
    fn stops_generation(&self, text: &str) -> bool {
        text.ends_with("<tool_call|>") || text.contains("<|tool_response>")
    }

    /// True if the template formats tools in *any* model-native protocol, so the
    /// backend can feed it structured tools rather than JSON prose.
    ///
    /// One sniffer for six families, which is what this profile is: a concrete
    /// profile checks only its own family's literal, and cannot be fooled by
    /// another's.
    fn template_formats_tools_natively(&self, template: &str) -> bool {
        // Gemma 4's three spellings, then GPT-OSS's Harmony channel marker.
        template.contains("<|tool_call>")
            || template.contains("<|tool>")
            || template.contains("declaration:")
            || template.contains("<|channel|>")
            // Taken from the parsers rather than re-spelled: the tag a template
            // declares is the tag the model then emits, and DeepSeek's is easy
            // to get wrong by hand — it delimits with U+FF5C (fullwidth
            // vertical bar, "｜"), not the ASCII "|" of every format above, so
            // a `<|…|>`-shaped literal would never match it.
            || template.contains(wire::minimax::WRAPPER_OPEN)
            || template.contains(wire::dsml::WRAPPER_OPEN)
    }
}

/// These were `llm_local`'s tests before profiles existed, and they are kept as
/// tests of *this* profile rather than split per format: what they pin down is
/// the cascade — that each format still parses when five others are also willing
/// to try, which is exactly the property a per-format test cannot see. The
/// concrete family profiles get their own, narrower tests.
#[cfg(test)]
mod tests {
    use super::*;

    fn calls_of(text: &str) -> Vec<ToolCallInfo> {
        Generic.tool_calls(text, &[])
    }

    fn calls_with(text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        Generic.tool_calls(text, tools)
    }

    fn cleaned(text: &str) -> String {
        Generic.clean_reply(text)
    }

    #[test]
    fn parses_bare_object() {
        let calls = calls_of(r#"{"name": "read", "arguments": {"path": "a.txt"}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn parses_object_wrapped_in_prose_and_fences() {
        let calls = calls_of(
            "Sure, I'll do that.\n```json\n{\"name\": \"glob\", \"arguments\": {\"pattern\": \"*.rs\"}}\n```",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
    }

    #[test]
    fn parses_array_of_calls_with_unique_ids() {
        let calls = calls_of(r#"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn parses_openai_shape_with_stringified_args() {
        let calls = calls_of(
            r#"{"tool_calls": [{"function": {"name": "read", "arguments": "{\"path\": \"x\"}"}}]}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "x");
    }

    #[test]
    fn parses_call_after_think_block() {
        let calls = calls_of(
            "<think>The user wants me to read a file. I should use {read}.</think>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.txt\"}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn parses_gemma_native_tool_call() {
        // Gemma's native envelope, calling an MCP tool (godevmcp's search-godoc).
        // The hyphens exercise the name charset (`[A-Za-z0-9_.-]`) on both sides.
        let calls = calls_of("<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc");
        assert_eq!(calls[0].arguments["query"], "mcp-go");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn parses_gemma_call_with_mixed_args() {
        let calls =
            calls_of("<|tool_call>call:grep{pattern:<|\"|>foo<|\"|>, limit:50}<tool_call|>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments["pattern"], "foo");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    #[test]
    fn plain_prose_is_not_a_gemma_call() {
        assert!(calls_of("Sure, I'll call the search tool for you.").is_empty());
    }

    #[test]
    fn gemma_call_with_braced_source_still_parses() {
        // Regression for gemma-4-26B leaking raw `<|tool_call>` markup: when a
        // string arg holds content with `{`/`}`, the whole native call must still
        // parse through the full chain (JSON → python → gemma fallback),
        // otherwise the turn is misread as a final text answer. Payload mirrors
        // the real leaked reply (channel wrapper + braced arg value).
        let raw = "<|channel>thought<channel|><|tool_call>call:write\
            {file_path:<|\"|>a.json<|\"|>,content:<|\"|>{ \"loop\": true, \"body\": { \"n\": 3 } }\
            <|\"|>}<tool_call|>";
        let calls = calls_of(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
        assert_eq!(calls[0].arguments["file_path"], "a.json");
        assert!(
            calls[0].arguments["content"]
                .as_str()
                .unwrap()
                .contains("\"body\": { \"n\": 3 }"),
            "braced content must survive intact"
        );
    }

    #[test]
    fn parses_python_style_bracket_call() {
        let calls = calls_of(r#"[read(file_path="codeword.txt")]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "codeword.txt");
    }

    #[test]
    fn parses_multiple_python_calls() {
        let calls = calls_of(r#"[glob(pattern="*.rs"), grep(pattern="fn main", path="src")]"#);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments["path"], "src");
    }

    #[test]
    fn prose_mentioning_a_function_is_not_a_call() {
        assert!(calls_of("You can use the read() function to open files.").is_empty());
    }

    #[test]
    fn plain_text_yields_no_calls() {
        assert!(calls_of("The capital of France is Paris.").is_empty());
    }

    #[test]
    fn parses_minimax_native_tool_call() {
        // No schema needed when every argument is a plain string.
        let calls = calls_of(
            "I should read the file.\n</think>\n\n<minimax:tool_call>\n\
             <invoke name=\"read\">\n<parameter name=\"file_path\">a.txt</parameter>\n</invoke>\n\
             </minimax:tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn minimax_call_decodes_argument_types_from_the_tool_schema() {
        // The wire format renders string args raw and everything else
        // tojson-encoded, so "50" (a string that looks numeric) and 50 (an
        // actual integer) are byte-identical on the wire — only the schema
        // tells them apart.
        let tools = [ToolDefinition {
            name: "grep".to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "limit": {"type": "integer"},
                    "case_sensitive": {"type": "boolean"},
                }
            }),
        }];
        let calls = calls_with(
            "<minimax:tool_call>\n<invoke name=\"grep\">\n\
             <parameter name=\"pattern\">50</parameter>\n\
             <parameter name=\"limit\">50</parameter>\n\
             <parameter name=\"case_sensitive\">true</parameter>\n\
             </invoke>\n</minimax:tool_call>",
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "50"); // string, kept raw
        assert_eq!(calls[0].arguments["limit"], 50); // integer, decoded
        assert_eq!(calls[0].arguments["case_sensitive"], true); // boolean, decoded
    }

    #[test]
    fn minimax_call_with_unknown_tool_defaults_every_argument_to_string() {
        // No schema to consult (MCP tool, or the model hallucinated a name) —
        // the lossless guess is to keep the raw text rather than gamble on JSON.
        let calls = calls_of(
            "<minimax:tool_call>\n<invoke name=\"mystery\">\n\
             <parameter name=\"n\">50</parameter>\n</invoke>\n</minimax:tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["n"], "50");
    }

    #[test]
    fn minimax_call_with_multiline_string_argument_survives_intact() {
        // Go source as a raw (unescaped) string arg — the shape a real
        // MultiEdit call takes; braces and newlines must not confuse the
        // <parameter> boundary.
        let tools = [ToolDefinition {
            name: "write".to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string"}}
            }),
        }];
        let calls = calls_with(
            "<minimax:tool_call>\n<invoke name=\"write\">\n\
             <parameter name=\"content\">func main() {\n\tfmt.Println(\"hi\")\n}</parameter>\n\
             </invoke>\n</minimax:tool_call>",
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments["content"],
            "func main() {\n\tfmt.Println(\"hi\")\n}"
        );
    }

    #[test]
    fn minimax_string_argument_containing_a_literal_closing_tag_is_not_truncated() {
        // Regression for a PR #106 review comment: the wire format has no
        // escaping, so a write-tool payload can legally contain the literal
        // text `</parameter>` (e.g. documentation of this very wire format,
        // or a stray HTML-ish closing tag in the file being written). A
        // naive "first </parameter>" scan truncates the value there and
        // drops everything after it — this is the *stray closing tag*
        // case, not a value that also fakes a matching opening tag (an
        // unescaped format can't distinguish that from a real one, and
        // nothing here tries to).
        let tools = [ToolDefinition {
            name: "write".to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string"}}
            }),
        }];
        let calls = calls_with(
            "<minimax:tool_call>\n<invoke name=\"write\">\n\
             <parameter name=\"content\">Close a call with </parameter> then keep writing.</parameter>\n\
             </invoke>\n</minimax:tool_call>",
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments["content"],
            "Close a call with </parameter> then keep writing."
        );
    }

    #[test]
    fn minimax_multiple_invokes_in_one_wrapper_split_correctly() {
        // The template renders every tool_calls entry inside a single
        // <minimax:tool_call>...</minimax:tool_call> wrapper, not one wrapper
        // per call — and a string argument containing `</invoke>` must not
        // be mistaken for the boundary between the two real calls, nor leak
        // into the second call's name/arguments.
        let calls = calls_of(
            "<minimax:tool_call>\n\
             <invoke name=\"read\">\n<parameter name=\"file_path\">notes on </invoke> tags.txt</parameter>\n</invoke>\n\
             <invoke name=\"glob\">\n<parameter name=\"pattern\">*.rs</parameter>\n</invoke>\n\
             </minimax:tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            calls[0].arguments["file_path"],
            "notes on </invoke> tags.txt"
        );
        assert_eq!(calls[1].name, "glob");
        assert_eq!(calls[1].arguments["pattern"], "*.rs");
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn minimax_last_value_does_not_leak_the_wrapper_closing_tag() {
        // The template puts a literal newline between the last </invoke> and
        // </minimax:tool_call> (`~ '\n'` in the render, not template-source
        // whitespace jinja trims away) — the last argument's value must not
        // pick up that newline or any part of the wrapper's own close tag.
        let tools = [ToolDefinition {
            name: "write".to_string(),
            description: String::new(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"content": {"type": "string"}}
            }),
        }];
        let calls = calls_with(
            "<minimax:tool_call>\n<invoke name=\"write\">\n\
             <parameter name=\"content\">done</parameter>\n</invoke>\n</minimax:tool_call>",
            &tools,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "done");
    }

    #[test]
    fn parses_dsml_native_tool_call() {
        let calls = calls_of(
            "I should read the file.\n</think>\n\n<｜DSML｜tool_calls>\n\
             <｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"file_path\" string=\"true\">a.txt</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "a.txt");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn dsml_call_decodes_argument_types_from_the_string_attribute() {
        // Unlike MiniMax, DSML names each parameter's type on the wire
        // itself, so no tool schema is needed to tell "50" (string) from 50
        // (integer) — both render identically except for `string="..."`.
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"grep\">\n\
             <｜DSML｜parameter name=\"pattern\" string=\"true\">50</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"limit\" string=\"false\">50</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"case_sensitive\" string=\"false\">true</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["pattern"], "50"); // string, kept raw
        assert_eq!(calls[0].arguments["limit"], 50); // integer, decoded
        assert_eq!(calls[0].arguments["case_sensitive"], true); // boolean, decoded
    }

    #[test]
    fn dsml_call_with_missing_or_malformed_string_attribute_defaults_to_string() {
        // Same lossless-default reasoning as MiniMax's unknown-tool case: if
        // `string=` is absent or not exactly "false", keep the raw text
        // rather than gamble on a JSON parse.
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"mystery\">\n\
             <｜DSML｜parameter name=\"n\" string=\"maybe\">50</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["n"], "50");
    }

    #[test]
    fn dsml_call_with_no_string_attribute_at_all_defaults_to_string_and_keeps_scanning() {
        // Regression: the first cut of dsml's parameter_boundaries searched
        // for `string="` unbounded past the current tag and aborted the
        // whole scan (`break`) when none was found at all — silently
        // dropping this parameter *and* every one after it, rather than
        // defaulting just this one to string and continuing. A parameter
        // with no `string=` attribute must not swallow the one that follows
        // it in the same invoke.
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"mystery\">\n\
             <｜DSML｜parameter name=\"n\">50</｜DSML｜parameter>\n\
             <｜DSML｜parameter name=\"m\" string=\"false\">7</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["n"], "50"); // no string= — defaulted to string
        assert_eq!(calls[0].arguments["m"], 7); // string="false" still decodes
    }

    #[test]
    fn dsml_string_argument_containing_a_literal_closing_tag_is_not_truncated() {
        // Same no-escaping hazard as MiniMax's wire format: a write-tool
        // payload can legally contain the literal text
        // `</｜DSML｜parameter>`, and a naive first-match scan would truncate
        // the value there.
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write\">\n\
             <｜DSML｜parameter name=\"content\" string=\"true\">Close a call with </｜DSML｜parameter> then keep writing.</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments["content"],
            "Close a call with </｜DSML｜parameter> then keep writing."
        );
    }

    #[test]
    fn dsml_multiple_invokes_in_one_wrapper_split_correctly() {
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n\
             <｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"file_path\" string=\"true\">notes on </｜DSML｜invoke> tags.txt</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"glob\">\n\
             <｜DSML｜parameter name=\"pattern\" string=\"true\">*.rs</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            calls[0].arguments["file_path"],
            "notes on </｜DSML｜invoke> tags.txt"
        );
        assert_eq!(calls[1].name, "glob");
        assert_eq!(calls[1].arguments["pattern"], "*.rs");
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn dsml_last_value_does_not_leak_the_wrapper_closing_tag() {
        let calls = calls_of(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write\">\n\
             <｜DSML｜parameter name=\"content\" string=\"true\">done</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "done");
    }

    /// Reasoning is stripped before *any* format is tried, not just before the
    /// brace scan that needed it first. A model weighing a call it decided
    /// against has not made one, and every native format is as findable inside a
    /// `<think>` block as JSON is — so this is checked on a native format, where
    /// a regression would mean gallium running a tool the model talked itself
    /// out of.
    #[test]
    fn a_tool_call_the_model_only_reasoned_about_is_not_executed() {
        let native = calls_of(
            "<think>I could write <minimax:tool_call>\n<invoke name=\"Bash\">\n\
             <parameter name=\"command\">rm -rf /</parameter>\n</invoke>\n\
             </minimax:tool_call> but that would be destructive.</think>\n\
             I won't run that.",
        );
        assert!(native.is_empty(), "{native:?}");

        let gemma = calls_of(
            "<think>The syntax is <|tool_call>call:write{file_path:<|\"|>x<|\"|>}<tool_call|>, \
             but I have nothing to write.</think>\nNothing to do here.",
        );
        assert!(gemma.is_empty(), "{gemma:?}");
    }

    #[test]
    fn template_supports_native_tools_recognizes_dsml() {
        assert!(Generic.template_formats_tools_natively(
            "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block"
        ));
    }

    #[test]
    fn template_supports_native_tools_recognizes_harmony() {
        assert!(Generic.template_formats_tools_natively(
            "# Valid channels: analysis, commentary, final.\n\
             {{- \"<|start|>assistant<|channel|>final<|message|>\" }}"
        ));
    }

    #[test]
    fn a_plain_template_needs_the_json_prose_fallback() {
        assert!(!Generic.template_formats_tools_natively(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        ));
    }

    #[test]
    fn parses_gpt_oss_harmony_tool_call() {
        // The exact text a real gpt-oss-120b run leaked as a "final answer"
        // before Harmony detection existed: llm_local.rs never recognized
        // the GGUF's template as native, so the model (fine-tuned on
        // Harmony) ignored gallium's generic JSON-prose instructions and
        // emitted Harmony syntax anyway, which nothing understood.
        let calls = calls_of(
            "<|channel|>analysis<|message|>We need to read Cargo.toml.<|end|>\
             <|start|>assistant<|channel|>commentary to=Read <|constrain|>json<|message|>\
             {\"file_path\":\"Cargo.toml\",\"limit\":200}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["file_path"], "Cargo.toml");
        assert_eq!(calls[0].arguments["limit"], 200);
    }

    #[test]
    fn parses_gpt_oss_harmony_tool_call_with_functions_namespace() {
        // What the model emits once render_native properly declares the
        // "functions" namespace (see harmony::parse_tool_calls's doc
        // comment) — the shape after this fix, not just the leaked-text
        // shape from before it.
        let calls = calls_of(
            "<|start|>assistant to=functions.Glob<|channel|>commentary <|constrain|>json<|message|>\
             {\"pattern\":\"crates/*\"}<|call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Glob");
        assert_eq!(calls[0].arguments["pattern"], "crates/*");
    }

    #[test]
    fn harmony_final_channel_is_cleaned_from_the_reply() {
        let raw = "<|channel|>analysis<|message|>Thinking it through.<|end|>\
                   <|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>";
        assert_eq!(cleaned(raw), "The answer is 42.");
    }

    /// The reply from a real gemma4-12b session, which reached the user with the
    /// channel wrapper still on it. This is that bug.
    #[test]
    fn a_gemma_channel_wrapper_does_not_reach_the_reply() {
        let raw = "<|channel>thought\n<channel|>This project, **rs-gallium**, is a \
                   research-oriented LLM inference framework written in Rust.";

        let reply = cleaned(raw);

        assert!(reply.starts_with("This project"), "{reply:?}");
        assert!(!reply.contains("channel"), "{reply:?}");
    }

    /// Thinking with content in it, not just an empty channel — everything up to
    /// the close belongs to the model, not the reader.
    #[test]
    fn thinking_inside_the_channel_is_dropped_with_it() {
        let raw = "<|channel>thought\nThe user asked about the repo. I should \
                   check git log first.<channel|>Here is what I found.";

        assert_eq!(cleaned(raw), "Here is what I found.");
    }

    /// The other shape the same model uses.
    #[test]
    fn a_paired_think_wrapper_is_dropped_too() {
        assert_eq!(
            cleaned("<|think|>reasoning here<|/think|>The answer."),
            "The answer."
        );
    }

    /// What a reasoning model like LFM2.5 emits. Same leak, same site — it just
    /// had not been reported yet.
    #[test]
    fn a_think_block_is_dropped_from_the_reply() {
        assert_eq!(
            cleaned("<think>Let me work through this.</think>\nThe answer."),
            "The answer."
        );
    }

    /// MiniMax-M2.7's template pre-fills `<think>\n` into the *prompt*, so the
    /// generated text carries only the closing tag — everything before it is
    /// still the model's reasoning, not the reply.
    #[test]
    fn a_bare_closing_think_tag_with_no_opener_still_strips_the_reasoning() {
        let raw = "The user wants a summary. Let me write one.\n</think>\n\n## Summary\nDone.";
        assert_eq!(cleaned(raw), "## Summary\nDone.");
    }

    /// Turkish İ (U+0130) grows by a byte under `to_lowercase()` ("i̇", i +
    /// combining dot above). A byte offset found in that lowercased copy,
    /// applied to the original string, lands one byte into the following
    /// multi-byte character instead of before it — a `replace_range` panic
    /// ("not a character boundary"), reproduced here with `é` sitting right
    /// after the tag so the drift lands mid-character.
    #[test]
    fn reasoning_with_a_length_changing_lowercase_character_does_not_panic() {
        assert_eq!(cleaned("İ</think>éxyz"), "éxyz");
    }

    /// A model that emits neither must come through untouched, since this runs
    /// over every GGUF the backend serves.
    #[test]
    fn an_ordinary_reply_is_left_alone() {
        let plain = "The capital of France is Paris.";
        assert_eq!(cleaned(plain), plain);
    }

    /// Prose that merely mentions the words is not a wrapper.
    #[test]
    fn prose_about_thinking_is_not_mistaken_for_it() {
        let text = "I was thinking about how the channel abstraction works.";
        assert_eq!(cleaned(text), text);
    }

    /// The Gemma boundary the sampler stops at, and the two different tests it
    /// takes: the tool-call close must be *at the end* (the model is done
    /// writing the call), while a tool-response marker anywhere means the model
    /// has started hallucinating the result.
    #[test]
    fn generation_stops_at_a_gemma_tool_boundary() {
        assert!(Generic.stops_generation("<|tool_call>call:read{}<tool_call|>"));
        assert!(Generic.stops_generation("<|tool_response>response:read{...}"));
        assert!(!Generic.stops_generation("<|tool_call>call:read{path:<|\"|>a"));
        assert!(!Generic.stops_generation("The answer is 42."));
    }
}
