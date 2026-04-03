//! Model protocol adapters: convert ChatMessage history ↔ raw model prompts.
//!
//! Each protocol knows:
//!   - `format_prompt`: render message list into a raw string the model expects
//!   - `parse_response`: extract user-facing reply from raw decoded output (skip_special=false)
//!   - `format_prompt_with_tools`: like `format_prompt` but embeds tool definitions
//!   - `supports_tools`: whether this protocol can generate/parse tool calls
//!   - `tool_stop_tokens`: extra EOS token strings for tool call termination
//!   - `parse_tool_call`: detect/parse a tool call from raw decoded output (skip_special=false)
//!
//! # Protocols
//!
//! | Protocol        | Model    | Tools | Thinking |
//! |-----------------|----------|-------|----------|
//! | HarmonyProtocol | GPT-OSS  | yes   | no       |
//! | GemmaProtocol  | Gemma 4  | yes   | optional |
//! | QwenProtocol    | Qwen 3.5 | no    | no       |
//!
//! ## Harmony channel and tool call format
//!
//! GPT-OSS uses the [Harmony protocol](https://github.com/openai/harmony).
//!
//! ### Chat output channels
//!
//! The model writes to named channels per turn. After `decode(skip_special=false)`,
//! special delimiters appear literally but channel name text tokens are embedded:
//!
//! ```text
//! <|channel|>analysis<|message|>REASONING<|end|>
//! <|start|>assistant<|channel|>final<|message|>ANSWER<|end|>
//! ```
//!
//! `parse_response` finds the last word-boundary "final" and returns everything after it.
//!
//! ### Tool call format
//!
//! Tool definitions are embedded in the system prompt as a TypeScript namespace:
//!
//! ```text
//! namespace functions {
//!   // description
//!   type func_name = (_: { param: string }) => any;
//! }
//! ```
//!
//! The model emits tool calls using:
//!
//! ```text
//! <|start|>assistant to=functions.FUNC<|channel|>commentary<|constrain|>json<|message|>{"arg":"val"}<|call|>
//! ```
//!
//! After decode, the plain text ` to=functions.FUNC` is detectable regardless of
//! skip_special setting (role + recipient are text tokens). `parse_harmony_tool_call`
//! detects `functions.FUNC_NAME` and extracts the JSON args from the last `{…}` block.
//!
//! Tool results are formatted as:
//!
//! ```text
//! <|start|>tool functions.FUNC_NAME<|message|>RESULT<|end|>
//! ```
//!
//! ## Gemma 4 / GemmaProtocol
//!
//! Uses Gemma 4's native function-calling token format.
//!
//! ### Tool declaration (prepended to prompt)
//!
//! ```text
//! <|tool>declaration:FUNC_NAME{description:<|"|>DESC<|"|>,parameters:{properties:{PARAM:{description:<|"|>DESC<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>PARAM<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>
//! ```
//!
//! ### Tool call (model output; stops at `<tool_call|>` EOS)
//!
//! ```text
//! <|tool_call>call:FUNC_NAME{param:<|"|>value<|"|>}<tool_call|>
//! ```
//!
//! ### Tool result (injected back into prompt)
//!
//! ```text
//! <|tool_response>response:FUNC_NAME{output:<|"|>RESULT<|"|>}<tool_response|>
//! ```
//!
//! ### Thinking (optional, enabled via `GemmaProtocol::with_thinking()`)
//!
//! When thinking is enabled, `<|think|>` is added to the system turn to activate
//! the model's internal reasoning. The model wraps reasoning in:
//!
//! ```text
//! <|channel>thought
//! REASONING
//! <channel|>FINAL ANSWER
//! ```
//!
//! `parse_response` looks for `<channel|>` and returns everything after it.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{ChatMessage, ChatRole, ToolDefinition};

// ============================================================================
// Trait
// ============================================================================

pub trait ModelProtocol {
    /// Render a message history into a raw prompt string for the model.
    fn format_prompt(&self, messages: &[ChatMessage]) -> String;

    /// Render a message history with tool definitions into a raw prompt string.
    /// Default: delegates to `format_prompt` (ignores tools).
    fn format_prompt_with_tools(&self, messages: &[ChatMessage], _tools: &[ToolDefinition]) -> String {
        self.format_prompt(messages)
    }

    /// Extract the user-facing reply from raw model output decoded with skip_special=false.
    fn parse_response(&self, raw: &str) -> String;

    /// Whether this protocol supports generating and parsing tool calls.
    fn supports_tools(&self) -> bool {
        false
    }

    /// Extra token strings that should act as EOS during tool call generation.
    /// GalliumProvider will look these up in the tokenizer vocabulary on construction.
    fn tool_stop_tokens(&self) -> &[&'static str] {
        &[]
    }

    /// Detect and parse a tool call from raw model output decoded with skip_special=false.
    /// Returns `(function_name, args_json)` if a tool call is detected.
    fn parse_tool_call(&self, _raw: &str) -> Option<(String, serde_json::Value)> {
        None
    }
}

// ============================================================================
// HarmonyProtocol — GPT-OSS
// ============================================================================

/// Harmony protocol adapter for GPT-OSS.
pub struct HarmonyProtocol;

impl HarmonyProtocol {
    /// Build the canonical Harmony system content, merging in the optional
    /// caller-provided system message and tool namespace.
    fn build_system_content(date: &str, extra: Option<&str>, tool_ns: Option<&str>) -> String {
        let mut s = format!(
            "You are ChatGPT, a large language model trained by OpenAI.\n\
             Knowledge cutoff: 2024-06\n\
             Current date: {date}\n\
             \n\
             Reasoning: medium\n\
             \n\
             # Valid channels: analysis, commentary, final. Channel must be included for every message."
        );
        if let Some(e) = extra {
            s.push_str("\n\n");
            s.push_str(e);
        }
        if let Some(ns) = tool_ns {
            s.push_str("\n\n");
            s.push_str(ns);
        }
        s
    }

    /// Render the non-system, non-tool-call portion of a message list.
    fn append_messages(s: &mut String, messages: &[ChatMessage]) {
        for msg in messages {
            match msg.role {
                ChatRole::System => {} // handled separately
                ChatRole::User => {
                    s.push_str(&format!("<|start|>user<|message|>{}<|end|>", msg.content));
                }
                ChatRole::Tool => {
                    // Tool result: <|start|>tool functions.NAME<|message|>CONTENT<|end|>
                    let func = msg.tool_name.as_deref().unwrap_or("unknown");
                    s.push_str(&format!(
                        "<|start|>tool functions.{}<|message|>{}<|end|>",
                        func, msg.content
                    ));
                }
                ChatRole::Assistant => {
                    if let Some(ref calls) = msg.tool_calls {
                        // One Harmony call block per tool invocation.
                        for call in calls {
                            let args = serde_json::to_string(&call.arguments)
                                .unwrap_or_else(|_| "{}".to_string());
                            s.push_str(&format!(
                                "<|start|>assistant to=functions.{}<|channel|>commentary<|constrain|>json<|message|>{}<|call|>",
                                call.name, args
                            ));
                        }
                    } else if !msg.content.is_empty() {
                        s.push_str(&format!("<|start|>assistant\n{}<|end|>", msg.content));
                    }
                }
            }
        }
    }
}

impl ModelProtocol for HarmonyProtocol {
    fn supports_tools(&self) -> bool {
        true
    }

    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let date = current_date_ymd();
        let extra = messages.iter().find_map(|m| {
            if m.role == ChatRole::System { Some(m.content.as_str()) } else { None }
        });
        let system = Self::build_system_content(&date, extra, None);
        let mut s = format!("<|start|>system<|message|>{system}<|end|>");
        Self::append_messages(&mut s, messages);
        s.push_str("<|start|>assistant\n");
        s
    }

    fn format_prompt_with_tools(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
        let date = current_date_ymd();
        let extra = messages.iter().find_map(|m| {
            if m.role == ChatRole::System { Some(m.content.as_str()) } else { None }
        });
        let ns = if tools.is_empty() {
            None
        } else {
            Some(tools_to_harmony_namespace(tools))
        };
        let system = Self::build_system_content(&date, extra, ns.as_deref());
        let mut s = format!("<|start|>system<|message|>{system}<|end|>");
        Self::append_messages(&mut s, messages);
        s.push_str("<|start|>assistant\n");
        s
    }

    fn parse_response(&self, raw: &str) -> String {
        extract_harmony_final(raw)
    }

    fn parse_tool_call(&self, raw: &str) -> Option<(String, serde_json::Value)> {
        parse_harmony_tool_call(raw)
    }
}

// ============================================================================
// Harmony tool call parsing
// ============================================================================

/// Detect and parse a Harmony tool call from decoded model output.
///
/// After `decode(skip_special=true)`, a tool call looks like:
/// ```text
/// "assistant to=functions.FUNC_NAMEcommentaryjson{"arg":"val"}"
/// ```
/// (special tokens stripped; role+recipient+channel text tokens remain)
///
/// Returns `(function_name, args_json)` if a tool call is detected.
pub fn parse_harmony_tool_call(decoded: &str) -> Option<(String, serde_json::Value)> {
    // Detect by presence of "functions." in the decoded text.
    // This pattern only appears in tool call recipient tokens.
    let marker = "functions.";
    let pos = decoded.find(marker)?;
    let after = &decoded[pos + marker.len()..];

    // Extract function name: identifier chars (alphanumeric + underscore).
    let func_name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();

    if func_name.is_empty() {
        return None;
    }

    // Find the JSON arguments. The generated text contains exactly one top-level
    // JSON object (the call's arguments), but its `content` field may itself
    // contain `{` and `}` (e.g. `func main() {`), so `rfind('{')` would land
    // inside the string literal. Use the FIRST `{` after the function-name
    // marker and the LAST `}` before any trailing markers.
    let json_start = decoded[pos..].find('{').map(|i| pos + i)?;
    let json_end = decoded.rfind('}')?;
    if json_end < json_start {
        return None;
    }
    let args: serde_json::Value =
        serde_json::from_str(&decoded[json_start..=json_end]).ok()?;

    tracing::debug!("Harmony tool call: {}({:?})", func_name, args);
    Some((func_name, args))
}

// ============================================================================
// Harmony helpers
// ============================================================================

/// Build a TypeScript namespace block from tool definitions.
///
/// ```text
/// namespace functions {
///   // description
///   type func_name = (_: {
///     // param description
///     param: string,
///     optional?: number,
///   }) => any;
/// }
/// ```
fn tools_to_harmony_namespace(tools: &[ToolDefinition]) -> String {
    let mut s = String::from("namespace functions {\n");
    for tool in tools {
        s.push_str(&format!("// {}\n", tool.description));
        s.push_str(&format!("type {} = (_: {{\n", tool.name));
        if let Some(props) = tool.parameters.get("properties").and_then(|p| p.as_object()) {
            let required: Vec<&str> = tool.parameters
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            for (name, schema) in props {
                if let Some(desc) = schema.get("description").and_then(|d| d.as_str()) {
                    s.push_str(&format!("  // {}\n", desc));
                }
                let opt = if required.contains(&name.as_str()) { "" } else { "?" };
                s.push_str(&format!("  {}{}: {},\n", name, opt, json_schema_to_ts(schema)));
            }
        }
        s.push_str("}) => any;\n\n");
    }
    s.push('}');
    s
}

fn json_schema_to_ts(schema: &serde_json::Value) -> &'static str {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("string") => "string",
        Some("integer") | Some("number") => "number",
        Some("boolean") => "boolean",
        Some("array") => "any[]",
        Some("object") => "object",
        _ => "any",
    }
}

// (Gemma type mapping is in json_schema_to_gemma_type above)

/// Find the `final` Harmony channel in decoded output (word-boundary scan).
///
/// After `decode(skip_special=true)`, special tokens are stripped but channel name
/// text tokens remain. "final" appears directly adjacent to the answer content.
/// We find the last word-boundary occurrence and return everything after it.
fn extract_harmony_final(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let needle = b"final";
    let nlen = needle.len();
    let mut last_end: Option<usize> = None;

    let mut i = 0;
    while i + nlen <= bytes.len() {
        if bytes[i..i + nlen] == *needle {
            let pre_ok = i == 0 || !bytes[i - 1].is_ascii_alphabetic();
            let end = i + nlen;
            let post_ok = end >= bytes.len() || !bytes[end].is_ascii_alphabetic();
            if pre_ok && post_ok {
                last_end = Some(end);
            }
        }
        i += 1;
    }

    if let Some(end) = last_end {
        let after = &raw[end..];
        let trimmed = after.trim_start_matches(|c: char| c.is_whitespace());
        let result = trimmed.trim_end().to_string();
        if !result.is_empty() {
            return result;
        }
    }

    raw.trim().to_string()
}

// ============================================================================
// GemmaProtocol — Gemma 4
// ============================================================================

/// Gemma 4 protocol adapter.
///
/// ## Turn format
///
/// Gemma 4 uses special tokens for turn delimiters (NOT Gemma 2 text markers):
/// - `<|turn>` (ID 105) — start of a turn (`sot_token`)
/// - `<turn|>` (ID 106) — end of a turn (`eot_token`)
///
/// Gemma 2 `<start_of_turn>` / `<end_of_turn>` tokenize as 7 regular BPE pieces
/// and are NOT recognized as turn boundaries by Gemma 4.
///
/// ## Tool calling (native Gemma 4 format)
///
/// Special tokens (all in the added-tokens vocabulary):
/// - `<|tool>` (46) / `<tool|>` (47) — tool declaration start/end
/// - `<|tool_call>` (48) / `<tool_call|>` (49) — tool call start/end
/// - `<|tool_response>` (50) / `<tool_response|>` (51) — tool response start/end
/// - `<|"|>` (52) — string value delimiter (`escape_token`)
///
/// ### Format (matches the Gemma 4 IT chat template exactly):
///
/// Tool declarations go inside the system turn:
/// ```text
/// <|turn>system
/// <|tool>declaration:write{description:<|"|>DESC<|"|>,parameters:{properties:{content:{...},file_path:{...}},required:[<|"|>file_path<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>
/// <turn|>
/// ```
///
/// Tool call (model output, stops at `<tool_call|>` EOS):
/// ```text
/// <|tool_call>call:write{content:<|"|>pkg main;...<|"|>,file_path:<|"|>hello.go<|"|>}<tool_call|>
/// ```
/// Note: argument keys are sorted alphabetically; values wrapped in `<|"|>`.
///
/// Tool response (injected inline, same model turn, no closing `<turn|>` before it):
/// ```text
/// <|tool_response>response:write{value:<|"|>ok<|"|>}<tool_response|>
/// ```
///
/// After all call+response pairs, the next model turn opens for continuation:
/// ```text
/// <|turn>model
/// (next tool call or final answer)
/// ```
///
/// ## Thinking
///
/// Optional; activated with `GemmaProtocol::with_thinking()`.
/// Adds `<|think|>` to the system turn; `parse_response` strips
/// `<|channel>thought...<channel|>`.
pub struct GemmaProtocol {
    pub thinking: bool,
    /// Tracks what was prepended as the tool-call prefix in the last
    /// `format_prompt_with_tools` call (e.g. `"write{content:<|\"|\>"`).
    /// `parse_tool_call` prepends this to the raw output so the standard
    /// `parse_gemini_tool_call` can find a complete `<|tool_call>call:NAME{...}`.
    tool_call_prefill: std::cell::RefCell<String>,
}

impl GemmaProtocol {
    pub fn new() -> Self {
        Self { thinking: false, tool_call_prefill: std::cell::RefCell::new(String::new()) }
    }

    pub fn with_thinking() -> Self {
        Self { thinking: true, tool_call_prefill: std::cell::RefCell::new(String::new()) }
    }
}

impl Default for GemmaProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelProtocol for GemmaProtocol {
    fn supports_tools(&self) -> bool {
        true
    }

    fn tool_stop_tokens(&self) -> &[&'static str] {
        // <tool_call|> (ID 49): end of a native Gemma 4 tool call block.
        //   Stops generation so we can inject the tool response.
        // <turn|> (ID 106): end of a model turn.
        //   Stops text responses (model signals "I'm done with this turn").
        &["<tool_call|>", "<turn|>"]
    }

    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut s = String::new();
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    let thinking_tag = if self.thinking { "<|think|>\n" } else { "" };
                    s.push_str(&format!(
                        "<|turn>system\n{}{}<turn|>\n",
                        thinking_tag, msg.content
                    ));
                }
                ChatRole::User => {
                    s.push_str(&format!("<|turn>user\n{}<turn|>\n", msg.content));
                }
                ChatRole::Tool => {
                    s.push_str(&format!("<|turn>user\n{}<turn|>\n", msg.content));
                }
                ChatRole::Assistant => {
                    if msg.tool_calls.is_none() && !msg.content.is_empty() {
                        s.push_str(&format!("<|turn>model\n{}<turn|>\n", msg.content));
                    }
                }
            }
        }
        s.push_str("<|turn>model\n");
        s
    }

    /// Format prompt with tools using the native Gemma 4 IT chat template.
    ///
    /// Matches the exact format from the official tokenizer chat template:
    ///
    /// ```text
    /// <bos>
    /// <|turn>system
    /// [opt: thinking tag]
    /// [opt: user system message]
    /// <|tool>declaration:write{description:<|"|>DESC<|"|>,...}<tool|>
    /// <|tool>declaration:done{...}<tool|>
    /// <turn|>
    /// <|turn>user
    /// user message<turn|>
    /// <|turn>model
    /// <|tool_call>call:write{content:<|"|>...<|"|>,file_path:<|"|>hello.go<|"|>}<tool_call|>
    /// <|tool_response>response:write{value:<|"|>ok<|"|>}<tool_response|>
    /// <|turn>model
    /// (next generation or tool call)
    /// ```
    ///
    /// Key points:
    /// - Tool declarations use `<|tool>...<tool|>` inside the system turn
    /// - Properties sorted alphabetically; values wrapped in `<|"|>`
    /// - Tool responses are inline in the same model turn as the call, no `<turn|>` separator
    /// - After tool response(s), the next model turn opens for continuation
    /// - No prefill: model generates `<|tool_call>call:...` naturally from context
    fn format_prompt_with_tools(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
        let thinking_tag = if self.thinking { "<|think|>\n" } else { "" };

        // Build system turn: thinking tag + optional user system message + tool declarations.
        let system_content = messages.iter().find_map(|m| {
            if m.role == ChatRole::System { Some(m.content.as_str()) } else { None }
        });

        let mut system_body = String::new();
        if self.thinking {
            system_body.push_str(thinking_tag);
        }
        if let Some(sc) = system_content {
            system_body.push_str(sc.trim());
            system_body.push('\n');
        }
        // Tool declarations (alphabetically sorted properties per Gemma 4 template).
        for tool in tools {
            system_body.push_str(&gemini_tool_declaration(tool));
        }

        let mut s = format!("<|turn>system\n{system_body}<turn|>\n");

        // Render messages, pairing (Assistant tool_calls) + (Tool results) in one model turn.
        // `in_model_turn` tracks whether the previous emission left the model turn open
        // (tool_call followed by inline tool_response leaves it open per the Gemma 4
        // chat template — the next tool_call or text continues the same turn).
        let mut in_model_turn = false;
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::System => { i += 1; } // already in system turn above
                ChatRole::User => {
                    s.push_str(&format!("<|turn>user\n{}<turn|>\n", msg.content));
                    in_model_turn = false;
                    i += 1;
                }
                ChatRole::Tool => {
                    // Orphan Tool result (no preceding Assistant call) — skip.
                    i += 1;
                }
                ChatRole::Assistant => {
                    if let Some(ref calls) = msg.tool_calls {
                        if !in_model_turn {
                            s.push_str("<|turn>model\n");
                            in_model_turn = true;
                        }
                        for call in calls {
                            let args_str = gemini_format_args(&call.arguments);
                            s.push_str(&format!("<|tool_call>call:{}{args_str}<tool_call|>", call.name));
                        }
                        i += 1;

                        // Consume all immediately following Tool messages and inline their responses.
                        while i < messages.len() && messages[i].role == ChatRole::Tool {
                            let tool_msg = &messages[i];
                            let func = tool_msg.tool_name.as_deref().unwrap_or("unknown");
                            let encoded = gemini_str_value(&tool_msg.content);
                            s.push_str(&format!(
                                "<|tool_response>response:{func}{{value:{encoded}}}<tool_response|>"
                            ));
                            i += 1;
                        }
                        // Note: no <turn|> — the model turn with call+response stays open;
                        // the next assistant message continues in the same turn.
                    } else if !msg.content.is_empty() {
                        if !in_model_turn {
                            s.push_str("<|turn>model\n");
                        }
                        // Defense in depth: even if thinking somehow reached memory
                        // (e.g. a pre-existing message from an older build), strip it
                        // before replaying so the model never sees prior thinking.
                        let body = strip_thinking_blocks(&msg.content);
                        s.push_str(&format!("{}<turn|>\n", body.trim()));
                        in_model_turn = false;
                        i += 1;
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Open a new model turn for generation only if the previous emission closed
        // its turn.  After a tool_call+tool_response pair the model turn is still
        // open and the generator continues in-place (matching the official template's
        // `add_generation_prompt` logic).
        if !in_model_turn {
            s.push_str("<|turn>model\n");
        }
        s
    }

    /// Extract the final answer from model output decoded with skip_special=false.
    ///
    /// Strips both forms of thinking content (`<|channel>…<channel|>` prefix and
    /// paired `<|think|>…<|/think|>` blocks) so memory never holds thinking text
    /// — the model card forbids replaying prior thinking into subsequent turns.
    /// Then trims trailing `<turn|>` / `<eos>` markers.
    fn parse_response(&self, raw: &str) -> String {
        let cleaned = strip_thinking_blocks(raw);
        strip_gemma_specials(&cleaned).to_string()
    }

    fn parse_tool_call(&self, raw: &str) -> Option<(String, serde_json::Value)> {
        parse_gemini_tool_call(raw)
            .or_else(|| parse_gemini_tool_call_continuation(raw))
            .or_else(|| parse_gemma_json_tool_call(raw))
            .or_else(|| parse_gemma_action_tool_call(raw))
    }
}


/// Parse the continuation of the `{"tool":"` prompt prefill.
///
/// The model only needs to complete `TOOL_NAME","args":{...}}`.
/// In practice the model often produces slightly malformed JSON (missing commas,
/// merged keys like `"hello.go.content": "..."`).  This function:
///
/// 1. Extracts the tool name (up to the first unescaped `"`).
/// 2. Tries a strict JSON reconstruction first.
/// 3. Falls back to a lenient key-value scanner for malformed output.
/// 4. Normalises the merged `"filename.param": "value"` pattern.
pub fn parse_gemma_prefill_continuation(raw: &str) -> Option<(String, serde_json::Value)> {
    // 1. Tool name is everything before the first '"'
    let tool_end = raw.find('"')?;
    let tool_name = raw[..tool_end].trim().to_lowercase();
    if tool_name.is_empty() {
        return None;
    }

    // 2. Try strict JSON reconstruction (works when model output is well-formed)
    let reconstructed = format!("{{\"tool\":\"{raw}");
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&reconstructed) {
        if let Some(args_obj) = v.get("args").and_then(|a| a.as_object()) {
            if !args_obj.is_empty() {
                tracing::debug!("Gemma prefill (strict): {}({:?})", tool_name, args_obj);
                return Some((tool_name, serde_json::Value::Object(args_obj.clone())));
            }
        }
    }

    // 3. Lenient scan: extract all "key": "value" pairs from after the tool name.
    //    raw[tool_end] is the closing '"' of the tool name — skip it so the
    //    scanner doesn't treat it as the opening of a new string.
    let rest = &raw[tool_end + 1..];
    let mut args = scan_string_kvpairs(rest);

    // 4. Handle merged `"filename.param": "value"` — the model sometimes
    //    concatenates file_path and a param name into one key.
    let merged_keys: Vec<String> = args
        .keys()
        .filter(|k| k.contains('.'))
        .cloned()
        .collect();
    for mk in merged_keys {
        if let Some(v) = args.remove(&mk) {
            if let Some(dot) = mk.find('.') {
                let file_part = mk[..dot].to_string();
                let param_part = mk[dot + 1..].to_string();
                args.entry("file_path".to_string())
                    .or_insert_with(|| serde_json::Value::String(file_part));
                args.entry(param_part).or_insert(v);
            }
        }
    }

    // Normalise "file" → "file_path"
    if let Some(v) = args.remove("file") {
        args.entry("file_path".to_string()).or_insert(v);
    }

    if args.is_empty() && tool_name == "write" {
        return None; // not enough info to call write
    }
    tracing::debug!("Gemma prefill (lenient): {}({:?})", tool_name, args);
    Some((tool_name, serde_json::Value::Object(args)))
}

/// Read one JSON string starting at `chars[*i]` (which should be `"`).
/// Advances `*i` past the closing `"`.  Returns `None` if not at a `"`.
fn read_json_string(chars: &[char], i: &mut usize) -> Option<String> {
    if *i >= chars.len() || chars[*i] != '"' {
        return None;
    }
    *i += 1; // skip opening '"'
    let mut s = String::new();
    let mut escaped = false;
    while *i < chars.len() {
        let c = chars[*i];
        *i += 1;
        if escaped {
            escaped = false;
            match c {
                'n'  => s.push('\n'),
                't'  => s.push('\t'),
                '"'  => s.push('"'),
                '\\' => s.push('\\'),
                _    => { s.push('\\'); s.push(c); }
            }
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            break;
        } else {
            s.push(c);
        }
    }
    Some(s)
}

/// Scan `text` for `"key": "value"` patterns and return an args map.
///
/// Skips the structural keys `"tool"` and `"args"`.
///
/// Handles the merged-key pattern the model sometimes emits:
///   `"file": "hello.go.content": "actual content"`
/// where `hello.go.content` is two parameters merged into a "value" that
/// is immediately followed by another `:`.  In that case we split at `.`
/// to recover `file_path = "hello.go"` and `content = "actual content"`.
fn scan_string_kvpairs(text: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut args = serde_json::Map::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Skip to the next '"' — potential start of a key
        if chars[i] != '"' { i += 1; continue; }

        let key = match read_json_string(&chars, &mut i) {
            Some(k) => k,
            None    => { i += 1; continue; }
        };

        // Must be a valid identifier-like key (no whitespace, not structural)
        if key.is_empty()
            || key == "tool"
            || key == "args"
            || !key.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }

        // Skip whitespace then require ':'
        while i < chars.len() && chars[i].is_whitespace() { i += 1; }
        if i >= chars.len() || chars[i] != ':' { continue; }
        i += 1;

        // Skip whitespace then require '"' for string value
        while i < chars.len() && chars[i].is_whitespace() { i += 1; }
        if i >= chars.len() || chars[i] != '"' { continue; }

        let val_start = i;
        let val = match read_json_string(&chars, &mut i) {
            Some(v) => v,
            None    => continue,
        };

        // Lookahead: if `:` follows immediately after the value's closing `"`,
        // this "value" is actually a merged key (model confusion pattern).
        // e.g. "file": "hello.go.content": "package main..."
        let next_non_ws = chars[i..].iter().find(|&&c| !c.is_whitespace());
        if next_non_ws == Some(&':') {
            // The "value" is really a merged "filepath.param_name" key.
            // Split at the last '.' to get the file path and the param name.
            if let Some(dot) = val.rfind('.') {
                let file_val = val[..dot].to_string();
                let param_key = val[dot + 1..].to_string();

                // The actual value for param_key is the NEXT string after the ':'
                while i < chars.len() && chars[i] != ':' { i += 1; }
                if i < chars.len() { i += 1; } // skip ':'
                while i < chars.len() && chars[i].is_whitespace() { i += 1; }
                let actual_val = read_json_string(&chars, &mut i);

                // Normalise the outer key to file_path
                let file_key = if key == "file" || key == "path" { "file_path".to_string() } else { key.clone() };
                args.entry(file_key).or_insert_with(|| serde_json::Value::String(file_val));
                if let Some(av) = actual_val {
                    args.entry(param_key).or_insert_with(|| serde_json::Value::String(av));
                }
            } else {
                // No dot: just treat the whole thing as the value for `key`
                // (skip the extra `:` and the value that follows it)
                let norm_key = if key == "file" || key == "path" { "file_path".to_string() } else { key };
                args.entry(norm_key).or_insert_with(|| serde_json::Value::String(val));
                while i < chars.len() && chars[i] != ':' { i += 1; }
                if i < chars.len() { i += 1; }
            }
        } else {
            // Normal case: store key → val, normalising known aliases
            let norm_key = if key == "file" || key == "path" { "file_path".to_string() } else { key };
            args.entry(norm_key).or_insert_with(|| serde_json::Value::String(val));
        }

        let _ = val_start; // suppress unused warning
    }
    args
}

/// Parse a plain-text TOOL: format tool call.
///
/// The model prompt ends with `TOOL: ` as a prefill; `raw` is the continuation.
/// Expected format (starting from the continuation):
/// ```text
/// write            ← tool name (already prefixed by "TOOL: " in the prompt)
/// file_path: hello.go
/// content: package main;import "fmt";func main(){fmt.Println("Hello, World!")}
/// DONE
/// ```
///
/// `done` is treated as a text-response signal by `GalliumProvider`.
pub fn parse_gemma_tool_format(raw: &str) -> Option<(String, serde_json::Value)> {
    let mut lines = raw.lines();

    // First line is the tool name (completion of the "TOOL: " prefill)
    let tool_name = lines.next()?.trim().to_lowercase();
    if tool_name.is_empty() {
        return None;
    }
    // Strip any trailing punctuation the model might add
    let tool_name = tool_name.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
    if tool_name.is_empty() {
        return None;
    }

    let mut args = serde_json::Map::new();
    for line in lines {
        let line = line.trim();
        if line == "DONE" || line.starts_with("<end_of_turn>") || line.starts_with("<eos>") {
            break;
        }
        if let Some(colon) = line.find(": ") {
            let key = line[..colon].trim().to_lowercase();
            let val = line[colon + 2..].to_string();
            if !key.is_empty() {
                // Normalise "file" / "path" aliases
                let key = if key == "file" || key == "path" { "file_path".to_string() } else { key };
                args.entry(key).or_insert_with(|| serde_json::Value::String(val));
            }
        }
    }

    tracing::debug!("Gemma TOOL format: {}({:?})", tool_name, args);
    Some((tool_name, serde_json::Value::Object(args)))
}

/// Detect a JSON tool call in model output (fallback path).
///
/// Handles `{"tool":"NAME","args":{...}}` on a single line.
pub fn parse_gemma_json_tool_call(raw: &str) -> Option<(String, serde_json::Value)> {
    for line in raw.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(name) = v.get("tool").and_then(|n| n.as_str()) else {
            continue;
        };
        let args = v.get("args").cloned().unwrap_or(serde_json::json!({}));
        tracing::debug!("Gemma JSON tool call: {}({:?})", name, args);
        return Some((name.to_string(), args));
    }
    None
}

/// Detect a Gemma 4 native `Action: TOOL\n  key: value` tool call block.
///
/// The model emits this format and terminates with `<execute>` (used as EOS).
/// We take the **last** value seen for each key since the model sometimes
/// revises its parameters while generating.
pub fn parse_gemma_action_tool_call(raw: &str) -> Option<(String, serde_json::Value)> {
    let mut func_name: Option<String> = None;
    let mut args: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

    for line in raw.lines() {
        let line = line.trim();
        // "Action: write" or "Action: write." (trailing punct)
        if let Some(name) = line.strip_prefix("Action:").or_else(|| line.strip_prefix("action:")) {
            let name = name.trim().trim_matches('"').trim_matches('\'')
                .trim_end_matches('.').trim_end_matches(',').trim().to_lowercase();
            if !name.is_empty() {
                func_name = Some(name);
                // Don't clear args — keep accumulating across multiple Action: lines.
                // The model sometimes emits args under an earlier Action: and then
                // repeats the tool name as a bare "Action: write" with no args.
            }
            continue;
        }
        if func_name.is_none() {
            continue;
        }
        // "  file_path: "hello.go""
        if let Some(colon) = line.find(':') {
            let key = line[..colon].trim().trim_matches('"');
            // Only accept simple identifier keys
            if key.is_empty() || !key.chars().all(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            let val_str = line[colon + 1..].trim();
            if val_str.is_empty() {
                continue;
            }
            // Parse as JSON value (handles quoted strings with escape sequences)
            let val = serde_json::from_str::<serde_json::Value>(val_str)
                .unwrap_or_else(|_| serde_json::Value::String(val_str.trim_matches('"').to_string()));
            args.insert(key.to_string(), val);
        }
    }

    let name = func_name?;
    tracing::debug!("Gemma action tool call: {}({:?})", name, args);
    Some((name, serde_json::Value::Object(args)))
}

// ============================================================================
// Gemma 4 native special-token helpers
// ============================================================================

/// Build a Gemma 4 native tool declaration block.
///
/// Matches the Gemma 4 IT chat template exactly:
/// ```text
/// <|tool>declaration:FUNC{description:<|"|>DESC<|"|>,parameters:{properties:{content:{description:<|"|>D<|"|>,type:<|"|>STRING<|"|>},file_path:{...}},required:[<|"|>file_path<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>
/// ```
/// Properties are sorted alphabetically (matching the template's `| dictsort`).
fn gemini_tool_declaration(tool: &ToolDefinition) -> String {
    let mut s = format!("<|tool>declaration:{}", tool.name);
    s.push('{');
    s.push_str("description:");
    s.push_str(&gemini_str_value(&tool.description));

    if let Some(props) = tool.parameters.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<&str> = tool.parameters
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Sort properties alphabetically (Gemma 4 template uses dictsort).
        let mut sorted_props: Vec<(&String, &serde_json::Value)> = props.iter().collect();
        sorted_props.sort_by_key(|(k, _)| k.as_str());

        s.push_str(",parameters:{properties:{");
        for (i, (name, schema)) in sorted_props.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(name);
            s.push_str(":{");
            if let Some(desc) = schema.get("description").and_then(|d| d.as_str()) {
                s.push_str("description:");
                s.push_str(&gemini_str_value(desc));
                s.push(',');
            }
            s.push_str("type:");
            s.push_str(&gemini_str_value(json_schema_to_gemma_type(schema)));
            s.push('}');
        }

        s.push_str("},required:[");
        for (i, req) in required.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&gemini_str_value(req));
        }
        s.push_str("],type:");
        s.push_str(&gemini_str_value("OBJECT"));
        s.push('}');
    }
    s.push_str("}<tool|>");
    s
}

/// Encode a string value in Gemma 4's `<|"|>value<|"|>` format.
fn gemini_str_value(s: &str) -> String {
    format!("<|\"|>{}<|\"|>", s)
}

/// Encode tool call arguments in Gemma 4's `{key:<|"|>val<|"|>,...}` format.
///
/// Keys are sorted alphabetically (matching the Gemma 4 IT chat template's `| dictsort`).
/// String values are wrapped in `<|"|>` (ID 52); keys are bare identifiers.
fn gemini_format_args(args: &serde_json::Value) -> String {
    let mut s = String::from('{');
    if let Some(obj) = args.as_object() {
        let mut sorted: Vec<(&String, &serde_json::Value)> = obj.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());
        for (i, (key, val)) in sorted.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(key);
            s.push(':');
            match val {
                serde_json::Value::String(v) => s.push_str(&gemini_str_value(v)),
                serde_json::Value::Number(n) => s.push_str(&n.to_string()),
                serde_json::Value::Bool(b)   => s.push_str(if *b { "true" } else { "false" }),
                serde_json::Value::Null      => s.push_str("null"),
                other                        => s.push_str(&gemini_str_value(&other.to_string())),
            }
        }
    }
    s.push('}');
    s
}

#[allow(dead_code)]
fn json_schema_to_gemma_type(schema: &serde_json::Value) -> &'static str {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("string") => "STRING",
        Some("integer") => "INTEGER",
        Some("number") => "NUMBER",
        Some("boolean") => "BOOLEAN",
        Some("array") => "ARRAY",
        Some("object") => "OBJECT",
        _ => "STRING",
    }
}

/// Detect and parse a Gemma 4 native-token tool call.
///
/// The model output (before `<tool_call|>` EOS) looks like:
/// ```text
/// <|tool_call>call:write{file_path:<|"|>hello.go<|"|>,content:<|"|>...<|"|>}
/// ```
///
/// The model may also emit a verbose name prefix like `call:TOOL_name:write{...}`;
/// we extract the tool name as the last colon-separated segment before `{`.
pub fn parse_gemini_tool_call(raw: &str) -> Option<(String, serde_json::Value)> {
    const MARKER: &str = "<|tool_call>";
    const CALL_PREFIX: &str = "call:";

    let start = raw.find(MARKER)? + MARKER.len();
    let rest = raw[start..].trim_start();

    if !rest.starts_with(CALL_PREFIX) {
        return None;
    }
    let rest = &rest[CALL_PREFIX.len()..];

    let brace = rest.find('{')?;
    let raw_name = rest[..brace].trim();
    if raw_name.is_empty() {
        return None;
    }

    // If the model emits "PREFIX:ACTUAL_NAME", take the last segment after ':'.
    let raw_func = raw_name.rsplit(':').next().unwrap_or(raw_name).trim().to_lowercase();
    if raw_func.is_empty() {
        return None;
    }
    let func_name = normalise_tool_name(&raw_func);

    // Find the outer closing brace.
    let args_section = &rest[brace..];
    let close = args_section.rfind('}')?;
    let inner = &args_section[1..close]; // between { and }

    let mut args_val = parse_gemma_kv_args(inner);

    // Normalise "file" / "path" → "file_path" (model sometimes uses short aliases).
    if let Some(map) = args_val.as_object_mut() {
        if let Some(v) = map.remove("file") {
            map.entry("file_path".to_string()).or_insert(v);
        }
        if let Some(v) = map.remove("path") {
            map.entry("file_path".to_string()).or_insert(v);
        }
    }

    tracing::debug!("Gemini tool call: {}({:?})", func_name, args_val);
    Some((func_name, args_val))
}

/// Normalise common tool name aliases the model may emit.
///
/// Gemma 4 4B at low temperature often completes `call:` with `write_file`,
/// `create_file`, etc. instead of the registered `write` name.
fn normalise_tool_name(name: &str) -> String {
    match name {
        "write_file" | "create_file" | "file_write" | "write_to_file" | "writefile"
        | "write_tool" | "writetool" | "write_content" | "create" => {
            "write".to_string()
        }
        "read_file" | "file_read" | "readfile" | "open_file" | "read_tool" => "read".to_string(),
        "list_files" | "list_file" | "ls" | "find_files" | "glob_tool" => "glob".to_string(),
        "edit_file" | "file_edit" | "update_file" | "patch_file" | "edit_tool" => "edit".to_string(),
        _ => name.to_string(),
    }
}

/// Parse the continuation of the `<|tool_call>call:` prefill.
///
/// When `format_prompt_with_tools` ends with `<|tool_call>call:` as prefill,
/// the raw model output is `NAME{args}<tool_call|>` (no leading `<|tool_call>call:`).
pub fn parse_gemini_tool_call_continuation(raw: &str) -> Option<(String, serde_json::Value)> {
    // Strip leading thinking junk if present (model sometimes emits text before the call)
    let raw = raw.trim_start();

    // Find the first `{` to delimit the function name
    let brace = raw.find('{')?;
    let raw_name = raw[..brace].trim();
    if raw_name.is_empty() {
        return None;
    }

    // Name must look like an identifier (no whitespace, newlines, etc.)
    // If there's a newline in the "name" part, this is not a valid continuation.
    if raw_name.contains('\n') || raw_name.contains(' ') {
        return None;
    }

    // If the model emits "PREFIX:ACTUAL_NAME", take the last segment after ':'.
    let raw_func = raw_name.rsplit(':').next().unwrap_or(raw_name).trim().to_lowercase();
    if raw_func.is_empty() {
        return None;
    }

    // Normalise common tool name aliases the model may emit
    let func_name = normalise_tool_name(&raw_func);

    // Find the outer closing brace.
    let args_section = &raw[brace..];
    let close = args_section.rfind('}')?;
    let inner = &args_section[1..close];

    let mut args_val = parse_gemma_kv_args(inner);

    // Normalise "file" / "path" → "file_path"
    if let Some(map) = args_val.as_object_mut() {
        if let Some(v) = map.remove("file") {
            map.entry("file_path".to_string()).or_insert(v);
        }
        if let Some(v) = map.remove("path") {
            map.entry("file_path".to_string()).or_insert(v);
        }
    }

    tracing::debug!("Gemini tool call continuation: {}({:?})", func_name, args_val);
    Some((func_name, args_val))
}

/// Parse Gemma 4 key-value argument pairs (native-token format).
///
/// Format: `key:<|"|>strval<|"|>,key2:42,...`
/// String values are delimited by `<|"|>`, non-strings are bare.
#[allow(dead_code)]
fn parse_gemma_kv_args(inner: &str) -> serde_json::Value {
    const STR_DELIM: &str = "<|\"|>";
    let mut map = serde_json::Map::new();
    let mut s = inner;

    loop {
        s = s.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
        if s.is_empty() {
            break;
        }

        // Read key until ':'
        let colon = match s.find(':') {
            Some(p) => p,
            None => break,
        };
        let key = s[..colon].trim().to_string();
        s = &s[colon + 1..];
        if key.is_empty() {
            break;
        }

        if s.starts_with(STR_DELIM) {
            // String value enclosed in <|"|>...<|"|>
            s = &s[STR_DELIM.len()..];
            match s.find(STR_DELIM) {
                Some(end) => {
                    let val = s[..end].to_string();
                    map.insert(key, serde_json::Value::String(val));
                    s = &s[end + STR_DELIM.len()..];
                }
                None => {
                    // Malformed: consume rest
                    map.insert(key, serde_json::Value::String(s.to_string()));
                    break;
                }
            }
        } else {
            // Non-string: read until next comma or end
            let end = s.find(',').unwrap_or(s.len());
            let val_str = s[..end].trim();
            map.insert(key, parse_gemma_scalar(val_str));
            s = &s[end..];
        }
    }

    serde_json::Value::Object(map)
}

#[allow(dead_code)]
fn parse_gemma_scalar(s: &str) -> serde_json::Value {
    match s {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "null" => serde_json::Value::Null,
        _ => {
            if let Ok(n) = s.parse::<i64>() {
                return serde_json::json!(n);
            }
            if let Ok(f) = s.parse::<f64>() {
                return serde_json::json!(f);
            }
            serde_json::Value::String(s.to_string())
        }
    }
}

/// Strip known Gemma 4 special token strings from decoded output and trim.
///
/// With `skip_special=false`, special tokens decode to their name strings.
/// Gemma 4 uses `<turn|>` (ID 106) as end-of-turn and `<eos>` (ID 1) as EOS.
fn strip_gemma_specials(s: &str) -> &str {
    let mut s = s.trim();
    loop {
        let prev = s;
        s = s.trim_end_matches("<turn|>").trim();
        s = s.trim_end_matches("<eos>").trim();
        // Also strip Gemma 2 format for compatibility
        s = s.trim_end_matches("<end_of_turn>").trim();
        if s == prev {
            break;
        }
    }
    s
}

/// Remove Gemma 4 thinking blocks from a message body.
///
/// Per the model card, multi-turn prompts must NOT include previous thinking
/// content. We strip both forms the model may emit:
///   - `<|think|>…<|/think|>` paired wrappers
///   - `<|channel>…<channel|>` (retain only the text after the last channel close)
///
/// Applied to assistant history *and* to the freshly parsed response stored in
/// memory, so thinking content never re-enters a subsequent prompt.
fn strip_thinking_blocks(s: &str) -> String {
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

// ============================================================================
// QwenProtocol — Qwen 3.5 (ChatML)
// ============================================================================

/// Qwen 3.5 ChatML adapter (`<|im_start|>role`).
pub struct QwenProtocol;

impl ModelProtocol for QwenProtocol {
    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut s = String::new();
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::User | ChatRole::Tool => {
                    s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::Assistant => {
                    if msg.tool_calls.is_none() && !msg.content.is_empty() {
                        s.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", msg.content));
                    }
                }
            }
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }

    /// Strip trailing `<|im_end|>` that appears when decoding with skip_special=false.
    fn parse_response(&self, raw: &str) -> String {
        let s = raw.trim();
        let s = s.strip_suffix("<|im_end|>").unwrap_or(s).trim();
        s.to_string()
    }
}

// ============================================================================
// Date helpers
// ============================================================================

fn current_date_ymd() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    epoch_days_to_ymd(secs / 86400)
}

fn epoch_days_to_ymd(mut days: u64) -> String {
    let mut year = 1970u32;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year { break; }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 1u32;
    for &md in &month_days {
        if days < md { break; }
        days -= md;
        month += 1;
    }
    format!("{year:04}-{month:02}-{:02}", days + 1)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Gemma action-format tool call parsing ---

    #[test]
    fn test_parse_gemma_action_tool_call_simple() {
        let raw = "Action: write\n  file_path: \"hello.go\"\n  content: \"package main\\n\"";
        let (name, args) = parse_gemma_action_tool_call(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.go");
        assert_eq!(args["content"], "package main\n");
    }

    #[test]
    fn test_parse_gemma_action_tool_call_takes_last_value() {
        // Model sometimes emits the same key multiple times; last value wins.
        let raw = "Action: write\n  file_path: \"a.go\"\n  content: \"v1\"\n  content: \"v2\"";
        let (_, args) = parse_gemma_action_tool_call(raw).unwrap();
        assert_eq!(args["content"], "v2");
    }

    #[test]
    fn test_parse_gemma_action_multi_action_keeps_args() {
        // Model emits args under an earlier Action line, then repeats tool name bare.
        let raw = "Action: FILE\nparam: \"content\"\nAction: write.\nfile_path: \"hello.txt\"\nparam: \"content\"\nAction: \"write\"\n<eos>";
        let (name, args) = parse_gemma_action_tool_call(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.txt");
    }

    #[test]
    fn test_parse_gemma_action_trailing_dot() {
        let raw = "Action: write.\n  file_path: \"out.txt\"\n  content: \"hi\"";
        let (name, args) = parse_gemma_action_tool_call(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "out.txt");
    }

    #[test]
    fn test_parse_gemma_action_case_insensitive() {
        let raw = "action: read\n  file_path: \"main.rs\"";
        let (name, args) = parse_gemma_action_tool_call(raw).unwrap();
        assert_eq!(name, "read");
        assert_eq!(args["file_path"], "main.rs");
    }

    // --- Gemma prefill continuation parsing ---

    #[test]
    fn test_parse_gemma_prefill_strict() {
        // Well-formed continuation: model output is valid JSON after the prefill
        let raw = r#"write","args":{"file_path":"hello.go","content":"package main\n"}}"#;
        let (name, args) = parse_gemma_prefill_continuation(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.go");
    }

    #[test]
    fn test_parse_gemma_prefill_merged_key() {
        // Model merges file_path + content into "file.content" key
        let raw = "write\"\n\"args\": {\"file\": \"hello.go.content\": \"package main\\n\"}";
        let (name, args) = parse_gemma_prefill_continuation(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.go");
    }

    #[test]
    fn test_parse_gemma_prefill_file_alias() {
        // Model uses "file" instead of "file_path"
        let raw = "write\"\n\"args\": {\"file\": \"hello.go\", \"content\": \"hi\"}";
        let (name, args) = parse_gemma_prefill_continuation(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.go");
        assert_eq!(args["content"], "hi");
    }

    // --- Gemma JSON tool call parsing (fallback) ---

    #[test]
    fn test_parse_gemma_json_tool_call_simple() {
        let raw = r#"{"tool":"write","args":{"file_path":"hello.go","content":"package main\n"}}"#;
        let (name, args) = parse_gemma_json_tool_call(raw).unwrap();
        assert_eq!(name, "write");
        assert_eq!(args["file_path"], "hello.go");
    }

    #[test]
    fn test_parse_gemma_json_tool_call_no_match() {
        assert!(parse_gemma_json_tool_call("Hello, I am a model.").is_none());
        assert!(parse_gemma_json_tool_call(r#"{"key":"value"}"#).is_none());
    }

    // --- Gemma 4 native tool format ---

    #[test]
    fn test_gemma_format_with_tools_native_declaration() {
        let proto = GemmaProtocol::new();
        let tools = vec![ToolDefinition {
            name: "write".to_string(),
            description: "Create or overwrite a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path" },
                    "content":   { "type": "string", "description": "Content" }
                },
                "required": ["file_path", "content"]
            }),
        }];
        let msgs = vec![crate::llm::ChatMessage::user("Write hello.go".to_string())];
        let prompt = proto.format_prompt_with_tools(&msgs, &tools);
        // Should have tool declaration in system turn
        assert!(prompt.contains("<|tool>declaration:write{"), "expected tool declaration");
        assert!(prompt.contains("<tool|>"), "expected tool declaration end");
        assert!(prompt.contains("file_path"), "expected file_path param");
        // Properties sorted alphabetically: content before file_path
        let content_pos = prompt.find("content:{").unwrap_or(usize::MAX);
        let file_pos = prompt.find("file_path:{").unwrap_or(usize::MAX);
        assert!(content_pos < file_pos, "properties should be sorted: content before file_path");
        // Model turn opener at the end
        assert!(prompt.ends_with("<|turn>model\n"),
            "expected model turn opener at end, got: {:?}", &prompt[prompt.len().saturating_sub(60)..]);
    }

    #[test]
    fn test_gemma_format_with_tools_replay() {
        use crate::llm::{ChatMessage, ToolCallInfo};
        let proto = GemmaProtocol::new();
        let tools = vec![ToolDefinition {
            name: "write".to_string(),
            description: "Create or overwrite a file".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
        }];
        let msgs = vec![
            ChatMessage::user("Write hello.go".to_string()),
            ChatMessage {
                role: crate::llm::ChatRole::Assistant,
                content: String::new(),
                tool_calls: Some(vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "write".to_string(),
                    arguments: serde_json::json!({"file_path":"hello.go","content":"hi"}),
                }]),
                tool_call_id: None,
                tool_name: None,
                images: vec![],
            },
            ChatMessage {
                role: crate::llm::ChatRole::Tool,
                content: "ok".to_string(),
                tool_calls: None,
                tool_call_id: Some("c1".to_string()),
                tool_name: Some("write".to_string()),
                images: vec![],
            },
        ];
        let prompt = proto.format_prompt_with_tools(&msgs, &tools);
        assert!(prompt.contains("<|tool_call>call:write{"), "expected tool call replay");
        assert!(prompt.contains("<|tool_response>response:write{"), "expected tool response");
        // Tool response should be inline (no <|turn>user between call and response)
        let call_pos = prompt.find("<|tool_call>call:write{").unwrap();
        let resp_pos = prompt.find("<|tool_response>response:write{").unwrap();
        let user_after_call = prompt[call_pos..].find("<|turn>user");
        assert!(user_after_call.is_none() || user_after_call.unwrap() > resp_pos - call_pos,
            "tool response should come before any <|turn>user after the call");
    }

    // --- Gemma parse_response with thinking ---

    #[test]
    fn test_parse_response_strips_thinking() {
        let proto = GemmaProtocol::with_thinking();
        let raw = "<|channel>thought\nThis is my reasoning\n<channel|>This is the answer.";
        let result = proto.parse_response(raw);
        assert_eq!(result, "This is the answer.");
    }

    #[test]
    fn test_parse_response_no_thinking() {
        let proto = GemmaProtocol::new();
        // Gemma 4 uses <turn|> (ID 106) as end-of-turn; <end_of_turn> (Gemma 2) is also handled.
        let raw = "Hello, world!<turn|>";
        assert_eq!(proto.parse_response(raw), "Hello, world!");
        let raw2 = "Hello, world!<end_of_turn>";
        assert_eq!(proto.parse_response(raw2), "Hello, world!");
    }

    // --- Harmony tool call detection still works with specials in output ---

    #[test]
    fn test_harmony_tool_call_with_specials() {
        // With skip_special=false, raw output includes literal special token strings.
        let raw = "<|start|>assistant to=functions.read_file<|channel|>commentary<|constrain|>json<|message|>{\"file_path\":\"src/main.rs\"}<|call|>";
        let result = parse_harmony_tool_call(raw);
        assert!(result.is_some());
        let (name, args) = result.unwrap();
        assert_eq!(name, "read_file");
        assert_eq!(args["file_path"], "src/main.rs");
    }

    // --- Qwen parse_response strips trailing im_end ---

    #[test]
    fn test_qwen_parse_response_strips_im_end() {
        let proto = QwenProtocol;
        let raw = "The answer is 42.<|im_end|>";
        assert_eq!(proto.parse_response(raw), "The answer is 42.");
    }
}
