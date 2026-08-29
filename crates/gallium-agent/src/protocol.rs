//! Prompt renderers: convert ChatMessage history → raw model prompt strings,
//! for the native candle backend.
//!
//! This module used to also own *parsing* raw decoded output back into a reply
//! or a tool call — the other half of what ADR 0003
//! (`docs/adr/0003-model-profiles.md`) calls the wire layer. That half has
//! moved to `crate::profile`, which both local backends now share (step 3-b);
//! a [`PromptRenderer`] renders a prompt and nothing else, since prompt
//! rendering is the one thing that legitimately differs per engine — llama.cpp
//! has the GGUF's own jinja template, candle has none and must build the
//! format itself.
//!
//! Each renderer knows:
//!   - `format_prompt`: render message list into a raw string the model expects
//!   - `format_prompt_with_tools`: like `format_prompt` but embeds tool definitions
//!
//! # Renderers
//!
//! | Renderer        | Model    | Tools | Thinking |
//! |-----------------|----------|-------|----------|
//! | HarmonyProtocol | GPT-OSS  | yes   | always on, effort text configurable |
//! | GemmaProtocol   | Gemma 4  | yes   | configurable, off by default |
//! | QwenProtocol    | Qwen 3.6 | yes   | configurable, on by default |
//! | Lfm2Protocol    | LFM2.5   | yes   | always on, not configurable |
//!
//! The wire format each renderer builds — and, since a renderer's output is
//! also what the model's raw reply looks like, what the matching
//! `crate::profile` parses back out — is documented where that parsing
//! actually lives now: [`crate::harmony`] for GPT-OSS's channels and Harmony
//! tool-call syntax, [`crate::gemma`] for Gemma 4's native
//! declaration/call/response tokens. `GemmaProtocol`'s own doc comment below
//! covers the parts specific to *building* the prompt (turn structure,
//! thinking activation) that `crate::gemma` has no reason to know.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{ChatMessage, ChatRole, ToolDefinition};

// ============================================================================
// Trait
// ============================================================================

pub trait PromptRenderer {
    /// Render a message history into a raw prompt string for the model.
    fn format_prompt(&self, messages: &[ChatMessage]) -> String;

    /// Render a message history with tool definitions into a raw prompt string.
    /// Default: delegates to `format_prompt` (ignores tools).
    fn format_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> String {
        self.format_prompt(messages)
    }
}

/// Every system message's content, in order, joined by a blank line — `None`
/// when there are none.
///
/// gallium sends **several** system messages in a real turn: a profile's agent
/// preamble, the operator's own prompt, the project's `AGENTS.md` / `CLAUDE.md`,
/// the skill catalog. Every renderer in this file used to take the first one
/// with `find_map` and drop the rest without a word, so a candle turn was
/// missing its project context and its skills while the llama.cpp turn had them
/// — one conversation, two different system prompts, decided by the engine.
///
/// The llama.cpp path has merged them since #184
/// (`llm_local::merge_system_messages`), for a template that admits only one.
/// This is the same merge with the same separator, so the two engines say the
/// same thing.
///
/// A blank line rather than a delimiter, for #184's reason: the seams are what
/// the separate messages were for, and inventing a marker would put a token in
/// the prompt that no model was trained on.
fn system_content(messages: &[ChatMessage]) -> Option<String> {
    let joined = messages
        .iter()
        .filter(|m| m.role == ChatRole::System)
        .map(|m| m.content.trim())
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!joined.is_empty()).then_some(joined)
}

// ============================================================================
// HarmonyProtocol — GPT-OSS
// ============================================================================

/// Harmony protocol adapter for GPT-OSS.
///
/// `effort_text` is the literal that follows `Reasoning: ` in the system
/// message — Harmony's own reasoning-effort control, driven by
/// `crate::profile::GptOss::reasoning_params` at load (see
/// `llm_candle.rs::Arch::renderer`). Defaults to `"medium"`, matching what
/// this was hardcoded to before `reasoningEffort` reached the candle
/// backend.
pub struct HarmonyProtocol {
    effort_text: &'static str,
}

impl HarmonyProtocol {
    pub fn new() -> Self {
        Self {
            effort_text: "medium",
        }
    }

    pub fn with_effort(effort_text: &'static str) -> Self {
        Self { effort_text }
    }

    /// Build the canonical Harmony system content, merging in the optional
    /// caller-provided system message and tool namespace.
    fn build_system_content(
        date: &str,
        effort_text: &str,
        extra: Option<&str>,
        tool_ns: Option<&str>,
    ) -> String {
        let mut s = format!(
            "You are ChatGPT, a large language model trained by OpenAI.\n\
             Knowledge cutoff: 2024-06\n\
             Current date: {date}\n\
             \n\
             Reasoning: {effort_text}\n\
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

impl Default for HarmonyProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptRenderer for HarmonyProtocol {
    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let date = current_date_ymd();
        let extra = system_content(messages);
        let extra = extra.as_deref();
        let system = Self::build_system_content(&date, self.effort_text, extra, None);
        let mut s = format!("<|start|>system<|message|>{system}<|end|>");
        Self::append_messages(&mut s, messages);
        s.push_str("<|start|>assistant\n");
        s
    }

    fn format_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> String {
        let date = current_date_ymd();
        let extra = system_content(messages);
        let extra = extra.as_deref();
        let ns = if tools.is_empty() {
            None
        } else {
            Some(tools_to_harmony_namespace(tools))
        };
        let system = Self::build_system_content(&date, self.effort_text, extra, ns.as_deref());
        let mut s = format!("<|start|>system<|message|>{system}<|end|>");
        Self::append_messages(&mut s, messages);
        s.push_str("<|start|>assistant\n");
        s
    }
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
        if let Some(props) = tool
            .parameters
            .get("properties")
            .and_then(|p| p.as_object())
        {
            let required: Vec<&str> = tool
                .parameters
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            for (name, schema) in props {
                if let Some(desc) = schema.get("description").and_then(|d| d.as_str()) {
                    s.push_str(&format!("  // {}\n", desc));
                }
                let opt = if required.contains(&name.as_str()) {
                    ""
                } else {
                    "?"
                };
                s.push_str(&format!(
                    "  {}{}: {},\n",
                    name,
                    opt,
                    json_schema_to_ts(schema)
                ));
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
/// Adds `<|think|>` to the system turn; `crate::profile::Gemma4::clean_reply`
/// strips `<|channel>thought...<channel|>`.
/// Gemma's beginning-of-sequence token, which its prompts open with.
///
/// Written into the prompt rather than left to the tokenizer, because this
/// tokenizer will not add it: `tokenizer.json`'s post-processor is a
/// `TemplateProcessing` whose `special_tokens` is empty, so `encode(text, true)`
/// adds nothing. Google's own documented prompt begins with it
/// (<https://ai.google.dev/gemma/docs/capabilities/thinking>), and llama.cpp
/// gets it from the chat template embedded in the GGUF — this is the native
/// candle backend catching up with both.
///
/// How much it matters depends on the prompt. With the chat template below, a
/// Gemma 4 E4B answers correctly either way; given a bare completion prompt it
/// degenerates into echoing its own input without one. Since the template is
/// what this builds, the honest description is spec conformance rather than a
/// bug fix.
const BOS: &str = "<bos>";

pub struct GemmaProtocol {
    pub thinking: bool,
}

impl GemmaProtocol {
    pub fn new() -> Self {
        Self { thinking: false }
    }

    pub fn with_thinking() -> Self {
        Self { thinking: true }
    }
}

impl Default for GemmaProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptRenderer for GemmaProtocol {
    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut s = String::from(BOS);
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
    fn format_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> String {
        let thinking_tag = if self.thinking { "<|think|>\n" } else { "" };

        // Build system turn: thinking tag + optional user system message + tool declarations.
        let system = system_content(messages);
        let system_content = system.as_deref();

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

        let mut s = format!("{BOS}<|turn>system\n{system_body}<turn|>\n");

        // Render messages, pairing (Assistant tool_calls) + (Tool results) in one model turn.
        // `in_model_turn` tracks whether the previous emission left the model turn open
        // (tool_call followed by inline tool_response leaves it open per the Gemma 4
        // chat template — the next tool_call or text continues the same turn).
        let mut in_model_turn = false;
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role {
                ChatRole::System => {
                    i += 1;
                } // already in system turn above
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
                            s.push_str(&format!(
                                "<|tool_call>call:{}{args_str}<tool_call|>",
                                call.name
                            ));
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
                        let body = crate::gemma::strip_thinking_blocks(&msg.content);
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

    if let Some(props) = tool
        .parameters
        .get("properties")
        .and_then(|p| p.as_object())
    {
        let required: Vec<&str> = tool
            .parameters
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Sort properties alphabetically (Gemma 4 template uses dictsort).
        let mut sorted_props: Vec<(&String, &serde_json::Value)> = props.iter().collect();
        sorted_props.sort_by_key(|(k, _)| k.as_str());

        s.push_str(",parameters:{properties:{");
        for (i, (name, schema)) in sorted_props.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
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
            if i > 0 {
                s.push(',');
            }
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
            if i > 0 {
                s.push(',');
            }
            s.push_str(key);
            s.push(':');
            match val {
                serde_json::Value::String(v) => s.push_str(&gemini_str_value(v)),
                serde_json::Value::Number(n) => s.push_str(&n.to_string()),
                serde_json::Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
                serde_json::Value::Null => s.push_str("null"),
                other => s.push_str(&gemini_str_value(&other.to_string())),
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

// ============================================================================
// QwenProtocol — Qwen 3.5 (ChatML)
// ============================================================================

/// Qwen 3.5 ChatML adapter (`<|im_start|>role`).
///
/// Uses the official Qwen3.5 chat template tool-calling format (XML-parameter style):
///
/// ## System turn (with tools)
///
/// ```text
/// <|im_start|>system
/// # Tools
///
/// You have access to the following functions:
///
/// <tools>
/// {"description": "...", "name": "write", "parameters": {...}}
/// </tools>
///
/// If you choose to call a function ONLY reply in the following format with NO suffix:
///
/// <tool_call>
/// <function=example_function_name>
/// <parameter=example_parameter_1>
/// value_1
/// </parameter>
/// </function>
/// </tool_call>
///
/// <IMPORTANT>
/// Reminder:
/// - Function calls MUST follow the specified format
/// - Required parameters MUST be specified
/// </IMPORTANT>
/// <|im_end|>
/// ```
///
/// ## Generation prefix (non-thinking mode)
///
/// ```text
/// <|im_start|>assistant
/// <think>
///
/// </think>
///
/// ```
///
/// ## Tool call (model output, stops at `</tool_call>`)
///
/// ```text
/// <tool_call>
/// <function=write>
/// <parameter=file_path>
/// hello.go
/// </parameter>
/// <parameter=content>
/// package main...
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// ## Tool result (injected as user message)
///
/// ```text
/// <|im_start|>user
/// <tool_response>
/// RESULT
/// </tool_response>
/// <|im_end|>
/// ```
/// `thinking` gates whether the generation prefix opens an unclosed
/// `<think>\n` (the model reasons) or a pre-closed `<think>\n\n</think>\n\n`
/// (the model skips straight to its answer) — driven by
/// `crate::profile::Qwen3::reasoning_params` at load (see
/// `llm_candle.rs::Arch::renderer`). Defaults to `true`, matching Qwen 3.6's
/// own real chat template (`enable_thinking` is on unless explicitly set
/// `false`). Before `reasoningEffort` reached the candle backend this was
/// hardcoded and inconsistent: off in [`PromptRenderer::format_prompt`],
/// on in [`PromptRenderer::format_prompt_with_tools`] — an accident of
/// which path happened to be written first, not a deliberate choice, which
/// is why unifying it here changes observed behavior for a no-tools turn
/// with `reasoningEffort` left unset.
pub struct QwenProtocol {
    thinking: bool,
    /// The `reasoning_effort` this turn asked for, as
    /// `crate::profile::ReasoningParams::effort_text` carries it. `None` is the
    /// template's own default (`xhigh`), or thinking being off entirely.
    effort_text: Option<&'static str>,
}

impl QwenProtocol {
    /// Matches the template's own default.
    pub fn new() -> Self {
        Self {
            thinking: true,
            effort_text: None,
        }
    }

    pub fn without_thinking() -> Self {
        Self {
            thinking: false,
            effort_text: None,
        }
    }

    /// Both axes this family has, from one
    /// `crate::profile::ReasoningParams`.
    ///
    /// Two, not one: `Qwen3::reasoning_params` sets `thinking` *and*
    /// `effort_text`, and a renderer that read only the first would make the
    /// same `reasoningEffort` mean different things on the two backends —
    /// llama.cpp's template reads both, so `Medium` through `Max` would be
    /// four distinct prompts there and one prompt here.
    pub fn with_reasoning(thinking: bool, effort_text: Option<&'static str>) -> Self {
        Self {
            thinking,
            effort_text,
        }
    }

    /// The system-prompt line this family's chat template emits for an effort
    /// level, transcribed from `Qwen/Qwen3.8-27B/chat_template.jinja` (pinned
    /// as a fixture in `tests/fixtures/chat_templates/qwen3.8.jinja`).
    ///
    /// This is the whole mechanism: `reasoning_effort` is not a token or a
    /// sampler setting, it is a sentence the template puts at the top of the
    /// system turn. Rendering it here is what makes the candle path agree with
    /// the llama.cpp one, the same way this renderer already hand-transcribes
    /// that template's `# Tools` block.
    ///
    /// `medium` deliberately produces nothing — the template sets
    /// `reasoning_instructions` to `''` and only assigns it in the `xhigh` and
    /// `low` branches, so "medium" is the *absence* of an instruction rather
    /// than an instruction to be moderate. Inventing a sentence for it would be
    /// text no Qwen was trained on.
    ///
    /// Thinking off produces nothing either, matching the template's own guard:
    /// the instructions are computed only `if enable_thinking is undefined or
    /// enable_thinking is true`.
    fn reasoning_instructions(&self) -> Option<&'static str> {
        if !self.thinking {
            return None;
        }
        match self.effort_text {
            // Nothing configured means nothing added. The llama.cpp path does
            // differ here — the real template defaults the variable to `xhigh`
            // and emits that instruction — but closing *that* gap would mean
            // writing the sentence into every unconfigured turn, including for
            // Qwen 3.6, whose own template has no `reasoning_effort` variable
            // at all and would never produce it. `Arch::Qwen35` cannot tell the
            // two generations apart. So an explicit level agrees across
            // backends, which is what a setting has to do, and an absent one
            // leaves each prompt builder at its own default.
            None => None,
            Some("xhigh") => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, \
                 validate key assumptions, consider plausible alternatives, and prioritize \
                 correctness, consistency, and clarity in the final answer.",
            ),
            Some("low") => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
                 directly to the conclusion without unnecessary elaboration.",
            ),
            _ => None,
        }
    }
}

impl Default for QwenProtocol {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip the thinking block from Qwen 3 output.
///
/// The `<think>` special token (ID 248068) decodes to `""` (empty string), so
/// the raw output may start directly with `</think>` or with thinking content
/// followed by `</think>`. `rfind` finds the last close and discards everything
/// before it, handling both cases uniformly.
fn strip_qwen_thinking(s: &str) -> &str {
    if let Some(pos) = s.rfind("</think>") {
        s[pos + "</think>".len()..].trim_start()
    } else {
        s.trim_start()
    }
}

/// Serialize a tool definition to JSON matching the Qwen3 chat template format.
///
/// The official Jinja2 template does `tool | tojson` where `tool` is the full
/// OpenAI wrapper `{"type":"function","function":{...}}`. We must match that exactly.
fn qwen_tool_json(tool: &ToolDefinition) -> String {
    // Build JSON manually to match Python's json.dumps insertion order:
    // {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    // serde_json::Map uses BTreeMap internally (alphabetical), so we can't rely on it for order.
    let params = sort_json_keys(&tool.parameters);
    let params_str = serde_json::to_string(&params).unwrap_or_default();
    let name_json = serde_json::to_string(&tool.name).unwrap_or_default();
    let desc_json = serde_json::to_string(&tool.description).unwrap_or_default();
    let compact = format!(
        r#"{{"type":"function","function":{{"name":{},"description":{},"parameters":{}}}}}"#,
        name_json, desc_json, params_str
    );
    python_style_json(&compact)
}

/// Convert compact JSON to Python json.dumps style: add space after ':' and ','.
///
/// Python's json.dumps default uses `separators=(', ', ': ')`.
/// We replicate this by inserting a space after every `:` and `,` that
/// appear at the structural level (not inside string values).
fn python_style_json(compact: &str) -> String {
    let mut out = String::with_capacity(compact.len() + compact.len() / 4);
    let chars: Vec<char> = compact.chars().collect();
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if escaped {
            escaped = false;
            out.push(c);
        } else if c == '\\' && in_string {
            escaped = true;
            out.push(c);
        } else if c == '"' {
            in_string = !in_string;
            out.push(c);
        } else if !in_string && (c == ':' || c == ',') {
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// Recursively sort JSON object keys alphabetically (matches Python's json.dumps / Jinja2 tojson).
fn sort_json_keys(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let sorted: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (k.clone(), sort_json_keys(v)))
                .collect();
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(sort_json_keys).collect())
        }
        other => other.clone(),
    }
}

/// Format a replayed Qwen3.5 tool call in the XML-parameter format matching the model's training data.
fn qwen_tool_call_block(name: &str, args: &serde_json::Value) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    if let Some(obj) = args.as_object() {
        for (param_name, value) in obj {
            let value_str = match value {
                serde_json::Value::String(vs) => vs.clone(),
                _ => value.to_string(),
            };
            s.push_str(&format!(
                "<parameter={param_name}>\n{value_str}\n</parameter>\n"
            ));
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

impl PromptRenderer for QwenProtocol {
    fn format_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut s = String::new();
        // The template emits the reasoning instructions in the system turn on
        // this path too — folded into the existing one, or as a system turn of
        // their own when the conversation has none.
        let instructions = self.reasoning_instructions();
        if let Some(instructions) = instructions {
            if !messages.iter().any(|m| m.role == ChatRole::System) {
                s.push_str(&format!("<|im_start|>system\n{instructions}<|im_end|>\n"));
            }
        }
        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    let content = match instructions {
                        Some(i) => format!("{i}\n\n{}", msg.content),
                        None => msg.content.clone(),
                    };
                    s.push_str(&format!("<|im_start|>system\n{content}<|im_end|>\n"));
                }
                ChatRole::User | ChatRole::Tool => {
                    s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::Assistant => {
                    if msg.tool_calls.is_none() && !msg.content.is_empty() {
                        let body = strip_qwen_thinking(&msg.content);
                        if !body.is_empty() {
                            s.push_str(&format!(
                                "<|im_start|>assistant\n{}<|im_end|>\n",
                                body.trim()
                            ));
                        }
                    }
                }
            }
        }
        s.push_str(if self.thinking {
            "<|im_start|>assistant\n<think>\n"
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        });
        s
    }

    fn format_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> String {
        let system = system_content(messages);
        let system_content = system.as_deref();

        let mut system_body = String::new();

        if !tools.is_empty() {
            system_body
                .push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in tools {
                system_body.push('\n');
                system_body.push_str(&qwen_tool_json(tool));
            }
            system_body.push_str(concat!(
                "\n</tools>",
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n",
                "\n<tool_call>",
                "\n<function=example_function_name>",
                "\n<parameter=example_parameter_1>",
                "\nvalue_1",
                "\n</parameter>",
                "\n<parameter=example_parameter_2>",
                "\nThis is the value for the second parameter\nthat can span\nmultiple lines",
                "\n</parameter>",
                "\n</function>",
                "\n</tool_call>",
                "\n\n<IMPORTANT>",
                "\nReminder:",
                "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags",
                "\n- Required parameters MUST be specified",
                "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after",
                "\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
                "\n</IMPORTANT>",
            ));
        }
        if let Some(sc) = system_content {
            if !system_body.is_empty() {
                system_body.push_str("\n\n");
            }
            system_body.push_str(sc.trim());
        }

        // The template puts `reasoning_instructions + '\n\n'` at the very top of
        // the system turn, before `# Tools`. Same position here.
        if let Some(instructions) = self.reasoning_instructions() {
            let rest = std::mem::take(&mut system_body);
            system_body.push_str(instructions);
            if !rest.trim().is_empty() {
                system_body.push_str("\n\n");
                system_body.push_str(&rest);
            }
        }

        let mut s = format!("<|im_start|>system\n{}<|im_end|>\n", system_body.trim());

        for msg in messages {
            match msg.role {
                ChatRole::System => {}
                ChatRole::User => {
                    s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::Tool => {
                    // Tool results are wrapped in <tool_response> inside a user turn.
                    s.push_str(&format!(
                        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                        msg.content
                    ));
                }
                ChatRole::Assistant => {
                    if let Some(ref calls) = msg.tool_calls {
                        // Replay previous tool calls in the official <function=...> format.
                        let mut call_s = String::new();
                        for call in calls {
                            call_s.push_str(&qwen_tool_call_block(&call.name, &call.arguments));
                        }
                        s.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", call_s));
                    } else if !msg.content.is_empty() {
                        let body = strip_qwen_thinking(&msg.content);
                        if !body.is_empty() {
                            s.push_str(&format!(
                                "<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n",
                                body.trim()
                            ));
                        }
                    }
                }
            }
        }

        s.push_str(if self.thinking {
            "<|im_start|>assistant\n<think>\n"
        } else {
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        });
        s
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
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u32;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    format!("{year:04}-{month:02}-{:02}", days + 1)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ============================================================================
// LFM2.5 (Liquid) — ChatML template + `[func(arg=val)]` tool calls
// ============================================================================

/// Protocol for LFM2.5 (`lfm2moe`). ChatML turns like Qwen, but tools are listed
/// in the system prompt and tool calls are emitted as
/// `<|tool_call_start|>[func_name(arg=value, ...)]<|tool_call_end|>`. The model
/// is a reasoning model: it emits a `<think>…</think>` block before the answer.
pub struct Lfm2Protocol;

impl PromptRenderer for Lfm2Protocol {
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
                        let body = strip_lfm2_think(&msg.content);
                        if !body.is_empty() {
                            s.push_str(&format!(
                                "<|im_start|>assistant\n{}<|im_end|>\n",
                                body.trim()
                            ));
                        }
                    }
                }
            }
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }

    fn format_prompt_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> String {
        let system = system_content(messages);
        let system_content = system.as_deref();

        let mut system_body = String::new();
        if let Some(sc) = system_content {
            system_body.push_str(sc.trim());
        }
        if !tools.is_empty() {
            if !system_body.is_empty() {
                system_body.push('\n');
            }
            system_body.push_str("List of tools: [");
            for (i, tool) in tools.iter().enumerate() {
                if i > 0 {
                    system_body.push_str(", ");
                }
                system_body.push_str(&lfm2_tool_json(tool));
            }
            system_body.push(']');
        }

        let mut s = String::new();
        if !system_body.is_empty() {
            s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system_body));
        }

        for msg in messages {
            match msg.role {
                ChatRole::System => {}
                ChatRole::User => {
                    s.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::Tool => {
                    // Tool results come back in a `tool` turn.
                    s.push_str(&format!("<|im_start|>tool\n{}<|im_end|>\n", msg.content));
                }
                ChatRole::Assistant => {
                    if let Some(ref calls) = msg.tool_calls {
                        let mut call_s = String::from("<|tool_call_start|>[");
                        for (i, call) in calls.iter().enumerate() {
                            if i > 0 {
                                call_s.push_str(", ");
                            }
                            call_s.push_str(&lfm2_render_call(&call.name, &call.arguments));
                        }
                        call_s.push_str("]<|tool_call_end|>");
                        s.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", call_s));
                    } else if !msg.content.is_empty() {
                        let body = strip_lfm2_think(&msg.content);
                        if !body.is_empty() {
                            s.push_str(&format!(
                                "<|im_start|>assistant\n{}<|im_end|>\n",
                                body.trim()
                            ));
                        }
                    }
                }
            }
        }

        s.push_str("<|im_start|>assistant\n");
        s
    }
}

/// Strip a leading/embedded `<think>…</think>` reasoning block.
fn strip_lfm2_think(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => rest = &after[end + "</think>".len()..],
            None => {
                // Unclosed — drop the rest (model didn't finish thinking).
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Render a tool as the JSON object LFM2 lists in its system prompt.
fn lfm2_tool_json(tool: &ToolDefinition) -> String {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
    .to_string()
}

/// Render one call as `name(arg=value, ...)` for assistant-history replay,
/// matching the template's `format_arg_value` (strings single-quoted, mappings
/// as JSON, everything else stringified).
fn lfm2_render_call(name: &str, args: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    if let Some(map) = args.as_object() {
        for (k, v) in map {
            let rendered = match v {
                serde_json::Value::String(s) => format!("'{}'", lfm2_escape(s)),
                serde_json::Value::Object(_) | serde_json::Value::Array(_) => v.to_string(),
                _ => v.to_string(),
            };
            parts.push(format!("{k}={rendered}"));
        }
    }
    format!("{name}({})", parts.join(", "))
}

/// Escape a string for the single-quoted literal above — the inverse of
/// [`crate::profile::wire::python`]'s unescaping, which is what reads these back.
///
/// Unescaped, an argument containing a `'` closed its literal early and
/// everything after it became structure — the replay of the model's own previous
/// call arrived malformed. Source code carries apostrophes and backslashes
/// routinely, so that is the common case, not the edge one. Newlines are escaped
/// too: a call is one line in this format.
///
/// Note this **deviates from the model's own template**, whose `format_arg_value`
/// concatenates `'` + value + `'` and escapes nothing. The deviation is the
/// lesser of the two: an unescapable literal cannot be read back by anything,
/// including the model.
fn lfm2_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(
            prompt.contains("<|tool>declaration:write{"),
            "expected tool declaration"
        );
        assert!(prompt.contains("<tool|>"), "expected tool declaration end");
        assert!(prompt.contains("file_path"), "expected file_path param");
        // Properties sorted alphabetically: content before file_path
        let content_pos = prompt.find("content:{").unwrap_or(usize::MAX);
        let file_pos = prompt.find("file_path:{").unwrap_or(usize::MAX);
        assert!(
            content_pos < file_pos,
            "properties should be sorted: content before file_path"
        );
        // Model turn opener at the end
        assert!(
            prompt.ends_with("<|turn>model\n"),
            "expected model turn opener at end, got: {:?}",
            &prompt[prompt.len().saturating_sub(60)..]
        );
    }

    /// Both Gemma prompt builders open with `<bos>`, once.
    ///
    /// The tokenizer will not supply it — its post-processor declares no special
    /// tokens — so if this is not in the string the model never sees one. Google
    /// documents the prompt as starting with it.
    #[test]
    fn gemma_prompts_begin_with_exactly_one_bos() {
        use crate::llm::ChatMessage;
        let proto = GemmaProtocol::new();
        let msgs = vec![ChatMessage::user("Hello.".to_string())];

        for (what, prompt) in [
            ("format_prompt", proto.format_prompt(&msgs)),
            (
                "format_prompt_with_tools",
                proto.format_prompt_with_tools(&msgs, &[]),
            ),
        ] {
            assert!(
                prompt.starts_with("<bos>"),
                "{what} lost its <bos>: {prompt:?}"
            );
            assert_eq!(
                prompt.matches("<bos>").count(),
                1,
                "{what} repeated <bos>: {prompt:?}"
            );
            // The turn structure still follows immediately — `<bos>` is a prefix
            // to the dialogue, not a turn of its own.
            assert!(
                prompt[5..].starts_with("<|turn>"),
                "{what} put something between <bos> and the first turn: {prompt:?}"
            );
        }
    }

    /// `<bos>` is Gemma's alone.
    ///
    /// It reached Qwen and LFM2 in the first cut of this change — one careless
    /// find-and-replace across three `format_prompt`s that open identically —
    /// and nothing failed, because no test asserted what a ChatML prompt starts
    /// with. `<bos>` is not in their vocabularies, so it would not even have
    /// been a wrong special token: it tokenizes as literal text at the head of
    /// every prompt.
    #[test]
    fn only_gemma_prompts_carry_a_bos() {
        use crate::llm::ChatMessage;
        let msgs = vec![ChatMessage::user("Hello.".to_string())];

        let qwen = QwenProtocol::new().format_prompt(&msgs);
        assert!(
            !qwen.contains("<bos>"),
            "Qwen is ChatML and has no <bos>: {qwen:?}"
        );
        assert!(
            qwen.starts_with("<|im_start|>"),
            "Qwen prompts open with a ChatML turn: {qwen:?}"
        );

        let lfm2 = Lfm2Protocol.format_prompt(&msgs);
        assert!(
            !lfm2.contains("<bos>"),
            "LFM2 is ChatML and has no <bos>: {lfm2:?}"
        );
        assert!(
            lfm2.starts_with("<|im_start|>"),
            "LFM2 prompts open with a ChatML turn: {lfm2:?}"
        );
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
                media: vec![],
                reasoning: None,
            },
            ChatMessage {
                role: crate::llm::ChatRole::Tool,
                content: "ok".to_string(),
                tool_calls: None,
                tool_call_id: Some("c1".to_string()),
                tool_name: Some("write".to_string()),
                media: vec![],
                reasoning: None,
            },
        ];
        let prompt = proto.format_prompt_with_tools(&msgs, &tools);
        assert!(
            prompt.contains("<|tool_call>call:write{"),
            "expected tool call replay"
        );
        assert!(
            prompt.contains("<|tool_response>response:write{"),
            "expected tool response"
        );
        // Tool response should be inline (no <|turn>user between call and response)
        let call_pos = prompt.find("<|tool_call>call:write{").unwrap();
        let resp_pos = prompt.find("<|tool_response>response:write{").unwrap();
        let user_after_call = prompt[call_pos..].find("<|turn>user");
        assert!(
            user_after_call.is_none() || user_after_call.unwrap() > resp_pos - call_pos,
            "tool response should come before any <|turn>user after the call"
        );
    }

    // --- HarmonyProtocol reasoning effort ---

    #[test]
    fn harmony_defaults_to_medium_and_with_effort_overrides_it() {
        use crate::llm::ChatMessage;
        let msgs = vec![ChatMessage::user("Hi.".to_string())];

        let default = HarmonyProtocol::new().format_prompt(&msgs);
        assert!(default.contains("Reasoning: medium\n"), "{default:?}");

        let high = HarmonyProtocol::with_effort("high").format_prompt(&msgs);
        assert!(high.contains("Reasoning: high\n"), "{high:?}");
    }

    // --- QwenProtocol thinking toggle ---

    #[test]
    fn qwen_thinking_controls_both_prompt_paths_uniformly() {
        use crate::llm::ChatMessage;
        let msgs = vec![ChatMessage::user("Hi.".to_string())];
        let tools: Vec<ToolDefinition> = vec![];

        let on_no_tools = QwenProtocol::new().format_prompt(&msgs);
        assert!(
            on_no_tools.ends_with("<|im_start|>assistant\n<think>\n"),
            "{on_no_tools:?}"
        );
        let on_with_tools = QwenProtocol::new().format_prompt_with_tools(&msgs, &tools);
        assert!(
            on_with_tools.ends_with("<|im_start|>assistant\n<think>\n"),
            "{on_with_tools:?}"
        );

        let off_no_tools = QwenProtocol::without_thinking().format_prompt(&msgs);
        assert!(
            off_no_tools.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{off_no_tools:?}"
        );
        let off_with_tools =
            QwenProtocol::without_thinking().format_prompt_with_tools(&msgs, &tools);
        assert!(
            off_with_tools.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "{off_with_tools:?}"
        );
    }

    /// The render above and [`crate::profile::wire::python`] are two halves of
    /// one round trip: the model's previous call is replayed to it in this
    /// format, and read back out of its next reply by that parser. A payload
    /// carrying the format's own delimiters is where an unescaped render used to
    /// hand the model a malformed transcript of its own turn.
    #[test]
    fn an_lfm2_native_call_round_trips_through_the_python_parser() {
        let args = serde_json::json!({
            "file_path": "hello.go",
            "content": "package main\n\nfunc main() {\n\tfmt.Println(\"it's here\")\n}",
        });
        let rendered = lfm2_render_call("Write", &args);
        assert!(
            !rendered.contains('\n'),
            "a call is one line in this format: {rendered}"
        );

        let calls = crate::profile::wire::python::parse_calls(&format!("[{rendered}]"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["content"], args["content"]);
        assert_eq!(calls[0].arguments["file_path"], "hello.go");
    }
}
