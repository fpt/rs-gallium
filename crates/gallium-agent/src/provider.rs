//! GalliumProvider: wraps a local gallium-core CausalLM as an LlmProvider.
//!
//! Prompt formatting and response parsing are delegated to a [`ModelProtocol`]
//! adapter. See [`protocol`] for available protocols:
//!
//! - [`HarmonyProtocol`] — GPT-OSS: full ReAct with tool calling via Harmony format
//! - [`GemmaProtocol`] — Gemma 4: native function-calling + optional thinking
//! - [`QwenProtocol`]   — Qwen 3.5: ChatML template, plain chat
//!
//! ## Generation and decoding
//!
//! `run_generate_ids` runs the model and returns raw token IDs. All paths decode
//! with `skip_special=false` so that `parse_response` and `parse_tool_call` have
//! access to special-token markers (e.g. `<channel|>` for Gemma thinking,
//! `<|channel|>final` for Harmony channels).
//!
//! ## Tool calling
//!
//! When `protocol.supports_tools()` is true, `chat_with_tools()`:
//!
//! 1. Formats the prompt via `protocol.format_prompt_with_tools()`.
//! 2. Runs generation; `protocol.tool_stop_tokens()` are added to the EOS set so
//!    generation stops as soon as the model signals a tool call.
//! 3. Decodes with skip_special=false and calls `protocol.parse_tool_call()`.
//!    If a tool call is detected, returns `LlmResponse::ToolCalls`.
//! 4. Otherwise extracts the response text via `protocol.parse_response()`.

use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use gallium_core::{generate, CausalLM, SamplingParams};
use tokenizers::Tokenizer;

use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo, ToolDefinition};
use crate::protocol::ModelProtocol;

pub struct GalliumProvider {
    model: RefCell<Box<dyn CausalLM>>,
    tokenizer: Tokenizer,
    params: SamplingParams,
    /// EOS token IDs (includes <|end|>, </s>, <|call|>, model-specific terminators).
    eos_tokens: Vec<u32>,
    max_new_tokens: usize,
    protocol: Box<dyn ModelProtocol>,
}

// GalliumProvider is used only from single-threaded binary context (REPL) or
// under a Mutex (UniFFI). The RefCell is never accessed from multiple threads concurrently.
unsafe impl Send for GalliumProvider {}
unsafe impl Sync for GalliumProvider {}

impl GalliumProvider {
    pub fn new(
        model: Box<dyn CausalLM>,
        tokenizer: Tokenizer,
        params: SamplingParams,
        max_new_tokens: usize,
        protocol: Box<dyn ModelProtocol>,
    ) -> Self {
        let tool_stops = protocol.tool_stop_tokens();
        // Use get_vocab(true) — includes both the base BPE vocabulary AND added tokens.
        // get_added_vocabulary().get_vocab() misses tokens like <|im_end|> that appear
        // in the base BPE vocab for some models (e.g. Qwen3.5) rather than the added layer.
        let eos_tokens: Vec<u32> = tokenizer
            .get_vocab(true)
            .into_iter()
            .filter(|(k, _)| {
                // NOTE: do NOT match the bare "<|end|>" token — in Harmony it's a
                // message/channel separator (analysis → commentary → final), not a
                // turn terminator. The turn ends on "<|return|>" or "<|call|>".
                k.contains("eos")
                    || k == "<|endoftext|>"
                    || k.contains("</s>")
                    || k.contains("<end_of_turn>")
                    || k.contains("<|im_end|>")
                    || k == "<|call|>"              // Harmony tool call terminator
                    || k == "<|return|>"            // Harmony end-of-turn terminator
                    || tool_stops.contains(&k.as_str()) // protocol-specific tool stops
            })
            .map(|(_, v)| v)
            .collect();

        tracing::info!(
            "GalliumProvider: {} EOS tokens, max_new_tokens={}",
            eos_tokens.len(),
            max_new_tokens
        );

        Self {
            model: RefCell::new(model),
            tokenizer,
            params,
            eos_tokens,
            max_new_tokens,
            protocol,
        }
    }

    /// Encode `prompt`, run generation, return the raw generated token IDs.
    fn run_generate_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("GalliumProvider: prompt_tokens={}", prompt_tokens.len());

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut model = self.model.borrow_mut();
        generate(
            model.as_mut(),
            &prompt_tokens,
            &self.params,
            self.max_new_tokens,
            &self.eos_tokens,
            |id| generated_ids.push(id),
        )
        .map_err(|e| anyhow::anyhow!("generate error: {e}"))?;

        tracing::info!("GalliumProvider: generated {} tokens", generated_ids.len());
        Ok(generated_ids)
    }

    /// Convenience: generate and decode with skip_special=false (for parse_response / parse_tool_call).
    fn run_generate(&self, prompt: &str) -> Result<String> {
        let ids = self.run_generate_ids(prompt)?;
        let raw = self
            .tokenizer
            .decode(&ids, false)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;
        tracing::debug!("GalliumProvider raw output: {:?}", raw);
        Ok(raw)
    }
}

impl LlmProvider for GalliumProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.protocol.format_prompt(messages);
        tracing::debug!("GalliumProvider prompt ({} chars)", prompt.len());
        let raw = self.run_generate(&prompt)?;
        Ok(self.protocol.parse_response(&raw))
    }

    fn supports_tools(&self) -> bool {
        self.protocol.supports_tools()
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let prompt = self.protocol.format_prompt_with_tools(messages, tools);
        tracing::debug!("GalliumProvider tool prompt ({} chars)", prompt.len());
        // Decode with skip_special=false so parse_tool_call can see all markers.
        let raw = self.run_generate(&prompt)?;

        if let Some((func_name, args)) = self.protocol.parse_tool_call(&raw) {
            tracing::info!("GalliumProvider: tool call '{}'", func_name);
            let call_id = format!(
                "call_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos()
            );
            return Ok(LlmResponse::ToolCalls(
                vec![ToolCallInfo { id: call_id, name: func_name, arguments: args }],
                None,
            ));
        }

        // No tool call — extract response text.
        Ok(LlmResponse::Text {
            content: self.protocol.parse_response(&raw),
            reasoning: None,
            usage: None,
        })
    }
}
