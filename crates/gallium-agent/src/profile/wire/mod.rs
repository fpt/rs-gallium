//! Wire formats: one module per way a model writes a tool call or marks its
//! reasoning.
//!
//! Each `parse_calls` here is a pure function of the model's decoded output (plus,
//! where the format cannot say it itself, the tool schema). None of them knows
//! which model produced the text — deciding *which* formats to run is the
//! profile's job, and that separation is the point: before profiles existed every
//! parser ran against every model's output, and a family's parser misreading
//! another family's text was the bug class that motivated [ADR
//! 0003](../../../../docs/adr/0003-model-profiles.md).
//!
//! Two more parsers of the same kind live at the crate root rather than here:
//! [`crate::harmony`] (GPT-OSS) and [`crate::gemma`] (Gemma 4). They were
//! extracted earlier so `protocol.rs` and `llm_local.rs` could share them, and
//! they stay put while the candle backend still reaches for them directly;
//! [`harmony_calls`] and [`gemma_calls`] below adapt their output to
//! [`ToolCallInfo`].

pub mod dsml;
pub mod json;
pub mod minimax;
pub mod python;
pub mod qwen_xml;
pub mod tags;
pub mod think;

use crate::llm::{ToolCallInfo, ToolDefinition};

/// The formats a model reaches for when following gallium's *prose* tool
/// instructions rather than its own training: the JSON protocol
/// `llm_local::tool_instructions` actually asks for, and the Python-ish call
/// list some instruction-tuned models substitute for it.
///
/// Neither is family-specific, which is why every profile falls back to them
/// after its own native format finds nothing. A model that ignored its training
/// and complied with the prompt is the common case for a heavily quantized
/// model — `configs/deepseek-v4-flash.toml` records a 1-bit quant that dropped
/// its native DSML block on 4 of 4 runs at temperature 0.6 — and refusing to
/// read the reply gallium *asked* for would turn a recoverable miss into a dead
/// turn.
///
/// `text` must already have its reasoning removed; see [`json::parse_calls`].
pub fn fallback_calls(text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
    let calls = json::parse_calls(text, tools);
    if !calls.is_empty() {
        return calls;
    }
    python::parse_calls(text)
}

/// Trim any run of `markers` from the tail of `s`.
///
/// Exists because of an engine difference, not a model one: the candle backend
/// decodes with special tokens kept as text, so a reply arrives ending in
/// `<|im_end|>` / `<turn|>` and would show them to the user, while on llama.cpp
/// the same tokens are EOG and stop generation before reaching the string. So
/// this is load-bearing on one engine and a no-op on the other, and every family
/// whose turn marker is a token needs it.
///
/// Looped rather than single-pass: a reply can end `…<turn|><eos>`.
pub fn strip_trailing_markers<'a>(s: &'a str, markers: &[&str]) -> &'a str {
    let mut s = s.trim();
    loop {
        let prev = s;
        for m in markers {
            s = s.trim_end_matches(m).trim();
        }
        if s == prev {
            return s;
        }
    }
}

/// Give each call in a batch its sequential `call_N` id.
///
/// Applied once, by [`super::ModelProfile::tool_calls`], rather than by each
/// wire parser: the ids number a *reply's* calls, which is a fact about the
/// batch and not about the format it arrived in. Parsers leave `id` empty.
pub fn number_ids(calls: &mut [ToolCallInfo]) {
    for (i, call) in calls.iter_mut().enumerate() {
        call.id = format!("call_{i}");
    }
}

/// GPT-OSS's Harmony calls, as `ToolCallInfo`.
///
/// `to=functions.NAME<|channel|>commentary …<|message|>{…}<|call|>` — parsed by
/// [`crate::harmony`], which both local backends share since both decode with
/// special tokens kept as literal text.
pub fn harmony_calls(text: &str) -> Vec<ToolCallInfo> {
    crate::harmony::parse_tool_calls(text)
        .into_iter()
        .map(|c| ToolCallInfo {
            id: String::new(),
            name: c.name,
            arguments: c.arguments,
        })
        .collect()
}

/// Gemma 4's native calls, as `ToolCallInfo`.
///
/// `<|tool_call>call:NAME{key:<|"|>value<|"|>, key2:123}<tool_call|>`, where
/// `<|"|>` is the model's quote token — parsed by [`crate::gemma`]. Names are
/// kept verbatim: the llama.cpp path is the general-purpose local backend and
/// must not fold mixed-case MCP tool names.
pub fn gemma_calls(text: &str) -> Vec<ToolCallInfo> {
    crate::gemma::parse_native_tool_calls(text)
        .into_iter()
        .map(|c| ToolCallInfo {
            id: String::new(),
            name: c.name,
            arguments: c.arguments,
        })
        .collect()
}
