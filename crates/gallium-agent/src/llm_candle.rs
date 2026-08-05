//! CandleProvider: wraps a local gallium-core CausalLM as an LlmProvider.
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
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use gallium_core::{generate, CausalLM, SamplingParams};
use tokenizers::Tokenizer;

use crate::cancel::CancellationToken;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo, ToolDefinition};
use crate::protocol::{GemmaProtocol, HarmonyProtocol, Lfm2Protocol, ModelProtocol, QwenProtocol};

pub struct CandleProvider {
    model: RefCell<Box<dyn CausalLM>>,
    tokenizer: Tokenizer,
    params: SamplingParams,
    /// EOS token IDs (includes <|end|>, </s>, <|call|>, model-specific terminators).
    eos_tokens: Vec<u32>,
    max_new_tokens: usize,
    protocol: Box<dyn ModelProtocol>,
    /// From the model's own metadata — GGUF's `<arch>.context_length` or
    /// safetensors' `max_position_embeddings`. `None` when the file is silent.
    context_window: Option<u32>,
}

// CandleProvider is used only from single-threaded binary context (REPL) or
// under a Mutex (app-server). The RefCell is never accessed from multiple threads concurrently.
unsafe impl Send for CandleProvider {}
unsafe impl Sync for CandleProvider {}

impl CandleProvider {
    pub fn new(
        model: Box<dyn CausalLM>,
        tokenizer: Tokenizer,
        params: SamplingParams,
        max_new_tokens: usize,
        protocol: Box<dyn ModelProtocol>,
        context_window: Option<u32>,
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
            "CandleProvider: {} EOS tokens, max_new_tokens={}",
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
            context_window,
        }
    }

    /// Encode `prompt`, run generation, return the raw generated token IDs and
    /// the size the prompt encoded to.
    ///
    /// That second number is returned rather than recomputed because a caller
    /// cannot recover it: the prompt string is gone by then, and re-encoding it
    /// would be a guess at what this tokenizer did.
    ///
    /// `cancel` is checked between sampled tokens — the only interruption point
    /// a decode loop has. A cancelled generation returns the tokens it managed,
    /// and the caller turns that into `AgentError::Cancelled` rather than
    /// handing a truncated reply to the model.
    fn run_generate_ids(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
    ) -> Result<(Vec<u32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("CandleProvider: prompt_tokens={}", prompt_tokens.len());

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut model = self.model.borrow_mut();
        // Timed in two halves because they scale differently and a single
        // average hides which one a change actually moved: prefill is one
        // forward over the whole prompt, decode is one forward per token.
        let started = Instant::now();
        let mut first_token_at: Option<Instant> = None;
        // Per-token, not just the total: a single stall (a buffer pool growing, a
        // page-in) averages out to a plausible-looking rate over a short run and
        // would send you optimising the steady state instead of the stall.
        let mut token_times: Vec<f64> = Vec::new();
        let mut last_token_at = started;
        generate(
            model.as_mut(),
            &prompt_tokens,
            &self.params,
            self.max_new_tokens,
            &self.eos_tokens,
            |id| {
                let now = Instant::now();
                if first_token_at.is_none() {
                    first_token_at = Some(now);
                } else {
                    token_times.push(now.duration_since(last_token_at).as_secs_f64());
                }
                last_token_at = now;
                generated_ids.push(id);
                if cancel.is_cancelled() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .map_err(|e| anyhow::anyhow!("generate error: {e}"))?;

        report_generation_rate(
            model.device(),
            prompt_tokens.len(),
            generated_ids.len(),
            started,
            first_token_at,
            &mut token_times,
        );
        Ok((generated_ids, prompt_tokens.len()))
    }

    /// Convenience: generate and decode with skip_special=false (for parse_response / parse_tool_call).
    ///
    /// Also reports what it cost. The counts are exact rather than estimated —
    /// they are the tokens this tokenizer produced and this loop sampled — which
    /// is what makes them usable as a context gauge rather than as a hint.
    fn run_generate(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
    ) -> Result<(String, TokenUsage)> {
        let (ids, prompt_tokens) = self.run_generate_ids(prompt, cancel)?;
        // Checked after generating rather than only before: a turn cancelled
        // mid-reply has a partial, usually mid-sentence string, and passing that
        // on as if the model had finished is worse than stopping.
        cancel.check()?;
        let raw = self
            .tokenizer
            .decode(&ids, false)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;
        tracing::debug!("CandleProvider raw output: {:?}", raw);

        let input = prompt_tokens as u64;
        let output = ids.len() as u64;
        Ok((raw, TokenUsage::single(input, output, input + output)))
    }
}

/// Log prefill and decode throughput for one generation.
///
/// The device is named in the line because these numbers only mean something
/// next to another device's, and a run that silently fell back to CPU otherwise
/// looks like a Metal run that gained nothing.
fn report_generation_rate(
    device: &candle_core::Device,
    prompt_tokens: usize,
    generated: usize,
    started: Instant,
    first_token_at: Option<Instant>,
    token_times: &mut [f64],
) {
    let Some(first) = first_token_at else {
        tracing::info!("CandleProvider: generated 0 tokens");
        return;
    };
    // Prefill is the whole prompt in one forward, so its rate is per prompt
    // token; the first sampled token is the point it has finished.
    let prefill = first.duration_since(started).as_secs_f64();
    // The first token is prefill's output, not decode's, hence `generated - 1`
    // over the time after it.
    let decode = first.elapsed().as_secs_f64();
    let decode_tokens = generated.saturating_sub(1);

    let rate = |n: usize, secs: f64| {
        if secs > 0.0 {
            n as f64 / secs
        } else {
            f64::NAN
        }
    };
    tracing::info!(
        "CandleProvider: {} — prefill {prompt_tokens} tok in {prefill:.2}s \
         ({:.1} tok/s), decode {decode_tokens} tok in {decode:.2}s ({:.1} tok/s)",
        gallium_core::device_name(device),
        rate(prompt_tokens, prefill),
        rate(decode_tokens, decode),
    );

    if !token_times.is_empty() {
        token_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = token_times[token_times.len() / 2];
        let slowest = token_times[token_times.len() - 1];
        tracing::info!(
            "CandleProvider: per-token median {:.0} ms ({:.1} tok/s), slowest {:.0} ms",
            median * 1000.0,
            if median > 0.0 { 1.0 / median } else { f64::NAN },
            slowest * 1000.0,
        );
    }
}

impl LlmProvider for CandleProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.protocol.format_prompt(messages);
        tracing::debug!("CandleProvider prompt ({} chars)", prompt.len());
        let (raw, _usage) = self.run_generate(&prompt, &CancellationToken::new())?;
        Ok(self.protocol.parse_response(&raw))
    }

    fn supports_tools(&self) -> bool {
        self.protocol.supports_tools()
    }

    /// The window the loaded model was configured for, from its own metadata.
    fn context_window(&self) -> Option<u32> {
        self.context_window
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        self.chat_with_tools_cancellable(messages, tools, &CancellationToken::new())
    }

    fn chat_with_tools_cancellable(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        cancel: &CancellationToken,
    ) -> Result<LlmResponse> {
        crate::llm::reject_images(messages, "the candle backend")?;
        let prompt = self.protocol.format_prompt_with_tools(messages, tools);
        tracing::debug!("CandleProvider tool prompt ({} chars)", prompt.len());
        // Decode with skip_special=false so parse_tool_call can see all markers.
        let (raw, usage) = self.run_generate(&prompt, cancel)?;

        if let Some((func_name, args)) = self.protocol.parse_tool_call(&raw) {
            tracing::info!("CandleProvider: tool call '{}'", func_name);
            let call_id = format!(
                "call_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos()
            );
            // Usage on this arm too, not only on the text one: a tool-using turn
            // is where the prompt actually grows, so reporting only the final
            // answer would gauge the context at its smallest.
            return Ok(LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: call_id,
                    name: func_name,
                    arguments: args,
                }],
                Some(usage),
            ));
        }

        // No tool call — extract response text.
        Ok(LlmResponse::Text {
            content: self.protocol.parse_response(&raw),
            reasoning: None,
            usage: Some(usage),
        })
    }
}

// ============================================================================
// Loader — build a CandleProvider from a plain model path
// ============================================================================
//
// The model path is exactly the spec the llama.cpp backend accepts (an
// `hf:ORG/REPO/…` spec or a local path) — the engine is chosen out of band via
// `llm.inference_engine` / `INFERENCE_ENGINE`, so the path carries no engine or
// arch marker. Arch and format are auto-detected from the model itself.

/// Which hand-written model implementation to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    GptOss,
    Qwen35,
    Gemma4,
    Lfm2,
}

impl Arch {
    /// Map an architecture hint — GGUF `general.architecture`, or config.json
    /// `model_type` / `architectures[]` — to a supported arch, by substring.
    fn from_hint(hint: &str) -> Option<Self> {
        let h = hint.to_ascii_lowercase();
        if h.contains("gemma") {
            Some(Arch::Gemma4)
        } else if h.contains("qwen") {
            Some(Arch::Qwen35)
        } else if h.contains("gptoss") || h.contains("gpt-oss") || h.contains("gpt_oss") {
            Some(Arch::GptOss)
        } else if h.contains("lfm2") {
            Some(Arch::Lfm2)
        } else {
            None
        }
    }

    fn protocol(self) -> Box<dyn ModelProtocol> {
        match self {
            Arch::GptOss => Box::new(HarmonyProtocol),
            Arch::Qwen35 => Box::new(QwenProtocol),
            Arch::Lfm2 => Box::new(Lfm2Protocol),
            Arch::Gemma4 => {
                // Gemma 4 supports an optional thinking channel.
                if env_flag("GALLIUM_THINKING") {
                    Box::new(GemmaProtocol::with_thinking())
                } else {
                    Box::new(GemmaProtocol::new())
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Safetensors,
    Gguf,
}

impl Format {
    /// Detect the on-disk format from a model path: a `.gguf` suffix (whether an
    /// `hf:` spec or a local file) is GGUF; anything else (a bare `hf:ORG/REPO`
    /// repo or a local directory of shards) is safetensors.
    fn detect(model_path: &str) -> Self {
        if model_path
            .trim_end_matches('/')
            .to_ascii_lowercase()
            .ends_with(".gguf")
        {
            Format::Gguf
        } else {
            Format::Safetensors
        }
    }
}

/// Build a [`CandleProvider`] from a plain model path — the same `hf:ORG/REPO…`
/// or local spec the llama.cpp backend accepts.
///
/// - **GGUF** (`….gguf`): resolved via the shared model downloader
///   ([`crate::model_downloader::ensure_model`]) and arch is read from the GGUF
///   `general.architecture` metadata. The tokenizer comes from a `tokenizer.json`
///   beside the GGUF, else it is fetched from the model's HF repo (llama.cpp uses
///   the GGUF's embedded tokenizer; gallium needs the HF `tokenizer.json`).
/// - **safetensors** (a bare `hf:ORG/REPO` repo or a local directory of shards):
///   the repo is fetched (or the directory used as-is) and arch is read from
///   `config.json`.
///
/// Env knobs: `GALLIUM_TOKENIZER_REPO` (tokenizer.json source repo),
/// `GALLIUM_DTYPE` (`f16`/`bf16`/`f32`, safetensors only, default `f16`),
/// `GALLIUM_THINKING` (Gemma 4 thinking channel).
pub fn load_candle_provider(
    model_path: &str,
    temperature: Option<f32>,
    max_tokens: u32,
    tokenizer_path: Option<&str>,
) -> Result<CandleProvider> {
    use candle_core::DType;

    // Metal on macOS, CUDA where it was built in, else CPU — `GALLIUM_DEVICE`
    // overrides, and naming a device that is not there fails rather than quietly
    // running on the CPU.
    let device = gallium_core::resolve_device(std::env::var("GALLIUM_DEVICE").ok().as_deref())?;
    tracing::info!("Candle device: {}", gallium_core::device_name(&device));
    let params = SamplingParams {
        temperature: temperature.unwrap_or(0.7),
        ..Default::default()
    };
    // Already resolved by the caller: the env var wins over the config's
    // `tokenizerPath`, and a config path has been made absolute. Falling back to
    // the env var here keeps a direct caller of this function working.
    let tok_spec = tokenizer_path
        .map(String::from)
        .or_else(|| std::env::var("GALLIUM_TOKENIZER_REPO").ok());

    // The window comes out of the model's own metadata, and is `None` when the
    // file does not say — a gauge shows nothing rather than a guess.
    let (arch, model, tokenizer, context_window): (
        Arch,
        Box<dyn CausalLM>,
        Tokenizer,
        Option<u32>,
    ) = match Format::detect(model_path) {
        Format::Gguf => {
            // Same hf:/local resolution as the llama.cpp backend.
            let gguf = crate::model_downloader::ensure_model(model_path)
                .map_err(|e| anyhow::anyhow!("failed to resolve '{model_path}': {e}"))?;
            tracing::info!("Loading GGUF candle model from {:?}", gguf);
            let (metadata, vb) = gallium_core::load_gguf(&gguf, &device)?;

            let hint = metadata.get_str("general.architecture").unwrap_or_default();
            let arch = Arch::from_hint(&hint).ok_or_else(|| {
                anyhow::anyhow!(
                    "could not detect gallium arch from GGUF general.architecture '{hint}' \
                         (supported: qwen35, gemma4, gpt-oss)"
                )
            })?;

            // GGUF names this per architecture (`qwen3.context_length`,
            // `gemma3.context_length`, …), keyed by the same string that
            // chose the arch above.
            let window = metadata.get_u32(&format!("{hint}.context_length")).ok();

            let tokenizer = resolve_gguf_tokenizer(&gguf, model_path, tok_spec.as_deref())?;
            let model: Box<dyn CausalLM> = match arch {
                Arch::GptOss => Box::new(gallium_models::gpt_oss_q::GptOssQ::load(
                    &metadata, &vb, &device,
                )?),
                Arch::Qwen35 => Box::new(gallium_models::qwen35_q::Qwen35Q::load(
                    &metadata, &vb, &device,
                )?),
                Arch::Gemma4 => Box::new(gallium_models::gemma4_q::Gemma4Q::load(
                    &metadata, &vb, &device,
                )?),
                Arch::Lfm2 => Box::new(gallium_models::lfm2moe_q::Lfm2MoeQ::load(
                    &metadata, &vb, &device,
                )?),
            };
            (arch, model, tokenizer, window)
        }
        Format::Safetensors => {
            let dir = resolve_safetensors_dir(model_path, tok_spec.as_deref())?;
            tracing::info!("Loading safetensors candle model from {:?}", dir);

            let config_path = dir.join("config.json");
            let full: serde_json::Value = gallium_models::loader::load_config(&config_path)?;
            let arch = detect_safetensors_arch(&full).ok_or_else(|| {
                anyhow::anyhow!(
                    "could not detect gallium arch from {:?} \
                         (supported: qwen35, gemma4, gpt-oss)",
                    config_path
                )
            })?;

            let dtype = match std::env::var("GALLIUM_DTYPE")
                .unwrap_or_else(|_| "f16".to_string())
                .as_str()
            {
                "f32" => DType::F32,
                "f16" => DType::F16,
                "bf16" => DType::BF16,
                other => anyhow::bail!("unsupported GALLIUM_DTYPE '{other}'"),
            };
            let shards: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
                .collect();
            if shards.is_empty() {
                anyhow::bail!("no .safetensors files in {:?}", dir);
            }
            let vb = gallium_models::loader::load_safetensors(&shards, dtype, &device)?;
            let tokenizer = resolve_safetensors_tokenizer(&dir, tok_spec.as_deref())?;
            // GPT-OSS parses the whole config; Qwen/Gemma nest theirs under
            // `text_config` (multimodal configs) and fall back to the root.
            let text = full.get("text_config").unwrap_or(&full);
            // HuggingFace's name for the same fact. Read from `text` for the
            // same reason the model config is: on a multimodal config the
            // root describes the wrapper, not the language model.
            let window = text
                .get("max_position_embeddings")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| u32::try_from(n).ok());
            let model: Box<dyn CausalLM> = match arch {
                Arch::GptOss => {
                    let cfg: gallium_models::gpt_oss::GptOssConfig =
                        serde_json::from_value(full.clone())
                            .map_err(|e| anyhow::anyhow!("GptOss config error: {e}"))?;
                    Box::new(gallium_models::gpt_oss::GptOss::load(
                        &cfg, vb, &shards, &device,
                    )?)
                }
                Arch::Qwen35 => {
                    let cfg: gallium_models::qwen35::Qwen35Config =
                        serde_json::from_value(text.clone())
                            .map_err(|e| anyhow::anyhow!("Qwen35 config error: {e}"))?;
                    Box::new(gallium_models::qwen35::Qwen35::load(&cfg, vb, &device)?)
                }
                Arch::Gemma4 => {
                    let cfg: gallium_models::gemma4::Gemma4Config =
                        serde_json::from_value(text.clone())
                            .map_err(|e| anyhow::anyhow!("Gemma4 config error: {e}"))?;
                    Box::new(gallium_models::gemma4::Gemma4::load(&cfg, vb, &device)?)
                }
                Arch::Lfm2 => anyhow::bail!(
                    "LFM2 is only supported as GGUF for now; use an `hf:…/….gguf` model path"
                ),
            };
            (arch, model, tokenizer, window)
        }
    };

    tracing::info!(
        "Candle model loaded (arch: {:?}, context window: {}).",
        arch,
        context_window
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    Ok(CandleProvider::new(
        model,
        tokenizer,
        params,
        max_tokens as usize,
        arch.protocol(),
        context_window,
    ))
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {:?}: {e}", path))
}

/// The `tokenizer.json` a spec names on disk, if it names one at all.
///
/// A bare `ORG/REPO` is indistinguishable from a relative path, so existence is
/// what decides — the same rule `config::resolve_tokenizer_path` applies, and
/// applying it again here means a spec that arrived through the env var (which
/// never goes through the config) gets the same treatment.
///
/// A directory is accepted for the obvious reason: people point at the model
/// directory more often than at the file inside it.
fn local_tokenizer_file(spec: &str) -> Option<PathBuf> {
    let path = Path::new(spec);
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    let inside = path.join("tokenizer.json");
    inside.is_file().then_some(inside)
}

/// The tokenizer a `tokenizerPath` / `GALLIUM_TOKENIZER_REPO` spec names: a
/// local file or directory if it is one, otherwise a HuggingFace repo to fetch
/// `tokenizer.json` from.
///
/// Both model formats resolve an explicit spec through here, so "a local path
/// works" is one fact about the setting rather than something the GGUF path
/// happens to support.
fn tokenizer_from_spec(spec: &str) -> Result<Tokenizer> {
    if let Some(local) = local_tokenizer_file(spec) {
        tracing::info!("Loading tokenizer.json from {:?}", local);
        return load_tokenizer(&local);
    }
    tracing::info!("Fetching tokenizer.json from HuggingFace: {spec}");
    let local = crate::model_downloader::ensure_repo_file(spec, "tokenizer.json").map_err(|e| {
        anyhow::anyhow!(
            "tokenizer '{spec}' is neither a path on disk nor a HuggingFace \
             repo with a tokenizer.json: {e}"
        )
    })?;
    load_tokenizer(&local)
}

/// Find a `tokenizer.json` for a GGUF: an explicit spec, else one sitting
/// beside the file (the shared downloader can place it there), else the GGUF's
/// own model repo.
///
/// An explicit spec wins over the one beside the model, which is the answer to
/// "I named a tokenizer and got a different one". It also means the setting
/// behaves the same whichever way the model was obtained.
fn resolve_gguf_tokenizer(
    gguf: &Path,
    model_path: &str,
    tok_spec: Option<&str>,
) -> Result<Tokenizer> {
    if let Some(spec) = tok_spec {
        return tokenizer_from_spec(spec);
    }

    let beside = gguf
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.json");
    if beside.exists() {
        return load_tokenizer(&beside);
    }

    let repo = hf_repo_of(model_path).ok_or_else(|| {
        anyhow::anyhow!(
            "no tokenizer.json for {:?}: none beside the model, and nothing to \
             fetch one from. Set `tokenizerPath` in the config (a local path, \
             or a HuggingFace repo that has one) or GALLIUM_TOKENIZER_REPO",
            gguf
        )
    })?;
    tokenizer_from_spec(&repo)
}

/// Find a `tokenizer.json` for a safetensors model: an explicit spec, else the
/// one in the model's own directory.
fn resolve_safetensors_tokenizer(dir: &Path, tok_spec: Option<&str>) -> Result<Tokenizer> {
    match tok_spec {
        Some(spec) => tokenizer_from_spec(spec),
        None => load_tokenizer(&dir.join("tokenizer.json")),
    }
}

/// Resolve a safetensors model path to a local directory of shards, downloading
/// the repo from HuggingFace for an `hf:` spec.
fn resolve_safetensors_dir(model_path: &str, tok_spec: Option<&str>) -> Result<PathBuf> {
    if let Some(hf) = hf_spec(model_path) {
        return download_safetensors_repo(hf, tok_spec.is_some());
    }
    let dir = PathBuf::from(model_path);
    if dir.is_dir() {
        Ok(dir)
    } else {
        anyhow::bail!("safetensors model path is not a directory: {model_path}");
    }
}

/// Download a full-precision safetensors repo (shards + config.json +
/// tokenizer.json) into the HuggingFace cache and return its directory.
fn download_safetensors_repo(hf: &str, tokenizer_named_elsewhere: bool) -> Result<PathBuf> {
    use crate::model_downloader::{ensure_repo_file, list_repo_files};

    let repo_id = hf.trim_end_matches('/');
    tracing::info!("Fetching safetensors repo from HuggingFace: {repo_id}");
    let shards: Vec<String> = list_repo_files(repo_id)?
        .into_iter()
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files found in {repo_id}");
    }
    let config_local = ensure_repo_file(repo_id, "config.json")?;
    // Only when nothing else names one. This used to fetch `tokenizer.json`
    // from the tokenizer repo and then load it from *this* repo's snapshot
    // directory, which are different places — so the setting quietly did
    // nothing here, or failed on a repo that ships no tokenizer. Choosing the
    // tokenizer is `resolve_safetensors_tokenizer`'s job now.
    if !tokenizer_named_elsewhere {
        ensure_repo_file(repo_id, "tokenizer.json")?;
    }
    for shard in &shards {
        ensure_repo_file(repo_id, shard)?;
    }
    Ok(config_local.parent().unwrap().to_path_buf())
}

/// Detect the arch from a parsed `config.json`: try `model_type` and
/// `architectures[]`, at the root and under a nested `text_config`.
fn detect_safetensors_arch(config: &serde_json::Value) -> Option<Arch> {
    fn hints(v: &serde_json::Value) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(mt) = v.get("model_type").and_then(|x| x.as_str()) {
            out.push(mt.to_string());
        }
        if let Some(arr) = v.get("architectures").and_then(|x| x.as_array()) {
            out.extend(arr.iter().filter_map(|a| a.as_str().map(String::from)));
        }
        out
    }
    let mut all = hints(config);
    if let Some(text) = config.get("text_config") {
        all.extend(hints(text));
    }
    all.iter().find_map(|h| Arch::from_hint(h))
}

/// Strip a leading `hf:` / `hf://` scheme, returning the `ORG/REPO[/…]` body.
fn hf_spec(model_path: &str) -> Option<&str> {
    model_path
        .strip_prefix("hf://")
        .or_else(|| model_path.strip_prefix("hf:"))
}

/// The `ORG/REPO` of an `hf:` spec (dropping any `@revision` and file path).
fn hf_repo_of(model_path: &str) -> Option<String> {
    let rest = hf_spec(model_path)?;
    let mut segs = rest.splitn(3, '/');
    let org = segs.next()?;
    let name = segs.next()?;
    let name = name.split('@').next().unwrap_or(name);
    if org.is_empty() || name.is_empty() {
        return None;
    }
    Some(format!("{org}/{name}"))
}

fn env_flag(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn format_detects_gguf_by_suffix() {
        assert_eq!(
            Format::detect("hf:unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"),
            Format::Gguf
        );
        assert_eq!(Format::detect("/models/x.GGUF"), Format::Gguf);
        assert_eq!(Format::detect("hf:org/repo"), Format::Safetensors);
        assert_eq!(Format::detect("/models/qwen-dir/"), Format::Safetensors);
    }

    #[test]
    fn arch_from_hint_maps_known_names() {
        assert_eq!(Arch::from_hint("qwen3"), Some(Arch::Qwen35));
        assert_eq!(Arch::from_hint("Qwen3MoeForCausalLM"), Some(Arch::Qwen35));
        assert_eq!(Arch::from_hint("gemma3"), Some(Arch::Gemma4));
        assert_eq!(Arch::from_hint("Gemma3ForCausalLM"), Some(Arch::Gemma4));
        assert_eq!(Arch::from_hint("gpt-oss"), Some(Arch::GptOss));
        assert_eq!(Arch::from_hint("gpt_oss"), Some(Arch::GptOss));
        assert_eq!(Arch::from_hint("llama"), None);
    }

    #[test]
    fn detect_arch_from_config_json_shapes() {
        assert_eq!(
            detect_safetensors_arch(&json!({"model_type": "qwen3"})),
            Some(Arch::Qwen35)
        );
        assert_eq!(
            detect_safetensors_arch(&json!({"architectures": ["Gemma3ForCausalLM"]})),
            Some(Arch::Gemma4)
        );
        // Hint nested under text_config (multimodal wrapper).
        assert_eq!(
            detect_safetensors_arch(&json!({"text_config": {"model_type": "gpt_oss"}})),
            Some(Arch::GptOss)
        );
        assert_eq!(
            detect_safetensors_arch(&json!({"model_type": "phi3"})),
            None
        );
    }

    #[test]
    fn hf_repo_of_drops_file_and_revision() {
        assert_eq!(
            hf_repo_of("hf:unsloth/Qwen3.5-9B-GGUF/x.gguf").as_deref(),
            Some("unsloth/Qwen3.5-9B-GGUF")
        );
        assert_eq!(
            hf_repo_of("hf://org/repo@abc123/sub/model.gguf").as_deref(),
            Some("org/repo")
        );
        assert_eq!(hf_repo_of("/local/path.gguf"), None);
    }

    /// The path-or-repo decision both model formats now share. Loading is not
    /// exercised here — a real `tokenizer.json` is megabytes — but choosing
    /// wrong is what sends a local path to the HuggingFace API as a repo id,
    /// which is the bug this consolidation fixed on the safetensors side.
    #[test]
    fn a_tokenizer_spec_naming_a_file_is_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, "{}").unwrap();

        assert_eq!(
            local_tokenizer_file(&file.to_string_lossy()),
            Some(file.clone())
        );
    }

    /// A directory is accepted whole: people point at the model directory more
    /// often than at the file inside it.
    #[test]
    fn a_tokenizer_spec_naming_a_directory_finds_the_file_inside() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, "{}").unwrap();

        assert_eq!(
            local_tokenizer_file(&dir.path().to_string_lossy()),
            Some(file)
        );
    }

    /// A directory without one is not a tokenizer, and must not be reported as
    /// a path that later fails to load.
    #[test]
    fn a_directory_without_a_tokenizer_is_not_a_path() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(local_tokenizer_file(&dir.path().to_string_lossy()), None);
    }

    /// Nothing of that name on disk, so it is a repo id — the shape a bare
    /// `ORG/REPO` has, and what `GALLIUM_TOKENIZER_REPO` has always meant.
    #[test]
    fn a_tokenizer_spec_that_is_not_on_disk_is_a_repo() {
        assert_eq!(local_tokenizer_file("unsloth/gemma-4-E4B-it"), None);
    }
}
