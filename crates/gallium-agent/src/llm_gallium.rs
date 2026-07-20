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
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use gallium_core::{generate, CausalLM, SamplingParams};
use tokenizers::Tokenizer;

use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo, ToolDefinition};
use crate::protocol::{GemmaProtocol, HarmonyProtocol, Lfm2Protocol, ModelProtocol, QwenProtocol};

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

// ============================================================================
// Loader — build a GalliumProvider from a plain model path
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
                if env_flag("KESSEL_GALLIUM_THINKING") {
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

/// Build a [`GalliumProvider`] from a plain model path — the same `hf:ORG/REPO…`
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
/// Env knobs: `KESSEL_GALLIUM_TOKENIZER_REPO` (tokenizer.json source repo),
/// `KESSEL_GALLIUM_DTYPE` (`f16`/`bf16`/`f32`, safetensors only, default `f16`),
/// `KESSEL_GALLIUM_THINKING` (Gemma 4 thinking channel).
pub fn load_gallium_provider(
    model_path: &str,
    temperature: Option<f32>,
    max_tokens: u32,
) -> Result<GalliumProvider> {
    use candle_core::{DType, Device};

    let device = Device::Cpu;
    let params = SamplingParams {
        temperature: temperature.unwrap_or(0.7),
        ..Default::default()
    };
    let tok_repo = std::env::var("KESSEL_GALLIUM_TOKENIZER_REPO").ok();

    let (arch, model, tokenizer): (Arch, Box<dyn CausalLM>, Tokenizer) =
        match Format::detect(model_path) {
            Format::Gguf => {
                // Same hf:/local resolution as the llama.cpp backend.
                let gguf = crate::model_downloader::ensure_model(model_path)
                    .map_err(|e| anyhow::anyhow!("failed to resolve '{model_path}': {e}"))?;
                tracing::info!("Loading GGUF gallium model from {:?}", gguf);
                let (metadata, vb) = gallium_core::load_gguf(&gguf, &device)?;

                let hint = metadata.get_str("general.architecture").unwrap_or_default();
                let arch = Arch::from_hint(&hint).ok_or_else(|| {
                    anyhow::anyhow!(
                        "could not detect gallium arch from GGUF general.architecture '{hint}' \
                         (supported: qwen35, gemma4, gpt-oss)"
                    )
                })?;

                let tokenizer = resolve_gguf_tokenizer(&gguf, model_path, tok_repo.as_deref())?;
                let model: Box<dyn CausalLM> = match arch {
                    Arch::GptOss => Box::new(gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device)?),
                    Arch::Qwen35 => Box::new(gallium_models::qwen35_q::Qwen35Q::load(&metadata, &vb, &device)?),
                    Arch::Gemma4 => Box::new(gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device)?),
                    Arch::Lfm2 => Box::new(gallium_models::lfm2moe_q::Lfm2MoeQ::load(&metadata, &vb, &device)?),
                };
                (arch, model, tokenizer)
            }
            Format::Safetensors => {
                let dir = resolve_safetensors_dir(model_path, tok_repo.as_deref())?;
                tracing::info!("Loading safetensors gallium model from {:?}", dir);

                let config_path = dir.join("config.json");
                let full: serde_json::Value = gallium_models::loader::load_config(&config_path)?;
                let arch = detect_safetensors_arch(&full).ok_or_else(|| {
                    anyhow::anyhow!(
                        "could not detect gallium arch from {:?} \
                         (supported: qwen35, gemma4, gpt-oss)",
                        config_path
                    )
                })?;

                let dtype = match std::env::var("KESSEL_GALLIUM_DTYPE")
                    .unwrap_or_else(|_| "f16".to_string())
                    .as_str()
                {
                    "f32" => DType::F32,
                    "f16" => DType::F16,
                    "bf16" => DType::BF16,
                    other => anyhow::bail!("unsupported KESSEL_GALLIUM_DTYPE '{other}'"),
                };
                let shards: Vec<PathBuf> = std::fs::read_dir(&dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map(|ext| ext == "safetensors").unwrap_or(false))
                    .collect();
                if shards.is_empty() {
                    anyhow::bail!("no .safetensors files in {:?}", dir);
                }
                let vb = gallium_models::loader::load_safetensors(&shards, dtype, &device)?;
                let tokenizer = load_tokenizer(&dir.join("tokenizer.json"))?;
                // GPT-OSS parses the whole config; Qwen/Gemma nest theirs under
                // `text_config` (multimodal configs) and fall back to the root.
                let text = full.get("text_config").unwrap_or(&full);
                let model: Box<dyn CausalLM> = match arch {
                    Arch::GptOss => {
                        let cfg: gallium_models::gpt_oss::GptOssConfig =
                            serde_json::from_value(full.clone())
                                .map_err(|e| anyhow::anyhow!("GptOss config error: {e}"))?;
                        Box::new(gallium_models::gpt_oss::GptOss::load(&cfg, vb, &shards, &device)?)
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
                (arch, model, tokenizer)
            }
        };

    tracing::info!("Gallium model loaded (arch: {:?}).", arch);
    Ok(GalliumProvider::new(
        model,
        tokenizer,
        params,
        max_tokens as usize,
        arch.protocol(),
    ))
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {:?}: {e}", path))
}

/// Find a `tokenizer.json` for a GGUF: prefer one sitting beside the file (the
/// shared downloader can place it there), otherwise fetch it from HuggingFace —
/// an explicit `KESSEL_GALLIUM_TOKENIZER_REPO`, else the GGUF's own model repo.
fn resolve_gguf_tokenizer(gguf: &Path, model_path: &str, tok_repo: Option<&str>) -> Result<Tokenizer> {
    let beside = gguf
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.json");
    if beside.exists() {
        return load_tokenizer(&beside);
    }

    let repo = tok_repo.map(String::from).or_else(|| hf_repo_of(model_path));
    let repo = repo.ok_or_else(|| {
        anyhow::anyhow!(
            "tokenizer.json not found beside {:?}; set KESSEL_GALLIUM_TOKENIZER_REPO \
             to its HuggingFace repo",
            gguf
        )
    })?;
    use hf_hub::api::sync::Api;
    tracing::info!("Fetching tokenizer.json from HuggingFace: {repo}");
    let local = Api::new()?.model(repo).get("tokenizer.json")?;
    load_tokenizer(&local)
}

/// Resolve a safetensors model path to a local directory of shards, downloading
/// the repo from HuggingFace for an `hf:` spec.
fn resolve_safetensors_dir(model_path: &str, tok_repo: Option<&str>) -> Result<PathBuf> {
    if let Some(hf) = hf_spec(model_path) {
        return download_safetensors_repo(hf, tok_repo);
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
fn download_safetensors_repo(hf: &str, tok_repo: Option<&str>) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;

    let repo_id = hf.trim_end_matches('/');
    tracing::info!("Fetching safetensors repo from HuggingFace: {repo_id}");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let info = repo.info()?;
    let shards: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    if shards.is_empty() {
        anyhow::bail!("no .safetensors files found in {repo_id}");
    }
    let config_local = repo.get("config.json")?;
    api.model(tok_repo.unwrap_or(repo_id).to_string())
        .get("tokenizer.json")?;
    for shard in &shards {
        repo.get(shard)?;
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
        assert_eq!(detect_safetensors_arch(&json!({"model_type": "phi3"})), None);
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
}
