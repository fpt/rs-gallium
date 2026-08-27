//! In-process llama.cpp provider via FFI (llama-cpp-2 crate).
//!
//! Loads a GGUF model directly — no server needed.
//!
//! Tool calling: llama-cpp-2 0.1.150 removed the OAI-compat chat layer
//! (`apply_chat_template_oaicompat` / `ChatTemplateResult` / `parse_response_oaicompat`),
//! so we implement tool calling ourselves: the available tools are injected into
//! the system prompt with a JSON output protocol, and the model's reply is parsed
//! leniently for a tool-call object. The model's own jinja chat template (from the
//! GGUF) is still used to format the conversation via `apply_chat_template`, which
//! only accepts role+content messages.

use anyhow::Result;
use parking_lot::Mutex;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde_json::Value;

use crate::cancel::CancellationToken;
use crate::llm::{
    fmt_rate, ChatMessage, ChatRole, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo,
    ToolDefinition,
};
use crate::profile::{self, DetectHints, ModelProfile, ReasoningEffort, ReasoningParams};
use crate::AgentError;

/// Template-level tests over the real embedded chat templates, kept in their
/// own file because they are a suite rather than a handful of unit tests. A
/// child module, so they reach this one's private message shaping.
#[cfg(test)]
#[path = "llm_local_templates.rs"]
mod llm_local_templates;

/// The llama.cpp backend is process-global: `LlamaBackend::init()` guards itself
/// with an atomic and returns `BackendAlreadyInitialized` on any later call. So a
/// second provider in one process cannot init its own — it has to share this one.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
/// Serializes the init race; `OnceLock::get_or_init` cannot host a fallible init.
static BACKEND_INIT: Mutex<()> = Mutex::new(());

/// The process-wide llama.cpp backend, initialized on first use.
fn shared_backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let _guard = BACKEND_INIT.lock();
    // Another thread may have won the race while we waited.
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    let mut backend =
        LlamaBackend::init().map_err(|e| anyhow::anyhow!("Failed to init llama backend: {}", e))?;
    backend.void_logs();
    Ok(BACKEND.get_or_init(|| backend))
}

/// One attachment on its way to the projector: the raw file bytes exactly as
/// they arrived, plus enough to name it if it will not decode.
struct MediaAttachment {
    bytes: Vec<u8>,
    /// `"image"` or `"audio"` — the modality the projector is being asked for.
    kind: &'static str,
    /// The declared media type, so a failure says *which* attachment broke.
    label: String,
}

/// How many conversations may keep a warm KV cache at once. `0` switches reuse
/// off entirely.
///
/// Default 1, deliberately: a slot is a *whole KV cache*, sized at the
/// context's `n_ctx`, so the second slot costs as much memory as the first.
/// One is right for the REPL, which has one conversation. An app-server running
/// concurrent threads wants one per conversation it expects to interleave, and
/// pays for them.
fn kv_cache_slots() -> usize {
    std::env::var("GALLIUM_KV_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1)
}

/// How many leading ids two token sequences share.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Evaluate `tokens` into `ctx` starting at position `start`, asking for logits
/// on the last one. Returns the position the context now sits at.
///
/// Shared by the cold and warm paths, which differ only in `start` and in how
/// much of the prompt they hand over.
fn feed(ctx: &mut LlamaContext, tokens: &[LlamaToken], start: i32, n_ctx: u32) -> Result<i32> {
    if tokens.is_empty() {
        anyhow::bail!("nothing to evaluate: an empty prompt produces no logits to sample from");
    }
    if start as u32 + tokens.len() as u32 > n_ctx {
        anyhow::bail!(
            "prompt of {} tokens at position {start} does not fit a context of {n_ctx}",
            tokens.len()
        );
    }
    // Sized for what is actually being fed, not for the whole context: with a
    // warm cache this batch holds a tool result, while `n_ctx` is the entire
    // conversation.
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len().saturating_sub(1) as i32;
    for (offset, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, start + offset, &[0], offset == last)
            .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {}", e))?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("Initial decode failed: {}", e))?;
    Ok(start + batch.n_tokens())
}

/// One retained `LlamaContext` and the exact token sequence its KV cache holds.
///
/// `tokens` is the contract: positions `0..tokens.len()` of the cache hold
/// exactly these ids. Everything about reuse follows from keeping that true —
/// a slot whose record disagrees with its cache produces plausible wrong logits,
/// which nothing downstream can detect.
struct Slot {
    /// SAFETY: this actually borrows [`LlamaLocalProvider::model`]. See the
    /// safety note on that field for why the `'static` is sound and what must
    /// stay true for it to remain so.
    ctx: LlamaContext<'static>,
    /// The ids currently in the cache, prompt *and* generated.
    tokens: Vec<LlamaToken>,
    /// The size this context was built at. A prompt that outgrows it cannot be
    /// served by this slot, and llama.cpp will not grow one in place.
    n_ctx: u32,
    /// Bumped from a counter on every use, so eviction can pick the coldest
    /// slot. A monotonic tick rather than a clock: it only has to order.
    last_used: u64,
}

pub struct LlamaLocalProvider {
    /// Retained contexts, newest-used tracked per slot. **Declared before
    /// `model`** — struct fields drop in declaration order, and every slot
    /// borrows the model, so the model must outlive them.
    ///
    /// `None` when reuse is switched off, which is also what makes the
    /// determinism test possible: the same turn, one path against the other.
    slots: Option<Mutex<SlotPool>>,
    backend: &'static LlamaBackend,
    /// Boxed for a **stable address**: the slots hold pointers into this model,
    /// and moving the provider must not move what they point at.
    model: Box<LlamaModel>,
    /// The model's embedded jinja chat template (rendered via minijinja). None if
    /// the GGUF has no template — then we fall back to a manual ChatML format.
    template_src: Option<String>,
    /// What this model family does on the wire: how it writes a tool call, how it
    /// marks its reasoning, where generation stops. Settled once at load from
    /// what the GGUF reports about itself, and shared with the candle backend —
    /// see `crate::profile` and docs/adr/0003-model-profiles.md.
    profile: &'static dyn ModelProfile,
    /// Literal BOS/EOS token text (e.g. "<bos>", "<eos>"), fed to the template.
    bos: String,
    eos: String,
    temperature: f32,
    /// Nucleus-sampling threshold. `None` skips the top_p stage entirely
    /// rather than running it as a `top_p=1.0` no-op.
    top_p: Option<f32>,
    /// Top-k sampling cutoff. `None` skips the top_k stage entirely rather
    /// than running it as a `top_k=vocab_size` no-op. When `Some`, always
    /// `1..=i32::MAX` — `llm::validated_top_k` is the only constructor of a
    /// `LocalModelOptions::top_k` that reaches here, and it enforces both
    /// bounds before this value is stored, so the `as i32` cast at the
    /// sampler-chain call site cannot wrap.
    top_k: Option<u32>,
    /// `profile.reasoning_params()` for the effort this provider was
    /// configured with, resolved once at load — same shape as
    /// `stop_marker_ids`, since it also needs `profile` (only known once
    /// the GGUF's architecture is detected) before it can be computed.
    /// `ReasoningParams::default()` (both fields `None`) when no effort was
    /// configured at all, which merges nothing into the render context and
    /// leaves the model's own template defaults untouched.
    reasoning: ReasoningParams,
    max_tokens: u32,
    n_ctx: u32,
    /// What the model was trained to hold, straight from the GGUF. The ceiling
    /// worth reporting: unlike `n_ctx`, it does not move.
    n_ctx_train: u32,
    /// llama.cpp's multimodal front end, present only when a projector was
    /// configured. `None` is a text-only provider, which is what every run
    /// without `mmprojPath` gets — and it costs nothing, since the projector is
    /// what holds the vision and audio weights.
    ///
    /// Behind a `Mutex` because `eval_chunks` is documented as not thread-safe,
    /// while a provider is shared across turns. Only multimodal turns contend
    /// for it; text turns never take the lock.
    mtmd: Option<Mutex<MtmdContext>>,
    /// `profile.stop_markers()` resolved to this vocabulary's token ids —
    /// `Some` only when every marker is exactly one token here, which is what
    /// lets `sample_until_done` compare ids instead of scanning decoded text
    /// (ADR 0003 step 5). `None` for a profile with no markers (most of them)
    /// and for one whose markers don't resolve singly on this GGUF; either
    /// way `profile.stops_generation` keeps running as the fallback.
    stop_marker_ids: Option<Vec<LlamaToken>>,
    /// The string mtmd looks for when splitting a prompt around its media —
    /// `<__media__>` by default. Read from the params we built the context with
    /// rather than hardcoded, since it is the projector that has to agree.
    media_marker: String,
    /// What the projector can actually do, asked once at load. A projector with
    /// no audio encoder must refuse a clip rather than encode silence.
    supports_vision: bool,
    supports_audio: bool,
}

/// The retained contexts, and the tick that orders their use.
///
/// Slots are **checked out** for the duration of a generation rather than used
/// under the pool lock. Holding the lock across a decode would serialize
/// concurrent turns, which the app-server runs in parallel today — that would
/// trade a prefill win for a latency regression on the frontend that has the
/// most to gain. A turn that finds every slot busy falls back to a throwaway
/// context and is exactly as fast as it is now.
struct SlotPool {
    idle: Vec<Slot>,
    /// Checked out right now. `idle.len() + busy` is how many exist, which is
    /// what `capacity` bounds.
    busy: usize,
    /// How many slots may exist. Each one is a **whole KV cache**, so this is a
    /// memory knob as much as a concurrency knob — see `GALLIUM_KV_CACHE_SLOTS`.
    capacity: usize,
    tick: u64,
}

/// What the pool decided, handed back so the slow parts happen outside its lock.
enum Reservation {
    /// A warm slot, ready to generate against.
    Ready(Slot),
    /// Capacity is reserved (`busy` already counts it). The caller drops
    /// `victim` if there is one, then builds a context of `n_ctx`. On failure it
    /// must give the capacity back.
    Build {
        n_ctx: u32,
        victim: Option<Slot>,
        tick: u64,
    },
    /// Every slot is checked out and none may be created.
    AllBusy,
}

// LlamaModel is Send+Sync and read-only once loaded; the backend is the shared
// process-global one.
//
// The retained contexts are *not* read-only, which is why they sit behind a
// `Mutex`: a `LlamaContext` is mutated by every decode, and a provider is
// shared across concurrent turns. Before slots existed this impl could argue
// that nothing was mutated per call at all; that argument is gone, and the
// mutex is what replaces it.
unsafe impl Send for LlamaLocalProvider {}
unsafe impl Sync for LlamaLocalProvider {}

/// Slots borrow the model, so they must be gone before it is. Field order
/// already guarantees this (`slots` is declared first), but an explicit drop
/// states the requirement where a future reorder would be reviewed, rather than
/// leaving it as a property of the declaration order that reads like style.
impl Drop for LlamaLocalProvider {
    fn drop(&mut self) {
        if let Some(slots) = &self.slots {
            slots.lock().idle.clear();
        }
    }
}

/// Resolve `profile.stop_markers()` against this model's vocabulary — see
/// [`crate::profile::ModelProfile::stop_markers`] for why an all-or-nothing
/// result is what a caller wants: `Some` only when *every* marker tokenizes
/// to exactly one id, since a caller that got `Some` for two markers and
/// silently dropped a third that didn't resolve would stop early on the two
/// it kept and never learn the profile expected a third.
///
/// `parse_special=true` is `str_to_token`'s hardcoded behavior (not a flag
/// here) — required for a marker like `<tool_call|>` to tokenize to its
/// single added-vocabulary id rather than being split as literal text a
/// model was never trained to treat specially.
fn resolve_stop_markers(
    model: &LlamaModel,
    profile: &'static dyn ModelProfile,
) -> Option<Vec<LlamaToken>> {
    let markers = profile.stop_markers();
    if markers.is_empty() {
        return None;
    }
    let mut ids = Vec::with_capacity(markers.len());
    for marker in markers {
        match model.str_to_token(marker, AddBos::Never) {
            Ok(tokens) if tokens.len() == 1 => ids.push(tokens[0]),
            Ok(tokens) => {
                tracing::warn!(
                    "  Model profile '{}': stop marker {marker:?} tokenizes to {} ids, \
                     not 1 — falling back to stops_generation's decoded-text check for \
                     this model (ADR 0003 step 5)",
                    profile.name(),
                    tokens.len(),
                );
                return None;
            }
            Err(e) => {
                tracing::warn!(
                    "  Model profile '{}': stop marker {marker:?} failed to tokenize ({e}) — \
                     falling back to stops_generation's decoded-text check for this model",
                    profile.name(),
                );
                return None;
            }
        }
    }
    tracing::info!(
        "  Model profile '{}': {} stop marker(s) resolved to token ids, \
         replacing the decoded-text check",
        profile.name(),
        ids.len(),
    );
    Some(ids)
}

/// How to load one GGUF on this machine.
///
/// A struct rather than more parameters on [`LlamaLocalProvider::new`], which had
/// reached seven of them: these are all *settings*, resolved `env > config >
/// default` by the caller before they get here, and a caller passing seven bare
/// values in the right order is one transposition away from a silent
/// misconfiguration. It also gives the explanations below a home that is not a
/// call signature.
pub struct LocalModelOptions<'a> {
    /// The multimodal projector (`mmproj-*.gguf`). `None` is text only.
    pub mmproj_path: Option<&'a str>,
    pub temperature: f32,
    /// See `LlamaLocalProvider::top_p`.
    pub top_p: Option<f32>,
    /// See `LlamaLocalProvider::top_k`.
    pub top_k: Option<u32>,
    /// See `LlamaLocalProvider::reasoning`. `None` means unconfigured, same
    /// as `top_p` — distinct from any particular [`ReasoningEffort`]
    /// variant, which is why this is `Option<ReasoningEffort>` rather than
    /// defaulting to e.g. `Medium`.
    pub reasoning_effort: Option<ReasoningEffort>,
    pub max_tokens: u32,
    /// The floor a context is built at; a longer prompt raises it (see
    /// `context_size_for`), so this is not a ceiling.
    pub n_ctx: u32,
    /// Resolved layers-to-offload (`GALLIUM_GPU_LAYERS` / `[llm] gpuLayers`).
    /// `None` means neither was set, and falls back to llama.cpp's own default.
    pub gpu_layers: Option<u32>,
    /// Move every MoE expert tensor (`ffn_(up|down|gate)_exps`, all layers)
    /// to CPU, keeping attention and the KV cache on the GPU. Mirrors
    /// llama.cpp's `--n-cpu-moe` in spirit (though this is all-or-nothing,
    /// not layer-graduated — see the comment at the use site). Cuts VRAM
    /// pressure sharply for a sparse MoE, since the expert tensors are most
    /// of the file but only a few are read per token; the CPU-side cost is
    /// paid only for the experts actually routed to, same as GPU-side.
    pub cpu_moe: bool,
    /// Which model profile reads this model's output (`GALLIUM_PROFILE` /
    /// `[llm] profile`). `None` means detect it from what the GGUF reports.
    pub profile: Option<&'a str>,
}

impl LlamaLocalProvider {
    pub fn new(model_path: &str, opts: LocalModelOptions<'_>) -> Result<Self> {
        let LocalModelOptions {
            mmproj_path,
            temperature,
            top_p,
            top_k,
            reasoning_effort,
            max_tokens,
            n_ctx,
            gpu_layers,
            cpu_moe,
            profile,
        } = opts;

        tracing::info!("Initializing local llama.cpp provider (FFI)");
        tracing::info!("  Model path: {}", model_path);
        tracing::info!("  Context size: {}", n_ctx);

        let backend = shared_backend()?;

        // On iOS simulator, Metal doesn't support residency sets — use CPU only.
        // Elsewhere, offload layers to the GPU backend (Metal/CUDA/Vulkan,
        // depending on the build features). On a CPU-only build these layers are
        // simply ignored by llama.cpp.
        let use_gpu = if cfg!(target_os = "ios") && cfg!(target_abi = "sim") {
            tracing::info!("  iOS simulator detected — using CPU only (no GPU)");
            false
        } else {
            true
        };

        if !use_gpu {
            // Prevent Metal residency set assertions on simulator
            unsafe {
                std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
            }
        }

        // Layers to offload to the GPU. `gpu_layers` is already resolved by
        // the caller (`GALLIUM_GPU_LAYERS` / `[llm] gpuLayers`, env winning);
        // fit a smaller VRAM budget by setting it lower (e.g. a 6 GB card
        // can't hold a 5 GB model plus KV cache, so partial offload like 20
        // avoids an OOM). Default 999 = offload everything.
        let gpu_layers: u32 = if use_gpu {
            gpu_layers.unwrap_or(999)
        } else {
            0
        };
        tracing::info!("  GPU layers to offload: {}", gpu_layers);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
        // `add_cpu_moe_override` stores a pointer into the params' own regex
        // buffer, so the struct must be pinned before the call and never moved
        // after — hence building the by-value params above first, then
        // pinning, per llama-cpp-2's own doc example for this family of
        // methods (`append_kv_override`).
        let mut model_params = std::pin::pin!(model_params);
        if cpu_moe {
            tracing::info!("  MoE experts: CPU (attention + KV cache stay on GPU)");
            model_params.as_mut().add_cpu_moe_override();
        }
        let model_params: &LlamaModelParams = &model_params;

        let model = Box::new(
            // llama.cpp returns a null pointer on load failure with no typed
            // reason attached, so `e` alone rarely says why. Print the setting
            // that shaped this attempt alongside it rather than guess a cause.
            LlamaModel::load_from_file(backend, Path::new(model_path), model_params).map_err(
                |e| {
                    anyhow::anyhow!(
                        "Failed to load model: {e} (gpu_layers={gpu_layers}, cpu_moe={cpu_moe})"
                    )
                },
            )?,
        );

        tracing::info!("  Model loaded: {} params", model.n_params());
        let n_ctx_train = model.n_ctx_train();
        tracing::info!("  Context train: {}", n_ctx_train);

        let template_src = match model.chat_template(None).and_then(|t| Ok(t.to_string()?)) {
            Ok(src) => Some(strip_unsupported_jinja(&src)),
            Err(_) => {
                tracing::warn!("No chat template in model, using ChatML fallback");
                None
            }
        };

        // Which model family's wire rules to read this model's output by. Both
        // hints come from the file itself: `general.architecture` is what
        // llama.cpp's own loader dispatches on, and the template is what a
        // family's native tool format is spelled out in. A configured name
        // overrides both, and one that does not exist fails the load here rather
        // than quietly running the permissive fallback.
        let arch = model.meta_val_str("general.architecture").ok();
        let profile = profile::resolve(
            profile,
            &DetectHints {
                arch: arch.as_deref(),
                chat_template: template_src.as_deref(),
                model_id: Some(model_path),
            },
        )?;
        let stop_marker_ids = resolve_stop_markers(&model, profile);

        // Can't be computed any earlier than this: it needs `profile`, which
        // itself needs the GGUF's own metadata (just resolved above).
        // `ReasoningParams::default()` (both fields `None`) when unconfigured,
        // which merges nothing into the render context and leaves the
        // model's own template default untouched — the same behavior as
        // before this field existed.
        let reasoning = reasoning_effort
            .map(|effort| profile.reasoning_params(effort))
            .unwrap_or_default();
        tracing::info!(
            "  Reasoning effort: {:?} -> {:?}",
            reasoning_effort,
            reasoning
        );

        // Literal BOS/EOS strings (e.g. "<bos>"/"<eos>") so the jinja template's
        // `{{ bos_token }}`/`{{ eos_token }}` render to real special tokens.
        let mut dec = encoding_rs::UTF_8.new_decoder();
        let bos = model
            .token_to_piece(model.token_bos(), &mut dec, true, None)
            .unwrap_or_default();
        let mut dec = encoding_rs::UTF_8.new_decoder();
        let eos = model
            .token_to_piece(model.token_eos(), &mut dec, true, None)
            .unwrap_or_default();

        // The projector, when one was named. Loaded here rather than lazily so a
        // bad path or a projector built for another model is a startup failure
        // with the file in the message, not a surprise on the first turn that
        // attaches an image.
        let (mtmd, media_marker, supports_vision, supports_audio) = match mmproj_path {
            None => (None, String::new(), false, false),
            Some(path) => {
                tracing::info!("  Multimodal projector: {}", path);
                let params = MtmdContextParams {
                    // The projector rides along with the model, so it follows
                    // the same decision the model's layers did.
                    use_gpu: gpu_layers > 0,
                    ..MtmdContextParams::default()
                };
                let marker = params.media_marker.to_string_lossy().into_owned();
                let ctx = MtmdContext::init_from_file(path, &model, &params)
                    .map_err(|e| anyhow::anyhow!("Failed to load mmproj '{}': {}", path, e))?;
                let vision = ctx.support_vision();
                let audio = ctx.support_audio();
                // Said out loud, because "the model ignored my image" and "this
                // projector has no vision encoder" look identical from outside.
                tracing::info!("  Projector supports: vision={}, audio={}", vision, audio);
                if let Some(rate) = ctx.get_audio_sample_rate() {
                    tracing::info!("  Projector audio sample rate: {} Hz", rate);
                }
                (Some(Mutex::new(ctx)), marker, vision, audio)
            }
        };

        // Reuse is on unless switched off. The switch exists mainly so the two
        // paths can be compared on the same turn — a prefix bug shows up as a
        // different token stream, not as a subtly worse answer — but it is also
        // the escape hatch if a model ever disagrees with the cache.
        let slots = match kv_cache_slots() {
            0 => {
                tracing::info!("  KV cache reuse: off");
                None
            }
            capacity => {
                tracing::info!(
                    "  KV cache reuse: on, {} slot(s) of up to {} tokens",
                    capacity,
                    n_ctx
                );
                Some(Mutex::new(SlotPool {
                    idle: Vec::new(),
                    busy: 0,
                    capacity,
                    tick: 0,
                }))
            }
        };

        Ok(Self {
            slots,
            backend,
            model,
            template_src,
            profile,
            bos,
            eos,
            temperature,
            top_p,
            top_k,
            reasoning,
            max_tokens,
            n_ctx,
            n_ctx_train,
            mtmd,
            media_marker,
            supports_vision,
            supports_audio,
            stop_marker_ids,
        })
    }

    /// Render the conversation through the model's chat template, injecting tool
    /// definitions into the system message when tools are available.
    fn build_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<String> {
        // Some templates format tools natively — Gemma 4's `<|tool>declaration:…`
        // / `<|tool_call>` / `<|tool_response>`, MiniMax-M2.7's
        // `<minimax:tool_call><invoke name="...">`. Feed those templates
        // structured tool inputs so the model sees tools in the exact form it
        // was trained on. Fall back to the generic JSON-prose protocol if it
        // doesn't render.
        if let Some(tools) = tools {
            if self.template_supports_native_tools() {
                match self.render_native(messages, tools) {
                    Ok(prompt) => return Ok(prompt),
                    Err(e) => tracing::warn!(
                        "native tool template render failed ({e}); using JSON-prose fallback"
                    ),
                }
            }
        }

        // (role, content) pairs using only system/user/assistant roles, which
        // every chat template supports (a "tool" role is not universal).
        let mut pairs: Vec<(&'static str, String)> =
            messages.iter().map(Self::render_message).collect();

        if let Some(tools) = tools {
            let instr = Self::tool_instructions(tools);
            if let Some(sys) = pairs.iter_mut().find(|(role, _)| *role == "system") {
                sys.1 = format!("{}\n\n{}", sys.1, instr);
            } else {
                pairs.insert(0, ("system", instr));
            }
        }

        // No embedded template: go straight to the manual ChatML fallback.
        if self.template_src.is_none() {
            return Ok(self.chatml_fallback(&Self::fold_system(pairs)));
        }

        // Render the model's own jinja template. If it rejects the system role
        // (e.g. gemma calls raise_exception), fold system into the first user
        // turn and retry; if it still fails, fall back to manual ChatML.
        match self.render_template(&pairs) {
            Ok(prompt) => return Ok(prompt),
            Err(e) => tracing::debug!("template render failed ({e}); folding system and retrying"),
        }
        let folded = Self::fold_system(pairs);
        match self.render_template(&folded) {
            Ok(prompt) => Ok(prompt),
            Err(e) => {
                tracing::warn!(
                    "template render failed after system-fold ({e}); using ChatML fallback"
                );
                Ok(self.chatml_fallback(&folded))
            }
        }
    }

    /// Build a minijinja environment with the model's chat template registered.
    ///
    /// The environment itself is [`chat_env`], a free function, so a test can
    /// build the *real* one over a fixture template instead of a lookalike —
    /// see `llm_local_templates.rs`. This method is only the part that needs a
    /// loaded model: finding the template it carries.
    fn jinja_env(&self) -> std::result::Result<minijinja::Environment<'static>, minijinja::Error> {
        let src = self.template_src.as_deref().ok_or_else(|| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "no chat template")
        })?;
        chat_env(src)
    }
}

/// A minijinja environment with `src` registered as the template `chat`, plus
/// the functions and filters real GGUF chat templates call.
///
/// Free rather than a method because everything here is about the *template*,
/// not about a loaded model, and because the tests that matter most are the
/// ones that run the same code a model would (`llm_local_templates.rs`). A
/// second environment built to look like this one would pass while this one
/// failed, which is exactly the bug class the harness exists to catch.
pub(crate) fn chat_env(
    src: &str,
) -> std::result::Result<minijinja::Environment<'static>, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    // Support Python-ish str methods (.strip/.startswith/.split/.get/...).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // Templates call raise_exception(...) to reject unsupported inputs.
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // Some newer templates call strftime_now(fmt); a stub is sufficient here.
    env.add_function(
        "strftime_now",
        |_fmt: String| -> std::result::Result<String, minijinja::Error> { Ok(String::new()) },
    );
    // Shadows minijinja's own `tojson` (the `json` cargo feature): that one
    // calls `args.assert_all_used()` and rejects any keyword it doesn't
    // recognize, so MiniMax-M2.7's template — which calls
    // `tojson(ensure_ascii=False)`, a Python/Jinja2 kwarg minijinja's
    // reimplementation never modeled — failed on every single call with
    // "unknown keyword argument 'ensure_ascii'", silently falling back to
    // the JSON-prose tool-call protocol (issue #105's actual live-model
    // verification caught this after landing the parser for it). Nothing
    // here needs to consume `ensure_ascii` — serde_json's output has no
    // ASCII-only mode to opt in or out of — so this version just never
    // asserts every kwarg was used, tolerating any Jinja2-ism a template
    // passes.
    env.add_filter("tojson", lenient_tojson);
    env.add_template_owned("chat", strip_generation_markers(src))?;
    Ok(env)
}

/// Turn `{% generation %}` / `{% endgeneration %}` into empty comments,
/// preserving each tag's whitespace control.
///
/// These are transformers' assistant-masking extension: they mark which spans
/// of a rendered conversation are assistant tokens, so a training run can build
/// a loss mask. They contribute nothing to the rendered text, and minijinja has
/// no statement by that name — so a template carrying them does not merely
/// render differently, it **fails to parse**, and `chat_env` is where the
/// template is registered, so that failure takes out every path that needs it:
/// `render_native`, `render_template`, and the system-folded retry. The model
/// then never sees its own format at all (issue #182 — LFM2.5 had been running
/// on the manual ChatML fallback since the day its config landed).
///
/// A comment rather than deletion because the whitespace control is not
/// decoration: `{%- generation -%}` strips the newline and indentation on both
/// sides of itself, and `{#- -#}` strips exactly the same ones. Deleting the
/// tag would leave that whitespace in the prompt, which is a different prompt.
///
/// Teaching minijinja the statement instead would buy nothing: nothing here
/// consumes an assistant mask, and what gallium wants is the rendered text —
/// which is identical either way.
fn strip_generation_markers(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;

    while let Some(open) = rest.find("{%") {
        let after_open = &rest[open + 2..];
        let Some(close) = after_open.find("%}") else {
            break;
        };
        let inner = &after_open[..close];

        // `- generation -` → `generation`. Dashes first: they are the
        // whitespace control, and they sit outside any spacing.
        if matches!(
            inner.trim_matches('-').trim(),
            "generation" | "endgeneration"
        ) {
            out.push_str(&rest[..open]);
            out.push_str(if inner.starts_with('-') { "{#-" } else { "{#" });
            out.push_str(if inner.ends_with('-') { "-#}" } else { "#}" });
        } else {
            out.push_str(&rest[..open + 2 + close + 2]);
        }
        rest = &after_open[close + 2..];
    }

    out.push_str(rest);
    out
}

/// Replacement for minijinja's own `tojson` filter (see `jinja_env`'s doc
/// comment on why: it rejects any keyword argument it doesn't recognize,
/// which broke on MiniMax-M2.7's `tojson(ensure_ascii=False)`). A free
/// function, not a closure, so a test can register it on a bare
/// `minijinja::Environment` without needing a whole `LlamaLocalProvider`.
fn lenient_tojson(
    value: minijinja::Value,
    indent: Option<minijinja::Value>,
    kwargs: minijinja::value::Kwargs,
) -> std::result::Result<String, minijinja::Error> {
    let indent = indent.or_else(|| kwargs.get("indent").ok());
    let pretty = indent
        .map(|v| bool::try_from(v).unwrap_or(false))
        .unwrap_or(false);
    let result = if pretty {
        serde_json::to_string_pretty(&value)
    } else {
        serde_json::to_string(&value)
    };
    result.map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
}

/// Render `messages` and `tools` through an already-built [`chat_env`].
///
/// Split out of [`LlamaLocalProvider::render_native`] so the template harness
/// (`llm_local_templates.rs`) renders through the same code a real turn does —
/// the same context keys, in the same shapes, from the same
/// [`LlamaLocalProvider::render_message_native`]. A test that rebuilt this
/// context by hand would keep passing after this one changed.
///
/// `add_generation_prompt` is a parameter only because the prefix-stability
/// check needs it off: the trailing assistant header is the one part of a
/// render that is *not* a prefix of the next one.
pub(crate) fn render_native_prompt(
    env: &minijinja::Environment<'static>,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    bos: &str,
    eos: &str,
    reasoning: &ReasoningParams,
    add_generation_prompt: bool,
) -> std::result::Result<String, minijinja::Error> {
    let msgs: Vec<Value> = messages
        .iter()
        .map(LlamaLocalProvider::render_message_native)
        .collect();
    let tool_defs: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect();

    env.get_template("chat")?.render(minijinja::context! {
        messages => msgs,
        tools => tool_defs,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos,
        eos_token => eos,
        ..reasoning_context(reasoning),
    })
}

/// `params` as extra template context keys — only the ones it actually has
/// `Some` for. Built as a map rather than passed as fixed-shape `Option`
/// fields because minijinja's `is defined` sees a difference between "key
/// absent" and "key present with a null value," and at least one family's
/// template (DeepSeek-V4's) branches on `... is defined` — a null-valued key
/// would trip that check differently than a truly absent one.
///
/// `thinking` and `enable_thinking` are both set to the same value: the two
/// literal variable names found across the families surveyed for this (see
/// `profile::ReasoningParams`'s docs and issue #138). Harmless for a
/// template that reads only one of them.
///
/// Free function rather than a `&self` method so it's testable against a
/// bare `minijinja::Environment` without needing a loaded model — see
/// `tests::reasoning_context_omits_unset_keys_rather_than_nulling_them`.
fn reasoning_context(params: &ReasoningParams) -> minijinja::Value {
    let mut extra = std::collections::BTreeMap::new();
    if let Some(thinking) = params.thinking {
        extra.insert("thinking", minijinja::Value::from(thinking));
        extra.insert("enable_thinking", minijinja::Value::from(thinking));
    }
    if let Some(effort) = params.effort_text {
        extra.insert("reasoning_effort", minijinja::Value::from(effort));
    }
    minijinja::Value::from_serialize(&extra)
}

impl LlamaLocalProvider {
    /// Render the model's embedded jinja chat template with minijinja.
    fn render_template(
        &self,
        pairs: &[(&'static str, String)],
    ) -> std::result::Result<String, minijinja::Error> {
        let env = self.jinja_env()?;
        let messages: Vec<Value> = pairs
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();

        let tmpl = env.get_template("chat")?;
        tmpl.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => self.bos,
            eos_token => self.eos,
            ..reasoning_context(&self.reasoning),
        })
    }

    /// True if the embedded chat template formats tools in this model's own
    /// native protocol, so we can feed it structured tools rather than JSON
    /// prose. Which literals count is the profile's answer, not this file's.
    fn template_supports_native_tools(&self) -> bool {
        self.template_src
            .as_deref()
            .is_some_and(|src| self.profile.template_formats_tools_natively(src))
    }

    /// Render via the model's native tool protocol: pass the OpenAI-style tools
    /// array and full message objects (with `tool_calls` / tool results) so the
    /// template emits its own wire format — Gemma 4's `<|tool>declaration:…` /
    /// `<|tool_call>` / `<|tool_response>`, MiniMax-M2.7's
    /// `<minimax:tool_call><invoke name="...">`, DeepSeek-V4's
    /// `<｜DSML｜tool_calls><｜DSML｜invoke name="...">`, or GPT-OSS's Harmony
    /// `to=functions.NAME<|channel|>commentary...<|message|>{...}<|call|>`
    /// (see `crate::harmony`, shared with `protocol::HarmonyProtocol`).
    fn render_native(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<String> {
        let env = self
            .jinja_env()
            .map_err(|e| anyhow::anyhow!("jinja env: {e}"))?;
        let rendered = render_native_prompt(
            &env,
            messages,
            tools,
            &self.bos,
            &self.eos,
            &self.reasoning,
            true,
        )
        .map_err(|e| anyhow::anyhow!("render: {e}"))?;
        tracing::debug!("rendered {} tools via native tool protocol", tools.len());
        Ok(rendered)
    }

    /// Convert a ChatMessage to the message object the native template expects,
    /// preserving assistant `tool_calls` and `tool` results.
    fn render_message_native(msg: &ChatMessage) -> Value {
        match msg.role {
            ChatRole::System => serde_json::json!({"role": "system", "content": msg.content}),
            ChatRole::User => serde_json::json!({"role": "user", "content": msg.content}),
            ChatRole::Assistant => {
                let mut m = serde_json::json!({"role": "assistant", "content": msg.content});
                if let Some(calls) = &msg.tool_calls {
                    let tc: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            })
                        })
                        .collect();
                    m["tool_calls"] = Value::Array(tc);
                }
                m
            }
            ChatRole::Tool => serde_json::json!({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
                "name": msg.tool_name,
            }),
        }
    }

    /// Last-resort manual ChatML layout when the embedded template is missing or
    /// won't render.
    fn chatml_fallback(&self, pairs: &[(&'static str, String)]) -> String {
        let mut out = String::new();
        out.push_str(&self.bos);
        for (role, content) in pairs {
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    /// Fold all system messages into the first user turn, for templates that
    /// don't support a system role (e.g. gemma).
    fn fold_system(pairs: Vec<(&'static str, String)>) -> Vec<(&'static str, String)> {
        let mut system = String::new();
        let mut rest: Vec<(&'static str, String)> = Vec::new();
        for (role, content) in pairs {
            if role == "system" {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&content);
            } else {
                rest.push((role, content));
            }
        }
        if system.is_empty() {
            return rest;
        }
        if let Some(first_user) = rest.iter_mut().find(|(role, _)| *role == "user") {
            first_user.1 = format!("{}\n\n{}", system, first_user.1);
        } else {
            rest.insert(0, ("user", system));
        }
        rest
    }

    /// Map a ChatMessage to a (role, content) pair. Assistant tool calls and tool
    /// results are folded into text so the model sees prior turns in the same
    /// protocol format we ask it to emit.
    fn render_message(msg: &ChatMessage) -> (&'static str, String) {
        match msg.role {
            ChatRole::System => ("system", msg.content.clone()),
            ChatRole::User => ("user", msg.content.clone()),
            ChatRole::Assistant => {
                if let Some(ref calls) = msg.tool_calls {
                    let json = Self::serialize_tool_calls(calls);
                    let content = if msg.content.is_empty() {
                        json
                    } else {
                        format!("{}\n{}", msg.content, json)
                    };
                    ("assistant", content)
                } else {
                    ("assistant", msg.content.clone())
                }
            }
            ChatRole::Tool => ("user", format!("Tool result: {}", msg.content)),
        }
    }

    /// The tool-use instruction block appended to the system prompt.
    fn tool_instructions(tools: &[ToolDefinition]) -> String {
        let list = Self::tools_to_json(tools);
        format!(
            "You have access to the following tools (described as JSON Schema):\n\
             {list}\n\n\
             To call a tool, respond with ONLY a single JSON object and nothing else:\n\
             {{\"name\": \"<tool_name>\", \"arguments\": {{ ...json args... }}}}\n\
             To call several tools at once, respond with a JSON array of such objects.\n\
             If no tool is needed, reply normally in plain text (do not output JSON)."
        )
    }

    /// Serialize ToolDefinitions to an OpenAI-style tools JSON array string.
    fn tools_to_json(tools: &[ToolDefinition]) -> String {
        let json_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();

        serde_json::to_string(&json_tools).unwrap_or_else(|_| "[]".to_string())
    }

    /// Serialize prior assistant tool calls into the protocol JSON.
    fn serialize_tool_calls(calls: &[ToolCallInfo]) -> String {
        let arr: Vec<Value> = calls
            .iter()
            .map(|c| serde_json::json!({"name": c.name, "arguments": c.arguments}))
            .collect();
        if arr.len() == 1 {
            arr[0].to_string()
        } else {
            Value::Array(arr).to_string()
        }
    }

    /// Core generation loop. Tokenize, decode, sample until EOG or max tokens.
    /// Returns (generated_text, token_usage).
    ///
    /// `cancel` is checked once per sampled token, which is the only point this
    /// loop yields: a decode of a single token is short, so a cancelled turn
    /// stops in about the time one token takes.
    fn generate(&self, prompt: &str, cancel: &CancellationToken) -> Result<(String, TokenUsage)> {
        // Prefill is timed from here, not from the first `decode` call:
        // tokenization and finding or building a context are part of what a
        // user waits through before the first token, so leaving them out would
        // report a TTFT nobody experiences.
        let started = Instant::now();
        // The template usually emits {{ bos_token }} already; only add a BOS at
        // tokenization time if the prompt doesn't already start with it.
        let add_bos = if !self.bos.is_empty() && prompt.starts_with(self.bos.as_str()) {
            AddBos::Never
        } else {
            AddBos::Always
        };
        let tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        match &self.slots {
            Some(pool) => self.generate_reusing(&tokens, pool, started, cancel),
            None => self.generate_fresh(&tokens, started, cancel),
        }
    }

    /// The pre-reuse path: build a context, prefill the whole prompt, drop it.
    /// Still what a `kvCacheSlots = 0` run does, and the reference the reuse
    /// path is tested against.
    fn generate_fresh(
        &self,
        tokens: &[LlamaToken],
        started: Instant,
        cancel: &CancellationToken,
    ) -> Result<(String, TokenUsage)> {
        let n_prompt = tokens.len() as u32;
        let n_ctx = self.context_size_for(n_prompt);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);

        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        let n_past = feed(&mut ctx, tokens, 0, n_ctx)?;
        // A cold context evaluates every prompt token, so `evaluated` is the
        // whole prompt.
        let (text, _decoded, usage) =
            self.sample_until_done(&mut ctx, n_past, n_prompt, n_prompt, started, cancel)?;
        Ok((text, usage))
    }

    /// Decode only what the chosen slot does not already hold.
    ///
    /// The whole win is in `common_prefix_len`: iteration *N*'s prompt is a
    /// prefix of iteration *N+1*'s, so the suffix is a tool result and a few
    /// framing tokens where the prompt is thousands.
    ///
    /// Matching is on **token ids**, never on the assumption that the prompt was
    /// appended to. The next prompt is re-rendered through the jinja template
    /// from scratch, and the assistant turn inside it is gallium's
    /// re-serialization of the tool call rather than the tokens the model
    /// emitted; those can differ. Comparing ids means a divergence anywhere
    /// simply shortens the reuse instead of poisoning it.
    fn generate_reusing(
        &self,
        tokens: &[LlamaToken],
        pool: &Mutex<SlotPool>,
        started: Instant,
        cancel: &CancellationToken,
    ) -> Result<(String, TokenUsage)> {
        let n_prompt = tokens.len() as u32;
        let needed = self.context_size_for(n_prompt);

        // Checked out, not borrowed: the lock is held long enough to pick a
        // slot and no longer — not across a decode, and not across an
        // allocation either, which is why building one happens out here.
        let mut slot = match self.reserve(pool, tokens, needed) {
            Reservation::Ready(slot) => slot,
            Reservation::Build {
                n_ctx,
                victim,
                tick,
            } => {
                // Freed before the replacement is allocated, so two contexts'
                // worth of KV cache never exist at once — on a card chosen to
                // fit exactly one, the overlap would be an OOM. Dropping out
                // here rather than under the lock for the same reason the build
                // is out here: releasing a KV cache is not instant.
                drop(victim);
                match self.new_slot(n_ctx) {
                    Ok(mut slot) => {
                        slot.last_used = tick;
                        slot
                    }
                    // The reservation counted this slot as live. Give the
                    // capacity back, or a failed allocation would shrink the
                    // pool permanently.
                    Err(e) => {
                        pool.lock().busy -= 1;
                        return Err(e);
                    }
                }
            }
            Reservation::AllBusy => {
                tracing::debug!("KV cache: every slot busy, falling back to a fresh context");
                return self.generate_fresh(tokens, started, cancel);
            }
        };

        let result = self.generate_in_slot(&mut slot, tokens, n_prompt, started, cancel);

        // Returned whichever way the generation went. A slot dropped instead of
        // returned would shrink the pool for the life of the process.
        let mut pool = pool.lock();
        pool.busy -= 1;
        pool.idle.push(slot);
        result
    }

    /// Run one generation against a checked-out slot, keeping its token record
    /// exactly equal to what its KV cache holds.
    ///
    /// The invariant is one-directional and that is what makes the error paths
    /// safe: the cache may hold *more* than the record, never less. Anything
    /// past the record is cleared before the next call reads it, so a
    /// generation that dies halfway leaves a slot that is stale, not wrong.
    fn generate_in_slot(
        &self,
        slot: &mut Slot,
        tokens: &[LlamaToken],
        n_prompt: u32,
        started: Instant,
        cancel: &CancellationToken,
    ) -> Result<(String, TokenUsage)> {
        // One token must be left to decode: the sampler reads the logits of the
        // last position *evaluated*, and a fully-cached prompt evaluates
        // nothing. Reusing `len - 1` costs one token and keeps the loop's entry
        // condition identical to a cold start.
        let reuse = common_prefix_len(&slot.tokens, tokens).min(tokens.len().saturating_sub(1));

        // Drop the divergent tail. `p1 = None` means "to the end", so whatever
        // the slot held past this point — a previous turn's answer, or a
        // re-rendered assistant turn that tokenized differently — is gone.
        //
        // The `bool` this returns is not a "did I throw, ignore it" detail —
        // it is llama.cpp reporting whether the trim actually happened, and a
        // hybrid/recurrent model (Gated DeltaNet, Mamba, RWKV — anything
        // `LlamaModel::is_recurrent`/`is_hybrid`) can say no. Its state isn't
        // a per-position log the way an attention KV cache is; rolling it
        // back to an arbitrary earlier position needs a bounded snapshot
        // trail (llama.cpp's `n_rs_seq`, default 0 — this crate never sets
        // it) that a partial `reuse > 0` almost always exceeds
        // (`llama_memory_recurrent::seq_rm`, "models like Mamba or RWKV
        // can't have a state partially erased... because their state isn't
        // preserved for previous tokens"). A full clear (`p0 == 0`) is a
        // different, always-successful code path — only genuine partial
        // reuse can be refused.
        //
        // Trusting a refusal blindly (the bug behind issue #98) desyncs the
        // record from the cache: `slot.tokens` would say `reuse`, but the
        // model's own memory is still sitting at its pre-trim position, and
        // the next `feed()` submits token positions the model has no
        // consecutive continuation for — `llama_batch_allocr::init` rejects
        // it, surfacing as `Decode Error -1` (misleadingly labeled
        // `NTokensZero` by this crate's error type; the actual llama.cpp
        // reason, muted by `void_logs()`, is "positions must remain
        // consecutive"). So: ask for what was actually trimmed, and if it's
        // less than requested, fall back to the one range that is always
        // honored — clearing everything — rather than record a reuse that
        // didn't happen.
        let trimmed = slot
            .ctx
            .clear_kv_cache_seq(Some(0), Some(reuse as u32), None)
            .map_err(|e| anyhow::anyhow!("Failed to trim the KV cache: {}", e))?;
        let reuse = if trimmed {
            reuse
        } else {
            slot.ctx
                .clear_kv_cache_seq(Some(0), None, None)
                .map_err(|e| anyhow::anyhow!("Failed to reset the KV cache: {}", e))?;
            0
        };
        slot.tokens.truncate(reuse);

        tracing::debug!(
            "KV cache: reusing {}/{} prompt tokens, evaluating {}{}",
            reuse,
            tokens.len(),
            tokens.len() - reuse,
            if trimmed {
                ""
            } else {
                " (partial trim refused by llama.cpp — recurrent/hybrid memory can't roll back mid-sequence; reset to a full re-evaluation instead)"
            },
        );

        let n_past = feed(&mut slot.ctx, &tokens[reuse..], reuse as i32, slot.n_ctx)?;
        slot.tokens.extend_from_slice(&tokens[reuse..]);

        let evaluated = (tokens.len() - reuse) as u32;
        // On the way out by `?`, the record already covers the whole prompt and
        // simply does not claim the tokens sampled after it. Those stay in the
        // cache unrecorded, which is exactly the harmless direction: the next
        // call clears everything past its own prefix before reading.
        let (text, decoded, usage) =
            self.sample_until_done(&mut slot.ctx, n_past, n_prompt, evaluated, started, cancel)?;

        // The cache now holds the prompt plus everything decoded. Recording
        // exactly that is what lets the next call trust its prefix.
        slot.tokens.extend_from_slice(&decoded);
        Ok((text, usage))
    }

    /// Decide, under the lock, how this prompt gets a slot — without doing any
    /// of the work.
    ///
    /// Allocating a `LlamaContext` allocates a whole KV cache, which is slow
    /// enough that doing it here would block every other turn from checking a
    /// slot in or out. So the lock only ever reserves: capacity is claimed by
    /// bumping `busy`, and the caller builds outside it. A concurrent turn that
    /// arrives mid-allocation sees the pool as full and takes a fresh context,
    /// which is the right answer — the slot is spoken for.
    ///
    /// Preference order: the warmest slot that shares a prefix, then a new slot
    /// if the pool may still grow, then the coldest idle one — reused as-is when
    /// it is large enough, rebuilt when it is not. llama.cpp cannot grow a
    /// context in place, so a slot built for a shorter conversation is replaced
    /// rather than stretched: the cache it held is lost, which is a slow turn
    /// and not a wrong one.
    fn reserve(&self, pool: &Mutex<SlotPool>, tokens: &[LlamaToken], needed: u32) -> Reservation {
        let mut pool = pool.lock();
        pool.tick += 1;
        let tick = pool.tick;

        let warmest = pool
            .idle
            .iter()
            .enumerate()
            .filter(|(_, s)| s.n_ctx >= needed)
            .map(|(i, s)| (common_prefix_len(&s.tokens, tokens), i))
            .max();

        // Any shared prefix is worth having. A slot that shares nothing is only
        // worth taking over once the pool may not grow — until then a second
        // conversation deserves its own cache rather than evicting the first.
        if let Some((shared, index)) = warmest {
            if shared > 0 || pool.idle.len() + pool.busy >= pool.capacity {
                let mut slot = pool.idle.remove(index);
                slot.last_used = tick;
                pool.busy += 1;
                return Reservation::Ready(slot);
            }
        }

        if pool.idle.len() + pool.busy < pool.capacity {
            pool.busy += 1;
            return Reservation::Build {
                n_ctx: needed,
                victim: None,
                tick,
            };
        }

        // At capacity, and every idle slot is too small for this prompt. Take
        // the coldest out of the pool so the caller can drop it and build its
        // replacement; `busy` covers it in the meantime.
        let Some(coldest) = pool
            .idle
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i)
        else {
            // Nothing idle at all — every slot is mid-generation.
            return Reservation::AllBusy;
        };

        tracing::debug!("KV cache: rebuilding the coldest slot at {needed} tokens");
        let victim = pool.idle.remove(coldest);
        pool.busy += 1;
        Reservation::Build {
            n_ctx: needed,
            victim: Some(victim),
            tick,
        }
    }

    /// A fresh context, retained.
    fn new_slot(&self, n_ctx: u32) -> Result<Slot> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);
        let ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        // SAFETY: `ctx` borrows `*self.model`, which is behind a `Box` — its
        // address does not move when the provider does — and which outlives
        // every slot: slots are declared before `model`, so they drop first,
        // and `Drop for LlamaLocalProvider` clears them explicitly. The
        // extension is to `'static` because a field cannot name the lifetime of
        // its own struct; nothing here escapes the provider.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        Ok(Slot {
            ctx,
            tokens: Vec::new(),
            n_ctx,
            last_used: 0,
        })
    }

    /// The context size a prompt of `n_prompt` needs: room for it and for
    /// everything the model may generate after it, **rounded up**.
    ///
    /// The rounding is what makes reuse survive a growing transcript. Sized
    /// exactly, a slot built for an 11 836-token prompt is too small for the
    /// 11 882-token prompt of the next iteration, and llama.cpp cannot grow a
    /// context in place — so every iteration would rebuild, discard its cache,
    /// and re-prefill, which is precisely the cost this is here to remove. A
    /// turn's growth per iteration is a tool result; a whole chunk of headroom
    /// buys many of them for one allocation.
    fn context_size_for(&self, n_prompt: u32) -> u32 {
        const CHUNK: u32 = 4096;
        let needed = self.n_ctx.max(n_prompt.saturating_add(self.max_tokens));
        needed.div_ceil(CHUNK).saturating_mul(CHUNK)
    }

    /// Sample from a context whose prompt has already been fed, until EOG, the
    /// token budget, or a Gemma tool boundary.
    ///
    /// Split out because the two ways a prompt gets in — plain tokens, and
    /// mtmd's chunks — differ only in the feeding. Everything after the first
    /// logits is identical, and duplicating it would mean maintaining the
    /// tool-boundary and UTF-8 handling twice.
    ///
    /// `n_past` is where the fed prompt left off; `n_prompt` is what to report
    /// as the input cost; `evaluated` is how much of that prompt this call
    /// actually computed, which is smaller when a warm cache served the rest;
    /// `started` is when the caller began building the prompt, so prefill can be
    /// priced from the moment the call did.
    ///
    /// Returns the text, **the tokens the cache now holds beyond the prompt**,
    /// and the usage. A caller retaining the context needs that second value to
    /// keep its record of the cache exact.
    ///
    /// `cancel` is checked once per sampled token, which is the only point this
    /// loop yields: a decode of a single token is short, so a cancelled turn
    /// stops in about the time one token takes.
    fn sample_until_done(
        &self,
        ctx: &mut LlamaContext,
        n_past: i32,
        n_prompt: u32,
        evaluated: u32,
        started: Instant,
        cancel: &CancellationToken,
    ) -> Result<(String, Vec<LlamaToken>, TokenUsage)> {
        let mut batch =
            LlamaBatch::new(self.n_ctx.max(n_past as u32 + self.max_tokens) as usize, 1);

        // top_k before top_p before temp, matching llama.cpp's own default
        // chain order (top_k/typical/top_p/min_p all narrow the candidate
        // set before temp rescales what's left) — applying temp first would
        // let a high temperature flatten the distribution top_k/top_p then
        // sample from.
        let mut stages = Vec::with_capacity(4);
        if let Some(k) = self.top_k {
            // Safe: `self.top_k` is always `1..=i32::MAX` when `Some` — see
            // its field doc.
            stages.push(LlamaSampler::top_k(k as i32));
        }
        if let Some(p) = self.top_p {
            stages.push(LlamaSampler::top_p(p, 1));
        }
        stages.push(LlamaSampler::temp(self.temperature));
        stages.push(LlamaSampler::dist(1234));
        let mut sampler = LlamaSampler::chain_simple(stages);

        // Generate tokens
        let mut n_cur = n_past;
        let batch_start = n_cur;
        let max_tokens = n_cur + self.max_tokens as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut generated_text = String::new();
        // Only the tokens that were *decoded* land here — the one that ends the
        // loop is sampled but never fed, so it is not in the cache and must not
        // be recorded as if it were.
        let mut decoded: Vec<LlamaToken> = Vec::new();
        // -1 is llama.cpp's "the last logits in the context", which is the only
        // thing both feeding paths can name: after `eval_chunks` there is no
        // batch of ours to index into.
        let mut logits_idx: i32 = -1;
        // Stamped on the first sample, before the EOG check: a model whose very
        // first token ends the turn still paid for the prefill, and pricing that
        // call as if it never produced anything would hide the cost.
        let mut first_token_at: Option<Instant> = None;

        while n_cur <= max_tokens {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled.into());
            }

            let token = sampler.sample(ctx, logits_idx);
            if first_token_at.is_none() {
                first_token_at = Some(Instant::now());
            }

            if self.model.is_eog_token(token) {
                break;
            }

            // A single token that can't be decoded (e.g. an unused/control id)
            // shouldn't abort the whole generation — skip it and keep going.
            match self
                .model
                .token_to_piece_bytes(token, 8, false, None)
                .or_else(|_| self.model.token_to_piece_bytes(token, 256, false, None))
            {
                Ok(output_bytes) => {
                    let mut output_string = String::with_capacity(32);
                    let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                    generated_text.push_str(&output_string);
                }
                Err(e) => tracing::debug!("skipping undecodable token {token:?}: {e}"),
            }

            // Stop where this model's own profile says a turn ends — e.g. Gemma
            // 4 closing a tool call, which must run rather than have its result
            // hallucinated. When every marker resolved to a single token for
            // this vocabulary (`stop_marker_ids`, ADR 0003 step 5) this is an
            // id comparison on the token just sampled; otherwise it falls back
            // to `stops_generation`'s decoded-text scan, same as a profile
            // with no markers at all always does (its default is `false` and
            // never touches `generated_text`).
            let stopped = match &self.stop_marker_ids {
                Some(ids) => ids.contains(&token),
                None => self.profile.stops_generation(&generated_text),
            };
            if stopped {
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| anyhow::anyhow!("Failed to add generated token: {}", e))?;
            decoded.push(token);
            n_cur += 1;
            // From here on the batch holds exactly the one token just decoded,
            // so its logits are at 0. (-1 would work too; being explicit costs
            // nothing and says which position is meant.)
            logits_idx = 0;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
        }

        let n_output = (n_cur - batch_start) as u64;
        // With no token sampled at all there is no boundary between the halves,
        // so the whole elapsed time is charged to prefill — which is where the
        // work went.
        let (prefill, decode) = match first_token_at {
            Some(first) => (first.duration_since(started), first.elapsed()),
            None => (started.elapsed(), Duration::ZERO),
        };
        let usage = TokenUsage::timed_partial_prefill(
            n_prompt as u64,
            n_output,
            n_prompt as u64 + n_output,
            evaluated as u64,
            prefill,
            decode,
        );
        // `evaluated` is named beside `input` because with a warm cache they
        // differ, and the gap is the whole point of the change: 11836 of 11882
        // reused is a working cache, 0 of 11882 twice in a row is not.
        tracing::info!(
            "Local LLM usage: input={} (evaluated {}), output={}, total={} — \
             prefill {:.2}s ({}), decode {:.2}s ({})",
            usage.input_tokens,
            evaluated,
            usage.output_tokens,
            usage.total_tokens,
            prefill.as_secs_f64(),
            fmt_rate(usage.prefill_rate()),
            decode.as_secs_f64(),
            fmt_rate(usage.decode_rate()),
        );

        Ok((generated_text, decoded, usage))
    }

    /// Refuse a turn whose media this provider cannot actually encode, naming
    /// the missing piece.
    ///
    /// There are two different "no" here and they want different fixes: no
    /// projector configured is a config line the user can add, while a
    /// projector that has no vision encoder is the wrong file for the job. A
    /// single "cannot see images" would send them looking in the wrong place.
    ///
    /// Still a refusal rather than a silent drop, for the reason it always was:
    /// a model handed a caption without its picture answers confidently about
    /// an image it never received.
    fn refuse_unsupported_media(&self, images: usize, clips: usize) -> Result<()> {
        if self.mtmd.is_none() {
            anyhow::bail!(
                "the llama.cpp backend has no multimodal projector \
                 ({images} image(s), {clips} audio clip(s) attached). \
                 Set `[llm] mmprojPath` to this model's mmproj-*.gguf — GGUF repos publish one \
                 beside the model — or MMPROJ_PATH in the environment."
            );
        }
        // Asked per modality, because a projector can carry one and not the
        // other and the fix differs: a missing audio encoder is the wrong file,
        // not a missing config line.
        if images > 0 && !self.supports_vision {
            anyhow::bail!(
                "the configured projector has no vision encoder ({images} image(s) attached). \
                 It reports vision={}, audio={} — check that mmprojPath names the projector \
                 built for this model.",
                self.supports_vision,
                self.supports_audio
            );
        }
        if clips > 0 && !self.supports_audio {
            anyhow::bail!(
                "the configured projector has no audio encoder ({clips} audio clip(s) attached). \
                 It reports vision={}, audio={} — this model's projector may be vision-only.",
                self.supports_vision,
                self.supports_audio
            );
        }
        Ok(())
    }

    /// Rewrite `messages` so every attachment appears as a media marker in the
    /// text, and collect the attachment bytes in the same order.
    ///
    /// Both halves come from one pass on purpose. mtmd pairs markers with
    /// bitmaps *positionally*, so a prompt whose markers and byte list were
    /// built separately could drift and hand the model the wrong picture — an
    /// error nothing downstream could detect.
    ///
    /// The marker goes before the text of the message it belongs to, which is
    /// what llama.cpp's own `mtmd-cli` does and what the Gemma templates expect:
    /// the image is the thing being asked about, so it reads first.
    fn stage_media(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(Vec<ChatMessage>, Vec<MediaAttachment>)> {
        use base64::Engine;

        let mut staged = Vec::with_capacity(messages.len());
        let mut media = Vec::new();

        for msg in messages {
            if msg.media.is_empty() {
                staged.push(msg.clone());
                continue;
            }

            let mut markers = String::new();
            // One marker per attachment, emitted while walking `media` in its
            // own order — so the Nth marker and the Nth bitmap are the Nth
            // attachment, by construction rather than by agreement between two
            // loops. Markers are identical strings; only position carries the
            // pairing, which is why this walk must not reorder anything.
            for item in &msg.media {
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(item.base64())
                    .map_err(|e| {
                        anyhow::anyhow!("attached {} is not valid base64: {}", item.media_type(), e)
                    })?;
                media.push(MediaAttachment {
                    bytes,
                    kind: item.kind(),
                    label: item.media_type().to_string(),
                });
                markers.push_str(&self.media_marker);
                markers.push('\n');
            }

            let mut with_marker = msg.clone();
            with_marker.content = format!("{}{}", markers, msg.content);
            // The bytes now live in `media`; leaving them on the message too
            // would tempt a later pass into counting them twice.
            with_marker.media = Vec::new();
            staged.push(with_marker);
        }

        Ok((staged, media))
    }

    /// Generate from a prompt that has media attached, via mtmd.
    ///
    /// The shape differs from [`Self::generate`] in one place: instead of
    /// tokenizing a string and decoding one batch, the prompt is split by mtmd
    /// around its media markers into chunks — text runs and encoded media — and
    /// `eval_chunks` feeds them in order, running the projector where it must.
    /// What comes out is a context with the whole prompt in it and an `n_past`,
    /// which is exactly what the sampling loop wants.
    ///
    /// `media` must be in the same order the markers appear in `prompt`, and
    /// there must be one of each: mtmd matches them positionally and refuses a
    /// mismatch, which is why the markers are inserted by the same pass that
    /// collects the bytes.
    fn generate_with_media(
        &self,
        prompt: &str,
        media: &[MediaAttachment],
        cancel: &CancellationToken,
    ) -> Result<(String, TokenUsage)> {
        // Includes decoding the attachments and running the projector, which on
        // this path is most of what happens before the first token.
        let started = Instant::now();
        let mtmd = self
            .mtmd
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no multimodal projector loaded"))?
            .lock();

        // Raw file bytes, straight through. llama.cpp links stb_image and
        // miniaudio and sniffs the format from the magic bytes, so a PNG and a
        // WAV take the same call and gallium decodes neither.
        let bitmaps = media
            .iter()
            .map(|m| {
                MtmdBitmap::from_buffer(&mtmd, &m.bytes, false).map_err(|e| {
                    anyhow::anyhow!(
                        "{} could not be decoded as {} ({} bytes): {}",
                        m.label,
                        m.kind,
                        m.bytes.len(),
                        e
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

        // Same rule the text path uses: the chat template usually emits BOS
        // itself, so only ask for one when the prompt does not already open
        // with it.
        let add_special = self.bos.is_empty() || !prompt.starts_with(self.bos.as_str());

        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: prompt.to_string(),
                    add_special,
                    // The rendered template is full of `<start_of_turn>` and
                    // friends; without this they would tokenize as literal text.
                    parse_special: true,
                },
                &bitmap_refs,
            )
            .map_err(|e| anyhow::anyhow!("mtmd tokenization failed: {}", e))?;

        // Media is expensive in tokens — an image can be hundreds — so the
        // context is sized from what mtmd actually produced, not from the
        // prompt string's length.
        let n_prompt = chunks.total_tokens() as u32;
        let n_ctx = self.n_ctx.max(n_prompt + self.max_tokens);
        tracing::debug!(
            "mtmd: {} chunk(s), {} token(s), {} position(s), n_ctx={}",
            chunks.len(),
            n_prompt,
            chunks.total_positions(),
            n_ctx
        );

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);
        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        // Runs the projector on media chunks and `llama_decode` on text ones,
        // in order, leaving the context holding the whole prompt.
        let n_past = chunks
            .eval_chunks(&mtmd, &ctx, 0, 0, n_ctx as i32, true)
            .map_err(|e| anyhow::anyhow!("mtmd chunk evaluation failed: {}", e))?;

        // The projector is only needed to *build* the prompt. Releasing it here
        // lets another multimodal turn start encoding while this one spends the
        // far longer stretch sampling.
        drop(mtmd);

        // Media never reuses a slot: every prompt token here was just evaluated.
        let (text, _decoded, usage) =
            self.sample_until_done(&mut ctx, n_past, n_prompt, n_prompt, started, cancel)?;
        Ok((text, usage))
    }
}

impl LlmProvider for LlamaLocalProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.build_prompt(messages, None)?;
        tracing::debug!("Prompt length: {} chars", prompt.len());

        let (text, _usage) = self.generate(&prompt, &CancellationToken::new())?;

        tracing::debug!("Generated: {}", text);
        Ok(text)
    }

    fn supports_tools(&self) -> bool {
        true
    }

    /// What the GGUF says the model was trained to hold.
    ///
    /// Deliberately *not* `n_ctx`, the size a context is actually built at.
    /// That one is elastic — `generate` opens each context at
    /// `n_ctx.max(n_prompt + max_tokens)` so a long prompt is never refused —
    /// so a gauge drawn against it would show a share of a denominator that
    /// grows to meet the numerator, and never approach full.
    fn context_window(&self) -> Option<u32> {
        Some(self.n_ctx_train)
    }

    fn agent_preamble(&self) -> Option<std::borrow::Cow<'static, str>> {
        self.profile.agent_preamble()
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
        // Two ways in, decided by whether this turn carries media at all. A
        // text turn takes exactly the path it always did — same tokenizer, same
        // batch, no projector touched — so enabling mtmd changes nothing for
        // the runs that do not use it.
        let (images, clips) = crate::llm::count_media(messages);
        let (generated, usage) = if images == 0 && clips == 0 {
            let prompt = self.build_prompt(messages, Some(tools))?;
            tracing::debug!("Prompt: {} chars, {} tools", prompt.len(), tools.len());
            self.generate(&prompt, cancel)?
        } else {
            self.refuse_unsupported_media(images, clips)?;
            let (staged, media) = self.stage_media(messages)?;
            let prompt = self.build_prompt(&staged, Some(tools))?;
            tracing::debug!(
                "Prompt: {} chars, {} tools, {} attachment(s)",
                prompt.len(),
                tools.len(),
                media.len()
            );
            self.generate_with_media(&prompt, &media, cancel)?
        };
        tracing::debug!("Raw generated: {}", generated);

        let calls = self.profile.tool_calls(&generated, tools);
        if !calls.is_empty() {
            tracing::info!("Local LLM returned {} tool call(s)", calls.len());
            return Ok(LlmResponse::ToolCalls(calls, Some(usage)));
        }

        // The reply goes to a person, so the model's thinking must not be in it.
        // Only the tool-call scan stripped anything before, and only to keep
        // braces inside a `<think>` block from being mistaken for JSON — so a
        // turn that ended in text handed the wrapper straight through, and
        // Gemma 4 opens every reply with `<|channel>thought … <channel|>`.
        Ok(LlmResponse::Text {
            content: self.profile.clean_reply(&generated),
            reasoning: None,
            usage: Some(usage),
        })
    }
}

/// Strip HF chat-template extensions minijinja can't parse. The `{% generation %}`
/// / `{% endgeneration %}` markers only tag assistant tokens for training masks
/// and are no-ops at inference time.
fn strip_unsupported_jinja(src: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"\{%-?\s*(end)?generation\s*-?%\}").unwrap());
    re.replace_all(src, "").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lenient_tojson_accepts_ensure_ascii_like_minimax_templates_use() {
        // Regression: minijinja's own `tojson` (the `json` feature) calls
        // `args.assert_all_used()` and rejects any kwarg it doesn't
        // recognize. MiniMax-M2.7's real chat template calls
        // `tojson(ensure_ascii=False)` — a Jinja2-ism minijinja never
        // modeled — which made every native-tool-protocol render fail with
        // "unknown keyword argument 'ensure_ascii'" and silently fall back
        // to the JSON-prose protocol, so parse_minimax_calls never actually
        // ran against real model output despite passing its own unit tests.
        let mut env = minijinja::Environment::new();
        env.add_filter("tojson", lenient_tojson);
        env.add_template("t", r#"{{ value | tojson(ensure_ascii=False) }}"#)
            .unwrap();
        let rendered = env
            .get_template("t")
            .unwrap()
            .render(minijinja::context! { value => "hé" })
            .expect("tojson(ensure_ascii=False) must not error");
        assert_eq!(rendered, "\"hé\"");
    }

    /// All four whitespace-control spellings become comments carrying the same
    /// control, and nothing else in the template is touched.
    #[test]
    fn generation_markers_become_comments_keeping_their_whitespace_control() {
        assert_eq!(strip_generation_markers("a{% generation %}b"), "a{##}b");
        assert_eq!(strip_generation_markers("a{%- generation -%}b"), "a{#--#}b");
        assert_eq!(strip_generation_markers("a{%- generation %}b"), "a{#-#}b");
        assert_eq!(strip_generation_markers("a{% generation -%}b"), "a{#-#}b");
        assert_eq!(
            strip_generation_markers("a{%- endgeneration -%}b"),
            "a{#--#}b"
        );
    }

    /// The neighbouring statements survive intact — including `for`, whose name
    /// a sloppier match could catch, and `add_generation_prompt`, which
    /// contains the word.
    #[test]
    fn other_statements_are_left_alone() {
        let src =
            "{%- for m in messages -%}{%- if add_generation_prompt -%}x{%- endif -%}{%- endfor -%}";
        assert_eq!(strip_generation_markers(src), src);
    }

    /// The claim the comment substitution rests on: replacing the marker must
    /// not change one character of rendered output. Asserted by rendering the
    /// same template twice — once with the markers, once with them absent — and
    /// requiring the results to be equal, whitespace included.
    #[test]
    fn stripping_a_marker_does_not_change_the_rendered_text() {
        let with_markers = "{%- for m in messages -%}\n    {%- generation -%}\n    {{- m -}}\n    {%- endgeneration -%}\n{%- endfor -%}";
        let without = "{%- for m in messages -%}\n    {{- m -}}\n{%- endfor -%}";

        let render = |src: &str| {
            chat_env(src)
                .unwrap()
                .get_template("chat")
                .unwrap()
                .render(minijinja::context! { messages => vec!["a", "b"] })
                .unwrap()
        };

        assert_eq!(render(with_markers), render(without));
        assert_eq!(render(with_markers), "ab");
    }

    /// The template that motivated this: LFM2.5's, which minijinja rejects
    /// outright until the markers are gone. `chat_env` is where a template is
    /// registered, so a parse failure here is a model that never sees its own
    /// format — see issue #182.
    #[test]
    fn lfm2s_real_template_registers() {
        let src = include_str!("../tests/fixtures/chat_templates/lfm2-8b-a1b.jinja");
        assert!(
            src.contains("{%- generation -%}"),
            "the fixture no longer carries the marker this test is about"
        );
        assert!(
            chat_env(src).is_ok(),
            "LFM2's own chat template must register"
        );
    }

    /// The correctness point `reasoning_context`'s doc comment calls out:
    /// an unset `ReasoningParams` must leave `thinking` **undefined** in the
    /// template, not defined-and-null, because DeepSeek-V4's real template
    /// (and this snippet, modeled on it) branches on `is defined` to decide
    /// whether to apply its own default at all. A null-valued key would
    /// satisfy `is defined` and skip that default silently.
    #[test]
    fn reasoning_context_omits_unset_keys_rather_than_nulling_them() {
        let mut env = minijinja::Environment::new();
        env.add_template(
            "t",
            "{%- if not thinking is defined -%}{%- set thinking = false -%}{%- endif -%}\
             {%- if thinking -%}ON{%- else -%}OFF{%- endif -%}\
             {%- if reasoning_effort == 'high' -%}-HIGH{%- endif -%}",
        )
        .unwrap();
        let tmpl = env.get_template("t").unwrap();

        let unset = ReasoningParams::default();
        assert_eq!(
            tmpl.render(minijinja::context! { ..reasoning_context(&unset) })
                .unwrap(),
            "OFF",
            "an omitted key must let the template's own `is defined` default apply"
        );

        let high = ReasoningParams {
            thinking: Some(true),
            effort_text: Some("high"),
        };
        assert_eq!(
            tmpl.render(minijinja::context! { ..reasoning_context(&high) })
                .unwrap(),
            "ON-HIGH"
        );
    }

    /// `LlamaBackend::init()` is guarded by a process-global atomic and fails
    /// with `BackendAlreadyInitialized` on every call after the first. Each
    /// provider used to call it directly, so the *second* local `thread/start`
    /// in a process died there. Every provider now shares one handle.
    #[test]
    fn the_backend_can_be_obtained_more_than_once() {
        let first = shared_backend().expect("first caller initializes the backend");
        let second = shared_backend().expect("a second provider must reuse it, not re-init");
        assert!(
            std::ptr::eq(first, second),
            "both callers must get the one process-wide backend"
        );
    }

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    /// The shape every ReAct iteration has: the previous prompt, the answer the
    /// model gave, and a tool result appended after it.
    #[test]
    fn a_grown_transcript_shares_everything_but_its_tail() {
        let cached = toks(&[1, 2, 3, 4, 5]); // prompt + what the model generated
        let next = toks(&[1, 2, 3, 4, 5, 6, 7]); // that, plus a tool result
        assert_eq!(common_prefix_len(&cached, &next), 5);
    }

    /// A re-rendered assistant turn can tokenize differently from what the model
    /// emitted. Reuse must then stop at the divergence rather than trust the
    /// rest — the case that makes this a token comparison and not an append.
    #[test]
    fn a_divergence_stops_the_reuse_there() {
        let cached = toks(&[1, 2, 3, 99, 100]);
        let next = toks(&[1, 2, 3, 42, 100]);
        assert_eq!(common_prefix_len(&cached, &next), 3);
    }

    #[test]
    fn an_unrelated_conversation_shares_nothing_useful() {
        assert_eq!(common_prefix_len(&toks(&[1, 2, 3]), &toks(&[9, 8, 7])), 0);
        assert_eq!(common_prefix_len(&[], &toks(&[1, 2])), 0);
        assert_eq!(common_prefix_len(&toks(&[1, 2]), &[]), 0);
    }

    /// A prompt already wholly in the cache still has to leave one token to
    /// evaluate: the sampler reads the logits of the last position *evaluated*,
    /// and reusing everything would evaluate nothing and sample from whatever
    /// the previous call left behind.
    #[test]
    fn a_fully_cached_prompt_keeps_one_token_to_evaluate() {
        let cached = toks(&[1, 2, 3, 4]);
        let next = toks(&[1, 2, 3, 4]);
        let reuse = common_prefix_len(&cached, &next).min(next.len().saturating_sub(1));
        assert_eq!(reuse, 3);
        assert_eq!(next.len() - reuse, 1, "exactly one token is evaluated");
    }

    /// The same rule when the cache runs *past* the new prompt — a slot holding
    /// last turn's answer, asked for a prompt that is a prefix of it.
    #[test]
    fn a_cache_longer_than_the_prompt_also_leaves_one_token() {
        let cached = toks(&[1, 2, 3, 4, 5, 6]);
        let next = toks(&[1, 2, 3]);
        let reuse = common_prefix_len(&cached, &next).min(next.len().saturating_sub(1));
        assert_eq!(reuse, 2);
    }

    /// The measured turn from issue #86: 11 836 tokens, then 11 882 after a tool
    /// result. Sized exactly, the second iteration would not fit the first's
    /// context and would rebuild — throwing away the cache this change exists to
    /// keep. Both must land on one size.
    #[test]
    fn a_growing_transcript_keeps_the_same_slot() {
        // A provider is a multi-GB model load, so exercise the arithmetic the
        // way `context_size_for` does it rather than building one.
        let size_for = |n_prompt: u32, floor: u32, max_tokens: u32| -> u32 {
            const CHUNK: u32 = 4096;
            floor
                .max(n_prompt.saturating_add(max_tokens))
                .div_ceil(CHUNK)
                .saturating_mul(CHUNK)
        };
        let first = size_for(11_836, 8192, 4096);
        let second = size_for(11_882, 8192, 4096);
        assert_eq!(first, second, "one tool result must not force a rebuild");
        assert!(
            second >= 11_882 + 4096,
            "the prompt and its generation budget both have to fit"
        );
    }
}
