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
use crate::AgentError;

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
    /// Literal BOS/EOS token text (e.g. "<bos>", "<eos>"), fed to the template.
    bos: String,
    eos: String,
    temperature: f32,
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

impl LlamaLocalProvider {
    pub fn new(
        model_path: &str,
        mmproj_path: Option<&str>,
        temperature: f32,
        max_tokens: u32,
        n_ctx: u32,
    ) -> Result<Self> {
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

        // Layers to offload to the GPU. Override with GALLIUM_GPU_LAYERS to
        // fit a smaller VRAM budget (e.g. a 6 GB card can't hold a 5 GB model
        // plus KV cache, so partial offload like 20 avoids an OOM). Default 999
        // = offload everything.
        let gpu_layers: u32 = if use_gpu {
            std::env::var("GALLIUM_GPU_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(999)
        } else {
            0
        };
        tracing::info!("  GPU layers to offload: {}", gpu_layers);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

        let model = Box::new(
            LlamaModel::load_from_file(backend, Path::new(model_path), &model_params)
                .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?,
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
            bos,
            eos,
            temperature,
            max_tokens,
            n_ctx,
            n_ctx_train,
            mtmd,
            media_marker,
            supports_vision,
            supports_audio,
        })
    }

    /// Render the conversation through the model's chat template, injecting tool
    /// definitions into the system message when tools are available.
    fn build_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<String> {
        // Gemma-4-style templates format tools natively (`<|tool>declaration:…`,
        // `<|tool_call>`, `<|tool_response>`). Feed those templates structured
        // tool inputs so the model sees tools in the exact form it was trained
        // on. Fall back to the generic JSON-prose protocol if it doesn't render.
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
    fn jinja_env(&self) -> std::result::Result<minijinja::Environment<'static>, minijinja::Error> {
        let src = self.template_src.as_deref().ok_or_else(|| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "no chat template")
        })?;

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
        env.add_template_owned("chat", src.to_string())?;
        Ok(env)
    }

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
        })
    }

    /// True if the embedded chat template formats tools in the Gemma-4 native
    /// protocol, so we can feed it structured tools rather than JSON prose.
    fn template_supports_native_tools(&self) -> bool {
        self.template_src.as_deref().map_or(false, |s| {
            s.contains("<|tool_call>") || s.contains("<|tool>") || s.contains("declaration:")
        })
    }

    /// Render via the model's native tool protocol: pass the OpenAI-style tools
    /// array and full message objects (with `tool_calls` / tool results) so the
    /// template emits `<|tool>declaration:…`, `<|tool_call>`, `<|tool_response>`.
    fn render_native(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<String> {
        let env = self
            .jinja_env()
            .map_err(|e| anyhow::anyhow!("jinja env: {e}"))?;

        let msgs: Vec<Value> = messages.iter().map(Self::render_message_native).collect();
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

        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("get template: {e}"))?;
        let rendered = tmpl
            .render(minijinja::context! {
                messages => msgs,
                tools => tool_defs,
                add_generation_prompt => true,
                bos_token => self.bos,
                eos_token => self.eos,
            })
            .map_err(|e| anyhow::anyhow!("render: {e}"))?;
        tracing::debug!(
            "rendered {} tools via native Gemma tool protocol",
            tools.len()
        );
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

        // Checked out, not borrowed: the lock is held only long enough to pick
        // a slot, never across a decode.
        let Some(mut slot) = self.check_out(pool, tokens, needed)? else {
            tracing::debug!("KV cache: every slot busy, falling back to a fresh context");
            return self.generate_fresh(tokens, started, cancel);
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
        slot.ctx
            .clear_kv_cache_seq(Some(0), Some(reuse as u32), None)
            .map_err(|e| anyhow::anyhow!("Failed to trim the KV cache: {}", e))?;
        slot.tokens.truncate(reuse);

        tracing::debug!(
            "KV cache: reusing {}/{} prompt tokens, evaluating {}",
            reuse,
            tokens.len(),
            tokens.len() - reuse,
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

    /// Take the best slot for this prompt out of the pool, or `None` when every
    /// slot is busy and no more may be created.
    ///
    /// Preference order: the warmest slot that shares a prefix, then a new slot
    /// if the pool may still grow, then the coldest idle one — reused as-is when
    /// it is large enough, rebuilt when it is not. llama.cpp cannot grow a
    /// context in place, so a slot built for a shorter conversation is replaced
    /// rather than stretched: the cache it held is lost, which is a slow turn
    /// and not a wrong one.
    fn check_out(
        &self,
        pool: &Mutex<SlotPool>,
        tokens: &[LlamaToken],
        needed: u32,
    ) -> Result<Option<Slot>> {
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
                return Ok(Some(slot));
            }
        }

        if pool.idle.len() + pool.busy < pool.capacity {
            let mut slot = self.new_slot(needed)?;
            slot.last_used = tick;
            pool.busy += 1;
            return Ok(Some(slot));
        }

        // At capacity, and every idle slot is too small for this prompt.
        let Some(coldest) = pool
            .idle
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i)
        else {
            // Nothing idle at all — every slot is mid-generation.
            return Ok(None);
        };

        tracing::debug!("KV cache: rebuilding the coldest slot at {needed} tokens");
        // Dropped before the replacement is built, so two contexts' worth of KV
        // cache never exist at once — on a card chosen to fit exactly one, the
        // overlap would be an OOM.
        pool.idle.remove(coldest);
        let mut slot = self.new_slot(needed)?;
        slot.last_used = tick;
        pool.busy += 1;
        Ok(Some(slot))
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

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature),
            LlamaSampler::dist(1234),
        ]);

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

            // Stop at Gemma-4 tool boundaries: once the model closes a tool call
            // (`<tool_call|>`) or emits a tool-response marker, stop so we can run
            // the tool instead of letting it hallucinate a result. These literals
            // are gemma-specific, so this is a no-op for other local models.
            if generated_text.ends_with("<tool_call|>")
                || generated_text.contains("<|tool_response>")
            {
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

    /// Leniently extract tool calls from the model's reply. Accepts the whole
    /// reply as JSON, or the first balanced `{...}`/`[...]` block (handles models
    /// that wrap JSON in prose or ``` fences). Returns empty if none found.
    fn parse_tool_calls(text: &str) -> Vec<ToolCallInfo> {
        // Reasoning models emit <think>…</think> first; drop it so the JSON scan
        // doesn't latch onto braces inside the chain-of-thought.
        let cleaned = strip_think_blocks(text);
        let text = cleaned.as_str();

        let mut candidates: Vec<String> = Vec::new();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            candidates.push(trimmed.to_string());
        }
        if let Some(block) = first_balanced_json(text) {
            if candidates.first().map(|c| c != &block).unwrap_or(true) {
                candidates.push(block);
            }
        }

        for candidate in candidates {
            if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
                let mut calls = Self::extract_calls(&val);
                if !calls.is_empty() {
                    Self::number_ids(&mut calls);
                    return calls;
                }
            }
        }

        // Python/Llama-style calls some models prefer: `[name(arg=val, ...)]` or a
        // bare `name(arg=val)`. Gate on the whole reply looking like a call list
        // to avoid matching function names mentioned in prose.
        let t = text.trim();
        let looks_like_calls = (t.starts_with('[') && t.ends_with(']')) || is_single_call(t);
        if looks_like_calls {
            let mut calls = parse_python_calls(t);
            if !calls.is_empty() {
                Self::number_ids(&mut calls);
                return calls;
            }
        }

        // Gemma-style native format: some models ignore the JSON protocol and emit
        // `<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>` (with `<|"|>` as a
        // quote token). Parse it leniently as a last resort.
        let mut gemma = parse_gemma_calls(text);
        if !gemma.is_empty() {
            Self::number_ids(&mut gemma);
            return gemma;
        }

        Vec::new()
    }

    fn number_ids(calls: &mut [ToolCallInfo]) {
        for (i, call) in calls.iter_mut().enumerate() {
            call.id = format!("call_{i}");
        }
    }

    /// Pull ToolCallInfo out of a parsed JSON value in any of the shapes a model
    /// might emit: a bare object, an array of objects, `{"tool_calls": [...]}`,
    /// and either `{name, arguments}` or `{function: {name, arguments}}`.
    fn extract_calls(val: &Value) -> Vec<ToolCallInfo> {
        fn one(v: &Value) -> Option<ToolCallInfo> {
            let obj = v.as_object()?;
            let (name, raw_args) = if let Some(f) = obj.get("function").and_then(|f| f.as_object())
            {
                (
                    f.get("name")?.as_str()?.to_string(),
                    f.get("arguments").cloned(),
                )
            } else {
                (
                    obj.get("name")?.as_str()?.to_string(),
                    obj.get("arguments").cloned(),
                )
            };
            let arguments = match raw_args {
                // OpenAI serializes arguments as a JSON string; accept that too.
                Some(Value::String(s)) => {
                    serde_json::from_str(&s).unwrap_or(Value::Object(Default::default()))
                }
                Some(v) => v,
                None => Value::Object(Default::default()),
            };
            Some(ToolCallInfo {
                id: "call_0".to_string(),
                name,
                arguments,
            })
        }

        match val {
            Value::Array(arr) => arr.iter().filter_map(one).collect(),
            Value::Object(o) if o.contains_key("tool_calls") => o
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(one).collect())
                .unwrap_or_default(),
            Value::Object(_) => one(val).into_iter().collect(),
            _ => Vec::new(),
        }
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

        let calls = Self::parse_tool_calls(&generated);
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
            content: clean_reply(&generated),
            reasoning: None,
            usage: Some(usage),
        })
    }
}

/// True if the whole string is a single `name(args)` call.
fn is_single_call(s: &str) -> bool {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^[A-Za-z_]\w*\s*\(.*\)$").unwrap());
    re.is_match(s.trim())
}

/// Parse Python/Llama-style tool calls: `[name(k=v, ...), ...]` or `name(k=v)`.
/// Values are parsed as quoted strings, numbers, booleans, or JSON.
fn parse_python_calls(text: &str) -> Vec<ToolCallInfo> {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"([A-Za-z_]\w*)\s*\(([^)]*)\)").unwrap());

    let mut calls = Vec::new();
    for cap in re.captures_iter(text) {
        let name = cap[1].to_string();
        let mut args = serde_json::Map::new();
        for part in split_top_commas(&cap[2]) {
            if let Some((k, v)) = part.split_once('=') {
                args.insert(k.trim().to_string(), parse_py_value(v.trim()));
            }
        }
        calls.push(ToolCallInfo {
            id: "call_0".to_string(),
            name,
            arguments: Value::Object(args),
        });
    }
    calls
}

/// Split argument text on top-level commas, ignoring commas inside quotes.
fn split_top_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in s.chars() {
        match c {
            '"' | '\'' if quote.is_none() => {
                quote = Some(c);
                cur.push(c);
            }
            c if Some(c) == quote => {
                quote = None;
                cur.push(c);
            }
            ',' if quote.is_none() => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse a Python-literal-ish value into JSON.
fn parse_py_value(v: &str) -> Value {
    let v = v.trim();
    if v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')))
    {
        return Value::String(v[1..v.len() - 1].to_string());
    }
    match v {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    if let Ok(n) = v.parse::<i64>() {
        return Value::from(n);
    }
    if let Ok(f) = v.parse::<f64>() {
        return Value::from(f);
    }
    serde_json::from_str::<Value>(v).unwrap_or_else(|_| Value::String(v.to_string()))
}

/// Parse the gemma-style native tool-call format that some models emit instead
/// of the JSON protocol we ask for:
/// `<|tool_call>call:NAME{key:<|"|>value<|"|>, key2:123}<tool_call|>`.
/// `<|"|>` is the model's quote token; tool names may contain hyphens (e.g. the
/// MCP tool `search-godoc`). Delimiter-agnostic — we key on `call:NAME{...}`.
fn parse_gemma_calls(text: &str) -> Vec<ToolCallInfo> {
    // Shared wire-format parser (see `crate::gemma`). Names are kept verbatim
    // here — the llama.cpp path is the general-purpose local backend and must
    // not fold mixed-case MCP tool names.
    crate::gemma::parse_native_tool_calls(text)
        .into_iter()
        .map(|c| ToolCallInfo {
            id: "call_0".to_string(),
            name: c.name,
            arguments: c.arguments,
        })
        .collect()
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

/// The reply with the model's thinking taken out of it.
///
/// Two shapes, because the backend serves every GGUF rather than one family:
/// Gemma 4's `<|channel>thought … <channel|>` and `<|think|>…<|/think|>`, and
/// the `<think>…</think>` that reasoning models like LFM2.5 emit. Each is
/// distinctive enough that running both over a model that emits neither is a
/// no-op.
///
/// Channel first: it keeps only what follows the last `<channel|>`, so running
/// it after would have to reason about markers already removed.
fn clean_reply(text: &str) -> String {
    let s = crate::gemma::strip_thinking_blocks(text);
    strip_think_blocks(&s).trim().to_string()
}

/// Remove well-formed `<think>...</think>` blocks (case-insensitive). An unclosed
/// `<think>` (model still reasoning, no answer yet) is left as-is.
fn strip_think_blocks(text: &str) -> String {
    let mut s = text.to_string();
    loop {
        let lower = s.to_lowercase();
        let Some(start) = lower.find("<think>") else {
            break;
        };
        let Some(end_rel) = lower[start..].find("</think>") else {
            break;
        };
        let end = start + end_rel + "</think>".len();
        s.replace_range(start..end, "");
    }
    s
}

/// Find the first balanced `{...}` or `[...]` span in `text`, respecting JSON
/// string literals (so braces inside strings don't unbalance it). Returns the
/// substring including the brackets, or None.
fn first_balanced_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(text[start..=i].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_object() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"{"name": "read", "arguments": {"path": "a.txt"}}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn parses_object_wrapped_in_prose_and_fences() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "Sure, I'll do that.\n```json\n{\"name\": \"glob\", \"arguments\": {\"pattern\": \"*.rs\"}}\n```",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
    }

    #[test]
    fn parses_array_of_calls_with_unique_ids() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn parses_openai_shape_with_stringified_args() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"{"tool_calls": [{"function": {"name": "read", "arguments": "{\"path\": \"x\"}"}}]}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "x");
    }

    #[test]
    fn parses_call_after_think_block() {
        let calls = LlamaLocalProvider::parse_tool_calls(
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
        let calls = LlamaLocalProvider::parse_tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc");
        assert_eq!(calls[0].arguments["query"], "mcp-go");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn parses_gemma_call_with_mixed_args() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "<|tool_call>call:grep{pattern:<|\"|>foo<|\"|>, limit:50}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments["pattern"], "foo");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    #[test]
    fn plain_prose_is_not_a_gemma_call() {
        let calls =
            LlamaLocalProvider::parse_tool_calls("Sure, I'll call the search tool for you.");
        assert!(calls.is_empty());
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
        let calls = LlamaLocalProvider::parse_tool_calls(raw);
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
        let calls = LlamaLocalProvider::parse_tool_calls(r#"[read(file_path="codeword.txt")]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "codeword.txt");
    }

    #[test]
    fn parses_multiple_python_calls() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"[glob(pattern="*.rs"), grep(pattern="fn main", path="src")]"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments["path"], "src");
    }

    #[test]
    fn prose_mentioning_a_function_is_not_a_call() {
        let calls =
            LlamaLocalProvider::parse_tool_calls("You can use the read() function to open files.");
        assert!(calls.is_empty());
    }

    #[test]
    fn plain_text_yields_no_calls() {
        let calls = LlamaLocalProvider::parse_tool_calls("The capital of France is Paris.");
        assert!(calls.is_empty());
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

    /// The reply from a real gemma4-12b session, which reached the user with the
    /// channel wrapper still on it. This is that bug.
    #[test]
    fn a_gemma_channel_wrapper_does_not_reach_the_reply() {
        let raw = "<|channel>thought\n<channel|>This project, **rs-gallium**, is a \
                   research-oriented LLM inference framework written in Rust.";

        let reply = clean_reply(raw);

        assert!(reply.starts_with("This project"), "{reply:?}");
        assert!(!reply.contains("channel"), "{reply:?}");
    }

    /// Thinking with content in it, not just an empty channel — everything up to
    /// the close belongs to the model, not the reader.
    #[test]
    fn thinking_inside_the_channel_is_dropped_with_it() {
        let raw = "<|channel>thought\nThe user asked about the repo. I should \
                   check git log first.<channel|>Here is what I found.";

        assert_eq!(clean_reply(raw), "Here is what I found.");
    }

    /// The other shape the same model uses.
    #[test]
    fn a_paired_think_wrapper_is_dropped_too() {
        assert_eq!(
            clean_reply("<|think|>reasoning here<|/think|>The answer."),
            "The answer."
        );
    }

    /// What a reasoning model like LFM2.5 emits. Same leak, same site — it just
    /// had not been reported yet.
    #[test]
    fn a_think_block_is_dropped_from_the_reply() {
        assert_eq!(
            clean_reply("<think>Let me work through this.</think>\nThe answer."),
            "The answer."
        );
    }

    /// A model that emits neither must come through untouched, since this runs
    /// over every GGUF the backend serves.
    #[test]
    fn an_ordinary_reply_is_left_alone() {
        let plain = "The capital of France is Paris.";
        assert_eq!(clean_reply(plain), plain);
    }

    /// Prose that merely mentions the words is not a wrapper.
    #[test]
    fn prose_about_thinking_is_not_mistaken_for_it() {
        let text = "I was thinking about how the channel abstraction works.";
        assert_eq!(clean_reply(text), text);
    }
}
