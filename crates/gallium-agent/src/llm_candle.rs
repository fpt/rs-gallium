//! CandleProvider: wraps a local gallium-core CausalLM as an LlmProvider.
//!
//! Prompt rendering and reply/tool-call parsing are two different jobs, per
//! ADR 0003 (`docs/adr/0003-model-profiles.md`, step 3-b) — and, since that
//! step, two different types. Rendering is engine-specific (candle has no
//! jinja template and must build the model's prompt format itself) and comes
//! from a [`PromptRenderer`] ([`protocol`]'s remaining role: [`HarmonyProtocol`],
//! [`GemmaProtocol`], [`QwenProtocol`], [`Lfm2Protocol`]). Parsing is a
//! property of the *model*, shared with the llama.cpp backend, and comes from
//! a `crate::profile::ModelProfile` — the same instances `llm_local.rs` uses.
//! [`Arch`] selects both: it is the total, exact mapping from a GGUF/
//! safetensors hint to one of the four families candle can actually load
//! weights for, so there is no separate detection pass here the way
//! `profile::detect` runs for llama.cpp's six.
//!
//! ## Generation and decoding
//!
//! `run_generate_ids` runs the model and returns raw token IDs. All paths decode
//! with `skip_special=false` so that reply cleaning and tool-call parsing have
//! access to special-token markers (e.g. `<channel|>` for Gemma thinking,
//! `<|channel|>final` for Harmony channels).
//!
//! ## Tool calling
//!
//! `chat_with_tools()`:
//!
//! 1. Formats the prompt via `renderer.format_prompt_with_tools()`.
//! 2. Runs generation, stopping early when `profile.stops_generation()` says
//!    the model has closed a call — the same check `llm_local.rs` makes,
//!    now shared rather than duplicated as a separate EOS-token-id list.
//! 3. Decodes with skip_special=false and calls `profile.tool_calls()`, which
//!    can return more than one — unlike the old per-engine `ModelProtocol`,
//!    which carried at most one call per reply. If any are found, returns
//!    `LlmResponse::ToolCalls`.
//! 4. Otherwise extracts the response text via `profile.clean_reply()`.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use gallium_core::{generate_reusing, CausalLM, SamplingParams};
use tokenizers::Tokenizer;

use crate::cancel::CancellationToken;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, MediaContent, TokenUsage, ToolDefinition};
use crate::profile::{Gemma4, GptOss, Lfm2, ModelProfile, Qwen3, ReasoningEffort};
use crate::protocol::{GemmaProtocol, HarmonyProtocol, Lfm2Protocol, PromptRenderer, QwenProtocol};
use crate::streaming::StreamingReply;

pub struct CandleProvider {
    model: RefCell<Box<dyn CausalLM>>,
    /// The token ids this model's cache holds, prompt *and* generated — the
    /// record that makes reuse across calls possible, and the one that must
    /// never claim more than the cache has. Stays empty for a model that does
    /// not expose its cache (`CausalLM::cache`), which then never reuses.
    cached: RefCell<Vec<u32>>,
    /// What a rewind needs for the layers a position cannot address, taken at
    /// the end of the last prompt. `None` for a pure-attention model, whose
    /// layers roll back from what they already hold.
    checkpoint: RefCell<Option<gallium_core::CacheCheckpoint>>,
    tokenizer: Tokenizer,
    params: SamplingParams,
    /// EOS token IDs (includes <|end|>, </s>, <|call|>, model-specific terminators).
    eos_tokens: Vec<u32>,
    max_new_tokens: usize,
    renderer: Box<dyn PromptRenderer>,
    /// Which family's wire rules read this model's output — the same shared
    /// instance `llm_local.rs` uses for the identical arch, so a fix to a
    /// parser lands on both engines at once. `Arch::profile()` names it; there
    /// is no separate detection pass on this path (see the module doc).
    profile: &'static dyn ModelProfile,
    /// From the model's own metadata — GGUF's `<arch>.context_length` or
    /// safetensors' `max_position_embeddings`. `None` when the file is silent.
    context_window: Option<u32>,
    /// Whether `profile.stop_markers()` resolved to single ids for this
    /// tokenizer — see `resolve_stop_markers`. The ids themselves already
    /// live in `eos_tokens` when this is `Some` (that's what actually stops
    /// generation); this field only tells `run_generate_ids` whether it can
    /// skip its `profile.stops_generation()` fallback.
    stop_marker_ids: Option<Vec<u32>>,
    /// The Gemma 4 image path. `Some` only when the loaded model has a vision
    /// tower (`CausalLM::accepts_image_features`) and a `vision_config` was
    /// found — carries the preprocessor and the three literal image markers
    /// (`<start_of_image>` / `<image_soft_token>` / `<end_of_image>`) resolved
    /// from this tokenizer. `None` → images are refused, as before.
    vision: Option<VisionSupport>,
    /// Image feature rows `stage_images` produced for the *whole* prompt,
    /// parked here rather than handed straight to the model: how many of the
    /// leading rows the model must NOT see depends on the KV-reuse split,
    /// which `run_generate_ids` only knows after tokenizing. It trims and
    /// forwards them (see `stage_pending_image_features`); `take()`n every
    /// call, so a row staged by a call that failed never leaks into the next.
    staged_image_features: RefCell<Option<candle_core::Tensor>>,
}

/// Everything the candle backend needs to feed an image to a Gemma 4 model.
struct VisionSupport {
    processor: gallium_models::gemma4_image::Gemma4ImageProcessor,
    /// `<image_soft_token>` — one per soft token, replaced in place by
    /// `Gemma4Multimodal::inject_image_features`.
    soft_token: String,
    begin_token: String,
    end_token: String,
}

// CandleProvider is used only from single-threaded binary context (REPL) or
// under a Mutex (app-server). The RefCell is never accessed from multiple threads concurrently.
unsafe impl Send for CandleProvider {}
unsafe impl Sync for CandleProvider {}

/// Gemma 4's well-known multimodal token ids (config.json `image_token_id` /
/// `boi_token_id` / `eoi_token_id`). Every published Gemma 4 shares them.
const GEMMA4_IMAGE_SOFT_TOKEN_ID: u32 = 258_880;
const GEMMA4_BEGIN_OF_IMAGE_ID: u32 = 255_999;
const GEMMA4_END_OF_IMAGE_ID: u32 = 258_882;

impl VisionSupport {
    fn build(
        tokenizer: &Tokenizer,
        vc: &gallium_models::gemma4_vision::Gemma4VisionConfig,
    ) -> Result<Self> {
        let tok = |id: u32, name: &str| -> Result<String> {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| anyhow::anyhow!("tokenizer has no {name} token (id {id})"))
        };
        Ok(Self {
            processor: gallium_models::gemma4_image::Gemma4ImageProcessor::from_config(vc),
            soft_token: tok(GEMMA4_IMAGE_SOFT_TOKEN_ID, "<image_soft_token>")?,
            begin_token: tok(GEMMA4_BEGIN_OF_IMAGE_ID, "<start_of_image>")?,
            end_token: tok(GEMMA4_END_OF_IMAGE_ID, "<end_of_image>")?,
        })
    }

    /// `<start_of_image>` + `<image_soft_token>` × `n` + `<end_of_image>`.
    fn marker_block(&self, n: usize) -> String {
        let mut s = String::with_capacity(
            self.begin_token.len() + self.end_token.len() + n * self.soft_token.len(),
        );
        s.push_str(&self.begin_token);
        for _ in 0..n {
            s.push_str(&self.soft_token);
        }
        s.push_str(&self.end_token);
        s
    }
}

impl CandleProvider {
    // One caller (`load_candle_provider`), each parameter independently
    // required (not a settings bundle like `llm_local::LocalModelOptions` —
    // these are the loaded model's own pieces, not tunables), so an options
    // struct for the eighth one would be ceremony over the threshold rather
    // than a real grouping.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Box<dyn CausalLM>,
        tokenizer: Tokenizer,
        params: SamplingParams,
        max_new_tokens: usize,
        renderer: Box<dyn PromptRenderer>,
        profile: &'static dyn ModelProfile,
        context_window: Option<u32>,
        declared_eos_ids: &[u32],
        vision_config: Option<gallium_models::gemma4_vision::Gemma4VisionConfig>,
    ) -> Self {
        // Use get_vocab(true) — includes both the base BPE vocabulary AND added tokens.
        // get_added_vocabulary().get_vocab() misses tokens like <|im_end|> that appear
        // in the base BPE vocab for some models (e.g. Qwen3.5) rather than the added layer.
        //
        // This is a *heuristic* supplement to `declared_eos_ids` below, not the
        // primary source: it can only ever be as precise as `is_eos_like`'s
        // literal spellings, so it exists for stop points a single declared id
        // can't name at all (Harmony's `<|call|>`/`<|return|>` aren't "the"
        // EOS, they're this format's own turn boundaries) rather than as a
        // guess at what the model's real EOS is spelled like.
        let mut eos_tokens: Vec<u32> = tokenizer
            .get_vocab(true)
            .into_iter()
            .filter(|(k, _)| is_eos_like(k))
            .map(|(_, v)| v)
            .collect();
        let heuristic_count = eos_tokens.len();

        // The declared id(s) — GGUF `tokenizer.ggml.eos_token_id` /
        // `eot_token_id`, or safetensors `config.json`'s `eos_token_id` — are
        // what actually answers "is this model's real EOS in the set at
        // all," which no string heuristic can guarantee. This is what fixes
        // Gemma 4: its real terminator is `<turn|>` (declared id 106), which
        // contains none of `is_eos_like`'s literal spellings and so was never
        // in this set before declared ids were read at all.
        eos_tokens.extend(declared_eos_ids);

        // Tool-call closing markers (Gemma's `<tool_call|>`, …) used to be
        // added here too, by hand, as a per-protocol list built from format
        // knowledge this file doesn't own any more. `resolve_stop_markers`
        // does that job now, driven by `profile.stop_markers()` — the same
        // names `llm_local.rs` resolves for the identical model. Folded
        // straight into `eos_tokens` when resolution succeeds: `generate`
        // already checks this set before every step, so a resolved marker
        // needs no separate per-token check in `run_generate_ids` — only an
        // *unresolved* one does, falling back to `profile.stops_generation`
        // on decoded text there.
        let stop_marker_ids = resolve_stop_markers(&tokenizer, profile);
        if let Some(ids) = &stop_marker_ids {
            eos_tokens.extend(ids);
        }

        tracing::info!(
            "CandleProvider: {} EOS tokens ({heuristic_count} heuristic, {} declared, {} stop \
             marker(s)), max_new_tokens={max_new_tokens}",
            eos_tokens.len(),
            declared_eos_ids.len(),
            stop_marker_ids.as_ref().map_or(0, Vec::len),
        );

        // Vision is wired only when the model actually took a tower *and* a
        // `vision_config` came through. The marker strings are resolved from
        // this tokenizer by their well-known Gemma 4 ids so a divergent
        // tokenizer fails loudly here rather than silently mis-tokenizing the
        // prompt.
        let vision = vision_config
            .filter(|_| model.accepts_image_features())
            .and_then(|vc| match VisionSupport::build(&tokenizer, &vc) {
                Ok(v) => {
                    tracing::info!(
                        "CandleProvider: Gemma 4 vision enabled (soft token {:?}, {} max soft tokens/image)",
                        v.soft_token,
                        vc.default_output_length,
                    );
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!("CandleProvider: vision tower loaded but disabled: {e}");
                    None
                }
            });

        Self {
            model: RefCell::new(model),
            cached: RefCell::new(Vec::new()),
            checkpoint: RefCell::new(None),
            tokenizer,
            params,
            eos_tokens,
            max_new_tokens,
            renderer,
            profile,
            context_window,
            stop_marker_ids,
            vision,
            staged_image_features: RefCell::new(None),
        }
    }

    /// Encode `prompt`, run generation, return the raw generated token IDs, the
    /// size the prompt encoded to, and the `(prefill, decode)` split.
    ///
    /// That second number is returned rather than recomputed because a caller
    /// cannot recover it: the prompt string is gone by then, and re-encoding it
    /// would be a guess at what this tokenizer did.
    ///
    /// The two durations are returned raw rather than as a [`crate::llm::Timing`] so the
    /// token counts that price them are attached in exactly one place —
    /// [`TokenUsage::timed`] — and cannot drift from the counts reported beside
    /// them.
    ///
    /// `cancel` is checked between sampled tokens — the only interruption point
    /// a decode loop has. A cancelled generation returns the tokens it managed,
    /// and the caller turns that into `AgentError::Cancelled` rather than
    /// handing a truncated reply to the model.
    fn run_generate_ids(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<(Vec<u32>, usize, usize, Duration, Duration)> {
        // Timed in two halves because they scale differently and a single
        // average hides which one a change actually moved: prefill is one
        // forward over the whole prompt, decode is one forward per token.
        //
        // The clock starts here, before tokenization and before the model is
        // borrowed, so `prefill` means the same thing on both local backends:
        // everything between entering the provider and the first token. Both of
        // those are part of the wait, and a TTFT that omits them is not one
        // anybody experiences.
        let started = Instant::now();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        tracing::info!("CandleProvider: prompt_tokens={}", prompt_tokens.len());
        if self.vision.is_some() {
            let n = count_image_soft_tokens(&prompt_tokens);
            if n > 0 {
                // The count-vs-staged-rows check itself lives in
                // `stage_pending_image_features`, after the reuse split.
                tracing::debug!("CandleProvider: prompt carries {n} image soft token id(s)");
            }
        }

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut model = self.model.borrow_mut();

        // How much of this prompt the cache already holds. One token is left to
        // evaluate whatever happens: the sampler reads the logits of the last
        // position *forwarded*, so a fully cached prompt would produce none.
        let reuse = self.reusable_prefix(model.as_mut(), &prompt_tokens);
        // Image features are staged only now, *after* the reuse split is known:
        // the rows whose markers sit inside the reused prefix are already in
        // the KV cache and must not be injected again at the suffix's markers.
        let reuse = self.stage_pending_image_features(model.as_mut(), &prompt_tokens, reuse);
        let evaluated = prompt_tokens.len() - reuse;
        let mut first_token_at: Option<Instant> = None;
        // Per-token, not just the total: a single stall (a buffer pool growing, a
        // page-in) averages out to a plausible-looking rate over a short run and
        // would send you optimising the steady state instead of the stall.
        let mut token_times: Vec<f64> = Vec::new();
        let mut last_token_at = started;
        let mut stream = StreamingReply::default();
        // A prompt ending in a dangling `<think>` means the model's own output
        // opens mid-reasoning; prepend the opener so `stream_reply` can see it.
        let think_prefix = if crate::streaming::prompt_prefills_thinking(prompt) {
            "<think>"
        } else {
            ""
        };
        let (_, checkpoint) = generate_reusing(
            model.as_mut(),
            &prompt_tokens,
            reuse,
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
                    return ControlFlow::Break(());
                }

                // Decode the reply-so-far once — both the stop check (for a
                // profile whose markers didn't resolve to single ids) and the
                // delta stream need it, and it is the loop's one non-trivial
                // per-token cost, so it is skipped when neither wants it.
                let needs_stop_scan = self.stop_marker_ids.is_none();
                let wants_stream = on_delta.is_some() && !stream.frozen;
                let decoded = if needs_stop_scan || wants_stream {
                    self.tokenizer.decode(&generated_ids, false).ok()
                } else {
                    None
                };

                if wants_stream {
                    if let Some(text) = &decoded {
                        let raw = format!("{think_prefix}{text}");
                        if let Some(visible) = self.profile.stream_reply(&raw) {
                            if let Some(chunk) = stream.advance(&visible, false) {
                                if let Some(cb) = &mut on_delta {
                                    cb(chunk);
                                }
                            }
                        }
                    }
                }

                // Stop where this model's own profile says a turn ends — e.g.
                // Gemma 4 closing a tool call, which must run rather than have
                // its result hallucinated. When `stop_marker_ids` resolved
                // (ADR 0003 step 5) the marker ids are already folded into
                // `self.eos_tokens`, so `generate` itself stops there and this
                // check is redundant — skipped entirely. Unresolved (`None`,
                // the default for every profile but Gemma 4) falls back to
                // decoding the whole reply so far and checking
                // `profile.stops_generation()` on the text, the same check
                // `llm_local.rs` makes.
                if needs_stop_scan
                    && decoded
                        .as_deref()
                        .map(|text| self.profile.stops_generation(text))
                        .unwrap_or(false)
                {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            },
        )
        .map_err(|e| anyhow::anyhow!("generate error: {e}"))?;

        // Flush the tail held back while it might have been a marker: generation
        // is done, so what is left is answer. The `TurnCompleted` message still
        // carries the whole thing, but this lands the last words a beat sooner.
        if let Some(cb) = &mut on_delta {
            if !stream.frozen {
                if let Ok(text) = self.tokenizer.decode(&generated_ids, false) {
                    let raw = format!("{think_prefix}{text}");
                    if let Some(visible) = self.profile.stream_reply(&raw) {
                        if let Some(chunk) = stream.advance(&visible, true) {
                            cb(chunk);
                        }
                    }
                }
            }
        }

        // Record what the cache now holds — prompt *and* what was generated,
        // since the next prompt renders both and a longer match is a shorter
        // evaluation. Checked against the cache's own length rather than
        // assumed: a record that leads the cache is the one failure that
        // produces confident wrong logits, so a disagreement drops the record
        // and costs the next call its reuse instead.
        self.remember(model.as_mut(), &prompt_tokens, &generated_ids, checkpoint);

        tracing::info!(
            "CandleProvider: prompt {} tokens (evaluated {evaluated}), generated {}",
            prompt_tokens.len(),
            generated_ids.len(),
        );

        report_generation_rate(
            model.device(),
            evaluated,
            generated_ids.len(),
            started,
            first_token_at,
            &mut token_times,
        );
        // Same split the log line reports, handed to the caller so a frontend
        // can show it instead of the user having to turn on tracing. With no
        // token sampled there is no boundary between the halves, so the whole
        // elapsed time is charged to prefill — which is where the work went.
        let (prefill, decode) = match first_token_at {
            Some(first) => (first.duration_since(started), first.elapsed()),
            None => (started.elapsed(), Duration::ZERO),
        };
        Ok((
            generated_ids,
            prompt_tokens.len(),
            evaluated,
            prefill,
            decode,
        ))
    }

    /// How many leading tokens of `prompt` this model's cache already holds,
    /// after rolling it back to exactly that point.
    ///
    /// Returns 0 when there is nothing to reuse, when the model does not expose
    /// its cache, or when the rollback is refused — `ModelCache::rewind` decides
    /// that and leaves the cache untouched when it says no, so the caller can
    /// simply start cold.
    fn reusable_prefix(&self, model: &mut dyn CausalLM, prompt: &[u32]) -> usize {
        let cached = self.cached.borrow();
        // One token must be left to forward: the sampler reads the logits of the
        // last position evaluated.
        let shared = cached
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len().saturating_sub(1));
        if shared == 0 {
            return 0;
        }
        let checkpoint = self.checkpoint.borrow();
        let Some(cache) = model.cache() else {
            return 0;
        };
        // A recurrent layer can only go back to where a checkpoint was taken, so
        // that is the target even when the prompts agree past it — which they
        // routinely do, because the re-rendered assistant turn usually starts
        // with the same tokens the model generated. Asking for the longer prefix
        // instead gets the whole rewind refused and costs a full re-evaluation.
        let target = if cache.needs_checkpoint() {
            match checkpoint.as_ref().map(|c| c.len()) {
                Some(len) if len > 0 && len <= shared => len,
                _ => return 0,
            }
        } else {
            shared
        };
        match cache.rewind(target, checkpoint.as_ref()) {
            Ok(true) => {
                // Read the length back rather than returning `target`: what the
                // caller forwards is positioned against what the cache actually
                // holds, and the two disagreeing is not subtle — it is
                // `broadcast_add` failing on a mask of one length against scores
                // of another, which is how an earlier version of this was found.
                let held = cache.len();
                tracing::debug!(
                    "candle KV cache: reusing {held}/{} prompt tokens ({shared} shared)",
                    prompt.len(),
                );
                held
            }
            Ok(false) => {
                tracing::debug!(
                    "candle KV cache: rewind to {target} refused — re-evaluating the prompt"
                );
                0
            }
            Err(e) => {
                tracing::warn!("candle KV cache: rewind failed ({e}); re-evaluating the prompt");
                cache.reset();
                0
            }
        }
    }

    /// Record what the cache holds after a call, or forget it.
    ///
    /// The record is **cut to the cache's own length** rather than assumed to be
    /// prompt-plus-generated, because those are not the same thing: the decode
    /// loop samples a token from the last forward and stops, so the final token
    /// it hands back was never fed back in — the cache holds one fewer than the
    /// caller has. Requiring them to be equal instead read as a disagreement on
    /// every single call, which switched reuse off entirely while leaving it
    /// looking enabled.
    ///
    /// Cutting to `len` also keeps the invariant in its one safe direction: the
    /// record may lag the cache, never lead it. A record that leads it makes the
    /// next call reuse tokens the model never saw.
    fn remember(
        &self,
        model: &mut dyn CausalLM,
        prompt: &[u32],
        generated: &[u32],
        checkpoint: Option<gallium_core::CacheCheckpoint>,
    ) {
        let held = model.cache().map_or(0, |c| c.len());
        let full: Vec<u32> = prompt.iter().chain(generated).copied().collect();
        let mut cached = self.cached.borrow_mut();
        cached.clear();
        // `held > full.len()` cannot come from this loop; it would mean the cache
        // holds tokens this call did not put there, so forget everything rather
        // than describe a cache nobody understands.
        if held > 0 && held <= full.len() {
            cached.extend_from_slice(&full[..held]);
            *self.checkpoint.borrow_mut() = checkpoint;
        } else {
            *self.checkpoint.borrow_mut() = None;
        }
    }

    /// Convenience: generate and decode with skip_special=false (for `profile.clean_reply` / `profile.tool_calls`).
    ///
    /// Also reports what it cost. The counts are exact rather than estimated —
    /// they are the tokens this tokenizer produced and this loop sampled — which
    /// is what makes them usable as a context gauge rather than as a hint.
    fn run_generate(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
        on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<(String, Vec<u32>, TokenUsage)> {
        let (ids, prompt_tokens, evaluated, prefill, decode) =
            self.run_generate_ids(prompt, cancel, on_delta)?;
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
        // `evaluated` is what was forwarded; the rest of the prompt came from
        // the model's own KV cache (`reusable_prefix`). candle has no checkpoint
        // path, so reuse is either a clean prefix hit or a cold start.
        let reused = input.saturating_sub(evaluated as u64);
        let kv = Some(if reused > 0 {
            crate::llm::KvProvenance::SlotReuse {
                reused_tokens: reused,
            }
        } else {
            crate::llm::KvProvenance::FreshContext
        });
        let mut usage = TokenUsage::timed_partial_prefill(
            input,
            output,
            input + output,
            evaluated as u64,
            prefill,
            decode,
        );
        usage.kv = kv;
        Ok((raw, ids, usage))
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

/// Whether a vocabulary token's literal string looks like a turn/generation
/// terminator, for `CandleProvider::new`'s heuristic EOS set.
///
/// Exact matches, or `contains` only where the pattern is distinctive enough
/// that an ordinary word-piece can't produce it by accident. `"eos"` on its
/// own was tried here and pulled in `▁videos`, `ideos`, `▁homeostasis`, and
/// `▁vídeos` from a real Gemma 4 vocabulary — a candle reply would stop dead
/// on the word "videos". `"<eos>"` (with the angle brackets) is what was
/// actually meant, and the other patterns below were already this specific;
/// only the bare `"eos"` case was the false-positive source.
///
/// This is a heuristic of last resort — `CandleProvider::new` also extends
/// its EOS set with `declared_eos_ids`, the GGUF/config's own stated token
/// id(s), which is what actually names a model's real EOS (see that call
/// site's comment for why a string can't).
fn is_eos_like(token: &str) -> bool {
    // NOTE: do NOT match the bare "<|end|>" token — in Harmony it's a
    // message/channel separator (analysis → commentary → final), not a
    // turn terminator. The turn ends on "<|return|>" or "<|call|>".
    token == "<eos>"
        || token == "<|eos|>"
        || token == "<|endoftext|>"
        || token.contains("</s>")
        || token.contains("<end_of_turn>")
        || token.contains("<|im_end|>")
        || token == "<|call|>" // Harmony tool call terminator
        || token == "<|return|>" // Harmony end-of-turn terminator
}

#[cfg(test)]
mod eos_heuristic_tests {
    use super::is_eos_like;

    #[test]
    fn known_eos_spellings_match() {
        for tok in [
            "<eos>",
            "<|eos|>",
            "<|endoftext|>",
            "</s>",
            "<end_of_turn>",
            "<|im_end|>",
            "<|call|>",
            "<|return|>",
        ] {
            assert!(is_eos_like(tok), "{tok:?} should be EOS-like");
        }
    }

    /// The exact false positives a real Gemma 4 vocabulary produced under
    /// the old bare `contains("eos")` check.
    #[test]
    fn ordinary_words_containing_eos_do_not_match() {
        for tok in ["▁videos", "ideos", "▁homeostasis", "▁vídeos"] {
            assert!(!is_eos_like(tok), "{tok:?} should not be EOS-like");
        }
    }

    /// Harmony's channel separator is not a turn terminator — see the
    /// comment on `is_eos_like`.
    #[test]
    fn harmony_channel_separator_does_not_match() {
        assert!(!is_eos_like("<|end|>"));
    }

    /// Gemma 4's real terminator has none of these literal spellings — it is
    /// only ever caught via `declared_eos_ids`, not this heuristic. Pinned
    /// here so nobody "fixes" this by adding `<turn|>` to the pattern list
    /// instead of trusting the declared id.
    #[test]
    fn gemma4_turn_marker_is_not_caught_by_the_heuristic() {
        assert!(!is_eos_like("<turn|>"));
    }
}

/// Resolve `profile.stop_markers()` against this tokenizer's vocabulary —
/// candle's counterpart to `llm_local.rs::resolve_stop_markers`. `Some` only
/// when *every* marker tokenizes to exactly one id, for the same
/// all-or-nothing reason: a caller that kept two ids and quietly dropped a
/// third would stop early on the two it kept, never knowing the profile
/// expected a third.
///
/// `encode(marker, false)` — no added special tokens — so a marker like
/// `<tool_call|>` resolves through the tokenizer's own added-vocabulary
/// entry rather than being wrapped in a template the way a real prompt is.
fn resolve_stop_markers(
    tokenizer: &Tokenizer,
    profile: &'static dyn ModelProfile,
) -> Option<Vec<u32>> {
    let markers = profile.stop_markers();
    if markers.is_empty() {
        return None;
    }
    let mut ids = Vec::with_capacity(markers.len());
    for marker in markers {
        match tokenizer.encode(*marker, false) {
            Ok(encoding) if encoding.get_ids().len() == 1 => ids.push(encoding.get_ids()[0]),
            Ok(encoding) => {
                tracing::warn!(
                    "CandleProvider: model profile '{}': stop marker {marker:?} tokenizes to \
                     {} ids, not 1 — falling back to stops_generation's decoded-text check \
                     for this model (ADR 0003 step 5)",
                    profile.name(),
                    encoding.get_ids().len(),
                );
                return None;
            }
            Err(e) => {
                tracing::warn!(
                    "CandleProvider: model profile '{}': stop marker {marker:?} failed to \
                     tokenize ({e}) — falling back to stops_generation's decoded-text check \
                     for this model",
                    profile.name(),
                );
                return None;
            }
        }
    }
    tracing::info!(
        "CandleProvider: model profile '{}': {} stop marker(s) resolved to token ids, \
         replacing the decoded-text check",
        profile.name(),
        ids.len(),
    );
    Some(ids)
}

impl LlmProvider for CandleProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.renderer.format_prompt(messages);
        tracing::debug!("CandleProvider prompt ({} chars)", prompt.len());
        let (raw, _ids, _usage) = self.run_generate(&prompt, &CancellationToken::new(), None)?;
        Ok(self.profile.clean_reply(&raw))
    }

    /// Every profile has a fallback (gallium's own JSON-prose protocol at
    /// minimum), so this is unconditional — matching `llm_local.rs`, which
    /// answers the same way for the same reason.
    fn supports_tools(&self) -> bool {
        true
    }

    /// The window the loaded model was configured for, from its own metadata.
    fn context_window(&self) -> Option<u32> {
        self.context_window
    }

    /// Same profile instance `llm_local.rs` uses for the identical arch (see
    /// the module doc), so a family that has earned a preamble gets it
    /// regardless of which local backend loaded the model.
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
        self.generate_response(messages, tools, cancel, None)
    }

    /// Streams answer fragments to `on_delta` as they decode — see
    /// [`StreamingReply`] for the filtering. This is the only provider that
    /// overrides the default (which does not stream).
    fn chat_with_tools_streaming(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        cancel: &CancellationToken,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<LlmResponse> {
        self.generate_response(messages, tools, cancel, Some(on_delta))
    }
}

impl CandleProvider {
    fn generate_response(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        cancel: &CancellationToken,
        on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<LlmResponse> {
        // Two ways in, decided by whether this model has a vision tower. A
        // text-only candle model refuses any attachment, exactly as before.
        // A Gemma 4 multimodal model runs its images through the tower and
        // stages the soft tokens for the prefill `forward` to inject; the
        // prompt then carries the expanded `<image_soft_token>` runs so the
        // token positions line up with the staged features.
        let staged;
        let messages: &[ChatMessage] = match &self.vision {
            None => {
                crate::llm::reject_media(messages, "the candle backend")?;
                messages
            }
            Some(vision) => {
                crate::llm::reject_audio(messages, "the candle backend")?;
                staged = self.stage_images(vision, messages)?;
                &staged
            }
        };

        let prompt = self.renderer.format_prompt_with_tools(messages, tools);
        tracing::debug!("CandleProvider tool prompt ({} chars)", prompt.len());
        // Decode with skip_special=false so tool-call parsing can see all markers.
        let (raw, ids, mut usage) = self.run_generate(&prompt, cancel, on_delta)?;
        // Hash the render the model was handed (docs/TODO.md §9.2).
        usage.prompt_sha256 = Some(crate::llm::prompt_digest(&prompt));

        let calls = self.profile.tool_calls(&raw, tools);
        if !calls.is_empty() {
            tracing::info!(
                "CandleProvider: {} tool call(s): {}",
                calls.len(),
                calls
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            // Usage on this arm too, not only on the text one: a tool-using turn
            // is where the prompt actually grows, so reporting only the final
            // answer would gauge the context at its smallest.
            // From the same raw decode the calls were parsed out of, before
            // anything strips it — see #177.
            let reasoning = self.profile.reasoning_content(&raw);
            return Ok(LlmResponse::ToolCalls {
                calls,
                usage: Some(usage),
                reasoning,
                // The decode exactly as `profile.tool_calls` saw it, before any
                // stripping — plus the token ids it came from, so a §9.1
                // analysis can tell a mangled decode from a mangled generation
                // (docs/TODO.md §9.1).
                raw: Some(crate::llm::RawGeneration::with_token_ids(raw.clone(), ids)),
            });
        }

        // No tool call — extract response text.
        Ok(LlmResponse::Text {
            content: self.profile.clean_reply(&raw),
            reasoning: self.profile.reasoning_content(&raw),
            usage: Some(usage),
            raw: Some(crate::llm::RawGeneration::with_token_ids(raw, ids)),
        })
    }

    /// Run every attached image through the vision tower, stage the projected
    /// soft tokens for the next prefill, and return `messages` with each
    /// image's `<start_of_image>…<end_of_image>` block spliced into its turn.
    ///
    /// Order is the contract: `Gemma4Multimodal::inject_image_features`
    /// replaces the `<image_soft_token>` positions in reading order with the
    /// staged feature rows in row order, so the per-image feature blocks are
    /// concatenated in exactly the order their markers appear in the prompt —
    /// message order, then attachment order within a message.
    fn stage_images(
        &self,
        vision: &VisionSupport,
        messages: &[ChatMessage],
    ) -> Result<Vec<ChatMessage>> {
        let device = self.model.borrow().device().clone();

        // Preprocess + encode every image first (immutable borrow of the model),
        // collecting the per-image feature block and its soft-token count.
        let mut per_message_counts: Vec<Vec<usize>> = Vec::with_capacity(messages.len());
        let mut feature_blocks: Vec<candle_core::Tensor> = Vec::new();
        {
            let model = self.model.borrow();
            for msg in messages {
                let mut counts = Vec::new();
                for media in &msg.media {
                    let MediaContent::Image(img) = media else {
                        continue;
                    };
                    let bytes = base64_decode(&img.base64)?;
                    let processed = vision
                        .processor
                        .process(&bytes, &device)
                        .map_err(|e| anyhow::anyhow!("image preprocessing failed: {e}"))?;
                    let feats = model
                        .encode_image(&processed.pixel_values, &processed.pixel_position_ids)
                        .map_err(|e| anyhow::anyhow!("vision tower failed: {e}"))?;
                    tracing::debug!(
                        "CandleProvider: image → {} soft tokens ({:?})",
                        processed.num_soft_tokens,
                        feats.dims()
                    );
                    counts.push(processed.num_soft_tokens);
                    feature_blocks.push(feats);
                }
                per_message_counts.push(counts);
            }
        }

        if feature_blocks.is_empty() {
            return Ok(messages.to_vec());
        }

        let all_features = candle_core::Tensor::cat(&feature_blocks, 0)
            .map_err(|e| anyhow::anyhow!("concatenating image features: {e}"))?;
        // A quick sanity line: the tower should hand back finite, roughly
        // unit-scale rows. NaN or a blown-up range here means the vision
        // encoder diverged.
        if let Ok(f) = all_features
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        {
            let (mn, mx) = f
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            tracing::debug!(
                "CandleProvider: image features range [{mn:.3}, {mx:.3}], nan={}",
                f.iter().filter(|v| v.is_nan()).count(),
            );
        }
        tracing::info!(
            "CandleProvider: staged {} image soft token(s) across {} image(s)",
            all_features.dim(0).unwrap_or(0),
            feature_blocks.len(),
        );
        // Parked on the provider, not handed to the model: these rows cover
        // every image marker in the prompt, but the prefill may evaluate only
        // a suffix of it — `run_generate_ids` trims to that suffix once it
        // knows the KV-reuse split (`stage_pending_image_features`).
        *self.staged_image_features.borrow_mut() = Some(all_features);

        // Splice the marker blocks into each message that carried images.
        let mut out = messages.to_vec();
        for (msg, counts) in out.iter_mut().zip(per_message_counts) {
            if counts.is_empty() {
                continue;
            }
            let mut blocks = String::new();
            for n in counts {
                blocks.push_str(&vision.marker_block(n));
                blocks.push('\n');
            }
            // In practice only a user turn ever carries media
            // (`ChatMessage::user_with_media`); prepending to its content puts
            // the image where the reference template does.
            msg.content = format!("{blocks}{}", msg.content);
            msg.media.clear();
            tracing::debug!(
                "CandleProvider: expanded user turn head: {:?}",
                &msg.content.chars().take(60).collect::<String>()
            );
        }
        Ok(out)
    }

    /// Hand the staged image features to the model, trimmed to the suffix the
    /// coming prefill will actually evaluate. Returns the (possibly forfeited)
    /// reuse length.
    ///
    /// `stage_images` encodes every image in the history, so the staged rows
    /// cover every `<image_soft_token>` in the *whole* prompt — but
    /// `Gemma4Multimodal::inject_image_features` matches rows to the image
    /// positions of the chunk it is handed, first row to first position, and
    /// with KV-prefix reuse that chunk is only the fresh suffix. The rows whose
    /// markers the cache already holds were injected by the call that evaluated
    /// them and must be dropped here first: without the trim, the second image
    /// of a conversation was injected with the *first* image's rows — silently,
    /// which is exactly the "model answers about a picture it never saw"
    /// failure the multimodal path promises to refuse rather than produce.
    ///
    /// Counts that do not line up (a literal soft-token string typed into the
    /// prompt, a template change mid-conversation) forfeit reuse instead of
    /// guessing: all rows are staged and the whole prompt re-evaluated —
    /// `generate_reusing` resets the cache on `reuse == 0` — which is the
    /// pre-reuse behavior and never wrong about which image is which.
    fn stage_pending_image_features(
        &self,
        model: &mut dyn CausalLM,
        prompt_tokens: &[u32],
        reuse: usize,
    ) -> usize {
        let Some(feats) = self.staged_image_features.borrow_mut().take() else {
            return reuse;
        };
        let total = feats.dim(0).unwrap_or(0);
        match image_rows_for_suffix(prompt_tokens, reuse, total) {
            Some((_, 0)) => {
                // Every marker sits inside the reused prefix: the features are
                // already in the KV cache. Staging nothing also spares the
                // prefill `inject_image_features`'s host round-trip of the
                // whole suffix embedding for positions it does not have.
                tracing::debug!(
                    "CandleProvider: all {total} staged image row(s) already in the reused prefix"
                );
                reuse
            }
            Some((skip, take)) => match feats.narrow(0, skip, take) {
                Ok(trimmed) => {
                    if skip > 0 {
                        tracing::debug!(
                            "CandleProvider: dropping {skip} cached image row(s), staging {take}"
                        );
                    }
                    model.set_image_features(trimmed);
                    reuse
                }
                Err(e) => {
                    // The bounds were just checked, so this is unreachable in
                    // practice — but a wrong picture is the one failure this
                    // path must not produce, so fall back to the full prefill.
                    tracing::warn!(
                        "CandleProvider: trimming staged image features failed ({e}); re-evaluating the prompt"
                    );
                    model.set_image_features(feats);
                    0
                }
            },
            None => {
                tracing::warn!(
                    "CandleProvider: prompt carries {} image soft token(s) but {total} feature row(s) \
                     are staged — forfeiting KV reuse and re-evaluating the prompt",
                    count_image_soft_tokens(prompt_tokens),
                );
                model.set_image_features(feats);
                0
            }
        }
    }
}

fn count_image_soft_tokens(tokens: &[u32]) -> usize {
    tokens
        .iter()
        .filter(|&&t| t == GEMMA4_IMAGE_SOFT_TOKEN_ID)
        .count()
}

/// Which staged feature rows the prefill of `prompt_tokens[reuse..]` needs:
/// `Some((skip, take))` — drop `skip` leading rows (their markers are inside
/// the reused prefix), stage the next `take`. `None` when the marker count and
/// the staged row count disagree, in which case the caller forfeits reuse.
fn image_rows_for_suffix(
    prompt_tokens: &[u32],
    reuse: usize,
    total_rows: usize,
) -> Option<(usize, usize)> {
    let cached = count_image_soft_tokens(&prompt_tokens[..reuse]);
    let fresh = count_image_soft_tokens(&prompt_tokens[reuse..]);
    (cached + fresh == total_rows).then_some((cached, fresh))
}

/// Base64-decode an image payload (standard alphabet, as `input.rs` encodes it).
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .map_err(|e| anyhow::anyhow!("invalid base64 image data: {e}"))
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

    /// The renderer for this family's prompt format. Prompt rendering is the
    /// one thing that legitimately differs per engine (candle has no jinja
    /// template and must build the format itself), so this stays a
    /// `Box<dyn PromptRenderer>` rather than a shared static — every renderer
    /// but `Lfm2Protocol` carries per-load state driven by `reasoning_effort`
    /// below.
    ///
    /// Reuses the same `ModelProfile::reasoning_params` mapping the
    /// llama.cpp backend uses (ADR 0003 step 3-b already shares `Gemma4`/
    /// `GptOss`/`Qwen3` between the two backends for parsing; this extends
    /// that sharing to reasoning control) rather than re-deriving
    /// per-family logic here. `unwrap_or` on each `.thinking`/`.effort_text`
    /// covers a profile that has nothing to say for the *other* axis (e.g.
    /// `GptOss::reasoning_params` never sets `.thinking`), not a real "no
    /// opinion" case for the field this call site actually reads.
    ///
    /// **A family may set both, and then both must be read.** `Qwen3` does now,
    /// and reading only `.thinking` made `Medium` through `Max` four distinct
    /// prompts on llama.cpp — whose template consumes `reasoning_effort` — and
    /// one prompt here, from a single config value. A setting that means
    /// different things depending on which engine loaded the model is worse
    /// than one that does nothing, because only one of the two is visible.
    /// `llm_local_templates::both_backends_render_the_same_reasoning_instruction`
    /// is what holds the two together; nothing else can, since one renders the
    /// model's own template and the other hand-transcribes it.
    fn renderer(self, reasoning_effort: Option<ReasoningEffort>) -> Box<dyn PromptRenderer> {
        match self {
            Arch::GptOss => {
                let effort_text = reasoning_effort
                    .and_then(|e| GptOss.reasoning_params(e).effort_text)
                    .unwrap_or("medium");
                Box::new(HarmonyProtocol::with_effort(effort_text))
            }
            Arch::Qwen35 => {
                // Both axes: this family sets `thinking` *and* `effort_text`,
                // and reading only the first made `Medium` through `Max` one
                // prompt here and four on llama.cpp, from the same config.
                let params = reasoning_effort.map(|e| Qwen3.reasoning_params(e));
                Box::new(QwenProtocol::with_reasoning(
                    params.and_then(|p| p.thinking).unwrap_or(true),
                    params.and_then(|p| p.effort_text),
                ))
            }
            Arch::Lfm2 => Box::new(Lfm2Protocol),
            Arch::Gemma4 => {
                let thinking = reasoning_effort
                    .map(|e| Gemma4.reasoning_params(e).thinking.unwrap_or(false))
                    .unwrap_or(false);
                if thinking {
                    Box::new(GemmaProtocol::with_thinking())
                } else {
                    Box::new(GemmaProtocol::new())
                }
            }
        }
    }

    /// Which family's wire rules read this model's output — the same shared
    /// instance `llm_local.rs` selects for the identical GGUF `general.architecture`.
    ///
    /// A direct match, not a `profile::detect` call: `Arch::from_hint` is
    /// already the total, exact mapping from a model's hint to one of these
    /// four families (candle can only load weights for four; there is no
    /// candle-side `Generic` to fall back to), so a second detection pass
    /// could only disagree with this one, never usefully refine it. An
    /// unsupported model is refused at `Arch::from_hint` — the same outcome
    /// ADR 0003 describes for a profile with no renderer, by the mechanism
    /// that was already here. `[llm] profile` / `GALLIUM_PROFILE` stay
    /// llama.cpp-only for the same reason: they exist to disambiguate among
    /// llama.cpp's six families from loose signals, which this exact mapping
    /// has no ambiguity to resolve.
    fn profile(self) -> &'static dyn ModelProfile {
        match self {
            Arch::GptOss => &GptOss,
            Arch::Qwen35 => &Qwen3,
            Arch::Gemma4 => &Gemma4,
            Arch::Lfm2 => &Lfm2,
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

/// What `load_candle_provider`'s format match resolves before it builds the
/// provider: the arch, the model, its tokenizer, the context window and
/// declared EOS ids from the file's metadata, and a Gemma 4 `vision_config`
/// when the checkpoint has one.
type LoadedCandleModel = (
    Arch,
    Box<dyn CausalLM>,
    Tokenizer,
    Option<u32>,
    Vec<u32>,
    Option<gallium_models::gemma4_vision::Gemma4VisionConfig>,
);

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
/// `reasoning_effort` drives each family's thinking control via
/// `Arch::renderer` (see there) — `[llm] reasoningEffort` / `REASONING_EFFORT`,
/// same config key the llama.cpp backend reads.
///
/// Env knobs: `GALLIUM_TOKENIZER_REPO` (tokenizer.json source repo),
/// `GALLIUM_DTYPE` (`f16`/`bf16`/`f32`, safetensors only, default `f16`).
#[allow(clippy::too_many_arguments)]
pub fn load_candle_provider(
    model_path: &str,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: u32,
    tokenizer_path: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<CandleProvider> {
    use candle_core::DType;

    // Metal on macOS, CUDA where it was built in, else CPU — `GALLIUM_DEVICE`
    // overrides, and naming a device that is not there fails rather than quietly
    // running on the CPU.
    let device = gallium_core::resolve_device(std::env::var("GALLIUM_DEVICE").ok().as_deref())?;
    tracing::info!("Candle device: {}", gallium_core::device_name(&device));
    // `topK` / `topP` reach this engine too. They used to be llama.cpp-only —
    // set in a config, silently inert here — which is the worst shape for a
    // setting: it works on one backend and does nothing on the other, and
    // nothing says so.
    let params = SamplingParams {
        temperature: temperature.unwrap_or(0.7),
        top_k: top_k.map(|k| k as usize),
        top_p,
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
    //
    // `declared_eos_ids` is the model file's own stated EOS/EOT id(s) — GGUF
    // `tokenizer.ggml.eos_token_id` / `eot_token_id`, or safetensors
    // `config.json`'s `eos_token_id` — read here because this is the one
    // place both formats' metadata is in scope. `CandleProvider::new` folds
    // these into its EOS set alongside (not instead of) the string heuristic;
    // see its call site for why a declared id is what actually matters.
    let (arch, model, tokenizer, context_window, declared_eos_ids, vision_config): LoadedCandleModel =
        match Format::detect(model_path) {
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
            // Not per-architecture — llama.cpp reads these two keys the same
            // way for every model (`llama-arch.cpp`'s LLM_KV_TOKENIZER_EOS_ID
            // / _EOT_ID). Gemma 4's real terminator (`<turn|>`) is the
            // `eos_token_id` entry here, which is what makes this the fix
            // rather than the string heuristic below.
            let declared_eos_ids: Vec<u32> = [
                metadata.get_u32("tokenizer.ggml.eos_token_id").ok(),
                metadata.get_u32("tokenizer.ggml.eot_token_id").ok(),
            ]
            .into_iter()
            .flatten()
            .collect();

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
            // No vision on the GGUF candle path — `gemma4_q` is text-only, and
            // the projector lives in a separate `mmproj` GGUF the llama.cpp
            // backend handles.
            (arch, model, tokenizer, window, declared_eos_ids, None)
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
            // HF configs spell this either a single id or a list — Llama 3's
            // `config.json` has three (regular EOS plus two turn/message
            // terminators), for instance.
            let declared_eos_ids: Vec<u32> = match text.get("eos_token_id") {
                Some(serde_json::Value::Array(ids)) => ids
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .filter_map(|n| u32::try_from(n).ok())
                    .collect(),
                Some(v) => v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .into_iter()
                    .collect(),
                None => Vec::new(),
            };
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
                Arch::Gemma4 if full.get("vision_config").is_some() => {
                    // Multimodal checkpoint: text weights under
                    // `model.language_model.*`, the tower under
                    // `model.vision_tower.*` / `model.embed_vision.*`.
                    let cfg: gallium_models::gemma4_vision::Gemma4MultimodalConfig =
                        serde_json::from_value(full.clone())
                            .map_err(|e| anyhow::anyhow!("Gemma4 multimodal config error: {e}"))?;
                    Box::new(gallium_models::gemma4_vision::Gemma4Multimodal::load(
                        &cfg, vb, &device,
                    )?)
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
            // Parsed for the provider's image preprocessor; `None` unless this
            // is a Gemma 4 checkpoint that actually carries a `vision_config`.
            let vision_config = full
                .get("vision_config")
                .filter(|_| arch == Arch::Gemma4)
                .map(|vc| serde_json::from_value(vc.clone()))
                .transpose()
                .map_err(|e| anyhow::anyhow!("Gemma4 vision_config error: {e}"))?;
            (
                arch,
                model,
                tokenizer,
                window,
                declared_eos_ids,
                vision_config,
            )
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
        arch.renderer(reasoning_effort),
        arch.profile(),
        context_window,
        &declared_eos_ids,
        vision_config,
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

    // ── image_rows_for_suffix ────────────────────────────────────────────────
    //
    // The trim that keeps KV-prefix reuse honest about which image is which:
    // rows whose markers sit inside the reused prefix are already in the cache
    // and must be skipped, or the suffix's markers get an earlier image's rows.

    const IMG: u32 = GEMMA4_IMAGE_SOFT_TOKEN_ID;

    #[test]
    fn cold_prefill_stages_every_row() {
        let prompt = [1, IMG, IMG, 2, 3];
        assert_eq!(image_rows_for_suffix(&prompt, 0, 2), Some((0, 2)));
    }

    #[test]
    fn a_second_image_skips_the_cached_first_images_rows() {
        // Turn 1's image (2 tokens) is inside the reused prefix; turn 2's
        // image (3 tokens) is in the fresh suffix and must get rows 2..5,
        // not 0..3.
        let prompt = [1, IMG, IMG, 2, 3, IMG, IMG, IMG, 4];
        assert_eq!(image_rows_for_suffix(&prompt, 4, 5), Some((2, 3)));
    }

    #[test]
    fn a_reuse_boundary_inside_a_marker_run_splits_the_rows_there() {
        let prompt = [1, IMG, IMG, IMG, 2];
        assert_eq!(image_rows_for_suffix(&prompt, 2, 3), Some((1, 2)));
    }

    #[test]
    fn a_fully_cached_image_stages_nothing() {
        // Text-only continuation of an image turn: markers all in the prefix.
        let prompt = [1, IMG, IMG, 2, 3, 4];
        assert_eq!(image_rows_for_suffix(&prompt, 5, 2), Some((2, 0)));
    }

    #[test]
    fn a_marker_count_mismatch_forfeits_reuse() {
        // More markers than staged rows (a literal soft-token string typed
        // into the prompt) — the caller must fall back to a full prefill.
        let prompt = [1, IMG, IMG, IMG, 2];
        assert_eq!(image_rows_for_suffix(&prompt, 0, 2), None);
        // Fewer markers than rows, likewise.
        assert_eq!(image_rows_for_suffix(&prompt, 0, 4), None);
    }
}
