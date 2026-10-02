use std::ops::ControlFlow;

use candle_core::{DType, Device, Result, Tensor};

use crate::sampling::{sample, SamplingParams};

/// Core trait for causal language models.
/// All models implement this for generation.
pub trait CausalLM {
    /// Forward pass: token IDs (batch, seq_len) -> logits (batch, vocab_size).
    /// `pos` is the starting position for this chunk (for KV cache offset).
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor>;

    /// Reset internal caches (start a new conversation).
    fn reset(&mut self);

    /// This model's cache, when it keeps one of the standard shape.
    ///
    /// Returning `Some` is what opts a model into **reuse across calls**: with
    /// it, [`generate_reusing`] can be handed a prompt whose first `reuse`
    /// tokens the cache already holds and evaluate only the rest, which is the
    /// difference between an agent turn costing one prefill and costing one per
    /// ReAct iteration.
    ///
    /// Default `None`, so a model nobody has checked keeps today's behaviour —
    /// every call re-evaluates its whole prompt — rather than silently reusing a
    /// cache whose layout this crate has assumed.
    fn cache(&mut self) -> Option<&mut crate::ModelCache> {
        None
    }

    /// This model's cache for **VRAM accounting** (issue #343): what the
    /// [`crate::VramLedger`] is attached to, what a refused forward is rolled
    /// back on, and what the context ceiling is planned from. Defaults to
    /// [`Self::cache`]; a model that keeps a standard cache but has not been
    /// verified for reuse across calls returns it here and not there, so it
    /// gets the accounting without the reuse.
    fn ledger_cache(&mut self) -> Option<&mut crate::ModelCache> {
        self.cache()
    }

    /// Device the model lives on.
    fn device(&self) -> &Device;

    // ── Vision (multimodal) ────────────────────────────────────────────────
    //
    // Default no-ops so a text-only model stays a unit struct with one
    // `forward`. A model with a vision tower overrides these three; the
    // provider drives them: `encode_image` once per attached image, then a
    // single `set_image_features` with the concatenated soft tokens, before the
    // prefill `forward` that injects them at the image-token positions.

    /// Whether this model can take image soft tokens via [`Self::set_image_features`].
    fn accepts_image_features(&self) -> bool {
        false
    }

    /// Run the vision tower: preprocessed patches → projected soft tokens in the
    /// language model's embedding space, `[num_soft_tokens, hidden]`.
    ///
    /// `pixel_values`: `[b, num_patches, 3 * patch_size^2]` f32.
    /// `pixel_position_ids`: `[b, num_patches, 2]` i64 `(x, y)`, padding `(-1,-1)`.
    fn encode_image(&self, _pixel_values: &Tensor, _pixel_position_ids: &Tensor) -> Result<Tensor> {
        candle_core::bail!("this model has no vision tower")
    }

    /// Stage image features for the next prefill `forward` to inject. Cleared by
    /// [`Self::reset`] and consumed by the forward pass that uses them.
    fn set_image_features(&mut self, _features: Tensor) {}

    /// Whether image features are staged *right now*, waiting for a prefill
    /// `forward` to scatter them over the image-token positions of the batch it
    /// is handed. When `true`, [`generate_reusing`] does the prefill in one
    /// shot rather than in windows — a split could put the image tokens in a
    /// chunk other than the one that consumes the features. A text turn through
    /// a vision-capable model stages nothing, so it still chunks.
    fn has_staged_image_features(&self) -> bool {
        false
    }

    /// Device bytes one `forward` over `seq_len` tokens at `pos` allocates and
    /// frees again before it returns — activations, attention scores, any
    /// weights it expands — at its peak. [`generate_reusing`] books this on the
    /// cache's [`crate::VramLedger`] around each call, so the expert cache gives
    /// the room up *before* the forward rather than the driver refusing it
    /// midway (issue #343). An estimate, and meant to err high; the safety
    /// margin the ledger's budget was cut with absorbs what it misses.
    ///
    /// Default 0: a model that has not estimated it leaves the margin to cover
    /// it, which is what every model did before the ledger existed.
    fn transient_bytes(&self, _seq_len: usize, _pos: usize) -> usize {
        0
    }
}

/// Smallest prefill window a refused transient booking steps down to (issue
/// #343 stage 2). Below this the per-chunk transient is mostly fixed overhead
/// (weights' dequant scratch, the residual stream), so halving further would
/// buy little room for a lot of extra forwards.
const MIN_PREFILL_CHUNK: usize = 64;

/// Why a [`try_forward`] did not produce logits.
enum ForwardRefusal {
    /// The transient booking was refused **before** the forward ran — the
    /// cache is untouched, so the same span can be retried smaller.
    Transient(candle_core::Error),
    /// The forward itself failed. The cache may hold part of this span.
    Forward(candle_core::Error),
}

/// `model.forward(input, pos)`, with its [`CausalLM::transient_bytes`] booked
/// on the cache's ledger for the duration (when there is one).
fn try_forward(
    model: &mut dyn CausalLM,
    input: &Tensor,
    pos: usize,
) -> std::result::Result<Tensor, ForwardRefusal> {
    let ledger = model.ledger_cache().and_then(|c| c.ledger().cloned());
    // Before a prefill window, settle the budget against what the driver shows
    // (`VramLedger::recalibrate`): pool slack, workspaces — anything nobody
    // books — is taken off it, and the expert cache evicts and trims to match
    // before this window allocates. Learning it only after a call, as the
    // provider does, let a 4k-token prefill's own slack accumulate mid-call
    // and run Qwen3.8-27B out of memory. A decode step's scratch is small and
    // the same every token, so it is not worth the synchronize.
    if let (Some(l), true) = (&ledger, input.dim(1).unwrap_or(1) > 1) {
        if let Some(free) =
            crate::vram::free_device_memory(model.device()).map_err(ForwardRefusal::Forward)?
        {
            l.recalibrate(free);
        }
    }
    let n = input.dim(1).map_err(ForwardRefusal::Forward)?;
    let estimated = model.transient_bytes(n, pos);
    let _transient = match &ledger {
        Some(l) => {
            let booked = if n > 1 {
                l.prefill_booking(pos, estimated)
            } else {
                estimated
            };
            Some(
                l.reserve_transient(booked, "forward transient")
                    .map_err(ForwardRefusal::Transient)?,
            )
        }
        None => None,
    };
    // What the forward really allocated, fed back so the next one is booked
    // for it (`VramLedger::observe_transient`). KV grown during the forward is
    // booked already, so it is taken off.
    let probe = ledger.as_ref().map(|l| {
        (
            crate::vram::TransientProbe::start(model.device()),
            l.persistent(),
        )
    });
    let out = model.forward(input, pos).map_err(ForwardRefusal::Forward);
    let out = out?;
    // A prefill window's scratch is gone now; give its chunks back so the next,
    // larger window's do not stack on top of them.
    if ledger.is_some() && input.dim(1).unwrap_or(1) > 1 {
        crate::vram::trim_default_pool(model.device()).map_err(ForwardRefusal::Forward)?;
    }
    if let (Some(l), Some((probe, kv_before))) = (&ledger, probe) {
        if let Some(peak) = probe.peak() {
            let kv_grown = l.persistent().saturating_sub(kv_before);
            l.observe_transient(n, pos, estimated, peak.saturating_sub(kv_grown));
        }
    }
    Ok(out)
}

/// Two prefill windows of dummy tokens — at position 0 and right after it —
/// with the expert cache closed, then the model reset: run once at load so the
/// VRAM ledger learns real scratch before any request can fill the room it
/// needs. The first window is the expensive one: it also allocates the
/// recurrent states, the first KV buffers and the kernels' workspaces, and it
/// is the one a cache still filling up competes with (Qwen3.8-27B OOM'd
/// there); the second is what every later window scales from. Returns what a
/// first window is now booked at. A no-op without a ledger.
pub fn calibrate_transient(model: &mut dyn CausalLM, token: u32) -> Result<Option<usize>> {
    let Some(ledger) = model.ledger_cache().and_then(|c| c.ledger().cloned()) else {
        return Ok(None);
    };
    let n = match prefill_chunk() {
        0 => 512,
        c => c,
    };
    let device = model.device().clone();
    let ids = Tensor::from_vec(vec![token; n], (1, n), &device)?.to_dtype(DType::U32)?;
    model.reset();
    let result = ledger.with_cache_closed(|| {
        forward_reserved(model, &ids, 0)?;
        forward_reserved(model, &ids, n)
    });
    model.reset();
    result?;
    Ok(Some(ledger.prefill_booking(0, model.transient_bytes(n, 0))))
}

/// After a failed forward at `pos`: put the cache back to exactly `pos`
/// positions, or give it up whole. A forward refused part way — a KV growth
/// the ledger turned down at layer *i* — leaves layers before *i* holding the
/// span and the rest not, a state no prompt describes. A windowed layer that
/// compacted during the span cannot be rolled back by position, so that case
/// resets, which costs the next call its reuse and nothing else. `true` when
/// the cache holds exactly `pos` positions afterwards.
fn roll_back_to(model: &mut dyn CausalLM, pos: usize) -> bool {
    let Some(cache) = model.ledger_cache() else {
        return false;
    };
    if matches!(cache.rewind(pos, None), Ok(true)) {
        return cache.len() == pos;
    }
    cache.reset();
    pos == 0
}

/// [`try_forward`] for the decode loop: any failure rolls the cache back to
/// `pos` before it is returned.
fn forward_reserved(model: &mut dyn CausalLM, input: &Tensor, pos: usize) -> Result<Tensor> {
    try_forward(model, input, pos).map_err(|refusal| {
        if let ForwardRefusal::Forward(_) = refusal {
            roll_back_to(model, pos);
        }
        match refusal {
            ForwardRefusal::Transient(e) | ForwardRefusal::Forward(e) => e,
        }
    })
}

/// The longest context `model` can hold inside its cache's ledger budget, up to
/// `upper`: the most positions whose KV peak ([`crate::ModelCache::peak_bytes_at`])
/// plus one prefill chunk's transient at that depth fits. `None` when there is
/// no ledger or the cache cannot say what it will cost.
///
/// This is what a provider reports as its context window, so compaction — which
/// fires at a fraction of the reported window — trims the conversation *before*
/// a reservation would be refused, the candle counterpart of llama.cpp's
/// `ctx_ceiling`. The expert cache is not counted: it is what gives way.
pub fn vram_context_ceiling(model: &mut dyn CausalLM, upper: usize) -> Option<usize> {
    let chunk = prefill_chunk();
    let ledger = model.ledger_cache()?.ledger()?.clone();
    let budget = ledger.budget();
    let cost = |model: &mut dyn CausalLM, p: usize| -> Option<usize> {
        let kv = model.ledger_cache()?.peak_bytes_at(p)?;
        let n = if chunk == 0 { p } else { chunk.min(p) };
        Some(kv + ledger.prefill_booking(p - n, model.transient_bytes(n, p - n)))
    };
    cost(model, 1)?;
    let fits = |model: &mut dyn CausalLM, p: usize| cost(model, p).is_some_and(|c| c <= budget);
    if fits(model, upper) {
        return Some(upper);
    }
    // Largest `p` in `[0, upper)` that fits; cost is non-decreasing in `p`.
    let (mut lo, mut hi) = (0usize, upper);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if fits(model, mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// Longest fresh-prompt slice fed to `CausalLM::forward` in one prefill call.
/// Default 512; `GALLIUM_PREFILL_CHUNK` overrides it, `0` disables chunking
/// (one forward over the whole prompt, the pre-chunking behaviour). See
/// [`generate_reusing`]'s prefill for why.
fn prefill_chunk() -> usize {
    std::env::var("GALLIUM_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(512)
}

/// Run auto-regressive generation.
///
/// Returns the generated token IDs (not including the prompt).
///
/// An EOS token ends generation but is **not** reported: it is neither passed
/// to `on_token` nor included in the returned vec, so a streaming frontend
/// never prints it and the token-id record matches the visible text. `on_token`
/// sees each *kept* sampled token and decides whether to keep going:
/// `ControlFlow::Break` stops after that token, and the tokens produced so far
/// are returned normally. Sampling one token at a time is the only interruption
/// point a decode loop has, so this is what a caller that has to abandon a
/// generation — a cancelled turn, a stop sequence, a token budget — hooks into.
pub fn generate(
    model: &mut dyn CausalLM,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    max_new_tokens: usize,
    eos_tokens: &[u32],
    on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<Vec<u32>> {
    let (tokens, _) = generate_reusing(
        model,
        prompt_tokens,
        0,
        params,
        max_new_tokens,
        eos_tokens,
        on_token,
    )?;
    Ok(tokens)
}

/// [`generate`], for a caller that keeps the cache warm between calls.
///
/// `reuse` is how many leading tokens of `prompt_tokens` the model's cache
/// **already holds** — the caller has rolled it back to exactly that point with
/// [`crate::ModelCache::rewind`] — so only the rest is evaluated. `0` means a
/// cold start and resets the model, which is what [`generate`] passes.
///
/// The returned [`crate::CacheCheckpoint`] is taken at the **end of the prompt**,
/// before a single token is generated, and is `Some` only for a model whose
/// cache needs one ([`crate::ModelCache::needs_checkpoint`]). That is the one
/// moment it can be taken: a recurrent layer's state is a rolling summary, so
/// once generation moves it past the prompt there is no way back to that point,
/// and the *next* call's rewind target is exactly there.
///
/// The caller is responsible for the other half of the contract — knowing which
/// tokens the cache holds, and never claiming more than it does. Nothing here
/// can check that, and a record that leads the cache produces confident wrong
/// logits.
#[allow(clippy::too_many_arguments)]
pub fn generate_reusing(
    model: &mut dyn CausalLM,
    prompt_tokens: &[u32],
    reuse: usize,
    params: &SamplingParams,
    max_new_tokens: usize,
    eos_tokens: &[u32],
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<(Vec<u32>, Option<crate::CacheCheckpoint>)> {
    let device = model.device().clone();
    if reuse == 0 {
        model.reset();
    }

    // Prefill: forward what the cache does not already hold, in windows.
    //
    // A single forward over the whole fresh prompt allocates attention-score
    // scratch that grows with `seq_len` (on a global/full-attention layer the
    // `q·kᵀ` tensor is `[heads, seq_len, pos + seq_len]`), so a ~20k-token
    // prompt OOMs a 12 GB GPU while CPU — which pages — survives. Feeding the
    // prompt in `PREFILL_CHUNK`-token windows bounds each forward's scratch;
    // the KV cache carries context across them, exactly as it does for the
    // decode loop and for a KV-reused ReAct suffix (`reuse > 0`), so no model
    // sees a new code path. `GALLIUM_PREFILL_CHUNK` overrides the window; `0`
    // disables chunking. A prefill with image features staged is never chunked
    // — its `forward` scatters them over the soft-token positions of the batch
    // it is handed, which a split would break; a text turn stages nothing and
    // still chunks.
    //
    // The window also **steps down** when the VRAM ledger refuses it (issue
    // #343 stage 2): the same span is retried at half the size, down to
    // `MIN_PREFILL_CHUNK`, because the forward's transient — booked for the
    // whole forward and scaling with the window — is what a smaller window
    // frees. A refusal of the transient comes before the forward runs and
    // leaves the cache untouched; one *inside* it (a KV growth at layer *i*)
    // is rolled back to the window's start first, and is retried only if that
    // rollback was exact — otherwise the earlier positions are gone and there
    // is nothing to continue from. A staged image is never split, so it is
    // never stepped down either.
    let fresh = &prompt_tokens[reuse.min(prompt_tokens.len())..];
    let chunk = prefill_chunk();
    let splittable = !model.has_staged_image_features();
    let mut window = if chunk == 0 || !splittable {
        fresh.len()
    } else {
        chunk.min(fresh.len())
    };
    let mut logits = None;
    let mut off = 0;
    while off < fresh.len() {
        let end = (off + window).min(fresh.len());
        let win = Tensor::from_vec(fresh[off..end].to_vec(), (1, end - off), &device)?
            .to_dtype(DType::U32)?;
        match try_forward(model, &win, reuse + off) {
            Ok(l) => {
                logits = Some(l);
                off = end;
            }
            Err(refusal) => {
                let (e, intact) = match refusal {
                    ForwardRefusal::Transient(e) => (e, true),
                    ForwardRefusal::Forward(e) => {
                        let intact = roll_back_to(model, reuse + off);
                        (e, intact)
                    }
                };
                if !(intact
                    && splittable
                    && end - off > MIN_PREFILL_CHUNK
                    && crate::is_vram_exhausted(&e))
                {
                    return Err(e);
                }
                window = ((end - off) / 2).max(MIN_PREFILL_CHUNK);
                tracing::info!(
                    "prefill window at position {} refused by the VRAM ledger ({e}); \
                     retrying at {window} tokens",
                    reuse + off
                );
            }
        }
    }
    let logits = logits.ok_or_else(|| {
        candle_core::Error::Msg("generate: nothing to evaluate (empty prompt past `reuse`)".into())
    })?;

    // Here, and only here, the cache holds exactly the prompt.
    let checkpoint = model
        .cache()
        .filter(|c| c.needs_checkpoint())
        .map(|c| c.checkpoint());
    // logits shape: (1, vocab_size) — last token's logits
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

    let mut next_token = sample(&logits, params, &all_tokens)?;
    let mut generated: Vec<u32> = Vec::new();
    // An EOS as the very first token means the model produced nothing — return
    // an empty vec rather than a one-element `[eos]`.
    let mut stop = eos_tokens.contains(&next_token);
    if !stop {
        generated.push(next_token);
        all_tokens.push(next_token);
        stop = on_token(next_token).is_break();
    }

    // Decode: one token at a time. `generated` always holds the tokens kept so
    // far; the last one has been sampled but not yet fed back, which is the
    // invariant the KV-cache reuse below `generate_reusing` relies on.
    let mut step = 1;
    while !stop && step < max_new_tokens {
        let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?.to_dtype(DType::U32)?;
        let pos = prompt_tokens.len() + generated.len() - 1;
        let logits = forward_reserved(model, &input, pos)?;
        next_token = sample(&logits, params, &all_tokens)?;
        if eos_tokens.contains(&next_token) {
            break;
        }
        generated.push(next_token);
        all_tokens.push(next_token);
        stop = on_token(next_token).is_break();
        step += 1;
    }

    Ok((generated, checkpoint))
}

#[cfg(test)]
mod vram_refusal_tests {
    //! Issue #343 stage 2: what `generate_reusing` does when the VRAM ledger
    //! says no — on CPU, with byte counts only.
    use super::*;
    use crate::{KvCache, LayerCache, ModelCache, VramLedger};
    use std::sync::Arc;

    const VOCAB: usize = 8;
    /// `(1, 1, n, 4)` f32 per layer: 16 bytes per tensor, 32 across K and V.
    const ROW: usize = 32;

    /// Two attention layers, logits that always pick `(last token + 1) %
    /// VOCAB`, and a transient of `per_token` bytes per token forwarded.
    struct ToyLm {
        cache: ModelCache,
        per_token: usize,
        windows: Vec<usize>,
    }

    impl ToyLm {
        fn new(per_token: usize, ledger: Option<Arc<VramLedger>>) -> Self {
            let mut cache = ModelCache::new(vec![
                LayerCache::Kv(KvCache::new(1 << 16)),
                LayerCache::Kv(KvCache::new(1 << 16)),
            ]);
            if let Some(l) = ledger {
                cache.attach_ledger(l).unwrap();
            }
            Self {
                cache,
                per_token,
                windows: Vec::new(),
            }
        }

        fn layer_lens(&self) -> Vec<usize> {
            self.cache
                .layers
                .iter()
                .filter_map(|l| l.as_kv().map(|kv| kv.len()))
                .collect()
        }
    }

    impl CausalLM for ToyLm {
        fn forward(&mut self, ids: &Tensor, _pos: usize) -> Result<Tensor> {
            let n = ids.dim(1)?;
            self.windows.push(n);
            let kv = Tensor::zeros((1, 1, n, 4), DType::F32, &Device::Cpu)?;
            for i in 0..2 {
                self.cache.get_kv(i).unwrap().append(&kv, &kv)?;
            }
            let last = ids.squeeze(0)?.to_vec1::<u32>()?[n - 1] as usize;
            let mut logits = vec![0f32; VOCAB];
            logits[(last + 1) % VOCAB] = 1.0;
            Tensor::from_vec(logits, (1, VOCAB), &Device::Cpu)
        }
        fn reset(&mut self) {
            self.cache.reset();
        }
        fn cache(&mut self) -> Option<&mut ModelCache> {
            Some(&mut self.cache)
        }
        fn device(&self) -> &Device {
            &Device::Cpu
        }
        fn transient_bytes(&self, seq_len: usize, _pos: usize) -> usize {
            seq_len * self.per_token
        }
    }

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..Default::default()
        }
    }

    fn run(model: &mut ToyLm, prompt: &[u32]) -> Result<Vec<u32>> {
        generate_reusing(model, prompt, 0, &greedy(), 4, &[], |_| {
            ControlFlow::Continue(())
        })
        .map(|(t, _)| t)
    }

    /// A transient refused at the default 512-token window is retried at half
    /// the size until it fits — and the answer is the one an unconstrained run
    /// gives.
    #[test]
    fn a_refused_transient_steps_the_window_down() {
        let prompt: Vec<u32> = (0..600).map(|i| (i % VOCAB) as u32).collect();
        let expected = run(&mut ToyLm::new(1000, None), &prompt).unwrap();

        // KV for 600 + 4 positions in two layers is 2 * 1024 * ROW at most;
        // the rest of the budget fits a 128-token transient, not a 256 one.
        let ledger = VramLedger::new(2 * 1024 * ROW + 2 * 256 * ROW + 200_000, None);
        let mut model = ToyLm::new(1000, Some(ledger));
        assert_eq!(run(&mut model, &prompt).unwrap(), expected);
        // 512 is refused before it runs; the first 256 fits; the second 256's
        // KV growth is refused part way, rolled back, and redone as 128s.
        let prefill: Vec<usize> = model.windows.iter().copied().filter(|&n| n > 1).collect();
        assert_eq!(prefill, vec![256, 256, 128, 128, 88]);
        // The prompt, plus the three generated tokens fed back: each position
        // once, in both layers — the rolled-back attempt left nothing behind.
        assert_eq!(model.layer_lens(), vec![603, 603]);
    }

    /// When even the smallest window's transient is refused, the refusal is
    /// returned — nothing was forwarded, and the cache holds nothing.
    #[test]
    fn a_refusal_at_the_floor_is_returned_with_the_cache_empty() {
        let ledger = VramLedger::new(10_000, None);
        let mut model = ToyLm::new(1000, Some(ledger.clone()));
        let prompt: Vec<u32> = (0..300).map(|i| (i % VOCAB) as u32).collect();
        let err = run(&mut model, &prompt).unwrap_err();
        assert!(crate::is_vram_exhausted(&err), "{err}");
        assert_eq!(model.layer_lens(), vec![0, 0]);
        assert!(model.windows.is_empty(), "no forward ran");
        assert_eq!(ledger.reserved(), 0, "nothing left booked");
    }

    /// A KV growth refused *inside* a forward — the first layer grew, the
    /// second could not — rolls the cache back to the window's start, so the
    /// layers never disagree about what they hold. With no transient, smaller
    /// windows cross the same growth step, so the retries end at the floor and
    /// the refusal comes back.
    #[test]
    fn a_kv_growth_refused_mid_forward_rolls_back() {
        // Room for both layers' first 256-row buffers, then for the first
        // layer's 512-row growth only.
        let ledger = VramLedger::new(2 * 256 * ROW + 512 * ROW, None);
        let mut model = ToyLm::new(0, Some(ledger));
        let prompt: Vec<u32> = (0..300).map(|i| (i % VOCAB) as u32).collect();
        // The first 256 positions fit; the next window's growth does not.
        let first = generate_reusing(&mut model, &prompt[..200], 0, &greedy(), 1, &[], |_| {
            ControlFlow::Continue(())
        });
        assert!(first.is_ok());
        let held = model.layer_lens();
        let err = generate_reusing(&mut model, &prompt, 200, &greedy(), 1, &[], |_| {
            ControlFlow::Continue(())
        })
        .map(|(t, _)| t)
        .unwrap_err();
        assert!(crate::is_vram_exhausted(&err), "{err}");
        assert_eq!(model.layer_lens(), held, "both layers back where they were");
    }
}
