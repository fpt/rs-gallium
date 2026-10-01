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

/// `model.forward(input, pos)`, with its [`CausalLM::transient_bytes`] booked
/// on the cache's ledger for the duration (when there is one).
fn forward_reserved(model: &mut dyn CausalLM, input: &Tensor, pos: usize) -> Result<Tensor> {
    let ledger = model.cache().and_then(|c| c.ledger().cloned());
    let _transient = match &ledger {
        Some(l) => Some(l.reserve_transient(
            model.transient_bytes(input.dim(1)?, pos),
            "forward transient",
        )?),
        None => None,
    };
    model.forward(input, pos)
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
    let budget = model.cache()?.ledger()?.budget();
    let cost = |model: &mut dyn CausalLM, p: usize| -> Option<usize> {
        let kv = model.cache()?.peak_bytes_at(p)?;
        let n = if chunk == 0 { p } else { chunk.min(p) };
        Some(kv + model.transient_bytes(n, p - n))
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
    let fresh = &prompt_tokens[reuse.min(prompt_tokens.len())..];
    let chunk = prefill_chunk();
    let logits = if chunk == 0 || fresh.len() <= chunk || model.has_staged_image_features() {
        let prompt =
            Tensor::from_vec(fresh.to_vec(), (1, fresh.len()), &device)?.to_dtype(DType::U32)?;
        forward_reserved(model, &prompt, reuse)?
    } else {
        let mut logits = None;
        let mut off = 0;
        while off < fresh.len() {
            let end = (off + chunk).min(fresh.len());
            let win = Tensor::from_vec(fresh[off..end].to_vec(), (1, end - off), &device)?
                .to_dtype(DType::U32)?;
            logits = Some(forward_reserved(model, &win, reuse + off)?);
            off = end;
        }
        logits.expect("fresh is non-empty on this branch")
    };

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
