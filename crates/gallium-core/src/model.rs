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

    // Prefill: forward what the cache does not already hold.
    let fresh = &prompt_tokens[reuse.min(prompt_tokens.len())..];
    let prompt =
        Tensor::from_vec(fresh.to_vec(), (1, fresh.len()), &device)?.to_dtype(DType::U32)?;
    let logits = model.forward(&prompt, reuse)?;

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
        let logits = model.forward(&input, pos)?;
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
