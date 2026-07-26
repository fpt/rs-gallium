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

    /// Device the model lives on.
    fn device(&self) -> &Device;
}

/// Run auto-regressive generation.
///
/// Returns the generated token IDs (not including the prompt).
///
/// `on_token` sees each sampled token and decides whether to keep going:
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
    mut on_token: impl FnMut(u32) -> ControlFlow<()>,
) -> Result<Vec<u32>> {
    let device = model.device().clone();
    model.reset();

    // Prefill: forward all prompt tokens at once
    let prompt = Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_tokens.len()), &device)?
        .to_dtype(DType::U32)?;
    let logits = model.forward(&prompt, 0)?;
    // logits shape: (1, vocab_size) — last token's logits
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

    let mut next_token = sample(&logits, params, &all_tokens)?;
    let mut keep_going = on_token(next_token);
    let mut generated = vec![next_token];
    all_tokens.push(next_token);

    // Decode: one token at a time
    for _step in 1..max_new_tokens {
        if eos_tokens.contains(&next_token) || keep_going.is_break() {
            break;
        }
        let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?.to_dtype(DType::U32)?;
        let pos = prompt_tokens.len() + generated.len() - 1;
        let logits = model.forward(&input, pos)?;
        next_token = sample(&logits, params, &all_tokens)?;
        keep_going = on_token(next_token);
        generated.push(next_token);
        all_tokens.push(next_token);
    }

    Ok(generated)
}
