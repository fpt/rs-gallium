use candle_core::{Result, Tensor};
use rand::distributions::Distribution;
use rand::SeedableRng;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k filtering: keep only the k highest probability tokens.
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filtering: keep tokens until cumulative prob >= p.
    pub top_p: Option<f32>,
    /// Repetition penalty applied multiplicatively (1.0 = no penalty).
    pub repetition_penalty: Option<f32>,
    /// Presence penalty: subtract this value from logits of tokens already
    /// generated (prevents repetition in thinking/tool-call mode).
    pub presence_penalty: Option<f32>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }
}

/// Sample a token from logits (shape: (vocab_size,) or (1, vocab_size)).
pub fn sample(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> Result<u32> {
    // Flatten to 1D
    let logits = logits.squeeze(0)?;
    let mut logits_vec: Vec<f32> = logits.to_vec1()?;

    // Apply repetition penalty (multiplicative: divides positive logits, multiplies negative)
    if let Some(penalty) = params.repetition_penalty {
        if penalty != 1.0 {
            for &tok in previous_tokens {
                let idx = tok as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= penalty;
                    } else {
                        logits_vec[idx] *= penalty;
                    }
                }
            }
        }
    }

    // Apply presence penalty (additive: subtract fixed value from logits of seen tokens)
    if let Some(penalty) = params.presence_penalty {
        if penalty != 0.0 {
            // Collect unique already-seen token ids.
            let mut seen = std::collections::HashSet::new();
            for &tok in previous_tokens {
                let idx = tok as usize;
                if seen.insert(idx) && idx < logits_vec.len() {
                    logits_vec[idx] -= penalty;
                }
            }
        }
    }

    // Greedy
    if params.temperature == 0.0 {
        let (max_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        return Ok(max_idx as u32);
    }

    // Apply temperature
    let temp = params.temperature as f64;
    for v in logits_vec.iter_mut() {
        *v = (*v as f64 / temp) as f32;
    }

    // Build (index, logit) pairs and sort descending
    let mut indexed: Vec<(usize, f32)> = logits_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top-k
    if let Some(k) = params.top_k {
        indexed.truncate(k);
    }

    // Softmax on remaining
    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|(idx, logit)| (*idx, (logit - max_logit).exp()))
        .collect();

    // Top-p (nucleus)
    if let Some(p) = params.top_p {
        let sum: f32 = probs.iter().map(|(_, prob)| prob).sum();
        let mut cumulative = 0.0;
        let mut cutoff = probs.len();
        for (i, (_, prob)) in probs.iter().enumerate() {
            cumulative += prob / sum;
            if cumulative >= p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Normalize
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Sample
    let mut rng = match params.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let dist = rand::distributions::WeightedIndex::new(probs.iter().map(|(_, p)| *p as f64))
        .map_err(|e| candle_core::Error::Msg(format!("sampling error: {e}")))?;
    let sampled = dist.sample(&mut rng);
    Ok(probs[sampled].0 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sampling() {
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 2.0], (4,), &Device::Cpu).unwrap();
        let token = sample(&logits, &SamplingParams::greedy(), &[]).unwrap();
        assert_eq!(token, 1); // index of max value (5.0)
    }

    #[test]
    fn test_temperature_sampling() {
        let logits =
            Tensor::from_vec(vec![100.0f32, 0.0, 0.0, 0.0], (4,), &Device::Cpu).unwrap();
        let params = SamplingParams {
            temperature: 0.1,
            seed: Some(42),
            ..Default::default()
        };
        // With very low temperature and huge logit difference, should always pick 0
        let token = sample(&logits, &params, &[]).unwrap();
        assert_eq!(token, 0);
    }
}
