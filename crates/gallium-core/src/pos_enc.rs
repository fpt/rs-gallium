use candle_core::{DType, Device, IndexOp, Result, Tensor};
use serde::Deserialize;

/// RoPE scaling strategy.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "rope_type", rename_all = "lowercase")]
pub enum RoPEScaling {
    None,
    Linear {
        factor: f64,
    },
    #[serde(rename = "yarn")]
    YaRN {
        factor: f64,
        original_max_position_embeddings: usize,
        #[serde(default = "default_beta_fast")]
        beta_fast: f64,
        #[serde(default = "default_beta_slow")]
        beta_slow: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: usize,
    },
    #[serde(rename = "ntk")]
    NTK {
        factor: f64,
    },
}

fn default_beta_fast() -> f64 {
    32.0
}
fn default_beta_slow() -> f64 {
    1.0
}

impl Default for RoPEScaling {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for Rotary Position Embeddings.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub theta: f64,
    pub scaling: RoPEScaling,
    /// Fraction of head_dim to apply rotary to (1.0 = full, 0.25 = partial).
    pub partial_rotary_factor: f64,
    /// Per-dimension frequency factors (e.g., Gemma 4 proportional RoPE).
    pub freq_factors: Option<Vec<f64>>,
}

impl Default for RoPEConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            max_seq_len: 4096,
            theta: 10000.0,
            scaling: RoPEScaling::None,
            partial_rotary_factor: 1.0,
            freq_factors: None,
        }
    }
}

/// Precomputed cos/sin tables for RoPE.
pub struct RoPE {
    cos: Tensor, // (max_seq_len, rotary_dim/2)
    sin: Tensor,
    rotary_dim: usize,
}

impl RoPE {
    pub fn new(cfg: &RoPEConfig, dtype: DType, device: &Device) -> Result<Self> {
        let rotary_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let half_dim = rotary_dim / 2;

        // Compute inverse frequencies
        let mut inv_freq = Vec::with_capacity(half_dim);
        let theta = match &cfg.scaling {
            RoPEScaling::NTK { factor } => {
                cfg.theta * factor.powf((rotary_dim as f64) / (rotary_dim as f64 - 2.0))
            }
            _ => cfg.theta,
        };

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / rotary_dim as f64);
            inv_freq.push(freq);
        }

        // Apply frequency factors if provided (e.g., proportional RoPE)
        if let Some(ref factors) = cfg.freq_factors {
            for (i, f) in inv_freq.iter_mut().enumerate() {
                if i < factors.len() {
                    *f /= factors[i];
                }
            }
        }

        // Apply scaling
        match &cfg.scaling {
            RoPEScaling::Linear { factor } => {
                for f in inv_freq.iter_mut() {
                    *f /= factor;
                }
            }
            RoPEScaling::YaRN {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
            } => {
                let low = (*original_max_position_embeddings as f64
                    / (*beta_fast * 2.0 * std::f64::consts::PI))
                    .floor();
                let high = (*original_max_position_embeddings as f64
                    / (*beta_slow * 2.0 * std::f64::consts::PI))
                    .floor();
                for (i, f) in inv_freq.iter_mut().enumerate() {
                    let wavelength = 2.0 * std::f64::consts::PI / *f;
                    let dim_ratio = wavelength / *original_max_position_embeddings as f64;
                    if dim_ratio < low as f64 / rotary_dim as f64 {
                        // High frequency: keep as is
                    } else if dim_ratio > high as f64 / rotary_dim as f64 {
                        // Low frequency: scale down
                        *f /= factor;
                    } else {
                        // Interpolation
                        let t = (i as f64 - low) / (high - low);
                        let scale = 1.0 / (1.0 + (factor - 1.0) * t);
                        *f *= scale;
                    }
                }
            }
            RoPEScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let old_ctx = *original_max_position_embeddings as f64;
                let low_freq_wavelen = old_ctx / low_freq_factor;
                let high_freq_wavelen = old_ctx / high_freq_factor;
                for f in inv_freq.iter_mut() {
                    let wavelen = 2.0 * std::f64::consts::PI / *f;
                    if wavelen < high_freq_wavelen {
                        // High frequency: keep
                    } else if wavelen > low_freq_wavelen {
                        // Low frequency: scale
                        *f /= factor;
                    } else {
                        // Smooth interpolation
                        let smooth = (old_ctx / wavelen - *low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        *f = (1.0 - smooth) * (*f / factor) + smooth * *f;
                    }
                }
            }
            RoPEScaling::None | RoPEScaling::NTK { .. } => {}
        }

        // Build cos/sin tables: (max_seq_len, half_dim).
        //
        // The outer product runs in F32, not F64: the scaling math above needs
        // f64 on the host, but the table itself is F32 in the references this is
        // checked against (transformers keeps `inv_freq` in float32 and matmuls
        // there; llama.cpp's rope carries a float theta), and it is cast down to
        // the model dtype two lines below anyway. It is also what makes this run
        // on a GPU at all — Metal has no F64 matmul, so the F64 version failed at
        // load time on every model, all of which build a RoPE this way.
        let inv_freq_tensor = Tensor::from_vec(
            inv_freq.iter().map(|&f| f as f32).collect::<Vec<_>>(),
            (1, half_dim),
            device,
        )?;
        let positions: Vec<f32> = (0..cfg.max_seq_len).map(|p| p as f32).collect();
        let pos_tensor = Tensor::from_vec(positions, (cfg.max_seq_len, 1), device)?;
        let freqs = pos_tensor.matmul(&inv_freq_tensor)?; // (max_seq_len, half_dim)

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    /// Apply rotary embeddings. Input shape: (batch, n_heads, seq_len, head_dim).
    /// `pos` is the position offset for KV cache.
    pub fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;

        // Context-window fail-fast (docs/TODO.md §1.2). The cos/sin tables are
        // built to `max_position_embeddings` rows; a position past the last one
        // has no rotation to look up. `self.cos.i(pos..pos + seq_len)` would
        // fail here anyway — with candle's `narrow invalid args start + len >
        // dim_len`, which names neither the cause nor the fix — and it is the
        // *first* thing to fail on overflow (attention applies RoPE before it
        // touches the KV cache or a mask), so this is where a turn that has
        // outgrown the model's trained context gets told so. Every LLM here
        // routes Q/K through this method, so the one check covers them all.
        let max_pos = self.cos.dim(0)?;
        if pos + seq_len > max_pos {
            candle_core::bail!(
                "context window exceeded: position {pos}..{} is past this model's \
                 trained context length of {max_pos} tokens. Start a new conversation \
                 or send a shorter prompt — this engine has no sliding-window eviction.",
                pos + seq_len
            );
        }

        // Slice cos/sin for current positions
        let cos = self.cos.i(pos..pos + seq_len)?.unsqueeze(0)?; // (1, seq_len, half_dim)
        let sin = self.sin.i(pos..pos + seq_len)?.unsqueeze(0)?;

        if self.rotary_dim == head_dim {
            // Full rotary: use candle-nn's rope
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        } else {
            // Partial rotary: split, apply to first part, concat back
            // narrow() returns a non-contiguous view; rope requires contiguous input.
            let x_rot = x.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let x_pass = x
                .narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?
                .contiguous()?;
            let x_rot = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&x_rot, &x_pass], 3)
        }
    }

    /// Build RoPE from precomputed inverse frequencies (e.g. loaded from GGUF rope_freqs tensor).
    /// `inv_freq` length = head_dim / 2; produces cos/sin of shape (max_seq_len, head_dim/2).
    /// Apply to the full head_dim — zero-frequency entries act as identity rotations.
    pub fn from_inv_freq(
        inv_freq: Vec<f64>,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = inv_freq.len();
        let rotary_dim = half_dim * 2;
        let inv_freq_t = Tensor::from_vec(
            inv_freq.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            (1, half_dim),
            device,
        )?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let pos_t = Tensor::from_vec(positions, (max_seq_len, 1), device)?;
        let freqs = pos_t.matmul(&inv_freq_t)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            rotary_dim,
        })
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let cfg = RoPEConfig {
            head_dim: 64,
            max_seq_len: 128,
            theta: 10000.0,
            ..Default::default()
        };
        let rope = RoPE::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.rotary_dim(), 64);
    }

    #[test]
    fn test_rope_partial() {
        let cfg = RoPEConfig {
            head_dim: 256,
            max_seq_len: 128,
            theta: 10000.0,
            partial_rotary_factor: 0.25,
            ..Default::default()
        };
        let rope = RoPE::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.rotary_dim(), 64);
    }

    #[test]
    fn test_rope_apply() {
        let cfg = RoPEConfig {
            head_dim: 64,
            max_seq_len: 128,
            theta: 10000.0,
            ..Default::default()
        };
        let rope = RoPE::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 4, 8, 64), &Device::Cpu).unwrap();
        let out = rope.apply(&x, 0).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8, 64]);
    }

    /// docs/TODO.md §1.2: a decode step past the model's trained context length
    /// fails with a message that names the cause and the fix, not candle's
    /// `narrow invalid args start + len > dim_len`. The last valid position is
    /// `max_seq_len - 1`; a single query there is fine, one past it is not.
    #[test]
    fn apply_past_the_context_window_fails_with_a_clear_message() {
        let cfg = RoPEConfig {
            head_dim: 64,
            max_seq_len: 128,
            theta: 10000.0,
            ..Default::default()
        };
        let rope = RoPE::new(&cfg, DType::F32, &Device::Cpu).unwrap();
        let one = |pos: usize| {
            let x = Tensor::randn(0f32, 1.0, (1, 4, 1, 64), &Device::Cpu).unwrap();
            rope.apply(&x, pos)
        };

        assert!(
            one(127).is_ok(),
            "the last in-window position still decodes"
        );

        let err = one(128).unwrap_err().to_string();
        assert!(err.contains("context window exceeded"), "{err}");
        assert!(err.contains("128"), "the limit is named: {err}");

        // A prompt longer than the whole window is refused up front, not
        // mid-decode.
        let long = Tensor::randn(0f32, 1.0, (1, 4, 200, 64), &Device::Cpu).unwrap();
        assert!(rope
            .apply(&long, 0)
            .unwrap_err()
            .to_string()
            .contains("context window exceeded"));
    }
}
