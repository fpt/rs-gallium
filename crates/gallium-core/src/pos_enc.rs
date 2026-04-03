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
            RoPEScaling::NTK { factor } => cfg.theta * factor.powf((rotary_dim as f64) / (rotary_dim as f64 - 2.0)),
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

        // Build cos/sin tables: (max_seq_len, half_dim)
        let inv_freq_tensor = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
        let positions: Vec<f64> = (0..cfg.max_seq_len).map(|p| p as f64).collect();
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
            let x_pass = x.narrow(3, self.rotary_dim, head_dim - self.rotary_dim)?.contiguous()?;
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
        Ok(Self { cos, sin, rotary_dim })
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
}
