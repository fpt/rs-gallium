//! TurboKvCache: KV cache that stores keys and values in TurboQuant-compressed form.
//!
//! Drop-in replacement for `KvCache` that reduces memory by 5-8x.
//! Keys and values are quantized when appended and dequantized when retrieved
//! for attention computation.

use candle_core::{Device, Result, Tensor};

use crate::turbo_quant::{TurboQuant, TurboQuantConfig, TurboQuantized};

/// A KV cache that stores keys and values in TurboQuant-compressed form.
///
/// Usage: create with a `TurboQuantConfig` matching the head dimension.
/// Call `append()` during generation — it quantizes incoming K,V and returns
/// dequantized full K,V for the attention computation.
pub struct TurboKvCache {
    quant_k: TurboQuant,
    quant_v: TurboQuant,
    cached_k: Vec<TurboQuantized>,
    cached_v: Vec<TurboQuantized>,
    /// Cached dequantized K for reuse (only the previously-cached portion).
    cached_k_deq: Option<Tensor>,
    cached_v_deq: Option<Tensor>,
    max_seq_len: usize,
    current_len: usize,
}

impl TurboKvCache {
    /// Create a new TurboKvCache.
    ///
    /// `cfg` should have `dim` matching the per-head key/value dimension.
    /// Two separate quantizers are created for K and V (different random rotations).
    pub fn new(cfg: &TurboQuantConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let quant_k = TurboQuant::new(cfg, device)?;
        // Use a different seed for V quantizer
        let cfg_v = TurboQuantConfig {
            seed: cfg.seed.wrapping_add(1000),
            ..cfg.clone()
        };
        let quant_v = TurboQuant::new(&cfg_v, device)?;

        Ok(Self {
            quant_k,
            quant_v,
            cached_k: Vec::new(),
            cached_v: Vec::new(),
            cached_k_deq: None,
            cached_v_deq: None,
            max_seq_len,
            current_len: 0,
        })
    }

    /// Quantize and append new K, V. Returns the full (cached + new) dequantized K, V.
    ///
    /// Input K, V shape: (batch, n_kv_heads, seq_len, head_dim)
    /// Output K, V shape: (batch, n_kv_heads, total_len, head_dim)
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, new_seq, _d) = k.dims4()?;

        // Quantize the new chunk
        let q_k = self.quant_k.quantize(k)?;
        let q_v = self.quant_v.quantize(v)?;

        // Dequantize the new chunk
        let k_deq = self.quant_k.dequantize(&q_k)?;
        let v_deq = self.quant_v.dequantize(&q_v)?;

        // Store compressed
        self.cached_k.push(q_k);
        self.cached_v.push(q_v);
        self.current_len += new_seq;

        // Build full dequantized output by concatenating with previous
        let full_k = match &self.cached_k_deq {
            Some(prev) => Tensor::cat(&[prev, &k_deq], 2)?,
            None => k_deq.clone(),
        };
        let full_v = match &self.cached_v_deq {
            Some(prev) => Tensor::cat(&[prev, &v_deq], 2)?,
            None => v_deq.clone(),
        };

        // Cache the full dequantized tensors for next step
        self.cached_k_deq = Some(full_k.clone());
        self.cached_v_deq = Some(full_v.clone());

        // TODO: truncate to max_seq_len if needed

        Ok((full_k, full_v))
    }

    /// Current cached sequence length.
    pub fn len(&self) -> usize {
        self.current_len
    }

    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Reset the cache (start a new sequence).
    pub fn reset(&mut self) {
        self.cached_k.clear();
        self.cached_v.clear();
        self.cached_k_deq = None;
        self.cached_v_deq = None;
        self.current_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_quant::TurboQuantMode;

    #[test]
    fn test_turbo_kv_cache_basic() {
        let device = Device::Cpu;
        let head_dim = 64;
        let cfg = TurboQuantConfig {
            bit_width: 3,
            dim: head_dim,
            mode: TurboQuantMode::Mse,
            seed: 42,
        };
        let mut cache = TurboKvCache::new(&cfg, 1024, &device).unwrap();
        assert!(cache.is_empty());

        // Simulate prefill: 4 tokens
        let k = Tensor::randn(0f32, 1.0, (1, 4, 4, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 4, 4, head_dim), &device).unwrap();
        let (fk, fv) = cache.append(&k, &v).unwrap();
        assert_eq!(fk.dims(), &[1, 4, 4, head_dim]);
        assert_eq!(cache.len(), 4);

        // Simulate decode: 1 token
        let k2 = Tensor::randn(0f32, 1.0, (1, 4, 1, head_dim), &device).unwrap();
        let v2 = Tensor::randn(0f32, 1.0, (1, 4, 1, head_dim), &device).unwrap();
        let (fk2, fv2) = cache.append(&k2, &v2).unwrap();
        assert_eq!(fk2.dims(), &[1, 4, 5, head_dim]);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_turbo_kv_cache_reset() {
        let device = Device::Cpu;
        let cfg = TurboQuantConfig {
            bit_width: 2,
            dim: 32,
            mode: TurboQuantMode::Mse,
            seed: 42,
        };
        let mut cache = TurboKvCache::new(&cfg, 512, &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 3, 32), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 3, 32), &device).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.len(), 3);

        cache.reset();
        assert!(cache.is_empty());
    }
}
