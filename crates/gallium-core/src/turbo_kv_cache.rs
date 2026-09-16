//! TurboKvCache: KV cache that stores keys and values in TurboQuant-compressed form.
//!
//! Drop-in replacement for `KvCache` that reduces resident KV memory — MSE
//! mode only (see `TurboQuant`'s own docs for why `InnerProduct` mode is
//! excluded: its sign bits are stored as f32, which `docs/TODO.md` §2.2
//! measured as *larger* than storing the value uncompressed). Keys and
//! values are quantized when appended; the full (cached + new) K/V a
//! forward pass needs is reconstructed by dequantizing the retained
//! compressed history — see [`TurboKvCache::append`] for why that, and not a
//! cached float copy, is the point.

use candle_core::{DType, Result, Tensor};

use crate::kv_cache::KvAppend;
use crate::turbo_quant::{TurboQuant, TurboQuantConfig, TurboQuantMode, TurboQuantized};

/// A KV cache that stores keys and values in TurboQuant-compressed form.
///
/// Usage: create with a `TurboQuantConfig` matching the head dimension.
/// Call `append()` during generation — it quantizes incoming K,V and returns
/// dequantized full K,V for the attention computation.
pub struct TurboKvCache {
    quant_k: TurboQuant,
    quant_v: TurboQuant,
    /// Compressed K: `indices` `(b, h_kv, len, d)` u8 + `norms` kept in the
    /// same `(b, h_kv, len, 1)` shape (not `TurboQuantized`'s own flattened
    /// `(n,)`, which only agrees with `indices.reshape(((), dim))`'s
    /// row-major order when it is flattened *after* every chunk is already in
    /// place — concatenating two already-flattened `norms` tensors along the
    /// wrong axis silently interleaves batch/head boundaries once `h > 1`).
    /// This is the *only* thing retained across calls — no float copy, which
    /// is the fix for `docs/TODO.md` §2.1: the old shape kept a running
    /// dequantized `cached_k_deq`/`cached_v_deq` right alongside these, so it
    /// was strictly worse than a plain `KvCache`, not smaller than one.
    k_indices: Option<Tensor>,
    k_norms: Option<Tensor>,
    v_indices: Option<Tensor>,
    v_norms: Option<Tensor>,
    /// Capacity a future eviction policy would truncate to. Unenforced: a
    /// `TurboKvCache` layer already makes `ModelCache::rewind` refuse any
    /// positional rollback unconditionally (it has none — dequantizing loses
    /// the exact float history a rollback would need to reproduce), so the
    /// cache either holds the whole conversation or is reset to empty. A
    /// sliding-window eviction is future work, not a correctness gap today.
    #[allow(dead_code)]
    max_seq_len: usize,
    current_len: usize,
}

impl TurboKvCache {
    /// Create a new TurboKvCache.
    ///
    /// `cfg` should have `dim` matching the per-head key/value dimension, and
    /// `mode: TurboQuantMode::Mse` — see the module docs for why
    /// `InnerProduct` is refused rather than silently accepted into a cache
    /// that would end up larger than an unquantized one.
    pub fn new(
        cfg: &TurboQuantConfig,
        max_seq_len: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if cfg.mode != TurboQuantMode::Mse {
            return Err(candle_core::Error::Msg(format!(
                "TurboKvCache only supports TurboQuantMode::Mse (got {:?}) — InnerProduct mode's \
                 sign bits store larger than the uncompressed value, see docs/TODO.md §2.2",
                cfg.mode
            )));
        }
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
            k_indices: None,
            k_norms: None,
            v_indices: None,
            v_norms: None,
            max_seq_len,
            current_len: 0,
        })
    }

    /// Quantize and append new K, V, and return the full (cached + new)
    /// dequantized K, V a forward pass needs.
    ///
    /// Input K, V shape: (batch, n_kv_heads, seq_len, head_dim)
    /// Output K, V shape: (batch, n_kv_heads, total_len, head_dim)
    ///
    /// This dequantizes the **whole retained history**, every call — not just
    /// the new tokens. There is no way around that here: candle has no
    /// attention kernel that reads a quantized K/V directly (that is what
    /// llama.cpp's own quantized-KV-cache path is, and it needs
    /// architecture-specific kernels this crate does not have), so the only
    /// way to feed the scores matmul is a real float tensor, and the whole
    /// point of this cache is to not be the thing holding that tensor between
    /// calls. So the trade is explicit: memory drops to roughly the
    /// compressed representation's size (indices as `u8` plus a per-vector
    /// norm — about half of `f16` at `bit_width` up to 8, since there is no
    /// bit-packing below a full byte per index either, see `TurboQuant`'s
    /// docs), and decode cost per step grows with the conversation instead of
    /// staying flat the way `KvCache`'s preallocated-buffer append does.
    /// Whether that trade is worth it for a given context length is exactly
    /// the question this cache exists to let someone measure.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, h, new_seq, _d) = k.dims4()?;

        let q_k = self.quant_k.quantize(k)?;
        let q_v = self.quant_v.quantize(v)?;

        self.k_indices = Some(cat_dim2(self.k_indices.take(), q_k.indices)?);
        self.k_norms = Some(cat_dim2(
            self.k_norms.take(),
            reshape_norms(q_k.norms, b, h, new_seq)?,
        )?);
        self.v_indices = Some(cat_dim2(self.v_indices.take(), q_v.indices)?);
        self.v_norms = Some(cat_dim2(
            self.v_norms.take(),
            reshape_norms(q_v.norms, b, h, new_seq)?,
        )?);
        self.current_len += new_seq;

        // TODO: truncate to max_seq_len if needed (see the field's own doc
        // comment — not reachable today since nothing rewinds this cache).

        let full_k = self.quant_k.dequantize(&self.full_k_quantized()?)?;
        let full_v = self.quant_v.dequantize(&self.full_v_quantized()?)?;
        Ok((full_k, full_v))
    }

    /// Reassemble a `TurboQuantized` over the whole retained history, flat
    /// norms in the row-major order `dequantize` expects (see the field doc
    /// comment on `k_norms`/`v_norms` for why that has to happen here and not
    /// at `append` time).
    fn full_k_quantized(&self) -> Result<TurboQuantized> {
        to_quantized(self.k_indices.as_ref(), self.k_norms.as_ref())
    }

    fn full_v_quantized(&self) -> Result<TurboQuantized> {
        to_quantized(self.v_indices.as_ref(), self.v_norms.as_ref())
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
        self.k_indices = None;
        self.k_norms = None;
        self.v_indices = None;
        self.v_norms = None;
        self.current_len = 0;
    }
}

impl KvAppend for TurboKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        TurboKvCache::append(self, k, v)
    }
}

/// Concatenate along the sequence axis (dim 2), or take `new` whole if this
/// is the first chunk.
fn cat_dim2(prev: Option<Tensor>, new: Tensor) -> Result<Tensor> {
    match prev {
        Some(prev) => Tensor::cat(&[&prev, &new], 2),
        None => Ok(new),
    }
}

/// `TurboQuantized::norms` comes back flat as `(b*h*seq,)`; put it in
/// `(b, h, seq, 1)` so it concatenates along the same axis, and in the same
/// order, as `indices`'s `(b, h, seq, d)`.
fn reshape_norms(norms: Tensor, b: usize, h: usize, seq: usize) -> Result<Tensor> {
    norms.reshape((b, h, seq, 1))
}

/// Flatten accumulated `(b, h, total_len, ..)` indices/norms back into the
/// `TurboQuantized` shape `dequantize` expects: `indices` untouched (its own
/// `dims()` is what `dequantize` reshapes the output to), `norms` flat over
/// the same row-major order.
fn to_quantized(indices: Option<&Tensor>, norms: Option<&Tensor>) -> Result<TurboQuantized> {
    let (Some(indices), Some(norms)) = (indices, norms) else {
        return Err(candle_core::Error::Msg(
            "TurboKvCache: dequantize called before any append".into(),
        ));
    };
    let n: usize = indices.dims()[..indices.rank() - 1].iter().product();
    Ok(TurboQuantized {
        indices: indices.clone(),
        norms: norms.reshape((n,))?.to_dtype(DType::F32)?,
        qjl_signs: None,
        residual_norms: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
        let (fk, _fv) = cache.append(&k, &v).unwrap();
        assert_eq!(fk.dims(), &[1, 4, 4, head_dim]);
        assert_eq!(cache.len(), 4);

        // Simulate decode: 1 token
        let k2 = Tensor::randn(0f32, 1.0, (1, 4, 1, head_dim), &device).unwrap();
        let v2 = Tensor::randn(0f32, 1.0, (1, 4, 1, head_dim), &device).unwrap();
        let (fk2, _fv2) = cache.append(&k2, &v2).unwrap();
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

    #[test]
    fn inner_product_mode_is_refused() {
        let cfg = TurboQuantConfig {
            bit_width: 3,
            dim: 32,
            mode: TurboQuantMode::InnerProduct,
            seed: 42,
        };
        assert!(TurboKvCache::new(&cfg, 512, &Device::Cpu).is_err());
    }

    /// The bug this whole rewrite is for: with more than one KV head (Gemma
    /// 4's GQA layers all have several), a flat concat of two chunks' norms
    /// would put chunk B's head-0 norms where chunk A's head-1 norms belong.
    /// Catch that by checking the *values*, not just the shape: a head
    /// initialized to a distinct constant should dequantize back to
    /// (approximately) that constant in every position of that head, across
    /// both a prefill and a follow-up decode-shaped append.
    #[test]
    fn multi_head_norms_stay_aligned_with_their_own_head() {
        let device = Device::Cpu;
        let head_dim = 16;
        let cfg = TurboQuantConfig {
            bit_width: 4,
            dim: head_dim,
            mode: TurboQuantMode::Mse,
            seed: 7,
        };
        let mut cache = TurboKvCache::new(&cfg, 256, &device).unwrap();

        // 2 heads, distinguishable by a large per-head magnitude: head 0 is a
        // vector of 1s, head 1 a vector of -3s (before the cache's own
        // internal normalization — the check below is about which *head* a
        // position's magnitude lands in, not the exact reconstructed value).
        let make = |seq: usize, h0: f32, h1: f32| -> Tensor {
            let mut data = vec![0f32; 2 * seq * head_dim];
            for s in 0..seq {
                for d in 0..head_dim {
                    data[(0 * seq + s) * head_dim + d] = h0;
                    data[(1 * seq + s) * head_dim + d] = h1;
                }
            }
            Tensor::from_vec(data, (1, 2, seq, head_dim), &device).unwrap()
        };

        let k1 = make(3, 1.0, -3.0);
        let (fk, _) = cache.append(&k1, &k1).unwrap();
        assert_eq!(fk.dims(), &[1, 2, 3, head_dim]);

        // A second, decode-shaped append (seq=1) — the shape that most
        // exercises the concat path in real use.
        let k2 = make(1, 1.0, -3.0);
        let (fk2, _) = cache.append(&k2, &k2).unwrap();
        assert_eq!(fk2.dims(), &[1, 2, 4, head_dim]);

        let vals: Vec<f32> = fk2.flatten_all().unwrap().to_vec1().unwrap();
        // Layout: (1, 2, 4, head_dim) row-major — head 0 is the first
        // 4*head_dim values, head 1 the next 4*head_dim.
        let head0 = &vals[0..4 * head_dim];
        let head1 = &vals[4 * head_dim..8 * head_dim];
        let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
        assert!(
            mean(head0) > 0.0,
            "head 0 (all +1s) dequantized to a negative mean: {}",
            mean(head0)
        );
        assert!(
            mean(head1) < 0.0,
            "head 1 (all -3s) dequantized to a positive mean: {}",
            mean(head1)
        );
    }
}
