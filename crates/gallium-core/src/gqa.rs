//! Grouped-query attention products that never materialise the expanded K/V.
//!
//! GQA gives every group of `rep = h / h_kv` query heads one KV head. The obvious
//! way to reach the two matmuls is to make the shapes match by growing K and V to
//! `h` heads — `unsqueeze(2).expand(..).contiguous()` — which copies the whole
//! cache, per layer, per decode step. On gemma-4-12B at context 1577 that copy is
//! 2.89 GB and 700 ms per step on Metal (73 ms on the CPU); the matmuls it feeds
//! are ~150 ms. See docs/CANDLE_METAL.md.
//!
//! The copy is avoidable. Q's head axis is laid out so that query head `i` belongs
//! to KV head `i / rep` — heads of a group are adjacent — so splitting that axis
//! into `(h_kv, rep)` and folding `rep` into the *row* axis leaves a tensor whose
//! batch dims already agree with K's:
//!
//! ```text
//! Q  (b, h, s, d)      -> (b, h_kv, rep*s, d)   ⎫ batch dims agree, so this is
//! Kᵀ (b, h_kv, d, t)                            ⎬ one plain matmul — no
//! => (b, h_kv, rep*s, t) -> (b, h, s, t)        ⎭ broadcast, no expansion
//! ```
//!
//! Both reshapes are metadata-only on a contiguous tensor: query head `i = j*rep +
//! r` sits at row offset `(j*rep + r)*s + u`, which is exactly row `r*s + u` of
//! group `j`. Nothing moves.
//!
//! It is also the better matmul. At decode `s == 1`, so the expanding form asks for
//! `h` matmuls with a single row each; this form asks for `h_kv` with `rep` rows —
//! the same arithmetic in a shape a GEMM can actually use. The 1-KV-head layers
//! gain most, being the ones that expanded 16x.
//!
//! What remains is `Kᵀ`'s own `contiguous()`, and it now copies `h_kv` heads
//! instead of `h` (docs/CANDLE_METAL.md item 2 is to remove it altogether by
//! caching K pre-transposed).

use candle_core::{Result, Tensor, D};

/// `Q·Kᵀ`: `(b, h, s, d)` x `(b, h_kv, t, d)` -> `(b, h, s, t)`.
///
/// Unscaled and unmasked — scaling, softcapping, masking and softmax stay at the
/// call site, which is where they differ between models.
pub fn gqa_scores(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    let (b, h, s, _) = q.dims4()?;
    let (_, h_kv, t, _) = k.dims4()?;
    let rep = group_size(h, h_kv, "q")?;

    // (b, h_kv, t, d) -> (b, h_kv, d, t). The one copy left in this path.
    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    if rep == 1 {
        return q.contiguous()?.matmul(&k_t);
    }
    group_rows(q, h_kv, rep)?
        .matmul(&k_t)?
        .reshape((b, h, s, t))
}

/// `P·V`: `(b, h, s, t)` x `(b, h_kv, t, d)` -> `(b, h, s, d)`.
///
/// `probs` is post-softmax attention weights. Kept separate from [`gqa_scores`]
/// because what happens between the two is model-specific: a mask, Gemma's logit
/// softcapping, GPT-OSS's sink column.
pub fn gqa_weighted_sum(probs: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (b, h, s, _) = probs.dims4()?;
    let (_, h_kv, _, d) = v.dims4()?;
    let rep = group_size(h, h_kv, "probs")?;

    let v = v.contiguous()?;
    if rep == 1 {
        return probs.contiguous()?.matmul(&v);
    }
    group_rows(probs, h_kv, rep)?
        .matmul(&v)?
        .reshape((b, h, s, d))
}

/// `h / h_kv`, rejecting a head count that is not a whole number of groups —
/// which would silently pair query heads with the wrong KV head.
fn group_size(h: usize, h_kv: usize, what: &str) -> Result<usize> {
    if h_kv == 0 || h % h_kv != 0 {
        return Err(candle_core::Error::Msg(format!(
            "GQA needs {what}'s {h} heads to be a whole multiple of the KV \
             tensor's {h_kv}"
        )));
    }
    Ok(h / h_kv)
}

/// Fold the `rep` heads of each group into the row axis:
/// `(b, h, s, x)` -> `(b, h_kv, rep*s, x)`. Metadata-only once contiguous.
fn group_rows(t: &Tensor, h_kv: usize, rep: usize) -> Result<Tensor> {
    let (b, _, s, x) = t.dims4()?;
    t.contiguous()?.reshape((b, h_kv, rep * s, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The expansion this module exists to avoid, kept as the reference the fast
    /// path is checked against: query head `i` must still see KV head `i / rep`.
    fn expand_kv(kv: &Tensor, h: usize) -> Tensor {
        let (b, h_kv, t, d) = kv.dims4().unwrap();
        if h == h_kv {
            return kv.clone();
        }
        kv.unsqueeze(2)
            .unwrap()
            .expand((b, h_kv, h / h_kv, t, d))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, h, t, d))
            .unwrap()
    }

    fn arange(shape: (usize, usize, usize, usize), dev: &Device) -> Tensor {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        // Spread values so a mispaired head shows up as a wrong number, which a
        // tensor of ones or of one repeated pattern would hide.
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin()).collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Both products must equal the expanding form they replace, for every
    /// grouping a real model uses: gemma-4's 8-KV/16-Q sliding layers, its
    /// 1-KV/16-Q global layers (the 16x expansion), and plain MHA.
    #[test]
    fn matches_the_expanding_form() {
        let dev = Device::Cpu;
        let (b, s, t, d) = (1, 3, 5, 8);
        for (h, h_kv) in [(16, 8), (16, 1), (8, 8), (4, 2)] {
            let q = arange((b, h, s, d), &dev);
            let k = arange((b, h_kv, t, d), &dev);
            let v = arange((b, h_kv, t, d), &dev);

            let want = q
                .matmul(&expand_kv(&k, h).transpose(D::Minus2, D::Minus1).unwrap())
                .unwrap();
            let got = gqa_scores(&q, &k).unwrap();
            assert_eq!(got.dims(), &[b, h, s, t], "scores shape for {h}/{h_kv}");
            assert!(
                max_abs_diff(&want, &got) < 1e-5,
                "scores differ for {h}q/{h_kv}kv"
            );

            let probs = candle_nn::ops::softmax_last_dim(&got).unwrap();
            let want = probs.matmul(&expand_kv(&v, h)).unwrap();
            let got = gqa_weighted_sum(&probs, &v).unwrap();
            assert_eq!(got.dims(), &[b, h, s, d], "output shape for {h}/{h_kv}");
            assert!(
                max_abs_diff(&want, &got) < 1e-5,
                "weighted sum differs for {h}q/{h_kv}kv"
            );
        }
    }

    /// Decode is the case that matters (one row, long context) and the one where a
    /// wrong reshape is easiest to miss, `s == 1` making several wrong groupings
    /// the right shape.
    #[test]
    fn single_row_decode_step() {
        let dev = Device::Cpu;
        let (h, h_kv, t, d) = (16, 1, 64, 32);
        let q = arange((1, h, 1, d), &dev);
        let k = arange((1, h_kv, t, d), &dev);

        let want = q
            .matmul(&expand_kv(&k, h).transpose(D::Minus2, D::Minus1).unwrap())
            .unwrap();
        assert!(max_abs_diff(&want, &gqa_scores(&q, &k).unwrap()) < 1e-5);
    }

    /// A non-contiguous Q is what a caller that skipped `.contiguous()` after
    /// `transpose` hands in; the reshape inside must not reinterpret its strides.
    #[test]
    fn a_transposed_q_is_handled() {
        let dev = Device::Cpu;
        let (h, h_kv, s, t, d) = (4, 2, 3, 5, 8);
        // (b, s, h, d) -> (b, h, s, d), left non-contiguous on purpose.
        let q = arange((1, s, h, d), &dev).transpose(1, 2).unwrap();
        assert!(!q.is_contiguous());
        let k = arange((1, h_kv, t, d), &dev);

        let want = q
            .contiguous()
            .unwrap()
            .matmul(&expand_kv(&k, h).transpose(D::Minus2, D::Minus1).unwrap())
            .unwrap();
        assert!(max_abs_diff(&want, &gqa_scores(&q, &k).unwrap()) < 1e-5);
    }

    /// A head count that is not a whole number of groups is a config error, not
    /// something to paper over with a truncating division.
    #[test]
    fn a_ragged_grouping_is_an_error() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 6, 2, 4), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 3, 4), DType::F32, &dev).unwrap();
        assert!(gqa_scores(&q, &k).is_err());
    }
}
