use candle_core::{DType, Device, Result, Tensor};

/// Build a causal attention mask: (seq_len, total_len) where total_len = pos + seq_len.
/// Entries are 0.0 (attend) or -inf (block).
pub fn build_causal_mask(seq_len: usize, pos: usize, device: &Device) -> Result<Tensor> {
    let total_len = pos + seq_len;
    let mask = Tensor::zeros((seq_len, total_len), DType::F32, device)?;
    // For each query position i (0..seq_len), it can attend to positions 0..=(pos+i).
    // Mask out positions (pos+i+1)..total_len with -inf.
    if seq_len <= 1 {
        return Ok(mask);
    }
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        let query_pos = pos + i;
        for j in (query_pos + 1)..total_len {
            mask_data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_data, (seq_len, total_len), device)
}

/// Build a sliding-window + causal mask: each query attends to at most `window_size`
/// previous positions (inclusive of itself), with causal constraint.
pub fn build_sliding_window_mask(
    seq_len: usize,
    pos: usize,
    window_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = pos + seq_len;
    if seq_len <= 1 && total_len <= window_size {
        return Tensor::zeros((seq_len, total_len), DType::F32, device);
    }
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        let query_pos = pos + i;
        for j in 0..total_len {
            // Block if: future (causal) or too far in the past (sliding window)
            let is_future = j > query_pos;
            let is_outside_window = query_pos >= window_size && j < query_pos - window_size + 1;
            if is_future || is_outside_window {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (seq_len, total_len), device)
}

/// A sliding-window + causal mask whose **key axis is already narrowed** to the
/// span a window ever reaches, so the scores matmul that consumes it is
/// `window`-wide instead of whole-context-wide.
///
/// The union of every query's window over `pos..pos+seq_len` is the last
/// `min(total_len, seq_len + window_size - 1)` cache positions. The returned
/// mask is `[seq_len, kv_len]` for that many columns; column `j` is absolute
/// cache position `total_len - kv_len + j`. The caller narrows K and V the same
/// way: `k.narrow(2, total_len - kv_len, kv_len)`.
///
/// When nothing can be dropped (`kv_len == total_len`, the prefill case) this is
/// exactly [`build_sliding_window_mask`].
pub fn build_sliding_window_mask_narrowed(
    seq_len: usize,
    pos: usize,
    window_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = pos + seq_len;
    let kv_len = (seq_len + window_size.saturating_sub(1)).min(total_len);
    let kv_start = total_len - kv_len;

    if seq_len <= 1 && kv_len <= window_size {
        // Every kept column is inside the window and not in the future.
        return Tensor::zeros((seq_len, kv_len), DType::F32, device);
    }

    let mut mask_data = vec![0.0f32; seq_len * kv_len];
    for i in 0..seq_len {
        let query_pos = pos + i;
        for j in 0..kv_len {
            let key_pos = kv_start + j;
            let is_future = key_pos > query_pos;
            let is_outside_window =
                query_pos >= window_size && key_pos + window_size < query_pos + 1;
            if is_future || is_outside_window {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (seq_len, kv_len), device)
}

/// Whether an attention forward pass of `seq_len` new tokens at absolute
/// position `pos` needs an explicit mask at all.
///
/// This is the decision that regressed twice — first in GPT-OSS, then the same
/// bug independently in Gemma 4 (docs/TODO.md §1.1) — because each model spelled
/// it inline at its own call site. It lives here now, as one tested function
/// the three model forward passes call.
///
/// - **Global / full-attention layer** (`window = None`): a mask is needed only
///   to keep the queries within one batch from seeing each other's futures, i.e.
///   `seq_len > 1`. At decode (`seq_len == 1`) there is no future to hide —
///   attending to the entire past is exactly what causal attention means.
/// - **Sliding-window layer** (`window = Some(w)`): a mask is needed whenever a
///   query could reach a key outside its window. That is always true during
///   prefill, and becomes true at decode once the cache grows past the window
///   (`pos + seq_len > w`). Below that threshold every key is inside the window
///   and the mask would be all-zeros, so skipping it is equivalent and cheaper
///   — [`build_sliding_window_mask`] short-circuits to a zeros tensor in exactly
///   the same range, so a caller that builds one anyway is also correct.
pub fn attention_mask_needed(seq_len: usize, pos: usize, window: Option<usize>) -> bool {
    match window {
        None => seq_len > 1,
        Some(w) => seq_len > 1 || pos + seq_len > w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let mask = build_causal_mask(4, 0, &device).unwrap();
        assert_eq!(mask.dims(), &[4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Device::Cpu;
        let mask = build_causal_mask(3, 0, &device).unwrap();
        let data: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        // Row 0: can attend to pos 0 only
        assert_eq!(data[0][0], 0.0);
        assert!(data[0][1].is_infinite());
        // Row 2: can attend to 0,1,2
        assert_eq!(data[2][0], 0.0);
        assert_eq!(data[2][1], 0.0);
        assert_eq!(data[2][2], 0.0);
    }

    /// Decode, past the window: one query, a long cache. This is the shape the
    /// Gemma 4 forward passes used to discard by skipping the mask whenever
    /// `seq_len <= 1`, which let a sliding layer attend to the whole history.
    #[test]
    fn sliding_window_mask_at_decode_bounds_a_single_query() {
        let device = Device::Cpu;
        let window = 4;
        // Query at absolute position 9, cache holding 0..=9.
        let mask = build_sliding_window_mask(1, 9, window, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 10]);
        let row = mask.to_vec2::<f32>().unwrap()[0].clone();

        // Visible: the window ending at the query — positions 6..=9.
        // Blocked: everything older.
        for (j, v) in row.iter().enumerate() {
            if j >= 6 {
                assert_eq!(*v, 0.0, "position {j} must be inside the window");
            } else {
                assert!(
                    v.is_infinite() && v.is_sign_negative(),
                    "position {j} must be outside the window, got {v}"
                );
            }
        }
    }

    /// While the cache still fits the window there is nothing to block, and the
    /// builder short-circuits to zeros — so masking every decode step costs
    /// nothing until it starts mattering.
    #[test]
    fn sliding_window_mask_at_decode_is_all_visible_inside_the_window() {
        let device = Device::Cpu;
        let mask = build_sliding_window_mask(1, 3, 8, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 4]);
        assert!(mask.to_vec2::<f32>().unwrap()[0].iter().all(|v| *v == 0.0));
    }

    /// The regression guard for docs/TODO.md §1.1: a sliding layer must be told
    /// it needs a mask at decode once the cache passes the window, and a global
    /// layer must be told it does *not* at decode. The boundary is
    /// `pos + seq_len > window`, off-by-one included.
    #[test]
    fn attention_mask_decision_matches_the_window_boundary() {
        // Global layer: mask only for a multi-token (prefill) batch.
        assert!(attention_mask_needed(8, 0, None));
        assert!(!attention_mask_needed(1, 0, None));
        assert!(!attention_mask_needed(1, 10_000, None));

        // Sliding layer, decode (seq_len == 1): the mask starts mattering the
        // step the cache outgrows the window.
        let w = 512;
        assert!(!attention_mask_needed(1, w - 1, Some(w))); // cache = 512, fits
        assert!(attention_mask_needed(1, w, Some(w))); // cache = 513, one over
        assert!(attention_mask_needed(1, 5_000, Some(w))); // long session

        // Sliding layer, prefill: always.
        assert!(attention_mask_needed(2, 0, Some(w)));
        assert!(attention_mask_needed(64, 0, Some(w)));
    }

    /// Where `attention_mask_needed` says "no mask" for a sliding layer,
    /// `build_sliding_window_mask` produces an all-zeros tensor — so a call site
    /// that skips the mask and one that builds it anyway compute the same thing.
    #[test]
    fn no_mask_needed_agrees_with_an_all_zeros_sliding_mask() {
        let device = Device::Cpu;
        let w = 8;
        for pos in 0..w {
            assert!(!attention_mask_needed(1, pos, Some(w)));
            let m = build_sliding_window_mask(1, pos, w, &device).unwrap();
            assert!(
                m.to_vec2::<f32>().unwrap()[0].iter().all(|v| *v == 0.0),
                "pos {pos}: builder should be a no-op mask here"
            );
        }
    }

    #[test]
    fn test_sliding_window_mask() {
        let device = Device::Cpu;
        let mask = build_sliding_window_mask(4, 0, 2, &device).unwrap();
        let data: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        // Row 0 (pos 0, window=2): attend to [0]
        assert_eq!(data[0][0], 0.0);
        // Row 2 (pos 2, window=2): attend to [1, 2], not [0]
        assert!(data[2][0].is_infinite());
        assert_eq!(data[2][1], 0.0);
        assert_eq!(data[2][2], 0.0);
        // Row 3 (pos 3, window=2): attend to [2, 3], not [0, 1]
        assert!(data[3][0].is_infinite());
        assert!(data[3][1].is_infinite());
        assert_eq!(data[3][2], 0.0);
        assert_eq!(data[3][3], 0.0);
    }

    /// Decode past the window: the narrowed mask keeps only `window` columns and
    /// they are exactly the window ending at the query, all visible.
    #[test]
    fn narrowed_mask_at_decode_is_window_wide_and_all_visible() {
        let device = Device::Cpu;
        let window = 1024;
        // One query at absolute position 5000, cache holding 0..=5000.
        let mask = build_sliding_window_mask_narrowed(1, 5000, window, &device).unwrap();
        assert_eq!(mask.dims(), &[1, window], "key axis narrowed to the window");
        assert!(
            mask.to_vec2::<f32>().unwrap()[0].iter().all(|v| *v == 0.0),
            "the kept span is exactly the visible window"
        );
    }

    /// The narrowed mask must mask the same key positions as the full one over
    /// the columns they share — checked column-for-column across a prefill-sized
    /// batch of queries that is *longer* than the window.
    #[test]
    fn narrowed_mask_agrees_with_the_full_mask_on_shared_columns() {
        let device = Device::Cpu;
        let (seq_len, pos, window) = (12, 40, 6);
        let total_len = pos + seq_len;

        let full = build_sliding_window_mask(seq_len, pos, window, &device).unwrap();
        let narrowed = build_sliding_window_mask_narrowed(seq_len, pos, window, &device).unwrap();

        let kv_len = narrowed.dim(1).unwrap();
        assert_eq!(kv_len, (seq_len + window - 1).min(total_len));
        let kv_start = total_len - kv_len;

        let full: Vec<Vec<f32>> = full.to_vec2().unwrap();
        let narrowed: Vec<Vec<f32>> = narrowed.to_vec2().unwrap();
        for i in 0..seq_len {
            for j in 0..kv_len {
                assert_eq!(
                    narrowed[i][j].is_infinite(),
                    full[i][kv_start + j].is_infinite(),
                    "row {i} col {j} (abs {}) disagrees",
                    kv_start + j
                );
            }
            // Every full-mask column the narrowed one dropped was blocked anyway.
            for j in 0..kv_start {
                assert!(full[i][j].is_infinite(), "dropped col {j} was visible");
            }
        }
    }
}
