//! Baseline kernels: portable Rust, no intrinsics.
//!
//! This is the reference implementation: clear, correct, and readable.
//! Every other backend delegates to these functions for any op it does not
//! specialise; correctness tests run against this output.

use super::Kernels;

#[derive(Debug)]
pub struct BaselineKernels;

impl Kernels for BaselineKernels {
    fn name(&self) -> &'static str {
        "baseline"
    }

    /// Triple-loop sgemm.  Inner loop accesses contiguous memory in both `a`
    /// (row i of a) and `b` (row j of b, since b is stored transposed), so
    /// auto-vectorisation picks this up readily.
    fn sgemm(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                let a_row = &a[i * k..][..k];
                let b_row = &b[j * k..][..k];
                for p in 0..k {
                    acc += a_row[p] * b_row[p];
                }
                out[i * n + j] = acc;
            }
        }
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        let n = x.len();
        // Compute mean of squares.
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for i in 0..n {
            out[i] = x[i] * inv_rms * w[i];
        }
    }

    fn rope_row(&self, row: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = row.len() / 2;
        for i in 0..half {
            let x0 = row[2 * i];
            let x1 = row[2 * i + 1];
            row[2 * i]     = x0 * cos[i] - x1 * sin[i];
            row[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
        }
    }

    fn dequant_dot_q8_0(&self, quant_row: &[u8], x: &[f32]) -> f32 {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = 2 + BLOCK_SIZE; // 2 for f16 scale + 32 × i8
        let n_blocks = x.len() / BLOCK_SIZE;
        let mut total = 0.0f32;
        for blk in 0..n_blocks {
            let base = blk * BLOCK_BYTES;
            let scale = f16_le(&quant_row[base..]);
            let x_base = blk * BLOCK_SIZE;
            let mut block_dot = 0.0f32;
            for j in 0..BLOCK_SIZE {
                let q = quant_row[base + 2 + j] as i8;
                block_dot += q as f32 * x[x_base + j];
            }
            total += scale * block_dot;
        }
        total
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Decode a little-endian IEEE 754 half-precision float from `bytes[0..2]`.
/// No dependency on the `half` crate; used by all backends.
#[inline]
pub(super) fn f16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
    let sign = (bits >> 15) << 31;
    let exp  = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f32_bits = match exp {
        0 if mant == 0 => sign,                           // ± zero
        0 => {
            // Subnormal: normalise by finding leading 1 bit.
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mant << 13),         // ± inf / NaN
        e  => sign | ((e + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(f32_bits)
}
