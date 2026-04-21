//! ARM NEON kernels for aarch64.
//!
//! NEON is mandatory on aarch64, so no runtime check is needed; `NeonKernels`
//! is always selected on that architecture.
//!
//! All functions use `#[target_feature(enable = "neon")]` so the compiler may
//! emit optimised code even without a global `-C target-feature=+neon` flag.
//! The `unsafe` invariant is satisfied by `NeonKernels` only being constructed
//! on aarch64, where the feature is guaranteed present.
//!
//! On non-aarch64 platforms the struct still compiles and delegates to
//! [`BaselineKernels`] so it can be named from any context.

use super::{baseline::BaselineKernels, Kernels};

#[derive(Debug)]
pub struct NeonKernels;

impl Kernels for NeonKernels {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn sgemm(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        #[cfg(target_arch = "aarch64")]
        // Safety: NEON is mandatory on aarch64.
        return unsafe { sgemm_neon(out, a, b, m, k, n) };
        #[cfg(not(target_arch = "aarch64"))]
        BaselineKernels.sgemm(out, a, b, m, k, n)
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        #[cfg(target_arch = "aarch64")]
        return unsafe { rmsnorm_neon(out, x, w, eps) };
        #[cfg(not(target_arch = "aarch64"))]
        BaselineKernels.rmsnorm(out, x, w, eps)
    }

    fn rope_row(&self, row: &mut [f32], cos: &[f32], sin: &[f32]) {
        // RoPE operates on pairs — load two floats, rotate, store.
        // Scalar is already fast; NEON saves a branch per pair but the gains
        // are small relative to the attention compute.  Implement as scalar.
        BaselineKernels.rope_row(row, cos, sin)
    }

    fn dequant_dot_q8_0(&self, quant_row: &[u8], x: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        return unsafe { dequant_dot_q8_0_neon(quant_row, x) };
        #[cfg(not(target_arch = "aarch64"))]
        BaselineKernels.dequant_dot_q8_0(quant_row, x)
    }
}

// ── NEON implementations (aarch64 only) ──────────────────────────────────────

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// 4-wide f32 FMA sgemm.
///
/// `vmlaq_f32(acc, a, b)` computes `acc += a * b` lane-wise.
/// `vaddvq_f32` reduces the 4-lane accumulator to a scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sgemm_neon(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        for j in 0..n {
            let b_row = b.as_ptr().add(j * k);
            let mut acc = vdupq_n_f32(0.0);
            let mut p = 0usize;
            while p + 4 <= k {
                let av = vld1q_f32(a_row.add(p));
                let bv = vld1q_f32(b_row.add(p));
                acc = vmlaq_f32(acc, av, bv);
                p += 4;
            }
            let mut sum = vaddvq_f32(acc);
            while p < k {
                sum += *a_row.add(p) * *b_row.add(p);
                p += 1;
            }
            out[i * n + j] = sum;
        }
    }
}

/// 4-wide RMSNorm.
/// Pass 1: accumulate sum-of-squares with `vmlaq_f32`.
/// Pass 2: scale with `vmulq_f32(vmulq_f32(x, inv_rms), w)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rmsnorm_neon(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut vsum = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(x.as_ptr().add(i));
        vsum = vmlaq_f32(vsum, v, v);
        i += 4;
    }
    let mut sum_sq = vaddvq_f32(vsum);
    while i < n { sum_sq += x[i] * x[i]; i += 1; }

    let inv_rms = (sum_sq / n as f32 + eps).sqrt().recip();
    let vs = vdupq_n_f32(inv_rms);

    let mut i = 0usize;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vw = vld1q_f32(w.as_ptr().add(i));
        let vo = vmulq_f32(vmulq_f32(vx, vs), vw);
        vst1q_f32(out.as_mut_ptr().add(i), vo);
        i += 4;
    }
    while i < n { out[i] = x[i] * inv_rms * w[i]; i += 1; }
}

/// Q8_0 dequant-dot: 4 i8 values at a time with NEON.
///
/// `vmovl_s8`  → s8×8  to s16×8
/// `vmovl_s16` → s16×4 to s32×4
/// `vcvtq_f32_s32` → s32×4 to f32×4
/// Then `vmlaq_f32` accumulates; `vaddvq_f32` reduces to scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_dot_q8_0_neon(quant_row: &[u8], x: &[f32]) -> f32 {
    use super::baseline::f16_le;
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n_blocks = x.len() / BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let base = blk * BLOCK_BYTES;
        let scale = f16_le(&quant_row[base..]);
        let qs = quant_row.as_ptr().add(base + 2) as *const i8;
        let xp = x.as_ptr().add(blk * BLOCK_SIZE);

        let mut vacc = vdupq_n_f32(0.0);
        let mut j = 0usize;
        // Process 8 i8 values at a time (one `vld1_s8` load, two 4-lane rounds).
        while j + 8 <= BLOCK_SIZE {
            let qi8x8 = vld1_s8(qs.add(j));        // 8 × i8
            let qi16  = vmovl_s8(qi8x8);            // 8 × i16
            // Lower 4 lanes
            let qi32_lo = vmovl_s16(vget_low_s16(qi16));
            let qf32_lo = vcvtq_f32_s32(qi32_lo);
            let xv_lo   = vld1q_f32(xp.add(j));
            vacc = vmlaq_f32(vacc, qf32_lo, xv_lo);
            // Upper 4 lanes
            let qi32_hi = vmovl_s16(vget_high_s16(qi16));
            let qf32_hi = vcvtq_f32_s32(qi32_hi);
            let xv_hi   = vld1q_f32(xp.add(j + 4));
            vacc = vmlaq_f32(vacc, qf32_hi, xv_hi);
            j += 8;
        }
        let mut block_dot = vaddvq_f32(vacc);
        while j < BLOCK_SIZE {
            block_dot += (*qs.add(j)) as f32 * *xp.add(j);
            j += 1;
        }
        total += scale * block_dot;
    }
    total
}
