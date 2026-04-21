//! AVX2 + FMA kernels for x86-64.
//!
//! Every `unsafe` fn here is decorated with `#[target_feature(enable="avx2,fma")]`
//! so the compiler can emit the right instructions.  The public `Avx2Kernels`
//! struct is the only entry point; it forwards to these fns after the runtime
//! check in `KernelSet::detect()` confirms the features are present.
//!
//! On non-x86 platforms the struct still exists (so the type is always nameable)
//! but each method delegates to `BaselineKernels`.

use super::{baseline::BaselineKernels, Kernels};

#[derive(Debug)]
pub struct Avx2Kernels;

impl Kernels for Avx2Kernels {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn sgemm(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        #[cfg(target_arch = "x86_64")]
        // Safety: constructed only after KernelSet::detect() confirms avx2 + fma.
        return unsafe { sgemm_avx2(out, a, b, m, k, n) };
        #[cfg(not(target_arch = "x86_64"))]
        BaselineKernels.sgemm(out, a, b, m, k, n)
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        #[cfg(target_arch = "x86_64")]
        return unsafe { rmsnorm_avx2(out, x, w, eps) };
        #[cfg(not(target_arch = "x86_64"))]
        BaselineKernels.rmsnorm(out, x, w, eps)
    }

    fn rope_row(&self, row: &mut [f32], cos: &[f32], sin: &[f32]) {
        // RoPE pairs are not wide enough to amortise gather overhead on small
        // head_dim values; baseline scalar is competitive here.
        BaselineKernels.rope_row(row, cos, sin)
    }

    fn dequant_dot_q8_0(&self, quant_row: &[u8], x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        return unsafe { dequant_dot_q8_0_avx2(quant_row, x) };
        #[cfg(not(target_arch = "x86_64"))]
        BaselineKernels.dequant_dot_q8_0(quant_row, x)
    }
}

// ── AVX2 implementations (x86-64 only) ──────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// 8-wide f32 FMA dot-product inner loop.
/// Each (i, j) pair scans row i of A and row j of B (both `[k]` f32 slices),
/// accumulating 8 products at a time and reducing to a scalar at the end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sgemm_avx2(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        for j in 0..n {
            let b_row = b.as_ptr().add(j * k);
            let mut acc = _mm256_setzero_ps();
            let mut p = 0usize;
            // Main 8-wide loop.
            while p + 8 <= k {
                let av = _mm256_loadu_ps(a_row.add(p));
                let bv = _mm256_loadu_ps(b_row.add(p));
                acc = _mm256_fmadd_ps(av, bv, acc);
                p += 8;
            }
            // Horizontal reduce to scalar.
            let mut sum = hsum256(acc);
            // Scalar tail (k not a multiple of 8).
            while p < k {
                sum += *a_row.add(p) * *b_row.add(p);
                p += 1;
            }
            out[i * n + j] = sum;
        }
    }
}

/// Vectorised RMSNorm.
/// Two passes over `x`: first to accumulate the sum of squares (8 at a time),
/// then to scale and write `out`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rmsnorm_avx2(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    // Pass 1: sum of squares.
    let mut vsum = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        vsum = _mm256_fmadd_ps(v, v, vsum);
        i += 8;
    }
    let mut sum_sq = hsum256(vsum);
    while i < n { sum_sq += x[i] * x[i]; i += 1; }

    let inv_rms = (sum_sq / n as f32 + eps).sqrt().recip();
    let vscale = _mm256_set1_ps(inv_rms);

    // Pass 2: out[i] = x[i] * inv_rms * w[i].
    let mut i = 0usize;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vw = _mm256_loadu_ps(w.as_ptr().add(i));
        let vout = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vout);
        i += 8;
    }
    while i < n { out[i] = x[i] * inv_rms * w[i]; i += 1; }
}

/// Q8_0 dequant-dot: process 32 i8 values per block, 8 at a time with AVX2.
///
/// Converts each block of 32 i8 values to f32 via `_mm256_cvtepi8_epi32` +
/// `_mm256_cvtepi32_ps`, then does 8-wide FMA against the corresponding x.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_dot_q8_0_avx2(quant_row: &[u8], x: &[f32]) -> f32 {
    use super::baseline::f16_le;
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n_blocks = x.len() / BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let base = blk * BLOCK_BYTES;
        let scale = f16_le(&quant_row[base..]);
        let qs = quant_row.as_ptr().add(base + 2); // i8 × 32
        let xp = x.as_ptr().add(blk * BLOCK_SIZE);

        let mut vacc = _mm256_setzero_ps();
        let mut j = 0usize;
        while j + 8 <= BLOCK_SIZE {
            // Load 8 bytes of i8 into a 64-bit integer register, then widen.
            let qi32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                qs.add(j) as *const __m128i,
            ));
            let qf32 = _mm256_cvtepi32_ps(qi32);
            let xv   = _mm256_loadu_ps(xp.add(j));
            vacc = _mm256_fmadd_ps(qf32, xv, vacc);
            j += 8;
        }
        let mut block_dot = hsum256(vacc);
        // Any remaining elements (block sizes that aren't multiples of 8).
        while j < BLOCK_SIZE {
            block_dot += (*qs.add(j) as i8) as f32 * *xp.add(j);
            j += 1;
        }
        total += scale * block_dot;
    }
    total
}

// ── AVX2 helpers ─────────────────────────────────────────────────────────────

/// Horizontal sum of an 8-lane f32 vector.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    // Add the two 128-bit halves together.
    let hi  = _mm256_extractf128_ps(v, 1);
    let lo  = _mm256_castps256_ps128(v);
    let s   = _mm_add_ps(hi, lo);            // [a+e, b+f, c+g, d+h]
    let s2  = _mm_hadd_ps(s, s);             // [a+b+e+f, c+d+g+h, ...]
    let s3  = _mm_hadd_ps(s2, s2);           // [sum, sum, ...]
    _mm_cvtss_f32(s3)
}
