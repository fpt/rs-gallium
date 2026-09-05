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

    fn dequant_dot_mxfp4(&self, quant_row: &[u8], x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        return unsafe { dequant_dot_mxfp4_avx2(quant_row, x) };
        #[cfg(not(target_arch = "x86_64"))]
        BaselineKernels.dequant_dot_mxfp4(quant_row, x)
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
    while i < n {
        sum_sq += x[i] * x[i];
        i += 1;
    }

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
    while i < n {
        out[i] = x[i] * inv_rms * w[i];
        i += 1;
    }
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
            let qi32 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(qs.add(j) as *const __m128i));
            let qf32 = _mm256_cvtepi32_ps(qi32);
            let xv = _mm256_loadu_ps(xp.add(j));
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

/// MXFP4 dequant-dot: one block (32 elements, 17 bytes) per iteration.
///
/// Reuses the nibble-unpack from `quantized::dequantize_mxfp4_avx2` — mask to
/// low/high nibbles, resolve i8 values through a 16-entry in-register `pshufb`
/// LUT — but instead of storing the widened f32 weights it FMAs them against
/// `x`. Scale is applied once per block on the reduced scalar, matching the
/// baseline's `scale * block_dot` order (and `dequant_dot_q8_0_avx2`).
///
/// The 16 packed-nibble bytes are loaded **unaligned**: a 17-byte block never
/// lands on a 16-byte boundary. `_mm_loadu_si128` at `base + 1` reads bytes
/// `base+1 .. base+17`, exactly the block's tail, so a `quant_row` of
/// `n_blocks * 17` bytes is not over-read.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequant_dot_mxfp4_avx2(quant_row: &[u8], x: &[f32]) -> f32 {
    use crate::quantized::e8m0_to_f32;
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 17;

    // pshufb LUT: nibble index → E2M1 i8 value. Byte-lane 0 holds code 0.
    // E2M1_LUT = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12].
    let lut = _mm_set_epi8(-12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0_i8);
    let nibble_mask = _mm_set1_epi8(0x0F_u8 as i8);

    let n_blocks = x.len() / BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let rb = blk * BLOCK_BYTES;
        let xp = x.as_ptr().add(blk * BLOCK_SIZE);

        // 16 packed-nibble bytes for this block.
        let qs = _mm_loadu_si128(quant_row.as_ptr().add(rb + 1) as *const __m128i);
        let lo = _mm_and_si128(qs, nibble_mask); // codes for elements 0..15
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble_mask); // elements 16..31
        let lo_i8 = _mm_shuffle_epi8(lut, lo);
        let hi_i8 = _mm_shuffle_epi8(lut, hi);

        // Widen 8 i8 → 8 f32 and FMA against the matching x lane.
        // _mm256_cvtepi8_epi32 reads the low 8 bytes; shift by 8 for the rest.
        macro_rules! fma_i8x8 {
            ($acc:expr, $v:expr, $xoff:expr) => {
                _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32($v)),
                    _mm256_loadu_ps(xp.add($xoff)),
                    $acc,
                )
            };
        }

        let mut acc = _mm256_setzero_ps();
        acc = fma_i8x8!(acc, lo_i8, 0);
        acc = fma_i8x8!(acc, _mm_srli_si128::<8>(lo_i8), 8);
        acc = fma_i8x8!(acc, hi_i8, 16);
        acc = fma_i8x8!(acc, _mm_srli_si128::<8>(hi_i8), 24);

        total += e8m0_to_f32(quant_row[rb]) * hsum256(acc);
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
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(hi, lo); // [a+e, b+f, c+g, d+h]
    let s2 = _mm_hadd_ps(s, s); // [a+b+e+f, c+d+g+h, ...]
    let s3 = _mm_hadd_ps(s2, s2); // [sum, sum, ...]
    _mm_cvtss_f32(s3)
}
