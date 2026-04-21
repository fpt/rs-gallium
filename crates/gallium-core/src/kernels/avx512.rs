//! AVX-512 kernels for x86-64.
//!
//! Uses 16-wide f32 vectors (`__m512`).  The sgemm inner loop processes 16
//! products per cycle instead of AVX2's 8, roughly doubling throughput on
//! sufficiently wide matrices.
//!
//! Ops without a dedicated AVX-512 implementation delegate to [`Avx2Kernels`]
//! (which is still legal — AVX-512 implies AVX2 + FMA).

use super::{avx2::Avx2Kernels, Kernels};

#[derive(Debug)]
pub struct Avx512Kernels;

impl Kernels for Avx512Kernels {
    fn name(&self) -> &'static str {
        "avx512"
    }

    fn sgemm(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        #[cfg(target_arch = "x86_64")]
        return unsafe { sgemm_avx512(out, a, b, m, k, n) };
        #[cfg(not(target_arch = "x86_64"))]
        Avx2Kernels.sgemm(out, a, b, m, k, n)
    }

    fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
        // 16-wide rmsnorm follows the same pattern as the AVX2 variant;
        // delegate for now — profiling rarely shows rmsnorm as the bottleneck.
        Avx2Kernels.rmsnorm(out, x, w, eps)
    }

    fn rope_row(&self, row: &mut [f32], cos: &[f32], sin: &[f32]) {
        Avx2Kernels.rope_row(row, cos, sin)
    }

    fn dequant_dot_q8_0(&self, quant_row: &[u8], x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        return unsafe { dequant_dot_q8_0_avx512(quant_row, x) };
        #[cfg(not(target_arch = "x86_64"))]
        Avx2Kernels.dequant_dot_q8_0(quant_row, x)
    }
}

// ── AVX-512 implementations (x86-64 only) ────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// 16-wide f32 FMA sgemm.  Same structure as the AVX2 version but with
/// `__m512` registers — processes twice as many elements per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_avx512(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = a.as_ptr().add(i * k);
        for j in 0..n {
            let b_row = b.as_ptr().add(j * k);
            let mut acc = _mm512_setzero_ps();
            let mut p = 0usize;
            while p + 16 <= k {
                let av = _mm512_loadu_ps(a_row.add(p));
                let bv = _mm512_loadu_ps(b_row.add(p));
                acc = _mm512_fmadd_ps(av, bv, acc);
                p += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            // Scalar tail or fall into AVX2 8-wide tail.
            while p < k {
                sum += *a_row.add(p) * *b_row.add(p);
                p += 1;
            }
            out[i * n + j] = sum;
        }
    }
}

/// Q8_0 dequant-dot: 16 i8 values at a time with AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn dequant_dot_q8_0_avx512(quant_row: &[u8], x: &[f32]) -> f32 {
    use super::baseline::f16_le;
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let n_blocks = x.len() / BLOCK_SIZE;
    let mut total = 0.0f32;

    for blk in 0..n_blocks {
        let base = blk * BLOCK_BYTES;
        let scale = f16_le(&quant_row[base..]);
        let qs = quant_row.as_ptr().add(base + 2);
        let xp = x.as_ptr().add(blk * BLOCK_SIZE);

        // Load 32 i8 values in two 16-element rounds.
        let mut vacc = _mm512_setzero_ps();
        for half in 0..2usize {
            let off = half * 16;
            // _mm_loadu_si128 loads 16 bytes; _mm512_cvtepi8_epi32 widens to 16×i32.
            let qi32 = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                qs.add(off) as *const __m128i,
            ));
            let qf32 = _mm512_cvtepi32_ps(qi32);
            let xv   = _mm512_loadu_ps(xp.add(off));
            vacc = _mm512_fmadd_ps(qf32, xv, vacc);
        }
        total += scale * _mm512_reduce_add_ps(vacc);
    }
    total
}
