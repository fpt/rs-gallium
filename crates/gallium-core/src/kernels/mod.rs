//! CPU kernel dispatch: selects the best available SIMD implementation at runtime.
//!
//! # Purpose
//! Candle handles the high-level tensor graph; this module owns the hot numerical
//! loops that sit inside forward passes.  Having an explicit dispatch layer means
//! each backend can be benchmarked or swapped without touching model code.
//!
//! # Abstraction
//! [`Kernels`] is the trait every backend implements.  [`KernelSet`] holds a
//! boxed backend and delegates; callers call `KernelSet::detect()` once at model
//! load time and keep the result.
//!
//! # Available targets
//!
//! | Struct | Selected when |
//! |--------|--------------|
//! | [`Avx512Kernels`] | x86-64 with AVX-512F + AVX-512BW at runtime |
//! | [`Avx2Kernels`]   | x86-64 with AVX2 + FMA at runtime |
//! | [`NeonKernels`]   | aarch64 (NEON is mandatory on that ISA) |
//! | [`BaselineKernels`] | portable fallback on any platform |
//!
//! # Adding a backend
//! 1. Add a file `kernels/<name>.rs` with a `pub struct <Name>Kernels`.
//! 2. Implement `Kernels` for it; delegate unsupported ops to `BaselineKernels`.
//! 3. Add `pub mod <name>; pub use <name>::<Name>Kernels;` here.
//! 4. Wire `detect_impl()` to return `Box::new(<Name>Kernels)` at the right point.

use std::fmt;

pub mod avx2;
pub mod avx512;
pub mod baseline;
pub mod neon;

pub use avx2::Avx2Kernels;
pub use avx512::Avx512Kernels;
pub use baseline::BaselineKernels;
pub use neon::NeonKernels;

// ── Trait ────────────────────────────────────────────────────────────────────

/// A set of CPU kernels for the hot operations in gallium inference.
///
/// All slice-based.  Callers are responsible for matching lengths to the
/// documented contracts; implementations may panic or produce incorrect results
/// on mismatched lengths.
pub trait Kernels: Send + Sync + fmt::Debug {
    /// Human-readable backend name (`"baseline"`, `"avx2"`, `"neon"`, …).
    fn name(&self) -> &'static str;

    /// Matrix multiply: `out = a @ b_T`
    ///
    /// - `a`:   row-major `[m × k]`
    /// - `b`:   row-major `[n × k]` — each row is one output neuron's weights
    ///          (i.e., already transposed relative to the mathematical `b`)
    /// - `out`: row-major `[m × n]`, fully overwritten
    ///
    /// This matches the standard linear-layer shape: `W ∈ ℝ^{out×in}`,
    /// called as `sgemm(out, x, W, batch, in_features, out_features)`.
    fn sgemm(&self, out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize);

    /// RMS normalisation: `out[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`
    ///
    /// `out`, `x`, and `w` must all have length `n`.
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], w: &[f32], eps: f32);

    /// Apply rotary position embedding to one head-row in-place.
    ///
    /// - `row`:       length `head_dim`, modified in-place
    /// - `cos`, `sin`: each length `head_dim / 2`, precomputed for this position
    ///
    /// Rotation applied to adjacent pairs: `(r₀, r₁) → (r₀·c - r₁·s, r₀·s + r₁·c)`.
    fn rope_row(&self, row: &mut [f32], cos: &[f32], sin: &[f32]);

    /// Dot product of `x` with a **Q8_0**-quantised row.
    ///
    /// **Q8_0 block layout** — 34 bytes per 32 elements:
    /// ```text
    /// [f16 scale (2 bytes)] [i8 × 32 (32 bytes)]
    /// ```
    /// Result: `Σ_blocks  scale_b · Σ_j  q_bj · x_bj`
    ///
    /// - `quant_row`: raw bytes; length must equal `(x.len() / 32) * 34`
    /// - `x`:         length must be a multiple of 32
    fn dequant_dot_q8_0(&self, quant_row: &[u8], x: &[f32]) -> f32;

    /// Dot product of `x` with an **MXFP4** (OCP MX FP4, E2M1)-quantised row.
    ///
    /// **MXFP4 block layout** — 17 bytes per 32 elements:
    /// ```text
    /// [E8M0 scale (1 byte)] [16 bytes: byte i holds element i in the low
    ///                        nibble and element i + 16 in the high nibble]
    /// ```
    /// Result: `Σ_blocks  e8m0_to_f32(scale_b) · Σ_j  E2M1_LUT[nibble_bj] · x_bj`
    ///
    /// - `quant_row`: raw bytes; length must equal `(x.len() / 32) * 17`
    /// - `x`:         length must be a multiple of 32
    ///
    /// This is **not** bit-identical to dequantising the whole row and calling
    /// a BLAS `matmul` — the reduction order differs — so it is used only where
    /// that difference has been shown not to matter (a single-token decode
    /// through `gpt_oss_q.rs`, gated by `GALLIUM_GPT_OSS_FUSED_MXFP4` and an
    /// A/B testsuite run). The weight bytes are inherently unaligned: a
    /// 17-byte block never lands on a SIMD boundary, so every implementation
    /// here loads them unaligned.
    fn dequant_dot_mxfp4(&self, quant_row: &[u8], x: &[f32]) -> f32;
}

// ── KernelSet ────────────────────────────────────────────────────────────────

/// Runtime-selected kernel set.
///
/// Constructed once with [`KernelSet::detect`] and shared (e.g. stored in the
/// model struct).  `Deref` to `&dyn Kernels` gives direct method dispatch.
pub struct KernelSet(Box<dyn Kernels>);

impl KernelSet {
    /// Probe CPU features and return the best available backend.
    pub fn detect() -> Self {
        Self(detect_impl())
    }

    /// Force the portable baseline backend (useful for tests and benchmarks).
    pub fn baseline() -> Self {
        Self(Box::new(BaselineKernels))
    }

    /// The backend name selected by `detect()`.
    pub fn name(&self) -> &'static str {
        self.0.name()
    }
}

impl std::ops::Deref for KernelSet {
    type Target = dyn Kernels;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl fmt::Debug for KernelSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KernelSet({})", self.0.name())
    }
}

// ── Runtime detection ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn detect_impl() -> Box<dyn Kernels> {
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
        tracing::debug!("kernels: selected avx512");
        Box::new(Avx512Kernels)
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        tracing::debug!("kernels: selected avx2");
        Box::new(Avx2Kernels)
    } else {
        tracing::debug!("kernels: selected baseline (no avx2/fma detected)");
        Box::new(BaselineKernels)
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_impl() -> Box<dyn Kernels> {
    // NEON is mandatory on aarch64; no runtime check needed.
    tracing::debug!("kernels: selected neon");
    Box::new(NeonKernels)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_impl() -> Box<dyn Kernels> {
    tracing::debug!("kernels: selected baseline (unsupported arch)");
    Box::new(BaselineKernels)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a smoke-test of all four ops against the given backend.
    pub fn smoke_test_kernels(k: &dyn Kernels) {
        // sgemm: 2×3 @ (2×3).T → 2×2
        // a = [[1,2,3],[4,5,6]], b = [[1,0,0],[0,1,0]]  (2 rows, 3 cols each)
        // out = [[1,2],[4,5]]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut out = [0.0f32; 4];
        k.sgemm(&mut out, &a, &b, 2, 3, 2);
        let expected = [1.0f32, 2.0, 4.0, 5.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "sgemm[{i}]: got {got}, expected {exp}"
            );
        }

        // rmsnorm: all-ones input with all-ones weight → each element = 1.0
        let x = [1.0f32; 8];
        let w = [1.0f32; 8];
        let mut y = [0.0f32; 8];
        k.rmsnorm(&mut y, &x, &w, 1e-5);
        for (i, &v) in y.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-4, "rmsnorm[{i}]: got {v}");
        }

        // rope_row: 90° rotation with cos=0, sin=1 → (x0,x1) becomes (-x1, x0)
        let mut row = [1.0f32, 0.0, 0.0, 1.0]; // two pairs
        let cos = [0.0f32, 0.0];
        let sin = [1.0f32, 1.0];
        k.rope_row(&mut row, &cos, &sin);
        // (1,0) → (0,1); (0,1) → (-1,0)
        assert!((row[0] - 0.0).abs() < 1e-5);
        assert!((row[1] - 1.0).abs() < 1e-5);
        assert!((row[2] - (-1.0)).abs() < 1e-5);
        assert!((row[3] - 0.0).abs() < 1e-5);

        // dequant_dot_q8_0: manually crafted single block
        // scale = 1.0 (f16), qs = [2, 3, 0, ...], x = [1.0, 1.0, ...]
        // expected = 1.0 * (2 + 3) = 5.0
        let mut block = [0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0 in little-endian
        block[2] = 2i8 as u8;
        block[3] = 3i8 as u8;
        let x32 = [1.0f32; 32];
        let dot = k.dequant_dot_q8_0(&block, &x32);
        assert!((dot - 5.0).abs() < 1e-4, "dequant_dot: got {dot}");

        // dequant_dot_mxfp4: one hand-built block.
        // scale byte 127 -> e8m0_to_f32(127) = 2^(126-127) = 0.5.
        // E2M1_LUT = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12].
        // byte[1] = 0x21 -> low nibble 1 (elem 0 -> LUT[1]=1), high nibble 2
        //   (elem 16 -> LUT[2]=2); every other nibble 0.
        // x = all ones -> Σ = 0.5 * (1 + 2) = 1.5.
        let mut mx = [0u8; 17];
        mx[0] = 127;
        mx[1] = 0x21;
        let dot = k.dequant_dot_mxfp4(&mx, &x32);
        assert!((dot - 1.5).abs() < 1e-4, "dequant_dot_mxfp4: got {dot}");
    }

    #[test]
    fn baseline_smoke() {
        smoke_test_kernels(&BaselineKernels);
    }

    #[test]
    fn detect_smoke() {
        let ks = KernelSet::detect();
        smoke_test_kernels(&*ks);
    }

    /// The detected backend's `dequant_dot_mxfp4` must track the scalar
    /// baseline across many blocks of pseudo-random MXFP4 bytes — the SIMD
    /// reduction reorders, so this is a closeness check, not bit-equality.
    #[test]
    fn dequant_dot_mxfp4_matches_baseline() {
        let ks = KernelSet::detect();
        if ks.name() == "baseline" {
            return; // nothing to compare against
        }
        // 90 blocks = 2880 elements, GPT-OSS 20B's per-row block count.
        const N_BLOCKS: usize = 90;
        let mut lcg: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) as u32
        };
        let mut quant = vec![0u8; N_BLOCKS * 17];
        for (i, b) in quant.iter_mut().enumerate() {
            // Keep scale bytes in a realistic exponent range; nibble bytes free.
            *b = if i % 17 == 0 {
                120 + (next() % 16) as u8
            } else {
                (next() & 0xFF) as u8
            };
        }
        let x: Vec<f32> = (0..N_BLOCKS * 32)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();

        let got = ks.dequant_dot_mxfp4(&quant, &x);
        let want = BaselineKernels.dequant_dot_mxfp4(&quant, &x);
        let tol = want.abs() * 1e-4 + 1e-4;
        assert!(
            (got - want).abs() <= tol,
            "{} dequant_dot_mxfp4 {got} vs baseline {want} (tol {tol})",
            ks.name()
        );
    }
}
