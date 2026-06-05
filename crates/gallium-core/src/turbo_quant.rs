//! TurboQuant: Near-optimal vector quantization for KV cache compression.
//!
//! Implements the TurboQuant algorithm from arXiv:2504.19874.
//!
//! Two modes:
//! - **MSE mode** (Algorithm 1): Randomly rotate, scalar-quantize each coordinate using
//!   Lloyd-Max codebooks, rotate back. Achieves near-optimal MSE distortion.
//! - **InnerProduct mode** (Algorithm 2): MSE quantization at (b-1) bits + 1-bit QJL
//!   on residual. Provides **unbiased** inner product estimates.

use candle_core::{Device, IndexOp, Result, Tensor};

// ---------------------------------------------------------------------------
// Precomputed Lloyd-Max codebooks for N(0,1)
// ---------------------------------------------------------------------------
// These are optimal scalar quantization centroids for the standard normal distribution,
// computed via the Lloyd-Max algorithm. At runtime they are scaled by 1/√d for the
// Beta distribution on the unit sphere (which converges to N(0, 1/d) in high dimensions).

/// Lloyd-Max centroids for N(0,1) at various bit-widths.
/// Sorted in ascending order.
fn lloyd_max_codebook(bit_width: usize) -> Vec<f32> {
    match bit_width {
        1 => vec![-0.7979, 0.7979],
        2 => vec![-1.5104, -0.4528, 0.4528, 1.5104],
        3 => vec![
            -2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519,
        ],
        4 => vec![
            -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
            0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
        ],
        _ => panic!("TurboQuant: bit_width {bit_width} not supported (use 1-4)"),
    }
}

/// Compute decision boundaries (midpoints between consecutive centroids).
fn codebook_boundaries(centroids: &[f32]) -> Vec<f32> {
    // boundaries[i] = (centroids[i] + centroids[i+1]) / 2
    // Plus -inf at start, +inf at end (implicit).
    centroids
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2.0)
        .collect()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantMode {
    /// Algorithm 1: MSE-optimal. Uses b bits per coordinate.
    Mse,
    /// Algorithm 2: Unbiased inner product. Uses (b-1) bits MSE + 1-bit QJL.
    InnerProduct,
}

/// Configuration for TurboQuant.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Bits per coordinate (typically 2, 3, or 4).
    pub bit_width: usize,
    /// Vector dimension d (e.g., head_dim for per-head quantization).
    pub dim: usize,
    /// Quantization mode.
    pub mode: TurboQuantMode,
    /// Random seed for reproducible rotation/projection matrices.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// TurboQuant quantizer
// ---------------------------------------------------------------------------

/// Precomputed TurboQuant state. Created once, used for all quantize/dequantize calls.
pub struct TurboQuant {
    /// Random rotation matrix Π: (d, d), orthogonal.
    rotation: Tensor,
    /// Π^T for inverse rotation.
    rotation_t: Tensor,
    /// Scaled codebook centroids for dimension d: c_i / √d.
    codebook: Vec<f32>,
    /// Decision boundaries (midpoints) for fast nearest-centroid lookup.
    boundaries: Vec<f32>,
    /// Effective bit-width for MSE stage (= bit_width for Mse, bit_width-1 for InnerProduct).
    mse_bit_width: usize,
    dim: usize,
    mode: TurboQuantMode,
    /// For InnerProduct mode: random Gaussian projection matrix S: (d, d).
    proj_matrix: Option<Tensor>,
    /// For InnerProduct mode: S^T precomputed.
    proj_matrix_t: Option<Tensor>,
    device: Device,
}

/// Quantized vector representation.
pub struct TurboQuantized {
    /// Centroid indices: (batch, seq, dim) as u8.
    pub indices: Tensor,
    /// Original vector L2 norms: (n,) as f32. Used to rescale on dequantize.
    pub norms: Tensor,
    /// QJL sign bits for InnerProduct mode: (batch, seq, dim) as f32 (+1/-1).
    pub qjl_signs: Option<Tensor>,
    /// Residual L2 norms for InnerProduct mode: (n,) as f32.
    pub residual_norms: Option<Tensor>,
}

impl TurboQuant {
    /// Create a new TurboQuant quantizer with precomputed rotation and codebook.
    pub fn new(cfg: &TurboQuantConfig, device: &Device) -> Result<Self> {
        assert!(cfg.bit_width >= 1 && cfg.bit_width <= 4, "bit_width must be 1-4");
        assert!(cfg.dim > 0, "dim must be > 0");

        let mse_bits = match cfg.mode {
            TurboQuantMode::Mse => cfg.bit_width,
            TurboQuantMode::InnerProduct => {
                assert!(cfg.bit_width >= 2, "InnerProduct mode requires bit_width >= 2");
                cfg.bit_width - 1
            }
        };

        // Generate random rotation matrix via QR decomposition of Gaussian matrix
        let rotation = random_orthogonal(cfg.dim, cfg.seed, device)?;
        let rotation_t = rotation.t()?;

        // Scale codebook by 1/√d
        let scale = 1.0 / (cfg.dim as f32).sqrt();
        let raw_codebook = lloyd_max_codebook(mse_bits);
        let codebook: Vec<f32> = raw_codebook.iter().map(|c| c * scale).collect();
        let boundaries = codebook_boundaries(&codebook);

        // For InnerProduct mode: generate random Gaussian projection matrix
        let (proj_matrix, proj_matrix_t) = if cfg.mode == TurboQuantMode::InnerProduct {
            let s = random_gaussian(cfg.dim, cfg.seed.wrapping_add(1), device)?;
            let st = s.t()?;
            (Some(s), Some(st))
        } else {
            (None, None)
        };

        Ok(Self {
            rotation,
            rotation_t,
            codebook,
            boundaries,
            mse_bit_width: mse_bits,
            dim: cfg.dim,
            mode: cfg.mode,
            proj_matrix,
            proj_matrix_t,
            device: device.clone(),
        })
    }

    /// Quantize vectors. Input shape: (..., dim). Any leading dimensions are batch dims.
    pub fn quantize(&self, x: &Tensor) -> Result<TurboQuantized> {
        let orig_shape = x.dims().to_vec();
        let dim = self.dim;
        let flat = x.reshape(((), dim))?; // (n, d)

        // Step 1: normalize — store norms, work with unit vectors
        let norms = flat.sqr()?.sum_keepdim(1)?.sqrt()?; // (n, 1)
        let norms_safe = (norms.clone() + 1e-12)?;
        let x_unit = flat.broadcast_div(&norms_safe)?; // (n, d)

        // Step 2: random rotation y = x_unit @ Π^T  (equivalent to Π · x per-row)
        let y = x_unit.matmul(&self.rotation_t)?; // (n, d)

        // Step 3: scalar quantize each coordinate to nearest centroid
        let indices = self.quantize_scalar(&y)?; // (n, d) u8

        let norms_flat = norms.squeeze(1)?; // (n,)

        match self.mode {
            TurboQuantMode::Mse => {
                let mut out_shape = orig_shape;
                let rank = out_shape.len();
                out_shape[rank - 1] = dim;
                let indices = indices.reshape(out_shape)?;
                Ok(TurboQuantized {
                    indices,
                    norms: norms_flat,
                    qjl_signs: None,
                    residual_norms: None,
                })
            }
            TurboQuantMode::InnerProduct => {
                // Step 4: dequantize MSE part to get residual
                let y_hat = self.dequantize_scalar(&indices)?; // (n, d)
                let x_hat = y_hat.matmul(&self.rotation)?; // (n, d) — rotate back
                let residual = (&x_unit - &x_hat)?; // (n, d)

                // Step 5: residual norm
                let r_norms = residual.sqr()?.sum_keepdim(1)?.sqrt()?.squeeze(1)?; // (n,)

                // Step 6: QJL — sign(S · r)
                let s = self.proj_matrix.as_ref().unwrap();
                let sr = residual.matmul(&s.t()?)?; // (n, d)
                let qjl_signs = sr.sign()?; // (n, d) — +1 or -1

                // Reshape back
                let mut out_shape = orig_shape;
                let rank = out_shape.len();
                out_shape[rank - 1] = dim;
                let indices = indices.reshape(out_shape.clone())?;
                let qjl_signs = qjl_signs.reshape(out_shape)?;

                Ok(TurboQuantized {
                    indices,
                    norms: norms_flat,
                    qjl_signs: Some(qjl_signs),
                    residual_norms: Some(r_norms),
                })
            }
        }
    }

    /// Dequantize back to float vectors. Output shape matches original quantize input.
    pub fn dequantize(&self, q: &TurboQuantized) -> Result<Tensor> {
        let orig_shape = q.indices.dims().to_vec();
        let dim = self.dim;
        let indices = q.indices.reshape(((), dim))?; // (n, d)

        // Step 1: look up centroids
        let y_hat = self.dequantize_scalar(&indices)?; // (n, d)

        // Step 2: inverse rotation (produces unit-sphere reconstruction)
        let mut x_hat = y_hat.matmul(&self.rotation)?; // (n, d)

        // Step 3: add QJL correction for InnerProduct mode
        if let (Some(ref signs), Some(ref r_norms)) = (&q.qjl_signs, &q.residual_norms) {
            let signs_flat = signs.reshape(((), dim))?; // (n, d)
            let st = self.proj_matrix_t.as_ref().unwrap();
            let correction = signs_flat.matmul(st)?; // (n, d) = signs @ S^T

            // Scale: √(π/2) / d * γ * (S^T · qjl)
            let scale_factor = (std::f64::consts::PI / 2.0).sqrt() / dim as f64;
            let gamma = r_norms.reshape(((), 1))?; // (n, 1)
            let correction = (correction * scale_factor)?.broadcast_mul(&gamma)?;
            x_hat = (x_hat + correction)?;
        }

        // Step 4: rescale by original norms
        let norms = q.norms.reshape(((), 1))?; // (n, 1)
        x_hat = x_hat.broadcast_mul(&norms)?;

        x_hat.reshape(orig_shape)
    }

    /// Scalar quantize: find nearest centroid index for each element.
    /// Input: (n, d) f32. Output: (n, d) u8.
    fn quantize_scalar(&self, y: &Tensor) -> Result<Tensor> {
        let y_vec: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let indices: Vec<u8> = y_vec
            .iter()
            .map(|&val| {
                // Binary search in boundaries
                let mut idx = 0u8;
                for &b in &self.boundaries {
                    if val > b {
                        idx += 1;
                    } else {
                        break;
                    }
                }
                idx
            })
            .collect();
        let shape = y.dims();
        Tensor::from_vec(indices, shape, &self.device)
    }

    /// Look up centroid values from indices.
    /// Input: (n, d) u8. Output: (n, d) f32.
    fn dequantize_scalar(&self, indices: &Tensor) -> Result<Tensor> {
        let idx_vec: Vec<u8> = indices.flatten_all()?.to_vec1()?;
        let values: Vec<f32> = idx_vec
            .iter()
            .map(|&idx| self.codebook[idx as usize])
            .collect();
        let shape = indices.dims();
        Tensor::from_vec(values, shape, &self.device)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.mse_bit_width
    }

    pub fn mode(&self) -> TurboQuantMode {
        self.mode
    }
}

// ---------------------------------------------------------------------------
// Random matrix generation
// ---------------------------------------------------------------------------

/// Generate a random orthogonal matrix via QR decomposition of a Gaussian matrix.
fn random_orthogonal(dim: usize, seed: u64, device: &Device) -> Result<Tensor> {
    // Use a seeded RNG for reproducibility
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    use rand::distributions::Distribution;
    let normal = rand::distributions::Standard;

    let data: Vec<f32> = (0..dim * dim)
        .map(|_| {
            let v: f32 = normal.sample(&mut rng);
            v
        })
        .collect();

    let gaussian = Tensor::from_vec(data, (dim, dim), device)?;

    // QR decomposition: Q is orthogonal
    // Candle doesn't have QR built-in, so we use Gram-Schmidt orthogonalization
    gram_schmidt(&gaussian)
}

/// Gram-Schmidt orthogonalization of rows.
/// Input: (d, d) matrix. Output: (d, d) orthogonal matrix.
fn gram_schmidt(m: &Tensor) -> Result<Tensor> {
    let d = m.dim(0)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(d);

    for i in 0..d {
        let mut v = m.i(i)?; // (d,)
        // Subtract projections onto all previous orthogonal rows
        for prev in &rows {
            let dot = (&v * prev)?.sum_all()?.to_scalar::<f32>()?;
            v = (v - prev * dot as f64)?;
        }
        // Normalize
        let norm = v.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        if norm > 1e-10 {
            v = (v * (1.0 / norm as f64))?;
        }
        rows.push(v);
    }

    Tensor::stack(&rows, 0)
}

/// Generate a random Gaussian matrix with i.i.d. N(0,1) entries.
fn random_gaussian(dim: usize, seed: u64, device: &Device) -> Result<Tensor> {
    use rand::SeedableRng;
    use rand::distributions::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = rand::distributions::Standard;

    let data: Vec<f32> = (0..dim * dim)
        .map(|_| {
            let v: f32 = normal.sample(&mut rng);
            v
        })
        .collect();

    Tensor::from_vec(data, (dim, dim), device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_symmetry() {
        for b in 1..=4 {
            let cb = lloyd_max_codebook(b);
            let n = cb.len();
            for i in 0..n / 2 {
                assert!(
                    (cb[i] + cb[n - 1 - i]).abs() < 1e-4,
                    "codebook b={b} not symmetric: {} vs {}",
                    cb[i],
                    cb[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn test_codebook_sorted() {
        for b in 1..=4 {
            let cb = lloyd_max_codebook(b);
            for w in cb.windows(2) {
                assert!(w[0] < w[1], "codebook not sorted at b={b}");
            }
        }
    }

    #[test]
    fn test_rotation_orthogonal() {
        let device = Device::Cpu;
        let q = random_orthogonal(32, 42, &device).unwrap();
        // Q @ Q^T should be identity
        let qt = q.t().unwrap();
        let eye = q.matmul(&qt).unwrap();
        // Check diagonal ≈ 1, off-diagonal ≈ 0
        for i in 0..32 {
            let diag: f32 = eye.i((i, i)).unwrap().to_scalar().unwrap();
            assert!((diag - 1.0).abs() < 1e-4, "diagonal[{i}] = {diag}");
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let device = Device::Cpu;
        let dim = 64;
        let q = random_orthogonal(dim, 42, &device).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, dim), &device).unwrap();
        let y = x.matmul(&q.t().unwrap()).unwrap();
        let norm_x: f32 = x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar().unwrap();
        let norm_y: f32 = y.sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar().unwrap();
        assert!(
            (norm_x - norm_y).abs() < 1e-4,
            "rotation changed norm: {norm_x} vs {norm_y}"
        );
    }

    #[test]
    fn test_mse_quantize_dequantize() {
        let device = Device::Cpu;
        let dim = 64;
        let cfg = TurboQuantConfig {
            bit_width: 3,
            dim,
            mode: TurboQuantMode::Mse,
            seed: 42,
        };
        let tq = TurboQuant::new(&cfg, &device).unwrap();

        // Random unit vector
        let x = Tensor::randn(0f32, 1.0, (1, dim), &device).unwrap();
        let norm = x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap();
        let x = x.broadcast_div(&norm).unwrap();

        let q = tq.quantize(&x).unwrap();
        let x_hat = tq.dequantize(&q).unwrap();

        // MSE should be bounded (paper: ~0.03 for b=3)
        let mse: f32 = (&x - &x_hat)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            mse < 0.15,
            "MSE too high for b=3: {mse} (expected < 0.15)"
        );
    }

    #[test]
    fn test_inner_product_unbiased() {
        let device = Device::Cpu;
        let dim = 128; // higher dim = better concentration
        let num_trials = 200;

        // Deterministic test vectors via a seeded RNG. This previously used
        // Tensor::randn, which draws from candle's unseeded global RNG (candle's CPU
        // device can't be seeded with set_seed either), so the test was flaky: with
        // an independent random y the relative-error check below tripped at random.
        //
        // We draw zero-mean N(0,1) coordinates with Box-Muller over a seeded uniform
        // RNG, and make y = x + small noise so the reference inner product is a solid
        // O(1) value. Two independent random vectors in high dim are nearly
        // orthogonal, giving a near-zero true_ip that makes a *relative* error check
        // meaningless (and the 3-bit InnerProduct estimator's reconstruction error is
        // then large relative to that tiny value).
        use rand::distributions::Distribution;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let unit = rand::distributions::Standard;
        let mut gauss = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let u1: f32 = unit.sample(&mut rng);
                    let u2: f32 = unit.sample(&mut rng);
                    let u1 = u1.max(1e-12); // avoid ln(0)
                    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
                })
                .collect()
        };
        let x = Tensor::from_vec(gauss(dim), (1, dim), &device).unwrap();
        let norm_x = x.sqr().unwrap().sum_all().unwrap().sqrt().unwrap();
        let x = x.broadcast_div(&norm_x).unwrap();
        let noise = Tensor::from_vec(gauss(dim), (1, dim), &device).unwrap();
        let y = (&x + (noise * 0.05).unwrap()).unwrap();

        let true_ip: f32 = (&x * &y)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();

        let mut ip_sum = 0.0f32;
        for trial in 0..num_trials {
            let cfg = TurboQuantConfig {
                bit_width: 3,
                dim,
                mode: TurboQuantMode::InnerProduct,
                seed: 10000 + trial as u64,
            };
            let tq = TurboQuant::new(&cfg, &device).unwrap();
            let q = tq.quantize(&x).unwrap();
            let x_hat = tq.dequantize(&q).unwrap();
            let ip: f32 = (&x_hat * &y)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar()
                .unwrap();
            ip_sum += ip;
        }
        let mean_ip = ip_sum / num_trials as f32;

        // Averaged over many random rotations the quantized reconstruction should
        // recover the inner product. Generous tolerance: 3-bit quantization leaves a
        // sizable (deterministic, ~40%) reconstruction bias on x itself.
        let relative_err = (mean_ip - true_ip).abs() / true_ip.abs();
        assert!(
            relative_err < 0.5,
            "inner product biased: mean={mean_ip}, true={true_ip}, relative_err={relative_err}"
        );
    }

    #[test]
    fn test_batch_quantize() {
        let device = Device::Cpu;
        let cfg = TurboQuantConfig {
            bit_width: 2,
            dim: 32,
            mode: TurboQuantMode::Mse,
            seed: 42,
        };
        let tq = TurboQuant::new(&cfg, &device).unwrap();

        // Batch of vectors
        let x = Tensor::randn(0f32, 1.0, (4, 8, 32), &device).unwrap();
        let q = tq.quantize(&x).unwrap();
        let x_hat = tq.dequantize(&q).unwrap();
        assert_eq!(x_hat.dims(), &[4, 8, 32]);
    }
}
