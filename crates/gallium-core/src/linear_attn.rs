//! Gated DeltaNet: linear attention with gated delta update rule.
//!
//! Matches the `Qwen3_5GatedDeltaNet` reference in modeling_qwen3_5.py.
//! Key differences from a vanilla DeltaNet:
//!   - Separate projections: in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
//!   - Learnable per-head decay: g = -A_log.exp() * softplus(a + dt_bias)
//!   - GQA: num_v_heads may differ from num_k_heads (Q/K repeated to match V)
//!   - RMSNormGated output normalization (norm + silu gate)
//!   - l2-normalize Q and K before the recurrence (eps = 1e-6)
//!   - Convolution operates on the full QKV concat (key_dim*2 + value_dim channels)

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::kv_cache::RecurrentState;

/// Configuration for Gated DeltaNet linear attention.
#[derive(Debug, Clone)]
pub struct DeltaNetConfig {
    pub hidden_size: usize,
    /// Number of key/query heads in the linear attention layers.
    pub num_k_heads: usize,
    /// Number of value heads. Usually > num_k_heads (Q/K are repeated to match).
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    /// Causal convolution kernel size (typically 4).
    pub conv_kernel_dim: usize,
    pub rms_eps: f64,
}

/// Gated DeltaNet: O(n) linear attention with exponential decay and delta write rule.
///
/// Recurrence (per token t, per head h):
///   S = S * exp(g_t)                    # exponential decay
///   kv_mem = S^T @ k_t                  # read from state
///   delta = (v_t - kv_mem) * beta_t     # correction
///   S = S + k_t outer delta             # delta write
///   out_t = S^T @ q_t                   # read
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,  // hidden → key_dim*2 + value_dim
    in_proj_z: Linear,    // hidden → value_dim  (RMSNormGated gate)
    in_proj_b: Linear,    // hidden → num_v_heads (beta)
    in_proj_a: Linear,    // hidden → num_v_heads (decay a)
    out_proj: Linear,     // value_dim → hidden
    conv_weight: Tensor,  // (conv_dim, 1, kernel_size) — depthwise
    a_log: Tensor,        // (num_v_heads,) — learnable log-A
    dt_bias: Tensor,      // (num_v_heads,) — learnable dt bias
    norm_weight: Tensor,  // (value_head_dim,) — RMSNormGated scale
    cfg: DeltaNetConfig,
}

impl GatedDeltaNet {
    pub fn new(cfg: DeltaNetConfig, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.num_k_heads * cfg.key_head_dim;
        let value_dim = cfg.num_v_heads * cfg.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let in_proj_qkv = linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z   = linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b   = linear_no_bias(cfg.hidden_size, cfg.num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a   = linear_no_bias(cfg.hidden_size, cfg.num_v_heads, vb.pp("in_proj_a"))?;
        let out_proj    = linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?;

        let conv_weight = vb.get((conv_dim, 1, cfg.conv_kernel_dim), "conv1d.weight")?;
        let a_log       = vb.get(cfg.num_v_heads, "A_log")?;
        let dt_bias     = vb.get(cfg.num_v_heads, "dt_bias")?;
        let norm_weight = vb.get(cfg.value_head_dim, "norm.weight")?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            cfg,
        })
    }

    /// Forward pass.
    /// - `x`: (batch, seq_len, hidden_size)
    /// - `state`: mutable recurrent state (S matrix + conv buffer)
    /// Returns: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let n_k  = self.cfg.num_k_heads;
        let n_v  = self.cfg.num_v_heads;
        let dk   = self.cfg.key_head_dim;
        let dv   = self.cfg.value_head_dim;
        let key_dim   = n_k * dk;
        let value_dim = n_v * dv;

        // 1. Project and convolve QKV
        let mixed = self.in_proj_qkv.forward(x)?;          // (b, s, conv_dim)
        let mixed = self.apply_causal_conv(&mixed, state)?; // (b, s, conv_dim) with silu

        // 2. Split Q, K, V
        let q = mixed.narrow(2, 0,               key_dim)?;   // (b, s, key_dim)
        let k = mixed.narrow(2, key_dim,          key_dim)?;   // (b, s, key_dim)
        let v = mixed.narrow(2, key_dim * 2, value_dim)?;   // (b, s, value_dim)

        // 3. Gate projections
        let z = self.in_proj_z.forward(x)?;      // (b, s, value_dim)
        let b_raw = self.in_proj_b.forward(x)?;  // (b, s, n_v_heads)
        let a_raw = self.in_proj_a.forward(x)?;  // (b, s, n_v_heads)

        // beta = sigmoid(b)
        let beta = candle_nn::ops::sigmoid(&b_raw)?; // (b, s, n_v)

        // g = -A_log.exp() * softplus(a + dt_bias)  — always negative → decay in (0,1)
        let a_f32  = a_raw.to_dtype(DType::F32)?;
        let dt_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let alog_f32 = self.a_log.to_dtype(DType::F32)?;
        let a_plus_dt = a_f32.broadcast_add(&dt_f32)?; // (b, s, n_v)
        let g = (alog_f32.exp()?.broadcast_mul(&softplus(&a_plus_dt)?)?.neg()?.to_dtype(x.dtype()))?; // (b, s, n_v)

        // 4. Reshape to (b, s, n_heads, head_dim)
        let q = q.reshape((b, seq_len, n_k, dk))?;
        let k = k.reshape((b, seq_len, n_k, dk))?;
        let v = v.reshape((b, seq_len, n_v, dv))?;

        // 5. L2 normalize Q and K (eps = 1e-6)
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // 6. GQA: repeat Q and K if num_v_heads > num_k_heads
        let (q, k) = if n_v > n_k {
            let rep = n_v / n_k;
            let q = q.unsqueeze(3)?
                .expand((b, seq_len, n_k, rep, dk))?.contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            let k = k.unsqueeze(3)?
                .expand((b, seq_len, n_k, rep, dk))?.contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. Scale Q by 1/sqrt(key_head_dim)
        let scale = (dk as f64).powf(-0.5);
        let q = (q * scale)?;

        // 8. Recurrent gated delta rule
        let mut s = match state.state.take() {
            Some(s) => s.to_dtype(DType::F32)?,
            None    => Tensor::zeros((b, n_v, dk, dv), DType::F32, x.device())?,
        };

        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t    = q.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;    // (b, n_v, dk)
            let k_t    = k.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;    // (b, n_v, dk)
            let v_t    = v.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;    // (b, n_v, dv)
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_v)
            let g_t    = g.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;    // (b, n_v)

            // Decay: S = S * exp(g_t)   (g < 0, so exp(g) < 1)
            let decay = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (b, n_v, 1, 1)
            s = s.broadcast_mul(&decay.exp()?)?;

            // kv_mem = sum_dk(S * k_t.unsqueeze(-1)) = S^T @ k_t
            let kv_mem = (s.broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2))?; // (b, n_v, dv)

            // delta = (v - kv_mem) * beta
            let delta = (v_t - &kv_mem)?.broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?; // (b, n_v, dv)

            // Write: outer product k outer delta
            let write = k_t.unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?; // (b, n_v, dk, dv)
            s = (s + write)?;

            // Read: sum_dk(S * q_t.unsqueeze(-1)) = S^T @ q_t
            let o_t = (s.broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2))?; // (b, n_v, dv)
            outs.push(o_t.unsqueeze(1)?); // (b, 1, n_v, dv)
        }
        state.state = Some(s.to_dtype(x.dtype())?);

        // (b, seq, n_v, dv)
        let output = Tensor::cat(&outs, 1)?.to_dtype(x.dtype())?;

        // 9. RMSNormGated: norm(output) * weight * silu(z)
        let output_flat = output.reshape((b * seq_len * n_v, dv))?;
        let z_flat = z.reshape((b * seq_len * n_v, dv))?;
        let normed = self.rms_norm_gated(&output_flat, &z_flat)?;
        let output = normed.reshape((b, seq_len, value_dim))?;

        self.out_proj.forward(&output)
    }

    /// Gated RMSNorm: rms_norm(x * silu(gate)) * weight.
    /// Gated RMSNorm: rms_norm(x) * weight * silu(gate).
    /// Matches Python Qwen3_5RMSNormGated (norm-first, then gate).
    fn rms_norm_gated(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let orig = x.dtype();
        let xf  = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.cfg.rms_eps)?.sqrt()?)?;
        let w = self.norm_weight.to_dtype(DType::F32)?;
        let normed = normed.broadcast_mul(&w)?;
        (normed * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?.to_dtype(orig)
    }

    /// Causal depthwise conv1d with SiLU.
    /// x: (b, s, conv_dim) — note: conv is along the sequence dimension.
    fn apply_causal_conv(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let (b, seq_len, conv_dim) = x.dims3()?;
        let k = self.cfg.conv_kernel_dim;

        // Pad with stored conv state (left-pad for causal)
        let padded = match state.conv_state.take() {
            Some(prev) => Tensor::cat(&[&prev, x], 1)?,  // prev: (b, k-1, conv_dim)
            None => {
                let pad = Tensor::zeros((b, k - 1, conv_dim), x.dtype(), x.device())?;
                Tensor::cat(&[&pad, x], 1)?
            }
        };

        let total = padded.dim(1)?;
        state.conv_state = Some(padded.narrow(1, total - (k - 1), k - 1)?);

        // conv_weight: (conv_dim, 1, k) → squeeze → (conv_dim, k) → T → (k, conv_dim)
        let w = self.conv_weight.squeeze(1)?.transpose(0, 1)?; // (k, conv_dim)

        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(1, t, k)?; // (b, k, conv_dim)
            let out = window.broadcast_mul(&w)?.sum(1)?; // (b, conv_dim)
            outs.push(out.unsqueeze(1)?);
        }
        let result = Tensor::cat(&outs, 1)?; // (b, seq, conv_dim)
        candle_nn::ops::silu(&result)
    }
}

/// L2-normalize along last dimension with eps = 1e-6.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm_sq = x.sqr()?.sum_keepdim(D::Minus1)?;
    let norm = (norm_sq + 1e-6_f64)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Numerically stable softplus: log(1 + exp(x)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0_f64)?.log()
}
