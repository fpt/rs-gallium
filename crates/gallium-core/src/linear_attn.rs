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
    in_proj_qkv: Linear, // hidden → key_dim*2 + value_dim
    in_proj_z: Linear,   // hidden → value_dim  (RMSNormGated gate)
    in_proj_b: Linear,   // hidden → num_v_heads (beta)
    in_proj_a: Linear,   // hidden → num_v_heads (decay a)
    out_proj: Linear,    // value_dim → hidden
    conv_weight: Tensor, // (conv_dim, 1, kernel_size) — depthwise
    a_log: Tensor,       // (num_v_heads,) — learnable log-A
    dt_bias: Tensor,     // (num_v_heads,) — learnable dt bias
    norm_weight: Tensor, // (value_head_dim,) — RMSNormGated scale
    cfg: DeltaNetConfig,
}

impl GatedDeltaNet {
    pub fn new(cfg: DeltaNetConfig, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.num_k_heads * cfg.key_head_dim;
        let value_dim = cfg.num_v_heads * cfg.value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;

        let in_proj_qkv = linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(cfg.hidden_size, cfg.num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(cfg.hidden_size, cfg.num_v_heads, vb.pp("in_proj_a"))?;
        let out_proj = linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?;

        let conv_weight = vb.get((conv_dim, 1, cfg.conv_kernel_dim), "conv1d.weight")?;
        let a_log = vb.get(cfg.num_v_heads, "A_log")?;
        let dt_bias = vb.get(cfg.num_v_heads, "dt_bias")?;
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
        let n_k = self.cfg.num_k_heads;
        let n_v = self.cfg.num_v_heads;
        let dk = self.cfg.key_head_dim;
        let dv = self.cfg.value_head_dim;
        let key_dim = n_k * dk;
        let value_dim = n_v * dv;

        // 1. Project and convolve QKV
        let mixed = self.in_proj_qkv.forward(x)?; // (b, s, conv_dim)
        let mixed = self.apply_causal_conv(&mixed, state)?; // (b, s, conv_dim) with silu

        // 2. Split Q, K, V
        let q = mixed.narrow(2, 0, key_dim)?; // (b, s, key_dim)
        let k = mixed.narrow(2, key_dim, key_dim)?; // (b, s, key_dim)
        let v = mixed.narrow(2, key_dim * 2, value_dim)?; // (b, s, value_dim)

        // 3. Gate projections
        let z = self.in_proj_z.forward(x)?; // (b, s, value_dim)
        let b_raw = self.in_proj_b.forward(x)?; // (b, s, n_v_heads)
        let a_raw = self.in_proj_a.forward(x)?; // (b, s, n_v_heads)

        // beta = sigmoid(b)
        let beta = candle_nn::ops::sigmoid(&b_raw)?; // (b, s, n_v)

        // g = -A_log.exp() * softplus(a + dt_bias)  — always negative → decay in (0,1)
        let a_f32 = a_raw.to_dtype(DType::F32)?;
        let dt_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let alog_f32 = self.a_log.to_dtype(DType::F32)?;
        let a_plus_dt = a_f32.broadcast_add(&dt_f32)?; // (b, s, n_v)
        let g = (alog_f32
            .exp()?
            .broadcast_mul(&softplus(&a_plus_dt)?)?
            .neg()?
            .to_dtype(x.dtype()))?; // (b, s, n_v)

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
            let q = q
                .unsqueeze(3)?
                .expand((b, seq_len, n_k, rep, dk))?
                .contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            let k = k
                .unsqueeze(3)?
                .expand((b, seq_len, n_k, rep, dk))?
                .contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. Scale Q by 1/sqrt(key_head_dim)
        let scale = (dk as f64).powf(-0.5);
        let q = (q * scale)?;

        // 8. Gated delta rule — chunked for a prompt, one recurrent step for a
        //    decode token (see `gated_delta_rule`).
        let (output, s) = gated_delta_rule(&q, &k, &v, &g, &beta, state.state.take())?;
        state.state = Some(s.to_dtype(x.dtype())?);

        // (b, seq, n_v, dv)
        let output = output.to_dtype(x.dtype())?;

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
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.cfg.rms_eps)?.sqrt()?)?;
        let w = self.norm_weight.to_dtype(DType::F32)?;
        let normed = normed.broadcast_mul(&w)?;
        (normed * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?.to_dtype(orig)
    }

    /// Causal depthwise conv1d with SiLU.
    /// x: (b, s, conv_dim) — note: conv is along the sequence dimension.
    fn apply_causal_conv(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        // conv_weight: (conv_dim, 1, k) → squeeze → (conv_dim, k) → T → (k, conv_dim)
        let w = self.conv_weight.squeeze(1)?.transpose(0, 1)?;
        causal_conv1d(x, &w, state)
    }
}

/// Causal depthwise conv1d over the sequence axis, then SiLU, carrying the
/// last `k-1` inputs in `state.conv_state` so a later call continues the
/// sequence.
///
/// - `x`: `(b, s, c)`; `w`: `(k, c)`, one taps-by-channel kernel.
///
/// `k` shifted multiply-adds over the whole padded sequence rather than a
/// window per token: the output is the same, the dispatch count is `k` instead
/// of `s`, which on an accelerator is the difference between a conv that
/// costs nothing and one that costs a kernel launch per prompt token.
pub fn causal_conv1d(x: &Tensor, w: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
    let (b, seq_len, conv_dim) = x.dims3()?;
    let k = w.dim(0)?;

    // Left-pad with the stored conv state (zeros at the start of a sequence).
    let padded = match state.conv_state.take() {
        Some(prev) => Tensor::cat(&[&prev, x], 1)?, // prev: (b, k-1, conv_dim)
        None => {
            let pad = Tensor::zeros((b, k - 1, conv_dim), x.dtype(), x.device())?;
            Tensor::cat(&[&pad, x], 1)?
        }
    };
    let total = padded.dim(1)?;
    state.conv_state = Some(padded.narrow(1, total - (k - 1), k - 1)?);

    // out[t] = Σ_j padded[t + j] · w[j]
    let w = w.to_dtype(x.dtype())?;
    let mut acc: Option<Tensor> = None;
    for j in 0..k {
        let term = padded
            .narrow(1, j, seq_len)?
            .broadcast_mul(&w.narrow(0, j, 1)?)?; // (b, s, c) · (1, c)
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    candle_nn::ops::silu(&acc.expect("k >= 1"))
}

/// Batched matmul over any number of leading dims.
///
/// candle's `matmul` wants the batch dims of a rank-4+ operand to collapse
/// into one stride, and a contiguous tensor with a size-1 batch dim (one chunk)
/// fails that check as "non-contiguous". Flattening the leading dims to one
/// batch axis sidesteps it; both operands are made contiguous first, so the
/// reshape is a view.
fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (ad, bd) = (a.dims(), b.dims());
    let r = ad.len();
    let batch: usize = ad[..r - 2].iter().product();
    let a3 = a.contiguous()?.reshape((batch, ad[r - 2], ad[r - 1]))?;
    let b3 = b.contiguous()?.reshape((batch, bd[r - 2], bd[r - 1]))?;
    let out = a3.matmul(&b3)?;
    let mut shape = ad[..r - 2].to_vec();
    shape.extend_from_slice(&[ad[r - 2], bd[r - 1]]);
    out.reshape(shape)
}

/// Chunk length for [`chunk_gated_delta_rule`]. The reference uses 64; 32 is
/// where the dispatch count `~5·chunk + 6·(s/chunk)` bottoms out for a
/// 512-token forward chunk, since the intra-chunk solve is itself `chunk`
/// sequential steps here.
pub const DELTA_CHUNK: usize = 32;

/// The gated delta rule over a whole sequence, choosing the form by length.
///
/// Inputs are the per-head projections *after* the conv, l2-norm, GQA
/// expansion and query scaling:
/// - `q`, `k`: `(b, s, h, dk)`; `v`: `(b, s, h, dv)`
/// - `g` (log-space decay, `<= 0`), `beta` (write gate in `(0,1)`): `(b, s, h)`
/// - `state`: the recurrent `S` matrix `(b, h, dk, dv)` from the previous call,
///   or `None` at the start of a sequence.
///
/// Returns the read-out `(b, s, h, dv)` and the new state, both `f32`.
///
/// A single token (decode) takes one recurrent step. A prompt takes the
/// chunked form, which is the same recurrence regrouped so that within a chunk
/// everything is a matmul and only the chunk-to-chunk state hand-off is
/// sequential. On an accelerator that is the whole difference: the per-token
/// loop issued ~10 small kernels per token per layer, so a 2k-token prompt on
/// Qwen3.8-9B spent ~200 s in this function on Metal — 11 tok/s of prefill
/// against llama.cpp's 151.
pub fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (b, seq_len, h, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let state = match state {
        Some(s) => s.to_dtype(DType::F32)?,
        None => Tensor::zeros((b, h, dk, dv), DType::F32, q.device())?,
    };
    if seq_len == 1 {
        recurrent_gated_delta_rule(q, k, v, g, beta, state)
    } else {
        chunk_gated_delta_rule(q, k, v, g, beta, state, DELTA_CHUNK)
    }
}

/// The recurrence itself, one token at a time. Shapes as [`gated_delta_rule`];
/// `state` is `(b, h, dk, dv)` f32.
///
/// Per token `t`, per head:
/// ```text
///   S      = S · exp(g_t)              # decay
///   kv_mem = Sᵀ k_t                    # what the state predicts for k_t
///   delta  = (v_t − kv_mem) · beta_t   # the correction
///   S      = S + k_t ⊗ delta           # delta write
///   o_t    = Sᵀ q_t                    # read
/// ```
/// This is the reference the chunked form is checked against, and the path a
/// decode step takes.
pub fn recurrent_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    mut s: Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq_len = q.dim(1)?;
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, h, dk)
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, h, dk)
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, h, dv)
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, h)
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, h)

        let decay = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (b, h, 1, 1)
        s = s.broadcast_mul(&decay.exp()?)?;

        let kv_mem = s
            .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?; // (b, h, dv)
        let delta = (v_t - &kv_mem)?.broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?; // (b, h, dv)
        let write = k_t
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?; // (b, h, dk, dv)
        s = (s + write)?;

        let o_t = s
            .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?; // (b, h, dv)
        outs.push(o_t.unsqueeze(1)?); // (b, 1, h, dv)
    }
    Ok((Tensor::cat(&outs, 1)?, s))
}

/// The chunked (WY / UT-transform) form of the gated delta rule — the
/// `torch_chunk_gated_delta_rule` reference, in candle. Shapes as
/// [`gated_delta_rule`]; `state` is `(b, h, dk, dv)` f32.
///
/// Within a chunk of `chunk` tokens the `chunk` delta writes are condensed
/// into a unit-lower-triangular system `(I + L) u = β·v`, whose solution `u`
/// is what the chunk writes to the state and `(I + L)⁻¹ (β·k·exp(cum g))` is
/// how it reads the state it started from. candle has no triangular solver,
/// so the system is solved by forward substitution, one row per step, batched
/// over every chunk and head at once; the other loop is the state hand-off
/// across chunks. The work per layer is therefore ~`chunk` small steps plus
/// `s / chunk` state steps, instead of `s` steps of everything.
///
/// Padded tail positions get `q = k = v = 0`, `g = β = 0`: they read nothing,
/// write nothing and do not decay the state, so the returned state is the
/// state after exactly `s` tokens.
pub fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Tensor,
    chunk: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, seq_len, h, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let dev = q.device();

    // (b, h, s, d) in f32, padded to a whole number of chunks.
    let pad = (chunk - seq_len % chunk) % chunk;
    let heads_first = |t: &Tensor| -> Result<Tensor> {
        t.to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .pad_with_zeros(2, 0, pad)
    };
    let q = heads_first(q)?;
    let k = heads_first(k)?;
    let v = heads_first(v)?;
    let g = heads_first(g)?;
    let beta = heads_first(beta)?;
    let n = (seq_len + pad) / chunk;

    let q = q.reshape((b, h, n, chunk, dk))?;
    let k = k.reshape((b, h, n, chunk, dk))?;
    let v = v.reshape((b, h, n, chunk, dv))?;
    let g = g.reshape((b, h, n, chunk))?;
    let beta = beta.reshape((b, h, n, chunk, 1))?;

    // cum[t] = Σ_{i<=t} g_i within the chunk; decay[i][j] = exp(cum_i − cum_j)
    // for j <= i (the decay accumulated from token j to token i), 0 above the
    // diagonal. `g <= 0` makes every kept entry `<= 0` before the exp; the
    // clamp only keeps the masked-out upper half from overflowing to inf
    // (inf · 0 would be NaN).
    // `Tensor::cumsum` is a broadcast matmul against a stride-0 view of the
    // triangular matrix, which is not a layout every backend's matmul accepts;
    // `bmm` materializes the operand instead.
    let triu = Tensor::triu2(chunk, DType::F32, dev)?; // 1 for i <= j
    let cum = bmm(&g.unsqueeze(3)?, &triu.expand((b, h, n, chunk, chunk))?)?.squeeze(3)?; // (b, h, n, c)
    let tril = Tensor::tril2(chunk, DType::F32, dev)?; // 1 for j <= i
    let eye = Tensor::eye(chunk, DType::F32, dev)?;
    let strict_lower = (&tril - &eye)?;
    let decay = cum
        .unsqueeze(4)?
        .broadcast_sub(&cum.unsqueeze(3)?)? // [i][j] = cum_i − cum_j
        .clamp(f32::MIN, 0f32)?
        .exp()?
        .broadcast_mul(&tril)?; // (b, h, n, c, c)

    let k_beta = k.broadcast_mul(&beta)?;
    let v_beta = v.broadcast_mul(&beta)?;
    let k_t = k.transpose(3, 4)?.contiguous()?; // (b, h, n, dk, c)
    let ut = bmm(&k_beta, &k_t)?.mul(&decay)?; // (b, h, n, c, c)
    let attn = bmm(&q, &k_t)?.mul(&decay)?; // (b, h, n, c, c)

    // Solve (I + L) x = rhs for both right-hand sides at once, L the strictly
    // lower part of `ut`: x_i = rhs_i − Σ_{j<i} L_ij x_j, one row per step,
    // batched over every chunk and head. The inverse is never formed — with
    // near-duplicate keys and β ≈ 1 its entries reach 2^(c−1), and multiplying
    // through it in f32 is what `chunked_survives_correlated_keys` guards
    // against. Forward substitution on the right-hand side stays bounded by the
    // solution, which the recurrence already bounds.
    let cum_exp = cum.exp()?.unsqueeze(4)?; // (b, h, n, c, 1)
    let l = ut.broadcast_mul(&strict_lower)?; // (b, h, n, c, c)
    let rhs = Tensor::cat(&[&v_beta, &k_beta.broadcast_mul(&cum_exp)?], 4)?; // (b, h, n, c, dv + dk)
    let mut rows: Vec<Tensor> = Vec::with_capacity(chunk);
    rows.push(rhs.narrow(3, 0, 1)?);
    for i in 1..chunk {
        let solved = Tensor::cat(&rows, 3)?; // (b, h, n, i, dv + dk)
        let l_row = l.narrow(3, i, 1)?.narrow(4, 0, i)?; // (b, h, n, 1, i)
        rows.push((rhs.narrow(3, i, 1)? - bmm(&l_row, &solved)?)?);
    }
    let solved = Tensor::cat(&rows, 3)?; // (b, h, n, c, dv + dk)
    let u = solved.narrow(4, 0, dv)?.contiguous()?; // the chunk's writes
    let w = solved.narrow(4, dv, dk)?.contiguous()?; // its reads of S
    let q_dec = q.broadcast_mul(&cum_exp)?;
    let cum_last = cum.narrow(3, chunk - 1, 1)?; // (b, h, n, 1)
    let k_dec = k.broadcast_mul(&cum_last.broadcast_sub(&cum)?.exp()?.unsqueeze(4)?)?;
    let chunk_decay = cum_last.exp()?.unsqueeze(4)?; // (b, h, n, 1, 1)

    // The sequential part: one state hand-off per chunk.
    let mut s = state;
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let at = |t: &Tensor| -> Result<Tensor> { t.narrow(2, i, 1)?.squeeze(2)?.contiguous() };
        let v_new = (at(&u)? - bmm(&at(&w)?, &s)?)?; // (b, h, c, dv)
        let inter = bmm(&at(&q_dec)?, &s)?; // (b, h, c, dv)
        outs.push((inter + bmm(&at(&attn)?, &v_new)?)?);
        s = (s.broadcast_mul(&at(&chunk_decay)?)? + bmm(&at(&k_dec)?.transpose(2, 3)?, &v_new)?)?;
    }
    let out = Tensor::cat(&outs, 2)? // (b, h, n·c, dv)
        .narrow(2, 0, seq_len)?
        .transpose(1, 2)?
        .contiguous()?; // (b, s, h, dv)
    Ok((out, s))
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Random inputs shaped like a real layer's: unit-norm scaled q/k, a
    /// negative log-decay, a write gate in (0, 1).
    fn inputs(
        b: usize,
        s: usize,
        h: usize,
        dk: usize,
        dv: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (b, s, h, dk), &dev).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, s, h, dk), &dev).unwrap();
        let q = (l2_normalize(&q).unwrap() * (dk as f64).powf(-0.5)).unwrap();
        let k = l2_normalize(&k).unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, s, h, dv), &dev).unwrap();
        let g = softplus(&Tensor::randn(0f32, 1.0, (b, s, h), &dev).unwrap())
            .unwrap()
            .neg()
            .unwrap();
        let beta =
            candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0, (b, s, h), &dev).unwrap()).unwrap();
        (q, k, v, g, beta)
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

    /// The chunked form must be the recurrence, regrouped — same read-out,
    /// same final state — at lengths below, at, and across chunk boundaries,
    /// with and without a state to start from.
    #[test]
    fn chunked_matches_the_recurrence() {
        let (b, h, dk, dv) = (2, 3, 8, 6);
        for &s in &[1usize, 3, 63, 64, 65, 130] {
            for with_state in [false, true] {
                let (q, k, v, g, beta) = inputs(b, s, h, dk, dv);
                let s0 = if with_state {
                    Tensor::randn(0f32, 0.5, (b, h, dk, dv), &Device::Cpu).unwrap()
                } else {
                    Tensor::zeros((b, h, dk, dv), DType::F32, &Device::Cpu).unwrap()
                };
                let (o_ref, s_ref) =
                    recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone()).unwrap();
                let (o_chk, s_chk) =
                    chunk_gated_delta_rule(&q, &k, &v, &g, &beta, s0, DELTA_CHUNK).unwrap();
                assert_eq!(o_chk.dims(), &[b, s, h, dv]);
                let d_out = max_abs_diff(&o_ref, &o_chk);
                let d_state = max_abs_diff(&s_ref, &s_chk);
                assert!(d_out < 1e-4, "s={s} state={with_state}: out diff {d_out}");
                assert!(
                    d_state < 1e-4,
                    "s={s} state={with_state}: state diff {d_state}"
                );
            }
        }
    }

    /// A different chunk changes both loop lengths and the padding; the
    /// answer must not depend on it.
    #[test]
    fn chunk_size_does_not_change_the_answer() {
        let (q, k, v, g, beta) = inputs(1, 50, 2, 8, 8);
        let s0 = Tensor::zeros((1, 2, 8, 8), DType::F32, &Device::Cpu).unwrap();
        let (o_ref, _) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone()).unwrap();
        for chunk in [4usize, 16, 128] {
            let (o, _) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone(), chunk).unwrap();
            assert!(max_abs_diff(&o_ref, &o) < 1e-4, "chunk {chunk}");
        }
    }

    /// Splitting a sequence across two calls — a prompt then a decode token,
    /// state carried between — must equal one call over the whole thing.
    #[test]
    fn state_carries_across_calls() {
        let (q, k, v, g, beta) = inputs(1, 70, 2, 8, 8);
        let (o_all, s_all) = gated_delta_rule(&q, &k, &v, &g, &beta, None).unwrap();
        let cut = 69;
        let head = |t: &Tensor| t.narrow(1, 0, cut).unwrap();
        let tail = |t: &Tensor| t.narrow(1, cut, 70 - cut).unwrap();
        let (o1, s1) = gated_delta_rule(
            &head(&q),
            &head(&k),
            &head(&v),
            &head(&g),
            &head(&beta),
            None,
        )
        .unwrap();
        let (o2, s2) = gated_delta_rule(
            &tail(&q),
            &tail(&k),
            &tail(&v),
            &tail(&g),
            &tail(&beta),
            Some(s1),
        )
        .unwrap();
        let o12 = Tensor::cat(&[o1, o2], 1).unwrap();
        assert!(max_abs_diff(&o_all, &o12) < 1e-4);
        assert!(max_abs_diff(&s_all, &s2) < 1e-4);
    }

    /// The chunked form on an accelerator against the recurrence on the CPU,
    /// at a real layer's shape. The CPU tests above cannot see a backend
    /// whose matmul mishandles a layout the CPU path accepts.
    #[test]
    fn chunked_matches_the_recurrence_on_the_accelerator() {
        let Some(dev) = accelerator() else {
            eprintln!("no accelerator compiled in; skipping");
            return;
        };
        let (b, s, h, dk, dv) = (1, 300, 32, 128, 128);
        let (q, k, v, g, beta) = inputs(b, s, h, dk, dv);
        let s0 = Tensor::zeros((b, h, dk, dv), DType::F32, &Device::Cpu).unwrap();
        let (o_ref, s_ref) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone()).unwrap();
        let to = |t: &Tensor| t.to_device(&dev).unwrap();
        let (o_acc, s_acc) = chunk_gated_delta_rule(
            &to(&q),
            &to(&k),
            &to(&v),
            &to(&g),
            &to(&beta),
            to(&s0),
            DELTA_CHUNK,
        )
        .unwrap();
        let o_acc = o_acc.to_device(&Device::Cpu).unwrap();
        let s_acc = s_acc.to_device(&Device::Cpu).unwrap();
        let d_out = max_abs_diff(&o_ref, &o_acc);
        let d_state = max_abs_diff(&s_ref, &s_acc);
        assert!(d_out.is_finite() && d_out < 1e-3, "out diff {d_out}");
        assert!(
            d_state.is_finite() && d_state < 1e-3,
            "state diff {d_state}"
        );
    }

    /// The same comparison at the shapes and dtypes a real prompt arrives in:
    /// half-precision activations, a 512-token forward chunk, the 9B's 32
    /// heads, a decay of realistic magnitude and a non-zero starting state.
    #[test]
    fn chunked_matches_the_recurrence_on_the_accelerator_in_half_precision() {
        let Some(dev) = accelerator() else {
            eprintln!("no accelerator compiled in; skipping");
            return;
        };
        let (b, s, h, dk, dv) = (1, 512, 32, 128, 128);
        let (q, k, v, g, beta) = inputs(b, s, h, dk, dv);
        // Real layers see decays of tens per token, not ~1.
        let g = (g * 30.0).unwrap();
        let s0 = Tensor::randn(0f32, 0.5, (b, h, dk, dv), &Device::Cpu).unwrap();
        let (o_ref, s_ref) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone()).unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let to = |t: &Tensor| t.to_dtype(dtype).unwrap().to_device(&dev).unwrap();
            // Through the public entry, which is what casts the carried state
            // up to f32 the way the model path does.
            let (o_acc, s_acc) = gated_delta_rule(
                &to(&q),
                &to(&k),
                &to(&v),
                &to(&g),
                &to(&beta),
                Some(to(&s0)),
            )
            .unwrap();
            let o_acc = o_acc.to_device(&Device::Cpu).unwrap();
            let s_acc = s_acc.to_device(&Device::Cpu).unwrap();
            let d_out = max_abs_diff(&o_ref, &o_acc);
            let d_state = max_abs_diff(&s_ref, &s_acc);
            // Half-precision inputs move the answer; NaN or inf is a bug.
            assert!(d_out.is_finite(), "{dtype:?}: out diff {d_out}");
            assert!(d_state.is_finite(), "{dtype:?}: state diff {d_state}");
            assert!(d_out < 5e-2, "{dtype:?}: out diff {d_out}");
        }
    }

    /// Near-duplicate keys with a write gate near 1 and little decay — a
    /// prompt full of repeated tokens — make the intra-chunk system
    /// ill-conditioned. The inverse must still be computed stably.
    #[test]
    fn chunked_survives_correlated_keys() {
        let dev = Device::Cpu;
        let (b, s, h, dk, dv) = (1, 256, 2, 16, 16);
        let base = Tensor::randn(0f32, 1.0, (1, 1, h, dk), &dev).unwrap();
        let noise = Tensor::randn(0f32, 0.02, (b, s, h, dk), &dev).unwrap();
        let k = l2_normalize(&noise.broadcast_add(&base).unwrap()).unwrap();
        let q = (l2_normalize(&Tensor::randn(0f32, 1.0, (b, s, h, dk), &dev).unwrap()).unwrap()
            * (dk as f64).powf(-0.5))
        .unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, s, h, dv), &dev).unwrap();
        let g = Tensor::full(-0.01f32, (b, s, h), &dev).unwrap();
        let beta = Tensor::full(0.999f32, (b, s, h), &dev).unwrap();
        let s0 = Tensor::zeros((b, h, dk, dv), DType::F32, &dev).unwrap();
        let (o_ref, s_ref) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, s0.clone()).unwrap();
        let (o_chk, s_chk) =
            chunk_gated_delta_rule(&q, &k, &v, &g, &beta, s0, DELTA_CHUNK).unwrap();
        let d_out = max_abs_diff(&o_ref, &o_chk);
        let d_state = max_abs_diff(&s_ref, &s_chk);
        let scale = o_ref
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            d_out.is_finite() && d_out < 1e-3 * scale.max(1.0),
            "out diff {d_out} (max |out| {scale})"
        );
        assert!(
            d_state.is_finite() && d_state < 1e-3,
            "state diff {d_state}"
        );
    }

    fn accelerator() -> Option<Device> {
        if candle_core::utils::metal_is_available() {
            return Device::new_metal(0).ok();
        }
        if candle_core::utils::cuda_is_available() {
            return Device::new_cuda(0).ok();
        }
        None
    }

    /// The shifted-sum conv equals the per-token windowed one it replaced,
    /// and leaves the same `k-1` inputs behind as state.
    #[test]
    fn causal_conv1d_matches_the_windowed_form() {
        let dev = Device::Cpu;
        let (b, s, c, k) = (2, 9, 5, 4);
        let x = Tensor::randn(0f32, 1.0, (b, s, c), &dev).unwrap();
        let w = Tensor::randn(0f32, 1.0, (k, c), &dev).unwrap();
        let prev = Tensor::randn(0f32, 1.0, (b, k - 1, c), &dev).unwrap();

        let mut state = RecurrentState::new();
        state.conv_state = Some(prev.clone());
        let got = causal_conv1d(&x, &w, &mut state).unwrap();

        let padded = Tensor::cat(&[&prev, &x], 1).unwrap();
        let mut outs = Vec::new();
        for t in 0..s {
            let window = padded.narrow(1, t, k).unwrap();
            outs.push(
                window
                    .broadcast_mul(&w)
                    .unwrap()
                    .sum(1)
                    .unwrap()
                    .unsqueeze(1)
                    .unwrap(),
            );
        }
        let want = candle_nn::ops::silu(&Tensor::cat(&outs, 1).unwrap()).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-5);
        let kept = state.conv_state.unwrap();
        assert!(max_abs_diff(&kept, &x.narrow(1, s - (k - 1), k - 1).unwrap()) == 0.0);
    }
}
