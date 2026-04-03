use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::kv_cache::KvCache;
use crate::norm::Norm;
use crate::pos_enc::RoPE;

/// Configuration for standard (MHA/GQA/MQA) attention.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Whether projection layers have bias (GPT-OSS: true, most others: false).
    pub attn_bias: bool,
    /// Logit softcapping: tanh(scores / cap) * cap (Gemma 4: 30.0).
    pub attn_logit_softcapping: Option<f64>,
    /// If true, K and V share the same projection (Gemma 4 global layers).
    pub shared_kv: bool,
    /// If true, apply RMSNorm to Q per-head before RoPE (Gemma 4).
    pub q_norm: bool,
    /// If true, apply RMSNorm to K per-head before RoPE (Gemma 4).
    pub k_norm: bool,
    /// If true, apply RMSNorm (no learnable scale) to V per-head before caching (Gemma 4).
    pub v_norm: bool,
    pub q_norm_eps: f64,
    /// If true, load per-head sink logits and append to scores before softmax (GPT-OSS).
    pub attn_sinks: bool,
    /// Explicit attention score scale. If None, uses 1/sqrt(head_dim).
    /// Set to Some(1.0) for Gemma 4 (q_norm handles effective scaling).
    pub scale: Option<f64>,
    /// Qwen3.5: Q projection is 2×, split into query + output gate.
    /// After attention: output = attn_out * sigmoid(gate), then o_proj.
    pub q_output_gate: bool,
    /// Qwen3.5: q_norm / k_norm weights use zeros-init + (1+w) formula.
    /// Set to true when the model's norm weights are stored as deltas from 0.
    pub norm_one_plus: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_q_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            attn_bias: false,
            attn_logit_softcapping: None,
            shared_kv: false,
            q_norm: false,
            k_norm: false,
            v_norm: false,
            q_norm_eps: 1e-6,
            attn_sinks: false,
            scale: None,
            q_output_gate: false,
            norm_one_plus: false,
        }
    }
}

/// Apply RMSNorm without a learnable scale (Gemma 4 v_norm).
/// Input shape: (batch, n_heads, seq, head_dim) — norms over the last dim.
fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let sq_mean = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(sq_mean + eps)?.sqrt()?)?;
    normed.to_dtype(orig_dtype)
}

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Option<Linear>, // None when shared_kv (K=V)
    o_proj: Linear,
    q_norm: Option<Norm>,
    k_norm: Option<Norm>,
    /// Per-head sink logit appended to attention scores before softmax (GPT-OSS).
    sinks: Option<Tensor>,
    cfg: AttentionConfig,
}

impl Attention {
    pub fn new(cfg: AttentionConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = cfg.num_q_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let mk_linear = |in_d, out_d, name| {
            if cfg.attn_bias {
                linear(in_d, out_d, vb.pp(name))
            } else {
                linear_no_bias(in_d, out_d, vb.pp(name))
            }
        };
        // Qwen3.5: Q proj is 2× to provide both query and output gate
        let q_proj_dim = if cfg.q_output_gate { q_dim * 2 } else { q_dim };
        let q_proj = mk_linear(cfg.hidden_size, q_proj_dim, "q_proj")?;
        let k_proj = mk_linear(cfg.hidden_size, kv_dim, "k_proj")?;
        let v_proj = if cfg.shared_kv {
            None
        } else {
            Some(mk_linear(cfg.hidden_size, kv_dim, "v_proj")?)
        };
        let o_proj = mk_linear(q_dim, cfg.hidden_size, "o_proj")?;

        let mk_norm = |size, name| -> Result<Norm> {
            if cfg.norm_one_plus {
                Norm::rms_one_plus(size, cfg.q_norm_eps, vb.pp(name))
            } else {
                Norm::rms(size, cfg.q_norm_eps, vb.pp(name))
            }
        };
        let q_norm = if cfg.q_norm { Some(mk_norm(cfg.head_dim, "q_norm")?) } else { None };
        let k_norm = if cfg.k_norm { Some(mk_norm(cfg.head_dim, "k_norm")?) } else { None };

        let sinks = if cfg.attn_sinks {
            Some(vb.get((cfg.num_q_heads,), "sinks")?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            sinks,
            cfg,
        })
    }

    /// Forward pass.
    /// - `x`: (batch, seq_len, hidden_size)
    /// - `mask`: (seq_len, total_len) with 0.0 / -inf
    /// Returns: (batch, seq_len, hidden_size)
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let h = self.cfg.num_q_heads;
        let h_kv = self.cfg.num_kv_heads;
        let d = self.cfg.head_dim;

        // Project Q, K, V
        let q_raw = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = match &self.v_proj {
            Some(v_proj) => v_proj.forward(x)?,
            None => k.clone(), // shared K=V
        };

        // Qwen3.5: split Q proj into query and output gate
        let (q, attn_gate) = if self.cfg.q_output_gate {
            // q_raw: (b, s, h * d * 2) → view as (b, s, h, d*2) → split last dim
            let qg = q_raw.reshape((b, seq_len, h, d * 2))?;
            let q_part = qg.narrow(3, 0, d)?;         // (b, s, h, d)
            let gate = qg.narrow(3, d, d)?             // (b, s, h, d)
                .reshape((b, seq_len, h * d))?;         // (b, s, h*d)
            (q_part.transpose(1, 2)?, Some(gate))
        } else {
            (q_raw.reshape((b, seq_len, h, d))?.transpose(1, 2)?, None)
        };

        // Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        let q = q;
        let k = k.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;

        // Optional Q normalization (Gemma 4)
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };

        // Optional K normalization (Gemma 4)
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        // Apply RoPE to Q and K (tensors must be contiguous after transpose)
        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        // Optional V normalization (Gemma 4, no learnable scale)
        let v = if self.cfg.v_norm {
            rms_norm_no_scale(&v, self.cfg.q_norm_eps)?
        } else {
            v
        };

        // Update KV cache
        let (k, v) = kv_cache.append(&k, &v)?;

        // Repeat KV heads for GQA: (batch, h_kv, total, d) -> (batch, h, total, d)
        let (k, v) = if h != h_kv {
            let rep = h / h_kv;
            let k = k
                .unsqueeze(2)?
                .expand((b, h_kv, rep, k.dim(2)?, d))?
                .reshape((b, h, k.dim(2)?, d))?;
            let v = v
                .unsqueeze(2)?
                .expand((b, h_kv, rep, v.dim(2)?, d))?
                .reshape((b, h, v.dim(2)?, d))?;
            (k, v)
        } else {
            (k, v)
        };

        // Attention scores: (batch, h, seq, total)
        let scale = self.cfg.scale.unwrap_or(1.0 / (d as f64).sqrt());
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        // Optional logit softcapping (Gemma 4)
        if let Some(cap) = self.cfg.attn_logit_softcapping {
            scores = ((scores * (1.0 / cap))?.tanh()? * cap)?;
        }

        // Apply mask (cast to scores dtype to handle f16/f32 mismatch)
        if let Some(mask) = mask {
            let mask = mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
            scores = scores.broadcast_add(&mask)?;
        }

        // Attention sinks (GPT-OSS): append per-head sink logit, softmax, drop last col.
        let total_len = scores.dim(D::Minus1)?;
        let attn_weights = if let Some(sinks) = &self.sinks {
            // sinks: [n_heads] -> [1, n_heads, 1, 1] -> [b, n_heads, seq_len, 1]
            let s = sinks.reshape((1, h, 1, 1))?.expand((b, h, seq_len, 1))?.contiguous()?;
            let combined = Tensor::cat(&[&scores, &s], D::Minus1)?;
            let probs = candle_nn::ops::softmax_last_dim(&combined)?;
            probs.narrow(D::Minus1, 0, total_len)?
        } else {
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        let attn_out = attn_weights.matmul(&v)?; // (batch, h, seq, d)

        // Reshape back: (batch, h, seq, d) -> (batch, seq, h*d)
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;

        // Qwen3.5 output gate: attn_out * sigmoid(gate)
        let attn_out = if let Some(gate) = attn_gate {
            (attn_out * candle_nn::ops::sigmoid(&gate)?)?
        } else {
            attn_out
        };

        self.o_proj.forward(&attn_out)
    }

    /// KV-shared attention: project Q from current input, read K/V from cache without appending.
    /// Used by Gemma 4 shared layers which reuse earlier layers' K/V.
    pub fn forward_shared(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let h = self.cfg.num_q_heads;
        let h_kv = self.cfg.num_kv_heads;
        let d = self.cfg.head_dim;

        let q = self.q_proj.forward(x)?
            .reshape((b, seq_len, h, d))?.transpose(1, 2)?;
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let q = rope.apply(&q.contiguous()?, pos)?;

        // Read K, V from shared source cache — no append.
        let (k, v) = kv_cache
            .current_kv()
            .ok_or_else(|| candle_core::Error::Msg("shared KV cache is empty".into()))?;

        // k: (b, h_kv, total, d) — already includes the current token from the source layer.
        let total = k.dim(2)?;
        let (k, v) = if h != h_kv {
            let rep = h / h_kv;
            let k = k.unsqueeze(2)?
                .expand((b, h_kv, rep, total, d))?.contiguous()?
                .reshape((b, h, total, d))?;
            let v = v.unsqueeze(2)?
                .expand((b, h_kv, rep, total, d))?.contiguous()?
                .reshape((b, h, total, d))?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        let scale = self.cfg.scale.unwrap_or(1.0 / (d as f64).sqrt());
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        if let Some(mask) = mask {
            let mask = mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
            scores = scores.broadcast_add(&mask)?;
        }
        let attn_out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;
        self.o_proj.forward(&attn_out)
    }
}
