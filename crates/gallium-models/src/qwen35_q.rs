//! Quantized Qwen 3.5 model loaded from GGUF.
//!
//! Tensor naming in the GGUF:
//!   Linear attention layers:  blk.{i}.attn_qkv, attn_gate, ssm_alpha, ssm_beta,
//!                             ssm_out, ssm_conv1d, ssm_a, ssm_dt.bias, ssm_norm
//!   Full attention layers:    blk.{i}.attn_q (2×, gate fused), attn_k, attn_v,
//!                             attn_output, attn_q_norm, attn_k_norm
//!   Both:                     blk.{i}.attn_norm, post_attention_norm, ffn_{gate,up,down}

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;

use gallium_core::quantized::{GgufMetadata, QLinear, QNorm, QVarBuilder};
use gallium_core::*;

// -- Quantized full Attention ------------------------------------------------
// Handles q_output_gate (2× q_proj) and per-head q_norm / k_norm.

struct QAttention {
    q_proj: QLinear,  // out_dim = n_q_heads * head_dim * 2 (query + gate fused)
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: QNorm,
    k_norm: QNorm,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QAttention {
    fn load(vb: &QVarBuilder, num_q_heads: usize, num_kv_heads: usize, head_dim: usize, rms_eps: f64) -> Result<Self> {
        Ok(Self {
            q_proj:  QLinear::load(&vb.pp("attn_q"))?,
            k_proj:  QLinear::load(&vb.pp("attn_k"))?,
            v_proj:  QLinear::load(&vb.pp("attn_v"))?,
            o_proj:  QLinear::load(&vb.pp("attn_output"))?,
            q_norm:  QNorm::rms_load(rms_eps, &vb.pp("attn_q_norm"))?,
            k_norm:  QNorm::rms_load(rms_eps, &vb.pp("attn_k_norm"))?,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let h    = self.num_q_heads;
        let h_kv = self.num_kv_heads;
        let d    = self.head_dim;

        // q_proj is 2×: [query | gate] concatenated along the last dim.
        let q_raw = self.q_proj.forward(x)?;                         // (b, s, h*d*2)
        let qg    = q_raw.reshape((b, seq_len, h, d * 2))?;
        let q_part = qg.narrow(3, 0, d)?;                            // (b, s, h, d)
        let gate   = qg.narrow(3, d, d)?.reshape((b, seq_len, h * d))?; // (b, s, h*d)
        let q = q_part.transpose(1, 2)?;                              // (b, h, s, d)

        let k = self.k_proj.forward(x)?.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;

        // Per-head norms then RoPE (transpose produces non-contiguous views)
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // GQA head expansion
        let (k, v) = if h != h_kv {
            let rep  = h / h_kv;
            let total = k.dim(2)?;
            let k = k.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
            let v = v.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
            (k, v)
        } else {
            (k, v)
        };

        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let attn_out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;

        // Output gate: attn_out * sigmoid(gate)
        let attn_out = (attn_out * candle_nn::ops::sigmoid(&gate)?)?;

        self.o_proj.forward(&attn_out)
    }
}

// -- Quantized GatedDeltaNet -------------------------------------------------

struct QGatedDeltaNet {
    in_proj_qkv: QLinear, // attn_qkv:   hidden → key_dim*2 + value_dim
    in_proj_z:   QLinear, // attn_gate:  hidden → value_dim
    in_proj_b:   QLinear, // ssm_beta:   hidden → n_v_heads
    in_proj_a:   QLinear, // ssm_alpha:  hidden → n_v_heads
    out_proj:    QLinear, // ssm_out:    value_dim → hidden
    conv_weight: Tensor,  // ssm_conv1d: (conv_k, conv_dim) — dequantized F32
    a_log:       Tensor,  // ssm_a:      (n_v_heads,) F32  — stored as -exp(A_log) in GGUF
    dt_bias:     Tensor,  // ssm_dt.bias:(n_v_heads,) F32
    norm_weight: Tensor,  // ssm_norm.weight: (dv,) F32
    n_k: usize,
    n_v: usize,
    dk: usize,
    dv: usize,
    conv_k: usize,
    rms_eps: f64,
}

impl QGatedDeltaNet {
    fn load(
        vb: &QVarBuilder,
        n_k: usize,
        n_v: usize,
        dk: usize,
        dv: usize,
        conv_k: usize,
        rms_eps: f64,
    ) -> Result<Self> {
        let dev = vb.device();
        Ok(Self {
            in_proj_qkv: QLinear::from_arc(vb.get("attn_qkv.weight")?,  None)?,
            in_proj_z:   QLinear::from_arc(vb.get("attn_gate.weight")?, None)?,
            in_proj_b:   QLinear::from_arc(vb.get("ssm_beta.weight")?,  None)?,
            in_proj_a:   QLinear::from_arc(vb.get("ssm_alpha.weight")?, None)?,
            out_proj:    QLinear::from_arc(vb.get("ssm_out.weight")?,   None)?,
            conv_weight: vb.get("ssm_conv1d.weight")?.dequantize(dev)?, // (conv_k, conv_dim)
            a_log:       vb.get("ssm_a")?.dequantize(dev)?,
            dt_bias:     vb.get("ssm_dt.bias")?.dequantize(dev)?,
            norm_weight: vb.get("ssm_norm.weight")?.dequantize(dev)?,
            n_k, n_v, dk, dv, conv_k, rms_eps,
        })
    }

    fn forward(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let n_k = self.n_k;
        let n_v = self.n_v;
        let dk  = self.dk;
        let dv  = self.dv;
        let key_dim   = n_k * dk;
        let value_dim = n_v * dv;

        // 1. Project + causal conv + SiLU on QKV
        let mixed = self.in_proj_qkv.forward(x)?;
        let mixed = self.apply_causal_conv(&mixed, state)?; // (b, s, key_dim*2+value_dim)

        // 2. Split Q, K, V
        let q = mixed.narrow(2, 0,               key_dim)?;
        let k = mixed.narrow(2, key_dim,          key_dim)?;
        let v = mixed.narrow(2, key_dim * 2, value_dim)?;

        // 3. Gate projections
        let z     = self.in_proj_z.forward(x)?;      // (b, s, value_dim)
        let b_raw = self.in_proj_b.forward(x)?;      // (b, s, n_v)
        let a_raw = self.in_proj_a.forward(x)?;      // (b, s, n_v)

        let beta = candle_nn::ops::sigmoid(&b_raw)?; // (b, s, n_v)

        // g = ssm_a * softplus(a + dt_bias)
        // NOTE: GGUF stores ssm_a = -exp(A_log) pre-computed (see convert_hf_to_gguf.py:4759),
        // so no exp() or neg() needed here — just multiply directly.
        let a_f32    = a_raw.to_dtype(DType::F32)?;
        let dt_f32   = self.dt_bias.to_dtype(DType::F32)?;
        let alog_f32 = self.a_log.to_dtype(DType::F32)?;
        let a_plus_dt = a_f32.broadcast_add(&dt_f32)?;
        let g = alog_f32.broadcast_mul(&softplus(&a_plus_dt)?)?.to_dtype(x.dtype())?;

        // 4. Reshape to (b, s, n_heads, head_dim)
        let q = q.reshape((b, seq_len, n_k, dk))?;
        let k = k.reshape((b, seq_len, n_k, dk))?;
        let v = v.reshape((b, seq_len, n_v, dv))?;

        // 5. L2 normalize Q and K
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // 6. GQA: expand Q, K from n_k heads to n_v heads
        let (q, k) = if n_v > n_k {
            let rep = n_v / n_k;
            let q = q.unsqueeze(3)?.expand((b, seq_len, n_k, rep, dk))?.contiguous()?.reshape((b, seq_len, n_v, dk))?;
            let k = k.unsqueeze(3)?.expand((b, seq_len, n_k, rep, dk))?.contiguous()?.reshape((b, seq_len, n_v, dk))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. Scale Q by 1/sqrt(dk)
        let q = (q * (dk as f64).powf(-0.5))?;

        // 8. Recurrent gated delta rule
        let mut s = match state.state.take() {
            Some(s) => s.to_dtype(DType::F32)?,
            None    => Tensor::zeros((b, n_v, dk, dv), DType::F32, x.device())?,
        };

        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t    = q.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_v, dk)
            let k_t    = k.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
            let v_t    = v.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_v, dv)
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_v)
            let g_t    = g.narrow(1, t, 1)?.squeeze(1)?.to_dtype(DType::F32)?;

            // Decay: S = S * exp(g)
            let decay = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?; // (b, n_v, 1, 1)
            s = s.broadcast_mul(&decay.exp()?)?;

            // kv_mem = S^T @ k_t
            let kv_mem = s.broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?; // (b, n_v, dv)

            // delta = (v - kv_mem) * beta
            let delta = (v_t - &kv_mem)?.broadcast_mul(&beta_t.unsqueeze(D::Minus1)?)?;

            // Write: S += k outer delta
            let write = k_t.unsqueeze(D::Minus1)?.broadcast_mul(&delta.unsqueeze(D::Minus2)?)?; // (b, n_v, dk, dv)
            s = (s + write)?;

            // Read: o = S^T @ q_t
            let o_t = s.broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?.sum(D::Minus2)?; // (b, n_v, dv)
            outs.push(o_t.unsqueeze(1)?);
        }
        state.state = Some(s.to_dtype(x.dtype())?);

        // (b, seq, n_v, dv) → flatten heads
        let output = Tensor::cat(&outs, 1)?.to_dtype(x.dtype())?;

        // 9. RMSNormGated: rms_norm(output) * norm_weight * silu(z)
        let output_flat = output.reshape((b * seq_len * n_v, dv))?;
        let z_flat      = z.reshape((b * seq_len * n_v, dv))?;

        let normed = self.rms_norm_gated(&output_flat, &z_flat)?;
        let output = normed.reshape((b, seq_len, value_dim))?;

        self.out_proj.forward(&output)
    }

    /// Gated RMSNorm: rms_norm(x) * weight * silu(gate).
    /// Matches Python Qwen3_5RMSNormGated (norm-first, then gate).
    fn rms_norm_gated(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let orig = x.dtype();
        let xf   = x.to_dtype(DType::F32)?;
        let var  = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf.broadcast_div(&(var + self.rms_eps)?.sqrt()?)?;
        let w = self.norm_weight.to_dtype(DType::F32)?;
        let normed = normed.broadcast_mul(&w)?;
        (normed * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?.to_dtype(orig)
    }

    /// Causal depthwise conv1d with SiLU.
    /// conv_weight in GGUF is (conv_k, conv_dim) — used directly.
    fn apply_causal_conv(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let (b, seq_len, conv_dim) = x.dims3()?;
        let k = self.conv_k;

        let padded = match state.conv_state.take() {
            Some(prev) => Tensor::cat(&[&prev, x], 1)?,
            None => {
                let pad = Tensor::zeros((b, k - 1, conv_dim), x.dtype(), x.device())?;
                Tensor::cat(&[&pad, x], 1)?
            }
        };

        let total = padded.dim(1)?;
        state.conv_state = Some(padded.narrow(1, total - (k - 1), k - 1)?);

        // GGUF stores conv weight as (conv_dim, k); transpose to (k, conv_dim).
        let w = self.conv_weight.t()?.contiguous()?.to_dtype(x.dtype())?; // (k, conv_dim)
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(1, t, k)?;            // (b, k, conv_dim)
            let out    = window.broadcast_mul(&w)?.sum(1)?;  // (b, conv_dim)
            outs.push(out.unsqueeze(1)?);
        }
        candle_nn::ops::silu(&Tensor::cat(&outs, 1)?)
    }
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm_sq = x.sqr()?.sum_keepdim(D::Minus1)?;
    let norm    = (norm_sq + 1e-6_f64)?.sqrt()?;
    x.broadcast_div(&norm)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0_f64)?.log()
}

// -- Quantized GatedFFN ------------------------------------------------------

struct QGatedFFN {
    gate_proj: QLinear,
    up_proj:   QLinear,
    down_proj: QLinear,
}

impl QGatedFFN {
    fn load(vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::from_arc(vb.get("ffn_gate.weight")?, None)?,
            up_proj:   QLinear::from_arc(vb.get("ffn_up.weight")?,   None)?,
            down_proj: QLinear::from_arc(vb.get("ffn_down.weight")?, None)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up   = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// -- Per-layer attention dispatch --------------------------------------------

enum QLayerAttn {
    Full(QAttention),
    Linear(QGatedDeltaNet),
}

struct QTransformerBlock {
    pre_attn_norm:  QNorm,
    attn:           QLayerAttn,
    post_attn_norm: QNorm,
    ffn:            QGatedFFN,
}

impl QTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: Option<&mut KvCache>,
        recurrent: Option<&mut RecurrentState>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.pre_attn_norm.forward(&x.contiguous()?)?;
        let attn_out = match &self.attn {
            QLayerAttn::Full(attn) => {
                let kv = kv_cache.expect("full attention requires KV cache");
                attn.forward(&normed, rope, pos, kv, mask)?
            }
            QLayerAttn::Linear(delta) => {
                let rec = recurrent.expect("linear attention requires recurrent state");
                delta.forward(&normed, rec)?
            }
        };
        let h        = (attn_out + x)?;
        let residual = h.clone();
        let h        = self.post_attn_norm.forward(&h.contiguous()?)?;
        let h        = self.ffn.forward(&h)?;
        h + residual
    }
}

// -- Full Quantized Qwen 3.5 ------------------------------------------------

pub struct Qwen35Q {
    embed_tokens: Embedding,
    blocks:       Vec<QTransformerBlock>,
    final_norm:   QNorm,
    lm_head:      QLinear,
    rope:         RoPE,
    cache:        ModelCache,
    device:       Device,
}

impl Qwen35Q {
    pub fn load(metadata: &GgufMetadata, vb: &QVarBuilder, device: &Device) -> Result<Self> {
        let arch = metadata.get_str("general.architecture").unwrap_or_else(|_| "qwen35".to_string());
        let pfx  = &arch;

        let n_layers  = metadata.get_u32(&format!("{pfx}.block_count"))? as usize;
        let n_heads   = metadata.get_u32(&format!("{pfx}.attention.head_count"))? as usize;
        let n_kv_heads = metadata.get_u32(&format!("{pfx}.attention.head_count_kv"))? as usize;
        let n_embd    = metadata.get_u32(&format!("{pfx}.embedding_length"))? as usize;
        let head_dim  = metadata.get_u32_or(&format!("{pfx}.attention.key_length"), (n_embd / n_heads) as u32) as usize;
        let rope_freq = metadata.get_f32_or(&format!("{pfx}.rope.freq_base"), 10_000_000.0) as f64;
        let rms_eps   = metadata.get_f32_or(&format!("{pfx}.attention.layer_norm_rms_epsilon"), 1e-6) as f64;
        let max_seq   = metadata.get_u32_or(&format!("{pfx}.context_length"), 262144) as usize;
        let fa_interval = metadata.get_u32_or(&format!("{pfx}.full_attention_interval"), 4) as usize;
        let rope_dims = metadata.get_u32_or(&format!("{pfx}.rope.dimension_count"), (head_dim / 4) as u32) as usize;

        // SSM (linear attention) parameters
        let n_k_heads  = metadata.get_u32_or(&format!("{pfx}.ssm.group_count"),    16) as usize;
        let n_v_heads  = metadata.get_u32_or(&format!("{pfx}.ssm.time_step_rank"), 32) as usize;
        let dv         = metadata.get_u32_or(&format!("{pfx}.ssm.state_size"),    128) as usize;
        let value_dim  = metadata.get_u32_or(&format!("{pfx}.ssm.inner_size"),   4096) as usize;
        let conv_k     = metadata.get_u32_or(&format!("{pfx}.ssm.conv_kernel"),      4) as usize;
        let dk         = (value_dim / 2) / n_k_heads; // key_dim = (conv_dim - value_dim) / 2; key_dim / n_k_heads

        // partial_rotary_factor = rope_dims / head_dim
        let partial_rotary = rope_dims as f64 / head_dim as f64;

        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len: max_seq,
                theta: rope_freq,
                partial_rotary_factor: partial_rotary,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        let tok_embd = vb.get("token_embd.weight")?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embd, n_embd);

        // Layer i is full attention iff (i + 1) % fa_interval == 0.
        let mut cache_layers = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let bvb = vb.pp(format!("blk.{i}"));
                let is_full = (i + 1) % fa_interval == 0;
                let attn = if is_full {
                    cache_layers.push(LayerCache::Kv(KvCache::new(max_seq)));
                    QLayerAttn::Full(QAttention::load(&bvb, n_heads, n_kv_heads, head_dim, rms_eps)?)
                } else {
                    cache_layers.push(LayerCache::Recurrent(RecurrentState::new()));
                    QLayerAttn::Linear(QGatedDeltaNet::load(&bvb, n_k_heads, n_v_heads, dk, dv, conv_k, rms_eps)?)
                };
                Ok(QTransformerBlock {
                    pre_attn_norm:  QNorm::rms_load(rms_eps, &bvb.pp("attn_norm"))?,
                    attn,
                    post_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("post_attention_norm"))?,
                    ffn:            QGatedFFN::load(&bvb)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
            QLinear::from_arc(vb.get("token_embd.weight")?, None)?
        };

        Ok(Self {
            embed_tokens,
            blocks,
            final_norm,
            lm_head,
            rope,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
        })
    }
}

impl CausalLM for Qwen35Q {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed_tokens.forward(token_ids)?.contiguous()?;

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = match &block.attn {
                QLayerAttn::Full(_) if seq_len > 1 => {
                    Some(build_causal_mask(seq_len, pos, &self.device)?)
                }
                _ => None,
            };
            let (kv, recurrent) = self.cache.get_layer(i);
            h = block.forward(&h, &self.rope, pos, kv, recurrent, mask.as_ref())?.contiguous()?;
        }

        let h_final = self.final_norm.forward(&h)?;
        let logits = self.lm_head.forward(&h_final.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        logits.to_dtype(DType::F32)
    }

    fn reset(&mut self) { self.cache.reset(); }
    fn device(&self) -> &Device { &self.device }
}
