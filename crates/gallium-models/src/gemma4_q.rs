//! Quantized Gemma 4 model loaded from GGUF.
//!
//! GGUF tensor names differ from safetensors:
//!   attn_norm          → input_layernorm
//!   post_attention_norm → post_attention_layernorm
//!   ffn_norm           → pre_feedforward_layernorm
//!   post_ffw_norm      → post_feedforward_layernorm
//!   inp_gate           → per_layer_input_gate  (PLE)
//!   proj               → per_layer_projection  (PLE)
//!   post_norm          → post_per_layer_input_norm
//!   layer_output_scale → layer_scalar
//!   per_layer_model_proj → per_layer_model_projection
//!   per_layer_proj_norm  → per_layer_projection_norm
//!   per_layer_token_embd → embed_tokens_per_layer

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;

use gallium_core::quantized::{GgufMetadata, QExperts, QLinear, QNorm, QVarBuilder};
use gallium_core::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RMSNorm without a learnable scale (Gemma 4 v_norm). Normalizes over the last dim.
fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let sq_mean = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(sq_mean + eps)?.sqrt()?)?;
    normed.to_dtype(orig)
}

/// Build proportional RoPE inv_freq: `rope_angles` real freqs + `nope_angles` zeros.
/// Matches `_compute_proportional_rope_parameters` in modeling_rope_utils.py.
fn proportional_inv_freq(
    head_dim: usize,
    partial_rotary_factor: f64,
    theta: f64,
) -> Vec<f64> {
    let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
    let nope_angles = head_dim / 2 - rope_angles;
    let mut inv_freq: Vec<f64> = (0..rope_angles)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();
    inv_freq.extend(std::iter::repeat(0.0).take(nope_angles));
    inv_freq
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    /// Absent on shared-K=V layers (26B-A4B global attention): V is then the
    /// raw K projection output (before k_norm/RoPE), mirroring llama.cpp's
    /// `Vcur = wv ? wv(cur) : Kcur`.
    v_proj: Option<QLinear>,
    o_proj: QLinear,
    q_norm: QNorm,
    k_norm: QNorm,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    rms_eps: f64,
}

impl QAttention {
    fn load(
        vb: &QVarBuilder,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rms_eps: f64,
    ) -> Result<Self> {
        let v_proj = if vb.contains("attn_v.weight") {
            Some(QLinear::load(&vb.pp("attn_v"))?)
        } else {
            None
        };
        Ok(Self {
            q_proj: QLinear::load(&vb.pp("attn_q"))?,
            k_proj: QLinear::load(&vb.pp("attn_k"))?,
            v_proj,
            o_proj: QLinear::load(&vb.pp("attn_output"))?,
            q_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_q_norm"))?,
            k_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_k_norm"))?,
            n_q,
            n_kv,
            head_dim,
            rms_eps,
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
        let (b, s, _) = x.dims3()?;
        let (h, h_kv, d) = (self.n_q, self.n_kv, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let k_raw = self.k_proj.forward(x)?.reshape((b, s, h_kv, d))?.transpose(1, 2)?.contiguous()?;
        let v = match &self.v_proj {
            Some(vp) => vp.forward(x)?.reshape((b, s, h_kv, d))?.transpose(1, 2)?.contiguous()?,
            None => k_raw.clone(), // shared K=V: raw K projection, pre-norm/RoPE
        };

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k_raw)?;

        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;
        let v = rms_norm_no_scale(&v, self.rms_eps)?;

        let (k, v) = kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
        let (k, v) = expand_gqa(k, v, h, h_kv, b, d)?;

        // scale = 1.0: q_norm controls effective magnitude
        let mut scores = q.contiguous()?.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        self.o_proj.forward(&out.transpose(1, 2)?.reshape((b, s, h * d))?)
    }

    fn forward_shared(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        src_cache: &KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, h_kv, d) = (self.n_q, self.n_kv, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
        let q = self.q_norm.forward(&q)?;
        let q = rope.apply(&q.contiguous()?, pos)?;

        let (k, v) = src_cache.current_kv()
            .ok_or_else(|| candle_core::Error::Msg("shared KV source is empty".into()))?;

        let total = k.dim(2)?;
        let (k, v) = if h != h_kv {
            let rep = h / h_kv;
            let k = k.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
            let v = v.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        let mut scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?;
        self.o_proj.forward(&out.transpose(1, 2)?.reshape((b, s, h * d))?)
    }
}

fn expand_gqa(k: Tensor, v: Tensor, h: usize, h_kv: usize, b: usize, d: usize) -> Result<(Tensor, Tensor)> {
    if h == h_kv {
        return Ok((k, v));
    }
    let rep = h / h_kv;
    let total = k.dim(2)?;
    let k = k.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
    let v = v.unsqueeze(2)?.expand((b, h_kv, rep, total, d))?.contiguous()?.reshape((b, h, total, d))?;
    Ok((k, v))
}

// ---------------------------------------------------------------------------
// FFN: dense (E4B) or shared-MLP + routed MoE (26B-A4B)
// ---------------------------------------------------------------------------

/// Routed-expert FFN used by the MoE variant, alongside the dense shared MLP.
/// Mirrors llama.cpp `llama_model_gemma4::graph` (models/gemma4.cpp):
/// - the router operates on `attn_out` (weightless RMS × 1/√hidden × gate_inp_s),
///   NOT on the pre_ffw_norm_2-normed expert input;
/// - softmax gating, top-k, weights normalised over the selected experts;
/// - experts use a merged `gate_up` projection (gate = first n_ff rows), GEGLU,
///   then `down` with an optional per-expert output scale.
struct QGemmaMoe {
    router: QLinear,          // ffn_gate_inp: hidden -> n_experts
    router_scale: Tensor,     // ffn_gate_inp.scale: (hidden,)
    pre_norm_2: QNorm,        // pre_ffw_norm_2 (expert input norm)
    gate_up_exps: QExperts,   // [n_expert, 2*n_ff_exp, hidden]
    down_exps: QExperts,      // [n_expert, hidden, n_ff_exp]
    down_exps_scale: Option<Vec<f32>>, // ffn_down_exps.scale: (n_expert,)
    post_norm_1: QNorm,       // post_ffw_norm_1 (after the shared MLP)
    post_norm_2: QNorm,       // post_ffw_norm_2 (after the routed experts)
    n_experts: usize,
    top_k: usize,
    rms_eps: f64,
    hidden: usize,
    device: Device,
}

impl QGemmaMoe {
    fn load(
        vb: &QVarBuilder,
        n_experts: usize,
        top_k: usize,
        rms_eps: f64,
        hidden: usize,
        device: &Device,
    ) -> Result<Self> {
        let down_exps_scale = if vb.contains("ffn_down_exps.scale") {
            let t = vb.pp("ffn_down_exps").get("scale")?.dequantize(device)?;
            Some(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        } else {
            None
        };
        Ok(Self {
            router: QLinear::from_arc(vb.get("ffn_gate_inp.weight")?, None)?,
            router_scale: vb.pp("ffn_gate_inp").get("scale")?.dequantize(device)?,
            pre_norm_2: QNorm::rms_load(rms_eps, &vb.pp("pre_ffw_norm_2"))?,
            gate_up_exps: vb.get_experts("ffn_gate_up_exps.weight")?,
            down_exps: vb.get_experts("ffn_down_exps.weight")?,
            down_exps_scale,
            post_norm_1: QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm_1"))?,
            post_norm_2: QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm_2"))?,
            n_experts,
            top_k,
            rms_eps,
            hidden,
            device: device.clone(),
        })
    }

    /// The routed-experts half. `attn_out` is the post-attention residual stream.
    fn forward(&self, attn_out: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = attn_out.dims3()?;
        let num_tokens = b * seq_len;

        // Router logits from attn_out: weightless RMS, × 1/√hidden, ⊙ gate_inp_s.
        let tmp = rms_norm_no_scale(attn_out, self.rms_eps)?;
        let tmp = (tmp * (self.hidden as f64).powf(-0.5))?;
        let tmp = tmp.broadcast_mul(&self.router_scale.to_dtype(tmp.dtype())?)?;
        let logits = self.router.forward(&tmp.reshape((num_tokens, hidden))?)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs_vec: Vec<Vec<f32>> = probs.to_dtype(DType::F32)?.to_vec2()?;

        // Expert input: pre_ffw_norm_2(attn_out).
        let xin = self.pre_norm_2.forward(&attn_out.contiguous()?)?;
        let x_flat = xin.reshape((num_tokens, hidden))?;

        // Top-k by softmax prob; combine weights renormalised over the top-k.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for (tok_idx, p) in probs_vec.iter().enumerate() {
            let mut idx: Vec<usize> = (0..self.n_experts).collect();
            idx.sort_by(|&a, &c| p[c].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(self.top_k);
            let total: f32 = idx.iter().map(|&e| p[e]).sum::<f32>().max(6.1035e-5);
            for e in idx {
                expert_tokens[e].push((tok_idx, p[e] / total));
            }
        }

        let active: Vec<(usize, Vec<(usize, f32)>)> = expert_tokens
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();

        use rayon::prelude::*;
        let contributions: Vec<(Vec<usize>, Tensor)> = active
            .par_iter()
            .map(|(expert_idx, tok_weights)| -> Result<_> {
                let tok_idxs: Vec<usize> = tok_weights.iter().map(|(t, _)| *t).collect();
                let weights: Vec<f32> = tok_weights.iter().map(|(_, w)| *w).collect();

                let batch = Tensor::cat(
                    &tok_idxs.iter().map(|&i| x_flat.narrow(0, i, 1)).collect::<Result<Vec<_>>>()?,
                    0,
                )?; // (n_e, hidden)

                // Merged gate_up: rows [0, n_ff) are gate, [n_ff, 2*n_ff) are up.
                let gu_w = self.gate_up_exps.dequantize_expert(*expert_idx, &self.device)?; // (2*n_ff, hidden)
                let down_w = self.down_exps.dequantize_expert(*expert_idx, &self.device)?; // (hidden, n_ff)

                let gu = batch.matmul(&gu_w.t()?.to_dtype(batch.dtype())?)?; // (n_e, 2*n_ff)
                let n_ff = gu.dim(1)? / 2;
                let gate = gu.narrow(1, 0, n_ff)?;
                let up = gu.narrow(1, n_ff, n_ff)?;
                let act = (gate.gelu()? * up)?;

                let mut out = act.matmul(&down_w.t()?.to_dtype(act.dtype())?)?; // (n_e, hidden)
                if let Some(scales) = &self.down_exps_scale {
                    out = (out * scales[*expert_idx] as f64)?;
                }

                let w = Tensor::from_vec(weights, (tok_idxs.len(), 1), &self.device)?
                    .to_dtype(out.dtype())?;
                Ok((tok_idxs, out.broadcast_mul(&w)?))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut acc = Tensor::zeros((num_tokens, hidden), attn_out.dtype(), &self.device)?;
        for (tok_idxs, weighted) in contributions {
            let idx = Tensor::from_vec(
                tok_idxs.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                tok_idxs.len(),
                &self.device,
            )?;
            acc = acc.index_add(&idx, &weighted, 0)?;
        }
        let moe = acc.reshape((b, seq_len, hidden))?;
        self.post_norm_2.forward(&moe)
    }
}

// ---------------------------------------------------------------------------
// Block (4-norm + optional PLE + optional MoE + layer_scalar)
// ---------------------------------------------------------------------------

/// Per-layer-embedding sub-branch (E4B; absent on the 26B MoE variant).
struct QGemmaPle {
    inp_gate: QLinear,
    proj: QLinear,
    post_norm: QNorm,
}

struct QGemmaBlock {
    pre_attn_norm: QNorm,
    attn: QAttention,
    post_attn_norm: QNorm,
    pre_ffn_norm: QNorm,
    ffn_gate: QLinear,
    ffn_up: QLinear,
    ffn_down: QLinear,
    post_ffn_norm: QNorm,
    /// Routed experts (26B-A4B). The dense ffn_* above doubles as the shared MLP.
    moe: Option<QGemmaMoe>,
    ple: Option<QGemmaPle>,
    layer_scalar: Tensor,
    // Sharing
    kv_source: Option<usize>,
}

impl QGemmaBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &QVarBuilder,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rms_eps: f64,
        hidden: usize,
        n_experts: usize,
        top_k: usize,
        has_ple: bool,
        device: &Device,
        kv_source: Option<usize>,
    ) -> Result<Self> {
        let layer_scalar = vb.pp("layer_output_scale").get("weight")?.dequantize(device)?;

        let moe = if vb.contains("ffn_gate_inp.weight") {
            Some(QGemmaMoe::load(vb, n_experts, top_k, rms_eps, hidden, device)?)
        } else {
            None
        };
        let ple = if has_ple {
            Some(QGemmaPle {
                inp_gate: QLinear::load(&vb.pp("inp_gate"))?,
                proj: QLinear::load(&vb.pp("proj"))?,
                post_norm: QNorm::rms_load(rms_eps, &vb.pp("post_norm"))?,
            })
        } else {
            None
        };

        Ok(Self {
            pre_attn_norm:  QNorm::rms_load(rms_eps, &vb.pp("attn_norm"))?,
            attn:           QAttention::load(vb, n_q, n_kv, head_dim, rms_eps)?,
            post_attn_norm: QNorm::rms_load(rms_eps, &vb.pp("post_attention_norm"))?,
            pre_ffn_norm:   QNorm::rms_load(rms_eps, &vb.pp("ffn_norm"))?,
            ffn_gate:       QLinear::load(&vb.pp("ffn_gate"))?,
            ffn_up:         QLinear::load(&vb.pp("ffn_up"))?,
            ffn_down:       QLinear::load(&vb.pp("ffn_down"))?,
            post_ffn_norm:  QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm"))?,
            moe,
            ple,
            layer_scalar,
            kv_source,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        cache: &mut ModelCache,
        layer_idx: usize,
        mask: Option<&Tensor>,
        ple_i: Option<&Tensor>, // [b, s, ple_dim] when the model has PLE
    ) -> Result<Tensor> {
        // Attention branch
        let h = self.pre_attn_norm.forward(x)?;
        let h = if let Some(src) = self.kv_source {
            let src_kv = cache.layers[src].as_kv()
                .ok_or_else(|| candle_core::Error::Msg("shared KV source empty".into()))?;
            self.attn.forward_shared(&h, rope, pos, src_kv, mask)?
        } else {
            let kv = cache.get_kv(layer_idx).expect("layer has KV cache");
            self.attn.forward(&h, rope, pos, kv, mask)?
        };
        let h = self.post_attn_norm.forward(&h)?;
        let x = (x + h)?;

        // FFN branch: dense GEGLU (also the MoE variant's shared MLP).
        let h = self.pre_ffn_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&h)?.gelu()?;
        let up = self.ffn_up.forward(&h)?;
        let mlp = self.ffn_down.forward(&(gate * up)?)?;

        let h = if let Some(moe) = &self.moe {
            // Shared MLP gets its own post-norm; the routed experts run in
            // parallel off the same attn_out and the halves are summed.
            let mlp = moe.post_norm_1.forward(&mlp)?;
            let routed = moe.forward(&x)?;
            (mlp + routed)?
        } else {
            mlp
        };
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (x + h)?;

        // PLE branch (E4B only)
        let x = if let (Some(ple), Some(ple_i)) = (&self.ple, ple_i) {
            let gate = ple.inp_gate.forward(&x)?.gelu()?;
            let h = ple.proj.forward(&(gate * ple_i)?)?;
            let h = ple.post_norm.forward(&h)?;
            (x + h)?
        } else {
            x
        };

        // layer_scalar
        x.broadcast_mul(&self.layer_scalar.to_dtype(x.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Model-level PLE tensors (E4B; absent on the 26B MoE variant).
struct QGemmaPleModel {
    embed_tokens_per_layer: Embedding,
    per_layer_model_proj: QLinear,
    per_layer_proj_norm: QNorm,
}

pub struct Gemma4Q {
    embed_tokens: Embedding,
    ple: Option<QGemmaPleModel>,
    blocks: Vec<QGemmaBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope_sliding: RoPE,
    rope_global: RoPE,
    cache: ModelCache,
    device: Device,
    hidden_size: usize,
    n_layers: usize,
    ple_dim: usize,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    is_sliding: Vec<bool>,  // true = sliding attention, false = global
}

impl Gemma4Q {
    pub fn load(
        metadata: &GgufMetadata,
        vb: &QVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let prefix = metadata.get_str("general.architecture").unwrap_or_else(|_| "gemma4".to_string());

        let n_layers     = metadata.get_u32(&format!("{prefix}.block_count"))? as usize;
        let n_q          = metadata.get_u32(&format!("{prefix}.attention.head_count"))? as usize;
        // KV-head count: a scalar on E4B, a per-layer array on the 26B MoE
        // variant (sliding layers 8, global layers 2).
        let n_kv_per_layer: Vec<usize> = match metadata.get_u32(&format!("{prefix}.attention.head_count_kv")) {
            Ok(v) => vec![v as usize; n_layers],
            Err(_) => metadata
                .get_i64_array(&format!("{prefix}.attention.head_count_kv"))?
                .into_iter()
                .map(|v| v as usize)
                .collect(),
        };
        let hidden       = metadata.get_u32(&format!("{prefix}.embedding_length"))? as usize;
        let ple_dim      = metadata.get_u32_or(&format!("{prefix}.embedding_length_per_layer_input"), 256) as usize;
        // MoE (26B-A4B): routed experts alongside the dense shared MLP.
        let n_experts    = metadata.get_u32_or(&format!("{prefix}.expert_count"), 0) as usize;
        let top_k        = metadata.get_u32_or(&format!("{prefix}.expert_used_count"), 0) as usize;
        let rms_eps      = metadata.get_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-6) as f64;
        let sw           = metadata.get_u32_or(&format!("{prefix}.attention.sliding_window"), 512) as usize;
        let max_seq      = metadata.get_u32_or(&format!("{prefix}.context_length"), 131072) as usize;
        let n_kv_shared  = metadata.get_u32_or(&format!("{prefix}.attention.shared_kv_layers"), 0) as usize;
        let num_owned    = n_layers - n_kv_shared;

        // Global head_dim vs sliding head_dim
        let global_head_dim  = metadata.get_u32_or(&format!("{prefix}.attention.key_length"), 512) as usize;
        let sliding_head_dim = metadata.get_u32_or(&format!("{prefix}.attention.key_length_swa"), 256) as usize;

        // Layer type: true = sliding, false = global (full attention)
        let is_sliding: Vec<bool> = metadata.get_bool_array(
            &format!("{prefix}.attention.sliding_window_pattern")
        ).unwrap_or_else(|_| {
            // fallback: every 6th layer (1-indexed) is global
            (0..n_layers).map(|i| !((i + 1) % 6 == 0 || i == n_layers - 1)).collect()
        });

        // Sliding RoPE: standard, head_dim=256
        let theta_swa = metadata.get_f32_or(&format!("{prefix}.rope.freq_base_swa"), 10000.0) as f64;
        let rope_sliding = RoPE::new(
            &RoPEConfig {
                head_dim: sliding_head_dim,
                max_seq_len: max_seq,
                theta: theta_swa,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        // Global RoPE: proportional. rope_freqs.weight stores per-dim DIVISORS
        // (1.0 for rotated pairs, 1e30 for non-rotated — identity rotations),
        // NOT inv_freq itself. The base inv_freq comes from theta=freq_base and
        // is divided element-wise by the factors.
        let theta_global = metadata.get_f32_or(&format!("{prefix}.rope.freq_base"), 1_000_000.0) as f64;
        let rope_global = if vb.contains("rope_freqs.weight") {
            let freqs_t = vb.get("rope_freqs.weight")?.dequantize(device)?;
            let factors: Vec<f32> = freqs_t.to_vec1()?;
            let half = global_head_dim / 2;
            debug_assert_eq!(factors.len(), half, "rope_freqs length must match head_dim/2");
            let inv_freq: Vec<f64> = (0..half)
                .map(|i| {
                    let base = 1.0 / theta_global.powf(2.0 * i as f64 / global_head_dim as f64);
                    base / factors[i] as f64
                })
                .collect();
            RoPE::from_inv_freq(inv_freq, max_seq, DType::F32, device)?
        } else {
            // Fallback: compute proportional inv_freq from config
            let inv_freq = proportional_inv_freq(global_head_dim, 0.25, theta_global);
            RoPE::from_inv_freq(inv_freq, max_seq, DType::F32, device)?
        };

        // Embeddings
        let tok_embd = vb.get("token_embd.weight")?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embd, hidden);

        // PLE is an E4B feature; the 26B MoE variant has ple_dim == 0 and no
        // per-layer embedding tensors.
        let has_ple = ple_dim > 0 && vb.contains("per_layer_token_embd.weight");
        let ple = if has_ple {
            let ple_embd = vb.get("per_layer_token_embd.weight")?.dequantize(device)?;
            Some(QGemmaPleModel {
                embed_tokens_per_layer: Embedding::new(ple_embd, n_layers * ple_dim),
                per_layer_model_proj: QLinear::load(&vb.pp("per_layer_model_proj"))?,
                per_layer_proj_norm: QNorm::rms_load(rms_eps, &vb.pp("per_layer_proj_norm"))?,
            })
        } else {
            None
        };

        // Blocks
        let mut cache_layers: Vec<LayerCache> = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let sliding = *is_sliding.get(i).unwrap_or(&true);
                let head_dim = if sliding { sliding_head_dim } else { global_head_dim };
                let n_kv = *n_kv_per_layer.get(i).unwrap_or(&1);

                let kv_source = if i >= num_owned && n_kv_shared > 0 {
                    // All shared layers of the same type → last owned layer of that type
                    let source = (0..num_owned)
                        .filter(|&j| is_sliding.get(j).copied().unwrap_or(true) == sliding)
                        .last()
                        .unwrap_or(0);
                    cache_layers.push(LayerCache::Shared { source_layer: source });
                    Some(source)
                } else {
                    cache_layers.push(LayerCache::Kv(KvCache::new(max_seq)));
                    None
                };

                QGemmaBlock::load(
                    &vb.pp(format!("blk.{i}")),
                    n_q, n_kv, head_dim, rms_eps, hidden, n_experts, top_k, has_ple,
                    device, kv_source,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
            QLinear::from_arc(vb.get("token_embd.weight")?, None)?
        };

        let final_logit_softcapping = Some(
            metadata.get_f32_or(&format!("{prefix}.final_logit_softcapping"), 30.0) as f64,
        );

        Ok(Self {
            embed_tokens,
            ple,
            blocks,
            final_norm,
            lm_head,
            rope_sliding,
            rope_global,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            hidden_size: hidden,
            n_layers,
            ple_dim,
            sliding_window: sw,
            final_logit_softcapping,
            is_sliding,
        })
    }

    /// Compute per-layer inputs [b, s, n_layers, ple_dim] (E4B PLE only).
    fn compute_ple(&self, ple: &QGemmaPleModel, token_ids: &Tensor, h_embed: &Tensor) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        let (n, d) = (self.n_layers, self.ple_dim);

        // Token-level per-layer embeddings, scaled by sqrt(ple_dim)
        let ple_tok = (ple.embed_tokens_per_layer.forward(token_ids)? * (d as f64).sqrt())?;
        let ple_tok = ple_tok.reshape((b, s, n, d))?;

        // Projection of main embeddings, scaled by 1/sqrt(hidden)
        let proj = (ple.per_layer_model_proj.forward(h_embed)? * (self.hidden_size as f64).powf(-0.5))?;
        let proj = proj.reshape((b, s, n, d))?;
        let proj = ple.per_layer_proj_norm.forward(&proj)?;

        // Combine
        ((ple_tok + proj)? * 2.0_f64.powf(-0.5))
    }
}

impl CausalLM for Gemma4Q {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;

        // Main embeddings scaled by sqrt(hidden)
        let h_embed = (self.embed_tokens.forward(token_ids)? * (self.hidden_size as f64).sqrt())?;

        // Per-layer inputs [b, seq, n_layers, ple_dim] (E4B only)
        let per_layer = match &self.ple {
            Some(ple) => Some(self.compute_ple(ple, token_ids, &h_embed)?),
            None => None,
        };

        let mut h = h_embed;

        for (i, block) in self.blocks.iter().enumerate() {
            let sliding = *self.is_sliding.get(i).unwrap_or(&true);
            let rope = if sliding { &self.rope_sliding } else { &self.rope_global };

            let mask = if seq_len <= 1 {
                None
            } else if sliding {
                Some(build_sliding_window_mask(seq_len, pos, self.sliding_window, &self.device)?)
            } else {
                Some(build_causal_mask(seq_len, pos, &self.device)?)
            };

            let ple_i = match &per_layer {
                Some(pl) => Some(pl.narrow(2, i, 1)?.squeeze(2)?),
                None => None,
            };

            h = block.forward(&h, rope, pos, &mut self.cache, i, mask.as_ref(), ple_i.as_ref())?;
        }

        let h = self.final_norm.forward(&h)?;
        let mut logits = self.lm_head.forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;

        if let Some(cap) = self.final_logit_softcapping {
            logits = ((logits * (1.0 / cap))?.tanh()? * cap)?;
        }

        logits.to_dtype(DType::F32)
    }

    fn reset(&mut self) { self.cache.reset(); }
    fn device(&self) -> &Device { &self.device }
}
