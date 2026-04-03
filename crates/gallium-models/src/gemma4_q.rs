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

use gallium_core::quantized::{GgufMetadata, QLinear, QNorm, QVarBuilder};
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
    v_proj: QLinear,
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
        Ok(Self {
            q_proj: QLinear::load(&vb.pp("attn_q"))?,
            k_proj: QLinear::load(&vb.pp("attn_k"))?,
            v_proj: QLinear::load(&vb.pp("attn_v"))?,
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
        let k = self.k_proj.forward(x)?.reshape((b, s, h_kv, d))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((b, s, h_kv, d))?.transpose(1, 2)?.contiguous()?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

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
// Block (4-norm + PLE + layer_scalar)
// ---------------------------------------------------------------------------

struct QGemmaBlock {
    pre_attn_norm: QNorm,
    attn: QAttention,
    post_attn_norm: QNorm,
    pre_ffn_norm: QNorm,
    ffn_gate: QLinear,
    ffn_up: QLinear,
    ffn_down: QLinear,
    post_ffn_norm: QNorm,
    // PLE
    inp_gate: QLinear,
    proj: QLinear,
    post_norm: QNorm,
    layer_scalar: Tensor,
    // Sharing
    kv_source: Option<usize>,
}

impl QGemmaBlock {
    fn load(
        vb: &QVarBuilder,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rms_eps: f64,
        device: &Device,
        kv_source: Option<usize>,
    ) -> Result<Self> {
        let layer_scalar = vb.pp("layer_output_scale").get("weight")?.dequantize(device)?;

        Ok(Self {
            pre_attn_norm:  QNorm::rms_load(rms_eps, &vb.pp("attn_norm"))?,
            attn:           QAttention::load(vb, n_q, n_kv, head_dim, rms_eps)?,
            post_attn_norm: QNorm::rms_load(rms_eps, &vb.pp("post_attention_norm"))?,
            pre_ffn_norm:   QNorm::rms_load(rms_eps, &vb.pp("ffn_norm"))?,
            ffn_gate:       QLinear::load(&vb.pp("ffn_gate"))?,
            ffn_up:         QLinear::load(&vb.pp("ffn_up"))?,
            ffn_down:       QLinear::load(&vb.pp("ffn_down"))?,
            post_ffn_norm:  QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm"))?,
            inp_gate:       QLinear::load(&vb.pp("inp_gate"))?,
            proj:           QLinear::load(&vb.pp("proj"))?,
            post_norm:      QNorm::rms_load(rms_eps, &vb.pp("post_norm"))?,
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
        ple_i: &Tensor,   // [b, s, ple_dim]
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

        // FFN branch
        let h = self.pre_ffn_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&h)?.gelu()?;
        let up = self.ffn_up.forward(&h)?;
        let h = self.ffn_down.forward(&(gate * up)?)?;
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (x + h)?;

        // PLE branch
        let gate = self.inp_gate.forward(&x)?.gelu()?;
        let h = self.proj.forward(&(gate * ple_i)?)?;
        let h = self.post_norm.forward(&h)?;
        let x = (x + h)?;

        // layer_scalar
        x.broadcast_mul(&self.layer_scalar.to_dtype(x.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Gemma4Q {
    embed_tokens: Embedding,
    embed_tokens_per_layer: Embedding,
    per_layer_model_proj: QLinear,
    per_layer_proj_norm: QNorm,
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
        let n_kv         = metadata.get_u32(&format!("{prefix}.attention.head_count_kv"))? as usize;
        let hidden       = metadata.get_u32(&format!("{prefix}.embedding_length"))? as usize;
        let ple_dim      = metadata.get_u32_or(&format!("{prefix}.embedding_length_per_layer_input"), 256) as usize;
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

        let ple_embd = vb.get("per_layer_token_embd.weight")?.dequantize(device)?;
        let embed_tokens_per_layer = Embedding::new(ple_embd, n_layers * ple_dim);

        let per_layer_model_proj = QLinear::load(&vb.pp("per_layer_model_proj"))?;
        let per_layer_proj_norm  = QNorm::rms_load(rms_eps, &vb.pp("per_layer_proj_norm"))?;

        // Blocks
        let mut cache_layers: Vec<LayerCache> = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let sliding = *is_sliding.get(i).unwrap_or(&true);
                let head_dim = if sliding { sliding_head_dim } else { global_head_dim };

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
                    n_q, n_kv, head_dim, rms_eps, device, kv_source,
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
            embed_tokens_per_layer,
            per_layer_model_proj,
            per_layer_proj_norm,
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

    /// Compute per-layer inputs [b, s, n_layers, ple_dim].
    fn compute_ple(&self, token_ids: &Tensor, h_embed: &Tensor) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        let (n, d) = (self.n_layers, self.ple_dim);

        // Token-level per-layer embeddings, scaled by sqrt(ple_dim)
        let ple_tok = (self.embed_tokens_per_layer.forward(token_ids)? * (d as f64).sqrt())?;
        let ple_tok = ple_tok.reshape((b, s, n, d))?;

        // Projection of main embeddings, scaled by 1/sqrt(hidden)
        let proj = (self.per_layer_model_proj.forward(h_embed)? * (self.hidden_size as f64).powf(-0.5))?;
        let proj = proj.reshape((b, s, n, d))?;
        let proj = self.per_layer_proj_norm.forward(&proj)?;

        // Combine
        ((ple_tok + proj)? * 2.0_f64.powf(-0.5))
    }
}

impl CausalLM for Gemma4Q {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;

        // Main embeddings scaled by sqrt(hidden)
        let h_embed = (self.embed_tokens.forward(token_ids)? * (self.hidden_size as f64).sqrt())?;

        // Per-layer inputs [b, seq, n_layers, ple_dim]
        let per_layer = self.compute_ple(token_ids, &h_embed)?;

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

            let ple_i = per_layer.narrow(2, i, 1)?.squeeze(2)?;

            h = block.forward(&h, rope, pos, &mut self.cache, i, mask.as_ref(), &ple_i)?;
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
