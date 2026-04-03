//! Quantized GPT-OSS model loaded from GGUF.
//!
//! GGUF uses a different tensor naming convention than safetensors:
//!   blk.{i}.attn_q.weight   vs  model.layers.{i}.self_attn.q_proj.weight
//!   token_embd.weight       vs  model.embed_tokens.weight

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;

use gallium_core::quantized::{GgufMetadata, QLinear, QNorm, QVarBuilder, Tq2Tensor};
use gallium_core::*;

// -- Quantized Attention (uses QLinear) --------------------------------------

struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    /// Per-head sink logit appended to attention scores before softmax.
    sinks: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QAttention {
    fn load(vb: &QVarBuilder, num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        let q_proj = QLinear::load(&vb.pp("attn_q"))?;
        let k_proj = QLinear::load(&vb.pp("attn_k"))?;
        let v_proj = QLinear::load(&vb.pp("attn_v"))?;
        let o_proj = QLinear::load(&vb.pp("attn_output"))?;
        let sinks = vb.get("attn_sinks.weight")?.dequantize(vb.device())?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sinks,
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
        let h = self.num_q_heads;
        let h_kv = self.num_kv_heads;
        let d = self.head_dim;

        let q = self.q_proj.forward(x)?.reshape((b, seq_len, h, d))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, seq_len, h_kv, d))?.transpose(1, 2)?;

        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // Repeat KV for GQA
        let (k, v) = if h != h_kv {
            let rep = h / h_kv;
            let k = k.unsqueeze(2)?.expand((b, h_kv, rep, k.dim(2)?, d))?.reshape((b, h, k.dim(2)?, d))?;
            let v = v.unsqueeze(2)?.expand((b, h_kv, rep, v.dim(2)?, d))?.reshape((b, h, v.dim(2)?, d))?;
            (k, v)
        } else {
            (k, v)
        };

        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        }

        // Attention sinks: append per-head sink logit, softmax over seq+1, drop last col.
        let total_len = scores.dim(D::Minus1)?;
        let s = self.sinks.reshape((1, h, 1, 1))?.expand((b, h, seq_len, 1))?.contiguous()?;
        let combined = Tensor::cat(&[&scores, &s], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&combined)?;
        let attn_weights = probs.narrow(D::Minus1, 0, total_len)?;

        let attn_out = attn_weights.matmul(&v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;
        self.o_proj.forward(&attn_out)
    }
}

// -- Quantized GatedFFN (single expert, unused) ------------------------------

struct QGatedFFN {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    clamp: Option<f32>,
}

impl QGatedFFN {
    fn load(vb: &QVarBuilder, clamp: Option<f32>) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::load(&vb.pp("ffn_gate"))?,
            up_proj: QLinear::load(&vb.pp("ffn_up"))?,
            down_proj: QLinear::load(&vb.pp("ffn_down"))?,
            clamp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_raw = self.gate_proj.forward(x)?;
        let up_raw = self.up_proj.forward(x)?;
        let gate = if let Some(limit) = self.clamp {
            gate_raw.clamp(-1e38_f64, limit as f64)?
        } else {
            gate_raw
        };
        let sig = ((&gate * 0.851_f64)?.tanh()? + 1.0_f64)? * 0.5_f64;
        let glu = (gate * (sig)?)?;
        let up = if let Some(limit) = self.clamp {
            up_raw.clamp(-(limit as f64), limit as f64)?
        } else {
            up_raw
        };
        let up1 = (up + 1.0_f64)?;
        self.down_proj.forward(&(glu * up1)?)
    }
}

// -- Quantized MoE -----------------------------------------------------------
//
// GGUF stores expert weights as merged 3D MXFP4 tensors:
//   ffn_gate_exps.weight: [n_expert, n_ff, n_embd]
//   ffn_up_exps.weight:   [n_expert, n_ff, n_embd]
//   ffn_down_exps.weight: [n_expert, n_embd, n_ff]
//
// We store raw bytes and dequantize one expert at a time during forward.

struct QMoEFFN {
    /// Raw MXFP4 expert weights — dequantized lazily per expert during forward.
    gate_exps: Tq2Tensor, // dims: [n_expert, n_ff, n_embd]
    up_exps: Tq2Tensor,   // dims: [n_expert, n_ff, n_embd]
    down_exps: Tq2Tensor, // dims: [n_expert, n_embd, n_ff]
    /// Per-expert biases: shape [n_expert, n_ff] or [n_expert, n_embd].
    gate_bias: Tensor,  // [n_expert, n_ff]
    up_bias: Tensor,    // [n_expert, n_ff]
    down_bias: Tensor,  // [n_expert, n_embd]
    router: QLinear,
    num_experts_per_tok: usize,
    clamp: Option<f32>,
    device: Device,
}

impl QMoEFFN {
    fn load(
        vb: &QVarBuilder,
        num_experts: usize,
        num_experts_per_tok: usize,
        clamp: Option<f32>,
    ) -> Result<Self> {
        // Load merged TQ2_0 expert tensors as raw bytes for lazy per-expert dequant.
        let gate_exps = vb.get_tq2("ffn_gate_exps.weight")?;
        let up_exps = vb.get_tq2("ffn_up_exps.weight")?;
        let down_exps = vb.get_tq2("ffn_down_exps.weight")?;
        // Expert FFN biases: shape [n_expert, n_ff] or [n_expert, n_embd] after dim reversal.
        let gate_bias = vb.get("ffn_gate_exps.bias")?.dequantize(vb.device())?;
        let up_bias = vb.get("ffn_up_exps.bias")?.dequantize(vb.device())?;
        let down_bias = vb.get("ffn_down_exps.bias")?.dequantize(vb.device())?;
        let router = QLinear::load(&vb.pp("ffn_gate_inp"))?;
        let _ = num_experts; // used only to verify dims at load time if needed
        Ok(Self {
            gate_exps,
            up_exps,
            down_exps,
            gate_bias,
            up_bias,
            down_bias,
            router,
            num_experts_per_tok,
            clamp,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b * seq_len, hidden))?;
        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;
        let router_probs_vec: Vec<Vec<f32>> = router_probs.to_vec2()?;
        let num_tokens = b * seq_len;

        let mut output_data = Vec::with_capacity(num_tokens);
        for tok_idx in 0..num_tokens {
            let probs = &router_probs_vec[tok_idx];
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(self.num_experts_per_tok);
            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let token = x_flat.narrow(0, tok_idx, 1)?; // (1, hidden)
            let mut tok_out = Tensor::zeros((1, hidden), x.dtype(), x.device())?;

            for (expert_idx, weight) in &indexed {
                // Dequantize TQ2_0 for this expert only.
                let gate_w = self.gate_exps.dequantize_expert(*expert_idx, &self.device)?;
                let up_w = self.up_exps.dequantize_expert(*expert_idx, &self.device)?;
                let down_w = self.down_exps.dequantize_expert(*expert_idx, &self.device)?;

                // Bias for this expert: narrow [n_expert, n_ff] → [1, n_ff].
                let gb = self.gate_bias.narrow(0, *expert_idx, 1)?;
                let ub = self.up_bias.narrow(0, *expert_idx, 1)?;
                let db = self.down_bias.narrow(0, *expert_idx, 1)?;

                // Gate: clamp from above only, then apply scaled-sigmoid GLU.
                // sigmoid(x * 1.702) = (1 + tanh(x * 0.851)) / 2
                let gate_raw = (token.matmul(&gate_w.t()?)? + gb)?;
                let gate = if let Some(limit) = self.clamp {
                    gate_raw.clamp(-1e38_f64, limit as f64)?
                } else {
                    gate_raw.clone()
                };
                let sig = ((&gate * 0.851_f64)?.tanh()? + 1.0_f64)? * 0.5_f64;
                let glu = (gate * (sig)?)?;

                // Up: clamp both sides, then shift by +1.
                let up_raw = (token.matmul(&up_w.t()?)? + ub)?;
                let up = if let Some(limit) = self.clamp {
                    up_raw.clamp(-(limit as f64), limit as f64)?
                } else {
                    up_raw
                };
                let up1 = (up + 1.0_f64)?;

                let expert_out = ((glu * up1)?.matmul(&down_w.t()?)? + db)?;

                tok_out = (tok_out + expert_out * (*weight as f64 / total as f64))?;
            }
            output_data.push(tok_out);
        }
        Tensor::cat(&output_data, 0)?.reshape((b, seq_len, hidden))
    }
}

// -- Quantized Transformer Block ---------------------------------------------

struct QTransformerBlock {
    pre_attn_norm: QNorm,
    attn: QAttention,
    post_attn_norm: QNorm,
    ffn: QMoEFFN,
}

impl QTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let h_in = self.pre_attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&h_in, rope, pos, kv_cache, mask)?;
        let h = (attn_out + x)?;
        let residual = &h;
        let h = self.post_attn_norm.forward(&h)?;
        let h = self.ffn.forward(&h)?;
        h + residual
    }
}

// -- Full Quantized GPT-OSS Model --------------------------------------------

pub struct GptOssQ {
    embed_tokens: Embedding,
    blocks: Vec<QTransformerBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
    sliding_window: usize,
    layer_types: Vec<String>, // "full_attention" or "sliding_attention"
}

impl GptOssQ {
    /// Load from GGUF file.
    pub fn load(
        metadata: &GgufMetadata,
        vb: &QVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        // Extract config from GGUF metadata
        // GPT-OSS uses "gpt_oss" arch prefix in GGUF
        let arch = metadata.get_str("general.architecture").unwrap_or_else(|_| "llama".to_string());
        let prefix = &arch;

        let n_layers = metadata.get_u32(&format!("{prefix}.block_count"))? as usize;
        let n_heads = metadata.get_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let n_kv_heads = metadata.get_u32(&format!("{prefix}.attention.head_count_kv"))? as usize;
        let n_embd = metadata.get_u32(&format!("{prefix}.embedding_length"))? as usize;
        // GPT-OSS uses head_dim=64, which differs from n_embd/n_heads (=45).
        // Prefer the explicit key_length field; fall back to n_embd/n_heads.
        let head_dim = metadata.get_u32_or(
            &format!("{prefix}.attention.key_length"),
            (n_embd / n_heads) as u32,
        ) as usize;
        let rope_freq_base = metadata.get_f32_or(&format!("{prefix}.rope.freq_base"), 150000.0);
        let rope_scaling_factor = metadata.get_f32_or(&format!("{prefix}.rope.scaling.factor"), 1.0);
        let rope_orig_ctx = metadata.get_u32_or(&format!("{prefix}.rope.scaling.original_context_length"), 4096) as usize;
        let rms_eps = metadata.get_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5) as f64;
        let n_experts = metadata.get_u32_or(&format!("{prefix}.expert_count"), 32) as usize;
        let n_experts_used = metadata.get_u32_or(&format!("{prefix}.expert_used_count"), 4) as usize;
        let sliding_window = metadata.get_u32_or(&format!("{prefix}.attention.sliding_window"), 128) as usize;
        let max_seq_len = metadata.get_u32_or(&format!("{prefix}.context_length"), 131072) as usize;
        let swiglu_limit = metadata.get_f32_or(&format!("{prefix}.swiglu_limit"), 7.0);

        // Layer types from metadata (or default alternating).
        // HF transformers: `"sliding_attention" if bool((i+1)%2) else "full_attention"`.
        // Ollama (model/models/gptoss/model.go:38–39 "// Even layers are sliding window
        // attention." + `SetLayerType(i % 2)` with SWA cache at index 0) agrees:
        //   i=0 -> sliding, i=1 -> full, i=2 -> sliding, ...
        // The GGUF has no `attention.layer_type` array for this arch, so we always hit
        // this fallback — getting it wrong silently swaps every layer's mask.
        let layer_types: Vec<String> = metadata
            .get_str_array(&format!("{prefix}.attention.layer_type"))
            .unwrap_or_else(|_| {
                (0..n_layers)
                    .map(|i| {
                        if i % 2 == 0 {
                            "sliding_attention".to_string()
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect()
            });

        // RoPE with YaRN scaling if specified
        let rope_scaling = if rope_scaling_factor > 1.0 {
            RoPEScaling::YaRN {
                factor: rope_scaling_factor as f64,
                original_max_position_embeddings: rope_orig_ctx,
                beta_fast: 32.0,
                beta_slow: 1.0,
            }
        } else {
            RoPEScaling::None
        };
        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len,
                theta: rope_freq_base as f64,
                scaling: rope_scaling,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        // Embeddings: dequantize since embedding lookup needs float
        let tok_embd = vb.get("token_embd.weight")?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embd, n_embd);

        // Layers
        let mut cache_layers = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let bvb = vb.pp(format!("blk.{i}"));
                cache_layers.push(LayerCache::Kv(KvCache::new(max_seq_len)));
                Ok(QTransformerBlock {
                    pre_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("attn_norm"))?,
                    attn: QAttention::load(&bvb, n_heads, n_kv_heads, head_dim)?,
                    post_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("post_attention_norm"))?,
                    ffn: QMoEFFN::load(&bvb, n_experts, n_experts_used, Some(swiglu_limit))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
            // Tied embeddings: reuse token_embd
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
            sliding_window,
            layer_types,
        })
    }
}

impl CausalLM for GptOssQ {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed_tokens.forward(token_ids)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let is_sliding = self.layer_types
                .get(i)
                .map(|s| s.contains("sliding"))
                .unwrap_or(false);
            // Sliding layers need a mask even at seq_len=1 (decode) once the KV cache
            // exceeds the window — otherwise queries attend to evicted-by-design K/V.
            // Full-attention layers at seq_len=1 have nothing to mask (all K are in the
            // past, causal is automatic).
            let needs_mask = seq_len > 1
                || (is_sliding && pos + seq_len > self.sliding_window);
            let mask = if !needs_mask {
                None
            } else {
                let m = if is_sliding {
                    build_sliding_window_mask(seq_len, pos, self.sliding_window, &self.device)?
                } else {
                    build_causal_mask(seq_len, pos, &self.device)?
                };
                Some(m)
            };
            let kv = self.cache.get_kv(i);
            let kv = kv.expect("GPT-OSS layers all use KV cache");
            h = block.forward(&h, &self.rope, pos, kv, mask.as_ref())?;
        }

        let h = self.final_norm.forward(&h)?;
        let logits = self.lm_head.forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        Ok(logits.to_dtype(candle_core::DType::F32)?)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
