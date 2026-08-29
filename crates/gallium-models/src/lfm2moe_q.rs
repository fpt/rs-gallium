//! Quantized LFM2.5 (Liquid Foundation Model 2, `lfm2moe`) loaded from GGUF.
//!
//! LFM2.5-8B-A1B is a **hybrid MoE**:
//!   - Blocks are either **short-conv** (a double-gated depthwise causal conv) or
//!     **GQA attention** (with per-head QK-RMSNorm + RoPE). The schedule comes
//!     from the per-layer `lfm2moe.attention.head_count_kv` array: `0` marks a
//!     conv block, a non-zero KV-head count marks an attention block.
//!   - FFN is dense SwiGLU for the first `leading_dense_block_count` blocks and a
//!     sparse **sigmoid-gated MoE** (with an `exp_probs_b` selection bias) for the
//!     rest.
//!   - `token_embd_norm` is the final norm; the LM head is tied to `token_embd`.
//!
//! Tensor naming in the GGUF (per block `blk.{i}`):
//!   conv block:  shortconv.{in_proj,conv,out_proj}
//!   attn block:  attn_{q,k,v,output}, attn_{q,k}_norm
//!   both:        attn_norm (pre-operator), ffn_norm (pre-FFN)
//!   dense FFN:   ffn_{gate,up,down}
//!   MoE FFN:     ffn_gate_inp, exp_probs_b.bias, ffn_{gate,up,down}_exps

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;

use gallium_core::quantized::{GgufMetadata, QLinear, QNorm, QVarBuilder};
use gallium_core::*;

// -- Short-conv block --------------------------------------------------------

struct QShortConv {
    in_proj: QLinear,    // hidden -> 3*hidden  (B | C | x)
    out_proj: QLinear,   // hidden -> hidden
    conv_weight: Tensor, // depthwise kernel, dequantized F32; (hidden, l_cache) after load
    hidden: usize,
    l_cache: usize,
}

impl QShortConv {
    fn load(vb: &QVarBuilder, hidden: usize, l_cache: usize) -> Result<Self> {
        let dev = vb.device();
        Ok(Self {
            in_proj: QLinear::from_arc(vb.get("shortconv.in_proj.weight")?, None)?,
            out_proj: QLinear::from_arc(vb.get("shortconv.out_proj.weight")?, None)?,
            conv_weight: vb.get("shortconv.conv.weight")?.dequantize(dev)?,
            hidden,
            l_cache,
        })
    }

    fn forward(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let d = self.hidden;

        // in_proj -> [B | C | x], each of width `hidden`.
        let bcx = self.in_proj.forward(x)?; // (b, s, 3d)
        let bg = bcx.narrow(2, 0, d)?;
        let cg = bcx.narrow(2, d, d)?;
        let xg = bcx.narrow(2, 2 * d, d)?;

        // Bx = B * x, causal depthwise conv, then gate by C.
        let bx = (bg * xg)?;
        let conv_out = self.causal_conv(&bx, state)?;
        let y = (cg * conv_out)?;
        self.out_proj.forward(&y)
    }

    /// Depthwise causal conv1d (no activation). `conv_weight` is dequantized as
    /// (hidden, l_cache); we transpose to (l_cache, hidden) for the windowed
    /// broadcast-multiply-and-sum, and carry `l_cache-1` samples across steps.
    fn causal_conv(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        let (b, seq_len, dim) = x.dims3()?;
        let k = self.l_cache;

        let padded = match state.conv_state.take() {
            Some(prev) => Tensor::cat(&[&prev, x], 1)?,
            None => {
                let pad = Tensor::zeros((b, k - 1, dim), x.dtype(), x.device())?;
                Tensor::cat(&[&pad, x], 1)?
            }
        };
        let total = padded.dim(1)?;
        state.conv_state = Some(padded.narrow(1, total - (k - 1), k - 1)?);

        let w = self.conv_weight.t()?.contiguous()?.to_dtype(x.dtype())?; // (k, dim)
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(1, t, k)?; // (b, k, dim)
            let out = window.broadcast_mul(&w)?.sum(1)?; // (b, dim)
            outs.push(out.unsqueeze(1)?);
        }
        Tensor::cat(&outs, 1)
    }
}

// -- GQA attention (QK-norm + RoPE) ------------------------------------------

struct QAttention {
    q_proj: QLinear,
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
    fn load(
        vb: &QVarBuilder,
        num_q_heads: usize,
        num_kv_heads: usize,
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

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq_len, h, d))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq_len, h_kv, d))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq_len, h_kv, d))?
            .transpose(1, 2)?;

        // Per-head QK RMSNorm over head_dim, then RoPE.
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // K/V stay at h_kv heads; `gqa_scores` groups Q instead of expanding them.
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (gqa_scores(&q, &k)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = gqa_weighted_sum(&probs, &v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;
        self.o_proj.forward(&attn_out)
    }
}

// -- Dense SwiGLU FFN --------------------------------------------------------

struct QGatedFFN {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QGatedFFN {
    fn load(vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::from_arc(vb.get("ffn_gate.weight")?, None)?,
            up_proj: QLinear::from_arc(vb.get("ffn_up.weight")?, None)?,
            down_proj: QLinear::from_arc(vb.get("ffn_down.weight")?, None)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// `x · Wᵀ` for one expert, taking whichever of candle's two quantized paths is
/// accurate for this shape.
///
/// candle's quantized multiply is **not one implementation**, and the difference
/// is measurable against llama.cpp on the same tokens
/// (`gallium-agent/tests/engine_logits.rs`):
///
/// | rows of `x` | candle vs llama.cpp, top-1 logit |
/// |---|---|
/// | 1 (a decode step) | 10.895035 vs 10.895246 — **0.0002** |
/// | 2 | 37.38982 vs 37.661972 — 0.27 |
/// | 14 | 31.36 vs 30.84 — 0.5 |
///
/// One row takes the matvec kernel ported from ggml and is exact to four
/// decimals. More than one takes a different path and drifts. So decode — where
/// every expert sees exactly one token — is free to multiply quantized, while a
/// prefill, which is the whole prompt through the same code, is not: that drift
/// is what moved the first sampled token and cost the `refactoring` testcase.
///
/// Expanding for the many-row case costs one expansion per expert-projection
/// **per prefill**, not per token — 288 of them for this model, against the 288
/// *per token* the original code did. That is the distinction the first pass at
/// this missed by treating "expand" and "multiply quantized" as a single choice
/// for the whole model.
///
/// What ggml does on Metal is better than either: `kernel_mul_mv_q4_K_f32`
/// unpacks the weight block in registers and multiplies it against f32
/// activations, so it needs neither the expansion nor the accuracy loss. candle
/// has no such fused kernel for the many-row case, which is the gap this works
/// around.
fn expert_matmul(w: &QMatMul, x: &Tensor) -> Result<Tensor> {
    if x.dim(0)? <= 1 {
        return w.forward(x);
    }
    match w {
        QMatMul::QTensor(qt) => {
            let expanded = qt.dequantize(x.device())?;
            x.matmul(&expanded.t()?.to_dtype(x.dtype())?.contiguous()?)
        }
        // Already float — `forward` is the same multiply either way.
        _ => w.forward(x),
    }
}

// -- Sparse MoE FFN (sigmoid gating + selection bias) ------------------------

struct QMoEFFN {
    router: QLinear,      // ffn_gate_inp: hidden -> n_experts
    probs_bias: Vec<f32>, // exp_probs_b.bias: (n_experts,), for top-k selection only
    /// One matmul per expert per projection, built at load, multiplying against
    /// the **quantized** weight.
    ///
    /// These used to be expanded to f32 inside `forward`, once per active
    /// expert per token — which also re-uploaded the quantized bytes to the
    /// device each time. On this model that is 24 blocks × 4 active experts × 3
    /// projections = 288 expansions per token, 14.7 MB each: **4.23 GB
    /// allocated and freed per token**, resident memory cycling 8 GB → 12 GB →
    /// 8 GB, decode at 1.1 tok/s.
    ///
    /// Every way of keeping the expansion was measured and none is close — see
    /// `docs/CANDLE_BACKEND.md` §6 for the table. Keeping the quantized bytes
    /// resident so only the expansion remains: 1451 ms/token. Expanding to f16
    /// instead: 1615 ms. Caching the expansions cannot be sized: all 2304
    /// expert-projections are 16.9 GB in f16 against a 19 GB working set, and a
    /// partial cache cannot get the hit rate — this model's routing uses **all
    /// 32 experts of every layer**, so even 24 of 32 resident (12.7 GB) leaves
    /// 15% of selections missing, at 223 ms/token.
    ///
    /// What it costs: `QMatMul` runs ggml's kernel, which quantizes the
    /// *activations* to 8 bits, so the arithmetic changed — the same arithmetic
    /// llama.cpp performs for this GGUF, and one testsuite case (`refactoring`)
    /// that the expanded path passes and this one does not.
    gate_exps: Vec<QMatMul>, // [n_expert] of (n_ff, n_embd)
    up_exps: Vec<QMatMul>, // [n_expert] of (n_ff, n_embd)
    down_exps: Vec<QMatMul>, // [n_expert] of (n_embd, n_ff)
    n_experts: usize,
    top_k: usize,
    device: Device,
}

impl QMoEFFN {
    fn load(vb: &QVarBuilder, n_experts: usize, top_k: usize) -> Result<Self> {
        let probs_bias: Vec<f32> = vb
            .get("exp_probs_b.bias")?
            .dequantize(vb.device())?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(Self {
            router: QLinear::load(&vb.pp("ffn_gate_inp"))?,
            probs_bias,
            gate_exps: vb
                .get_experts("ffn_gate_exps.weight")?
                .qmatmuls(vb.device())?,
            up_exps: vb
                .get_experts("ffn_up_exps.weight")?
                .qmatmuls(vb.device())?,
            down_exps: vb
                .get_experts("ffn_down_exps.weight")?
                .qmatmuls(vb.device())?,
            n_experts,
            top_k,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b * seq_len, hidden))?;
        let num_tokens = b * seq_len;

        // Sigmoid gating; bias is added for SELECTION ONLY, weights are the raw
        // sigmoid probs of the chosen experts, normalised to sum to 1.
        let router_logits = self.router.forward(&x_flat)?;
        let probs = candle_nn::ops::sigmoid(&router_logits)?;
        let probs_vec: Vec<Vec<f32>> = probs.to_dtype(DType::F32)?.to_vec2()?;

        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for (tok_idx, p) in probs_vec.iter().enumerate() {
            let mut idx: Vec<usize> = (0..self.n_experts).collect();
            idx.sort_by(|&a, &c| {
                let sa = p[a] + self.probs_bias[a];
                let sc = p[c] + self.probs_bias[c];
                sc.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(self.top_k);
            let total: f32 = idx.iter().map(|&e| p[e]).sum::<f32>().max(1e-20);
            for e in idx {
                expert_tokens[e].push((tok_idx, p[e] / total));
            }
        }

        // Which experts this layer picked, for the question a partial cache of
        // *expanded* experts turns on: a cache only pays if the same few
        // experts come back. One line per MoE call, in layer order, so a
        // decode step is a run of `block_count` of them.
        if tracing::enabled!(target: "gallium::moe", tracing::Level::TRACE) {
            let picked: Vec<usize> = expert_tokens
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(e, _)| e)
                .collect();
            tracing::trace!(target: "gallium::moe", "experts {picked:?}");
        }

        let active: Vec<(usize, Vec<(usize, f32)>)> = expert_tokens
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();

        let one_expert =
            |(expert_idx, tok_weights): &(usize, Vec<(usize, f32)>)| -> Result<(Vec<usize>, Tensor)> {
                let tok_idxs: Vec<usize> = tok_weights.iter().map(|(t, _)| *t).collect();
                let weights: Vec<f32> = tok_weights.iter().map(|(_, w)| *w).collect();

                let batch = Tensor::cat(
                    &tok_idxs
                        .iter()
                        .map(|&i| x_flat.narrow(0, i, 1))
                        .collect::<Result<Vec<_>>>()?,
                    0,
                )?; // (n_e, hidden)

                let gate = candle_nn::ops::silu(&expert_matmul(
                    &self.gate_exps[*expert_idx],
                    &batch,
                )?)?;
                let up = expert_matmul(&self.up_exps[*expert_idx], &batch)?;
                let inter = (gate * up)?;
                let out = expert_matmul(&self.down_exps[*expert_idx], &inter)?; // (n_e, hidden)

                let w = Tensor::from_vec(weights, (tok_idxs.len(), 1), &self.device)?
                    .to_dtype(out.dtype())?;
                let weighted = out.broadcast_mul(&w)?;
                Ok((tok_idxs, weighted))
            };

        // Parallel on the CPU, serial on an accelerator — this loop is where the
        // NaN-on-Metal bug `par_map_on_cpu` documents was found.
        let contributions = par_map_on_cpu(&self.device, &active, one_expert)?;

        let mut acc = Tensor::zeros((num_tokens, hidden), x.dtype(), &self.device)?;
        for (tok_idxs, weighted) in contributions {
            let idx = Tensor::from_vec(
                tok_idxs.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                tok_idxs.len(),
                &self.device,
            )?;
            acc = acc.index_add(&idx, &weighted, 0)?;
        }
        acc.reshape((b, seq_len, hidden))
    }
}

// -- Per-layer dispatch ------------------------------------------------------

enum QOperator {
    Attn(QAttention),
    Conv(QShortConv),
}

enum QFfn {
    Dense(QGatedFFN),
    Moe(QMoEFFN),
}

struct QBlock {
    op_norm: QNorm,
    op: QOperator,
    ffn_norm: QNorm,
    ffn: QFfn,
}

impl QBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: Option<&mut KvCache>,
        recurrent: Option<&mut RecurrentState>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.op_norm.forward(&x.contiguous()?)?;
        let op_out = match &self.op {
            QOperator::Attn(a) => a.forward(
                &normed,
                rope,
                pos,
                kv_cache.expect("attn needs KV cache"),
                mask,
            )?,
            QOperator::Conv(c) => {
                c.forward(&normed, recurrent.expect("conv needs recurrent state"))?
            }
        };
        let h = (op_out + x)?;
        let residual = h.clone();
        let normed = self.ffn_norm.forward(&h.contiguous()?)?;
        let f = match &self.ffn {
            QFfn::Dense(d) => d.forward(&normed)?,
            QFfn::Moe(m) => m.forward(&normed)?,
        };
        f + residual
    }
}

// -- Full model --------------------------------------------------------------

pub struct Lfm2MoeQ {
    embed_tokens: Embedding,
    blocks: Vec<QBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
}

impl Lfm2MoeQ {
    pub fn load(metadata: &GgufMetadata, vb: &QVarBuilder, device: &Device) -> Result<Self> {
        let arch = metadata
            .get_str("general.architecture")
            .unwrap_or_else(|_| "lfm2moe".to_string());
        let pfx = &arch;

        let n_layers = metadata.get_u32(&format!("{pfx}.block_count"))? as usize;
        let n_heads = metadata.get_u32(&format!("{pfx}.attention.head_count"))? as usize;
        let n_embd = metadata.get_u32(&format!("{pfx}.embedding_length"))? as usize;
        let head_dim = metadata.get_u32_or(
            &format!("{pfx}.attention.key_length"),
            (n_embd / n_heads) as u32,
        ) as usize;
        let rope_freq = metadata.get_f32_or(&format!("{pfx}.rope.freq_base"), 1_000_000.0) as f64;
        let rms_eps =
            metadata.get_f32_or(&format!("{pfx}.attention.layer_norm_rms_epsilon"), 1e-5) as f64;
        let max_seq = metadata.get_u32_or(&format!("{pfx}.context_length"), 128_000) as usize;
        let l_cache = metadata.get_u32_or(&format!("{pfx}.shortconv.l_cache"), 3) as usize;
        let leading_dense =
            metadata.get_u32_or(&format!("{pfx}.leading_dense_block_count"), 0) as usize;
        let n_experts = metadata.get_u32_or(&format!("{pfx}.expert_count"), 0) as usize;
        let top_k = metadata.get_u32_or(&format!("{pfx}.expert_used_count"), 0) as usize;

        // Per-layer KV-head counts: 0 marks a conv block, non-zero an attention block.
        let kv_per_layer = metadata.get_i64_array(&format!("{pfx}.attention.head_count_kv"))?;

        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len: max_seq,
                theta: rope_freq,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        let tok_embd = vb.get("token_embd.weight")?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embd, n_embd);

        let mut cache_layers = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let bvb = vb.pp(format!("blk.{i}"));
                let n_kv = *kv_per_layer.get(i).unwrap_or(&0) as usize;
                let op = if n_kv > 0 {
                    cache_layers.push(LayerCache::Kv(KvCache::new(max_seq)));
                    QOperator::Attn(QAttention::load(&bvb, n_heads, n_kv, head_dim, rms_eps)?)
                } else {
                    cache_layers.push(LayerCache::Recurrent(RecurrentState::new()));
                    QOperator::Conv(QShortConv::load(&bvb, n_embd, l_cache)?)
                };
                let ffn = if i < leading_dense {
                    QFfn::Dense(QGatedFFN::load(&bvb)?)
                } else {
                    QFfn::Moe(QMoEFFN::load(&bvb, n_experts, top_k)?)
                };
                Ok(QBlock {
                    op_norm: QNorm::rms_load(rms_eps, &bvb.pp("attn_norm"))?,
                    op,
                    ffn_norm: QNorm::rms_load(rms_eps, &bvb.pp("ffn_norm"))?,
                    ffn,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Final norm is `token_embd_norm`; LM head is tied to `token_embd`.
        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("token_embd_norm"))?;
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

impl CausalLM for Lfm2MoeQ {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed_tokens.forward(token_ids)?.contiguous()?;

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = match &block.op {
                QOperator::Attn(_) if seq_len > 1 => {
                    Some(build_causal_mask(seq_len, pos, &self.device)?)
                }
                _ => None,
            };
            let (kv, recurrent) = self.cache.get_layer(i);
            h = block
                .forward(&h, &self.rope, pos, kv, recurrent, mask.as_ref())?
                .contiguous()?;
        }

        let h_final = self.final_norm.forward(&h)?;
        let logits = self
            .lm_head
            .forward(&h_final.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        Ok(logits.to_dtype(DType::F32)?)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    /// Opted into cross-call reuse: this model's cache is the standard
    /// `ModelCache`, so `generate_reusing` can roll it back and evaluate only
    /// what a new prompt adds. See [`gallium_core::CausalLM::cache`].
    fn cache(&mut self) -> Option<&mut ModelCache> {
        Some(&mut self.cache)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}
