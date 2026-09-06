//! GPT-OSS safetensors model.
//!
//! Architecture:
//!   - GQA (64 Q heads, 8 KV heads, head_dim=64) with biases and per-head attention sinks
//!   - Alternating sliding-window (128) and full-attention layers
//!   - MXFP4-quantized fused MoE FFN (32 experts, top-4)
//!   - YaRN RoPE (theta=150k, factor=32)

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder};
use serde::Deserialize;
use std::path::PathBuf;

use gallium_core::*;

// ─────────────────────────────────────────────────────────────────────────────
// MXFP4 E2M1 lookup: nibble (0–15) → float
//   format : sign=bit3, exp=bits[2:1], mant=bit0, exp_bias=1
//   block scale (E8M0): value = 2^(byte − 127)
//
// These are the *true* FP4 values, and `deq` below multiplies by the full
// `2^(e − 127)`. That is the safetensors convention (transformers'
// `convert_moe_packed_tensors`), and is deliberately not the split the GGUF
// path uses — `gallium_core::quantized` doubles this table and halves the
// scale (matching ggml). Both land on the same number; see the note on
// `quantized::e8m0_to_f32`.
// ─────────────────────────────────────────────────────────────────────────────
static MXFP4_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive (sign=0)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative (sign=1)
];

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

fn default_swiglu_limit() -> f32 {
    7.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    pub rope_scaling: Option<RopeScaling>,
}

impl GptOssConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MXFP4 MoE FFN
//
// Weights stored in OpenAI's MXFP4 block format:
//   blocks : [n_exp, out_dim, n_blocks, 16] U8  — 4-bit packed (2 values/byte)
//   scales : [n_exp, out_dim, n_blocks]    U8  — E8M0 per-block scale
//   bias   : [n_exp, out_dim]              BF16
//   n_blocks = hidden_size / 32  (32 values per 16-byte block)
// ─────────────────────────────────────────────────────────────────────────────

/// `y[o] = W[o, :] · x` for one MXFP4 weight in OpenAI's split block/scale
/// layout, streamed row by row without building an `[out_dim, in_dim]` f32
/// matrix. Each row's `nb` blocks are repacked into the GGUF 17-byte block
/// layout and dotted against `x` by the shared `Kernels::dequant_dot_mxfp4`.
///
/// - `blk_bytes`: `out_dim * nb * 16` — OpenAI byte `i` holds elements
///   `(2i, 2i+1)` (low, high nibble).
/// - `scl_bytes`: `out_dim * nb` — one raw E8M0 exponent byte per block.
/// - `x`:         `nb * 32`.
///
/// Not bit-identical to a whole-row dequantize + BLAS dot: the reduction
/// order differs.
fn mxfp4_matvec_openai(
    kernels: &KernelSet,
    blk_bytes: &[u8],
    scl_bytes: &[u8],
    nb: usize,
    out_dim: usize,
    x: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(x.len(), nb * 32);
    // 32-byte-aligned copy of x, reused for every output row (same trick as
    // `Tq2Tensor::matvec_expert`): a fresh `Vec<f32>` is only 4-byte aligned,
    // so over-allocate by 8 and slice to the first aligned offset.
    let mut scratch = vec![0.0f32; x.len() + 8];
    let pad = ((32 - (scratch.as_ptr() as usize % 32)) % 32) / 4;
    scratch[pad..pad + x.len()].copy_from_slice(x);
    let xa = &scratch[pad..pad + x.len()];

    use rayon::prelude::*;
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let mut gguf = vec![0u8; nb * 17];
            for blk in 0..nb {
                let oa = &blk_bytes[(o * nb + blk) * 16..][..16];
                let g = &mut gguf[blk * 17..][..17];
                g[0] = scl_bytes[o * nb + blk];
                for k in 0..8 {
                    // OpenAI element (2k, 2k+1) in byte k -> GGUF element
                    // (k, k+16); the second operand of each byte comes from
                    // OpenAI byte k+8 (element (2k+16, 2k+17)).
                    g[1 + 2 * k] = (oa[k] & 0x0F) | ((oa[k + 8] & 0x0F) << 4);
                    g[1 + 2 * k + 1] = ((oa[k] >> 4) & 0x0F) | (oa[k + 8] & 0xF0);
                }
            }
            kernels.dequant_dot_mxfp4(&gguf, xa)
        })
        .collect()
}

struct OssMoEFFN {
    gate_up_blocks: Tensor, // [n_exp, 2*inter, n_blocks, 16] U8
    gate_up_scales: Tensor, // [n_exp, 2*inter, n_blocks]    U8
    gate_up_bias: Tensor,   // [n_exp, 2*inter]              BF16
    down_blocks: Tensor,    // [n_exp, hidden,  n_blocks, 16] U8
    down_scales: Tensor,    // [n_exp, hidden,  n_blocks]    U8
    down_bias: Tensor,      // [n_exp, hidden]               BF16
    router_weight: Tensor,  // [n_exp, hidden]
    router_bias: Tensor,    // [n_exp]
    inter: usize,
    hidden: usize,
    n_blocks: usize, // = hidden / 32
    top_k: usize,
    swiglu_limit: f32,
    device: Device,
    /// CPU SIMD kernels for the fused MXFP4 matvec path below.
    kernels: KernelSet,
    /// Stream the MXFP4 bytes for a single-token decode (`matvec_expert`)
    /// instead of expanding each expert to an `[out_dim, in_dim]` f32 matrix
    /// (`deq`) for a `matmul`. On by default; `GALLIUM_GPT_OSS_FUSED_MXFP4=0`
    /// forces the expand path, for the A/B testsuite comparison — the same
    /// switch `gpt_oss_q.rs` reads for the GGUF path. Not bit-identical (the
    /// reduction order differs), so decode-only and CPU-only.
    fused_mxfp4: bool,
}

impl OssMoEFFN {
    fn load(cfg: &GptOssConfig, vb: &VarBuilder, vb_u8: &VarBuilder, i: usize) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let n_exp = cfg.num_local_experts;
        let n_blocks = hidden / 32; // 2880 / 32 = 90

        let pfx = format!("model.layers.{i}.mlp");
        let vb_exp = vb.pp(format!("{pfx}.experts"));
        let vu_exp = vb_u8.pp(format!("{pfx}.experts"));
        let vb_rtr = vb.pp(format!("{pfx}.router"));

        Ok(Self {
            gate_up_blocks: vu_exp.get((n_exp, inter * 2, n_blocks, 16), "gate_up_proj_blocks")?,
            gate_up_scales: vu_exp.get((n_exp, inter * 2, n_blocks), "gate_up_proj_scales")?,
            gate_up_bias: vb_exp.get((n_exp, inter * 2), "gate_up_proj_bias")?,
            down_blocks: vu_exp.get((n_exp, hidden, n_blocks, 16), "down_proj_blocks")?,
            down_scales: vu_exp.get((n_exp, hidden, n_blocks), "down_proj_scales")?,
            down_bias: vb_exp.get((n_exp, hidden), "down_proj_bias")?,
            router_weight: vb_rtr.get((n_exp, hidden), "weight")?,
            router_bias: vb_rtr.get(n_exp, "bias")?,
            inter,
            hidden,
            n_blocks,
            top_k: cfg.num_experts_per_tok,
            swiglu_limit: cfg.swiglu_limit,
            device: vb.device().clone(),
            kernels: KernelSet::detect(),
            fused_mxfp4: !matches!(
                std::env::var("GALLIUM_GPT_OSS_FUSED_MXFP4").as_deref(),
                Ok("0")
            ),
        })
    }

    /// Dequantize MXFP4 blocks+scales to [out_dim, hidden] F32.
    fn deq(&self, blocks: &Tensor, scales: &Tensor, out_dim: usize) -> Result<Tensor> {
        let b = blocks.flatten_all()?.to_vec1::<u8>()?;
        let s = scales.flatten_all()?.to_vec1::<u8>()?;
        let in_dim = self.n_blocks * 32;
        let mut out = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for blk in 0..self.n_blocks {
                let e = s[o * self.n_blocks + blk] as i32;
                let sc = if e == 0 { 0.0f32 } else { 2f32.powi(e - 127) };
                let bb = (o * self.n_blocks + blk) * 16;
                let ob = o * in_dim + blk * 32;
                for bi in 0..16 {
                    let byte = b[bb + bi];
                    out[ob + bi * 2] = MXFP4_TABLE[(byte & 0xF) as usize] * sc;
                    out[ob + bi * 2 + 1] = MXFP4_TABLE[(byte >> 4) as usize] * sc;
                }
            }
        }
        Tensor::from_vec(out, (out_dim, in_dim), &self.device)
    }

    /// `y[o] = W_expert[o, :] · x` for one expert and one activation row,
    /// streaming the MXFP4 weight straight from the `blocks`/`scales` tensors —
    /// **no `[out_dim, in_dim]` f32 matrix is built** (the counterpart of
    /// `gpt_oss_q.rs`'s `Tq2Tensor::matvec_expert` for the GGUF path).
    ///
    /// The shared kernel `Kernels::dequant_dot_mxfp4` expects the GGUF 17-byte
    /// block layout (`[E8M0 scale][16 bytes: byte i = element i low / i+16
    /// high]`), so each weight row's blocks are repacked into it here. The
    /// E8M0 scale byte and the E2M1/scale split already agree between the two
    /// conventions (`quantized::e8m0_to_f32` halves and doubles `E2M1_LUT` to
    /// match OpenAI's true FP4 table × `2^(e-127)`), so the repack is only a
    /// nibble permutation: OpenAI packs elements `(2i, 2i+1)` into byte `i`,
    /// GGUF packs `(i, i+16)`.
    fn matvec_expert(
        &self,
        blocks: &Tensor, // [n_exp, out_dim, n_blocks, 16] U8
        scales: &Tensor, // [n_exp, out_dim, n_blocks]     U8
        eidx: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let blk_bytes = blocks.i(eidx)?.flatten_all()?.to_vec1::<u8>()?; // out_dim*nb*16
        let scl_bytes = scales.i(eidx)?.flatten_all()?.to_vec1::<u8>()?; // out_dim*nb
        Ok(mxfp4_matvec_openai(
            &self.kernels,
            &blk_bytes,
            &scl_bytes,
            self.n_blocks,
            out_dim,
            x,
        ))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let ntok = b * seq;
        let xf = x.reshape((ntok, self.hidden))?.to_dtype(DType::F32)?;

        // Router logits → softmax → top-k selection
        let rw = self.router_weight.to_dtype(DType::F32)?;
        let rb = self.router_bias.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&xf.matmul(&rw.t()?)?.broadcast_add(&rb)?)?;
        let pv: Vec<Vec<f32>> = probs.to_vec2()?;

        // Stream the MXFP4 bytes (`matvec_expert`) instead of expanding each
        // expert to f32 for a `matmul` — a single-token decode on the CPU only,
        // same gate as `gpt_oss_q.rs`'s GGUF path.
        let fused = self.fused_mxfp4 && ntok == 1 && self.device.is_cpu();

        let mut out_toks = Vec::with_capacity(ntok);
        for tok in 0..ntok {
            let mut idx_w: Vec<(usize, f32)> = pv[tok].iter().copied().enumerate().collect();
            idx_w.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
            idx_w.truncate(self.top_k);
            let total = idx_w.iter().map(|(_, w)| w).sum::<f32>().max(1e-9);

            let tx = xf.narrow(0, tok, 1)?; // [1, hidden]
            let xrow: Vec<f32> = if fused {
                tx.flatten_all()?.to_vec1()?
            } else {
                Vec::new()
            };
            let mut tok_out = Tensor::zeros((1, self.hidden), DType::F32, &self.device)?;

            for (eidx, w) in &idx_w {
                let gu_b = self.gate_up_bias.i(*eidx)?.to_dtype(DType::F32)?;
                // Gate+up: [1, 2*inter]
                let gu = if fused {
                    let row = self.matvec_expert(
                        &self.gate_up_blocks,
                        &self.gate_up_scales,
                        *eidx,
                        self.inter * 2,
                        &xrow,
                    )?;
                    Tensor::from_vec(row, (1, self.inter * 2), &self.device)?
                        .broadcast_add(&gu_b)?
                } else {
                    let gu_w = self.deq(
                        &self.gate_up_blocks.i(*eidx)?,
                        &self.gate_up_scales.i(*eidx)?,
                        self.inter * 2,
                    )?;
                    tx.matmul(&gu_w.t()?)?.broadcast_add(&gu_b)?
                };
                // gate_up is interleaved: even indices = gate, odd indices = up.
                // Reshape [1, 2*inter] → [1, inter, 2] to split.
                let gu_split = gu.reshape((1, self.inter, 2))?;
                let gate = gu_split.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?; // [1, inter]
                let up = gu_split.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?; // [1, inter]

                // Gate: clamp from above only, then GLU: gate * sigmoid(gate * 1.702)
                // sigmoid(x) = (1 + tanh(x/2)) / 2  →  sigmoid(x*1.702) uses x*0.851
                let gate = gate.clamp(-1e38_f64, self.swiglu_limit as f64)?;
                let sig = ((&gate * 0.851_f64)?.tanh()? + 1.0_f64)? * 0.5_f64;
                let glu = (gate * (sig)?)?;

                // Up: clamp both sides, then shift by +1
                let up = up.clamp(-(self.swiglu_limit as f64), self.swiglu_limit as f64)?;
                let up1 = (up + 1.0_f64)?;

                let h = (glu * up1)?; // [1, inter]

                let d_b = self.down_bias.i(*eidx)?.to_dtype(DType::F32)?;
                // Down: [1, hidden]
                let eout = if fused {
                    let hv: Vec<f32> = h.flatten_all()?.to_vec1()?;
                    let row = self.matvec_expert(
                        &self.down_blocks,
                        &self.down_scales,
                        *eidx,
                        self.hidden,
                        &hv,
                    )?;
                    Tensor::from_vec(row, (1, self.hidden), &self.device)?.broadcast_add(&d_b)?
                } else {
                    let d_w = self.deq(
                        &self.down_blocks.i(*eidx)?,
                        &self.down_scales.i(*eidx)?,
                        self.hidden,
                    )?;
                    h.matmul(&d_w.t()?)?.broadcast_add(&d_b)?
                };

                tok_out = (tok_out + eout * (*w as f64 / total as f64))?;
            }
            out_toks.push(tok_out);
        }

        Tensor::cat(&out_toks, 0)?
            .reshape((b, seq, self.hidden))?
            .to_dtype(x.dtype())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer
// ─────────────────────────────────────────────────────────────────────────────

struct GptOssLayer {
    pre_norm: Norm,
    attn: Attention,
    post_norm: Norm,
    moe: OssMoEFFN,
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

pub struct GptOss {
    embed: Embedding,
    layers: Vec<GptOssLayer>,
    final_norm: Norm,
    lm_head: candle_nn::Linear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
    sliding_window: usize,
    layer_types: Vec<LayerType>,
    /// Narrow K/V to the window span before the scores matmul on sliding
    /// layers (exact — the dropped positions are the ones the mask sets to
    /// `-inf`; the attention sink rides the narrowed scores unchanged, see
    /// `gallium_core::attention::narrow_kv_to_mask`'s doc comment). GPT-OSS's
    /// window (128) is a quarter of Gemma 4 E4B's (512), so this is the
    /// architecture the narrowing was expected to help most — issue #232.
    /// `GALLIUM_GPT_OSS_KV_NARROW=0` forces the full-context path, for the A/B.
    kv_narrow: bool,
}

impl GptOss {
    /// Load from safetensors files.
    /// `vb` is a BF16/F32 VarBuilder for normal tensors.
    /// `safetensors_paths` is used to create a U8 VarBuilder for MXFP4 block tensors.
    pub fn load(
        cfg: &GptOssConfig,
        vb: VarBuilder,
        safetensors_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        // U8 VarBuilder for loading raw MXFP4 block/scale tensors without dtype conversion.
        let vb_u8 =
            unsafe { VarBuilder::from_mmaped_safetensors(safetensors_paths, DType::U8, device)? };

        let head_dim = cfg.head_dim();
        let scaling = match &cfg.rope_scaling {
            Some(rs) if rs.rope_type == "yarn" => RoPEScaling::YaRN {
                factor: rs.factor,
                original_max_position_embeddings: rs.original_max_position_embeddings,
                beta_fast: rs.beta_fast.unwrap_or(32.0),
                beta_slow: rs.beta_slow.unwrap_or(1.0),
            },
            _ => RoPEScaling::None,
        };
        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len: cfg.max_position_embeddings,
                theta: cfg.rope_theta,
                scaling,
                ..Default::default()
            },
            vb.dtype(),
            device,
        )?;

        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let attn_cfg = AttentionConfig {
            hidden_size: cfg.hidden_size,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            attn_bias: cfg.attention_bias,
            attn_sinks: true,
            ..Default::default()
        };

        let mut cache_layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp(format!("model.layers.{i}"));
            cache_layers.push(LayerCache::Kv(KvCache::new(cfg.max_position_embeddings)));
            layers.push(GptOssLayer {
                pre_norm: Norm::rms(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_l.pp("input_layernorm"),
                )?,
                attn: Attention::new(attn_cfg.clone(), vb_l.pp("self_attn"))?,
                post_norm: Norm::rms(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_l.pp("post_attention_layernorm"),
                )?,
                moe: OssMoEFFN::load(cfg, &vb, &vb_u8, i)?,
            });
        }

        let final_norm = Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            let w = vb
                .pp("model.embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            candle_nn::Linear::new(w, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed,
            layers,
            final_norm,
            lm_head,
            rope,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            sliding_window: cfg.sliding_window.unwrap_or(128),
            layer_types: cfg.layer_types.clone(),
            kv_narrow: !matches!(
                std::env::var("GALLIUM_GPT_OSS_KV_NARROW").as_deref(),
                Ok("0")
            ),
        })
    }
}

impl CausalLM for GptOss {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed.forward(token_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let is_sliding = matches!(self.layer_types[i], LayerType::SlidingAttention);
            // One tested decision for "does this layer need a mask this step",
            // shared with both Gemma 4 variants (gallium_core::mask, docs/TODO.md
            // §1.1). A sliding layer needs one at decode too, once the KV cache
            // outgrows the window; a full-attention layer at decode does not.
            let window = is_sliding.then_some(self.sliding_window);
            let mask = if !attention_mask_needed(seq_len, pos, window) {
                None
            } else if is_sliding && self.kv_narrow {
                Some(build_sliding_window_mask_narrowed(
                    seq_len,
                    pos,
                    self.sliding_window,
                    &self.device,
                )?)
            } else if is_sliding {
                Some(build_sliding_window_mask(
                    seq_len,
                    pos,
                    self.sliding_window,
                    &self.device,
                )?)
            } else {
                Some(build_causal_mask(seq_len, pos, &self.device)?)
            };

            let kv = self.cache.get_kv(i).expect("layer has kv cache");
            let h_norm = layer.pre_norm.forward(&h)?;
            let h_attn = layer
                .attn
                .forward(&h_norm, &self.rope, pos, kv, mask.as_ref())?;
            h = (h + h_attn)?;

            let h_norm = layer.post_norm.forward(&h)?;
            let h_moe = layer.moe.forward(&h_norm)?;
            h = (h + h_moe)?;
        }

        let h = self.final_norm.forward(&h)?;
        let logits = self
            .lm_head
            .forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        Ok(logits.to_dtype(DType::F32)?)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mxfp4_matvec_openai` (the fused stream-and-dot used for a single-token
    /// decode) must track a plain whole-row dequantize + dot in OpenAI's own
    /// convention — **not** bit-exactly (the reduction order differs), the same
    /// contract `gpt_oss_gguf_mxfp4_matvec_tracks_dequantize_matmul` holds for
    /// the GGUF path. This catches a repack / indexing bug; the real check is
    /// the `GALLIUM_GPT_OSS_FUSED_MXFP4` A/B testsuite run.
    #[test]
    fn fused_mxfp4_matvec_tracks_scalar_dequant_dot() {
        const NB: usize = 6; // 192 elements
        const OUT: usize = 40;
        let kernels = KernelSet::detect();

        let mut lcg: u64 = 0xda3e_39cb_94b6_95a5;
        let mut next = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) as u32
        };

        let mut blk = vec![0u8; OUT * NB * 16];
        for b in blk.iter_mut() {
            *b = (next() & 0xFF) as u8;
        }
        // E8M0 exponent bytes kept in a sane range (0 => zero block).
        let scl: Vec<u8> = (0..OUT * NB).map(|_| 120 + (next() % 16) as u8).collect();
        let x: Vec<f32> = (0..NB * 32)
            .map(|_| (next() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect();

        let got = mxfp4_matvec_openai(&kernels, &blk, &scl, NB, OUT, &x);

        // Reference: OpenAI-convention scalar dequant of each row, then dot.
        for (o, &g) in got.iter().enumerate() {
            let mut want = 0.0f32;
            for block in 0..NB {
                let e = scl[o * NB + block] as i32;
                let sc = if e == 0 { 0.0 } else { 2f32.powi(e - 127) };
                for bi in 0..16 {
                    let byte = blk[(o * NB + block) * 16 + bi];
                    let xb = block * 32 + bi * 2;
                    want += MXFP4_TABLE[(byte & 0xF) as usize] * sc * x[xb];
                    want += MXFP4_TABLE[(byte >> 4) as usize] * sc * x[xb + 1];
                }
            }
            let tol = want.abs() * 2e-3 + 1e-3;
            assert!(
                (g - want).abs() <= tol,
                "row {o}: fused {g} vs scalar {want} (tol {tol})"
            );
        }
    }
}
