//! Gemma 4 multimodal model: vision tower + language model.
//!
//! The vision tower is Gemma 4's own encoder (not SigLIP):
//!   - patch_size=16, hidden=768, 16 transformer layers
//!   - 2D positional RoPE applied independently per spatial axis
//!   - 3× spatial average pooling after encoding
//!   - Linear projection (768→text_hidden) into the language model embedding space
//!
//! Image tokens in the input sequence (token_id == image_token_id=258880) are replaced
//! in-place with projected vision features before the language model forward pass.
//!
//! # Usage
//!
//! ```ignore
//! // Preprocess image externally and call:
//! let feats = model.encode_image(&pixel_values, &pixel_position_ids)?;
//! model.set_image_features(feats);
//! // Then generate as usual via CausalLM::forward.
//! ```
//!
//! `pixel_values` shape: `[batch, num_patches, 3 * patch_size^2]` (f32, values in `[0, 1]`).
//! `pixel_position_ids` shape: `[batch, num_patches, 2]` (i64, (x, y); padding patches = (-1,-1)).

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use serde::Deserialize;

use crate::gemma4::{Gemma4, Gemma4Config};
use crate::gemma4_q::Gemma4Q;
use gallium_core::quantized::{GgufMetadata, QVarBuilder};
use gallium_core::*;

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,         // 768
    pub intermediate_size: usize,   // 3072
    pub num_hidden_layers: usize,   // 16
    pub num_attention_heads: usize, // 12
    #[serde(default)]
    pub head_dim: Option<usize>, // 64 (defaults to hidden/heads)
    pub rms_norm_eps: f64,          // 1e-6
    pub patch_size: usize,          // 16
    #[serde(default = "default_pooling")]
    pub pooling_kernel_size: usize, // 3
    #[serde(default = "default_pos_size")]
    pub position_embedding_size: usize, // 10240
    /// Soft tokens per image for the reference processor's default resize —
    /// `max_soft_tokens` in `Gemma4ImageProcessor`, `280` for every published
    /// Gemma 4. Drives the patch budget in [`crate::gemma4_image`].
    #[serde(default = "default_output_length")]
    pub default_output_length: usize, // 280
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>, // extract rope_theta from here
}

fn default_pooling() -> usize {
    3
}
fn default_pos_size() -> usize {
    10240
}
fn default_output_length() -> usize {
    280
}

impl Gemma4VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4MultimodalConfig {
    pub text_config: Gemma4Config,
    pub vision_config: Gemma4VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
}

fn default_image_token_id() -> u32 {
    258_880
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// `Gemma4ClippableLinear`: a bias-free Linear that clamps its input to
/// `[input_min, input_max]` and its output to `[output_min, output_max]`.
///
/// `use_clipped_linears: true` for every Gemma 4 vision tower, and the bounds
/// are **trained values stored in the checkpoint** (per-layer, e.g. ±6 at layer
/// 0 growing to ±20 deep in the stack) — not the `±inf` the module initialises
/// them to. Skipping the clamps lets activations run past the range the tower
/// was trained in, and the pooled features come out structurally wrong even
/// though they stay finite. Weights live under `<name>.linear.weight`; the four
/// bounds are 0-d tensors `<name>.{input,output}_{min,max}`.
struct ClippedLinear {
    linear: Linear,
    in_min: f64,
    in_max: f64,
    out_min: f64,
    out_max: f64,
}

impl ClippedLinear {
    fn load(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear_no_bias(in_features, out_features, vb.pp("linear"))?;
        let bound = |name: &str, default: f64| -> f64 {
            vb.get((), name)
                .and_then(|t| t.to_dtype(DType::F32)?.to_scalar::<f32>())
                .map(|v| v as f64)
                .unwrap_or(default)
        };
        Ok(Self {
            linear,
            in_min: bound("input_min", f64::NEG_INFINITY),
            in_max: bound("input_max", f64::INFINITY),
            out_min: bound("output_min", f64::NEG_INFINITY),
            out_max: bound("output_max", f64::INFINITY),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.clamp(self.in_min, self.in_max)?;
        self.linear.forward(&x)?.clamp(self.out_min, self.out_max)
    }
}

/// RMSNorm without a learned scale: `x / rms(x)`.
///
/// Used for v_norm in vision attention and the projector's pre-norm (with_scale=False
/// in the reference).
fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = (x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
}

/// Apply RoPE to a single [b, s, h, dim/2] chunk.
fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // cos/sin: [b, s, 1, dim/2] (heads already unsqueezed by caller)
    x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
}

/// Apply 2D spatial RoPE.
///
/// Rotates the first `head_dim/2` channels using x-position frequencies and
/// the last `head_dim/2` channels using y-position frequencies.
///
/// `qk`: `[b, s, heads, head_dim]`
/// `cos`, `sin`: `[b, s, head_dim]` (x-half then y-half concatenated)
fn apply_vision_rope(qk: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = qk.dim(candle_core::D::Minus1)?;
    let half = d / 2;

    let qk_x = qk.narrow(3, 0, half)?;
    let qk_y = qk.narrow(3, half, half)?;

    let cos_x = cos.narrow(2, 0, half)?.unsqueeze(2)?;
    let sin_x = sin.narrow(2, 0, half)?.unsqueeze(2)?;
    let cos_y = cos.narrow(2, half, half)?.unsqueeze(2)?;
    let sin_y = sin.narrow(2, half, half)?.unsqueeze(2)?;

    let rot_x = apply_rotary(&qk_x, &cos_x, &sin_x)?;
    let rot_y = apply_rotary(&qk_y, &cos_y, &sin_y)?;
    Tensor::cat(&[&rot_x, &rot_y], 3)
}

// ── 2D Vision RoPE ───────────────────────────────────────────────────────────

struct Vision2DRoPE {
    inv_freq: Tensor, // [spatial_dim/2] = [16]
}

impl Vision2DRoPE {
    fn new(head_dim: usize, theta: f64, device: &Device) -> Result<Self> {
        // spatial_dim = head_dim / 2 (each axis gets head_dim/2 channels)
        // inv_freq[j] = 1 / theta^(2j / spatial_dim) for j in 0..spatial_dim/2
        let spatial_dim = head_dim / 2; // 32
        let n = spatial_dim / 2; // 16
        let inv: Vec<f32> = (0..n)
            .map(|j| (1.0 / theta.powf(2.0 * j as f64 / spatial_dim as f64)) as f32)
            .collect();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv, n, device)?,
        })
    }

    /// Returns `(cos, sin)` each of shape `[b, num_patches, head_dim]`.
    fn compute(&self, pixel_position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // pixel_position_ids: [b, s, 2] i64 (padding = -1)
        let (b, s, _) = pixel_position_ids.dims3()?;

        // Clamp padding positions to 0 (padding patches are masked out via padding_positions).
        let pos = pixel_position_ids
            .to_dtype(DType::F32)?
            .clamp(0f32, f32::INFINITY)?; // [b, s, 2]

        let pos_x = pos.narrow(2, 0, 1)?.squeeze(2)?.reshape((b * s, 1))?; // [b*s, 1]
        let pos_y = pos.narrow(2, 1, 1)?.squeeze(2)?.reshape((b * s, 1))?;

        let inv = self.inv_freq.reshape((1, self.inv_freq.dim(0)?))?; // [1, n]

        // freqs: [b*s, n]
        let freqs_x = pos_x
            .broadcast_mul(&inv)?
            .reshape((b, s, self.inv_freq.dim(0)?))?;
        let freqs_y = pos_y
            .broadcast_mul(&inv)?
            .reshape((b, s, self.inv_freq.dim(0)?))?;

        // emb: [b, s, 2n=32] (cat freqs with itself for full rotation pairs)
        let emb_x = Tensor::cat(&[&freqs_x, &freqs_x], 2)?;
        let emb_y = Tensor::cat(&[&freqs_y, &freqs_y], 2)?;

        // cos/sin: [b, s, 64] (x-half || y-half)
        let cos = Tensor::cat(&[&emb_x.cos()?, &emb_y.cos()?], 2)?;
        let sin = Tensor::cat(&[&emb_x.sin()?, &emb_y.sin()?], 2)?;
        Ok((cos, sin))
    }
}

// ── Patch Embedder ───────────────────────────────────────────────────────────

struct VisionPatchEmbedder {
    input_proj: Linear, // 3*patch_size^2 → hidden
    pos_table: Tensor,  // [2, position_embedding_size, hidden]
    hidden_size: usize,
}

impl VisionPatchEmbedder {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_features = 3 * cfg.patch_size * cfg.patch_size;
        let input_proj = linear_no_bias(in_features, cfg.hidden_size, vb.pp("input_proj"))?;
        let pos_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            pos_table,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(
        &self,
        pixel_values: &Tensor,       // [b, s, 3*patch_size^2]
        pixel_position_ids: &Tensor, // [b, s, 2] i64
        padding_positions: &Tensor,  // [b, s] bool-like (1 = padding)
    ) -> Result<Tensor> {
        // Scale from [0,1] to [-1,1].
        let pv = ((pixel_values.to_dtype(self.pos_table.dtype())? * 2.0)? - 1.0)?;
        let hidden = self.input_proj.forward(&pv)?; // [b, s, hidden]

        let pos_emb = self.position_embeddings(pixel_position_ids, padding_positions)?;
        hidden + pos_emb
    }

    fn position_embeddings(
        &self,
        pixel_position_ids: &Tensor, // [b, s, 2] i64
        padding_positions: &Tensor,  // [b, s]
    ) -> Result<Tensor> {
        let (b, s, _) = pixel_position_ids.dims3()?;
        let pos_size = self.pos_table.dim(1)?;

        // Clamp to [0, pos_size-1] (padding patches at -1 → 0, zeroed later).
        let clamped = pixel_position_ids
            .to_dtype(DType::I64)?
            .clamp(0i64, (pos_size - 1) as i64)?;

        let cx = clamped
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .reshape((b * s,))?
            .to_dtype(DType::U32)?;
        let cy = clamped
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .reshape((b * s,))?
            .to_dtype(DType::U32)?;

        let table_x = self.pos_table.narrow(0, 0, 1)?.squeeze(0)?; // [pos_size, hidden]
        let table_y = self.pos_table.narrow(0, 1, 1)?.squeeze(0)?;

        // Lookup and sum x + y contributions.
        let emb = (table_x.index_select(&cx, 0)? + table_y.index_select(&cy, 0)?)?.reshape((
            b,
            s,
            self.hidden_size,
        ))?;

        // Zero out padding patches by multiplying by (1 - padding_mask).
        let not_pad = (padding_positions.to_dtype(emb.dtype())? - 1.0)?
            .neg()?
            .unsqueeze(2)?; // [b, s, 1]
        emb.broadcast_mul(&not_pad)
    }
}

// ── Vision Attention ─────────────────────────────────────────────────────────

struct VisionAttention {
    q_proj: ClippedLinear,
    k_proj: ClippedLinear,
    v_proj: ClippedLinear,
    o_proj: ClippedLinear,
    q_norm: Norm,
    k_norm: Norm,
    num_heads: usize,
    head_dim: usize,
    norm_eps: f64,
}

impl VisionAttention {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let qkv_out = cfg.num_attention_heads * head_dim;
        let q_proj = ClippedLinear::load(h, qkv_out, vb.pp("q_proj"))?;
        let k_proj = ClippedLinear::load(h, qkv_out, vb.pp("k_proj"))?;
        let v_proj = ClippedLinear::load(h, qkv_out, vb.pp("v_proj"))?;
        let o_proj = ClippedLinear::load(qkv_out, h, vb.pp("o_proj"))?;
        let q_norm = Norm::rms(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Norm::rms(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            head_dim,
            norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        x: &Tensor,   // [b, s, hidden]
        cos: &Tensor, // [b, s, head_dim]
        sin: &Tensor, // [b, s, head_dim]
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.num_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        // v_norm has no learned scale (with_scale=False in reference).
        let v = rms_norm_no_scale(&v, self.norm_eps)?;

        // Apply 2D RoPE to Q and K.
        let q = apply_vision_rope(&q, cos, sin)?;
        let k = apply_vision_rope(&k, cos, sin)?;

        // Transpose to [b, heads, s, head_dim] for matmul.
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Bidirectional attention, scale=1.0 (reference uses self.scaling = 1.0).
        let attn_w = q.matmul(&k.transpose(2, 3)?)?;
        let attn_w =
            candle_nn::ops::softmax(&attn_w.to_dtype(DType::F32)?, 3)?.to_dtype(q.dtype())?;
        let out = attn_w.matmul(&v)?; // [b, heads, s, head_dim]

        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

// ── Vision MLP ───────────────────────────────────────────────────────────────

struct VisionMLP {
    gate_proj: ClippedLinear,
    up_proj: ClippedLinear,
    down_proj: ClippedLinear,
}

impl VisionMLP {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: ClippedLinear::load(h, i, vb.pp("gate_proj"))?,
            up_proj: ClippedLinear::load(h, i, vb.pp("up_proj"))?,
            down_proj: ClippedLinear::load(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Vision Encoder Layer ──────────────────────────────────────────────────────

struct VisionBlock {
    input_norm: Norm,
    attn: VisionAttention,
    post_attn_norm: Norm,
    pre_ffn_norm: Norm,
    mlp: VisionMLP,
    post_ffn_norm: Norm,
}

impl VisionBlock {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        Ok(Self {
            input_norm: Norm::rms(h, eps, vb.pp("input_layernorm"))?,
            attn: VisionAttention::load(cfg, vb.pp("self_attn"))?,
            post_attn_norm: Norm::rms(h, eps, vb.pp("post_attention_layernorm"))?,
            pre_ffn_norm: Norm::rms(h, eps, vb.pp("pre_feedforward_layernorm"))?,
            mlp: VisionMLP::load(cfg, vb.pp("mlp"))?,
            post_ffn_norm: Norm::rms(h, eps, vb.pp("post_feedforward_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Attention branch: pre-norm → attn → post-norm → residual.
        let h = self.attn.forward(&self.input_norm.forward(x)?, cos, sin)?;
        let x = (x + self.post_attn_norm.forward(&h)?)?;

        // FFN branch: pre-norm → mlp → post-norm → residual.
        let h = self.mlp.forward(&self.pre_ffn_norm.forward(&x)?)?;
        &x + self.post_ffn_norm.forward(&h)?
    }
}

// ── Vision Encoder ────────────────────────────────────────────────────────────

struct VisionEncoder {
    layers: Vec<VisionBlock>,
    rope: Vision2DRoPE,
}

impl VisionEncoder {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| VisionBlock::load(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<_>>()?;
        let rope = Vision2DRoPE::new(cfg.head_dim(), cfg.rope_theta(), device)?;
        Ok(Self { layers, rope })
    }

    fn forward(&self, embeds: &Tensor, pixel_position_ids: &Tensor) -> Result<Tensor> {
        let (cos, sin) = self.rope.compute(pixel_position_ids)?;
        let cos = cos.to_dtype(embeds.dtype())?;
        let sin = sin.to_dtype(embeds.dtype())?;
        let mut h = embeds.clone();
        for block in &self.layers {
            h = block.forward(&h, &cos, &sin)?;
        }
        Ok(h)
    }
}

// ── Vision Pooler ─────────────────────────────────────────────────────────────

struct VisionPooler {
    scale: f64, // sqrt(hidden_size)
}

impl VisionPooler {
    fn new(hidden_size: usize) -> Self {
        Self {
            scale: (hidden_size as f64).sqrt(),
        }
    }

    /// Spatial average pooling + scaling.
    ///
    /// Groups patches into k×k blocks using their (x, y) positions and averages each block.
    /// k = sqrt(num_patches / output_len).
    fn pool(
        &self,
        hidden: &Tensor,             // [b, num_patches, hidden]
        pixel_position_ids: &Tensor, // [b, num_patches, 2] i64
        padding_positions: &Tensor,  // [b, num_patches]
        output_len: usize,
    ) -> Result<Tensor> {
        let (b, s, d) = hidden.dims3()?;
        let k = ((s / output_len) as f64).sqrt() as usize;

        // Zero out padding patches before pooling.
        let not_pad = (padding_positions.to_dtype(hidden.dtype())? - 1.0)?
            .neg()?
            .unsqueeze(2)?;
        let hidden = hidden.broadcast_mul(&not_pad)?;

        if s == output_len {
            return hidden * self.scale;
        }

        // CPU-based scatter-average (runs once during prefill, performance non-critical).
        let h_f32 = hidden.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let pos = pixel_position_ids.to_vec3::<i64>()?;
        let k2 = k * k;

        let mut result = vec![0.0f32; b * output_len * d];

        for bi in 0..b {
            let max_x = pos[bi].iter().map(|p| p[0].max(0)).max().unwrap_or(0) + 1;
            let num_cols = (max_x as usize).div_ceil(k);

            for si in 0..s {
                let px = pos[bi][si][0];
                let py = pos[bi][si][1];
                if px < 0 || py < 0 {
                    continue;
                } // padding

                let kx = px as usize / k;
                let ky = py as usize / k;
                let ki = kx + num_cols * ky;
                if ki < output_len {
                    let base = (bi * output_len + ki) * d;
                    for di in 0..d {
                        result[base + di] += h_f32[bi][si][di];
                    }
                }
            }
        }
        // Divide by k^2 and apply sqrt(hidden_size) scale.
        let combined_scale = self.scale / k2 as f64;
        for v in result.iter_mut() {
            *v *= combined_scale as f32;
        }

        Tensor::from_vec(result, (b, output_len, d), hidden.device())?.to_dtype(hidden.dtype())
    }
}

// ── Vision Projector ──────────────────────────────────────────────────────────

struct VisionProjector {
    pre_norm_eps: f64,
    proj: Linear, // vision_hidden → text_hidden
}

impl VisionProjector {
    fn load(
        vision_hidden: usize,
        text_hidden: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        // embedding_pre_projection_norm has no weight (with_scale=False); only proj has weights.
        Ok(Self {
            pre_norm_eps: norm_eps,
            proj: linear_no_bias(vision_hidden, text_hidden, vb.pp("embedding_projection"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = rms_norm_no_scale(x, self.pre_norm_eps)?;
        self.proj.forward(&normed)
    }
}

// ── Gemma4Text ────────────────────────────────────────────────────────────────

/// The language-model half of [`Gemma4Multimodal`], enum-dispatched over the
/// two checkpoint formats (house style: concrete structs + enum dispatch, not
/// a second trait). Both variants expose the same three-phase split — embed →
/// PLE → blocks — which is what lets one copy of the merge logic in
/// [`Gemma4Multimodal::forward`] serve both.
pub enum Gemma4Text {
    /// safetensors ([`Gemma4`]), weights under `model.language_model.*`.
    Full(Gemma4),
    /// GGUF ([`Gemma4Q`]), the same text model the text-only candle path loads.
    Quantized(Gemma4Q),
}

impl Gemma4Text {
    fn embed_scaled(&self, token_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Full(m) => m.embed_scaled(token_ids),
            Self::Quantized(m) => m.embed_scaled(token_ids),
        }
    }

    /// `None` only on a PLE-less variant (26B-A4B GGUF); the safetensors
    /// loader targets E4B, whose PLE is unconditional.
    fn compute_ple(&self, token_ids: &Tensor, h_embed: &Tensor) -> Result<Option<Tensor>> {
        match self {
            Self::Full(m) => Ok(Some(m.compute_ple(token_ids, h_embed)?)),
            Self::Quantized(m) => m.compute_ple_opt(token_ids, h_embed),
        }
    }

    fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        per_layer: Option<&Tensor>,
        pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Full(m) => {
                let ple = per_layer.ok_or_else(|| {
                    candle_core::Error::Msg("safetensors Gemma4 requires PLE inputs".into())
                })?;
                m.forward_embeds(inputs_embeds, ple, pos, seq_len)
            }
            Self::Quantized(m) => m.forward_embeds(inputs_embeds, per_layer, pos, seq_len),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Full(m) => CausalLM::reset(m),
            Self::Quantized(m) => CausalLM::reset(m),
        }
    }

    /// Delegates each variant's own answer: `Gemma4Q` exposes its cache (so the
    /// GGUF multimodal path reuses KV across ReAct iterations, exactly like the
    /// text-only GGUF path), `Gemma4` does not expose one today — the
    /// safetensors path re-evaluates each prompt whole, unchanged by this enum.
    fn cache(&mut self) -> Option<&mut ModelCache> {
        match self {
            Self::Full(m) => CausalLM::cache(m),
            Self::Quantized(m) => CausalLM::cache(m),
        }
    }
}

// ── Gemma4Multimodal ──────────────────────────────────────────────────────────

pub struct Gemma4Multimodal {
    text: Gemma4Text,
    patch_embedder: VisionPatchEmbedder,
    encoder: VisionEncoder,
    pooler: VisionPooler,
    projector: VisionProjector,
    image_token_id: u32,
    pad_token_id: u32,
    pooling_kernel_size: usize,
    pending_image_embeds: Option<Tensor>,
    device: Device,
}

/// The four vision-side pieces, shared by both checkpoint formats. `vb32` must
/// be an f32 builder rooted where `model.vision_tower.*` / `model.embed_vision.*`
/// resolve — the safetensors file itself, or the renamed-mmproj tensor map.
fn load_tower(
    vc: &Gemma4VisionConfig,
    text_hidden: usize,
    vb32: &VarBuilder,
    device: &Device,
) -> Result<(
    VisionPatchEmbedder,
    VisionEncoder,
    VisionPooler,
    VisionProjector,
)> {
    let vb_vt = vb32.pp("model.vision_tower");
    let patch_embedder = VisionPatchEmbedder::load(vc, vb_vt.pp("patch_embedder"))?;
    let encoder = VisionEncoder::load(vc, vb_vt.pp("encoder"), device)?;
    let pooler = VisionPooler::new(vc.hidden_size);
    let projector = VisionProjector::load(
        vc.hidden_size,
        text_hidden,
        vc.rms_norm_eps,
        vb32.pp("model.embed_vision"),
    )?;
    Ok((patch_embedder, encoder, pooler, projector))
}

impl Gemma4Multimodal {
    pub fn load(cfg: &Gemma4MultimodalConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        // Text model: weights live under model.language_model.* in the multimodal safetensors.
        let text = Gemma4::load(&cfg.text_config, vb.clone(), device)?;

        // The vision tower runs in **f32**, whatever dtype the text model loaded
        // at. The reference forces f32 through the pooler's `sqrt(hidden_size)`
        // scaling and the 2D-RoPE precisely because those overflow f16 (max
        // 65504); loading the whole tower f32 is the simplest way to match that
        // and it is a small model (16 layers, hidden 768). `encode_image`'s
        // output is cast back to the text model's dtype at injection.
        let vb32 = vb.to_dtype(DType::F32);
        let vc = &cfg.vision_config;
        let (patch_embedder, encoder, pooler, projector) =
            load_tower(vc, cfg.text_config.hidden_size, &vb32, device)?;

        Ok(Self {
            text: Gemma4Text::Full(text),
            patch_embedder,
            encoder,
            pooler,
            projector,
            image_token_id: cfg.image_token_id,
            pad_token_id: cfg.text_config.pad_token_id,
            pooling_kernel_size: vc.pooling_kernel_size,
            pending_image_embeds: None,
            device: device.clone(),
        })
    }

    /// Build the multimodal model from the two GGUF files the llama.cpp
    /// ecosystem distributes: the text model (`metadata` + `vb`, exactly what
    /// [`Gemma4Q::load`] takes) and the `mmproj-*.gguf` beside it, which holds
    /// the vision tower under llama.cpp's `clip` naming (`v.blk.*`, `mm.*`).
    ///
    /// The mmproj tensors are dequantized to f32 (the tower runs f32 — see
    /// `load` above), renamed to the HF safetensors paths, and fed through the
    /// same `load_tower` as the safetensors checkpoint, so there is exactly one
    /// description of the tower's structure. The renames were verified
    /// tensor-by-tensor against `unsloth/gemma-4-E4B-it`'s safetensors:
    /// bit-exact on every mapped tensor, including the ClippedLinear bounds and
    /// the conv→linear patch-embedding permutation (docs/MULTIMODAL.md).
    ///
    /// Returns the vision config alongside the model because the GGUF path has
    /// no `config.json` — the provider needs it for the image preprocessor.
    pub fn load_gguf(
        metadata: &GgufMetadata,
        vb: &QVarBuilder,
        mmproj_path: &std::path::Path,
        device: &Device,
    ) -> Result<(Self, Gemma4VisionConfig)> {
        let text = Gemma4Q::load(metadata, vb, device)?;
        let text_prefix = metadata
            .get_str("general.architecture")
            .unwrap_or_else(|_| "gemma4".to_string());
        let text_hidden = metadata.get_u32(&format!("{text_prefix}.embedding_length"))? as usize;
        // Gemma's pad id is 0; the GGUF may or may not record it.
        let pad_token_id = metadata.get_u32_or("tokenizer.ggml.padding_token_id", 0);

        let (mm_meta, mm_vb) = gallium_core::quantized::load_gguf(mmproj_path, device)?;
        if !mm_vb.contains("v.patch_embd.weight") {
            candle_core::bail!(
                "mmproj {} has no vision encoder (no v.patch_embd.weight) — \
                 an audio-only projector cannot drive the candle image path",
                mmproj_path.display()
            );
        }

        let vc = vision_config_from_mmproj(&mm_meta)?;
        let tensors = rename_mmproj_tensors(&mm_vb, vc.patch_size, device)?;
        let vb32 = VarBuilder::from_tensors(tensors, DType::F32, device);
        let (patch_embedder, encoder, pooler, projector) =
            load_tower(&vc, text_hidden, &vb32, device)?;

        let model = Self {
            text: Gemma4Text::Quantized(text),
            patch_embedder,
            encoder,
            pooler,
            projector,
            image_token_id: default_image_token_id(),
            pad_token_id,
            pooling_kernel_size: vc.pooling_kernel_size,
            pending_image_embeds: None,
            device: device.clone(),
        };
        Ok((model, vc))
    }

    /// Encode image patches to language model space.
    ///
    /// Returns `[batch * output_tokens, text_hidden]` — the projected soft tokens ready to
    /// scatter into the token embedding sequence.  For `batch=1` with no padding this is
    /// simply `[output_tokens, text_hidden]`.
    ///
    /// `pixel_values`: `[b, num_patches, 3 * patch_size^2]` floats in `[0, 1]`.
    /// `pixel_position_ids`: `[b, num_patches, 2]` i64 (x, y); padding patches = `(-1, -1)`.
    pub fn encode_image(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
    ) -> Result<Tensor> {
        // padding_positions: [b, num_patches] — 1.0 where patch is padding
        let px = pixel_position_ids.narrow(2, 0, 1)?.squeeze(2)?; // [b, s] i64
        let padding_positions = px.lt(0i64)?.to_dtype(DType::F32)?; // 1.0 = padding

        let embeds =
            self.patch_embedder
                .forward(pixel_values, pixel_position_ids, &padding_positions)?;

        let encoded = self.encoder.forward(&embeds, pixel_position_ids)?;

        let num_patches = pixel_values.dim(1)?;
        let output_len = num_patches / (self.pooling_kernel_size * self.pooling_kernel_size);

        let pooled =
            self.pooler
                .pool(&encoded, pixel_position_ids, &padding_positions, output_len)?;

        // Flatten batch dimension: [b, output_len, text_hidden] → [(b*output_len), text_hidden]
        let (b, ol, _) = pooled.dims3()?;
        let flat = pooled.reshape((b * ol, pooled.dim(2)?))?;
        self.projector.forward(&flat)
    }

    /// Store image features for injection during the next prefill forward pass.
    pub fn set_image_features(&mut self, features: Tensor) {
        self.pending_image_embeds = Some(features);
    }

    /// Replace `image_token_id` positions in `h_embed` with `img_feats` (in token order).
    fn inject_image_features(
        &self,
        token_ids: &Tensor, // [b, s] u32
        h_embed: Tensor,    // [b, s, hidden]
        img_feats: &Tensor, // [num_img_tokens, hidden]
    ) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        let img_token = self.image_token_id;

        let ids = token_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let img_vec: Vec<Vec<f32>> = img_feats.to_dtype(DType::F32)?.to_vec2()?;
        let mut h_vec: Vec<Vec<Vec<f32>>> = h_embed.to_dtype(DType::F32)?.to_vec3()?;

        let mut img_idx = 0;
        'outer: for bi in 0..b {
            for si in 0..s {
                if ids[bi][si] == img_token {
                    if img_idx >= img_vec.len() {
                        break 'outer;
                    }
                    h_vec[bi][si] = img_vec[img_idx].clone();
                    img_idx += 1;
                }
            }
        }

        Tensor::new(h_vec, &self.device)?.to_dtype(h_embed.dtype())
    }
}

/// One `v.blk.N.<rest>` tail → the HF module path inside
/// `encoder.layers.N.`, or `None` for a name this loader does not know.
fn block_suffix(rest: &str) -> Option<String> {
    let (module, leaf) = rest.rsplit_once('.')?;
    let mapped = match module {
        "ln1" => "input_layernorm".to_string(),
        "ln2" => "pre_feedforward_layernorm".to_string(),
        "attn_post_norm" => "post_attention_layernorm".to_string(),
        "ffn_post_norm" => "post_feedforward_layernorm".to_string(),
        "attn_q_norm" => "self_attn.q_norm".to_string(),
        "attn_k_norm" => "self_attn.k_norm".to_string(),
        "attn_q" | "attn_k" | "attn_v" | "attn_out" | "ffn_gate" | "ffn_up" | "ffn_down" => {
            let (prefix, name) = match module {
                "attn_q" => ("self_attn", "q_proj"),
                "attn_k" => ("self_attn", "k_proj"),
                "attn_v" => ("self_attn", "v_proj"),
                "attn_out" => ("self_attn", "o_proj"),
                "ffn_gate" => ("mlp", "gate_proj"),
                "ffn_up" => ("mlp", "up_proj"),
                _ => ("mlp", "down_proj"),
            };
            // The weight lives under `.linear.`; the clip bounds sit beside it
            // on the module itself.
            return Some(if leaf == "weight" {
                format!("{prefix}.{name}.linear.weight")
            } else {
                format!("{prefix}.{name}.{leaf}")
            });
        }
        _ => return None,
    };
    (leaf == "weight").then(|| format!("{mapped}.weight"))
}

/// [`Gemma4VisionConfig`] from an mmproj GGUF's `clip.vision.*` metadata —
/// the GGUF path's substitute for `config.json`'s `vision_config`.
fn vision_config_from_mmproj(meta: &GgufMetadata) -> Result<Gemma4VisionConfig> {
    let u = |k: &str| -> Result<usize> {
        meta.get_u32(&format!("clip.vision.{k}"))
            .map(|v| v as usize)
            .map_err(|e| candle_core::Error::Msg(format!("mmproj metadata clip.vision.{k}: {e}")))
    };
    Ok(Gemma4VisionConfig {
        hidden_size: u("embedding_length")?,
        intermediate_size: u("feed_forward_length")?,
        num_hidden_layers: u("block_count")?,
        num_attention_heads: u("attention.head_count")?,
        head_dim: None,
        rms_norm_eps: meta.get_f32_or("clip.vision.attention.layer_norm_epsilon", 1e-6) as f64,
        patch_size: u("patch_size")?,
        // llama.cpp's KEY_PROJ_SCALE_FACTOR; its clip loader defaults gemma4v
        // to 3 the same way.
        pooling_kernel_size: meta.get_u32_or("clip.vision.projector.scale_factor", 3) as usize,
        position_embedding_size: default_pos_size(),
        default_output_length: default_output_length(),
        // No rope key in the mmproj; llama.cpp hardcodes 100.0 for gemma4v,
        // which is also `Gemma4VisionConfig::rope_theta`'s default.
        rope_parameters: None,
    })
}

/// Dequantize the mmproj's vision tensors and rename them to the HF
/// safetensors paths `load_tower` reads, so both formats share one loader.
///
/// llama.cpp name → HF name, per block:
///   `ln1` → `input_layernorm`, `attn_post_norm` → `post_attention_layernorm`,
///   `ln2` → `pre_feedforward_layernorm`, `ffn_post_norm` → `post_feedforward_layernorm`,
///   `attn_{q,k,v,out}` → `self_attn.{q,k,v,o}_proj.linear` (+ their four
///   clip bounds, stored `[1]` in GGUF and reshaped to the 0-d scalars
///   `ClippedLinear` reads), `attn_{q,k}_norm` → `self_attn.{q,k}_norm`,
///   `ffn_{gate,up,down}` → `mlp.{gate,up,down}_proj.linear` (+ bounds).
///
/// Top level: `v.position_embd.weight` maps 1:1 to the `[2, pos, hidden]`
/// position table; `v.patch_embd.weight` is stored as a conv kernel
/// `[out, 3, ps, ps]` and becomes the patchify Linear via `permute(0,2,3,1)`
/// (the patch flattens (y, x, channel), channel innermost — see
/// `gemma4_image`); `mm.input_projection.weight` is the projector. Audio
/// tensors (`a.*`, `mm.a.*`) are skipped — there is no audio tower here.
/// Any *other* unrecognized `v.*`/`mm.*` name is an error, not a skip: a
/// tensor this function cannot place is a tower it does not understand.
fn rename_mmproj_tensors(
    mm_vb: &QVarBuilder,
    patch_size: usize,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    const VT: &str = "model.vision_tower";

    let mut out = HashMap::new();
    for name in mm_vb.tensor_names() {
        if name.starts_with("a.") || name.starts_with("mm.a.") {
            continue; // audio tower — not loaded
        }
        let t = mm_vb.get(name)?.dequantize(device)?.to_dtype(DType::F32)?;
        let (hf_name, t) = if name == "v.patch_embd.weight" {
            let (out_f, c, kh, kw) = t.dims4()?;
            if (c, kh, kw) != (3, patch_size, patch_size) {
                candle_core::bail!(
                    "v.patch_embd.weight is [{out_f}, {c}, {kh}, {kw}], expected [_, 3, {patch_size}, {patch_size}]"
                );
            }
            (
                format!("{VT}.patch_embedder.input_proj.weight"),
                t.permute((0, 2, 3, 1))?
                    .contiguous()?
                    .reshape((out_f, 3 * patch_size * patch_size))?,
            )
        } else if name == "v.position_embd.weight" {
            (format!("{VT}.patch_embedder.position_embedding_table"), t)
        } else if name == "mm.input_projection.weight" {
            (
                "model.embed_vision.embedding_projection.weight".to_string(),
                t,
            )
        } else if let Some(rest) = name.strip_prefix("v.blk.") {
            let Some((idx, tail)) = rest.split_once('.') else {
                candle_core::bail!("unrecognized mmproj tensor name: {name}");
            };
            let Some(suffix) = block_suffix(tail) else {
                candle_core::bail!("unrecognized mmproj tensor name: {name}");
            };
            let t = if tail.ends_with("_min") || tail.ends_with("_max") {
                t.reshape(())? // stored [1]; `ClippedLinear` reads a 0-d scalar
            } else {
                t
            };
            (format!("{VT}.encoder.layers.{idx}.{suffix}"), t)
        } else {
            candle_core::bail!("unrecognized mmproj tensor name: {name}");
        };
        out.insert(hf_name, t);
    }
    Ok(out)
}

impl CausalLM for Gemma4Multimodal {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, seq_len) = token_ids.dims2()?;

        // The PLE has two halves and the image positions want a *different*
        // source for each — this is the whole subtlety of the Gemma 4 merge:
        //
        //  - token-identity half (`get_per_layer_inputs`): looks up
        //    `embed_tokens_per_layer` by id. The reference blanks image ids to
        //    `pad_token_id` first (`get_placeholder_mask` → `where(pad)`), so
        //    that lookup uses `masked_ids` below.
        //  - context-projection half (`project_per_layer_inputs`): projects the
        //    *merged* `inputs_embeds` — which already has the vision features
        //    scattered in — so its image rows are `proj(vision_features)`, not
        //    `proj(pad)`.
        //
        // So: inject the vision features into `h` **first**, then compute the
        // PLE against that merged `h` (with `masked_ids` for the lookup half).
        // Computing the PLE before injection fed `proj(scaled pad)` at every
        // image position, and since `per_layer_projection_norm` then floors on
        // `eps` for that near-zero input the result was ~1.0 rel-diff off from
        // the reference at every one of the 42 layers — the drift that garbled
        // the caption.
        let masked_ids = if self.pending_image_embeds.is_some() && seq_len > 1 {
            let ids: Vec<u32> = token_ids.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
            let blanked: Vec<u32> = ids
                .iter()
                .map(|&t| {
                    if t == self.image_token_id {
                        self.pad_token_id
                    } else {
                        t
                    }
                })
                .collect();
            Tensor::from_vec(blanked, (b, seq_len), &self.device)?
        } else {
            token_ids.clone()
        };

        // Text embeddings, scaled by sqrt(hidden_size); image slots start as the
        // (scaled) pad row.
        let mut h = self.text.embed_scaled(&masked_ids)?;

        // Scatter the staged vision features over the image slots, before PLE.
        if seq_len > 1 {
            if let Some(img_feats) = self.pending_image_embeds.take() {
                tracing::debug!(
                    "Gemma4Multimodal: scattering {} vision soft token(s) over id {}",
                    img_feats.dim(0)?,
                    self.image_token_id,
                );
                h = self.inject_image_features(token_ids, h, &img_feats)?;
            }
        }

        // PLE: lookup half off `masked_ids` (pad at image slots), projection
        // half off the merged `h` (vision features at image slots). `None` on
        // a PLE-less text variant, where the blocks take embeddings alone.
        let ple = self.text.compute_ple(&masked_ids, &h)?;

        self.text.forward_embeds(&h, ple.as_ref(), pos, seq_len)
    }

    fn reset(&mut self) {
        // Deliberately does **not** drop `pending_image_embeds`. The provider
        // stages them right before a turn, and the reuse path calls `reset()`
        // between that staging and the prefill `forward` that consumes them
        // (via `.take()`). Clearing here would drop the image on every turn.
        //
        // A set that is never followed by a prefill is not guaranteed to be
        // overwritten — the provider stages nothing on a text turn — but it is
        // harmless: `inject_image_features` scatters over the positions holding
        // `image_token_id`, so a prompt with none leaves `h` untouched, and the
        // next image turn overwrites the slot before its own prefill.
        self.text.reset();
    }

    /// Each text variant's own answer — see [`Gemma4Text::cache`]. This is what
    /// lets the GGUF multimodal path reuse KV across ReAct iterations; the
    /// provider's staged-feature trim (`stage_pending_image_features`) already
    /// accounts for image markers inside a reused prefix.
    fn cache(&mut self) -> Option<&mut ModelCache> {
        self.text.cache()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn accepts_image_features(&self) -> bool {
        true
    }

    /// Forwards to the inherent [`Gemma4Multimodal::encode_image`] — the
    /// fully-qualified path keeps this from resolving back to itself.
    fn encode_image(&self, pixel_values: &Tensor, pixel_position_ids: &Tensor) -> Result<Tensor> {
        Gemma4Multimodal::encode_image(self, pixel_values, pixel_position_ids)
    }

    fn set_image_features(&mut self, features: Tensor) {
        Gemma4Multimodal::set_image_features(self, features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mmproj→HF rename table, pinned name by name. Verified bit-exact
    /// against `unsloth/gemma-4-E4B-it`'s safetensors (every mapped tensor,
    /// max abs diff 0.0) — these tests keep the table from drifting.
    #[test]
    fn mmproj_block_names_map_to_the_hf_module_paths() {
        for (gguf, hf) in [
            ("ln1.weight", "input_layernorm.weight"),
            ("ln2.weight", "pre_feedforward_layernorm.weight"),
            ("attn_post_norm.weight", "post_attention_layernorm.weight"),
            ("ffn_post_norm.weight", "post_feedforward_layernorm.weight"),
            ("attn_q_norm.weight", "self_attn.q_norm.weight"),
            ("attn_k_norm.weight", "self_attn.k_norm.weight"),
            ("attn_q.weight", "self_attn.q_proj.linear.weight"),
            ("attn_out.weight", "self_attn.o_proj.linear.weight"),
            ("ffn_gate.weight", "mlp.gate_proj.linear.weight"),
            ("ffn_down.weight", "mlp.down_proj.linear.weight"),
        ] {
            assert_eq!(block_suffix(gguf).as_deref(), Some(hf), "{gguf}");
        }
    }

    /// The clip bounds sit beside `.linear` on the module, not under it —
    /// `ClippedLinear::load` reads `<module>.{input,output}_{min,max}`.
    #[test]
    fn clip_bounds_map_beside_the_linear_not_under_it() {
        assert_eq!(
            block_suffix("attn_q.input_min").as_deref(),
            Some("self_attn.q_proj.input_min")
        );
        assert_eq!(
            block_suffix("ffn_up.output_max").as_deref(),
            Some("mlp.up_proj.output_max")
        );
    }

    /// A name this loader does not know must map to nothing — the caller turns
    /// that into a hard error, because a tensor it cannot place is a tower it
    /// does not understand.
    #[test]
    fn unknown_mmproj_names_are_refused_not_guessed() {
        assert_eq!(block_suffix("attn_q_rel.weight"), None); // audio-only module
        assert_eq!(block_suffix("conv_dw.weight"), None);
        assert_eq!(block_suffix("noleaf"), None);
    }
}
