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
//! ```
//! // Preprocess image externally and call:
//! let feats = model.encode_image(&pixel_values, &pixel_position_ids)?;
//! model.set_image_features(feats);
//! // Then generate as usual via CausalLM::forward.
//! ```
//!
//! `pixel_values` shape: `[batch, num_patches, 3 * patch_size^2]` (f32, values in `[0, 1]`).
//! `pixel_position_ids` shape: `[batch, num_patches, 2]` (i64, (x, y); padding patches = (-1,-1)).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use serde::Deserialize;

use gallium_core::*;
use crate::gemma4::{Gemma4, Gemma4Config};

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,           // 768
    pub intermediate_size: usize,     // 3072
    pub num_hidden_layers: usize,     // 16
    pub num_attention_heads: usize,   // 12
    #[serde(default)]
    pub head_dim: Option<usize>,      // 64 (defaults to hidden/heads)
    pub rms_norm_eps: f64,            // 1e-6
    pub patch_size: usize,            // 16
    #[serde(default = "default_pooling")]
    pub pooling_kernel_size: usize,   // 3
    #[serde(default = "default_pos_size")]
    pub position_embedding_size: usize, // 10240
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>, // extract rope_theta from here
}

fn default_pooling() -> usize { 3 }
fn default_pos_size() -> usize { 10240 }

impl Gemma4VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
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

fn default_image_token_id() -> u32 { 258_880 }

// ── Helpers ──────────────────────────────────────────────────────────────────

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
        let n = spatial_dim / 2;        // 16
        let inv: Vec<f32> = (0..n)
            .map(|j| (1.0 / theta.powf(2.0 * j as f64 / spatial_dim as f64)) as f32)
            .collect();
        Ok(Self { inv_freq: Tensor::from_vec(inv, n, device)? })
    }

    /// Returns `(cos, sin)` each of shape `[b, num_patches, head_dim]`.
    fn compute(&self, pixel_position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // pixel_position_ids: [b, s, 2] i64 (padding = -1)
        let (b, s, _) = pixel_position_ids.dims3()?;

        // Clamp padding positions to 0 (padding patches are masked out via padding_positions).
        let pos = pixel_position_ids.to_dtype(DType::F32)?
            .clamp(0f32, f32::INFINITY)?; // [b, s, 2]

        let pos_x = pos.narrow(2, 0, 1)?.squeeze(2)?.reshape((b * s, 1))?; // [b*s, 1]
        let pos_y = pos.narrow(2, 1, 1)?.squeeze(2)?.reshape((b * s, 1))?;

        let inv = self.inv_freq.reshape((1, self.inv_freq.dim(0)?))?; // [1, n]

        // freqs: [b*s, n]
        let freqs_x = pos_x.broadcast_mul(&inv)?.reshape((b, s, self.inv_freq.dim(0)?))?;
        let freqs_y = pos_y.broadcast_mul(&inv)?.reshape((b, s, self.inv_freq.dim(0)?))?;

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
    input_proj: Linear,   // 3*patch_size^2 → hidden
    pos_table: Tensor,    // [2, position_embedding_size, hidden]
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
        Ok(Self { input_proj, pos_table, hidden_size: cfg.hidden_size })
    }

    fn forward(
        &self,
        pixel_values: &Tensor,        // [b, s, 3*patch_size^2]
        pixel_position_ids: &Tensor,  // [b, s, 2] i64
        padding_positions: &Tensor,   // [b, s] bool-like (1 = padding)
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

        let cx = clamped.narrow(2, 0, 1)?.squeeze(2)?.reshape((b * s,))?.to_dtype(DType::U32)?;
        let cy = clamped.narrow(2, 1, 1)?.squeeze(2)?.reshape((b * s,))?.to_dtype(DType::U32)?;

        let table_x = self.pos_table.narrow(0, 0, 1)?.squeeze(0)?; // [pos_size, hidden]
        let table_y = self.pos_table.narrow(0, 1, 1)?.squeeze(0)?;

        // Lookup and sum x + y contributions.
        let emb = (table_x.index_select(&cx, 0)? + table_y.index_select(&cy, 0)?)?
            .reshape((b, s, self.hidden_size))?;

        // Zero out padding patches by multiplying by (1 - padding_mask).
        let not_pad = (padding_positions.to_dtype(emb.dtype())? - 1.0)?
            .neg()?
            .unsqueeze(2)?; // [b, s, 1]
        emb.broadcast_mul(&not_pad)
    }
}

// ── Vision Attention ─────────────────────────────────────────────────────────

struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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
        // Weights are nested under ".linear." because use_clipped_linears=True in the reference.
        let q_proj = linear_no_bias(h, cfg.num_attention_heads * head_dim, vb.pp("q_proj.linear"))?;
        let k_proj = linear_no_bias(h, cfg.num_attention_heads * head_dim, vb.pp("k_proj.linear"))?;
        let v_proj = linear_no_bias(h, cfg.num_attention_heads * head_dim, vb.pp("v_proj.linear"))?;
        let o_proj = linear_no_bias(cfg.num_attention_heads * head_dim, h, vb.pp("o_proj.linear"))?;
        let q_norm = Norm::rms(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Norm::rms(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                  num_heads: cfg.num_attention_heads, head_dim, norm_eps: cfg.rms_norm_eps })
    }

    fn forward(
        &self,
        x: &Tensor,          // [b, s, hidden]
        cos: &Tensor,        // [b, s, head_dim]
        sin: &Tensor,        // [b, s, head_dim]
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;

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
        let attn_w = candle_nn::ops::softmax(&attn_w.to_dtype(DType::F32)?, 3)?
            .to_dtype(q.dtype())?;
        let out = attn_w.matmul(&v)?; // [b, heads, s, head_dim]

        let out = out.transpose(1, 2)?.contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

// ── Vision MLP ───────────────────────────────────────────────────────────────

struct VisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMLP {
    fn load(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        // Weights nested under ".linear." (use_clipped_linears=True).
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj.linear"))?,
            up_proj:   linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj.linear"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj.linear"))?,
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
            input_norm:    Norm::rms(h, eps, vb.pp("input_layernorm"))?,
            attn:          VisionAttention::load(cfg, vb.pp("self_attn"))?,
            post_attn_norm: Norm::rms(h, eps, vb.pp("post_attention_layernorm"))?,
            pre_ffn_norm:  Norm::rms(h, eps, vb.pp("pre_feedforward_layernorm"))?,
            mlp:           VisionMLP::load(cfg, vb.pp("mlp"))?,
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
        Self { scale: (hidden_size as f64).sqrt() }
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
            return (hidden * self.scale);
        }

        // CPU-based scatter-average (runs once during prefill, performance non-critical).
        let h_f32 = hidden.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let pos = pixel_position_ids.to_vec3::<i64>()?;
        let k2 = k * k;

        let mut result = vec![0.0f32; b * output_len * d];

        for bi in 0..b {
            let max_x = pos[bi].iter().map(|p| p[0].max(0)).max().unwrap_or(0) + 1;
            let num_cols = (max_x as usize + k - 1) / k;

            for si in 0..s {
                let px = pos[bi][si][0];
                let py = pos[bi][si][1];
                if px < 0 || py < 0 { continue; } // padding

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

        Tensor::from_vec(result, (b, output_len, d), hidden.device())?
            .to_dtype(hidden.dtype())
    }
}

// ── Vision Projector ──────────────────────────────────────────────────────────

struct VisionProjector {
    pre_norm_eps: f64,
    proj: Linear, // vision_hidden → text_hidden
}

impl VisionProjector {
    fn load(vision_hidden: usize, text_hidden: usize, norm_eps: f64, vb: VarBuilder) -> Result<Self> {
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

// ── Gemma4Multimodal ──────────────────────────────────────────────────────────

pub struct Gemma4Multimodal {
    text: Gemma4,
    patch_embedder: VisionPatchEmbedder,
    encoder: VisionEncoder,
    pooler: VisionPooler,
    projector: VisionProjector,
    image_token_id: u32,
    pooling_kernel_size: usize,
    pending_image_embeds: Option<Tensor>,
    device: Device,
}

impl Gemma4Multimodal {
    pub fn load(cfg: &Gemma4MultimodalConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        // Text model: weights live under model.language_model.* in the multimodal safetensors.
        let text = Gemma4::load(&cfg.text_config, vb.clone(), device)?;

        let vc = &cfg.vision_config;
        let vb_vt = vb.pp("model.vision_tower");
        let patch_embedder = VisionPatchEmbedder::load(vc, vb_vt.pp("patch_embedder"))?;
        let encoder = VisionEncoder::load(vc, vb_vt.pp("encoder"), device)?;
        let pooler = VisionPooler::new(vc.hidden_size);
        let projector = VisionProjector::load(
            vc.hidden_size,
            cfg.text_config.hidden_size,
            vc.rms_norm_eps,
            vb.pp("model.embed_vision"),
        )?;

        Ok(Self {
            text,
            patch_embedder,
            encoder,
            pooler,
            projector,
            image_token_id: cfg.image_token_id,
            pooling_kernel_size: vc.pooling_kernel_size,
            pending_image_embeds: None,
            device: device.clone(),
        })
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
        let padding_positions = px.lt(0i64)?.to_dtype(DType::F32)?;  // 1.0 = padding

        let embeds = self.patch_embedder.forward(
            pixel_values,
            pixel_position_ids,
            &padding_positions,
        )?;

        let encoded = self.encoder.forward(&embeds, pixel_position_ids)?;

        let num_patches = pixel_values.dim(1)?;
        let output_len = num_patches / (self.pooling_kernel_size * self.pooling_kernel_size);

        let pooled = self.pooler.pool(&encoded, pixel_position_ids, &padding_positions, output_len)?;

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
                    if img_idx >= img_vec.len() { break 'outer; }
                    h_vec[bi][si] = img_vec[img_idx].clone();
                    img_idx += 1;
                }
            }
        }

        Tensor::new(h_vec, &self.device)?.to_dtype(h_embed.dtype())
    }
}

impl CausalLM for Gemma4Multimodal {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;

        // Text embeddings (scaled by sqrt(hidden_size)).
        let h = self.text.embed_scaled(token_ids)?;

        // PLE uses the original token_ids (image positions treated as regular tokens for PLE,
        // which is a small approximation vs. the reference that replaces them with PAD first).
        let ple = self.text.compute_ple(token_ids, &h)?;

        // On prefill, inject pending image features if available.
        let h = if seq_len > 1 {
            if let Some(img_feats) = self.pending_image_embeds.take() {
                self.inject_image_features(token_ids, h, &img_feats)?
            } else {
                h
            }
        } else {
            h
        };

        self.text.forward_embeds(&h, &ple, pos, seq_len)
    }

    fn reset(&mut self) {
        self.pending_image_embeds = None;
        self.text.reset();
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
