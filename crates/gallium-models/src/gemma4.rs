use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use serde::Deserialize;

use gallium_core::*;

// -- Config ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeLayerParams {
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    pub sliding_attention: Option<RopeLayerParams>,
    pub full_attention: Option<RopeLayerParams>,
}

fn default_sliding_window() -> usize { 512 }
fn default_softcap() -> Option<f64> { Some(30.0) }

/// Text config for Gemma 4 (deserialized from config.json's "text_config" key).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_softcap")]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    /// Per-layer input embedding dimension (PLE). Non-zero enables PLE.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

impl Gemma4Config {
    pub fn local_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn global_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.local_head_dim())
    }

    pub fn sliding_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|rp| rp.sliding_attention.as_ref())
            .and_then(|r| r.rope_theta)
            .unwrap_or(10_000.0)
    }

    pub fn global_rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|rp| rp.full_attention.as_ref())
            .and_then(|r| r.rope_theta)
            .unwrap_or(1_000_000.0)
    }

    pub fn global_partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.as_ref()
            .and_then(|rp| rp.full_attention.as_ref())
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(0.25)
    }

    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        if let Some(lt) = &self.layer_types {
            return lt.get(layer_idx).map(|t| t == "full_attention").unwrap_or(false);
        }
        layer_idx == self.num_hidden_layers - 1 || (layer_idx + 1) % 6 == 0
    }
}

// -- Block -------------------------------------------------------------------
//
// Gemma 4 block structure (from modeling_gemma4.py Gemma4TextDecoderLayer):
//   1. residual = h
//      h = input_layernorm(h)
//      h = attn(h)
//      h = post_attention_layernorm(h)
//      h = residual + h
//
//   2. residual = h
//      h = pre_feedforward_layernorm(h)
//      h = mlp(h)
//      h = post_feedforward_layernorm(h)
//      h = residual + h
//
//   3. PLE (if hidden_size_per_layer_input > 0):
//      residual = h
//      gate = gelu(per_layer_input_gate(h))      # [b, s, ple_dim]
//      h = gate * per_layer_input                 # [b, s, ple_dim]
//      h = per_layer_projection(h)               # [b, s, hidden]
//      h = post_per_layer_input_norm(h)
//      h = residual + h
//
//   4. h *= layer_scalar

struct GemmaBlock {
    pre_attn_norm: Norm,
    attn: Attention,
    post_attn_norm: Norm,
    pre_ffn_norm: Norm,
    ffn: GatedFFN,
    post_ffn_norm: Norm,
    // PLE
    per_layer_input_gate: Linear,    // hidden → ple_dim
    per_layer_projection: Linear,    // ple_dim → hidden
    post_ple_norm: Norm,
    layer_scalar: Tensor,
    /// Source layer index when K/V is shared (None = own cache).
    kv_source: Option<usize>,
}

impl GemmaBlock {
    fn load(
        cfg: &Gemma4Config,
        is_global: bool,
        kv_source: Option<usize>,
        ple_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (n_kv, head_dim, shared_kv) = if is_global {
            (
                cfg.num_global_key_value_heads.unwrap_or(cfg.num_key_value_heads),
                cfg.global_head_dim(),
                cfg.attention_k_eq_v,
            )
        } else {
            (cfg.num_key_value_heads, cfg.local_head_dim(), false)
        };

        let attn_cfg = AttentionConfig {
            hidden_size: cfg.hidden_size,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads: n_kv,
            head_dim,
            shared_kv,
            q_norm: true,
            k_norm: true,
            v_norm: true,
            q_norm_eps: cfg.rms_norm_eps,
            scale: Some(1.0),
            ..Default::default()
        };

        let ffn = GatedFFN::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            Activation::GeluTanh,
            None,
            vb.pp("mlp"),
        )?;

        let per_layer_input_gate = linear_no_bias(cfg.hidden_size, ple_dim, vb.pp("per_layer_input_gate"))?;
        let per_layer_projection = linear_no_bias(ple_dim, cfg.hidden_size, vb.pp("per_layer_projection"))?;
        let post_ple_norm = Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_per_layer_input_norm"))?;

        let layer_scalar = vb.get(&[1usize], "layer_scalar")?;

        Ok(Self {
            pre_attn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::new(attn_cfg, vb.pp("self_attn"))?,
            post_attn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            pre_ffn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_feedforward_layernorm"))?,
            ffn,
            post_ffn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_feedforward_layernorm"))?,
            per_layer_input_gate,
            per_layer_projection,
            post_ple_norm,
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
        ple_input: &Tensor,  // [b, s, ple_dim]
    ) -> Result<Tensor> {
        // Attention branch
        let h_pre = self.pre_attn_norm.forward(x)?;
        let h_attn = if let Some(src) = self.kv_source {
            let src_kv = cache.layers[src].as_kv()
                .ok_or_else(|| candle_core::Error::Msg("shared source KV cache is empty".into()))?;
            self.attn.forward_shared(&h_pre, rope, pos, src_kv, mask)?
        } else {
            let kv = cache.get_kv(layer_idx).expect("own layer has KV cache");
            self.attn.forward(&h_pre, rope, pos, kv, mask)?
        };
        let h = self.post_attn_norm.forward(&h_attn)?;
        let x = (x + h)?;

        // FFN branch
        let h = self.pre_ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (x + h)?;

        // PLE branch: gate = gelu(per_layer_input_gate(x)); h = per_layer_projection(gate * ple_input)
        let gate = self.per_layer_input_gate.forward(&x)?.gelu()?;
        let gated = (gate * ple_input)?;
        let h = self.per_layer_projection.forward(&gated)?;
        let h = self.post_ple_norm.forward(&h)?;
        let x = (x + h)?;

        // layer_scalar
        x.broadcast_mul(&self.layer_scalar.to_dtype(x.dtype())?)
    }
}

// -- Model -------------------------------------------------------------------

pub struct Gemma4 {
    embed_tokens: Embedding,
    // PLE global components
    embed_tokens_per_layer: Embedding,
    per_layer_model_projection: Linear,
    per_layer_projection_norm: Norm,
    blocks: Vec<GemmaBlock>,
    final_norm: Norm,
    lm_head: candle_nn::Linear,
    rope_sliding: RoPE,
    rope_global: RoPE,
    cache: ModelCache,
    device: Device,
    hidden_size: usize,
    is_global: Vec<bool>,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    n_layers: usize,
    ple_dim: usize,
}

impl Gemma4 {
    pub fn load(cfg: &Gemma4Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let vb_lm = vb.pp("model.language_model");

        let rope_sliding = RoPE::new(
            &RoPEConfig {
                head_dim: cfg.local_head_dim(),
                max_seq_len: cfg.max_position_embeddings,
                theta: cfg.sliding_rope_theta(),
                ..Default::default()
            },
            vb.dtype(),
            device,
        )?;

        let rope_global = RoPE::new(
            &RoPEConfig {
                head_dim: cfg.global_head_dim(),
                max_seq_len: cfg.max_position_embeddings,
                theta: cfg.global_rope_theta(),
                partial_rotary_factor: cfg.global_partial_rotary_factor(),
                ..Default::default()
            },
            vb.dtype(),
            device,
        )?;

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(0);
        let ple_vocab = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);

        let embed_tokens_per_layer = embedding(
            ple_vocab,
            cfg.num_hidden_layers * ple_dim,
            vb_lm.pp("embed_tokens_per_layer"),
        )?;
        let per_layer_model_projection = linear_no_bias(
            cfg.hidden_size,
            cfg.num_hidden_layers * ple_dim,
            vb_lm.pp("per_layer_model_projection"),
        )?;
        let per_layer_projection_norm = Norm::rms(ple_dim, cfg.rms_norm_eps, vb_lm.pp("per_layer_projection_norm"))?;

        let mut is_global_vec = Vec::new();
        let mut cache_layers = Vec::new();
        let num_owned = cfg.num_hidden_layers - cfg.num_kv_shared_layers;

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb_l = vb_lm.pp(format!("layers.{i}"));
                let is_global = cfg.is_global_layer(i);
                is_global_vec.push(is_global);

                let kv_source = if i >= num_owned && cfg.num_kv_shared_layers > 0 {
                    // All shared layers of the same type map to the LAST owned layer of that type.
                    let source = (0..num_owned)
                        .filter(|&j| cfg.is_global_layer(j) == is_global)
                        .last()
                        .unwrap_or(0);
                    cache_layers.push(LayerCache::Shared { source_layer: source });
                    Some(source)
                } else {
                    cache_layers.push(LayerCache::Kv(KvCache::new(cfg.max_position_embeddings)));
                    None
                };

                GemmaBlock::load(cfg, is_global, kv_source, ple_dim, vb_l)
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb_lm.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            let w = vb_lm.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            candle_nn::Linear::new(w, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_lm.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            blocks,
            final_norm,
            lm_head,
            rope_sliding,
            rope_global,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            hidden_size: cfg.hidden_size,
            is_global: is_global_vec,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
            n_layers: cfg.num_hidden_layers,
            ple_dim,
        })
    }

    /// Compute per-layer inputs: [b, s, n_layers, ple_dim].
    ///
    /// Combines per-token per-layer embeddings with a projection of the main embeddings.
    /// Matches `get_per_layer_inputs` + `project_per_layer_inputs` in the reference.
    fn compute_ple(
        &self,
        token_ids: &Tensor,
        h_embed: &Tensor,  // scaled main embeddings [b, s, hidden]
    ) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        let n = self.n_layers;
        let d = self.ple_dim;

        // Per-token per-layer embedding (with scale sqrt(ple_dim))
        let ple_scale = (d as f64).sqrt();
        let ple_embed = (self.embed_tokens_per_layer.forward(token_ids)? * ple_scale)?;
        // [b, s, n*d] → [b, s, n, d]
        let ple_embed = ple_embed.reshape((b, s, n, d))?;

        // Projection of main embeddings scaled by 1/sqrt(hidden_size)
        // (the sqrt(hidden) scale applied to h_embed cancels with this factor)
        let proj_scale = 1.0 / (self.hidden_size as f64).sqrt();
        let ple_proj = (self.per_layer_model_projection.forward(h_embed)? * proj_scale)?;
        let ple_proj = ple_proj.reshape((b, s, n, d))?;
        let ple_proj = self.per_layer_projection_norm.forward(&ple_proj)?;

        // Combine: (embed + proj) * 2^-0.5
        ((ple_embed + ple_proj)? * 2.0_f64.powf(-0.5))
    }
}

impl CausalLM for Gemma4 {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;

        // Gemma 4: scale embeddings by sqrt(hidden_size)
        let scale = (self.hidden_size as f64).sqrt();
        let h_embed = (self.embed_tokens.forward(token_ids)? * scale)?;

        // Per-layer inputs [b, s, n_layers, ple_dim]
        let per_layer_inputs = self.compute_ple(token_ids, &h_embed)?;

        let mut h = h_embed;

        for (i, block) in self.blocks.iter().enumerate() {
            let is_global = self.is_global[i];
            let rope = if is_global { &self.rope_global } else { &self.rope_sliding };

            let mask = if seq_len <= 1 {
                None
            } else if is_global {
                Some(build_causal_mask(seq_len, pos, &self.device)?)
            } else {
                Some(build_sliding_window_mask(seq_len, pos, self.sliding_window, &self.device)?)
            };

            // Extract PLE for this layer: [b, s, ple_dim]
            let ple_i = per_layer_inputs.narrow(2, i, 1)?.squeeze(2)?;

            h = block.forward(&h, rope, pos, &mut self.cache, i, mask.as_ref(), &ple_i)?;

            if std::env::var("GALLIUM_DEBUG").is_ok() && seq_len > 1 {
                let rms = h.to_dtype(candle_core::DType::F32)?
                    .sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                eprintln!("layer {i:2} ({}) rms={:.4}", if self.is_global[i] { "global" } else { "slide " }, rms);
            }
        }

        let h = self.final_norm.forward(&h)?;
        let mut logits = self.lm_head.forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;

        if let Some(cap) = self.final_logit_softcapping {
            logits = ((logits * (1.0 / cap))?.tanh()? * cap)?;
        }

        let logits = logits.to_dtype(DType::F32)?;
        if std::env::var("GALLIUM_DEBUG").is_ok() && seq_len > 1 {
            let lv: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            let mut top: Vec<(usize, f32)> = lv.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("top-10 logits: {:?}", &top[..10]);
        }
        Ok(logits)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
