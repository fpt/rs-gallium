use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder};
use serde::Deserialize;

use gallium_core::*;

// -- Config ------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Qwen35LayerType {
    FullAttention,
    LinearAttention,
}

/// RoPE parameters may be stored as a nested dict in the config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35RopeParameters {
    pub rope_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    /// Full-attention query heads.
    pub num_attention_heads: usize,
    /// Full-attention KV heads (GQA).
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    /// Legacy flat rope_theta; use rope_parameters if present.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Nested rope parameters dict (Qwen3.5 style).
    #[serde(default)]
    pub rope_parameters: Option<Qwen35RopeParameters>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Optional explicit layer type list. If absent, defaults to every 4th layer
    /// being full attention (indices 3,7,11,...).
    #[serde(default)]
    pub layer_types: Option<Vec<Qwen35LayerType>>,
    /// Partial rotary factor (legacy flat field).
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    // Linear attention config
    #[serde(default = "default_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_32")]
    pub linear_num_value_heads: usize,
    /// Qwen3.5: Q projection is 2× for output gating (from config as attn_output_gate).
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    // MoE (optional)
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
}

fn default_partial_rotary() -> f64 { 0.25 }
fn default_key_head_dim() -> usize { 128 }
fn default_value_head_dim() -> usize { 128 }
fn default_conv_kernel() -> usize { 4 }
fn default_16() -> usize { 16 }
fn default_32() -> usize { 32 }
fn default_true() -> bool { true }

impl Qwen35Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        if let Some(rp) = &self.rope_parameters {
            if let Some(t) = rp.rope_theta { return t; }
        }
        self.rope_theta.unwrap_or(10_000.0)
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        if let Some(rp) = &self.rope_parameters {
            if let Some(f) = rp.partial_rotary_factor { return f; }
        }
        self.partial_rotary_factor
    }

    pub fn layer_types(&self) -> Vec<Qwen35LayerType> {
        if let Some(lt) = &self.layer_types {
            return lt.clone();
        }
        // Default: every 4th layer (1-indexed) is full attention
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    Qwen35LayerType::FullAttention
                } else {
                    Qwen35LayerType::LinearAttention
                }
            })
            .collect()
    }
}

// -- Model -------------------------------------------------------------------

pub struct Qwen35 {
    embed_tokens: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: Norm,
    lm_head: candle_nn::Linear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
    layer_types: Vec<Qwen35LayerType>,
}

impl Qwen35 {
    pub fn load(cfg: &Qwen35Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let layer_types = cfg.layer_types();

        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len: cfg.max_position_embeddings,
                theta: cfg.rope_theta(),
                partial_rotary_factor: cfg.partial_rotary_factor(),
                ..Default::default()
            },
            vb.dtype(),
            device,
        )?;

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.language_model.embed_tokens"))?;

        // Full attention config: Qwen3.5 uses Q output gate and (1+w) norms
        let attn_cfg = AttentionConfig {
            hidden_size: cfg.hidden_size,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            q_norm: true,
            k_norm: true,
            norm_one_plus: true,    // q_norm/k_norm weights are zeros-init + (1+w)
            q_output_gate: cfg.attn_output_gate,
            q_norm_eps: cfg.rms_norm_eps,
            ..Default::default()
        };

        let delta_cfg = DeltaNetConfig {
            hidden_size: cfg.hidden_size,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            key_head_dim: cfg.linear_key_head_dim,
            value_head_dim: cfg.linear_value_head_dim,
            conv_kernel_dim: cfg.linear_conv_kernel_dim,
            rms_eps: cfg.rms_norm_eps,
        };

        let mut cache_layers = Vec::new();
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb_l = vb.pp(format!("model.language_model.layers.{i}"));
                let (attn_impl, cache) = match layer_types[i] {
                    Qwen35LayerType::FullAttention => (
                        AttnImpl::Standard(Attention::new(attn_cfg.clone(), vb_l.pp("self_attn"))?),
                        LayerCache::Kv(KvCache::new(cfg.max_position_embeddings)),
                    ),
                    Qwen35LayerType::LinearAttention => (
                        AttnImpl::LinearDeltaNet(
                            GatedDeltaNet::new(delta_cfg.clone(), vb_l.pp("linear_attn"))?
                        ),
                        LayerCache::Recurrent(RecurrentState::new()),
                    ),
                };
                cache_layers.push(cache);

                let ffn = match (cfg.num_local_experts, cfg.moe_intermediate_size) {
                    (Some(n_experts), Some(moe_inter)) if n_experts > 0 => {
                        FfnImpl::MoE(MoEFFN::new(
                            cfg.hidden_size,
                            moe_inter,
                            n_experts,
                            cfg.num_experts_per_tok.unwrap_or(8),
                            Activation::Silu,
                            None,
                            cfg.shared_expert_intermediate_size,
                            vb_l.pp("mlp"),
                        )?)
                    }
                    _ => FfnImpl::Gated(GatedFFN::new(
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        Activation::Silu,
                        None,
                        vb_l.pp("mlp"),
                    )?),
                };

                // Qwen3.5 norms: weight stored as delta-from-zero, scale = (1+w)
                Ok(TransformerBlock {
                    pre_attn_norm: Norm::rms_one_plus(
                        cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("input_layernorm"),
                    )?,
                    attn: attn_impl,
                    post_attn_norm: Norm::rms_one_plus(
                        cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("post_attention_layernorm"),
                    )?,
                    ffn,
                    per_layer_embed: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = Norm::rms_one_plus(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.language_model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            let w = vb.pp("model.language_model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            candle_nn::Linear::new(w, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            blocks,
            final_norm,
            lm_head,
            rope,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            layer_types,
        })
    }
}

impl CausalLM for Qwen35 {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed_tokens.forward(token_ids)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = match self.layer_types[i] {
                Qwen35LayerType::FullAttention if seq_len > 1 => {
                    Some(build_causal_mask(seq_len, pos, &self.device)?)
                }
                _ => None,
            };
            let (kv, recurrent) = self.cache.get_layer(i);
            h = block.forward(&h, &self.rope, pos, kv, recurrent, mask.as_ref())?;
        }

        let h = self.final_norm.forward(&h)?;
        let logits = self.lm_head.forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        logits.to_dtype(DType::F32)
    }

    fn reset(&mut self) { self.cache.reset(); }
    fn device(&self) -> &Device { &self.device }
}
