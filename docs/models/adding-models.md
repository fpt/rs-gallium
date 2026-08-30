# Adding a New Model

This guide shows how to add a new model architecture to rs-gallium.

## Step 1: Define the Config

Create `crates/gallium-models/src/your_model.rs` and define a config struct that deserializes from the HuggingFace `config.json`:

```rust
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct YourModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    // ... model-specific fields
}
```

## Step 2: Define the Model Struct

Wire together gallium-core building blocks:

```rust
use gallium_core::*;
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_core::{Device, Result, Tensor, DType, Module};

pub struct YourModel {
    embed_tokens: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: Norm,
    lm_head: candle_nn::Linear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
}
```

## Step 3: Implement Loading

The `load` function reads weights via candle's `VarBuilder`. The `vb.pp("prefix")` calls must match the weight names in the safetensors file (same as PyTorch's `state_dict` keys):

```rust
impl YourModel {
    pub fn load(cfg: &YourModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rope = RoPE::new(&RoPEConfig {
            head_dim,
            max_seq_len: cfg.max_position_embeddings,
            theta: cfg.rope_theta,
            ..Default::default()
        }, vb.dtype(), device)?;

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut cache_layers = Vec::new();
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb_l = vb.pp(format!("model.layers.{i}"));
                cache_layers.push(LayerCache::Kv(KvCache::new(cfg.max_position_embeddings)));
                Ok(TransformerBlock {
                    pre_attn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("input_layernorm"))?,
                    attn: AttnImpl::Standard(Attention::new(AttentionConfig {
                        hidden_size: cfg.hidden_size,
                        num_q_heads: cfg.num_attention_heads,
                        num_kv_heads: cfg.num_key_value_heads,
                        head_dim,
                        ..Default::default()
                    }, vb_l.pp("self_attn"))?),
                    post_attn_norm: Norm::rms(cfg.hidden_size, cfg.rms_norm_eps, vb_l.pp("post_attention_layernorm"))?,
                    ffn: FfnImpl::Gated(GatedFFN::new(
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        Activation::Silu,
                        None,
                        vb_l.pp("mlp"),
                    )?),
                    per_layer_embed: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // ... final_norm, lm_head, construct Self
    }
}
```

## Step 4: Implement CausalLM

```rust
impl CausalLM for YourModel {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let mut h = self.embed_tokens.forward(token_ids)?;

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = if seq_len > 1 {
                Some(build_causal_mask(seq_len, pos, &self.device)?)
            } else {
                None
            };
            let kv = self.cache.get_kv(i);
            h = block.forward(&h, &self.rope, pos, kv, None, mask.as_ref())?;
        }

        let h = self.final_norm.forward(&h)?;
        let logits = self.lm_head.forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        logits.to_dtype(DType::F32)
    }

    fn reset(&mut self) { self.cache.reset(); }
    fn device(&self) -> &Device { &self.device }
}
```

## Step 5: Register in lib.rs

Add to `crates/gallium-models/src/lib.rs`:
```rust
pub mod your_model;
```

And add a variant to the CLI's `ModelArch` enum.

## Step 6: Verify Weight Names

Check the model's `model.safetensors.index.json` on HuggingFace to verify your `vb.pp()` paths match. Common patterns:
- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

## Adding Novel Components

If a paper introduces a component that doesn't fit existing building blocks:

1. Create a new file in `crates/gallium-core/src/` (e.g., `diff_attention.rs`)
2. Give it the same function signature pattern (`forward(&self, x, ...) -> Result<Tensor>`)
3. Add a variant to `AttnImpl` or `FfnImpl` as needed
4. Use it in your model file

See `linear_attn.rs` (Gated DeltaNet) as an example of a completely custom attention mechanism.
