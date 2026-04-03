use candle_core::{Result, Tensor};

use crate::attention::Attention;
use crate::ffn::FfnImpl;
use crate::kv_cache::{KvCache, RecurrentState};
use crate::linear_attn::GatedDeltaNet;
use crate::norm::Norm;
use crate::pos_enc::RoPE;

/// Attention implementation variant.
pub enum AttnImpl {
    /// Standard multi-head / grouped-query / multi-query attention.
    Standard(Attention),
    /// Gated DeltaNet linear attention (Qwen 3.5).
    LinearDeltaNet(GatedDeltaNet),
}

/// A single transformer block: pre-norm -> attn -> residual -> post-norm -> ffn -> residual.
pub struct TransformerBlock {
    pub pre_attn_norm: Norm,
    pub attn: AttnImpl,
    pub post_attn_norm: Norm,
    pub ffn: FfnImpl,
    /// Per-layer embedding added to hidden states (Gemma 4 PLE).
    pub per_layer_embed: Option<Tensor>,
}

impl TransformerBlock {
    /// Forward pass.
    ///
    /// - `x`: (batch, seq_len, hidden_size)
    /// - `rope`: rotary embeddings for this layer (may differ per layer)
    /// - `pos`: position offset for KV cache
    /// - `kv_cache`: mutable KV cache for standard attention layers
    /// - `recurrent_state`: mutable recurrent state for linear attention layers
    /// - `mask`: attention mask (causal or sliding window)
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: Option<&mut KvCache>,
        recurrent_state: Option<&mut RecurrentState>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Optional per-layer embedding
        let x = if let Some(ref ple) = self.per_layer_embed {
            x.broadcast_add(ple)?
        } else {
            x.clone()
        };

        // Pre-attention norm
        let h = self.pre_attn_norm.forward(&x)?;

        // Attention
        let h = match &self.attn {
            AttnImpl::Standard(attn) => {
                let kv = kv_cache.expect("standard attention requires KV cache");
                attn.forward(&h, rope, pos, kv, mask)?
            }
            AttnImpl::LinearDeltaNet(delta) => {
                let state = recurrent_state.expect("DeltaNet requires recurrent state");
                delta.forward(&h, state)?
            }
        };

        // Residual
        let h = (h + &x)?;

        // Post-attention norm
        let residual = &h;
        let h = self.post_attn_norm.forward(&h)?;

        // FFN
        let h = self.ffn.forward(&h)?;

        // Residual
        h + residual
    }
}
