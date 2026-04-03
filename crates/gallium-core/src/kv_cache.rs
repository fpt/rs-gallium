use candle_core::{Result, Tensor};

/// Per-layer KV cache that accumulates K and V tensors across generation steps.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
        }
    }

    /// Append new K, V to cache. Returns the full (cached + new) K and V.
    /// K, V shape: (batch, n_kv_heads, seq_len, head_dim)
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_out, v_out) = match (&self.k, &self.v) {
            (Some(ck), Some(cv)) => {
                let k_cat = Tensor::cat(&[ck, k], 2)?;
                let v_cat = Tensor::cat(&[cv, v], 2)?;
                (k_cat, v_cat)
            }
            _ => (k.clone(), v.clone()),
        };
        // Truncate if exceeding max_seq_len
        let seq_len = k_out.dim(2)?;
        let (k_out, v_out) = if seq_len > self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            (
                k_out.narrow(2, start, self.max_seq_len)?,
                v_out.narrow(2, start, self.max_seq_len)?,
            )
        } else {
            (k_out, v_out)
        };
        self.k = Some(k_out.clone());
        self.v = Some(v_out.clone());
        Ok((k_out, v_out))
    }

    /// Read current K and V without modifying cache (for KV-shared layers).
    pub fn current_kv(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Current cached sequence length.
    pub fn len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(2).unwrap_or(0)).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Recurrent state for linear attention layers (e.g., Gated DeltaNet).
pub struct RecurrentState {
    /// Hidden state tensor, shape depends on the specific recurrent mechanism.
    pub state: Option<Tensor>,
    /// Short conv state for causal convolution layers.
    pub conv_state: Option<Tensor>,
}

impl RecurrentState {
    pub fn new() -> Self {
        Self {
            state: None,
            conv_state: None,
        }
    }

    pub fn reset(&mut self) {
        self.state = None;
        self.conv_state = None;
    }
}

impl Default for RecurrentState {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer cache — can be KV (standard attention), recurrent, shared, or TurboQuant-compressed.
#[allow(clippy::large_enum_variant)]
pub enum LayerCache {
    /// Standard KV cache for transformer attention.
    Kv(KvCache),
    /// Shared KV: this layer reuses the KV cache from `source_layer`.
    Shared { source_layer: usize },
    /// Recurrent state for linear attention (DeltaNet, etc.).
    Recurrent(RecurrentState),
    /// TurboQuant-compressed KV cache (5-8x memory reduction).
    TurboKv(crate::turbo_kv_cache::TurboKvCache),
}

impl LayerCache {
    pub fn as_kv(&self) -> Option<&KvCache> {
        match self {
            LayerCache::Kv(kv) => Some(kv),
            _ => None,
        }
    }
}

/// Collection of per-layer caches for a full model.
pub struct ModelCache {
    pub layers: Vec<LayerCache>,
}

impl ModelCache {
    pub fn new(layers: Vec<LayerCache>) -> Self {
        Self { layers }
    }

    /// Get mutable reference to a KV cache. Follows Shared pointers.
    pub fn get_kv(&mut self, layer: usize) -> Option<&mut KvCache> {
        // If this layer is shared, redirect to the source layer.
        let target = match &self.layers[layer] {
            LayerCache::Shared { source_layer } => *source_layer,
            _ => layer,
        };
        match &mut self.layers[target] {
            LayerCache::Kv(kv) => Some(kv),
            _ => None,
        }
    }

    /// Get mutable reference to a recurrent state.
    pub fn get_recurrent(&mut self, layer: usize) -> Option<&mut RecurrentState> {
        match &mut self.layers[layer] {
            LayerCache::Recurrent(state) => Some(state),
            _ => None,
        }
    }

    /// Get mutable references to either KV cache or recurrent state for a layer.
    /// Only one will be Some depending on the layer type.
    pub fn get_layer(
        &mut self,
        layer: usize,
    ) -> (Option<&mut KvCache>, Option<&mut RecurrentState>) {
        let target = match &self.layers[layer] {
            LayerCache::Shared { source_layer } => *source_layer,
            _ => layer,
        };
        match &mut self.layers[target] {
            LayerCache::Kv(kv) => (Some(kv), None),
            LayerCache::Recurrent(state) => (None, Some(state)),
            LayerCache::Shared { .. } => (None, None),
            LayerCache::TurboKv(_) => (None, None), // Use get_turbo_kv() instead
        }
    }

    /// Get mutable reference to a TurboKvCache.
    pub fn get_turbo_kv(&mut self, layer: usize) -> Option<&mut crate::turbo_kv_cache::TurboKvCache> {
        let target = match &self.layers[layer] {
            LayerCache::Shared { source_layer } => *source_layer,
            _ => layer,
        };
        match &mut self.layers[target] {
            LayerCache::TurboKv(tkv) => Some(tkv),
            _ => None,
        }
    }

    /// Reset all caches.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Kv(kv) => kv.reset(),
                LayerCache::Recurrent(state) => state.reset(),
                LayerCache::Shared { .. } => {}
                LayerCache::TurboKv(tkv) => tkv.reset(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KvCache::new(1024);
        let device = Device::Cpu;
        let k1 = Tensor::zeros((1, 4, 3, 64), candle_core::DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 4, 3, 64), candle_core::DType::F32, &device).unwrap();
        let (k, v) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);

        let k2 = Tensor::zeros((1, 4, 1, 64), candle_core::DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 4, 1, 64), candle_core::DType::F32, &device).unwrap();
        let (k, _v) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 4);
    }
}
