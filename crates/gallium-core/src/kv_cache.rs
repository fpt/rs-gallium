use candle_core::{Result, Tensor};

/// Per-layer KV cache that accumulates K and V tensors across generation steps.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    max_seq_len: usize,
    /// Whether this cache has ever dropped positions off the front to stay
    /// within `max_seq_len`.
    ///
    /// It gates [`Self::truncate`], because once the front is gone the tensors
    /// are no longer positions `0..len` and a rollback addressed by position
    /// would silently land somewhere else. In practice models here build this
    /// with the whole context length and apply a sliding window through the
    /// *mask* instead, so this stays false — which is the point: the one case
    /// that would break reuse is the one that says so.
    evicted: bool,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            max_seq_len,
            evicted: false,
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
            self.evicted = true;
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

    /// Whether this cache still holds positions `0..len()` — false once it has
    /// dropped anything off the front, after which [`Self::truncate`] is not a
    /// positional rollback any more.
    pub fn holds_prefix(&self) -> bool {
        !self.evicted
    }

    /// Drop everything past `len`, keeping positions `0..len`.
    ///
    /// This is the rollback an attention cache can do and a recurrent state
    /// cannot: K and V are a per-position log, so a prefix of them is a valid
    /// cache for that prefix. It is what lets iteration *N+1* of an agent turn
    /// evaluate only what iteration *N*'s prompt did not already contain.
    ///
    /// Made contiguous rather than left as a view: `narrow` on the sequence
    /// dimension gives a strided view, and the next `append`'s `cat` would carry
    /// that stride into everything after it. The copy happens only on a
    /// rollback, which is once per turn against a forward pass per token.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len == 0 {
            self.reset();
            return Ok(());
        }
        if len >= self.len() {
            return Ok(());
        }
        if let (Some(k), Some(v)) = (&self.k, &self.v) {
            self.k = Some(k.narrow(2, 0, len)?.contiguous()?);
            self.v = Some(v.narrow(2, 0, len)?.contiguous()?);
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.evicted = false;
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

    /// A copy of this state, to put back later.
    ///
    /// Cheap, and that is a property of how these tensors are used rather than a
    /// hope: candle tensors are reference-counted, and a step *replaces* the
    /// state with a new tensor instead of writing through the old one, so a
    /// snapshot shares storage until the next step and copies nothing.
    ///
    /// This is what a recurrent layer has instead of [`KvCache::truncate`]. Its
    /// state is not a per-position log — it is one rolling summary — so the only
    /// way back to an earlier position is to have kept it.
    pub fn snapshot(&self) -> Self {
        Self {
            state: self.state.clone(),
            conv_state: self.conv_state.clone(),
        }
    }

    pub fn restore(&mut self, from: &Self) {
        self.state = from.state.clone();
        self.conv_state = from.conv_state.clone();
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

/// The part of a [`ModelCache`] that a positional rollback cannot reproduce,
/// captured at a known token count.
///
/// Only the recurrent layers are in here. An attention layer needs nothing kept:
/// [`KvCache::truncate`] rolls it back to any earlier position from what it
/// already holds. So a hybrid model's rewind costs one clone per recurrent layer
/// and nothing per attention layer — the split llama.cpp does not expose, where
/// `llama_memory_hybrid::seq_rm` tries the recurrent half first and refuses the
/// whole operation without touching the attention half.
pub struct CacheCheckpoint {
    /// How many tokens the cache held when this was taken. A checkpoint is only
    /// usable to rewind to exactly this length: the recurrent state it holds is
    /// the summary of precisely these tokens.
    len: usize,
    /// `(layer index, state)` for every recurrent layer.
    recurrent: Vec<(usize, RecurrentState)>,
}

impl CacheCheckpoint {
    /// The token count this checkpoint restores to.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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

    /// How many tokens this cache holds, read from the attention layers.
    ///
    /// A recurrent layer cannot answer — its state is a summary with no length —
    /// so a model built entirely from them reports 0 and simply never reuses.
    pub fn len(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| match l {
                LayerCache::Kv(kv) => Some(kv.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a rewind needs a [`CacheCheckpoint`] — true when any layer holds
    /// state that cannot be rolled back by position.
    pub fn needs_checkpoint(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(l, LayerCache::Recurrent(_)))
    }

    /// Capture what a rewind to the current length would need.
    pub fn checkpoint(&self) -> CacheCheckpoint {
        CacheCheckpoint {
            len: self.len(),
            recurrent: self
                .layers
                .iter()
                .enumerate()
                .filter_map(|(i, l)| match l {
                    LayerCache::Recurrent(state) => Some((i, state.snapshot())),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Roll every layer back to `len`, using `checkpoint` for the layers that a
    /// position cannot address.
    ///
    /// `Ok(false)` means *not done and nothing changed*: feasibility is decided
    /// for the whole cache before a single layer is touched, because a rewind
    /// that gave up half way would leave the layers describing different
    /// prefixes of the conversation — a cache that produces plausible logits for
    /// a state no conversation was ever in.
    ///
    /// It is refused when a recurrent layer is present without a checkpoint at
    /// exactly `len`, and for a TurboQuant cache, which has no positional
    /// rollback at all. `len == 0` is always possible: that is a reset.
    pub fn rewind(&mut self, len: usize, checkpoint: Option<&CacheCheckpoint>) -> Result<bool> {
        if len == 0 {
            self.reset();
            return Ok(true);
        }
        let usable = checkpoint.filter(|c| c.len == len);
        let feasible = self.layers.iter().all(|l| match l {
            LayerCache::Kv(kv) => kv.holds_prefix() && kv.len() >= len,
            LayerCache::Shared { .. } => true,
            LayerCache::Recurrent(_) => usable.is_some(),
            LayerCache::TurboKv(_) => false,
        });
        if !feasible {
            return Ok(false);
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                LayerCache::Kv(kv) => kv.truncate(len)?,
                LayerCache::Recurrent(state) => {
                    if let Some((_, saved)) = usable
                        .expect("feasibility checked above")
                        .recurrent
                        .iter()
                        .find(|(idx, _)| *idx == i)
                    {
                        state.restore(saved);
                    }
                }
                LayerCache::Shared { .. } | LayerCache::TurboKv(_) => {}
            }
        }
        Ok(true)
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
    pub fn get_turbo_kv(
        &mut self,
        layer: usize,
    ) -> Option<&mut crate::turbo_kv_cache::TurboKvCache> {
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
        let (k, _v) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k.dim(2).unwrap(), 3);

        let k2 = Tensor::zeros((1, 4, 1, 64), candle_core::DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 4, 1, 64), candle_core::DType::F32, &device).unwrap();
        let (k, _v) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k.dim(2).unwrap(), 4);
    }
}

#[cfg(test)]
mod rewind_tests {
    use super::*;
    use candle_core::{DType, Device};

    /// A cache holding `len` positions, shaped the way a layer appends them.
    fn kv(len: usize) -> KvCache {
        let mut cache = KvCache::new(4096);
        if len > 0 {
            let t = Tensor::zeros((1, 2, len, 8), DType::F32, &Device::Cpu).unwrap();
            cache.append(&t, &t).unwrap();
        }
        cache
    }

    fn recurrent(marker: f32) -> RecurrentState {
        RecurrentState {
            state: Some(Tensor::full(marker, (1, 4), &Device::Cpu).unwrap()),
            conv_state: None,
        }
    }

    fn marker_of(state: &RecurrentState) -> f32 {
        state
            .state
            .as_ref()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    }

    #[test]
    fn an_attention_only_cache_rolls_back_from_what_it_holds() {
        let mut cache = ModelCache::new(vec![LayerCache::Kv(kv(10)), LayerCache::Kv(kv(10))]);
        assert_eq!(cache.len(), 10);
        assert!(
            !cache.needs_checkpoint(),
            "no checkpoint needed without a recurrent layer"
        );
        assert!(cache.rewind(4, None).unwrap());
        assert_eq!(cache.len(), 4);
    }

    /// The refusal, and the half of it that matters: a rewind that cannot finish
    /// must not start. Layers left describing different prefixes would produce
    /// plausible logits for a state no conversation was ever in.
    #[test]
    fn a_recurrent_layer_without_a_checkpoint_refuses_and_changes_nothing() {
        let mut cache = ModelCache::new(vec![
            LayerCache::Kv(kv(10)),
            LayerCache::Recurrent(recurrent(1.0)),
        ]);
        assert!(cache.needs_checkpoint());
        assert!(!cache.rewind(4, None).unwrap());
        assert_eq!(cache.len(), 10, "the attention layer was trimmed anyway");
    }

    #[test]
    fn a_checkpoint_at_the_wrong_length_is_refused_too() {
        let mut cache = ModelCache::new(vec![
            LayerCache::Kv(kv(10)),
            LayerCache::Recurrent(recurrent(1.0)),
        ]);
        let stale = cache.checkpoint(); // taken at 10
        assert!(!cache.rewind(4, Some(&stale)).unwrap());
        assert_eq!(cache.len(), 10);
    }

    /// The hybrid rewind llama.cpp will not do: the attention half by position,
    /// the recurrent half from a snapshot, in one operation.
    #[test]
    fn a_hybrid_cache_rewinds_both_halves_together() {
        let mut cache = ModelCache::new(vec![
            LayerCache::Kv(kv(4)),
            LayerCache::Recurrent(recurrent(1.0)),
        ]);
        let checkpoint = cache.checkpoint();
        assert_eq!(checkpoint.len(), 4);

        // The turn moves on: more tokens, a new recurrent state.
        let more = Tensor::zeros((1, 2, 6, 8), DType::F32, &Device::Cpu).unwrap();
        cache.get_kv(0).unwrap().append(&more, &more).unwrap();
        *cache.get_recurrent(1).unwrap() = recurrent(2.0);
        assert_eq!(cache.len(), 10);

        assert!(cache.rewind(4, Some(&checkpoint)).unwrap());
        assert_eq!(cache.len(), 4, "the attention half was not trimmed");
        assert_eq!(
            marker_of(cache.get_recurrent(1).unwrap()),
            1.0,
            "the recurrent half was not restored"
        );
    }

    /// A cache that dropped positions off the front is no longer addressed by
    /// position, so a rollback that looks like one is refused.
    #[test]
    fn an_evicted_cache_is_not_rolled_back_by_position() {
        let mut small = KvCache::new(8);
        let t = Tensor::zeros((1, 2, 10, 8), DType::F32, &Device::Cpu).unwrap();
        small.append(&t, &t).unwrap();
        assert!(
            !small.holds_prefix(),
            "10 into a cache of 8 must have evicted"
        );

        let mut cache = ModelCache::new(vec![LayerCache::Kv(small)]);
        assert!(!cache.rewind(4, None).unwrap());
    }

    #[test]
    fn a_rewind_to_zero_is_a_reset_and_always_possible() {
        let mut cache = ModelCache::new(vec![
            LayerCache::Kv(kv(10)),
            LayerCache::Recurrent(recurrent(1.0)),
        ]);
        assert!(cache.rewind(0, None).unwrap());
        assert_eq!(cache.len(), 0);
    }
}
