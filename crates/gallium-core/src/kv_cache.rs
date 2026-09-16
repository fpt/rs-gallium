use candle_core::{Result, Tensor};

/// Growth headroom: the smallest buffer a cache with no history is allowed to
/// allocate. Above this it grows by doubling, so a decode adds a position with
/// no allocation until the buffer fills — see [`KvCache::plan_capacity`].
const KV_MIN_CAPACITY: usize = 256;

/// Per-layer KV cache that accumulates K and V tensors across generation steps.
///
/// Backed by **preallocated** `[batch, n_kv_heads, capacity, head_dim]` buffers
/// that each append writes into with `slice_set`, rather than a `Tensor::cat`
/// that reallocates and copies the whole cache every step (measured 30 ms → 0.7
/// ms per decode step, `docs/CANDLE_BACKEND.md`). `capacity` starts small and
/// grows by doubling — clamped to `max_seq_len`, which stays the *logical* cap
/// (e.g. a model's whole `context_length`, far larger than any real cache) and
/// the eviction boundary.
pub struct KvCache {
    /// `[b, n_kv, capacity, head_dim]`; positions `[0, cur_len - base)` are
    /// live, the rest is scratch that the next append overwrites. `None`
    /// until the first append fixes the batch/head/dim shape and the
    /// dtype/device.
    k: Option<Tensor>,
    v: Option<Tensor>,
    /// Absolute count of tokens ever appended — what [`Self::len`] reports,
    /// and what position bookkeeping (RoPE angles for the next append)
    /// needs. Never shrinks except via [`Self::truncate`]/[`Self::reset`].
    cur_len: usize,
    /// Absolute position of the buffer's physical index 0 — `0` until the
    /// first eviction. The buffer physically holds live positions
    /// `[base, cur_len)`, `cur_len - base` of them; [`Self::holds_prefix`]
    /// is exactly `base == 0`.
    base: usize,
    /// Allocated positions along dim 2. `>= cur_len - base`.
    capacity: usize,
    max_seq_len: usize,
    /// Sliding-window retention: `Some((window, headroom))` means eviction
    /// compacts down to the most recent `window` positions once the live
    /// count would exceed `window + headroom` — the model never reads past
    /// `window` from a layer built this way (`narrow_kv_to_mask` cuts every
    /// scores matmul down to the mask's span first), so retaining more is
    /// pure waste (see issue #304). `headroom` buys slack between
    /// compactions the same way `capacity` doubling buys it for the
    /// unwindowed case below — without it every append past the window
    /// would pay the cat-and-copy eviction cost instead of the cheap
    /// `slice_set` path. `None` is today's unbounded cache, where `capacity`
    /// grows by doubling up to `max_seq_len` and eviction (if it ever fires)
    /// keeps the last `max_seq_len` positions — unchanged from before this
    /// field existed.
    window: Option<(usize, usize)>,
}

/// An independent copy of a windowed [`KvCache`]'s retained tail, for
/// [`CacheCheckpoint`]. Not a plain tensor clone — see [`KvCache::snapshot`].
pub struct KvWindowSnapshot {
    k: Tensor,
    v: Tensor,
    /// Absolute position of `k`/`v`'s own index 0, at the moment this was
    /// taken — independent of whatever `base` the live cache has moved to
    /// since (that's the whole point of taking a copy).
    base: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k: None,
            v: None,
            cur_len: 0,
            base: 0,
            capacity: 0,
            max_seq_len,
            window: None,
        }
    }

    /// A cache that retains only the most recent `window` positions once
    /// exceeded — for a sliding-window attention layer, whose scores matmul
    /// never reads further back than that anyway. `headroom` is spare
    /// capacity kept beyond `window` so most appends stay on the cheap
    /// `slice_set` path; the caller's own prefill-chunk size is a reasonable
    /// choice (a whole chunk can arrive in one append). See issue #304.
    pub fn windowed(window: usize, headroom: usize, max_seq_len: usize) -> Self {
        Self {
            window: Some((window, headroom)),
            ..Self::new(max_seq_len)
        }
    }

    /// Whether this cache uses sliding-window retention — [`ModelCache`]
    /// uses this to decide whether a rewind might need [`CacheCheckpoint`]
    /// rescue for it, the same way it already does for recurrent layers.
    pub fn is_windowed(&self) -> bool {
        self.window.is_some()
    }

    /// Capacity to allocate so `need` positions fit with room to grow: the next
    /// power of two at or above `need` (never below [`KV_MIN_CAPACITY`]), capped
    /// at `ceiling`. `need >= ceiling` returns `ceiling` — eviction takes it
    /// from there.
    fn plan_capacity(need: usize, ceiling: usize) -> usize {
        if need >= ceiling {
            return ceiling;
        }
        need.max(KV_MIN_CAPACITY)
            .checked_next_power_of_two()
            .unwrap_or(need)
            .min(ceiling)
    }

    /// The physical-growth ceiling: `max_seq_len` for an unwindowed cache
    /// (today's hard ceiling, unchanged), `window + headroom` for a windowed
    /// one — see the `window` field's own doc comment for why headroom
    /// exists. Not the same as what a compaction retains — see `append`'s own
    /// `retain` computation for why that additionally depends on the size of
    /// the append that triggered it.
    fn ceiling(&self) -> usize {
        match self.window {
            Some((window, headroom)) => window + headroom,
            None => self.max_seq_len,
        }
    }

    /// The plain window size a rewind target's own history needs — distinct
    /// from `append`'s per-call `retain`, which additionally covers the
    /// *current* chunk's span. `can_rewind_to` wants this one: a target is
    /// one position, not a chunk.
    fn window_size(&self) -> usize {
        self.window.map_or(self.max_seq_len, |(w, _)| w)
    }

    /// Append new K, V to the cache. Returns views of the whole live cache
    /// (cached + new). K, V shape: `(batch, n_kv_heads, seq_len, head_dim)`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = k.dim(2)?;
        let need = self.cur_len + n;
        let live = self.cur_len - self.base;
        let ceiling = self.ceiling();
        // What a compaction below retains. For a chunked prefill (`n > 1`)
        // this has to be `window + n - 1`, not just `window`: the *first*
        // query in the chunk needs keys back to `pos - window + 1`, the
        // *last* needs keys up to `pos + n - 1`, and the mask this feeds
        // (`build_sliding_window_mask_narrowed`) is sized for that whole
        // union — retaining only `window` here left the mask wider than the
        // K/V it was added to (`broadcast_add` shape mismatch, caught by
        // `a_windowed_cache_serves_a_multi_token_chunk_after_compacting`
        // below; a single-token decode (`n == 1`) reduces this to exactly
        // `window`, unchanged from the original design).
        let retain = match self.window {
            Some((window, _)) => window + n.saturating_sub(1),
            None => self.max_seq_len,
        };

        // Eviction — the live count would exceed `ceiling`. For an unwindowed
        // cache this is unreachable from a model forward pass (attention
        // applies RoPE before it appends here, and `RoPE::apply` fails first
        // on the same overflow — its cos/sin tables are the same
        // `max_position_embeddings` tall — the context-window fail-fast for
        // §1.2); it remains reachable only for direct `KvCache` users (tests).
        // For a windowed cache this is the normal, repeated retention path,
        // firing roughly once every `headroom` appends. Either way: rebuild a
        // buffer holding the last `retain` positions and fall back to the
        // copy-based path for it.
        if live + n > ceiling {
            let (fk, fv) = match (&self.k, &self.v) {
                (Some(ck), Some(cv)) => (
                    Tensor::cat(&[&ck.narrow(2, 0, live)?, &k.contiguous()?], 2)?,
                    Tensor::cat(&[&cv.narrow(2, 0, live)?, &v.contiguous()?], 2)?,
                ),
                _ => (k.contiguous()?, v.contiguous()?),
            };
            let total = fk.dim(2)?; // live + n
            let keep = total.min(retain);
            let start = total - keep;
            self.k = Some(fk.narrow(2, start, keep)?.contiguous()?);
            self.v = Some(fv.narrow(2, start, keep)?.contiguous()?);
            self.capacity = keep;
            self.cur_len = need;
            self.base = need - keep;
            let kb = self.k.as_ref().unwrap();
            let vb = self.v.as_ref().unwrap();
            return Ok((kb.clone(), vb.clone()));
        }

        // Grow (or first allocate) when the buffers cannot hold `live + n`.
        if self.k.is_none() || live + n > self.capacity {
            let (b, h, _, hd) = k.dims4()?;
            let new_cap =
                Self::plan_capacity((live + n).max(self.capacity.saturating_mul(2)), ceiling);
            let k_buf = Tensor::zeros((b, h, new_cap, hd), k.dtype(), k.device())?;
            let v_buf = Tensor::zeros((b, h, new_cap, hd), v.dtype(), v.device())?;
            if let (Some(ok), Some(ov)) = (&self.k, &self.v) {
                // `.contiguous()` matters: narrowing dim 2 of a buffer whose
                // live count < capacity yields a strided view, which
                // `slice_set` refuses. Single-token decodes never hit it
                // (they grow only when the buffer is exactly full, where the
                // narrow is the whole tensor) — a multi-token append that
                // jumps the boundary does, i.e. a KV-reused ReAct suffix
                // prefill.
                k_buf.slice_set(&ok.narrow(2, 0, live)?.contiguous()?, 2, 0)?;
                v_buf.slice_set(&ov.narrow(2, 0, live)?.contiguous()?, 2, 0)?;
            }
            self.k = Some(k_buf);
            self.v = Some(v_buf);
            self.capacity = new_cap;
        }

        let kb = self.k.as_ref().unwrap();
        let vb = self.v.as_ref().unwrap();
        kb.slice_set(&k.contiguous()?, 2, live)?;
        vb.slice_set(&v.contiguous()?, 2, live)?;
        self.cur_len = need;

        Ok((kb.narrow(2, 0, live + n)?, vb.narrow(2, 0, live + n)?))
    }

    /// Read the current K and V without modifying the cache (for KV-shared
    /// layers). Owned narrowed views — the buffers hold scratch past the live
    /// count.
    pub fn current_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => {
                let live = self.cur_len - self.base;
                Ok(Some((k.narrow(2, 0, live)?, v.narrow(2, 0, live)?)))
            }
            _ => Ok(None),
        }
    }

    /// Current cached sequence length — the absolute position, not the
    /// physically retained count (which can be smaller once windowed
    /// eviction has run; see the `base` field).
    pub fn len(&self) -> usize {
        self.cur_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_len == 0
    }

    /// Whether this cache still holds every position from the start — false
    /// once it has dropped anything off the front (a hard `max_seq_len`
    /// ceiling, or a sliding-window layer's own retention).
    ///
    /// This is the coarse, pre-windowing check: once false, [`Self::truncate`]
    /// is not a positional rollback any more *for any target*. A windowed
    /// cache that has evicted *something* can still often serve a rewind to a
    /// target still inside what it kept — see [`Self::can_rewind_to`] for the
    /// precise version [`ModelCache::rewind`] actually uses.
    pub fn holds_prefix(&self) -> bool {
        self.base == 0
    }

    /// Whether a rewind to absolute position `target` is possible from what
    /// this cache currently retains.
    ///
    /// For the unwindowed case this is exactly `holds_prefix() && len() >=
    /// target` (nothing's ever evicted, so any target up to `cur_len` is
    /// fine). For a windowed cache it's stricter than that pairing would
    /// suggest, and stricter in the way that matters: rewind is a
    /// front-anchored "keep positions before the target" operation, and
    /// window eviction *also* drops from the front — so a target is feasible
    /// only if the *whole window ending at it* (not just the target position
    /// itself) is still physically retained, i.e. `base <= target -
    /// window`. A target that isn't — most of them, once eviction has run
    /// past this position — needs the [`CacheCheckpoint`] rescue instead.
    pub fn can_rewind_to(&self, target: usize) -> bool {
        if target > self.cur_len {
            return false;
        }
        self.base <= target.saturating_sub(self.window_size())
    }

    /// Drop everything past `len`, keeping positions `0..len` of what's
    /// retained (all of it, if this cache has never evicted).
    ///
    /// This is the rollback an attention cache can do and a recurrent state
    /// cannot: K and V are a per-position log, so a prefix of them is a valid
    /// cache for that prefix. It is what lets iteration *N+1* of an agent turn
    /// evaluate only what iteration *N*'s prompt did not already contain.
    /// Callers check [`Self::can_rewind_to`] first — this just moves the
    /// write pointer back (`base` is untouched: the buffer may now retain a
    /// little more history than a window ending at `len` strictly needs,
    /// which the next eviction trims, not a correctness issue). Positions
    /// past `len` become scratch that the next append overwrites, and the
    /// preallocated buffer is kept — no copy.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len == 0 {
            self.reset();
            return Ok(());
        }
        if len < self.cur_len {
            self.cur_len = len;
        }
        Ok(())
    }

    /// An independent copy of the currently retained tail, for
    /// [`CacheCheckpoint`]. `None` before the first append.
    ///
    /// Not a plain tensor clone: unlike [`RecurrentState::snapshot`] — cheap
    /// because a recurrent step *replaces* its tensor rather than writing
    /// through it, so a clone shares storage nothing will touch again — this
    /// buffer is mutated **in place** by `slice_set` (candle: "modifies self
    /// in place"). A cloned handle would alias the live buffer's storage and
    /// the next append would silently corrupt the "snapshot" retroactively.
    /// `.narrow(...).contiguous()` is a real, independent copy here — the
    /// live count is generally smaller than `capacity`, so the narrow is
    /// non-contiguous (same reason the growth path above needs it) and
    /// `.contiguous()` isn't a formality.
    pub fn snapshot(&self) -> Result<Option<KvWindowSnapshot>> {
        let (Some(k), Some(v)) = (&self.k, &self.v) else {
            return Ok(None);
        };
        let live = self.cur_len - self.base;
        Ok(Some(KvWindowSnapshot {
            k: k.narrow(2, 0, live)?.contiguous()?,
            v: v.narrow(2, 0, live)?.contiguous()?,
            base: self.base,
        }))
    }

    /// Restore from a snapshot taken earlier, when this cache's own absolute
    /// position was `at_len` (the checkpoint's own `len` — the snapshot
    /// itself only records its tail's *own* base, not where the conversation
    /// had reached when it was taken).
    ///
    /// Rebuilds a fresh buffer sized the same way a windowed cache normally
    /// grows (room for `headroom` further appends before the next eviction),
    /// pre-populated with the snapshot's tail at the front — not a swap of
    /// the snapshot's own tensors into `self`, so the snapshot stays valid
    /// for a second restore if this rewind target turns out to need retrying.
    pub fn restore_snapshot(&mut self, snap: &KvWindowSnapshot, at_len: usize) -> Result<()> {
        let live = snap.k.dim(2)?;
        let cap = Self::plan_capacity(live, self.ceiling()).max(live);
        let (b, h, _, hd) = snap.k.dims4()?;
        let k_buf = Tensor::zeros((b, h, cap, hd), snap.k.dtype(), snap.k.device())?;
        let v_buf = Tensor::zeros((b, h, cap, hd), snap.v.dtype(), snap.v.device())?;
        k_buf.slice_set(&snap.k, 2, 0)?;
        v_buf.slice_set(&snap.v, 2, 0)?;
        self.k = Some(k_buf);
        self.v = Some(v_buf);
        self.capacity = cap;
        self.cur_len = at_len;
        self.base = snap.base;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.cur_len = 0;
        self.base = 0;
        self.capacity = 0;
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
    /// `(layer index, tail)` for every windowed KV layer — the same rescue
    /// recurrent layers get, for the same reason: a sliding-window cache has
    /// no positional rollback once eviction has run past the target (see
    /// `KvCache::can_rewind_to`), so a checkpoint taken while the target was
    /// still live is the only way back. See issue #304.
    windowed: Vec<(usize, KvWindowSnapshot)>,
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
    /// state that cannot always be rolled back by position (a recurrent
    /// layer, never; a windowed KV layer, once eviction has run past a given
    /// target — but the checkpoint has to exist *before* that happens, so
    /// this says yes as soon as the layer exists, not only once it's needed).
    pub fn needs_checkpoint(&self) -> bool {
        self.layers.iter().any(|l| match l {
            LayerCache::Recurrent(_) => true,
            LayerCache::Kv(kv) => kv.is_windowed(),
            _ => false,
        })
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
            windowed: self
                .layers
                .iter()
                .enumerate()
                .filter_map(|(i, l)| match l {
                    LayerCache::Kv(kv) if kv.is_windowed() => Some((
                        i,
                        kv.snapshot()
                            .expect("windowed KvCache snapshot")
                            .expect("windowed KvCache has appended at least once"),
                    )),
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
    /// exactly `len`, when a windowed KV layer has evicted past `len` without
    /// a checkpoint to rescue it, and for a TurboQuant cache, which has no
    /// positional rollback at all. `len == 0` is always possible: that is a
    /// reset.
    pub fn rewind(&mut self, len: usize, checkpoint: Option<&CacheCheckpoint>) -> Result<bool> {
        if len == 0 {
            self.reset();
            return Ok(true);
        }
        let usable = checkpoint.filter(|c| c.len == len);
        let windowed_rescue =
            |i: usize| usable.and_then(|c| c.windowed.iter().find(|(idx, _)| *idx == i));
        let feasible = self.layers.iter().enumerate().all(|(i, l)| match l {
            LayerCache::Kv(kv) => kv.can_rewind_to(len) || windowed_rescue(i).is_some(),
            LayerCache::Shared { .. } => true,
            LayerCache::Recurrent(_) => usable.is_some(),
            LayerCache::TurboKv(_) => false,
        });
        if !feasible {
            return Ok(false);
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                LayerCache::Kv(kv) => {
                    if kv.can_rewind_to(len) {
                        kv.truncate(len)?;
                    } else {
                        let (_, snap) = windowed_rescue(i).expect("feasibility checked above");
                        kv.restore_snapshot(snap, len)?;
                    }
                }
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
    use candle_core::{Device, IndexOp};

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

    /// A prefill then many single-token decodes — the buffer grows by doubling
    /// under it and every position that was written is still readable and equal
    /// to what went in. `slice_set` into a preallocated buffer must not disturb
    /// the positions before the write, and `narrow` must never expose scratch.
    #[test]
    fn append_preserves_every_written_position_across_growth() {
        let device = Device::Cpu;
        let mut cache = KvCache::new(100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();

        // Prefill of 5, then decode past 256: a prefill this small starts at
        // KV_MIN_CAPACITY, so position 256 is the first append that has to
        // reallocate and copy the live prefix into the doubled buffer — the
        // branch this test exists for. (A loop that stops short of the
        // boundary never runs it: 45 positions used to pass while covering
        // only the in-place writes.)
        let prefill = Tensor::cat(&(0..5).map(|i| step(i as f32)).collect::<Vec<_>>(), 2).unwrap();
        cache.append(&prefill, &prefill).unwrap();
        assert_eq!(
            cache.capacity, KV_MIN_CAPACITY,
            "small prefill starts at the floor"
        );
        let mut last = vec![];
        for i in 5..300 {
            let (k, v) = cache.append(&step(i as f32), &step(i as f32)).unwrap();
            assert_eq!(k.dim(2).unwrap(), i + 1);
            assert_eq!(v.dim(2).unwrap(), i + 1);
            last = k.i((0, 0, .., 0)).unwrap().to_vec1::<f32>().unwrap();
        }
        assert_eq!(cache.len(), 300);
        assert_eq!(cache.capacity, 512, "the 256 → 512 growth realloc happened");
        assert_eq!(
            last,
            (0..300).map(|i| i as f32).collect::<Vec<f32>>(),
            "position p still holds value p after growth"
        );
    }

    /// A multi-token append that jumps the growth boundary while the buffer is
    /// only part-full — the KV-reused ReAct shape: reuse leaves `cur_len`
    /// mid-buffer, then a suffix prefill pushes `need` past `capacity`. The
    /// live prefix copied into the doubled buffer is then a *strided* narrow
    /// (`cur_len < capacity`), which `slice_set` refuses unless made
    /// contiguous first; this failed with "slice-set only supports contiguous
    /// tensors" and cost the whole turn. Single-token decodes never hit it —
    /// they grow exactly at `cur_len == capacity`, where the narrow is the
    /// whole tensor — which is why `append_preserves_every_written_position_
    /// across_growth` above stayed green over the bug.
    #[test]
    fn a_partfull_buffer_survives_a_growth_jumping_append() {
        let device = Device::Cpu;
        let mut cache = KvCache::new(100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        let run = |a: usize, b: usize| {
            Tensor::cat(&(a..b).map(|i| step(i as f32)).collect::<Vec<_>>(), 2).unwrap()
        };

        // 200 positions into a KV_MIN_CAPACITY=256 buffer: part-full.
        cache.append(&run(0, 200), &run(0, 200)).unwrap();
        assert_eq!(cache.capacity, KV_MIN_CAPACITY);
        assert!(cache.len() < cache.capacity, "the narrow must be strided");

        // 100 more jump the boundary: grow-with-copy from the strided prefix.
        let (k, _) = cache.append(&run(200, 300), &run(200, 300)).unwrap();
        assert_eq!(cache.capacity, 512);
        assert_eq!(
            k.i((0, 0, .., 0)).unwrap().to_vec1::<f32>().unwrap(),
            (0..300).map(|i| i as f32).collect::<Vec<f32>>(),
            "every position survives the copy"
        );
    }

    /// Truncate is a pointer move: after it, an append overwrites from `len` and
    /// the earlier positions are untouched.
    #[test]
    fn truncate_then_append_overwrites_from_the_cut() {
        let device = Device::Cpu;
        let mut cache = KvCache::new(1024);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        for i in 0..10 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        cache.truncate(4).unwrap();
        assert_eq!(cache.len(), 4);
        let (k, _) = cache.append(&step(99.0), &step(99.0)).unwrap();
        let row: Vec<f32> = k.i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        assert_eq!(row, vec![0.0, 1.0, 2.0, 3.0, 99.0]);
    }

    /// A windowed cache's live length never exceeds `window + headroom`
    /// (the buffer is allowed that much slack between compactions — see the
    /// `window` field's own doc comment for why), however long the
    /// conversation runs — the whole point of issue #304. Values are
    /// positions, so the retained tail must always be the *most recent*
    /// positions, with nothing older than `window + headroom` back.
    #[test]
    fn a_windowed_cache_bounds_live_length_to_window_plus_headroom() {
        let device = Device::Cpu;
        let mut cache = KvCache::windowed(8, 4, 100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();

        let mut last_k = None;
        for i in 0..50 {
            let (k, _) = cache.append(&step(i as f32), &step(i as f32)).unwrap();
            assert!(
                k.dim(2).unwrap() <= 12,
                "position {i}: live length {} exceeds window+headroom 12",
                k.dim(2).unwrap()
            );
            last_k = Some(k);
        }
        // `len()` still reports the true absolute count (position bookkeeping
        // needs it), even though only the tail is physically retained.
        assert_eq!(cache.len(), 50);
        // Whatever length the last append settled on, it must be the
        // contiguous, correctly-ordered suffix of positions ending at 49 —
        // not a stale or reordered mix from an earlier compaction.
        let row: Vec<f32> = last_k.unwrap().i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        let expect: Vec<f32> = (50 - row.len()..50).map(|i| i as f32).collect();
        assert_eq!(row, expect);
    }

    /// The bug a real Gemma 4 candle run caught (`shape mismatch in
    /// broadcast_add, lhs: [.., 52, 1024], rhs: [.., 52, 1075]`): a chunked
    /// prefill's mask for `n` new queries against a sliding window needs
    /// `window + n - 1` key positions (the *first* query's own window reaches
    /// back `window - 1` further than the chunk's last position), not just
    /// `window`. A compaction that fires *during* such a multi-token append
    /// must retain enough for the whole chunk, or the returned K/V is
    /// narrower than the mask built for it.
    #[test]
    fn a_windowed_cache_serves_a_multi_token_chunk_after_compacting() {
        let device = Device::Cpu;
        let window = 8;
        let mut cache = KvCache::windowed(window, 4, 100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        let chunk = |a: i64, b: i64| {
            Tensor::cat(&(a..b).map(|i| step(i as f32)).collect::<Vec<_>>(), 2).unwrap()
        };

        // Fill well past the ceiling with single-token appends first (the
        // ordinary decode-shaped path), then one multi-token append (`n =
        // 5`) that itself triggers compaction.
        for i in 0..15 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        let five = chunk(15, 20);
        let (k, v) = cache.append(&five, &five).unwrap();

        let needed = window + 5 - 1; // 12
        assert!(
            k.dim(2).unwrap() >= needed,
            "chunk of 5 against window {window} needs >= {needed} key positions, got {}",
            k.dim(2).unwrap()
        );
        assert_eq!(k.dim(2).unwrap(), v.dim(2).unwrap());
        // And it's still the correct, contiguous, most-recent tail.
        let row: Vec<f32> = k.i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        let expect: Vec<f32> = (20 - row.len() as i64..20).map(|i| i as f32).collect();
        assert_eq!(row, expect);

        // A single-token decode step right after is unaffected — same as the
        // original, non-chunked case.
        let (k, _) = cache.append(&step(20.0), &step(20.0)).unwrap();
        assert!(k.dim(2).unwrap() <= window + 4 /* headroom */);
    }

    /// `can_rewind_to` is the precise, per-target check `ModelCache::rewind`
    /// needs: feasible only when the *whole window ending at the target* is
    /// still retained, not merely when the target position itself is.
    #[test]
    fn can_rewind_to_requires_the_whole_target_window_retained() {
        let device = Device::Cpu;
        let mut cache = KvCache::windowed(8, 4, 100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        for i in 0..20 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        // Live positions are now the tail of [0,20) that eviction kept.
        // A target far enough back that its own 8-window reaches before
        // whatever eviction has already dropped must be refused...
        assert!(!cache.can_rewind_to(5));
        // ...while the current length (a no-op rewind) is always fine.
        assert!(cache.can_rewind_to(20));
        assert!(!cache.can_rewind_to(21), "past cur_len is never feasible");
    }

    /// The bug `KvCache::snapshot`'s own doc comment is about: `slice_set`
    /// mutates in place, so a naive clone of the live buffer would alias it,
    /// and a *subsequent* append would silently corrupt the "snapshot". This
    /// pins the fix — take a snapshot, keep appending, and the snapshot's own
    /// values must be unchanged.
    #[test]
    fn a_window_snapshot_survives_later_appends() {
        let device = Device::Cpu;
        let mut cache = KvCache::windowed(8, 4, 100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        for i in 0..8 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        let snap = cache.snapshot().unwrap().unwrap();
        let before: Vec<f32> = snap.k.i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        assert_eq!(before, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Keep going — well past the point where the live buffer's own
        // storage gets overwritten and/or reallocated.
        for i in 8..40 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }

        let after: Vec<f32> = snap.k.i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        assert_eq!(after, before, "snapshot must not see later appends");
    }

    /// `restore_snapshot` rebuilds a live, appendable cache from a saved
    /// tail — not just a readable one. After restoring, further appends must
    /// work and grow from the restored position, not from wherever the live
    /// cache happened to be before the restore.
    #[test]
    fn restore_snapshot_produces_a_cache_appends_can_continue_from() {
        let device = Device::Cpu;
        let mut cache = KvCache::windowed(8, 4, 100_000);
        let step = |val: f32| Tensor::full(val, (1, 2, 1, 4), &device).unwrap();
        for i in 0..8 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        let snap = cache.snapshot().unwrap().unwrap();

        // The turn moves on, well past the window several times over.
        for i in 8..40 {
            cache.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        assert_eq!(cache.len(), 40);

        // Restore back to the checkpoint (absolute position 8).
        cache.restore_snapshot(&snap, 8).unwrap();
        assert_eq!(cache.len(), 8);
        assert!(cache.can_rewind_to(8), "restored cache holds its own tail");

        let (k, _) = cache.append(&step(100.0), &step(100.0)).unwrap();
        let row: Vec<f32> = k.i((0, 0, .., 0)).unwrap().to_vec1().unwrap();
        assert_eq!(row, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]);
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

    fn windowed_kv(window: usize, headroom: usize) -> KvCache {
        KvCache::windowed(window, headroom, 4096)
    }

    #[test]
    fn a_windowed_layer_needs_a_checkpoint() {
        let cache = ModelCache::new(vec![LayerCache::Kv(windowed_kv(8, 4))]);
        assert!(
            cache.needs_checkpoint(),
            "a windowed layer can't always be rolled back by position"
        );
    }

    /// Still inside the window: no checkpoint needed at all, same as an
    /// ordinary attention-only cache — windowing only changes behavior once
    /// eviction has actually run.
    #[test]
    fn a_windowed_layer_still_rolls_back_by_position_within_the_window() {
        let mut small = windowed_kv(8, 4);
        let t = Tensor::zeros((1, 2, 5, 8), DType::F32, &Device::Cpu).unwrap();
        small.append(&t, &t).unwrap();

        let mut cache = ModelCache::new(vec![LayerCache::Kv(small)]);
        assert!(cache.rewind(2, None).unwrap());
        assert_eq!(cache.len(), 2);
    }

    /// The scenario issue #304 is for: eviction has run past the rewind
    /// target, so a plain positional rewind is refused (same as the
    /// unwindowed `an_evicted_cache_is_not_rolled_back_by_position` case) —
    /// but a checkpoint taken while the target was still live rescues it.
    #[test]
    fn a_windowed_layer_past_its_window_needs_the_checkpoint_rescue() {
        let mut small = windowed_kv(8, 4);
        let step = |v: f32| Tensor::full(v, (1, 2, 1, 8), &Device::Cpu).unwrap();
        for i in 0..6 {
            small.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        let mut cache = ModelCache::new(vec![LayerCache::Kv(small)]);
        let checkpoint = cache.checkpoint(); // taken at 6, still fully live

        // Push well past the window — the checkpoint target is now gone from
        // the live buffer.
        for i in 6..30 {
            let t = step(i as f32);
            cache.get_kv(0).unwrap().append(&t, &t).unwrap();
        }
        assert_eq!(cache.len(), 30);
        assert!(
            !cache.get_kv(0).unwrap().can_rewind_to(6),
            "6 should no longer be reachable by position after eviction"
        );

        // Without the checkpoint: refused, same as the unwindowed case.
        assert!(!cache.rewind(6, None).unwrap());
        assert_eq!(cache.len(), 30, "a refused rewind changes nothing");

        // With it: rescued.
        assert!(cache.rewind(6, Some(&checkpoint)).unwrap());
        assert_eq!(cache.len(), 6);

        // And the restored cache is appendable, continuing from position 6.
        let t = step(200.0);
        cache.get_kv(0).unwrap().append(&t, &t).unwrap();
        assert_eq!(cache.len(), 7);
    }

    /// A checkpoint at the wrong length doesn't rescue a windowed layer any
    /// more than it rescues a recurrent one (`a_checkpoint_at_the_wrong_
    /// length_is_refused_too` above).
    #[test]
    fn a_windowed_checkpoint_at_the_wrong_length_is_refused_too() {
        let mut small = windowed_kv(8, 4);
        let step = |v: f32| Tensor::full(v, (1, 2, 1, 8), &Device::Cpu).unwrap();
        for i in 0..6 {
            small.append(&step(i as f32), &step(i as f32)).unwrap();
        }
        let mut cache = ModelCache::new(vec![LayerCache::Kv(small)]);
        let stale = cache.checkpoint(); // taken at 6

        for i in 6..30 {
            let t = step(i as f32);
            cache.get_kv(0).unwrap().append(&t, &t).unwrap();
        }
        assert!(!cache.rewind(5, Some(&stale)).unwrap());
        assert_eq!(cache.len(), 30);
    }
}
