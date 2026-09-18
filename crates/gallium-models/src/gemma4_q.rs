//! Quantized Gemma 4 model loaded from GGUF.
//!
//! GGUF tensor names differ from safetensors:
//!   attn_norm          → input_layernorm
//!   post_attention_norm → post_attention_layernorm
//!   ffn_norm           → pre_feedforward_layernorm
//!   post_ffw_norm      → post_feedforward_layernorm
//!   inp_gate           → per_layer_input_gate  (PLE)
//!   proj               → per_layer_projection  (PLE)
//!   post_norm          → post_per_layer_input_norm
//!   layer_output_scale → layer_scalar
//!   per_layer_model_proj → per_layer_model_projection
//!   per_layer_proj_norm  → per_layer_projection_norm
//!   per_layer_token_embd → embed_tokens_per_layer

use candle_core::{DType, Device, Module, Result, Tensor, D};

use gallium_core::quantized::{GgufMetadata, QExperts, QLinear, QNorm, QVarBuilder};
use gallium_core::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// RMSNorm without a learnable scale (Gemma 4 v_norm). Normalizes over the last dim.
fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let sq_mean = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(sq_mean + eps)?.sqrt()?)?;
    normed.to_dtype(orig)
}

/// Build proportional RoPE inv_freq: `rope_angles` real freqs + `nope_angles` zeros.
/// Matches `_compute_proportional_rope_parameters` in modeling_rope_utils.py.
fn proportional_inv_freq(head_dim: usize, partial_rotary_factor: f64, theta: f64) -> Vec<f64> {
    let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
    let nope_angles = head_dim / 2 - rope_angles;
    let mut inv_freq: Vec<f64> = (0..rope_angles)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();
    inv_freq.extend(std::iter::repeat(0.0).take(nope_angles));
    inv_freq
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Widest head the fused Metal *prefill* path is used at — see
/// `QAttention::fused_attention` for the measurement behind it.
const FUSED_PREFILL_MAX_HEAD_DIM: usize = 256;

struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    /// Absent on shared-K=V layers (26B-A4B global attention): V is then the
    /// raw K projection output (before k_norm/RoPE), mirroring llama.cpp's
    /// `Vcur = wv ? wv(cur) : Kcur`.
    v_proj: Option<QLinear>,
    o_proj: QLinear,
    q_norm: QNorm,
    k_norm: QNorm,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    rms_eps: f64,
    /// Storage dtype for the KV cache — `Some(F16)` when the caller's
    /// `kv_f16` (`[llm] gemma4KvF16` / `GALLIUM_GEMMA4_KV_F16`, default on)
    /// resolves true, `None` (f32) otherwise. `q`/`k`/`v` cast down after
    /// RoPE for the cache write and scores matmul; `scores` is upcast back
    /// to f32 for the mask add and softmax, `probs` cast back down to
    /// `v`'s dtype for the weighted sum — `scores` is the cheapest tensor
    /// here to keep at full precision, and softmax is where the precision
    /// sensitivity lives. See issue #305 for the measurement protocol and
    /// two rejected designs: the whole computation in `kv_dtype`
    /// (memory-positive, but a softmax-precision correctness bug under
    /// non-greedy sampling with no system prompt), and caching in
    /// `kv_dtype` with `k` upcast after the cache read (fixes that bug
    /// too, but grows VRAM with decode length to OOM — likely, not
    /// confirmed, an allocator effect from `k`'s size changing every
    /// step). Verified default-on across E4B/12B/26B-A4B with no
    /// regression attributable to it — docs/VERIFICATION_STATUS.md.
    kv_dtype: Option<DType>,
    /// Run attention through candle's fused Metal kernel
    /// (`candle_nn::ops::sdpa`) instead of `gqa_scores` → softmax →
    /// `gqa_weighted_sum`. Experiment for issue #308, `GALLIUM_GEMMA4_SDPA=1`,
    /// off by default; resolved (and gated on Metal + `kv_narrow` + f16) in
    /// `Gemma4Q::load`. See [`Self::fused_attention`] for which shapes take
    /// it and which fall back.
    fused: bool,
    /// Run *prefill* attention (all layers — issue #313's head_dim-512
    /// correctness bug is fixed as of the candle `0.11.0` bump) through
    /// `candle-flash-attn` (issue #308, CUDA-only, needs the `flash-attn`
    /// cargo feature — a build-time dependency, not turned on by `cuda`
    /// alone, since the nvcc build is a real cost most CUDA builds shouldn't
    /// pay for a dependency nothing uses without opting in at build time).
    /// On by default whenever the build and the runtime can actually take
    /// it (the `flash-attn` cargo feature is compiled in, the device is
    /// CUDA, and `kv_f16` is on); `GALLIUM_GEMMA4_FLASH_ATTN=0` opts back
    /// out. Resolved in `Gemma4Q::load`. Decode always stays on the matmul
    /// path — see [`Self::flash_attention`]'s doc comment for why.
    flash: bool,
}

impl QAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &QVarBuilder,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rms_eps: f64,
        kv_f16: bool,
        fused: bool,
        flash: bool,
    ) -> Result<Self> {
        let v_proj = if vb.contains("attn_v.weight") {
            Some(QLinear::load(&vb.pp("attn_v"))?)
        } else {
            None
        };
        let kv_dtype = kv_f16.then_some(DType::F16);
        Ok(Self {
            q_proj: QLinear::load(&vb.pp("attn_q"))?,
            k_proj: QLinear::load(&vb.pp("attn_k"))?,
            v_proj,
            o_proj: QLinear::load(&vb.pp("attn_output"))?,
            q_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_q_norm"))?,
            k_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_k_norm"))?,
            n_q,
            n_kv,
            head_dim,
            rms_eps,
            kv_dtype,
            fused,
            flash,
        })
    }

    /// One attention over `q` `(b, h, s, d)` and the cache views `k`/`v`
    /// `(b, h_kv, t, d)` — already narrowed to the mask's span — returning
    /// `(b, h, s, d)` in `out_dtype`. Shared by `forward` and
    /// `forward_shared`, which differ only in where K/V come from.
    ///
    /// `window` is the sliding layer's window width (`None` for a global
    /// layer) — needed by `flash_attention` alongside `mask`, since
    /// `candle-flash-attn`'s windowing is a kernel parameter rather than an
    /// additive tensor.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        window: Option<usize>,
        out_dtype: DType,
    ) -> Result<Tensor> {
        if let Some(out) = self.fused_attention(q, k, v, mask)? {
            return out.to_dtype(out_dtype);
        }
        if let Some(out) = self.flash_attention(q, k, v, window)? {
            return out.to_dtype(out_dtype);
        }
        // scale = 1.0: q_norm controls effective magnitude. `scores` is
        // upcast to f32 for the mask add and softmax — `[b, h, s, t]`, a
        // few bytes per attended token rather than a copy of the whole
        // K/V history — then `probs` is cast back to `v`'s dtype (a no-op
        // when `kv_dtype` is unset) for the weighted sum.
        let mut scores = gqa_scores(q, k)?.to_dtype(DType::F32)?;
        if let Some(mask) = mask {
            scores = scores
                .broadcast_add(&mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        gqa_weighted_sum(&probs, v)?.to_dtype(out_dtype)
    }

    /// The fused Metal path (`candle_nn::ops::sdpa`, issue #308), or `None`
    /// when this call has to take the matmul path instead.
    ///
    /// candle routes `s <= 8` to its *vector* kernel and `s > 8` to its
    /// *full* kernel, and only the full kernel reads a mask (`ops.rs` threads
    /// `self.mask` into `call_sdpa_full` alone). So:
    /// - `s == 1` (decode): vector kernel, no mask. Sound because
    ///   `Gemma4Q::load` gates `fused` on `kv_narrow`: a sliding layer's K
    ///   is already narrowed to exactly its window, so the mask it was
    ///   handed is all zeros (`build_sliding_window_mask_narrowed`'s
    ///   short-circuit), and a global layer has none at decode.
    /// - `2..=8` (a short KV-reused suffix): fall back — the vector kernel
    ///   would drop the sliding/causal mask silently.
    /// - `s > 8` (prefill): full kernel with the additive mask cast to
    ///   `q`'s dtype and broadcast over batch and heads as a strided view
    ///   (the kernel takes mask strides), so nothing `(b, h, s, t)`-sized is
    ///   materialized.
    /// - `s > 1` with no mask cannot happen (`attention_mask_needed` is true
    ///   for any multi-token batch); if it does, fall back rather than guess.
    fn fused_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if !self.fused {
            return Ok(None);
        }
        // `k`/`v` are handed over as the cache's own narrowed views (non-zero
        // start offset, head stride = the buffer's capacity): the kernels take
        // strides, and candle passes the offset in bytes since huggingface/candle
        // 0c5895368 (#3599). Before that rev the element count went in as a
        // byte offset and a narrowed view was read from the wrong place — the
        // greedy stream diverged from the second token — which is why the pin
        // is at or past that commit.
        let (b, h, s, _) = q.dims4()?;
        if s == 1 {
            // Isolation switch for the #308 experiment: `GALLIUM_GEMMA4_SDPA_DECODE=0`
            // keeps decode on the matmul path so prefill can be judged alone.
            if matches!(
                std::env::var("GALLIUM_GEMMA4_SDPA_DECODE").as_deref(),
                Ok("0")
            ) {
                return Ok(None);
            }
            return Ok(Some(candle_nn::ops::sdpa(q, k, v, None, false, 1.0, 1.0)?));
        }
        let Some(mask) = mask else {
            return Ok(None);
        };
        if s <= 8 {
            return Ok(None);
        }
        // The full kernel is 2x faster than the matmul path at head_dim 256
        // and 5-6x *slower* at 512 (E4B shapes, `metal_sdpa_bench`): its mlx
        // ancestor was tuned for <= 128 and the 512 instantiation spills. So a
        // prefill takes it only on the sliding layers; the global layers
        // (head_dim 512 on every Gemma 4) stay on the matmul path at prefill
        // and take the vector kernel — which does win at 512 — at decode.
        if self.head_dim > FUSED_PREFILL_MAX_HEAD_DIM {
            return Ok(None);
        }
        let t = k.dim(2)?;
        let mask = mask
            .to_dtype(q.dtype())?
            .reshape((1, 1, s, t))?
            .broadcast_as((b, h, s, t))?;
        Ok(Some(candle_nn::ops::sdpa(
            q,
            k,
            v,
            Some(&mask),
            false,
            1.0,
            1.0,
        )?))
    }

    /// The fused CUDA prefill path (`candle-flash-attn`, issue #308), or
    /// `None` when this call has to take the matmul path instead.
    ///
    /// **All layers**, sliding (`window: Some(_)`, head_dim 256) and global
    /// (`window: None`, head_dim 512) alike, as of the candle `0.11.0` bump
    /// (rev `31f35b1`). That was not always true: at the previous pin
    /// (`0c5895368`), the head_dim-512 causal kernel was measurably **wrong**
    /// — max |Δlogit| ~22 against an f32 reference and a degenerate greedy
    /// stream (repeated newlines within a few tokens) through the real
    /// model, root-caused down to the vendored kernel itself, outside
    /// gallium and outside Gemma 4 entirely
    /// (`flash_attn_hdim256_vs_hdim512_at_matched_magnitude` and its sibling
    /// probes, `gallium-models/tests/integration.rs`: random Q/K/V, no
    /// model, no GGUF weights — the `KvCache` stride gap and GQA each moved
    /// the delta a little, but *value magnitude* was what reproduced the
    /// real failure, and only at head_dim 512). Filed as issue #313. The
    /// `0.11.0` bump (huggingface/candle#3655's hdim512-specific kernel
    /// changes, landed for an unrelated multimodal-prefix feature) fixes it:
    /// the same probes now measure max |Δ| ~0.0002–0.0005 at baseline
    /// amplitude and ~0.2 at 10× — in the same range as head_dim 256's own
    /// numbers, not the ~20 the broken kernel produced. Through the real
    /// model, flash-f16 with global layers included now measures max |Δ|
    /// 1.11 against f32 (16/262144 vocab positions off by >1.0) — slightly
    /// *better* than the sliding-only figure (1.25, 17 positions) this path
    /// shipped with initially, not worse. #313 stays open only as the record
    /// of what was broken and where it got fixed, not as a live constraint
    /// on this function.
    ///
    /// **Prefill only** (`s > 1`) — decode stays on `gqa_scores` deliberately:
    /// the vendored FA2 build sets `num_splits = 1` and never reaches its
    /// split-KV decode kernel, so a `seqlen_q = 1` call walks the whole KV
    /// history serially in one block per (batch, head) rather than
    /// parallelizing across it — plausibly slower than the existing cuBLAS
    /// batched matmul at long context, not faster (unmeasured — there is no
    /// split-KV path here to benchmark against).
    ///
    /// `q`/`k`/`v` arrive `(b, h, s, d)`; `candle-flash-attn` wants
    /// `(b, s, h, d)`. `transpose(1, 2)` is metadata-only here — the kernel
    /// only requires the *last* dim contiguous (row/head strides are passed
    /// to the C side separately), which a `(b, h, s, d)`-contiguous buffer
    /// already satisfies after the transpose, so this costs no copy. GQA is
    /// native — `k`/`v` keep their own (smaller) head count; the kernel
    /// only requires it divides `q`'s.
    ///
    /// Windowing: a sliding layer passes `window_size_left = window - 1`,
    /// `window_size_right = 0` (`flash_attn_windowed`); a global layer
    /// passes plain `causal = true` (`flash_attn`) — global layers never
    /// narrow K, so there's no window width to give it. Both rely on FA2's
    /// bottom-right causal alignment for `seqlen_q < seqlen_k` — query row
    /// `i` is treated as sitting at `seqlen_k - seqlen_q + i`, which lines
    /// up with this cache's own window/causal math exactly when K has
    /// already been narrowed by `narrow_kv_to_mask` (always true by the
    /// time this is called, whether or not `kv_narrow` is on: narrowing is
    /// a no-op once K is no wider than the mask, which is exactly the
    /// un-narrowed case — the global layers' own case always).
    /// `gemma4_gguf_flash_attn_matches_matmul` pins this against an f32
    /// reference — not against the matmul-f16 path, which that same test
    /// found is not a trustworthy reference on this machine's CUDA build
    /// (its own drift from f32 is larger than this path's).
    #[cfg(feature = "flash-attn")]
    fn flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        if !self.flash {
            return Ok(None);
        }
        let s = q.dim(2)?;
        if s <= 1 || !matches!(q.dtype(), DType::F16 | DType::BF16) {
            return Ok(None);
        }
        let qt = q.transpose(1, 2)?;
        let kt = k.transpose(1, 2)?;
        let vt = v.transpose(1, 2)?;
        let out = match window {
            Some(w) => candle_flash_attn::flash_attn_windowed(
                &qt,
                &kt,
                &vt,
                1.0,
                Some(w.saturating_sub(1)),
                Some(0),
            )?,
            None => candle_flash_attn::flash_attn(&qt, &kt, &vt, 1.0, true)?,
        };
        Ok(Some(out.transpose(1, 2)?))
    }

    #[cfg(not(feature = "flash-attn"))]
    fn flash_attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        // `self.flash` is only ever read here, so it'd be flagged dead code
        // in a build without the `flash-attn` cargo feature otherwise — it's
        // resolved at load time regardless of this crate's own features
        // (`Gemma4Q::load` gates on `cfg!(feature = "flash-attn")` itself,
        // per the comment there, and warns rather than staying silent).
        let _ = self.flash;
        Ok(None)
    }

    /// Cast to `kv_dtype`, a no-op unless it's set. Applied to `q` (so it
    /// matches `k`/`v` for the scores matmul) and to `k`/`v` themselves
    /// right before they enter the cache — see `kv_dtype`'s own doc comment
    /// for why `scores`, not `k`/`v` on the way back out, is what gets
    /// upcast for softmax.
    fn to_cache_dtype(&self, t: &Tensor) -> Result<Tensor> {
        match self.kv_dtype {
            Some(dt) => t.to_dtype(dt),
            None => Ok(t.clone()),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, h_kv, d) = (self.n_q, self.n_kv, self.head_dim);

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_raw = self
            .k_proj
            .forward(x)?
            .reshape((b, s, h_kv, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = match &self.v_proj {
            Some(vp) => vp
                .forward(x)?
                .reshape((b, s, h_kv, d))?
                .transpose(1, 2)?
                .contiguous()?,
            None => k_raw.clone(), // shared K=V: raw K projection, pre-norm/RoPE
        };

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k_raw)?;

        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;
        let v = rms_norm_no_scale(&v, self.rms_eps)?;

        // Cast down once, right after RoPE (so the position encoding itself
        // is computed at full precision) — `q` alongside `k`/`v`, so the
        // scores matmul type-checks. See `kv_dtype`'s doc comment for the
        // two rejected alternatives and why upcasting `scores` (not `k`,
        // not the whole computation) is what's here instead.
        let out_dtype = q.dtype();
        let q = self.to_cache_dtype(&q)?;
        let k_store = self.to_cache_dtype(&k.contiguous()?)?;
        let v_store = self.to_cache_dtype(&v.contiguous()?)?;
        let (k, v) = kv_cache.append(&k_store, &v_store)?;
        let (k, v) = narrow_kv_to_mask(k, v, mask)?;

        // scale = 1.0: q_norm controls effective magnitude. `scores` is
        // upcast to f32 for the mask add and softmax — `[b, h, s, t]`, a
        // few bytes per attended token rather than a copy of the whole
        // K/V history — then `probs` is cast back to `v`'s dtype (a no-op
        // when `kv_dtype` is unset) for the weighted sum.
        let out = self.attend(&q, &k, &v, mask, window, out_dtype)?;
        self.o_proj
            .forward(&out.transpose(1, 2)?.reshape((b, s, h * d))?)
    }

    fn forward_shared(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        src_cache: &KvCache,
        mask: Option<&Tensor>,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, d) = (self.n_q, self.head_dim);

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.q_norm.forward(&q)?;
        let q = rope.apply(&q.contiguous()?, pos)?;
        let out_dtype = q.dtype();
        let q = self.to_cache_dtype(&q)?;

        let (k, v) = src_cache
            .current_kv()?
            .ok_or_else(|| candle_core::Error::Msg("shared KV source is empty".into()))?;
        // `src_cache` already holds `kv_dtype` (a global setting, so every
        // layer agrees on it) — `q` was cast to match just above, same as
        // `forward`'s own path. See `kv_dtype`'s doc comment.
        let (k, v) = narrow_kv_to_mask(k, v, mask)?;

        let out = self.attend(&q, &k, &v, mask, window, out_dtype)?;
        self.o_proj
            .forward(&out.transpose(1, 2)?.reshape((b, s, h * d))?)
    }
}

// ---------------------------------------------------------------------------
// FFN: dense (E4B) or shared-MLP + routed MoE (26B-A4B)
// ---------------------------------------------------------------------------

/// Routed-expert FFN used by the MoE variant, alongside the dense shared MLP.
/// Mirrors llama.cpp `llama_model_gemma4::graph` (models/gemma4.cpp):
/// - the router operates on `attn_out` (weightless RMS × 1/√hidden × gate_inp_s),
///   NOT on the pre_ffw_norm_2-normed expert input;
/// - softmax gating, top-k, weights normalised over the selected experts;
/// - experts use a merged `gate_up` projection (gate = first n_ff rows), GEGLU,
///   then `down` with an optional per-expert output scale.
struct QGemmaMoe {
    router: QLinear,                   // ffn_gate_inp: hidden -> n_experts
    router_scale: Tensor,              // ffn_gate_inp.scale: (hidden,)
    pre_norm_2: QNorm,                 // pre_ffw_norm_2 (expert input norm)
    gate_up_exps: QExperts,            // [n_expert, 2*n_ff_exp, hidden]
    down_exps: QExperts,               // [n_expert, hidden, n_ff_exp]
    down_exps_scale: Option<Vec<f32>>, // ffn_down_exps.scale: (n_expert,)
    post_norm_1: QNorm,                // post_ffw_norm_1 (after the shared MLP)
    post_norm_2: QNorm,                // post_ffw_norm_2 (after the routed experts)
    n_experts: usize,
    top_k: usize,
    rms_eps: f64,
    hidden: usize,
    /// Where the rest of the model runs — the routed activations arrive here
    /// and the expert outputs are moved back here for the scatter.
    device: Device,
    /// Where the expert matvec runs. `== device` normally; `Device::Cpu` under
    /// `cpuMoe`, so a big MoE fits a small card: the dense half stays on the
    /// accelerator and only `(n_e, hidden)` activations + outputs cross the bus.
    moe_device: Device,
    /// Use `QExperts::matvec_expert` (candle's quantized `QMatMul::forward`
    /// against this expert's bytes) for a single-token decode instead of
    /// `dequantize_expert` + `matmul`. On by default; `GALLIUM_GEMMA4_FUSED=0`
    /// forces the expand path, for the A/B. Not bit-identical — candle's
    /// quantized matmul quantises the activations to 8 bits — so **decode
    /// only** (`n_e == 1`); a multi-row prefill batch keeps the expand path.
    /// Runs on whatever `moe_device` is — CPU under `cpuMoe`, else the model's
    /// own device. It beats the expand path on CUDA too (half the upload, no
    /// f32 expansion, less VRAM); Metal keeps the expand path until it can be
    /// measured on one (see the `fused` computation in `forward`).
    fused: bool,
}

impl QGemmaMoe {
    fn load(
        vb: &QVarBuilder,
        n_experts: usize,
        top_k: usize,
        rms_eps: f64,
        hidden: usize,
        device: &Device,
        moe_device: &Device,
    ) -> Result<Self> {
        let down_exps_scale = if vb.contains("ffn_down_exps.scale") {
            let t = vb.pp("ffn_down_exps").get("scale")?.dequantize(device)?;
            Some(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        } else {
            None
        };
        Ok(Self {
            router: QLinear::from_arc(vb.get("ffn_gate_inp.weight")?, None)?,
            router_scale: vb.pp("ffn_gate_inp").get("scale")?.dequantize(device)?,
            pre_norm_2: QNorm::rms_load(rms_eps, &vb.pp("pre_ffw_norm_2"))?,
            gate_up_exps: vb.get_experts("ffn_gate_up_exps.weight")?,
            down_exps: vb.get_experts("ffn_down_exps.weight")?,
            down_exps_scale,
            post_norm_1: QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm_1"))?,
            post_norm_2: QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm_2"))?,
            n_experts,
            top_k,
            rms_eps,
            hidden,
            device: device.clone(),
            moe_device: moe_device.clone(),
            fused: !matches!(std::env::var("GALLIUM_GEMMA4_FUSED").as_deref(), Ok("0")),
        })
    }

    /// The routed-experts half. `attn_out` is the post-attention residual stream.
    fn forward(&self, attn_out: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = attn_out.dims3()?;
        let num_tokens = b * seq_len;

        // Router logits from attn_out: weightless RMS, × 1/√hidden, ⊙ gate_inp_s.
        let tmp = rms_norm_no_scale(attn_out, self.rms_eps)?;
        let tmp = (tmp * (self.hidden as f64).powf(-0.5))?;
        let tmp = tmp.broadcast_mul(&self.router_scale.to_dtype(tmp.dtype())?)?;
        let logits = self.router.forward(&tmp.reshape((num_tokens, hidden))?)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let probs_vec: Vec<Vec<f32>> = probs.to_dtype(DType::F32)?.to_vec2()?;

        // Expert input: pre_ffw_norm_2(attn_out).
        let xin = self.pre_norm_2.forward(&attn_out.contiguous()?)?;
        let x_flat = xin.reshape((num_tokens, hidden))?;

        // Top-k by softmax prob; combine weights renormalised over the top-k.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for (tok_idx, p) in probs_vec.iter().enumerate() {
            let mut idx: Vec<usize> = (0..self.n_experts).collect();
            idx.sort_by(|&a, &c| p[c].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(self.top_k);
            let total: f32 = idx.iter().map(|&e| p[e]).sum::<f32>().max(6.1035e-5);
            for e in idx {
                expert_tokens[e].push((tok_idx, p[e] / total));
            }
        }

        let active: Vec<(usize, Vec<(usize, f32)>)> = expert_tokens
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();

        let one_expert =
            |(expert_idx, tok_weights): &(usize, Vec<(usize, f32)>)| -> Result<(Vec<usize>, Tensor)> {
                let tok_idxs: Vec<usize> = tok_weights.iter().map(|(t, _)| *t).collect();
                let weights: Vec<f32> = tok_weights.iter().map(|(_, w)| *w).collect();

                // The expert compute happens on `moe_device` — the same device
                // normally, `Device::Cpu` under `cpuMoe`. Only these `(n_e,
                // hidden)` activations and the `(n_e, hidden)` output below
                // cross the bus; a no-op `to_device` when the devices match.
                let batch = Tensor::cat(
                    &tok_idxs
                        .iter()
                        .map(|&i| x_flat.narrow(0, i, 1))
                        .collect::<Result<Vec<_>>>()?,
                    0,
                )?
                .to_device(&self.moe_device)?; // (n_e, hidden)

                // A single-token decode routes one row to this expert, so the
                // projections are matrix-vector products: `matvec_expert`
                // (candle's quantized `QMatMul::forward` against this expert's
                // bytes) never expands the `[d_out, d_in]` weight to f32. On
                // CPU that is the fused MXFP4-analogue path; on **CUDA** it also
                // beats `dequantize_expert` + matmul — half the upload, no f32
                // expansion, ~0.4 GB less VRAM — the same `n_e <= 1` split
                // `lfm2moe_q` already makes on any device. More rows stay on
                // the expand path; candle's quantized matmul drifts there.
                //
                // Metal included as of the M3 measurement below: it was held
                // back only for want of a Mac to run it on, and the exactness
                // argument was always Metal's — `docs/CANDLE_BACKEND.md` §6b's
                // "one row takes the matvec kernel ported from ggml and agrees
                // to four decimals, more than one row drifts" is a Metal
                // measurement, so the `n_e <= 1` split is this device's own
                // rule rather than one borrowed from CUDA.
                let fused = self.fused && tok_idxs.len() == 1;

                // Merged gate_up: rows [0, n_ff) are gate, [n_ff, 2*n_ff) are up.
                let gu = if fused {
                    self.gate_up_exps
                        .matvec_expert(*expert_idx, &batch, &self.moe_device)?
                } else {
                    let gu_w = self
                        .gate_up_exps
                        .dequantize_expert(*expert_idx, &self.moe_device)?; // (2*n_ff, hidden)
                    batch.matmul(&gu_w.t()?.to_dtype(batch.dtype())?)?
                }; // (n_e, 2*n_ff)
                let n_ff = gu.dim(1)? / 2;
                let gate = gu.narrow(1, 0, n_ff)?;
                let up = gu.narrow(1, n_ff, n_ff)?;
                let act = (gate.gelu()? * up)?;

                let mut out = if fused {
                    self.down_exps
                        .matvec_expert(*expert_idx, &act, &self.moe_device)?
                } else {
                    let down_w = self
                        .down_exps
                        .dequantize_expert(*expert_idx, &self.moe_device)?; // (hidden, n_ff)
                    act.matmul(&down_w.t()?.to_dtype(act.dtype())?)?
                }; // (n_e, hidden)
                if let Some(scales) = &self.down_exps_scale {
                    out = (out * scales[*expert_idx] as f64)?;
                }

                let w = Tensor::from_vec(weights, (tok_idxs.len(), 1), &self.moe_device)?
                    .to_dtype(out.dtype())?;
                // Back to the model's device for the scatter into `acc`.
                let weighted = out.broadcast_mul(&w)?.to_device(&self.device)?;
                Ok((tok_idxs, weighted))
            };

        // Fan out across experts with rayon when they run on the CPU (whether
        // or not the rest of the model does) — keyed on `moe_device`.
        let contributions = par_map_on_cpu(&self.moe_device, &active, one_expert)?;

        let mut acc = Tensor::zeros((num_tokens, hidden), attn_out.dtype(), &self.device)?;
        for (tok_idxs, weighted) in contributions {
            let idx = Tensor::from_vec(
                tok_idxs.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                tok_idxs.len(),
                &self.device,
            )?;
            acc = acc.index_add(&idx, &weighted, 0)?;
        }
        let moe = acc.reshape((b, seq_len, hidden))?;
        self.post_norm_2.forward(&moe)
    }
}

// ---------------------------------------------------------------------------
// Block (4-norm + optional PLE + optional MoE + layer_scalar)
// ---------------------------------------------------------------------------

/// Per-layer-embedding sub-branch (E4B; absent on the 26B MoE variant).
struct QGemmaPle {
    inp_gate: QLinear,
    proj: QLinear,
    post_norm: QNorm,
}

struct QGemmaBlock {
    pre_attn_norm: QNorm,
    attn: QAttention,
    post_attn_norm: QNorm,
    pre_ffn_norm: QNorm,
    ffn_gate: QLinear,
    ffn_up: QLinear,
    ffn_down: QLinear,
    post_ffn_norm: QNorm,
    /// Routed experts (26B-A4B). The dense ffn_* above doubles as the shared MLP.
    moe: Option<QGemmaMoe>,
    ple: Option<QGemmaPle>,
    layer_scalar: Tensor,
    // Sharing
    kv_source: Option<usize>,
}

impl QGemmaBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &QVarBuilder,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rms_eps: f64,
        hidden: usize,
        n_experts: usize,
        top_k: usize,
        has_ple: bool,
        device: &Device,
        moe_device: &Device,
        kv_source: Option<usize>,
        kv_f16: bool,
        fused: bool,
        flash: bool,
    ) -> Result<Self> {
        let layer_scalar = vb
            .pp("layer_output_scale")
            .get("weight")?
            .dequantize(device)?;

        let moe = if vb.contains("ffn_gate_inp.weight") {
            Some(QGemmaMoe::load(
                vb, n_experts, top_k, rms_eps, hidden, device, moe_device,
            )?)
        } else {
            None
        };
        let ple = if has_ple {
            Some(QGemmaPle {
                inp_gate: QLinear::load(&vb.pp("inp_gate"))?,
                proj: QLinear::load(&vb.pp("proj"))?,
                post_norm: QNorm::rms_load(rms_eps, &vb.pp("post_norm"))?,
            })
        } else {
            None
        };

        Ok(Self {
            pre_attn_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_norm"))?,
            attn: QAttention::load(vb, n_q, n_kv, head_dim, rms_eps, kv_f16, fused, flash)?,
            post_attn_norm: QNorm::rms_load(rms_eps, &vb.pp("post_attention_norm"))?,
            pre_ffn_norm: QNorm::rms_load(rms_eps, &vb.pp("ffn_norm"))?,
            ffn_gate: QLinear::load(&vb.pp("ffn_gate"))?,
            ffn_up: QLinear::load(&vb.pp("ffn_up"))?,
            ffn_down: QLinear::load(&vb.pp("ffn_down"))?,
            post_ffn_norm: QNorm::rms_load(rms_eps, &vb.pp("post_ffw_norm"))?,
            moe,
            ple,
            layer_scalar,
            kv_source,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        cache: &mut ModelCache,
        layer_idx: usize,
        mask: Option<&Tensor>,
        window: Option<usize>,
        ple_i: Option<&Tensor>, // [b, s, ple_dim] when the model has PLE
    ) -> Result<Tensor> {
        // Attention branch
        let h = self.pre_attn_norm.forward(x)?;
        let h = if let Some(src) = self.kv_source {
            let src_kv = cache.layers[src]
                .as_kv()
                .ok_or_else(|| candle_core::Error::Msg("shared KV source empty".into()))?;
            self.attn
                .forward_shared(&h, rope, pos, src_kv, mask, window)?
        } else {
            let kv = cache.get_kv(layer_idx).expect("layer has KV cache");
            self.attn.forward(&h, rope, pos, kv, mask, window)?
        };
        let h = self.post_attn_norm.forward(&h)?;
        let x = (x + h)?;

        // FFN branch: dense GEGLU (also the MoE variant's shared MLP).
        let h = self.pre_ffn_norm.forward(&x)?;
        let gate = self.ffn_gate.forward(&h)?.gelu()?;
        let up = self.ffn_up.forward(&h)?;
        let mlp = self.ffn_down.forward(&(gate * up)?)?;

        let h = if let Some(moe) = &self.moe {
            // Shared MLP gets its own post-norm; the routed experts run in
            // parallel off the same attn_out and the halves are summed.
            let mlp = moe.post_norm_1.forward(&mlp)?;
            let routed = moe.forward(&x)?;
            (mlp + routed)?
        } else {
            mlp
        };
        let h = self.post_ffn_norm.forward(&h)?;
        let x = (x + h)?;

        // PLE branch (E4B only)
        let x = if let (Some(ple), Some(ple_i)) = (&self.ple, ple_i) {
            let gate = ple.inp_gate.forward(&x)?.gelu()?;
            let h = ple.proj.forward(&(gate * ple_i)?)?;
            let h = ple.post_norm.forward(&h)?;
            (x + h)?
        } else {
            x
        };

        // layer_scalar
        x.broadcast_mul(&self.layer_scalar.to_dtype(x.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Token ids as a host `Vec<u32>` — what a mmap row-gather indexes with.
fn cpu_ids(token_ids: &Tensor) -> Result<Vec<u32>> {
    token_ids
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()
}

/// Model-level PLE tensors (E4B; absent on the 26B MoE variant).
struct QGemmaPleModel {
    /// `[vocab, n_layers·ple_dim]`, left **quantized in the file mmap** and
    /// row-gathered per token (`QExperts::gather_rows`). Dequantized whole it
    /// is ~11 GB of f32 — more than a 12 GB card, and most of a candle load's
    /// wall time — while a forward only reads the rows for the current tokens,
    /// a few MB. Only those are ever expanded, and only they reach the device.
    per_layer_token_embd: QExperts,
    per_layer_model_proj: QLinear,
    per_layer_proj_norm: QNorm,
}

pub struct Gemma4Q {
    /// `[vocab, hidden]`, left **quantized in the file mmap** and row-gathered
    /// per forward (`QExperts::gather_rows`) — a 2-D table is the degenerate
    /// `QExperts` (`vocab` experts of shape `[hidden]`). Dequantized whole it is
    /// ~2.7 GB of device f32; see the note in `load`.
    embed_tokens: QExperts,
    ple: Option<QGemmaPleModel>,
    blocks: Vec<QGemmaBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope_sliding: RoPE,
    rope_global: RoPE,
    cache: ModelCache,
    device: Device,
    hidden_size: usize,
    n_layers: usize,
    ple_dim: usize,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    is_sliding: Vec<bool>, // true = sliding attention, false = global
    /// Narrow K/V to the window span before the scores matmul on sliding layers
    /// (exact — the dropped positions are the ones the mask sets to `-inf`).
    /// `GALLIUM_GEMMA4_KV_NARROW=0` forces the full-context path, for the A/B.
    kv_narrow: bool,
}

impl Gemma4Q {
    pub fn load(
        metadata: &GgufMetadata,
        vb: &QVarBuilder,
        device: &Device,
        moe_device: &Device,
        // `gemma4KvF16` / `GALLIUM_GEMMA4_KV_F16` (issue #305), already
        // resolved by the caller (`env > config`, default on). See
        // `QAttention::kv_dtype`'s doc comment for what this does.
        kv_f16: bool,
    ) -> Result<Self> {
        let prefix = metadata
            .get_str("general.architecture")
            .unwrap_or_else(|_| "gemma4".to_string());

        let n_layers = metadata.get_u32(&format!("{prefix}.block_count"))? as usize;
        let n_q = metadata.get_u32(&format!("{prefix}.attention.head_count"))? as usize;
        // KV-head count: a scalar on E4B, a per-layer array on the 26B MoE
        // variant (sliding layers 8, global layers 2).
        let n_kv_per_layer: Vec<usize> =
            match metadata.get_u32(&format!("{prefix}.attention.head_count_kv")) {
                Ok(v) => vec![v as usize; n_layers],
                Err(_) => metadata
                    .get_i64_array(&format!("{prefix}.attention.head_count_kv"))?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect(),
            };
        let hidden = metadata.get_u32(&format!("{prefix}.embedding_length"))? as usize;
        let ple_dim = metadata
            .get_u32_or(&format!("{prefix}.embedding_length_per_layer_input"), 256)
            as usize;
        // MoE (26B-A4B): routed experts alongside the dense shared MLP.
        let n_experts = metadata.get_u32_or(&format!("{prefix}.expert_count"), 0) as usize;
        let top_k = metadata.get_u32_or(&format!("{prefix}.expert_used_count"), 0) as usize;
        let rms_eps =
            metadata.get_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-6) as f64;
        let sw = metadata.get_u32_or(&format!("{prefix}.attention.sliding_window"), 512) as usize;
        let max_seq = metadata.get_u32_or(&format!("{prefix}.context_length"), 131072) as usize;
        let n_kv_shared =
            metadata.get_u32_or(&format!("{prefix}.attention.shared_kv_layers"), 0) as usize;
        let num_owned = n_layers - n_kv_shared;

        // Global head_dim vs sliding head_dim
        let global_head_dim =
            metadata.get_u32_or(&format!("{prefix}.attention.key_length"), 512) as usize;
        let sliding_head_dim =
            metadata.get_u32_or(&format!("{prefix}.attention.key_length_swa"), 256) as usize;

        // Layer type: true = sliding, false = global (full attention)
        let is_sliding: Vec<bool> = metadata
            .get_bool_array(&format!("{prefix}.attention.sliding_window_pattern"))
            .unwrap_or_else(|_| {
                // fallback: every 6th layer (1-indexed) is global
                (0..n_layers)
                    .map(|i| !((i + 1) % 6 == 0 || i == n_layers - 1))
                    .collect()
            });

        // Sliding RoPE: standard, head_dim=256
        let theta_swa =
            metadata.get_f32_or(&format!("{prefix}.rope.freq_base_swa"), 10000.0) as f64;
        let rope_sliding = RoPE::new(
            &RoPEConfig {
                head_dim: sliding_head_dim,
                max_seq_len: max_seq,
                theta: theta_swa,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        // Global RoPE: proportional. rope_freqs.weight stores per-dim DIVISORS
        // (1.0 for rotated pairs, 1e30 for non-rotated — identity rotations),
        // NOT inv_freq itself. The base inv_freq comes from theta=freq_base and
        // is divided element-wise by the factors.
        let theta_global =
            metadata.get_f32_or(&format!("{prefix}.rope.freq_base"), 1_000_000.0) as f64;
        let rope_global = if vb.contains("rope_freqs.weight") {
            let freqs_t = vb.get("rope_freqs.weight")?.dequantize(device)?;
            let factors: Vec<f32> = freqs_t.to_vec1()?;
            let half = global_head_dim / 2;
            debug_assert_eq!(
                factors.len(),
                half,
                "rope_freqs length must match head_dim/2"
            );
            let inv_freq: Vec<f64> = (0..half)
                .map(|i| {
                    let base = 1.0 / theta_global.powf(2.0 * i as f64 / global_head_dim as f64);
                    base / factors[i] as f64
                })
                .collect();
            RoPE::from_inv_freq(inv_freq, max_seq, DType::F32, device)?
        } else {
            // Fallback: compute proportional inv_freq from config
            let inv_freq = proportional_inv_freq(global_head_dim, 0.25, theta_global);
            RoPE::from_inv_freq(inv_freq, max_seq, DType::F32, device)?
        };

        // Embeddings. `token_embd` stays **quantized in the file mmap** and is
        // row-gathered per forward (`QExperts::gather_rows`), exactly like the
        // PLE table below. Dequantized whole it is ~2.7 GB of device f32 nobody
        // reads more than the current tokens' rows of per call — held for the
        // process lifetime and, stacked on the rest of E4B, the difference
        // between fitting a 12 GB card and not (12B did not fit at all). It was
        // tried before, produced NaN logits on Metal, and was blamed on the
        // backend; the real cause was a use-after-free in the gather itself
        // (`Cow::Owned` into `QStorage::from_data`), fixed 2026-09-03. The
        // gathered rows are bit-identical to a whole-table dequantization —
        // `gemma4_gguf_token_embd_gather_matches_whole_dequantize` checks it.
        let embed_tokens = vb.get_experts("token_embd.weight")?;

        // PLE is an E4B feature; the 26B MoE variant has ple_dim == 0 and no
        // per-layer embedding tensors.
        let has_ple = ple_dim > 0 && vb.contains("per_layer_token_embd.weight");
        let ple = if has_ple {
            Some(QGemmaPleModel {
                per_layer_token_embd: vb.get_experts("per_layer_token_embd.weight")?,
                per_layer_model_proj: QLinear::load(&vb.pp("per_layer_model_proj"))?,
                per_layer_proj_norm: QNorm::rms_load(rms_eps, &vb.pp("per_layer_proj_norm"))?,
            })
        } else {
            None
        };

        // Whether sliding layers are narrowed to the mask's span before the
        // scores matmul (see `kv_narrow`'s own field doc below) — computed
        // here, ahead of the closure below, because a windowed `KvCache`'s
        // own invariant depends on it: `narrow_kv_to_mask` is a no-op once its
        // input is already no wider than the mask (`kv_len >= total`), so a
        // windowed cache's short K/V pass straight through when `kv_narrow`
        // is on. With it off, `build_sliding_window_mask` (not narrowed)
        // stays full-width against a K that is now short — a shape mismatch
        // at the `broadcast_add`, not a graceful fallback. So windowing is
        // built only when `kv_narrow` is on; `GALLIUM_GEMMA4_KV_NARROW=0`
        // (the existing A/B escape hatch) falls every sliding layer back to
        // the plain, unwindowed cache instead of crashing on it. See issue
        // #304.
        let kv_narrow = !matches!(
            std::env::var("GALLIUM_GEMMA4_KV_NARROW").as_deref(),
            Ok("0")
        );
        if !kv_narrow {
            tracing::warn!(
                "GALLIUM_GEMMA4_KV_NARROW=0: sliding-layer KV caches are NOT windowed \
                 (issue #304) — every sliding layer retains the whole conversation, same \
                 as before this change"
            );
        }
        // Fused attention through candle's Metal `sdpa` kernel — the issue
        // #308 experiment, `GALLIUM_GEMMA4_SDPA=1`, off by default. Three
        // gates, each refused with a warning rather than silently taking
        // the matmul path: Metal (the kernel has no CPU or CUDA impl here),
        // `kv_narrow` (the decode path drops the mask, which is only sound
        // when a sliding layer's K is already narrowed to its window), and
        // `kv_f16` (the full kernel refuses f32 at head_dim 512, the global
        // layers' width). See `QAttention::fused_attention`.
        let fused = matches!(std::env::var("GALLIUM_GEMMA4_SDPA").as_deref(), Ok("1"));
        let fused = if !fused {
            false
        } else if !device.is_metal() {
            tracing::warn!("GALLIUM_GEMMA4_SDPA=1 ignored: fused sdpa is Metal-only (issue #308)");
            false
        } else if !kv_narrow {
            tracing::warn!(
                "GALLIUM_GEMMA4_SDPA=1 ignored: needs windowed sliding layers \
                 (GALLIUM_GEMMA4_KV_NARROW=0 is set)"
            );
            false
        } else if !kv_f16 {
            tracing::warn!("GALLIUM_GEMMA4_SDPA=1 ignored: needs the f16 KV cache (gemma4KvF16)");
            false
        } else {
            tracing::info!("GALLIUM_GEMMA4_SDPA=1: attention through Metal sdpa (issue #308)");
            true
        };
        // Fused *prefill* attention (all layers) through `candle-flash-attn`
        // — the issue #308 experiment. On by default once the build and the
        // runtime can actually take it: the `flash-attn` cargo feature (a
        // build-time opt-in kept separate from `cuda` — see that feature's
        // own comment in gallium-models/Cargo.toml), CUDA (the crate has no
        // CPU or Metal impl), and `kv_f16` (the kernel only takes f16/bf16
        // Q/K/V). `GALLIUM_GEMMA4_FLASH_ATTN=0` opts back out unconditionally;
        // `=1` is honored the same as the default but warns instead of
        // silently falling back when a gate isn't met, since asking
        // explicitly and getting the matmul path anyway should say why.
        // Decode always stays on the matmul path regardless of these gates;
        // see `QAttention::flash_attention`'s doc comment for why, and for
        // the head_dim-512 correctness history (issue #313, fixed by the
        // candle `0.11.0` bump).
        let requested = std::env::var("GALLIUM_GEMMA4_FLASH_ATTN").ok();
        let explicit_off = matches!(requested.as_deref(), Some("0") | Some("false"));
        let explicit_on = matches!(requested.as_deref(), Some("1") | Some("true"));
        let flash = if explicit_off {
            false
        } else if !cfg!(feature = "flash-attn") {
            if explicit_on {
                tracing::warn!(
                    "GALLIUM_GEMMA4_FLASH_ATTN=1 ignored: built without the `flash-attn` cargo \
                     feature (candle-flash-attn not compiled in, issue #308)"
                );
            }
            false
        } else if !device.is_cuda() {
            if explicit_on {
                tracing::warn!(
                    "GALLIUM_GEMMA4_FLASH_ATTN=1 ignored: candle-flash-attn is CUDA-only \
                     (issue #308)"
                );
            }
            false
        } else if !kv_f16 {
            if explicit_on {
                tracing::warn!(
                    "GALLIUM_GEMMA4_FLASH_ATTN=1 ignored: needs the f16 KV cache (gemma4KvF16)"
                );
            }
            false
        } else {
            tracing::info!(
                "prefill attention through candle-flash-attn (issue #308); \
                 GALLIUM_GEMMA4_FLASH_ATTN=0 to opt out"
            );
            true
        };
        // Spare capacity a windowed cache keeps beyond the window itself, so
        // most appends stay on the cheap `slice_set` path instead of paying
        // a compaction on every single one — see `KvCache::windowed`'s own
        // doc comment. Matches `model.rs`'s default prefill-chunk size: a
        // whole chunk can arrive in one append.
        const KV_WINDOW_HEADROOM: usize = 512;

        // Blocks
        let mut cache_layers: Vec<LayerCache> = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let sliding = *is_sliding.get(i).unwrap_or(&true);
                let head_dim = if sliding {
                    sliding_head_dim
                } else {
                    global_head_dim
                };
                let n_kv = *n_kv_per_layer.get(i).unwrap_or(&1);

                let kv_source = if i >= num_owned && n_kv_shared > 0 {
                    // All shared layers of the same type → last owned layer of that type
                    let source = (0..num_owned)
                        .filter(|&j| is_sliding.get(j).copied().unwrap_or(true) == sliding)
                        .last()
                        .unwrap_or(0);
                    cache_layers.push(LayerCache::Shared {
                        source_layer: source,
                    });
                    Some(source)
                } else if sliding && kv_narrow {
                    // The model never reads past `sw` positions back from a
                    // sliding layer (`narrow_kv_to_mask`, called with the
                    // narrowed mask's span) — retaining the whole
                    // conversation here is pure waste. See issue #304.
                    cache_layers.push(LayerCache::Kv(KvCache::windowed(
                        sw,
                        KV_WINDOW_HEADROOM,
                        max_seq,
                    )));
                    None
                } else {
                    cache_layers.push(LayerCache::Kv(KvCache::new(max_seq)));
                    None
                };

                QGemmaBlock::load(
                    &vb.pp(format!("blk.{i}")),
                    n_q,
                    n_kv,
                    head_dim,
                    rms_eps,
                    hidden,
                    n_experts,
                    top_k,
                    has_ple,
                    device,
                    moe_device,
                    kv_source,
                    kv_f16,
                    fused,
                    flash,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
            QLinear::from_arc(vb.get("token_embd.weight")?, None)?
        };

        let final_logit_softcapping =
            Some(metadata.get_f32_or(&format!("{prefix}.final_logit_softcapping"), 30.0) as f64);

        Ok(Self {
            embed_tokens,
            ple,
            blocks,
            final_norm,
            lm_head,
            rope_sliding,
            rope_global,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            hidden_size: hidden,
            n_layers,
            ple_dim,
            sliding_window: sw,
            final_logit_softcapping,
            is_sliding,
            kv_narrow,
        })
    }

    /// Main embeddings scaled by `sqrt(hidden)` — the first third of `forward`,
    /// split out so the multimodal wrapper ([`crate::gemma4_vision`]) can
    /// inject vision features between the embedding and the PLE.
    pub fn embed_scaled(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        // Gather just this call's rows from the mmap-resident quantized table
        // and move only those onto the compute device — a prefill pulls a few
        // MB, a decode step one row. Bit-identical to indexing a whole-table
        // dequantization (block dequant is per-block, rows are whole blocks).
        let rows = self
            .embed_tokens
            .gather_rows(&cpu_ids(token_ids)?, &self.device)?;
        rows.reshape((b, s, self.hidden_size))? * (self.hidden_size as f64).sqrt()
    }

    /// Per-layer inputs for the block stack, `None` on a variant without PLE
    /// (26B-A4B). Same contract as the safetensors `Gemma4::compute_ple`:
    /// `token_ids` feeds the lookup half, `h_embed` the projection half — the
    /// multimodal caller passes *masked* ids and *merged* embeddings, which is
    /// the whole reason the two are separate arguments.
    pub fn compute_ple_opt(&self, token_ids: &Tensor, h_embed: &Tensor) -> Result<Option<Tensor>> {
        match &self.ple {
            Some(ple) => Ok(Some(self.compute_ple(ple, token_ids, h_embed)?)),
            None => Ok(None),
        }
    }

    /// Run the block stack, final norm and head on pre-computed embeddings —
    /// the tail of `forward`, split out for the multimodal wrapper.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        per_layer: Option<&Tensor>,
        pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let mut h = inputs_embeds.clone();

        for (i, block) in self.blocks.iter().enumerate() {
            let sliding = *self.is_sliding.get(i).unwrap_or(&true);
            let rope = if sliding {
                &self.rope_sliding
            } else {
                &self.rope_global
            };

            // Sliding layers need their mask at decode as well — see the long
            // comment on the same decision in `gemma4.rs`, and
            // `attention_mask_needed` (gallium_core::mask, docs/TODO.md §1.1),
            // the shared tested spelling used here. In short: `KvCache` is never
            // truncated to the window, so an unmasked single-token query attends
            // to the entire history and a long session silently drifts outside
            // what the layer was trained for. The *narrowed* builder also cuts
            // the key axis to the window span, and the attention layers slice
            // K/V to `mask.dim(1)` to match — so the scores matmul is
            // window-wide, not whole-context-wide, on the 40 of 48 sliding
            // layers a long turn spends most of its time in. Below the window
            // the decision returns `false` and the K/V is left full-width, which
            // is the same span the narrowed mask would have kept there anyway.
            let window = sliding.then_some(self.sliding_window);
            let mask = if !attention_mask_needed(seq_len, pos, window) {
                None
            } else if !sliding {
                Some(build_causal_mask(seq_len, pos, &self.device)?)
            } else if self.kv_narrow {
                Some(build_sliding_window_mask_narrowed(
                    seq_len,
                    pos,
                    self.sliding_window,
                    &self.device,
                )?)
            } else {
                Some(build_sliding_window_mask(
                    seq_len,
                    pos,
                    self.sliding_window,
                    &self.device,
                )?)
            };

            let ple_i = match per_layer {
                Some(pl) => Some(pl.narrow(2, i, 1)?.squeeze(2)?),
                None => None,
            };

            h = block.forward(
                &h,
                rope,
                pos,
                &mut self.cache,
                i,
                mask.as_ref(),
                window,
                ple_i.as_ref(),
            )?;
        }

        let h = self.final_norm.forward(&h)?;
        let mut logits = self
            .lm_head
            .forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;

        if let Some(cap) = self.final_logit_softcapping {
            logits = ((logits * (1.0 / cap))?.tanh()? * cap)?;
        }

        logits.to_dtype(DType::F32)
    }

    /// Compute per-layer inputs [b, s, n_layers, ple_dim] (E4B PLE only).
    fn compute_ple(
        &self,
        ple: &QGemmaPleModel,
        token_ids: &Tensor,
        h_embed: &Tensor,
    ) -> Result<Tensor> {
        let (b, s) = token_ids.dims2()?;
        let (n, d) = (self.n_layers, self.ple_dim);

        // Token-level per-layer embeddings, scaled by sqrt(ple_dim). The table
        // stays quantized in the file mmap; dequantize this call's rows only
        // and move just those onto the compute device.
        let ple_tok = ple
            .per_layer_token_embd
            .gather_rows(&cpu_ids(token_ids)?, &self.device)?;
        let ple_tok = (ple_tok * (d as f64).sqrt())?;
        let ple_tok = ple_tok.reshape((b, s, n, d))?;

        // Projection of main embeddings, scaled by 1/sqrt(hidden)
        let proj =
            (ple.per_layer_model_proj.forward(h_embed)? * (self.hidden_size as f64).powf(-0.5))?;
        let proj = proj.reshape((b, s, n, d))?;
        let proj = ple.per_layer_proj_norm.forward(&proj)?;

        // Combine
        (ple_tok + proj)? * 2.0_f64.powf(-0.5)
    }
}

impl CausalLM for Gemma4Q {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = token_ids.dims2()?;
        let h_embed = self.embed_scaled(token_ids)?;
        let per_layer = self.compute_ple_opt(token_ids, &h_embed)?;
        self.forward_embeds(&h_embed, per_layer.as_ref(), pos, seq_len)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    /// Opted into cross-call reuse: this model's cache is the standard
    /// `ModelCache`, so `generate_reusing` can roll it back and evaluate only
    /// what a new prompt adds. See [`gallium_core::CausalLM::cache`].
    fn cache(&mut self) -> Option<&mut ModelCache> {
        Some(&mut self.cache)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}
