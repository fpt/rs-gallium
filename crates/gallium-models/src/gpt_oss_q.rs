//! Quantized GPT-OSS model loaded from GGUF.
//!
//! GGUF uses a different tensor naming convention than safetensors:
//!   blk.{i}.attn_q.weight   vs  model.layers.{i}.self_attn.q_proj.weight
//!   token_embd.weight       vs  model.embed_tokens.weight

use candle_core::{DType, Device, Module, Result, Tensor, D};

use gallium_core::quantized::{GgufMetadata, QExperts, QLinear, QNorm, QVarBuilder, Tq2Tensor};
use gallium_core::*;

/// Token ids as a host `Vec<u32>` — what a mmap row-gather indexes with.
/// Identical to `gemma4_q.rs`'s private helper of the same name; not shared
/// because the two crates' modules don't have a natural common home for five
/// lines, and duplicating them costs less than the indirection would.
fn cpu_ids(token_ids: &Tensor) -> Result<Vec<u32>> {
    token_ids
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()
}

// -- Quantized Attention (uses QLinear) --------------------------------------

struct QAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    /// Per-head sink logit appended to attention scores before softmax.
    sinks: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QAttention {
    fn load(
        vb: &QVarBuilder,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let q_proj = QLinear::load(&vb.pp("attn_q"))?;
        let k_proj = QLinear::load(&vb.pp("attn_k"))?;
        let v_proj = QLinear::load(&vb.pp("attn_v"))?;
        let o_proj = QLinear::load(&vb.pp("attn_output"))?;
        let sinks = vb.get("attn_sinks.weight")?.dequantize(vb.device())?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sinks,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let h = self.num_q_heads;
        let h_kv = self.num_kv_heads;
        let d = self.head_dim;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq_len, h, d))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq_len, h_kv, d))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq_len, h_kv, d))?
            .transpose(1, 2)?;

        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;
        let (k, v) = narrow_kv_to_mask(k, v, mask)?;

        // K/V stay at h_kv heads; `gqa_scores` groups Q instead of expanding them.
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (gqa_scores(&q, &k)? * scale)?;

        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        }

        // Attention sinks: append per-head sink logit, softmax over seq+1, drop last col.
        let total_len = scores.dim(D::Minus1)?;
        let s = self
            .sinks
            .reshape((1, h, 1, 1))?
            .expand((b, h, seq_len, 1))?
            .contiguous()?;
        let combined = Tensor::cat(&[&scores, &s], D::Minus1)?;
        let probs = candle_nn::ops::softmax_last_dim(&combined)?;
        let attn_weights = probs.narrow(D::Minus1, 0, total_len)?;

        let attn_out = gqa_weighted_sum(&attn_weights, &v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;
        self.o_proj.forward(&attn_out)
    }
}

// -- Quantized MoE -----------------------------------------------------------
//
// GGUF stores expert weights as merged 3D MXFP4 tensors:
//   ffn_gate_exps.weight: [n_expert, n_ff, n_embd]
//   ffn_up_exps.weight:   [n_expert, n_ff, n_embd]
//   ffn_down_exps.weight: [n_expert, n_embd, n_ff]
//
// We store raw bytes and dequantize one expert at a time during forward.

struct QMoEFFN {
    /// Raw MXFP4 expert weights — dequantized lazily per expert during forward.
    gate_exps: Tq2Tensor, // dims: [n_expert, n_ff, n_embd]
    up_exps: Tq2Tensor,   // dims: [n_expert, n_ff, n_embd]
    down_exps: Tq2Tensor, // dims: [n_expert, n_embd, n_ff]
    /// Per-expert biases: shape [n_expert, n_ff] or [n_expert, n_embd].
    gate_bias: Tensor, // [n_expert, n_ff]
    up_bias: Tensor,      // [n_expert, n_ff]
    down_bias: Tensor,    // [n_expert, n_embd]
    router: QLinear,
    num_experts_per_tok: usize,
    clamp: Option<f32>,
    /// Where the rest of the model runs — expert outputs are moved back here.
    device: Device,
    /// Where the expert matvec runs. `== device` normally; `Device::Cpu` under
    /// `cpuMoe`, so the ~60 GB of 120B experts (or 13 GB of 20B) stay in host
    /// RAM and only `(n_e, hidden)` activations + outputs cross the bus.
    moe_device: Device,
    /// CPU SIMD kernels for the fused MXFP4 matvec path below.
    kernels: KernelSet,
    /// Use `Tq2Tensor::matvec_expert` (stream the MXFP4 bytes, never expand
    /// the `[d_out, d_in]` weight to f32) for a single-token decode instead of
    /// `dequantize_expert` + `matmul`. On by default; `GALLIUM_GPT_OSS_FUSED_MXFP4=0`
    /// forces the expand path, for the A/B testsuite comparison. Not
    /// bit-identical — the reduction order differs — so decode only, and only
    /// when `moe_device` is CPU (the expand path uploads f32 to an accelerator).
    fused_mxfp4: bool,
    /// Use `Tq2Tensor::matmul_expert` (decode the expert a block of rows at a
    /// time and multiply each block against every routed token) when more
    /// than one token routes to an expert — a prefill step — instead of
    /// expanding the whole expert to f32 for `matmul` (issue #339). On by
    /// default; `GALLIUM_GPT_OSS_FUSED_PREFILL=0` forces the expand path, for
    /// the A/B. CPU `moe_device` only, like `fused_mxfp4`.
    fused_prefill: bool,
    /// `(gate, up, down)` biases on the model's device, copied over the first
    /// time `resident_forward` needs them — a resident expert would otherwise
    /// copy three small slices from the host per token, 288 transfers per
    /// decoded token on 20B. Tens to a couple of hundred MB; only made when an
    /// expert cache is attached.
    device_biases: std::sync::OnceLock<(Tensor, Tensor, Tensor)>,
}

impl QMoEFFN {
    fn load(
        vb: &QVarBuilder,
        num_experts: usize,
        num_experts_per_tok: usize,
        clamp: Option<f32>,
        moe_device: &Device,
    ) -> Result<Self> {
        // Load merged TQ2_0 expert tensors as raw bytes for lazy per-expert dequant.
        let gate_exps = vb.get_tq2("ffn_gate_exps.weight")?;
        let up_exps = vb.get_tq2("ffn_up_exps.weight")?;
        let down_exps = vb.get_tq2("ffn_down_exps.weight")?;
        // Expert FFN biases: shape [n_expert, n_ff] or [n_expert, n_embd] after
        // dim reversal. On `moe_device` — they are added to the expert output.
        let gate_bias = vb.get("ffn_gate_exps.bias")?.dequantize(moe_device)?;
        let up_bias = vb.get("ffn_up_exps.bias")?.dequantize(moe_device)?;
        let down_bias = vb.get("ffn_down_exps.bias")?.dequantize(moe_device)?;
        let router = QLinear::load(&vb.pp("ffn_gate_inp"))?;
        let _ = num_experts; // used only to verify dims at load time if needed
        Ok(Self {
            gate_exps,
            up_exps,
            down_exps,
            gate_bias,
            up_bias,
            down_bias,
            router,
            num_experts_per_tok,
            clamp,
            device: vb.device().clone(),
            moe_device: moe_device.clone(),
            kernels: KernelSet::detect(),
            fused_mxfp4: !matches!(
                std::env::var("GALLIUM_GPT_OSS_FUSED_MXFP4").as_deref(),
                Ok("0")
            ),
            fused_prefill: !matches!(
                std::env::var("GALLIUM_GPT_OSS_FUSED_PREFILL").as_deref(),
                Ok("0")
            ),
            device_biases: std::sync::OnceLock::new(),
        })
    }

    /// One decode expert computed on the accelerator from its device-resident
    /// MXFP4 bytes (`Tq2Tensor::resident_expert`, `gallium_core::mxfp4_matvec`)
    /// instead of on `moe_device` from the mmap. `None` — compute it the usual
    /// way — unless an expert cache is attached, the model is on an
    /// accelerator, exactly one token routes here (a prefill's many-row step
    /// stays on the blocked CPU kernel, #339/#340), and the expert — all three
    /// matrices, one cache entry — is resident or has room to become so.
    ///
    /// Same arithmetic as the host path below — GLU with the clamp, `(up + 1)`,
    /// biases, routing weight — but not bit-identical to it: the GPU kernel
    /// reduces in a different order, as the CPU fused matvec already does
    /// against expand-then-`matmul`.
    fn resident_forward(
        &self,
        (expert_idx, tok_weights): &(usize, Vec<(usize, f32)>),
        x_flat: &Tensor,
    ) -> Result<Option<(Vec<usize>, Tensor)>> {
        if tok_weights.len() != 1 || self.device.is_cpu() || !self.gate_exps.has_cache() {
            return Ok(None);
        }
        let idx = *expert_idx;
        let Some(parts) = gallium_core::quantized::resident_expert(
            &[&self.gate_exps, &self.up_exps, &self.down_exps],
            idx,
            &self.device,
        )?
        else {
            return Ok(None);
        };
        let [gw, uw, dw] = &parts[..] else {
            candle_core::bail!("resident_expert: expected gate, up and down");
        };
        let (tok, weight) = tok_weights[0];
        let (n_ff, hidden) = (self.gate_exps.dims[1], self.gate_exps.dims[2]);
        let (gb, ub, db) = match self.device_biases.get() {
            Some(b) => b,
            None => {
                let b = (
                    self.gate_bias.to_device(&self.device)?,
                    self.up_bias.to_device(&self.device)?,
                    self.down_bias.to_device(&self.device)?,
                );
                self.device_biases.get_or_init(|| b)
            }
        };

        let x = x_flat.narrow(0, tok, 1)?;
        let gate_raw = mxfp4_matvec(gw, &x, n_ff, hidden)?.broadcast_add(&gb.narrow(0, idx, 1)?)?;
        let up_raw = mxfp4_matvec(uw, &x, n_ff, hidden)?.broadcast_add(&ub.narrow(0, idx, 1)?)?;
        let inter = self.glu(gate_raw, up_raw)?;
        let out = mxfp4_matvec(dw, &inter, hidden, n_ff)?.broadcast_add(&db.narrow(0, idx, 1)?)?;
        Ok(Some((
            vec![tok],
            (out * weight as f64)?.to_dtype(x_flat.dtype())?,
        )))
    }

    /// GPT-OSS's gated activation: `gate` clamped above and `up` both ways at
    /// `clamp`, then `gate · σ(1.702 · gate) · (up + 1)`.
    fn glu(&self, gate_raw: Tensor, up_raw: Tensor) -> Result<Tensor> {
        let gate = if let Some(limit) = self.clamp {
            gate_raw.clamp(-1e38_f64, limit as f64)?
        } else {
            gate_raw
        };
        let sig = ((&gate * 0.851_f64)?.tanh()? + 1.0_f64)? * 0.5_f64;
        let glu = (gate * sig)?;
        let up = if let Some(limit) = self.clamp {
            up_raw.clamp(-(limit as f64), limit as f64)?
        } else {
            up_raw
        };
        glu * (up + 1.0_f64)?
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b * seq_len, hidden))?;
        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;
        let router_probs_vec: Vec<Vec<f32>> = router_probs.to_vec2()?;
        let num_tokens = b * seq_len;
        let n_experts = self.gate_exps.dims[0];

        // Build routing table: for each expert, which tokens route to it and with what weight.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        for tok_idx in 0..num_tokens {
            let probs = &router_probs_vec[tok_idx];
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(self.num_experts_per_tok);
            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            for (expert_idx, weight) in indexed {
                expert_tokens[expert_idx].push((tok_idx, weight / total));
            }
        }

        // For each active expert: gather its tokens, dequantize once, run batched matmul.
        let active: Vec<(usize, Vec<(usize, f32)>)> = expert_tokens
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .collect();

        let one_expert =
            |(expert_idx, tok_weights): &(usize, Vec<(usize, f32)>)| -> Result<(Vec<usize>, Tensor)> {
                let tok_idxs: Vec<usize> = tok_weights.iter().map(|(t, _)| *t).collect();
                let weights: Vec<f32> = tok_weights.iter().map(|(_, w)| *w).collect();

                // Gather all tokens routed to this expert → (n_e, hidden), on
                // `moe_device` (the model's device normally, `Device::Cpu`
                // under `cpuMoe`). Only this and the `(n_e, hidden)` output
                // cross the bus; `to_device` is a no-op when they match.
                let batch = Tensor::cat(
                    &tok_idxs
                        .iter()
                        .map(|&i| x_flat.narrow(0, i, 1))
                        .collect::<Result<Vec<_>>>()?,
                    0,
                )?
                .to_device(&self.moe_device)?;

                let gb = self.gate_bias.narrow(0, *expert_idx, 1)?; // (1, n_ff)
                let ub = self.up_bias.narrow(0, *expert_idx, 1)?;
                let db = self.down_bias.narrow(0, *expert_idx, 1)?;

                // Never expand the ~33 MB f32 weight on the CPU: a single-token
                // decode routes exactly one row here and takes the fused matvec
                // (`fused_mxfp4`); a prefill step routes several and takes the
                // blocked `matmul_expert` (`fused_prefill`). Both are A/B
                // switches back to expand-then-`matmul`, the path an
                // accelerator `moe_device` always takes.
                let n_e = tok_idxs.len();
                let fused = self.moe_device.is_cpu()
                    && if n_e == 1 { self.fused_mxfp4 } else { self.fused_prefill };
                let project = |w: &Tq2Tensor, x: &[f32]| -> Result<Vec<f32>> {
                    if n_e == 1 {
                        w.matvec_expert(*expert_idx, x, &self.kernels)
                    } else {
                        w.matmul_expert(*expert_idx, x, n_e, &self.kernels)
                    }
                };

                // gate/up input projections: (n_e, hidden) → (n_e, n_ff).
                let (gate_raw, up_raw) = if fused {
                    let xrows = batch.flatten_all()?.to_vec1::<f32>()?;
                    let g = project(&self.gate_exps, &xrows)?;
                    let u = project(&self.up_exps, &xrows)?;
                    let n_ff = g.len() / n_e;
                    (
                        Tensor::from_vec(g, (n_e, n_ff), &self.moe_device)?.broadcast_add(&gb)?,
                        Tensor::from_vec(u, (n_e, n_ff), &self.moe_device)?.broadcast_add(&ub)?,
                    )
                } else {
                    // Dequantize this expert's weights once for the entire batch.
                    let gate_w = self.gate_exps.dequantize_expert(*expert_idx, &self.moe_device)?;
                    let up_w = self.up_exps.dequantize_expert(*expert_idx, &self.moe_device)?;
                    (
                        batch.matmul(&gate_w.t()?)?.broadcast_add(&gb)?,
                        batch.matmul(&up_w.t()?)?.broadcast_add(&ub)?,
                    )
                };

                // GLU · (up + 1), then the down projection: (n_e, n_ff) → (n_e, hidden).
                let inter = self.glu(gate_raw, up_raw)?;
                let expert_out = if fused {
                    let iv = inter.flatten_all()?.to_vec1::<f32>()?;
                    let o = project(&self.down_exps, &iv)?;
                    let hid = o.len() / n_e;
                    Tensor::from_vec(o, (n_e, hid), &self.moe_device)?.broadcast_add(&db)?
                } else {
                    let down_w = self.down_exps.dequantize_expert(*expert_idx, &self.moe_device)?;
                    inter.matmul(&down_w.t()?)?.broadcast_add(&db)?
                };

                // Scale by the routing weight, then back to the model's device.
                let w_col = Tensor::from_slice(&weights, (weights.len(), 1), &self.moe_device)?;
                let weighted = expert_out.broadcast_mul(&w_col)?.to_device(&self.device)?;
                Ok((tok_idxs, weighted))
            };

        // Experts whose weights are resident on the accelerator (issue #343
        // stage 4) run there, on this thread; the rest fan out across experts
        // with rayon when they run on the CPU — keyed on `moe_device`, so
        // `cpuMoe` gets the fan-out even on an accelerator.
        let mut contributions = Vec::with_capacity(active.len());
        let mut host_side = Vec::with_capacity(active.len());
        for expert in active {
            match self.resident_forward(&expert, &x_flat)? {
                Some(c) => contributions.push(c),
                None => host_side.push(expert),
            }
        }
        contributions.extend(par_map_on_cpu(&self.moe_device, &host_side, one_expert)?);

        // Scatter: accumulate weighted expert outputs into per-token slots.
        let mut out_rows: Vec<Option<Tensor>> = (0..num_tokens).map(|_| None).collect();
        for (tok_idxs, weighted) in contributions {
            for (local_i, global_t) in tok_idxs.iter().enumerate() {
                let row = weighted.narrow(0, local_i, 1)?;
                out_rows[*global_t] = Some(match out_rows[*global_t].take() {
                    None => row,
                    Some(prev) => (prev + row)?,
                });
            }
        }

        let output_rows: Vec<Tensor> = out_rows
            .into_iter()
            .map(|t| t.expect("every token has at least one active expert"))
            .collect();

        Tensor::cat(&output_rows, 0)?.reshape((b, seq_len, hidden))
    }
}

// -- Quantized Transformer Block ---------------------------------------------

struct QTransformerBlock {
    pre_attn_norm: QNorm,
    attn: QAttention,
    post_attn_norm: QNorm,
    ffn: QMoEFFN,
}

impl QTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let h_in = self.pre_attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&h_in, rope, pos, kv_cache, mask)?;
        let h = (attn_out + x)?;
        let residual = &h;
        let h = self.post_attn_norm.forward(&h)?;
        let h = self.ffn.forward(&h)?;
        h + residual
    }
}

// -- Full Quantized GPT-OSS Model --------------------------------------------

pub struct GptOssQ {
    /// `[vocab, hidden]`, left **quantized in the file mmap** and row-gathered
    /// per forward (`QExperts::gather_rows`) — a 2-D table is the degenerate
    /// `QExperts` (`vocab` experts of shape `[hidden]`), same as `gemma4_q.rs`'s
    /// `embed_tokens` (issue #255, mirroring #252). Dequantized whole it was
    /// ~2.3 GB of device f32 (Q5_0, `[2880, 201088]` on the 20B GGUF) nobody
    /// read more than the current tokens' rows of per call, held for the
    /// process lifetime — low-urgency (the 20B GGUF already fit a 12 GB card
    /// either way, ~4.0 GiB peak), but it was the last dequantize-whole in
    /// this file. Bit-identical to a whole-table dequantization — block
    /// dequant is per-block and rows are whole blocks — checked by
    /// `gpt_oss_gguf_token_embd_gather_matches_whole_dequantize`.
    embed_tokens: QExperts,
    blocks: Vec<QTransformerBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
    sliding_window: usize,
    layer_types: Vec<String>, // "full_attention" or "sliding_attention"
    /// Narrow K/V to the window span before the scores matmul on sliding
    /// layers — same switch and same reasoning as `gpt_oss.rs` (issue #232);
    /// `GALLIUM_GPT_OSS_KV_NARROW=0` forces the full-context path, for the A/B.
    kv_narrow: bool,
    /// What [`CausalLM::transient_bytes`] estimates from.
    transient: TransientDims,
}

/// The dimensions a forward's peak scratch scales with (issue #343).
struct TransientDims {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_embd: usize,
    /// One expert's FFN width.
    n_ff: usize,
    /// The routed experts compute on the model's device (no `cpuMoe`), where a
    /// prefill expands them to f32 one at a time.
    experts_on_device: bool,
}

impl GptOssQ {
    /// Load from GGUF file.
    pub fn load(
        metadata: &GgufMetadata,
        vb: &QVarBuilder,
        device: &Device,
        moe_device: &Device,
    ) -> Result<Self> {
        // Extract config from GGUF metadata
        // GPT-OSS uses "gpt_oss" arch prefix in GGUF
        let arch = metadata
            .get_str("general.architecture")
            .unwrap_or_else(|_| "llama".to_string());
        let prefix = &arch;

        let n_layers = metadata.get_u32(&format!("{prefix}.block_count"))? as usize;
        let n_heads = metadata.get_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let n_kv_heads = metadata.get_u32(&format!("{prefix}.attention.head_count_kv"))? as usize;
        let n_embd = metadata.get_u32(&format!("{prefix}.embedding_length"))? as usize;
        // GPT-OSS uses head_dim=64, which differs from n_embd/n_heads (=45).
        // Prefer the explicit key_length field; fall back to n_embd/n_heads.
        let head_dim = metadata.get_u32_or(
            &format!("{prefix}.attention.key_length"),
            (n_embd / n_heads) as u32,
        ) as usize;
        let rope_freq_base = metadata.get_f32_or(&format!("{prefix}.rope.freq_base"), 150000.0);
        let rope_scaling_factor =
            metadata.get_f32_or(&format!("{prefix}.rope.scaling.factor"), 1.0);
        let rope_orig_ctx = metadata.get_u32_or(
            &format!("{prefix}.rope.scaling.original_context_length"),
            4096,
        ) as usize;
        let rms_eps =
            metadata.get_f32_or(&format!("{prefix}.attention.layer_norm_rms_epsilon"), 1e-5) as f64;
        let n_experts = metadata.get_u32_or(&format!("{prefix}.expert_count"), 32) as usize;
        let n_experts_used =
            metadata.get_u32_or(&format!("{prefix}.expert_used_count"), 4) as usize;
        let sliding_window =
            metadata.get_u32_or(&format!("{prefix}.attention.sliding_window"), 128) as usize;
        let max_seq_len = metadata.get_u32_or(&format!("{prefix}.context_length"), 131072) as usize;
        let swiglu_limit = metadata.get_f32_or(&format!("{prefix}.swiglu_limit"), 7.0);

        // Layer types from metadata (or default alternating).
        // HF transformers: `"sliding_attention" if bool((i+1)%2) else "full_attention"`.
        // Ollama (model/models/gptoss/model.go:38–39 "// Even layers are sliding window
        // attention." + `SetLayerType(i % 2)` with SWA cache at index 0) agrees:
        //   i=0 -> sliding, i=1 -> full, i=2 -> sliding, ...
        // The GGUF has no `attention.layer_type` array for this arch, so we always hit
        // this fallback — getting it wrong silently swaps every layer's mask.
        let layer_types: Vec<String> = metadata
            .get_str_array(&format!("{prefix}.attention.layer_type"))
            .unwrap_or_else(|_| {
                (0..n_layers)
                    .map(|i| {
                        if i % 2 == 0 {
                            "sliding_attention".to_string()
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect()
            });

        // RoPE with YaRN scaling if specified
        let rope_scaling = if rope_scaling_factor > 1.0 {
            RoPEScaling::YaRN {
                factor: rope_scaling_factor as f64,
                original_max_position_embeddings: rope_orig_ctx,
                beta_fast: 32.0,
                beta_slow: 1.0,
            }
        } else {
            RoPEScaling::None
        };
        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len,
                theta: rope_freq_base as f64,
                scaling: rope_scaling,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        // Embeddings. `token_embd` stays **quantized in the file mmap** and is
        // row-gathered per forward (`QExperts::gather_rows`) instead of
        // dequantized whole — see the field doc comment on `embed_tokens`.
        let embed_tokens = vb.get_experts("token_embd.weight")?;

        // Layers
        let mut cache_layers = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let bvb = vb.pp(format!("blk.{i}"));
                // K/V are cached in the f32 the activations run in.
                cache_layers.push(LayerCache::Kv(
                    KvCache::new(max_seq_len).with_position_bytes(2 * n_kv_heads * head_dim * 4),
                ));
                Ok(QTransformerBlock {
                    pre_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("attn_norm"))?,
                    attn: QAttention::load(&bvb, n_heads, n_kv_heads, head_dim)?,
                    post_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("post_attention_norm"))?,
                    ffn: QMoEFFN::load(
                        &bvb,
                        n_experts,
                        n_experts_used,
                        Some(swiglu_limit),
                        moe_device,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
            // Tied embeddings: reuse token_embd
            QLinear::from_arc(vb.get("token_embd.weight")?, None)?
        };

        Ok(Self {
            embed_tokens,
            blocks,
            final_norm,
            lm_head,
            rope,
            cache: ModelCache::new(cache_layers),
            device: device.clone(),
            sliding_window,
            layer_types,
            transient: TransientDims {
                n_heads,
                n_kv_heads,
                head_dim,
                n_embd,
                n_ff: metadata.get_u32_or(
                    &format!("{prefix}.expert_feed_forward_length"),
                    n_embd as u32,
                ) as usize,
                experts_on_device: moe_device.same_device(device),
            },
            kv_narrow: !matches!(
                std::env::var("GALLIUM_GPT_OSS_KV_NARROW").as_deref(),
                Ok("0")
            ),
        })
    }
}

impl CausalLM for GptOssQ {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, seq_len) = token_ids.dims2()?;
        // Gather just this call's rows from the mmap-resident quantized table
        // and move only those onto the compute device — see the field doc
        // comment on `embed_tokens`.
        let hidden = self.embed_tokens.expert_shape()[0];
        let mut h = self
            .embed_tokens
            .gather_rows(&cpu_ids(token_ids)?, &self.device)?
            .reshape((b, seq_len, hidden))?;

        for (i, block) in self.blocks.iter().enumerate() {
            let is_sliding = self
                .layer_types
                .get(i)
                .map(|s| s.contains("sliding"))
                .unwrap_or(false);
            // Sliding layers need a mask even at seq_len=1 (decode) once the KV cache
            // exceeds the window — otherwise queries attend to evicted-by-design K/V.
            // Full-attention layers at seq_len=1 have nothing to mask (all K are in the
            // past, causal is automatic).
            let needs_mask = seq_len > 1 || (is_sliding && pos + seq_len > self.sliding_window);
            let mask = if !needs_mask {
                None
            } else {
                let m = if is_sliding && self.kv_narrow {
                    build_sliding_window_mask_narrowed(
                        seq_len,
                        pos,
                        self.sliding_window,
                        &self.device,
                    )?
                } else if is_sliding {
                    build_sliding_window_mask(seq_len, pos, self.sliding_window, &self.device)?
                } else {
                    build_causal_mask(seq_len, pos, &self.device)?
                };
                Some(m)
            };
            let kv = self.cache.get_kv(i);
            let kv = kv.expect("GPT-OSS layers all use KV cache");
            h = block.forward(&h, &self.rope, pos, kv, mask.as_ref())?;
        }

        let h = self.final_norm.forward(&h)?;
        let logits = self
            .lm_head
            .forward(&h.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        Ok(logits.to_dtype(candle_core::DType::F32)?)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    /// The cache is booked against the VRAM ledger (issue #343) without being
    /// offered for reuse across calls — `cache()` stays `None`, so GPT-OSS
    /// still re-evaluates each prompt whole, as it always has.
    fn ledger_cache(&mut self) -> Option<&mut ModelCache> {
        Some(&mut self.cache)
    }

    /// One layer's scratch at a time on top of the residual stream, all f32,
    /// erring high. Dominated at length by the full-attention layers' scores,
    /// `[heads, s, pos + s]` — 64 heads, so ~8.6 GB per copy for a 512-token
    /// window at 131k — which is what limits how long a context this model can
    /// hold on a given card, and why the ledger has to know it. Counted five
    /// times: `QAttention::forward` holds the scaled scores, the masked ones,
    /// the sink-concatenated ones and the softmax at once, and the weighted
    /// sum copies the narrowed probabilities. Counting two (scores + probs)
    /// let a 4k-token prefill out-run its booking and OOM the card.
    fn transient_bytes(&self, s: usize, pos: usize) -> usize {
        const F: usize = 4;
        let d = &self.transient;
        let stream = 6 * s * d.n_embd * F;
        let attn = 2 * s * (d.n_heads + 2 * d.n_kv_heads) * d.head_dim * F
            + 5 * d.n_heads * s * (pos + s) * F;
        // Routed outputs come back to the device either way; on the device a
        // prefill also expands each expert (gate, up, down) to f32 — two
        // counted, for the one being freed as the next is built.
        let moe = 4 * s * d.n_embd * F
            + if d.experts_on_device && s > 1 {
                2 * 3 * d.n_ff * d.n_embd * F + 3 * s * d.n_ff * F
            } else {
                0
            };
        stream + attn + moe
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
