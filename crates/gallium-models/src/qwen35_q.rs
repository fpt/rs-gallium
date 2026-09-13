//! Quantized Qwen 3.5 model loaded from GGUF.
//!
//! Tensor naming in the GGUF:
//!   Linear attention layers:  blk.{i}.attn_qkv, attn_gate, ssm_alpha, ssm_beta,
//!                             ssm_out, ssm_conv1d, ssm_a, ssm_dt.bias, ssm_norm
//!   Full attention layers:    blk.{i}.attn_q (2×, gate fused), attn_k, attn_v,
//!                             attn_output, attn_q_norm, attn_k_norm
//!   Both:                     blk.{i}.attn_norm, post_attention_norm, ffn_{gate,up,down}

use candle_core::{DType, Device, Module, Result, Tensor};

use gallium_core::linear_attn::{causal_conv1d, gated_delta_rule, l2_normalize, rms_norm_gated};
use gallium_core::probe::StageTimer;
use gallium_core::quantized::{GgufMetadata, QExperts, QLinear, QNorm, QVarBuilder};
use gallium_core::*;

/// Token ids as a host `Vec<u32>` — what a mmap row-gather indexes with.
/// Identical to `gemma4_q.rs`'s / `gpt_oss_q.rs`'s private helper of the same
/// name; not shared because the two crates' modules don't have a natural
/// common home for five lines, and duplicating them costs less than the
/// indirection would.
fn cpu_ids(token_ids: &Tensor) -> Result<Vec<u32>> {
    token_ids
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()
}

// -- Quantized full Attention ------------------------------------------------
// Handles q_output_gate (2× q_proj) and per-head q_norm / k_norm.

struct QAttention {
    q_proj: QLinear, // out_dim = n_q_heads * head_dim * 2 (query + gate fused)
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: QNorm,
    k_norm: QNorm,
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
        rms_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: QLinear::load(&vb.pp("attn_q"))?,
            k_proj: QLinear::load(&vb.pp("attn_k"))?,
            v_proj: QLinear::load(&vb.pp("attn_v"))?,
            o_proj: QLinear::load(&vb.pp("attn_output"))?,
            q_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_q_norm"))?,
            k_norm: QNorm::rms_load(rms_eps, &vb.pp("attn_k_norm"))?,
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

        // q_proj is 2×: [query | gate] concatenated along the last dim.
        let q_raw = self.q_proj.forward(x)?; // (b, s, h*d*2)
        let qg = q_raw.reshape((b, seq_len, h, d * 2))?;
        let q_part = qg.narrow(3, 0, d)?; // (b, s, h, d)
        let gate = qg.narrow(3, d, d)?.reshape((b, seq_len, h * d))?; // (b, s, h*d)
        let q = q_part.transpose(1, 2)?; // (b, h, s, d)

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

        // Per-head norms then RoPE (transpose produces non-contiguous views)
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = rope.apply(&q.contiguous()?, pos)?;
        let k = rope.apply(&k.contiguous()?, pos)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        // K/V stay at h_kv heads; `gqa_scores` groups Q instead of expanding them.
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = (gqa_scores(&q, &k)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_out = gqa_weighted_sum(&probs, &v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, seq_len, h * d))?;

        // Output gate: attn_out * sigmoid(gate)
        let attn_out = (attn_out * candle_nn::ops::sigmoid(&gate)?)?;

        self.o_proj.forward(&attn_out)
    }
}

// -- Quantized GatedDeltaNet -------------------------------------------------

struct QGatedDeltaNet {
    in_proj_qkv: QLinear, // attn_qkv:   hidden → key_dim*2 + value_dim
    in_proj_z: QLinear,   // attn_gate:  hidden → value_dim
    /// ssm_beta / ssm_alpha: hidden → n_v_heads, stored dequantized and
    /// transposed, `(hidden, n_v)`. A 32-wide output is a poor shape for the
    /// quantized many-row kernel; a dense f32 matmul on the same dequantized
    /// values (what the quantized kernel would compute anyway) is several
    /// times faster at a prefill, and these two run every layer.
    in_proj_b_t: Tensor,
    in_proj_a_t: Tensor,
    out_proj: QLinear,   // ssm_out:    value_dim → hidden
    conv_weight: Tensor, // ssm_conv1d: (conv_k, conv_dim) — dequantized F32
    a_log: Tensor,       // ssm_a:      (n_v_heads,) F32  — stored as -exp(A_log) in GGUF
    dt_bias: Tensor,     // ssm_dt.bias:(n_v_heads,) F32
    norm_weight: Tensor, // ssm_norm.weight: (dv,) F32
    n_k: usize,
    n_v: usize,
    dk: usize,
    dv: usize,
    rms_eps: f64,
}

impl QGatedDeltaNet {
    fn load(
        vb: &QVarBuilder,
        n_k: usize,
        n_v: usize,
        dk: usize,
        dv: usize,
        conv_k: usize,
        rms_eps: f64,
    ) -> Result<Self> {
        let dev = vb.device();
        // candle's (conv_dim, conv_k) view of the GGUF's [conv_k, conv_dim]
        // tensor. The kernel length the conv actually runs with is read off
        // this shape (`causal_conv1d`), so the metadata's `ssm.conv_kernel` is
        // only checked against it — a disagreement is a file worth refusing,
        // not one to guess about.
        let conv_weight = vb.get("ssm_conv1d.weight")?.dequantize(dev)?;
        if conv_weight.dim(1)? != conv_k {
            candle_core::bail!(
                "ssm_conv1d.weight is {:?} but ssm.conv_kernel says {conv_k}",
                conv_weight.dims()
            );
        }
        Ok(Self {
            in_proj_qkv: QLinear::from_arc(vb.get("attn_qkv.weight")?, None)?,
            in_proj_z: QLinear::from_arc(vb.get("attn_gate.weight")?, None)?,
            in_proj_b_t: vb
                .get("ssm_beta.weight")?
                .dequantize(dev)?
                .t()?
                .contiguous()?,
            in_proj_a_t: vb
                .get("ssm_alpha.weight")?
                .dequantize(dev)?
                .t()?
                .contiguous()?,
            out_proj: QLinear::from_arc(vb.get("ssm_out.weight")?, None)?,
            conv_weight,
            a_log: vb.get("ssm_a")?.dequantize(dev)?,
            dt_bias: vb.get("ssm_dt.bias")?.dequantize(dev)?,
            norm_weight: vb.get("ssm_norm.weight")?.dequantize(dev)?,
            n_k,
            n_v,
            dk,
            dv,
            rms_eps,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        state: &mut RecurrentState,
        timer: &mut StageTimer,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let n_k = self.n_k;
        let n_v = self.n_v;
        let dk = self.dk;
        let dv = self.dv;
        let key_dim = n_k * dk;
        let value_dim = n_v * dv;

        // 1. Project + causal conv + SiLU on QKV
        let mixed = self.in_proj_qkv.forward(x)?;
        let mixed = self.apply_causal_conv(&mixed, state)?; // (b, s, key_dim*2+value_dim)
        timer.lap("dn.qkv+conv", &mixed);

        // 2. Split Q, K, V
        let q = mixed.narrow(2, 0, key_dim)?;
        let k = mixed.narrow(2, key_dim, key_dim)?;
        let v = mixed.narrow(2, key_dim * 2, value_dim)?;

        // 3. Gate projections
        let z = self.in_proj_z.forward(x)?; // (b, s, value_dim)
                                            // Dense f32 for the two narrow gate projections (see the field docs).
        let x2 = x.reshape((b * seq_len, x.dim(2)?))?;
        let b_raw = x2
            .matmul(&self.in_proj_b_t.to_dtype(x.dtype())?)?
            .reshape((b, seq_len, n_v))?; // (b, s, n_v)
        let a_raw = x2
            .matmul(&self.in_proj_a_t.to_dtype(x.dtype())?)?
            .reshape((b, seq_len, n_v))?; // (b, s, n_v)

        let beta = candle_nn::ops::sigmoid(&b_raw)?; // (b, s, n_v)

        // g = ssm_a * softplus(a + dt_bias)
        // NOTE: GGUF stores ssm_a = -exp(A_log) pre-computed (see convert_hf_to_gguf.py:4759),
        // so no exp() or neg() needed here — just multiply directly.
        let a_f32 = a_raw.to_dtype(DType::F32)?;
        let dt_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let alog_f32 = self.a_log.to_dtype(DType::F32)?;
        let a_plus_dt = a_f32.broadcast_add(&dt_f32)?;
        let g = alog_f32
            .broadcast_mul(&softplus(&a_plus_dt)?)?
            .to_dtype(x.dtype())?;
        timer.lap("dn.gates", &g);

        // 4. Reshape to (b, s, n_heads, head_dim)
        let q = q.reshape((b, seq_len, n_k, dk))?;
        let k = k.reshape((b, seq_len, n_k, dk))?;
        let v = v.reshape((b, seq_len, n_v, dv))?;

        // 5. L2 normalize Q and K
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // 6. GQA: expand Q, K from n_k heads to n_v heads (tiled layout: [h0..hN, h0..hN]).
        // GGUF for qwen35 stores V-heads in tiled (not interleaved) order, matching Ollama's
        // vHeadReordered=true path which uses Repeat4D rather than repeat_interleave.
        let (q, k) = if n_v > n_k {
            let rep = n_v / n_k;
            let q = q
                .unsqueeze(2)?
                .expand((b, seq_len, rep, n_k, dk))?
                .contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            let k = k
                .unsqueeze(2)?
                .expand((b, seq_len, rep, n_k, dk))?
                .contiguous()?
                .reshape((b, seq_len, n_v, dk))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. Scale Q by 1/sqrt(dk)
        let q = (q * (dk as f64).powf(-0.5))?;
        timer.lap("dn.prep", &q);

        // 8. Gated delta rule — chunked for a prompt, one recurrent step for a
        //    decode token (`gallium_core::linear_attn::gated_delta_rule`).
        let (output, s) = gated_delta_rule(&q, &k, &v, &g, &beta, state.state.take())?;
        timer.lap("dn.rule", &output);
        state.state = Some(s.to_dtype(x.dtype())?);

        // (b, seq, n_v, dv) → flatten heads
        let output = output.to_dtype(x.dtype())?;

        // 9. RMSNormGated: rms_norm(output) * norm_weight * silu(z)
        let output_flat = output.reshape((b * seq_len * n_v, dv))?;
        let z_flat = z.reshape((b * seq_len * n_v, dv))?;

        let normed = self.rms_norm_gated(&output_flat, &z_flat)?;
        let output = normed.reshape((b, seq_len, value_dim))?;

        let out = self.out_proj.forward(&output)?;
        timer.lap("dn.norm+out", &out);
        Ok(out)
    }

    /// See `gallium_core::linear_attn::rms_norm_gated`.
    fn rms_norm_gated(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        rms_norm_gated(x, gate, &self.norm_weight, self.rms_eps)
    }

    /// Causal depthwise conv1d with SiLU.
    /// conv_weight in GGUF is (conv_k, conv_dim) — used directly.
    fn apply_causal_conv(&self, x: &Tensor, state: &mut RecurrentState) -> Result<Tensor> {
        // GGUF stores conv weight as (conv_dim, conv_k); transpose to (conv_k, conv_dim).
        let w = self.conv_weight.t()?.contiguous()?; // (k, conv_dim)
        causal_conv1d(x, &w, state)
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // Numerically stable: max(x,0) + log(1 + exp(-|x|))
    let pos = x.clamp(0.0_f64, f64::MAX)?;
    let neg_abs = x.abs()?.neg()?;
    pos + (neg_abs.exp()? + 1.0_f64)?.log()?
}

// -- Quantized GatedFFN ------------------------------------------------------

struct QGatedFFN {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl QGatedFFN {
    fn load(vb: &QVarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: QLinear::from_arc(vb.get("ffn_gate.weight")?, None)?,
            up_proj: QLinear::from_arc(vb.get("ffn_up.weight")?, None)?,
            down_proj: QLinear::from_arc(vb.get("ffn_down.weight")?, None)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// -- Per-layer attention dispatch --------------------------------------------

enum QLayerAttn {
    Full(QAttention),
    Linear(QGatedDeltaNet),
}

struct QTransformerBlock {
    pre_attn_norm: QNorm,
    attn: QLayerAttn,
    post_attn_norm: QNorm,
    ffn: QGatedFFN,
}

impl QTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RoPE,
        pos: usize,
        kv_cache: Option<&mut KvCache>,
        recurrent: Option<&mut RecurrentState>,
        mask: Option<&Tensor>,
        timer: &mut StageTimer,
    ) -> Result<Tensor> {
        let normed = self.pre_attn_norm.forward(&x.contiguous()?)?;
        timer.lap("norm", &normed);
        let attn_out = match &self.attn {
            QLayerAttn::Full(attn) => {
                let kv = kv_cache.expect("full attention requires KV cache");
                let out = attn.forward(&normed, rope, pos, kv, mask)?;
                timer.lap("attn", &out);
                out
            }
            QLayerAttn::Linear(delta) => {
                let rec = recurrent.expect("linear attention requires recurrent state");
                delta.forward(&normed, rec, timer)?
            }
        };
        let h = (attn_out + x)?;
        let residual = h.clone();
        let h = self.post_attn_norm.forward(&h.contiguous()?)?;
        timer.lap("norm", &h);
        let h = self.ffn.forward(&h)?;
        timer.lap("ffn", &h);
        let out = (h + residual)?;
        timer.lap("residual", &out);
        Ok(out)
    }
}

// -- Full Quantized Qwen 3.5 ------------------------------------------------

/// `block_count` minus one if the checkpoint appends a trailing MTP head,
/// else `block_count` unchanged. Pure arithmetic split out of `Qwen35Q::load`
/// so the off-by-one (and the `saturating_sub` guarding a `block_count` of 0,
/// which a real GGUF never has but a hand-built metadata map in a test might)
/// has a test that doesn't need a multi-GB file.
fn real_layer_count(block_count: usize, has_trailing_mtp_head: bool) -> usize {
    if has_trailing_mtp_head {
        block_count.saturating_sub(1)
    } else {
        block_count
    }
}

/// DeltaNet's key head dim, derived from a real linear-attention layer's
/// actual `attn_qkv` output width (`qkv_out_dim`, `key_dim_total*2 +
/// value_dim`) rather than assumed from `value_dim` alone — see
/// `Qwen35Q::load`'s call site for why the naive `value_dim / 2` shortcut
/// this replaces is wrong in general, not just on the checkpoint that first
/// exposed it.
fn deltanet_key_head_dim(qkv_out_dim: usize, value_dim: usize, n_k_heads: usize) -> usize {
    (qkv_out_dim - value_dim) / 2 / n_k_heads
}

pub struct Qwen35Q {
    /// `[vocab, hidden]`, left **quantized in the file mmap** and row-gathered
    /// per forward (`QExperts::gather_rows`) — a 2-D table is the degenerate
    /// `QExperts` (`vocab` experts of shape `[hidden]`), same as
    /// `gemma4_q.rs`'s / `gpt_oss_q.rs`'s `embed_tokens`. Dequantized whole it
    /// was several GB of transient f32 for one row-lookup per token (issue
    /// #255's gpt-oss fix, applied here too).
    embed_tokens: QExperts,
    blocks: Vec<QTransformerBlock>,
    final_norm: QNorm,
    lm_head: QLinear,
    rope: RoPE,
    cache: ModelCache,
    device: Device,
}

impl Qwen35Q {
    pub fn load(metadata: &GgufMetadata, vb: &QVarBuilder, device: &Device) -> Result<Self> {
        let arch = metadata
            .get_str("general.architecture")
            .unwrap_or_else(|_| "qwen35".to_string());
        let pfx = &arch;

        let block_count = metadata.get_u32(&format!("{pfx}.block_count"))? as usize;
        // Some Qwen3.8 checkpoints append a trailing MTP ("next-token
        // prediction") head as the last `block_count` block, for speculative
        // decoding — not a normal transformer layer, and unused by plain
        // greedy/sampled generation (gallium has no MTP support). Its
        // `nextn.*` tensors are how it names itself; detected rather than
        // assumed always-present, since a checkpoint without MTP has none of
        // them and `block_count` already names exactly the real depth there.
        // Verified against `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf`:
        // block_count=65, blocks 0..64 are real layers (mixing DeltaNet and
        // full attention on `full_attention_interval`), block 64 is the MTP
        // head — loading it as a layer failed with "cannot find tensor:
        // blk.64.attn_qkv.weight" (the periodic pattern predicted a DeltaNet
        // layer there; the MTP head's attention tensors are shaped like full
        // attention's, which is itself incidental — the real tell is `nextn.*`).
        let has_trailing_mtp_head = block_count > 0
            && vb.contains(&format!("blk.{}.nextn.eh_proj.weight", block_count - 1));
        let n_layers = real_layer_count(block_count, has_trailing_mtp_head);
        let n_heads = metadata.get_u32(&format!("{pfx}.attention.head_count"))? as usize;
        let n_kv_heads = metadata.get_u32(&format!("{pfx}.attention.head_count_kv"))? as usize;
        let n_embd = metadata.get_u32(&format!("{pfx}.embedding_length"))? as usize;
        let head_dim = metadata.get_u32_or(
            &format!("{pfx}.attention.key_length"),
            (n_embd / n_heads) as u32,
        ) as usize;
        let rope_freq = metadata.get_f32_or(&format!("{pfx}.rope.freq_base"), 10_000_000.0) as f64;
        let rms_eps =
            metadata.get_f32_or(&format!("{pfx}.attention.layer_norm_rms_epsilon"), 1e-6) as f64;
        let max_seq = metadata.get_u32_or(&format!("{pfx}.context_length"), 262144) as usize;
        let fa_interval =
            metadata.get_u32_or(&format!("{pfx}.full_attention_interval"), 4) as usize;
        let rope_dims = metadata.get_u32_or(
            &format!("{pfx}.rope.dimension_count"),
            (head_dim / 4) as u32,
        ) as usize;

        // SSM (linear attention) parameters
        let n_k_heads = metadata.get_u32_or(&format!("{pfx}.ssm.group_count"), 16) as usize;
        let n_v_heads = metadata.get_u32_or(&format!("{pfx}.ssm.time_step_rank"), 32) as usize;
        let dv = metadata.get_u32_or(&format!("{pfx}.ssm.state_size"), 128) as usize;
        let value_dim = metadata.get_u32_or(&format!("{pfx}.ssm.inner_size"), 4096) as usize;
        let conv_k = metadata.get_u32_or(&format!("{pfx}.ssm.conv_kernel"), 4) as usize;
        // key_dim_total = (conv_dim - value_dim) / 2, where conv_dim is
        // attn_qkv's actual output width (key_dim*2 + value_dim) — NOT
        // `value_dim / 2`, which only equals this by the coincidence that
        // `unsloth/Qwen3.5-9B-GGUF` happens to have conv_dim == 2 * value_dim
        // (conv_dim 8192, value_dim 4096). `unsloth/Qwen3.8-27B-GGUF` breaks
        // that coincidence (conv_dim 10240, value_dim 6144 → conv_dim ≠
        // 2 * value_dim), and the old shortcut computed dk=192 instead of the
        // real 128, narrowing the fused QKV tensor past its actual width
        // ("start + len > dim_len") the moment `forward` ran. conv_dim has no
        // GGUF metadata key of its own, so it's read off a real linear-attention
        // layer's `attn_qkv.weight` — `elem_count() / n_embd` avoids caring
        // which axis candle's loader put "out" on.
        let first_linear = (0..n_layers)
            .find(|i| (i + 1) % fa_interval != 0)
            .unwrap_or(0);
        let qkv_elems = vb
            .pp(format!("blk.{first_linear}"))
            .get("attn_qkv.weight")?
            .shape()
            .elem_count();
        let dk = deltanet_key_head_dim(qkv_elems / n_embd, value_dim, n_k_heads);

        // partial_rotary_factor = rope_dims / head_dim
        let partial_rotary = rope_dims as f64 / head_dim as f64;

        let rope = RoPE::new(
            &RoPEConfig {
                head_dim,
                max_seq_len: max_seq,
                theta: rope_freq,
                partial_rotary_factor: partial_rotary,
                ..Default::default()
            },
            DType::F32,
            device,
        )?;

        // Left quantized in the mmap and row-gathered per forward instead of
        // dequantized whole — see the field doc comment on `embed_tokens`.
        let embed_tokens = vb.get_experts("token_embd.weight")?;

        // Layer i is full attention iff (i + 1) % fa_interval == 0.
        let mut cache_layers = Vec::new();
        let blocks = (0..n_layers)
            .map(|i| {
                let bvb = vb.pp(format!("blk.{i}"));
                let is_full = (i + 1) % fa_interval == 0;
                let attn = if is_full {
                    cache_layers.push(LayerCache::Kv(KvCache::new(max_seq)));
                    QLayerAttn::Full(QAttention::load(
                        &bvb, n_heads, n_kv_heads, head_dim, rms_eps,
                    )?)
                } else {
                    cache_layers.push(LayerCache::Recurrent(RecurrentState::new()));
                    QLayerAttn::Linear(QGatedDeltaNet::load(
                        &bvb, n_k_heads, n_v_heads, dk, dv, conv_k, rms_eps,
                    )?)
                };
                Ok(QTransformerBlock {
                    pre_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("attn_norm"))?,
                    attn,
                    post_attn_norm: QNorm::rms_load(rms_eps, &bvb.pp("post_attention_norm"))?,
                    ffn: QGatedFFN::load(&bvb)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let final_norm = QNorm::rms_load(rms_eps, &vb.pp("output_norm"))?;
        let lm_head = if vb.contains("output.weight") {
            QLinear::from_arc(vb.get("output.weight")?, None)?
        } else {
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
        })
    }
}

impl CausalLM for Qwen35Q {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, seq_len) = token_ids.dims2()?;
        // row-gathered per forward, not dequantized whole — see the field doc
        // comment on `embed_tokens`.
        let hidden = self.embed_tokens.expert_shape()[0];
        let mut timer = StageTimer::start();
        let mut h = self
            .embed_tokens
            .gather_rows(&cpu_ids(token_ids)?, &self.device)?
            .reshape((b, seq_len, hidden))?;
        timer.lap("embed", &h);

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = match &block.attn {
                QLayerAttn::Full(_) if seq_len > 1 => {
                    Some(build_causal_mask(seq_len, pos, &self.device)?)
                }
                _ => None,
            };
            let (kv, recurrent) = self.cache.get_layer(i);
            h = block
                .forward(
                    &h,
                    &self.rope,
                    pos,
                    kv,
                    recurrent,
                    mask.as_ref(),
                    &mut timer,
                )?
                .contiguous()?;
        }

        let h_final = self.final_norm.forward(&h)?;
        let logits = self
            .lm_head
            .forward(&h_final.narrow(1, seq_len - 1, 1)?.squeeze(1)?)?;
        let logits = logits.to_dtype(DType::F32)?;
        timer.lap("head", &logits);
        timer.report("qwen35", seq_len);
        Ok(logits)
    }

    fn reset(&mut self) {
        self.cache.reset();
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::{deltanet_key_head_dim, real_layer_count};
    use candle_core::{DType, Device, Tensor};

    /// GQA expansion must be tiled, not interleaved.
    ///
    /// With n_k=2 heads, n_v=4 heads (rep=2), the GGUF stores V-channels in
    /// tiled order: [K0_V0 | K1_V0 | K0_V1 | K1_V1].  The correct expansion
    /// maps:
    ///   output head 0 → K-head 0 (first copy)
    ///   output head 1 → K-head 1 (first copy)
    ///   output head 2 → K-head 0 (second copy)
    ///   output head 3 → K-head 1 (second copy)
    ///
    /// The wrong interleaved expansion would give [h0,h0,h1,h1], mispairing
    /// Q/K with V channels and producing garbage logits.
    #[test]
    fn tiled_gqa_expansion() {
        let dev = &Device::Cpu;
        let b = 1usize;
        let seq_len = 1usize;
        let n_k = 2usize;
        let n_v = 4usize;
        let rep = n_v / n_k;
        let dk = 3usize;

        // Q heads: head0 = [1,1,1], head1 = [2,2,2]
        let q_data: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let q = Tensor::from_vec(q_data, (b, seq_len, n_k, dk), dev).unwrap();

        // Tiled expand: unsqueeze(2) inserts rep-dim before n_k.
        let q_tiled = q
            .unsqueeze(2)
            .unwrap()
            .expand((b, seq_len, rep, n_k, dk))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, seq_len, n_v, dk))
            .unwrap();

        let out: Vec<f32> = q_tiled.flatten_all().unwrap().to_vec1().unwrap();
        // Expected tiled: [h0, h1, h0, h1] × dk
        let expected = vec![
            1.0, 1.0, 1.0, // head 0 → K-head 0 (copy 1)
            2.0, 2.0, 2.0, // head 1 → K-head 1 (copy 1)
            1.0, 1.0, 1.0, // head 2 → K-head 0 (copy 2)
            2.0, 2.0, 2.0, // head 3 → K-head 1 (copy 2)
        ];
        assert_eq!(
            out, expected,
            "GQA expansion must be tiled [h0,h1,h0,h1], not interleaved [h0,h0,h1,h1]"
        );
    }

    /// Interleaved expansion (wrong for this GGUF) produces a different layout,
    /// confirming the two patterns are distinguishable.
    #[test]
    fn interleaved_gqa_differs_from_tiled() {
        let dev = &Device::Cpu;
        let b = 1usize;
        let seq_len = 1usize;
        let n_k = 2usize;
        let n_v = 4usize;
        let rep = n_v / n_k;
        let dk = 3usize;

        let q_data: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let q = Tensor::from_vec(q_data, (b, seq_len, n_k, dk), dev).unwrap();

        // Interleaved: unsqueeze(3) inserts rep-dim after n_k.
        let q_interleaved = q
            .unsqueeze(3)
            .unwrap()
            .expand((b, seq_len, n_k, rep, dk))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, seq_len, n_v, dk))
            .unwrap();

        let out: Vec<f32> = q_interleaved.flatten_all().unwrap().to_vec1().unwrap();
        // Interleaved gives [h0,h0,h1,h1] — wrong for this GGUF.
        let interleaved_expected = vec![
            1.0, 1.0, 1.0, // head 0 → K-head 0 (copy 1)
            1.0, 1.0, 1.0, // head 1 → K-head 0 (copy 2)  ← wrong pairing with V
            2.0, 2.0, 2.0, // head 2 → K-head 1 (copy 1)
            2.0, 2.0, 2.0, // head 3 → K-head 1 (copy 2)  ← wrong pairing with V
        ];
        assert_eq!(out, interleaved_expected);

        // Must differ from tiled.
        let tiled_expected = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        assert_ne!(out, tiled_expected, "interleaved and tiled must differ");
    }

    /// Single-step DeltaNet recurrence: with identity-like state and known
    /// inputs, the output should be the V-vector (the state read after write).
    #[test]
    fn deltanet_single_step_write_then_read() {
        let dev = &Device::Cpu;
        // 1 head, dk=2, dv=2.
        let dk = 2usize;
        let dv = 2usize;

        // State S = zeros(dk, dv).
        let s = Tensor::zeros((1usize, dk, dv), DType::F32, dev).unwrap();

        // k = [1, 0], v = [3, 5], beta = 1 (full write), decay exp(g) = 1 (no decay).
        let k = Tensor::from_vec(vec![1.0f32, 0.0], (1usize, dk), dev).unwrap();
        let v = Tensor::from_vec(vec![3.0f32, 5.0], (1usize, dv), dev).unwrap();
        let beta = Tensor::from_vec(vec![1.0f32], (1usize, 1usize), dev).unwrap();

        // kv_mem = S^T @ k = zeros → delta = (v - 0) * 1 = v = [3, 5]
        // S += k ⊗ delta = [[3,5],[0,0]]
        let kv_mem = s
            .broadcast_mul(&k.unsqueeze(2).unwrap())
            .unwrap()
            .sum(1)
            .unwrap(); // (1, dv)
        let delta = (v.clone() - &kv_mem).unwrap().broadcast_mul(&beta).unwrap(); // (1, dv)
        let write = k
            .unsqueeze(2)
            .unwrap()
            .broadcast_mul(&delta.unsqueeze(1).unwrap())
            .unwrap(); // (1, dk, dv)
        let s_new = (s + write).unwrap();

        // q = [1, 0] → o = S^T @ q = [3, 5] (first column of S)
        let q = Tensor::from_vec(vec![1.0f32, 0.0], (1usize, dk), dev).unwrap();
        let o = s_new
            .broadcast_mul(&q.unsqueeze(2).unwrap())
            .unwrap()
            .sum(1)
            .unwrap(); // (1, dv)
        let o_vals: Vec<f32> = o.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (o_vals[0] - 3.0).abs() < 1e-5,
            "expected 3.0, got {}",
            o_vals[0]
        );
        assert!(
            (o_vals[1] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            o_vals[1]
        );
    }

    /// Conv weight transposition: GGUF stores ssm_conv1d as [conv_k, conv_dim].
    /// Candle reverses GGUF dimensions on load (gguf_file.rs:438), producing
    /// (conv_dim, conv_k).  After .t() we get (conv_k, conv_dim), matching
    /// a window of shape (b, conv_k, conv_dim) for broadcast multiply + sum(1).
    #[test]
    fn conv_weight_transpose_shape() {
        let dev = &Device::Cpu;
        let conv_k = 4usize;
        let conv_dim = 8usize;

        // Simulate what candle produces after loading GGUF (dimensions reversed):
        // GGUF shape [conv_k, conv_dim] → candle (conv_dim, conv_k)
        let conv_weight = Tensor::zeros((conv_dim, conv_k), DType::F32, dev).unwrap();

        // .t() gives (conv_k, conv_dim) — correct for window broadcast.
        let w = conv_weight.t().unwrap().contiguous().unwrap();
        assert_eq!(w.dims(), &[conv_k, conv_dim]);

        // Window: (b=1, conv_k, conv_dim) × (conv_k, conv_dim) → sum(1) → (b, conv_dim)
        let window = Tensor::ones((1usize, conv_k, conv_dim), DType::F32, dev).unwrap();
        let out = window.broadcast_mul(&w).unwrap().sum(1).unwrap();
        assert_eq!(out.dims(), &[1, conv_dim]);
    }

    /// The bug this exists to catch: loading a checkpoint whose `block_count`
    /// includes a trailing MTP head as though every block were a real layer
    /// failed with "cannot find tensor: blk.64.attn_qkv.weight" against
    /// `unsloth/Qwen3.8-27B-GGUF`.
    #[test]
    fn real_layer_count_drops_a_trailing_mtp_head() {
        assert_eq!(real_layer_count(65, true), 64);
        assert_eq!(real_layer_count(64, false), 64);
    }

    /// A `block_count` of 0 never happens on a real GGUF, but the arithmetic
    /// must not underflow if it somehow did.
    #[test]
    fn real_layer_count_does_not_underflow() {
        assert_eq!(real_layer_count(0, true), 0);
    }

    /// The bug this exists to catch: `value_dim / 2` happened to equal the
    /// real key head dim on the 9B checkpoint this code was first written
    /// against (`conv_dim == 2 * value_dim` there), and silently stopped
    /// being right the moment a checkpoint broke that coincidence — computing
    /// dk=192 instead of 128 and narrowing the fused QKV tensor past its
    /// actual width ("start + len > dim_len") at `forward` time.
    #[test]
    fn deltanet_key_head_dim_matches_the_27b_checkpoint() {
        // unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf: attn_qkv output
        // width 10240, ssm.inner_size (value_dim) 6144, ssm.group_count
        // (n_k_heads) 16.
        assert_eq!(deltanet_key_head_dim(10240, 6144, 16), 128);
    }

    /// The checkpoint that made the old `value_dim / 2` shortcut look right:
    /// this is the case where it and the real formula agree, so a fix here
    /// must not have moved this answer.
    #[test]
    fn deltanet_key_head_dim_matches_the_9b_checkpoint() {
        // unsloth/Qwen3.5-9B-GGUF: attn_qkv output width 8192, value_dim
        // 4096, n_k_heads 16 (docs/models/qwen35.md).
        assert_eq!(deltanet_key_head_dim(8192, 4096, 16), 128);
    }
}
