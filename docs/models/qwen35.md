# Qwen 3.5 Implementation Notes

Implementation notes for `crates/gallium-models/src/qwen35_q.rs` (GGUF, the
only candle path for this family). A safetensors implementation
(`qwen35.rs`) existed alongside it and is documented below wherever the
mechanics differ by format — it was dropped for maintenance cost; see git
history for the file, and the `Arch::Qwen35` match arm in
`gallium-agent/src/llm_candle.rs` for where a safetensors load now refuses
with a pointer back to GGUF.

Model: `Qwen/Qwen3.5-9B` (text component of a multimodal model). The numeric
detail below (layer count, hidden size, GGUF keys) is specific to this 9B
dense checkpoint — **Qwen3.8-27B is the currently targeted/recommended
checkpoint** (see `docs/models/architectures.md`, `configs/qwen3.8.toml`),
a larger dense checkpoint of the same `qwen35`-prefixed architecture family
this file documents the mechanics of (not the `qwen35moe` MoE variant Qwen
3.6-35B-A3B used before it). Its full tensor-shape table hasn't been written
up here — but see Bugs 2 and 3 under Debugging History below for the two
places its dimensions actually differ from the 9B numbers this file
otherwise documents (`attn_qkv` output width, `ssm.time_step_rank`), found by
running it for real (`docs/VERIFICATION_STATUS.md`, "Qwen3.8-27B on candle").

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 32 |
| Hidden size | 4096 |
| FFN intermediate | 12288 |
| Layer type | Hybrid: 24× GatedDeltaNet + 8× full attention |
| Full attention pattern | Every 4th layer (indices 3, 7, 11, 15, 19, 23, 27, 31) |
| RMSNorm eps | 1e-6 |
| Context length | 262144 |

### DeltaNet layers (24 of 32)

| Parameter | Value | GGUF key |
|-----------|-------|----------|
| num_k_heads | 16 | `qwen35.ssm.group_count` |
| num_v_heads | 32 | `qwen35.ssm.time_step_rank` |
| key_head_dim (dk) | 128 | derived: `inner_size / 2 / group_count` |
| value_head_dim (dv) | 128 | `qwen35.ssm.state_size` |
| key_dim total | 2048 | |
| value_dim total | 4096 | `qwen35.ssm.inner_size` |
| conv_kernel_dim | 4 | `qwen35.ssm.conv_kernel` |

QKV are fused into a single `attn_qkv` projection: `hidden → key_dim*2 + value_dim` = `hidden → 8192`.

K has GQA: num_v_heads (32) > num_k_heads (16), so Q and K are expanded 2× to match V using **tiled** repeat. See [V-head tiled layout](#v-head-tiled-layout-gguf) below.

### Full attention layers (8 of 32)

| Parameter | Value |
|-----------|-------|
| num_heads (Q) | 16 |
| num_kv_heads | 4 |
| head_dim | 256 |
| partial_rotary_factor | 0.25 (first 64 of 256 dims rotated) |
| RoPE theta | 10,000,000 |

Q projection size is `16 × 256 × 2 = 8192` (doubled): the first half is the query, the second half is a per-head gate applied as `sigmoid(gate)` to the attention output before `out_proj`.

## Block Structure

Both layer types use the same 2-norm pre-norm residual structure:

```
1. residual = h
   h = input_layernorm(h)
   h = self_attn(h)  | linear_attn(h)
   h = residual + h

2. residual = h
   h = post_attention_layernorm(h)
   h = mlp(h)
   h = residual + h
```

## GatedDeltaNet Recurrence

The linear attention recurrence (per token `t`, per head `h`):

```
S = S * exp(g_t)                   # exponential decay (g < 0 → exp(g) ∈ (0,1))
kv_mem = S^T @ k_t                 # read from state: (dv,)
delta = (v_t - kv_mem) * beta_t   # correction term
S = S + k_t ⊗ delta               # delta write: outer product (dk, dv)
o_t = S^T @ q_t                   # read output
```

State `S` has shape `(n_v_heads, dk, dv)`.

### Gate computation

```
g = -A_log.exp() * softplus(in_proj_a(x) + dt_bias)
beta = sigmoid(in_proj_b(x))
```

`g` is always negative (decay factor); `exp(g) ∈ (0,1)`.

### Output normalization (RMSNormGated)

After the recurrence, apply norm-first gated RMSNorm:

```
out = rms_norm(out) * norm_weight * silu(z)
```

where `z = in_proj_z(x)` (a separate `hidden → value_dim` projection). **The gate is applied after normalization**, not before.

### Causal convolution

QKV is convolved with a depthwise causal conv1d (kernel=4) with SiLU activation before the recurrence. During decode, the conv state (last `k-1` tokens) is stored in `RecurrentState.conv_state`.

## MRoPE

Qwen3.5 uses multimodal RoPE (MRoPE) with `dimension_sections = [11, 11, 10, 0]` splitting the 32 rotary dimensions across time, height, and width modalities. **For text-only inference**, all three modality positions are equal (same token index), making MRoPE identical to standard RoPE. No special handling is required.

## Norm Weight Convention

Qwen3_5RMSNorm initializes weight to zeros and applies:

```python
x * rsqrt(mean(x^2) + eps) * (1 + weight)
```

The `+1` makes the identity initialization (zero weights) equivalent to standard RMSNorm with ones. The (now-removed) **safetensors** implementation used `Norm::rms_one_plus`, which adds 1.0 to the loaded weight before applying: `scale = weight + 1.0`. GGUF does not need this at inference — see "Norm weights have +1 baked in" under GGUF Conventions below, where the conversion step folds the `+1` into the stored weight instead.

## Weight Namespace (safetensors, removed)

This section described the now-removed `qwen35.rs`; kept for context on why `text_config` unwrapping shows up elsewhere in this codebase (`gemma4.rs`, `gpt_oss.rs` share the same HF convention). Qwen3.5-9B is multimodal (`Qwen3_5VLForConditionalGeneration`). All text weights are nested under `model.text_model.*`. The top-level `config.json` wraps the text config under the `"text_config"` key, which had to be extracted before deserializing:

```rust
let full: serde_json::Value = loader::load_config(&config_path)?;
let text = full.get("text_config").unwrap_or(&full);
let cfg: Qwen35Config = serde_json::from_value(text.clone())?;
```

## GGUF Conventions

### ssm_a is pre-computed

The GGUF converter (`convert_hf_to_gguf.py`) pre-computes `A_log` at conversion time:

```python
# convert_hf_to_gguf.py
if name.endswith(".A_log"):
    data_torch = -torch.exp(data_torch)
```

So `ssm_a` in GGUF stores `-exp(A_log)` (the decay rate, always negative). **Do NOT apply `exp()` or `neg()` at inference.** Use directly:

```rust
// qwen35_q.rs — correct:
let g = alog_f32.broadcast_mul(&softplus(&a_plus_dt)?)?;
// (alog is already negative, softplus is positive → g is negative → exp(g) ∈ (0,1))
```

### Norm weights have +1 baked in

The GGUF converter adds 1 to all norm weights **except** `linear_attn.norm.weight` (ssm_norm):

```python
if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
    data_torch = data_torch + 1
```

So for `attn_norm.weight`, `post_attention_norm.weight`, `output_norm.weight` (final norm): use directly as the scale. For `ssm_norm.weight` (the RMSNormGated scale): use directly too, but note it was NOT incremented — use `norm_weight` as-is.

### V-head tiled layout (GGUF)

The unsloth GGUF (and Ollama's converter) stores DeltaNet V-related tensors with **tiled** head ordering, not interleaved. Ollama defaults `vHeadReordered=true` for the `qwen35` architecture (see `model/models/qwen3next/model.go:defaultVHeadReordered`).

With n_k_heads=16, n_v_heads=32 (rep=2):

```
Tiled   (GGUF): [K0_V0 | K1_V0 | … | K15_V0 | K0_V1 | … | K15_V1]
Interleaved:    [K0_V0 | K0_V1 | K1_V0 | K1_V1 | … | K15_V1]
```

Where `Ki_Vj` is the j-th V-head paired with K-head i.

**GQA expansion of Q and K must therefore use tiled repeat** (`unsqueeze(2)+expand`):

```rust
// Correct (tiled): all K-heads once, then all K-heads again
let q = q.unsqueeze(2)?.expand((b, seq_len, rep, n_k, dk))?.contiguous()?.reshape((b, seq_len, n_v, dk))?;

// Wrong (interleaved): would use unsqueeze(3) → [h0,h0,h1,h1,…]
```

Affected GGUF tensors (all reordered by the unsloth converter, consistent with tiled layout):
`attn_qkv.weight`, `attn_gate.weight`, `ssm_beta.weight`, `ssm_alpha.weight`,
`ssm_dt.bias`, `ssm_out.weight`, `ssm_conv1d.weight`, `ssm_a`.

**Diagnostic**: to confirm tiled vs interleaved ordering in a GGUF, inspect `ssm_a` (shape `[n_v_heads]`). For tiled order, `|val[i] - val[i + n_k]|` (same K-head, adjacent V-group) should be much smaller than `|val[i] - val[i+1]|` (different K-heads). In the Qwen3.5-9B-Q4_K_M.gguf: mean tiled-pair diff = 0.006, mean interleaved-pair diff = 0.052 (9× larger).

### GGUF conv weight shape

Candle reverses GGUF dimension order on load (see `gguf_file.rs` line 438). The `ssm_conv1d.weight` is stored in GGUF as `[conv_k=4, conv_dim=8192]`, which candle loads as `(8192, 4)` = `(conv_dim, conv_k)`. Transpose before use:

```rust
let w = self.conv_weight.t()?.contiguous()?; // (conv_k, conv_dim)
// window: (b, conv_k, conv_dim) → broadcast_mul(w) → sum(dim=1) → (b, conv_dim)
```

### GGUF tensor name mapping

| HF weight | GGUF tensor |
|-----------|-------------|
| `model.text_model.embed_tokens.weight` | `token_embd.weight` |
| `model.text_model.norm.weight` | `output_norm.weight` |
| `model.text_model.lm_head.weight` | `output.weight` (or tied to `token_embd.weight`) |
| `model.text_model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| `model.text_model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.post_attention_norm.weight` |
| `model.text_model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` (fused: Q‖gate, 2× width) |
| `model.text_model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `model.text_model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `model.text_model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `model.text_model.layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` |
| `model.text_model.layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_qkv.weight` | `blk.{i}.attn_qkv.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_z.weight` | `blk.{i}.attn_gate.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_a.weight` | `blk.{i}.ssm_alpha.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_b.weight` | `blk.{i}.ssm_beta.weight` |
| `model.text_model.layers.{i}.linear_attn.out_proj.weight` | `blk.{i}.ssm_out.weight` |
| `model.text_model.layers.{i}.linear_attn.A_log` | `blk.{i}.ssm_a` (pre-computed: `-exp(A_log)`) |
| `model.text_model.layers.{i}.linear_attn.dt_bias` | `blk.{i}.ssm_dt.bias` |
| `model.text_model.layers.{i}.linear_attn.conv1d.weight` | `blk.{i}.ssm_conv1d.weight` |
| `model.text_model.layers.{i}.linear_attn.norm.weight` | `blk.{i}.ssm_norm.weight` (NOT +1) |
| `model.text_model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `model.text_model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `model.text_model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |

### GGUF Key Metadata

| GGUF key | Used for |
|----------|----------|
| `qwen35.block_count` | number of layers |
| `qwen35.context_length` | max sequence length |
| `qwen35.embedding_length` | hidden_size |
| `qwen35.feed_forward_length` | intermediate_size |
| `qwen35.attention.head_count` | num_heads (full attention Q) |
| `qwen35.attention.head_count_kv` | num_kv_heads (full attention) |
| `qwen35.attention.key_length` | head_dim (full attention) |
| `qwen35.attention.layer_norm_rms_epsilon` | rms_eps |
| `qwen35.rope.freq_base` | RoPE theta |
| `qwen35.rope.dimension_count` | rotary_dim (= head_dim × partial_rotary_factor) |
| `qwen35.full_attention_interval` | layer stride for full attention (default 4) |
| `qwen35.ssm.conv_kernel` | DeltaNet conv kernel size |
| `qwen35.ssm.state_size` | value_head_dim (dv) |
| `qwen35.ssm.group_count` | num_k_heads |
| `qwen35.ssm.time_step_rank` | num_v_heads |
| `qwen35.ssm.inner_size` | value_dim = num_v_heads × dv |
| `tokenizer.ggml.eos_token_id` | EOS token ID |

## Usage Examples

Qwen3.5-9B is a **base model**, not an instruction-tuned model. It performs text completion, not question answering. For reliable factual output, provide few-shot context:

The `gallium` binary reads prompts from stdin, so a one-shot completion is a pipe:

```bash
export MODEL_PATH=hf:unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf

# Good: few-shot context guides the model
echo "The capital of Japan is Tokyo. The capital of France is" | gallium

# Bad: single-fact prompt with no context → unpredictable completions
echo "The capital of France is" | gallium
```

Custom sampling on the native candle engine, same GGUF (there is no
safetensors path any more — see the top of this doc):

```bash
MAX_TOKENS=32 LLM_TEMPERATURE=0.0 \
INFERENCE_ENGINE=candle \
GALLIUM_TOKENIZER_REPO=Qwen/Qwen3.5-9B \
MODEL_PATH=hf:unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  gallium
```

## Quality Notes

**Historical, from when `qwen35.rs` (safetensors) existed — not reproducible
through gallium today, kept because the underlying quantization-quality point
still holds and might matter again if this ever gets re-added.**
F16 safetensors beat Q4_K_M GGUF for in-context learning:

- F16 (full precision): correctly continues multi-fact few-shot chains (France→Paris, then Germany→Berlin)
- Q4_K_M (4-bit): sometimes loops back to the first fact in the chain (France→Paris again) rather than advancing

With greedy decoding (`LLM_TEMPERATURE=0.0`) and a short budget (`MAX_TOKENS=32`), Q4_K_M is adequate for simple single-step completions — true today, GGUF-only.

At higher temperatures or longer generations, Q4_K_M is prone to:
- Repetition loops ("the term is the term is...")
- Hallucinatory text (HTML fragments, glossary entries, foreign-language text)

Use `--repetition-penalty 1.1` to mitigate loops at the cost of slightly reduced coherence.

## Reference

- `references/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- Key functions: `Qwen3_5DecoderLayer.forward`, `Qwen3_5GatedDeltaNet.forward`, `Qwen3_5Attention.forward`, `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`

## Debugging History

### Bug 1: Interleaved vs tiled GQA expansion in DeltaNet (qwen35_q.rs)

**Symptom**: GGUF inference produced coherent text for the first token (e.g. "Paris" at rank 1 for few-shot prompts), but long-context generation (agent tool-calling prompt, ~200 tokens) produced garbled output — repeated `</function>` tokens, then silence.

**Root cause**: The DeltaNet GQA expansion used `unsqueeze(3)+expand` (interleaved repeat) instead of `unsqueeze(2)+expand` (tiled repeat). The GGUF stores V-heads in tiled order (Ollama `vHeadReordered=true`), so the wrong expansion caused Q/K heads to be paired with mismatched V-head channels.

**Evidence**: `ssm_a` values exhibit the tiled pattern: `|val[i] - val[i + n_k]|` is 9× smaller than `|val[i] - val[i+1]|`. Ollama source (`model/models/qwen3next/model.go`) explicitly sets `defaultVHeadReordered("qwen35") = true` and uses `Repeat4D` (tile) not `repeat_interleave`.

**Fix** (`crates/gallium-models/src/qwen35_q.rs`, DeltaNet GQA expansion):
```rust
// Before (wrong — interleaved):
let q = q.unsqueeze(3)?.expand((b, seq_len, n_k, rep, dk))?.contiguous()?.reshape((b, seq_len, n_v, dk))?;

// After (correct — tiled):
let q = q.unsqueeze(2)?.expand((b, seq_len, rep, n_k, dk))?.contiguous()?.reshape((b, seq_len, n_v, dk))?;
```

**Verification**: `qwen35_gguf` integration test passes with "Paris" at rank 1 (logit 19.0 vs next-best 12.9). Docker integration test `coding qwen35-gguf` passes end-to-end.

### Bug 2: `block_count` includes a trailing MTP head (qwen35_q.rs)

**Symptom**: loading `unsloth/Qwen3.8-27B-GGUF` (the current target checkpoint — see the top of this doc) failed at load time: `cannot find tensor: blk.64.attn_qkv.weight`.

**Root cause**: this checkpoint appends a "next-token prediction" (MTP) head, used for speculative-decoding training, as block 64 of a `qwen35.block_count = 65` GGUF — not a normal transformer layer, and unused by plain greedy/sampled generation (gallium has no MTP support). The periodic full/linear-attention pattern (`(i+1) % full_attention_interval == 0`) predicted block 64 should be a DeltaNet layer (fused `attn_qkv`); the MTP head has separate `attn_q`/`attn_k`/`attn_v` instead, plus its own `nextn.*` tensors.

**Fix**: detect `blk.{block_count-1}.nextn.eh_proj.weight` and, if present, treat `block_count - 1` as the real layer count (`real_layer_count`, a pure function with its own unit test — no multi-GB file needed to check the arithmetic).

**Verification**: `deltanet_key_head_dim_matches_the_27b_checkpoint` / `real_layer_count_drops_a_trailing_mtp_head` (`qwen35_q.rs::tests`); end-to-end run against the real checkpoint in `docs/VERIFICATION_STATUS.md` ("Qwen3.8-27B on candle").

### Bug 3: DeltaNet key head dim assumed `value_dim / 2` (qwen35_q.rs)

**Symptom**: past Bug 2's fix, forward failed: `narrow invalid args start + len > dim_len: [1, 2216, 10240], dim: 2, start: 6144, len: 6144`.

**Root cause**: `dk = (value_dim / 2) / n_k_heads` is not the real formula — it's `key_dim_total = (conv_dim - value_dim) / 2` where `conv_dim` is the fused `attn_qkv` projection's actual output width (`key_dim_total*2 + value_dim`). The two formulas agree only when `conv_dim == 2 * value_dim`, which happens to hold on the 9B checkpoint (8192 == 2×4096) this code was first written against — not a general identity. On 27B, `conv_dim` (10240) ≠ `2 * value_dim` (12288, since value_dim=6144 there), so the shortcut computed `dk=192` instead of the real 128 and narrowed the fused QKV tensor past its actual width.

**Fix**: `conv_dim` has no GGUF metadata key of its own, so read it off a real linear-attention layer's `attn_qkv.weight` shape (`elem_count() / n_embd`, which doesn't care which axis candle's loader calls "out") instead of deriving it from `value_dim` (`deltanet_key_head_dim`).

**Verification**: `deltanet_key_head_dim_matches_the_27b_checkpoint` (the new case, 128) and `deltanet_key_head_dim_matches_the_9b_checkpoint` (pins that the fix doesn't move the answer where the old shortcut happened to be right) — `qwen35_q.rs::tests`.
