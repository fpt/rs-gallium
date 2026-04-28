# Qwen 3.5 Implementation Notes

Implementation notes for `crates/gallium-models/src/qwen35.rs` (safetensors) and `qwen35_q.rs` (GGUF).

Model: `Qwen/Qwen3.5-9B` (text component of a multimodal model).

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

The `+1` makes the identity initialization (zero weights) equivalent to standard RMSNorm with ones. In the **safetensors** implementation, use `Norm::rms_one_plus` which adds 1.0 to the loaded weight before applying: `scale = weight + 1.0`.

## Weight Namespace

Qwen3.5-9B is multimodal (`Qwen3_5VLForConditionalGeneration`). All text weights are nested under `model.text_model.*`. The top-level `config.json` wraps the text config under the `"text_config"` key, which must be extracted before deserializing:

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

```bash
# Good: few-shot context guides the model
make run-qwen35-gguf PROMPT="The capital of Japan is Tokyo. The capital of France is"

# Bad: single-fact prompt with no context → unpredictable completions
make run-qwen35-gguf PROMPT="The capital of France is"
```

Safetensors (F16):

```bash
make run-qwen35 PROMPT="The capital of Japan is Tokyo. The capital of France is"
```

CLI with custom parameters:

```bash
cargo run --release -p gallium-cli -- \
    --arch qwen35 \
    --format gguf \
    --hf-repo unsloth/Qwen3.5-9B-GGUF \
    --hf-file Qwen3.5-9B-Q4_K_M.gguf \
    --hf-tokenizer-repo Qwen/Qwen3.5-9B \
    --prompt "The capital of Japan is Tokyo. The capital of France is" \
    --max-tokens 32 \
    --temperature 0.0
```

## Quality Notes

**F16 safetensors > Q4_K_M GGUF** for in-context learning:

- F16 (full precision): correctly continues multi-fact few-shot chains (France→Paris, then Germany→Berlin)
- Q4_K_M (4-bit): sometimes loops back to the first fact in the chain (France→Paris again) rather than advancing

With greedy decoding (`--temperature 0.0`) and short `--max-tokens` (32), Q4_K_M is adequate for simple single-step completions.

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
