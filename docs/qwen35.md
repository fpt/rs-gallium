# Qwen 3.5 Implementation Notes

Implementation notes for `crates/gallium-models/src/qwen35.rs` (safetensors) and `qwen35_q.rs` (GGUF).

Model: `Qwen/Qwen3.5-9B` (text component of a multimodal model).

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 32 |
| Hidden size | 4096 |
| FFN intermediate | 14336 |
| Layer type | Hybrid: 24× GatedDeltaNet + 8× full attention |
| Full attention pattern | Every 4th layer (indices 3, 7, 11, 15, 19, 23, 27, 31) |
| RMSNorm eps | 1e-6 |
| Context length | 32768 |

### DeltaNet layers (24 of 32)

| Parameter | Value |
|-----------|-------|
| num_k_heads | 16 |
| num_v_heads | 32 |
| key_head_dim (dk) | 128 |
| value_head_dim (dv) | 128 |
| key_dim total | 2048 |
| value_dim total | 4096 |
| conv_kernel_dim | 4 |

QKV are fused into a single `in_proj_qkv` projection: `hidden → key_dim*2 + value_dim` = `hidden → 8192`. K has GQA: num_v_heads (32) > num_k_heads (16), so Q and K are repeated 2× to match V.

### Full attention layers (8 of 32)

| Parameter | Value |
|-----------|-------|
| num_heads (Q) | 16 |
| num_kv_heads | 4 |
| head_dim | 256 |
| partial_rotary_factor | 0.25 (first 64 of 256 dims rotated) |
| RoPE theta | 1,000,000 |

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

So for `input_layernorm.weight`, `post_attention_layernorm.weight`, `norm.weight` (final norm): use directly as the scale. For `linear_attn.norm.weight` (the RMSNormGated scale): use directly too, but note it was NOT incremented — use `norm_weight` as-is (matches the GGUF value which is the raw HF weight without the +1).

### GGUF tensor name mapping

| HF weight | GGUF tensor |
|-----------|-------------|
| `model.text_model.embed_tokens.weight` | `token_embd.weight` |
| `model.text_model.norm.weight` | `output_norm.weight` |
| `model.text_model.lm_head.weight` | `output.weight` |
| `model.text_model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| `model.text_model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` |
| `model.text_model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| `model.text_model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `model.text_model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `model.text_model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `model.text_model.layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` |
| `model.text_model.layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_qkv.weight` | `blk.{i}.ssm_in.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_z.weight` | `blk.{i}.ssm_z.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_a.weight` | `blk.{i}.ssm_a_proj.weight` |
| `model.text_model.layers.{i}.linear_attn.in_proj_b.weight` | `blk.{i}.ssm_b_proj.weight` |
| `model.text_model.layers.{i}.linear_attn.out_proj.weight` | `blk.{i}.ssm_out.weight` |
| `model.text_model.layers.{i}.linear_attn.A_log` | `blk.{i}.ssm_a` (pre-computed: `-exp(A_log)`) |
| `model.text_model.layers.{i}.linear_attn.dt_bias` | `blk.{i}.ssm_dt_bias` |
| `model.text_model.layers.{i}.linear_attn.conv1d.weight` | `blk.{i}.ssm_conv1d.weight` |
| `model.text_model.layers.{i}.linear_attn.norm.weight` | `blk.{i}.ssm_norm.weight` (NOT +1) |
| `model.text_model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `model.text_model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `model.text_model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |

### GGUF conv weight shape

The `ssm_conv1d.weight` is stored as `(conv_dim, conv_k)` in GGUF (transposed from the HF `(conv_dim, 1, conv_k)` shape). When loading, squeeze the group dimension and transpose to `(conv_k, conv_dim)` before the dot product:

```rust
let w = self.conv_weight.transpose(0, 1)?; // (conv_k, conv_dim)
```

## GGUF Key Metadata

These fields are stored in the GGUF file header and used to configure the model:

| GGUF key | Used for |
|----------|----------|
| `qwen3moe.block_count` | number of layers |
| `qwen3moe.embedding_length` | hidden_size |
| `qwen3moe.feed_forward_length` | intermediate_size |
| `qwen3moe.ssm.key_length` | key_head_dim (dk) |
| `qwen3moe.ssm.value_length` | value_head_dim (dv) |
| `qwen3moe.ssm.conv_kernel` | conv_kernel_dim |
| `qwen3moe.ssm.inner_size` | num_v_heads × value_head_dim (value_dim) |
| `qwen3moe.attention.head_count` | num_heads (full attention) |
| `qwen3moe.attention.head_count_kv` | num_kv_heads (full attention) |
| `qwen3moe.attention.key_length` | head_dim (full attention) |
| `qwen3moe.attention.layer_norm_rms_epsilon` | rms_eps |
| `qwen3moe.rope.dimension_count` | rotary_dim (= head_dim × partial_rotary_factor) |
| `qwen3moe.rope.freq_base` | RoPE theta |
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
- `references/llama.cpp/src/models/qwen35moe.cpp`
- Key functions: `Qwen3_5DecoderLayer.forward`, `Qwen3_5GatedDeltaNet.forward`, `Qwen3_5Attention.forward`, `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`
