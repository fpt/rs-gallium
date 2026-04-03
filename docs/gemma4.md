# Gemma 4 Implementation Notes

Implementation notes for `crates/gallium-models/src/gemma4.rs` (safetensors) and `gemma4_q.rs` (GGUF).

Model: `google/gemma-4-E4B` (text component of a multimodal model).

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 42 |
| Hidden size | 2560 |
| Q heads | 8 |
| KV heads (sliding) | 2 |
| KV heads (global) | 2 |
| Head dim (sliding) | 256 |
| Head dim (global) | 512 |
| FFN | GatedFFN (GeGLU-tanh), intermediate_size=10240 |
| Attention scale | **1.0** (not 1/sqrt(head_dim)) |
| RoPE (sliding) | theta=10000 |
| RoPE (global) | theta=1e6, partial_rotary_factor=0.25 |
| Sliding window | 512 tokens |
| KV shared layers | 18 (layers 24–41 share from owned layers) |
| PLE dim | 256 (per-layer embeddings) |
| RMSNorm eps | 1e-6 |
| Context length | 131072 |
| Logit softcap | 30.0 |
| Layer pattern | 5× sliding + 1× global, repeating |

## Weight Namespace

This is a multimodal model (`Gemma4ForConditionalGeneration`). All text weights are nested under `model.language_model.*`. The top-level `config.json` wraps the text config under the `"text_config"` key, which must be extracted before deserializing.

## Block Structure (4-norm)

Gemma 4 uses a 4-norm block, unlike the standard 2-norm (pre-norm) transformer:

```
1. residual = h
   h = input_layernorm(h)
   h = self_attn(h)
   h = post_attention_layernorm(h)
   h = residual + h

2. residual = h
   h = pre_feedforward_layernorm(h)
   h = mlp(h)
   h = post_feedforward_layernorm(h)
   h = residual + h

3. if PLE:
   residual = h
   h = gelu(per_layer_input_gate(h)) * per_layer_input   # [b, s, ple_dim]
   h = per_layer_projection(h)                            # [b, s, hidden]
   h = post_per_layer_input_norm(h)
   h = residual + h

4. h *= layer_scalar                                      # per-layer learnable scalar
```

## Non-standard Components

### 1. Attention Scale = 1.0

**The attention dot-product scale is `1.0`, not `1/sqrt(head_dim)`.**

The reference sets `self.scaling = 1.0` in `Gemma4TextAttention`. This is intentional: the Q and K norms (see below) constrain the magnitude of queries and keys, so the standard `1/sqrt(head_dim)` dampening is not needed.

Using `1/sqrt(head_dim)` (the default for most models) makes scores 16–22× too small, causing the softmax to produce near-uniform attention weights — resulting in completely incoherent output.

**Implementation:** Set `AttentionConfig.scale = Some(1.0)` when building attention layers.

### 2. Q/K/V Norms

All layers (both sliding and global) apply per-head RMSNorm to Q, K, and V after projection and reshape:

| Norm | Learnable scale | When applied |
|------|----------------|--------------|
| `q_norm` | Yes | after Q projection, before RoPE |
| `k_norm` | Yes | after K projection, before RoPE |
| `v_norm` | **No** | after V projection, before KV cache update |

The V norm has no learnable weight (`with_scale=False`); it only normalizes the magnitude. It is applied in non-shared layers (shared layers read pre-normalized V from the source cache).

**Implementation:** `AttentionConfig.q_norm = true`, `k_norm = true`, `v_norm = true`. The v_norm is implemented inline as `rms_norm_no_scale()` in `attention.rs`.

### 3. KV Sharing

The last 18 layers (`num_kv_shared_layers`) share K and V from owned layers (0–23). The mapping rule: **all shared layers of the same type map to the last owned layer of that type.**

For the 42-layer model:
- Layer pattern repeats `[slide, slide, slide, slide, slide, global]` × 4 = layers 0–23
- Last owned sliding = layer 22
- Last owned global = layer 23
- All shared sliding layers (24–28, 30–34, 36–40) → source = layer 22
- All shared global layers (29, 35, 41) → source = layer 23

Shared layers:
- Project Q only; read K/V from source cache via `current_kv()` (no append)
- Do NOT have k_norm or v_norm weights (the source cache already has normalized K/V)

**Wrong implementation:** Mapping the nth shared layer of a type to the nth owned layer of that type cycles through all owned layers instead of always reading from the last one. This caused shape errors and garbage output.

### 4. Dual RoPE (Proportional)

Sliding and global layers use different RoPE configs:

| Layer type | theta | head_dim | partial_rotary_factor |
|-----------|-------|----------|-----------------------|
| sliding | 10,000 | 256 | 1.0 (full) |
| global | 1,000,000 | 512 | 0.25 (first 128 dims) |

Global layers only rotate the first `512 * 0.25 = 128` dimensions; the remaining 384 are passed through unchanged. This requires a contiguous slice before calling `candle_nn::rotary_emb::rope()` (non-contiguous views cause a panic).

**GGUF `rope_freqs.weight` convention (critical):** the GGUF global-RoPE tensor has shape `[head_dim/2] = [256]` and stores **per-dim divisors**, NOT `inv_freq` itself:

| Indices | Value | Meaning |
|---------|-------|---------|
| 0..64 | `1.0` | Rotated pairs — keep full frequency |
| 64..256 | `1e30` | Non-rotated pairs — divisor effectively zeros the frequency |

The correct interpretation matches ollama's `nn.RoPE(q, positions, ropeDims=512, ropeBase=1e6, WithFactors(rope_freqs))`:

```rust
inv_freq[i] = (1.0 / theta^(2i / head_dim)) / factors[i]
//                    ^^^^^^^^^^^^^^^^^^^^^ base proportional frequency
//                                              ^^^^^^^^^^ divide by GGUF factor
```

Feeding the tensor values directly as `inv_freq` produces nonsense RoPE (`inv_freq=1.0` uniformly for 64 dims, `inv_freq=1e30` for 192 dims). See Bug 10.

### 5. Per-Layer Embeddings (PLE)

Each block gets a `[batch, seq, ple_dim]` tensor derived from two sources:

**Source 1 — token-level per-layer embeddings:**
```
embed_tokens_per_layer(token_ids)               # [b, s, n_layers * ple_dim]
  .reshape(b, s, n_layers, ple_dim)
  * sqrt(ple_dim)                               # scale baked into Gemma4TextScaledWordEmbedding
```

**Source 2 — projection of main embeddings:**
```
per_layer_model_projection(inputs_embeds)       # [b, s, n_layers * ple_dim]
  * (1/sqrt(hidden_size))                       # per_layer_model_projection_scale
  .reshape(b, s, n_layers, ple_dim)
  → per_layer_projection_norm(...)              # RMSNorm over ple_dim
```

**Combined:**
```
per_layer_input = (source1 + source2) * 2^-0.5
```

Note: `inputs_embeds` passed to the projection is already scaled by `sqrt(hidden_size)` (from the main embedding lookup), so multiplying by `1/sqrt(hidden_size)` effectively undoes that scaling.

## Debugging History

**Bug 1 — Attention scale 1/sqrt(head_dim) instead of 1.0:** All scores were 16–22× too small. Softmax output was near-uniform, attention was effectively averaging all positions equally. RMS norms per layer were stable (~0.9–2.3) so this was invisible without checking logits. Fixed: `AttentionConfig.scale = Some(1.0)`.

**Bug 2 — Missing v_norm:** Values were not normalized before caching. The model's weights were trained expecting unit-magnitude values; without this the attention output was scaled incorrectly. Fixed: `rms_norm_no_scale()` applied to V before cache update.

**Bug 3 — Wrong KV source layer:** The code was cycling shared layers across owned layers (first shared → layer 0, second shared → layer 6, etc.) instead of all shared sliding → layer 22, all shared global → layer 23. This caused shape errors during prefill (K/V shape mismatches) and wrong activations. Fixed: use `.last()` instead of `.nth(same_type_idx)` when computing `kv_source`.

**Bug 4 — dtype mismatch (F16 attention mask):** The causal mask is built as F32 but attention scores are F16. Fixed: cast mask to scores dtype before `broadcast_add`.

**Bug 5 — Non-contiguous tensor in partial RoPE:** `x.narrow()` returns a non-contiguous view; `candle_nn::rotary_emb::rope()` requires contiguous input. Fixed: `.contiguous()?` after each `narrow()` in `pos_enc.rs`.

**Bug 6 — Wrong weight namespace:** Gemma 4 is multimodal; text weights are under `model.language_model.*` not `model.*`. Fixed: `vb_lm = vb.pp("model.language_model")`.

**Bug 7 — BF16 unsupported on Apple Silicon CPU:** Matmul is not implemented for BF16 on Metal/CPU. Use `--dtype f16`.

### GGUF-specific bugs (gemma4_q.rs)

**Bug 8 — `get_bool_array` always returned false:** `Value::to_u32()` only handles `U32` variants and returns Err for `Bool`. Using `.unwrap_or(0)` silently defaulted every bool to `false`, making `is_sliding` all-false so all layers used `global_head_dim=512`. The Q projection of a sliding layer outputs `8×256=2048` but the reshape expected `8×512=4096`, causing shape mismatch. Fixed: match on `Value::Bool(b)` explicitly before falling back to `to_u32()`.

**Bug 9 — `expand()` before `reshape()` without `.contiguous()`:** `Tensor::expand()` returns a non-contiguous view; the subsequent `reshape()` requires contiguous input. Fixed: `.contiguous()?` between `expand()` and `reshape()` in `expand_gqa()` and after every `transpose()` before passing to norms/matmul.

**Bug 10 — `rope_freqs.weight` interpreted as `inv_freq` instead of divisors (fixed 2026-04-20):** The GGUF tensor stores per-dim **divisors** (1.0 for rotated pairs, 1e30 for non-rotated) that scale a base proportional `inv_freq` computed from `theta=1e6`. The previous code fed those values straight into `RoPE::from_inv_freq`, so the global-layer RoPE used `inv_freq[i]=1.0` for dims 0..63 and `inv_freq[i]=1e30` for dims 64..255 — nonsense that corrupted every global-attention layer. Only 7 of 42 layers are global, so the symptom was subtle: short prompts (<100 tokens) still emitted plausible greedy continuations, but at seq_len≈600+ the model regurgitated earlier prompt fragments mid-word. Fixed in `gemma4_q.rs::Gemma4Q::load` by computing `inv_freq[i] = (1 / theta^(2i/head_dim)) / factors[i]`. Verified by running a 643-token tool-calling prompt through ours and ollama at `temperature=0.0` and confirming matching output. See `.claude/skills/ollama-narrow/` for the replay harness.

## Reference

- `references/transformers/src/transformers/models/gemma4/modeling_gemma4.py`
- Key functions: `Gemma4TextAttention.forward`, `Gemma4TextDecoderLayer.forward`, `project_per_layer_inputs`, `get_per_layer_inputs`

## Deferred Work

Items identified during agent-side tool-calling debugging on 2026-04-19 but not yet applied:

### (b) Strip thinking content from multi-turn history

The official Gemma 4 model card specifies:

> When providing multi-turn conversations as input, you should NOT include previous thinking content. For the E2B and E4B instruction-tuned variants, the thinking wrapper is NOT emitted in the response when thinking is disabled.

`GemmaProtocol::format_prompt_with_tools` in `crates/gallium-agent/src/protocol.rs` currently replays every prior assistant turn verbatim. If the model emitted a `<|think|> … <|/think|>` block (or a `<|channel>thinking … <channel|>` block) on a previous turn, that block is re-fed into the model's context on the next turn. This likely contributes to thinking loops observed when `--thinking` was enabled against the E4B model.

**Fix sketch:** in `format_prompt_with_tools`, when rendering a prior `ChatMessage::Assistant`, strip any text between `<|think|>`/`<|/think|>` markers (and/or `<|channel>` … `<channel|>` blocks whose channel name is `thinking`) before emitting it. Leave the current (last) turn alone — only strip *history*. Preserve tool-call and tool-response markers untouched.

**Why deferred:** want to confirm the `--top-k 64 / --top-p 0.95 / --temperature 1.0` sampling changes alone fix the coding test before adding more protocol complexity.

### (c) Attention scale confirmation — already resolved

During the backend review, the `scale = 1.0` (no `1/√d` factor) at `gemma4_q.rs:116` looked suspect vs. standard attention. **This is intentional and already documented above (see "Attention Scale = 1.0" and Bug 1).** The reference `Gemma4TextAttention` sets `self.scaling = 1.0`, compensated by the Q/K RMSNorms. Using `1/√d` here made output incoherent in earlier development. Leaving this note so future reviewers don't re-open the investigation.
