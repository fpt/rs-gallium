# Gemma 4 Implementation Notes

Implementation notes for `crates/gallium-models/src/gemma4.rs` (safetensors) and `gemma4_q.rs` (GGUF).

Model: `google/gemma-4-E4B` (text component of a multimodal model).

---

## Testing All Features with Gemma 4 E2B

Gemma 4 E2B (2B active parameters) is a good end-to-end testbed because it is small enough to run comfortably on Apple Silicon.

Local Gemma 4 inference goes through the `gallium-agent` binary (GGUF text inference, ReAct tools, sessions, skills).

### Local text inference — gallium-agent binary

Tests: GGUF loading, quantized inference, ReAct tools, session persistence, skill system.

**Prerequisites:** model downloads from HuggingFace on first run (~1.5 GB).

```bash
make run-gemma4-e2b-gguf
# Override sampling:
make run-gemma4-e2b-gguf MAX_TOKENS=512 TEMPERATURE=0.7
```

Expected: a `>` prompt. Type a question, press Enter. Use `/reset` to clear history, `/quit` to exit.

```
> What is the capital of France?
Paris is the capital of France.
[in=12 out=9 ctx=0%]
```

---

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

## Vision Architecture (`gemma4_vision.rs`)

Implementation notes for `crates/gallium-models/src/gemma4_vision.rs`.

The multimodal entry point is `Gemma4Multimodal`, which owns a `Gemma4` text model plus vision components. It implements `CausalLM` with an additional `encode_image` / `set_image_features` API for prefilling image features.

### Vision Encoder Config

| Component | Value |
|-----------|-------|
| Layers | 16 |
| Hidden size | 768 |
| Q/K/V heads | 8 each |
| Head dim | 96 (768/8) |
| Intermediate size | 3072 |
| Patch size | 16×16 pixels |
| Image size | 896×896 (default) |
| Position embedding size | 8192 |
| RoPE theta | 10000 |
| Pooling kernel size | 4×4 (pooling_kernel_size=4) |
| Attention scale | 1.0 (explicit, matches text model) |
| Normalization | 4-norm block; pre-attn, post-attn, pre-MLP, post-MLP |
| V norm | RMSNorm **without learnable scale** |

### Weight Namespace

All vision weights live under `model.vision_tower.*`. The projector lives under `model.embed_vision.*`.

```
model.vision_tower.vision_model.embeddings.patch_embedding.weight   [768, 3, 16, 16]
model.vision_tower.vision_model.embeddings.patch_embedding.bias     [768]
model.vision_tower.vision_model.embeddings.position_embedding.weight [8192, 768]

model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.linear.weight
model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.linear.bias
... (k_proj, v_proj, out_proj — all with .linear.weight / .linear.bias)
model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.linear.weight
model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.linear.weight
model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight
model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight
model.vision_tower.vision_model.encoder.layers.{i}.layer_norm3.weight
model.vision_tower.vision_model.encoder.layers.{i}.layer_norm4.weight

model.embed_vision.embedding_projection.weight                        [2560, 768]
```

**Critical: `use_clipped_linears=True`** — all attention and MLP weights in the vision encoder are wrapped in an extra `.linear.` level: `q_proj.linear.weight`, not `q_proj.weight`. Bias tensors also live at `q_proj.linear.bias`. This matches how `ClipLinear` weights are serialized for the E4B checkpoint.

The projector has **no bias** (bias=false in `embedding_projection`).

### Block Structure (4-norm)

```
1. residual = h
   h = layer_norm1(h)
   h = self_attn(h)      # 2D RoPE, scale=1.0, v_norm-no-scale
   h = layer_norm2(h)
   h = residual + h

2. residual = h
   h = layer_norm3(h)
   h = mlp(h)            # fc1 (gelu) + fc2 — no gate split; fc1 and fc2 are full-width
   h = layer_norm4(h)
   h = residual + h
```

Unlike the text model (GeGLU), the vision MLP uses plain GELU: `fc2(gelu(fc1(h)))`, no gate projection.

### 2D Spatial RoPE

The vision encoder uses **2D spatial RoPE** rather than sequential 1D RoPE. Each patch at pixel-position `(x, y)` gets:
- **First `head_dim/2 = 48` dimensions** of Q/K rotated by the *x*-coordinate
- **Last `head_dim/2 = 48` dimensions** of Q/K rotated by the *y*-coordinate

`inv_freq` is computed in the standard way but for half the head dim:

```
inv_freq[j] = 1 / (theta ^ (2j / 32))   for j in 0..16
```

(32 = `head_dim / 2 / num_inv_freq_elements`; there are 16 inv_freq values covering 16 pairs × 2 = 32 slots = `head_dim/2` dims.)

The caller supplies `pixel_position_ids: [batch, seq, 2]` where `[:, :, 0]` is the x-patch index and `[:, :, 1]` is the y-patch index. The RoPE module computes `(cos, sin)` of shape `[batch, seq, head_dim/2]` by applying `inv_freq` to both x and y, then concatenating:

```
cos_x = cos(x * inv_freq)   [b, s, 16]
sin_x = sin(x * inv_freq)   [b, s, 16]
cos_y = cos(y * inv_freq)   [b, s, 16]
sin_y = sin(y * inv_freq)   [b, s, 16]
cos = cat([cos_x, cos_y], dim=-1)   [b, s, 32]  ← expanded to head_dim/2=48? see impl
sin = cat([sin_x, sin_y], dim=-1)
```

Applied via `apply_rotary(q, cos, sin)` using the standard `rotate_half` formula (swap and negate the second half of the last axis).

### Pooler

`VisionPooler` reduces the `H×W` patch grid to `(H/k)×(W/k)` features by spatial average pooling with kernel size `k=pooling_kernel_size=4`. After pooling, features are scaled by `√hidden_size = √768 ≈ 27.7` to match the language model's embedding scale convention.

Implementation is CPU-based (scatter-average):
1. Sort patches into spatial buckets of size `k×k` using their pixel-position indices.
2. Average the `hidden_size`-dim vectors in each bucket.
3. Multiply the result by `√hidden_size`.

This runs only once per image (at prefill) so throughput is acceptable.

### Projector

`VisionProjector` (a.k.a. `Gemma4MultimodalEmbedder`) maps pooled vision features `[n_patches, 768]` to language model space `[n_patches, 2560]`:

```
features = rms_norm_no_scale(features)       # normalize without learnable scale
features = Linear(768 → 2560, no bias)(features)
```

The RMSNorm here has no learnable weight (same convention as the text model's `v_norm`). The linear weight is `model.embed_vision.embedding_projection.weight [2560, 768]`.

### `Gemma4Multimodal` Integration

```
Gemma4Multimodal {
    text: Gemma4,
    patch_embedder: VisionPatchEmbedder,
    encoder: VisionEncoder,
    pooler: VisionPooler,
    projector: VisionProjector,
    image_token_id: u32,       // 258880
    pooling_kernel_size: usize, // 4
    pending_image_embeds: Option<Tensor>,
    device: Device,
}
```

**Usage pattern:**

```rust
// 1. Before prefill, encode the image and store features
let vision_feats = model.encode_image(pixel_values, pixel_position_ids)?;
model.set_image_features(vision_feats);

// 2. Prefill with tokens containing image placeholders (token_id=258880)
//    forward() detects seq_len > 1 (prefill), extracts pending_image_embeds,
//    replaces placeholder embeddings with projected vision features, then runs the LM.
let logits = model.forward(&token_ids, 0)?;

// 3. Decode normally (no image injection needed)
let logits = model.forward(&next_token_ids, seq_len)?;
```

**Image feature injection** (`inject_image_features`):
- Build initial embeddings by calling `text.embed_scaled(token_ids)`.
- For each position where `token_id == image_token_id (258880)`, overwrite that row in `inputs_embeds` with the corresponding projected vision feature.
- Implemented via CPU `to_vec` scatter because Candle lacks `masked_scatter`. Acceptable at batch=1.
- `pending_image_embeds` is `None`-d out immediately after consumption to avoid accidental re-use.

### Implementation Notes

- **`rms_norm_no_scale`**: Used for `v_norm` in vision attention AND the projector's pre-norm. Implemented as `x / sqrt(mean(x²) + eps)` with no multiplication by a weight tensor.
- **Bidirectional attention**: The vision encoder uses full (non-causal) attention — no attention mask is applied.
- **Position embeddings**: The vision encoder also has a learnable `position_embedding` table `[8192, 768]` looked up by a flat patch index (0-based, row-major). This is added to patch embeddings before the encoder, separate from the 2D spatial RoPE applied inside each attention layer.
- **`CausalLM::reset()`**: Clears `pending_image_embeds` in addition to resetting the text model KV cache.

## Reference

- `references/transformers/src/transformers/models/gemma4/modeling_gemma4.py`
- Key functions: `Gemma4TextAttention.forward`, `Gemma4TextDecoderLayer.forward`, `project_per_layer_inputs`, `get_per_layer_inputs`
- Vision: `Gemma4VisionAttention.forward`, `Gemma4VisionEncoderLayer.forward`, `SiglipVisionEmbeddings.forward`, `Gemma4MultimodalProjector.forward`

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
