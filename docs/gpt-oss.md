# GPT-OSS Implementation Notes

Implementation notes for `crates/gallium-models/src/gpt_oss.rs` (safetensors) and `gpt_oss_q.rs` (GGUF).

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 24 |
| Hidden size | 2880 |
| Q heads / KV heads | 64 / 8 (GQA) |
| Head dim | 64 (note: ≠ hidden/n_heads = 45) |
| FFN | MoE, 32 experts, top-4, intermediate_size=2880 |
| Attention bias | true (all projections including o_proj) |
| RoPE | theta=150000, YaRN factor=32, base_context=4096 |
| Context length | 131072 |
| Sliding window | 128 tokens |
| Layer types | alternating full/sliding per `layer_types` in config |
| RMSNorm eps | 1e-5 |

## Non-standard Components

### 1. Attention Sinks

Each attention layer has a learnable `sinks` parameter of shape `[n_heads]`. It acts as a per-head "virtual key" that absorbs excess probability from the softmax.

**Forward pass:**
1. Compute raw attention scores `[b, n_heads, q_len, kv_len]` as usual (scale, mask applied).
2. Expand sinks to `[b, n_heads, q_len, 1]` — same scalar value for all query positions.
3. Concatenate along the key dimension: `combined = cat([scores, sinks], dim=-1)` → `[..., kv_len+1]`.
4. Apply softmax over the combined last dimension.
5. Drop the last column (the sink probability): `probs[..., :-1]`.
6. Use truncated probs for the V matmul as normal.

**Why it matters:** Without sinks, softmax weights over attended positions sum to exactly 1.0. With sinks, the model can learn to "discard" up to all of the attention probability for a given head, enabling controlled sparsity.

**Tensor locations:**
- Safetensors: `model.layers.{i}.self_attn.sinks`, shape `[64]`, dtype BF16
- GGUF: `blk.{i}.attn_sinks.weight`, shape `[64]`, dtype F32

**Implementation:** `AttentionConfig.attn_sinks = true` enables sink loading in `Attention::new`. Set in `gpt_oss.rs` when building `attn_cfg`. For GGUF, loaded directly in `QAttention::load`.

**Contiguity note:** After `sinks.reshape(...).expand(...)`, call `.contiguous()` before `Tensor::cat`. The `expand` produces a non-contiguous view that causes errors in subsequent ops.

### 2. MoE FFN Activation (not SwiGLU)

The GPT-OSS FFN looks like SwiGLU but has important differences. The activation function is:

```
glu  = gate * sigmoid(gate * 1.702)      # scaled-sigmoid, ≈ GELU
output = (up + 1) * glu                  # up shifted by +1
```

**Full per-expert computation:**
1. `gate_up = x @ W_gate_up.T + b_gate_up`   — joint projection
2. Split gate/up **interleaved**: `gate = gate_up[..., ::2]`, `up = gate_up[..., 1::2]`
3. `gate = clamp(gate, max=7.0)`             — one-sided clamp (no min)
4. `up   = clamp(up, -7.0, 7.0)`             — two-sided clamp
5. `glu  = gate * sigmoid(gate * 1.702)`
6. `out  = (up + 1) * glu @ W_down.T + b_down`

**Key differences from standard SwiGLU:**

| Property | GPT-OSS | Standard SwiGLU |
|----------|---------|-----------------|
| Gate/up split | interleaved (`::2` / `1::2`) | contiguous (first/second half) |
| Gate activation | `gate * sigmoid(gate * 1.702)` | `silu(gate)` |
| Gate clamp | max-only | both sides |
| Up clamp | both sides | none |
| Up shift | `up + 1` | none |

**Safetensors weight layout:** `gate_up_proj` has shape `[n_experts, hidden, 2*inter]` (note: `is_transposed=True` in the model code means the weight matrix rows correspond to hidden dim). Gate values occupy even output indices, up values occupy odd output indices. In the Rust code, after computing `tx @ gu_w.T`, reshape `[1, 2*inter]` → `[1, inter, 2]` and narrow along the last dimension.

**GGUF weight layout:** The unsloth converter pre-splits gate and up into separate tensors (`ffn_gate_exps` and `ffn_up_exps`), so no interleaved split is needed. Apply the same activation formula.

**sigmoid implementation in Rust:** Use the tanh identity `sigmoid(x) = (1 + tanh(x/2)) / 2`:
```rust
// sigmoid(gate * 1.702) = (1 + tanh(gate * 0.851)) / 2
let sig = ((&gate * 0.851_f64)?.tanh()? + 1.0_f64)? * 0.5_f64;
let glu = (gate * (sig)?)?;
```

### 3. MoE Router

The router selects top-k experts and normalizes their weights:

```
logits = x @ W_router.T + b_router       # shape [n_tokens, n_experts]
top_k_logits = topk(logits, k=4)
weights = softmax(top_k_logits, dim=-1)   # normalized over selected experts only
```

Our implementation uses full softmax over all experts then selects top-k and renormalizes — this is equivalent since `normalize(top-k of softmax(all)) = softmax(top-k logits)`.

## MXFP4 Expert Weights

In the safetensors model, MoE expert weights use OpenAI's MXFP4 (MX Float4 E2M1) block quantization:

- **Block size:** 32 elements per block
- **Block layout:** 1 byte E8M0 scale + 16 bytes of packed 4-bit E2M1 values (2 nibbles/byte)
- **Scale:** `2^(exponent_byte - 127)`
- **E2M1 LUT:** `[0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]` for positive codes
- **Storage:** `gate_up_proj_blocks [n_exp, 2*inter, n_blocks, 16]` and `gate_up_proj_scales [n_exp, 2*inter, n_blocks]`

In GGUF, these are stored as dtype 39 (custom type) with the same 17-bytes-per-32-elements layout. Loaded via `Tq2Tensor` with lazy per-expert dequantization at forward time to avoid materializing all expert weights simultaneously.

## Debugging History

These bugs caused garbage/NaN output and were identified by comparing hidden-state RMS norms layer by layer against the reference transformers implementation.

**Bug 1 — Wrong FFN activation:** Using `silu(gate)` instead of `gate * sigmoid(gate * 1.702)` and a contiguous split instead of interleaved. This caused the FFN output RMS to grow unboundedly across layers (from ~2.8 at L0 to ~625 at L23), making the final logits nearly uniform and the output incoherent.

**Bug 2 — Missing attention sinks:** Without sinks, attention weights summed to exactly 1.0 per query. With sinks trained into the model, the actual expected sum is < 1.0. The missing probability mass inflated attn outputs by ~28% per layer, compounding to ~100-180x over 24 layers and further distorting the residual stream direction.

**Verification:** Running with `SKIP_BLOCKS=1` (bypasses transformer layers, bigram-only) produced coherent logits with top-1 confidence ~25 vs ~8 with broken blocks — confirming both bugs were in the block computation, not embeddings or lm_head.

**Bug 3 — `layer_types` fallback inverted in the GGUF loader (2026-04-20, `gpt_oss_q.rs:325`).**
The unsloth GGUF does not set `gpt-oss.attention.layer_type`, so `GptOssQ::load` hit the fallback and assigned `i%2==0 → full_attention`, `i%2==1 → sliding_attention`. HF transformers (`configuration_gpt_oss.py:73`: `"sliding_attention" if bool((i+1)%2) else "full_attention"`) and ollama (`model/models/gptoss/model.go:38–39` with SWA cache at index 0 of the WrapperCache) both agree that **even layers are sliding, odd layers are full**. Because the GGUF never carries the explicit array, the swap silently ruins every layer's mask. Symptoms were short-prompt sanity was fine but long prompts (e.g. the coding test at 484 tokens) produced incoherent `We We We…` output; ollama on the exact same tokenised prompt produced the expected tool call.

**Bug 4 — No sliding-window mask at `seq_len=1` during decode (`gpt_oss_q.rs:416` and `gpt_oss.rs:342`).**
`forward` short-circuited `mask = None` whenever `seq_len <= 1`. That is correct for full-attention layers at decode time, but a sliding-window layer whose KV cache has grown past the window still needs a mask — otherwise each new query attends to all past K/V instead of just the last `sliding_window` tokens. Prefill was fine (masks were built there); divergence appeared on the first generated token. Fixed to keep building the SW mask while `pos + seq_len > sliding_window`.

**Bug 5 — `<|end|>` in the provider's EOS set (`provider.rs:57–71`).**
`GalliumProvider::new` matched `k.contains("<|end")` and pulled `<|end|>` into `eos_tokens`. In Harmony, `<|end|>` is a *channel separator* (analysis → commentary → final), not a turn terminator — the turn ends on `<|return|>` (plain chat) or `<|call|>` (tool call). With the bug, generation stopped the moment the model closed the `analysis` channel, before emitting the actual tool call. Fixed by narrowing the match to `<|endoftext|>` and adding `<|return|>` explicitly.

**Bug 6 — `parse_harmony_tool_call` used `rfind('{')` for the JSON start (`protocol.rs:277`).**
The Harmony tool call arguments are serialized JSON; for coding-tool calls the `content` field often contains a literal `{` (e.g. `func main() {`). `rfind` landed inside the string, so `serde_json::from_str` got a fragment and returned `None`, and the agent silently dropped the call. Fixed to `find('{')` after the `functions.<name>` marker while keeping `rfind('}')` for the close.

**How these four were found.** Ran the ollama-narrow skill (`.claude/skills/ollama-narrow/SKILL.md`): dumped our Harmony prompt, fed the byte-identical prompt to ollama's `gpt-oss:20b` at `raw:true, temperature:0.0`, compared first-token outputs, and then read ollama's `model/models/gptoss/model.go` side-by-side with `gpt_oss_q.rs`. The layer_type inversion fell out of that diff; the SW decode-mask and stop-token bugs fell out of rerunning the testcase after each fix.

## Reference

- `external/` contains the original `modeling_gpt_oss.py` from the transformers library.
- `_apply_gate` in that file defines the exact activation formula.
- `eager_attention_forward` defines the sinks concatenation.
- `GptOssTopKRouter.forward` defines the router (top-k then softmax, not softmax-then-top-k).
