# Target Model Architectures

## GPT-OSS (OpenAI)

**Paper/Source**: https://github.com/openai/gpt-oss

**Sizes**: 20B (3.6B active), 120B (5.1B active)

### Architecture

| Component | Details |
|-----------|---------|
| Attention | GQA: 64 Q heads, 8 KV heads, head_dim=64. Alternating full + sliding-window (128 tokens) per `layer_types` array |
| FFN | MoE: 32 experts (20B) / 128 experts (120B), top-4 routing. SwiGLU with `swiglu_limit=7.0` clamp |
| Position | RoPE theta=150000, YaRN scaling (factor=32x from 4K base) for 128K context |
| Normalization | Pre-LN RMSNorm, eps=1e-5 |
| Bias | attention_bias=true (unusual) |

### Key Novelties
- **Attention sinks via learnable bias**: softmax denominator bias lets heads attend to nothing
- **128-token sliding window**: much smaller than typical 1K-4K windows
- **MXFP4 quantization-aware post-training**: MoE weights at ~4.25 bits
- **SwiGLU clamping**: `clamp(-7.0, 7.0)` for numerical stability at low precision

### Implementation Notes
- `layer_types: Vec<LayerType>` from config.json determines full vs sliding per layer
- Same attention config for both types, only the mask differs
- Weight prefix: `model.layers.{i}.block_sparse_moe.experts.{j}.{gate,up,down}_proj`

---

## Qwen 3.5 (Alibaba)

**Paper/Source**: https://github.com/QwenLM/Qwen3.5

**Sizes**: 0.8B, 2B, 4B, 9B, 27B (dense), 35B-A3B, 122B, 397B-A17B (MoE)

### Architecture

| Component | Details |
|-----------|---------|
| Attention (full) | GQA with output gate, head_dim=256 |
| Attention (linear) | Gated DeltaNet: key_head_dim=128, value_head_dim=128, conv_kernel=4 |
| Layer pattern | 3 linear + 1 full (repeating), configured by `layer_types` array |
| FFN | SwiGLU (dense) or MoE with shared experts (256-512 experts, top 8-10) |
| Position | RoPE theta=10M, M-RoPE sections [11,11,10], partial_rotary_factor=0.25 |
| Normalization | RMSNorm, eps=1e-6 |

### Key Novelties
- **Gated DeltaNet**: O(n) linear attention with delta update rule + gating, replaces 75% of full attention layers
- **Hybrid attention**: mix of linear (recurrent) and full (quadratic) attention
- **M-RoPE**: multimodal RoPE with separate sections for spatial/temporal dims
- **Multi-Token Prediction**: trained to predict multiple next tokens (enables speculative decoding)
- **Shared experts**: MoE variants have a shared expert that all tokens pass through

### Implementation Notes
- `layer_types` array: `"full_attention"` or `"linear_attention"` per layer
- Linear layers use `RecurrentState` (not KV cache)
- DeltaNet includes short causal convolution (kernel=4) on Q input
- Full attention layers may have `attn_output_gate=true` (sigmoid gate on output)

---

## Gemma 4 (Google)

**Paper/Source**: https://ai.google.dev/gemma/docs/core/model_card_4

**Sizes**: E2B, E4B, 26B-A4B (MoE), 31B (dense)

### Architecture

| Component | Details |
|-----------|---------|
| Attention (sliding) | GQA, standard RoPE (theta=10K), sliding window (512-1024 tokens) |
| Attention (global) | GQA with shared K=V, Q-norm, proportional RoPE (theta=1M, partial_rotary=0.25) |
| Layer pattern | 5 sliding + 1 global (repeating), last layer always global |
| FFN | GeGLU (GELU with tanh approx) or MoE (128 experts, 8+1 active) |
| Normalization | RMSNorm, eps=1e-6 |
| Embedding | scaled by sqrt(hidden_size) |
| Logits | final softcapping at 30.0 |

### Key Novelties
- **Dual RoPE**: different theta per attention type (10K for sliding, 1M for global)
- **Shared K=V**: global layers share a single projection for keys and values (`attention_k_eq_v`)
- **Q normalization**: per-head RMSNorm on Q before RoPE (global layers)
- **Per-Layer Embeddings (PLE)**: learned per-layer conditioning vectors added to hidden states
- **KV cache sharing**: last N layers reuse KV cache from earlier layers of the same type
- **Logit softcapping**: `tanh(logits / 30) * 30` on final output (re-introduced from Gemma 2)
- **Variable head_dim**: global layers may use larger head_dim (e.g., 512 vs 256)

### Implementation Notes
- `global_attention_interval` in config (default 6 = 5 sliding + 1 global)
- Last layer is always global regardless of interval
- `num_kv_shared_layers` specifies how many trailing layers share KV with earlier layers
- PLE tensor shape: `(hidden_size,)` per layer, loaded from `model.layers.{i}.per_layer_embedding`
- Weight prefix for V: may be absent when `attention_k_eq_v=true` (K projection used for both)
