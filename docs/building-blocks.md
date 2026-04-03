# Building Blocks Reference

All building blocks live in `crates/gallium-core/src/`.

## Attention (`attention.rs`)

Standard multi-head attention supporting MHA, GQA, and MQA via the `num_kv_heads` ratio.

```rust
AttentionConfig {
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,      // == num_q_heads → MHA, == 1 → MQA, else GQA
    head_dim: usize,
    attn_bias: bool,           // projection bias (GPT-OSS)
    attn_logit_softcapping: Option<f64>,  // tanh softcapping (Gemma 4)
    shared_kv: bool,           // K=V shared projection (Gemma 4 global)
    q_norm: bool,              // per-head Q normalization (Gemma 4)
    q_norm_eps: f64,
}
```

**Sliding window**: handled via mask, not a separate attention struct. Pass `build_sliding_window_mask()` as the mask argument.

## Linear Attention (`linear_attn.rs`)

Gated DeltaNet: O(n) linear attention with delta update rule.

```rust
DeltaNetConfig {
    hidden_size: usize,
    num_heads: usize,
    key_head_dim: usize,       // typically 128
    value_head_dim: usize,     // typically 128
    conv_kernel_dim: usize,    // short causal conv, typically 4
    output_gate: bool,         // output gating (Qwen 3.5)
}
```

Uses `RecurrentState` instead of KV cache. Each head maintains a `(key_dim, value_dim)` state matrix updated via delta rule.

## Feed-Forward Networks (`ffn.rs`)

### GatedFFN

Gated linear unit with configurable activation:
```rust
GatedFFN::new(hidden_size, intermediate_size, activation, clamp, vb)
```

Computes: `down_proj(activation(gate_proj(x)) * up_proj(x))`

- `Activation::Silu` → SwiGLU (Llama, GPT-OSS, Qwen)
- `Activation::GeluTanh` → GeGLU (Gemma 4)
- `clamp: Some(7.0)` → numerical stability (GPT-OSS)

### MoEFFN

Mixture of Experts with top-k routing:
```rust
MoEFFN::new(
    hidden_size, intermediate_size,
    num_experts, num_experts_per_tok,
    activation, clamp,
    shared_expert_intermediate,  // Option<usize> for shared expert (Qwen 3.5 MoE)
    vb
)
```

## Normalization (`norm.rs`)

```rust
Norm::rms(size, eps, vb)    // RMSNorm
Norm::layer(size, eps, vb)  // LayerNorm
```

## Position Encodings (`pos_enc.rs`)

### RoPE

```rust
RoPEConfig {
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,                    // base frequency (10000, 150000, 1M, 10M, ...)
    scaling: RoPEScaling,          // None, YaRN, Linear, Llama3, NTK
    partial_rotary_factor: f64,    // 1.0 = full, 0.25 = partial (Qwen 3.5, Gemma 4 global)
    freq_factors: Option<Vec<f64>>, // per-dim factors (Gemma 4 proportional RoPE)
}
```

**Scaling variants**:
- `RoPEScaling::None` - standard RoPE
- `RoPEScaling::YaRN { factor, original_max_position_embeddings, beta_fast, beta_slow }` - GPT-OSS
- `RoPEScaling::Linear { factor }` - simple linear interpolation
- `RoPEScaling::Llama3 { factor, low_freq_factor, high_freq_factor, original_max_position_embeddings }`
- `RoPEScaling::NTK { factor }` - Neural Tangent Kernel scaling

## Transformer Block (`block.rs`)

```rust
TransformerBlock {
    pre_attn_norm: Norm,
    attn: AttnImpl,                    // Standard or LinearDeltaNet
    post_attn_norm: Norm,
    ffn: FfnImpl,                      // Gated or MoE
    per_layer_embed: Option<Tensor>,   // PLE (Gemma 4)
}
```

Flow: `pre_norm → attn → +residual → post_norm → ffn → +residual`

## KV Cache (`kv_cache.rs`)

```rust
// Standard: concatenates K,V across generation steps
KvCache::new(max_seq_len)

// Recurrent: hidden state for linear attention
RecurrentState::new()

// Per-model collection
ModelCache::new(vec![
    LayerCache::Kv(KvCache::new(4096)),          // standard attention layer
    LayerCache::Recurrent(RecurrentState::new()), // linear attention layer
    LayerCache::Shared { source_layer: 0 },       // shares KV with layer 0
])
```

## Masks (`mask.rs`)

```rust
build_causal_mask(seq_len, pos, device)            // standard causal
build_sliding_window_mask(seq_len, pos, window, device)  // sliding window + causal
```

Both return `(seq_len, total_len)` tensors with 0.0 (attend) / -inf (block).

## Sampling (`sampling.rs`)

```rust
SamplingParams {
    temperature: f32,          // 0.0 = greedy
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
}

sample(&logits, &params, &previous_tokens) -> Result<u32>
```

## Generation (`model.rs`)

```rust
generate(
    model: &mut dyn CausalLM,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    max_new_tokens: usize,
    eos_tokens: &[u32],
    on_token: impl FnMut(u32),  // streaming callback
) -> Result<Vec<u32>>
```

## TurboQuant (`turbo_quant.rs`)

Near-optimal vector quantization from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). Compresses KV cache vectors to 2-4 bits per coordinate with minimal distortion.

```rust
TurboQuantConfig {
    bit_width: usize,      // 1-4 bits per coordinate
    dim: usize,            // vector dimension (e.g., head_dim)
    mode: TurboQuantMode,  // Mse or InnerProduct
    seed: u64,             // for reproducible random matrices
}

let tq = TurboQuant::new(&cfg, &device)?;
let quantized = tq.quantize(&x)?;     // (..., dim) -> TurboQuantized
let x_hat = tq.dequantize(&quantized)?; // TurboQuantized -> (..., dim)
```

**Two modes**:
- `Mse`: optimal MSE distortion. Randomly rotates, scalar-quantizes each coordinate using Lloyd-Max codebooks, rotates back.
- `InnerProduct`: unbiased inner product estimation. Uses (b-1)-bit MSE + 1-bit QJL on residual.

**Distortion bounds** (from paper, unit vectors):
| Bits | MSE | Inner Product Error |
|------|-----|-------------------|
| 1 | 0.36 | 1.57/d |
| 2 | 0.117 | 0.56/d |
| 3 | 0.03 | 0.18/d |
| 4 | 0.009 | 0.047/d |

## TurboKvCache (`turbo_kv_cache.rs`)

Drop-in replacement for `KvCache` that stores K,V in TurboQuant-compressed form. 5-8x memory reduction.

```rust
let cfg = TurboQuantConfig { bit_width: 3, dim: head_dim, mode: Mse, seed: 42 };
let cache = TurboKvCache::new(&cfg, max_seq_len, &device)?;

// Use as LayerCache variant:
LayerCache::TurboKv(cache)
```

Attention code works unchanged — `TurboKvCache::append()` returns dequantized tensors.
