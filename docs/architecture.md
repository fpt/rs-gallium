# rs-gallium Architecture

## Overview

rs-gallium is a Rust LLM inference framework designed for **simplicity** and **rapid implementation of novel architectures from research papers**.

```
rs-gallium/
├── crates/
│   ├── gallium-core/       # Composable building blocks + generation
│   ├── gallium-models/     # Concrete model implementations
│   ├── gallium-cli/        # One-shot inference CLI
│   └── gallium-agent/      # Interactive ReAct agent
└── docs/                # Documentation
```

## Design Principles

### 1. Concrete structs over traits

Most components are concrete structs with enum dispatch for variants. Only one trait exists: `CausalLM` for the top-level model interface. This keeps code navigable -- you can always click through to any implementation directly.

### 2. Per-layer heterogeneous configuration

Modern architectures (GPT-OSS, Qwen 3.5, Gemma 4) use different attention types per layer. rs-gallium treats per-layer variance as first-class: each layer can have its own attention type, RoPE config, and FFN type.

### 3. Candle as tensor backend

We use `candle-core` and `candle-nn` for tensor operations. This gives CPU, CUDA, and Metal support without reimplementing low-level compute. Our framework focuses on the model-level abstractions on top.

### 4. Paper-mapping

Components map directly to how papers describe architectures:
- "We use grouped-query attention" → `AttentionConfig { num_kv_heads: 8, .. }`
- "SwiGLU activation with gating" → `GatedFFN { activation: Activation::Silu, .. }`
- "Rotary embeddings with theta=1M" → `RoPEConfig { theta: 1_000_000.0, .. }`

## Crate Responsibilities

### gallium-core

All reusable building blocks. Zero model-specific code. Modules:

| Module | Purpose |
|--------|---------|
| `attention.rs` | Standard attention: MHA, GQA, MQA, with optional sliding window mask, logit softcapping, shared K=V, Q-norm |
| `linear_attn.rs` | Gated DeltaNet: O(n) linear attention with delta update rule and short causal convolution |
| `ffn.rs` | GatedFFN (SwiGLU/GeGLU) with optional clamp, MoEFFN with top-k routing and optional shared expert |
| `norm.rs` | RMSNorm, LayerNorm wrappers |
| `pos_enc.rs` | RoPE with scaling variants: standard, YaRN, Linear, Llama3, NTK; supports partial rotary and per-dim frequency factors. **Note:** GGUF `rope_freqs.weight` stores per-dim DIVISORS (`inv_freq[i] = base / factor[i]`), not `inv_freq` itself — see `docs/gemma4.md` Bug 10 |
| `block.rs` | TransformerBlock: pre-norm -> attn -> residual -> post-norm -> ffn -> residual |
| `kv_cache.rs` | KV cache (per-layer, with cross-layer sharing), RecurrentState for linear attention |
| `mask.rs` | Causal mask and sliding-window mask builders |
| `sampling.rs` | Greedy, top-k, top-p, temperature, repetition penalty |
| `model.rs` | `CausalLM` trait and `generate()` function |

### gallium-models

Concrete model definitions. Each model file is ~150-200 lines because it delegates to gallium-core blocks:

| Model | File | Key Features |
|-------|------|-------------|
| GPT-OSS | `gpt_oss.rs` | Alternating full/sliding-window attn, MoE with SwiGLU + clamp, YaRN RoPE |
| Qwen 3.5 | `qwen35.rs` | Hybrid DeltaNet (linear) + full attention, MoE with shared experts |
| Gemma 4 | `gemma4.rs` | Dual RoPE, shared K=V, Q-norm, PLE, logit softcapping, KV cache sharing |

### gallium-cli

Thin binary: parse args, load tokenizer + model, run generation loop, stream output to stdout.

### gallium-agent

Interactive multi-turn ReAct agent. Modules:

| Module | Purpose |
|--------|---------|
| `llm.rs` | `LlmProvider` trait + `OpenAiProvider` (Responses API) |
| `memory.rs` | `ConversationMemory`: multi-turn history, token-based compaction |
| `tool.rs` | `ToolHandler` trait, `ToolRegistry`, built-in tools: `read`, `glob`, `tasks` |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `protocol.rs` | `ModelProtocol` trait + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol` |
| `provider.rs` | `GalliumProvider`: wraps a local `CausalLM`, delegates format/parse to protocol |
| `agent.rs` | `Agent`: orchestrates provider, memory, tools; routes to ReAct or plain chat |
| `main.rs` | REPL CLI — reads stdin, streams responses, handles `/reset`, `/help`, `/quit` |

Two execution paths depending on provider:

```
# Gallium provider (local model, supports_tools = false)
user input → memory.add → protocol.format_prompt(history) → generate() → protocol.parse_response() → response

# OpenAI provider (supports_tools = true)
user input → memory.add → format messages → ReAct loop:
    ├── client.chat_with_tools(messages, tool_defs)
    ├── if ToolCalls: execute each tool → append results → loop
    └── if Text: return response
```

The full conversation history is re-prefilled on every turn for the Gallium provider (no incremental KV cache across turns). `generate()` calls `model.reset()` internally.

### Protocol Adapters

`ModelProtocol` is the adapter layer between the agent's generic `ChatMessage` history and each model's raw prompt/response format. Each implementation handles two responsibilities:

1. **`format_prompt`** — renders a `Vec<ChatMessage>` into the model-specific token string
2. **`parse_response`** — extracts the user-facing reply from raw decoded output

| Protocol | Model | Format | Parse |
|---|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + `Valid channels` instructions; `<\|start\|>role<\|channel\|>ch<\|message\|>content<\|end\|>` | Extracts `final` channel, discards `analysis`/`commentary` |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template | Passthrough trim |
| `QwenProtocol` | Qwen 3.5 | ChatML `<\|im_start\|>role` template | Passthrough trim |

#### Harmony channel format (GPT-OSS)

GPT-OSS is trained on the [Harmony protocol](https://github.com/openai/harmony) and requires it to produce coherent output. The model writes to named channels per turn:

```
<|start|>assistant<|channel|>analysis<|message|>REASONING<|end|>
<|start|>assistant<|channel|>final<|message|>ANSWER<|end|>
```

After `tokenizer.decode(skip_special=true)`, special tokens are stripped but channel names remain as plain text:

```
analysis
<reasoning...>
assistant
final
<answer>
```

`HarmonyProtocol::parse_response` finds the last line containing exactly `"final"` and returns everything after it. This prevents verbose reasoning from being stored in memory and confusing subsequent turns.

## Data Flow

```
prompt text
    │
    ▼
tokenizer.encode()
    │
    ▼
[CausalLM::forward] ◄── prefill (all prompt tokens at once)
    │
    ├─ embed_tokens(token_ids)
    │
    ├─ for each layer:
    │   ├─ pre_attn_norm(x)
    │   ├─ attention(x, rope, kv_cache, mask)  ← or DeltaNet(x, recurrent_state)
    │   ├─ residual connection
    │   ├─ post_attn_norm(x)
    │   ├─ ffn(x)  ← GatedFFN or MoEFFN
    │   └─ residual connection
    │
    ├─ final_norm(x)
    ├─ lm_head(x) → logits
    └─ optional softcapping
    │
    ▼
sample(logits, params) → next token
    │
    ▼
[CausalLM::forward] ◄── decode (one token, using KV cache)
    │
    ▼
... repeat until EOS or max_tokens
    │
    ▼
tokenizer.decode(generated_tokens)
```

## Key Types

```rust
// The one trait in the framework
pub trait CausalLM {
    fn forward(&mut self, token_ids: &Tensor, pos: usize) -> Result<Tensor>;
    fn reset(&mut self);
    fn device(&self) -> &Device;
}

// Attention dispatch
pub enum AttnImpl {
    Standard(Attention),       // MHA/GQA/MQA
    LinearDeltaNet(GatedDeltaNet),  // O(n) linear attention
}

// FFN dispatch
pub enum FfnImpl {
    Gated(GatedFFN),  // SwiGLU, GeGLU, etc.
    MoE(MoEFFN),      // Mixture of Experts
}

// Per-layer cache dispatch
pub enum LayerCache {
    Kv(KvCache),
    Shared { source_layer: usize },
    Recurrent(RecurrentState),
}
```
