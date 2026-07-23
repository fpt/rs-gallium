# rs-gallium Architecture

## Overview

rs-gallium is a Rust LLM inference framework designed for **simplicity** and **rapid implementation of novel architectures from research papers**.

```
rs-gallium/
├── crates/
│   ├── gallium-core/       # Composable building blocks + generation
│   ├── gallium-models/     # Concrete model implementations
│   └── gallium-agent/      # The `gallium` binary: ReAct agent + app-server
├── configs/             # TOML configs for the agent (--config)
├── testsuite/           # Agent capability tests
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
| LFM2.5 | `lfm2moe_q.rs` | Hybrid short-conv + GQA MoE (GGUF only) |

Each has a `*_q.rs` GGUF counterpart. `gemma4_vision.rs` exists and compiles but has no caller.

### gallium-agent

The `gallium` binary: a multi-turn ReAct agent, usable as a REPL or as a JSON-RPC
whole-turn backend. Modules:

| Module | Purpose |
|--------|---------|
| `main.rs` | Mode selection (REPL vs `app-server`), env/config resolution, REPL loop (`/reset`, `/quit`) |
| `config.rs` | TOML `--config` schema (`[llm]`, `[agent]`, `[[mcpServers]]`) |
| `llm.rs` | `LlmProvider` trait, `OpenAiProvider` (Responses API), `InferenceEngine` selection |
| `llm_local.rs` | In-process llama.cpp backend; renders the GGUF's embedded jinja chat template |
| `llm_gallium.rs` | Native candle backend; `Arch` detection, model load, protocol dispatch |
| `protocol.rs` | `ModelProtocol` + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol`, `Lfm2Protocol` |
| `memory.rs` | `ConversationMemory`: multi-turn history with compaction |
| `tool.rs` | `ToolHandler`, `ToolRegistry`, `ApprovalSink`, and the built-in tools |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `skill.rs` / `situation.rs` / `github.rs` | SKILL.md loading, situation messages, GitHub tools |
| `mcp_client*.rs` / `mcp_server*.rs` | MCP over stdio and streamable HTTP, both directions |
| `appserver/` | JSON-RPC whole-turn backend on stdio |

One execution path, whatever the provider — OpenAI, llama.cpp, and native candle all
run the same ReAct loop:

```
user input → history → ReAct loop:
    ├── provider.chat_with_tools(messages, tool_defs)
    ├── if ToolCalls: approve if mutating → execute each → append results → loop
    └── if Text: return response
```

The local backends re-prefill the full conversation history on every turn (no
incremental KV cache across turns); `generate()` calls `model.reset()` internally.

### app-server mode

`gallium app-server` serves the agent over line-delimited JSON-RPC on stdio: the
client hands over a whole turn, and gallium runs its own ReAct loop, tools, and MCP
connections inside it. Inbound `initialize` (with `experimentalApi` capability
negotiation), `initialized`, `thread/start` (accepts client `dynamicTools`),
`turn/start`, `account/read`; outbound `item/*`, `turn/completed`, `turn/failed`,
and `item/fileChange/requestApproval`.

This is the same wire protocol codex's app-server presents — what `../rs-kessel` and
`../klein-cli` call "ACP". It is *not* the agentclientprotocol.com standard
(`session/new` / `session/prompt`), which was considered and declined (issue #15).

Because stdout carries the protocol stream in this mode, logging is redirected to
stderr in `main.rs`.

### Protocol Adapters

`ModelProtocol` is the adapter layer between the agent's generic `ChatMessage` history and each model's raw prompt/response format. **It applies to the native candle backend only** — the llama.cpp backend renders the chat template embedded in the GGUF instead. Each implementation handles two responsibilities:

1. **`format_prompt`** — renders a `Vec<ChatMessage>` into the model-specific token string
2. **`parse_response`** — extracts the user-facing reply from raw decoded output

| Protocol | Model | Format | Parse |
|---|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + `Valid channels` instructions; `<\|start\|>role<\|channel\|>ch<\|message\|>content<\|end\|>` | Extracts `final` channel, discards `analysis`/`commentary` |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template | Passthrough trim |
| `QwenProtocol` | Qwen 3.5 | ChatML `<\|im_start\|>role` template | Passthrough trim |
| `Lfm2Protocol` | LFM2.5 | ChatML-style template | Strips the leading `<think>` block |

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
