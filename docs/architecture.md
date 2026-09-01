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

### 5. The advanced harness is klein's job

gallium's agent surface deliberately stays small: the built-in tools cover a
local REPL (`Read`/`Write`/`Edit`/`Bash`/…), and safety is approval tiers, not
a sandbox. Everything beyond that — sandboxing, web fetch, rich workspace
tooling, editor integration — is the *client's* side of the app-server
protocol: `../klein-cli` brings those capabilities as `dynamicTools`, which run
on the client's machine under the client's policy, while gallium stays the
model-side runtime (inference engines, wire protocols, KV reuse, traces).

The rule of thumb for where a new capability belongs: if it needs the model
(a wire format, a profile, cache behavior, trace fidelity), it goes here; if it
needs the user's environment (the network, a sandbox, an editor), it goes in
the harness on the other side of the socket. This is why `WebFetchTool` was
removed rather than hardened, and why `--working-dir` containment was replaced
by approval tiers instead of a path jail (docs/TODO.md §4 has the retired
findings).

## Crate Responsibilities

### gallium-core

All reusable building blocks. Zero model-specific code. Modules:

| Module | Purpose |
|--------|---------|
| `attention.rs` | Standard attention: MHA, GQA, MQA, with optional sliding window mask, logit softcapping, shared K=V, Q-norm |
| `linear_attn.rs` | Gated DeltaNet: O(n) linear attention with delta update rule and short causal convolution |
| `ffn.rs` | GatedFFN (SwiGLU/GeGLU) with optional clamp, MoEFFN with top-k routing and optional shared expert |
| `norm.rs` | RMSNorm, LayerNorm wrappers |
| `pos_enc.rs` | RoPE with scaling variants: standard, YaRN, Linear, Llama3, NTK; supports partial rotary and per-dim frequency factors. **Note:** GGUF `rope_freqs.weight` stores per-dim DIVISORS (`inv_freq[i] = base / factor[i]`), not `inv_freq` itself — see `docs/models/gemma4.md` Bug 10 |
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
| Qwen 3.6 | `qwen35.rs` | Hybrid DeltaNet (linear) + full attention, MoE with shared experts |
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
| `llm_candle.rs` | Native candle backend; `Arch` detection, model load, protocol dispatch |
| `protocol.rs` | `ModelProtocol` + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol`, `Lfm2Protocol` |
| `memory.rs` | The compaction policy, applied by `runtime::run_turn` |
| `tool.rs` | `Tool`, `ToolDescriptor`, `ToolRegistry`, `ApprovalSink`, and the built-in tools |
| `cancel.rs` | `CancellationToken`, `TurnContext` — stopping a turn that is already running |
| `event.rs` | `AgentEvent` / `AgentObserver` — the progress stream frontends render from |
| `runtime.rs` | `run_turn` — the one turn path: compact → prompt → skill catalog → ReAct → reply. Used by the REPL and every app-server thread |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `skill.rs` / `project.rs` / `github.rs` | SKILL.md loading, AGENTS.md/CLAUDE.md project context, GitHub tools |
| `mcp_client*.rs` / `mcp_server*.rs` | MCP over stdio and streamable HTTP, both directions |
| `appserver/` | JSON-RPC whole-turn backend on stdio |

### One turn path

`runtime::run_turn` (`runtime.rs`) is where a turn happens, for every frontend:
compact if the last turn neared the window → append the prompt → inject the skill
catalog for this turn only → run the ReAct loop → append the reply. The REPL and each
app-server thread both call it.

Each frontend used to assemble that sequence by hand, and they had drifted: the
app-server built its `SkillRegistry` empty and never injected a catalog, so
`lookup_skill` was advertised to the model in every thread and could never find
anything. Cancellation, approval policy, and tracing land here once rather than once
per frontend.

The tool transcript stays in history — the next turn's model can see what it already
read, and compaction is what bounds the cost.

### The event stream

`AgentEvent` (`event.rs`) is what every frontend renders from — the app-server's
`item/*` notifications, the CLI's live tool output, and later a TTS renderer are all
translations of this one stream. `react::run_observed` and `Agent::step_observed` take
an `AgentObserver`; the plain `run` / `step` pass `None`.

Events borrow rather than own: they are emitted on the turn's own thread and every
consumer formats them immediately, so allocating per tool call would be waste.

The variant list is deliberately limited to what the agent emits — `ToolStarted`,
`ToolCompleted`, `Usage`, `TurnCompleted`, `Error`. Token deltas are the notable
absence: no provider streams yet (OpenAI uses the blocking Responses API, and both
local backends return a finished string), so a `MessageDelta` variant would be a
promise nothing keeps.

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
negotiation), `initialized`, `thread/start` (accepts client `dynamicTools` and
`skillPaths`), `turn/start`, `turn/steer`, `turn/interrupt`, `account/read`;
outbound `item/*`, `turn/completed` (whatever the outcome — the turn's `status`
is `completed`, `interrupted` or `failed`), `thread/tokenUsage/updated`, and
`item/fileChange/requestApproval`.

`thread/tokenUsage/updated` carries what each model call cost, the thread's
running total, and the context window they should be read against — enough for a
client to draw a context gauge, which is the one thing the protocol previously
made impossible.

That window is reported only when someone can vouch for it: the user configured
one, or the model's own file says (llama.cpp reads the GGUF's `n_ctx_train`,
candle reads `context_length` / `max_position_embeddings`). Gallium's fallbacks
exist so that *compaction* always has a threshold, and a gauge drawn against one
would present a guess as a measurement — so `modelContextWindow` is null
instead, and a client shows nothing. The same rule applies to the counts: a
provider that reports no usage produces no notification, because `0%` of a real
window is itself a claim.

A turn already in flight can be spoken to twice over. `turn/interrupt` stops it;
`turn/steer` adds user text to it, under the same turn id, and the turn carries
on. Both reach the running loop the same way — a shared cell on the
`TurnContext` that `react.rs` reads at its loop boundaries — so both are prompt
rather than instant: a steer lands after the current generation and the tool
calls it asked for. `expectedTurnId` is a precondition on steering for the same
reason `turnId` is on interrupting: a message meant for a turn that has already
ended must be refused, not delivered to whatever is running now.

Being refused is the point. The steering cell closes when the loop stops reading
it — as one step with the loop's decision to end, so there is no instant in which
a steer is accepted by a turn that will never look again — and `turn/steer`
reports that closure as an error. A client can then re-send into the next turn.
An acknowledged message that silently goes nowhere is the one failure it could
not detect for itself. For the same reason a steered round does not count against
`max_iterations`: a turn must not fail, and roll back an answer it already
produced, because the user spoke during its last iteration.

A thread's skills are the standard locations `skill::load_skills` searches, the
launch config's `agent.skillPaths`, then the client's `skillPaths` from
`thread/start` — increasing precedence, so the client's win. That last list is
what lets a client whose skills live in its own repo (`skills/`, say) have them
at all; a path that loads nothing is logged as a warning, since the response is
codex's `ThreadStartResponse` and has no field for a count.

This is the same wire protocol codex's app-server presents. It is *not* the
agentclientprotocol.com standard (`session/new` / `session/prompt`), which was
considered and declined (issue #15). `../klein-cli`'s `pkg/agentserver` is the
client for it.

Because stdout carries the protocol stream in this mode, logging is redirected to
stderr in `main.rs`.

Providers are built once and shared by every thread on the same model, keyed on the
local model path when there is one. One process serves many threads — klein starts a
thread per session — and a local provider owns multi-GB weights, so a provider per
thread would duplicate them. llama.cpp's backend is process-global besides: it
refuses to initialize twice, so a second local provider could not be built at all.

Each thread keeps its own history and compacts it between turns, using the shared
policy in `memory.rs` (`compaction_target` / `compact_messages`): once the previous
turn's prompt reaches 90% of `contextWindow`, the oldest history is dropped until it
is back under 50% of it. Dropping happens a whole exchange at a time — a user message
together with the assistant replies, tool calls, and tool results that answered it —
so the retained history always resumes at a user turn. That avoids both a `tool`
message whose call is gone (which providers reject) and an assistant reply whose
question is gone (dead context, and a hazard for GGUF chat templates that expect a
user-first history). When a backend reports no token usage — the native candle one
never does — the trigger falls back to gallium's own estimate of the history.

### Protocol Adapters

`ModelProtocol` is the adapter layer between the agent's generic `ChatMessage` history and each model's raw prompt/response format. **It applies to the native candle backend only** — the llama.cpp backend renders the chat template embedded in the GGUF instead. Each implementation handles two responsibilities:

1. **`format_prompt`** — renders a `Vec<ChatMessage>` into the model-specific token string
2. **`parse_response`** — extracts the user-facing reply from raw decoded output

| Protocol | Model | Format | Parse |
|---|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + `Valid channels` instructions; `<\|start\|>role<\|channel\|>ch<\|message\|>content<\|end\|>` | Extracts `final` channel, discards `analysis`/`commentary` |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template | Passthrough trim |
| `QwenProtocol` | Qwen 3.6 | ChatML `<\|im_start\|>role` template | Passthrough trim |
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
