# CLAUDE.md

## Project Overview

rs-gallium is a simple, paper-friendly LLM inference framework in Rust. It provides composable building blocks (attention, FFN, RoPE, normalization) that researchers can wire together to implement new model architectures quickly.

Target models: GPT-OSS, Qwen 3.5, Gemma 4. Also includes `gallium-agent`, an interactive ReAct agent backed by local gallium models or OpenAI.

## Essential Commands

```bash
# Build everything (Rust + UniFFI gen + Swift)
make build

# Build Rust only
make build-rust

# Regenerate Swift bindings from UDL (after Rust lib is built)
make gen-uniffi

# Build Swift frontend only (after gen-uniffi)
make build-swift

# Check (fast compile check, Rust only)
cargo check --workspace

# Run tests
cargo test --workspace

# Format
cargo fmt --all

# Clippy
cargo clippy --workspace

# Run Swift CLI (text mode, reads configs/default.yaml)
make run-text

# Available configs (all in configs/):
#   default.yaml              OpenAI gpt-5.4-mini  (Swift CLI only)

# Run gallium-agent with a local model (canned shortcuts)
make run-gpt-oss              # GPT-OSS 20B safetensors
make run-gpt-oss-gguf         # GPT-OSS 20B Q4_K_M GGUF
make run-gemma4-e2b-gguf      # Gemma 4 E2B Q4_K_M GGUF
make run-gemma4-gguf          # Gemma 4 E4B Q4_K_M GGUF
make run-qwen35-gguf          # Qwen 3.5 9B Q4_K_M GGUF

# Override dtype / max-tokens / temperature on any canned target
make run-gpt-oss DTYPE=bf16 MAX_TOKENS=1024 TEMPERATURE=0.5

# Generic targets for any repo/file
make run-agent-gguf ARCH=gemma4 HF_REPO=unsloth/gemma-4-E4B-it-GGUF \
     HF_FILE=gemma-4-E4B-it-Q4_K_M.gguf HF_TOKENIZER_REPO=google/gemma-4-E4B
make run-agent-local ARCH=gemma4 HF_REPO=google/gemma-4-E4B DTYPE=bf16

# Run gallium-agent with OpenAI (full ReAct with tools)
cargo run -p gallium-agent -- --provider openai --openai-model gpt-5.4-mini
```

## Architecture

### Workspace Layout

- `crates/gallium-core/` — All reusable building blocks. Zero model-specific code.
- `crates/gallium-models/` — Concrete model implementations using gallium-core blocks.
- `crates/gallium-agent/` — Interactive ReAct agent (multi-turn, tool calling). Also compiled as a static lib for Swift via UniFFI.
- `swift/` — Swift frontend (macOS 26+): text REPL + voice mode. Built with `swift build`.
- `docs/` — Documentation.
- `references/` — Reference implementations (transformers, llama.cpp, vllm, mistral.rs). Cloned via `bash references/setup.sh`. Gitignored, not built by cargo.

### Key Design Decisions

- **Concrete structs + enum dispatch** over traits. Only one trait: `CausalLM`.
- **Per-layer heterogeneous config**: layers can have different attention types, RoPE, FFN.
- **candle-core/candle-nn** as tensor backend (git dependency pinned to rev 097655a2).

### Core Modules (gallium-core)

| File | Responsibility |
|------|---------------|
| `attention.rs` | Standard attention (MHA/GQA/MQA), sliding window via mask, logit softcapping, shared K=V, Q-norm |
| `linear_attn.rs` | Gated DeltaNet linear attention with recurrent state |
| `ffn.rs` | GatedFFN (SwiGLU/GeGLU + clamp), MoEFFN (top-k routing + shared expert) |
| `quantized.rs` | GGUF loading: `QVarBuilder`, `QLinear`, `QNorm`, `GgufMetadata` |
| `turbo_quant.rs` | TurboQuant: near-optimal vector quantization (MSE + InnerProduct modes) |
| `turbo_kv_cache.rs` | TurboKvCache: KV cache with TurboQuant compression (5-8x memory reduction) |
| `block.rs` | TransformerBlock combinator |
| `pos_enc.rs` | RoPE with scaling variants (YaRN, Linear, Llama3, NTK), partial rotary, freq factors |
| `norm.rs` | RMSNorm, LayerNorm wrappers around candle-nn |
| `kv_cache.rs` | KV cache, RecurrentState, cross-layer sharing |
| `mask.rs` | Causal and sliding-window mask builders |
| `sampling.rs` | Greedy, top-k, top-p, temperature sampling |
| `model.rs` | `CausalLM` trait, `generate()` with streaming callback |

### Model Files (gallium-models)

| File | Model |
|------|-------|
| `gpt_oss.rs` | GPT-OSS (safetensors): alternating full/SW attn, MoE, YaRN RoPE |
| `gpt_oss_q.rs` | GPT-OSS (GGUF): quantized variant using QLinear |
| `qwen35.rs` | Qwen 3.5 (safetensors): hybrid DeltaNet + full attn |
| `qwen35_q.rs` | Qwen 3.5 (GGUF): quantized variant |
| `gemma4.rs` | Gemma 4 (safetensors): dual RoPE, shared K=V, PLE, softcapping |
| `gemma4_q.rs` | Gemma 4 (GGUF): quantized variant |
| `loader.rs` | safetensors loading via VarBuilder |

### Adding a New Model

1. Add `your_model.rs` in `crates/gallium-models/src/`
2. Define config struct (serde deserialize from HuggingFace `config.json`)
3. Wire gallium-core blocks in `load()`, implement `CausalLM`
4. Add `pub mod your_model;` to `lib.rs` and an arch variant in `gallium-agent/src/main.rs`
5. Verify `vb.pp()` paths match safetensors weight names

### Adding a Novel Component

1. Add `your_component.rs` in `crates/gallium-core/src/`
2. Follow the same forward signature pattern
3. Add variant to `AttnImpl` or `FfnImpl` enum if needed
4. Export from `lib.rs`

### Weight Loading

Uses candle-nn `VarBuilder::from_mmaped_safetensors`. The `vb.pp("prefix")` calls must match PyTorch `state_dict` key hierarchy. Check the model's `model.safetensors.index.json` on HuggingFace.

### gallium-agent Modules

| File | Responsibility |
|------|---------------|
| `llm.rs` | `LlmProvider` trait + `OpenAiProvider` (Responses API with tool calling) |
| `memory.rs` | `ConversationMemory`: multi-turn history with token-based compaction |
| `tool.rs` | `ToolHandler` trait, `ToolRegistry`, built-in tools: `read`, `glob`, `tasks` |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `protocol.rs` | `ModelProtocol` trait + `HarmonyProtocol` (GPT-OSS), `GemmaProtocol` (Gemma 4), `QwenProtocol` (Qwen 3.5) |
| `provider.rs` | `GalliumProvider`: wraps a local `CausalLM`, delegates prompt format/parse to `ModelProtocol` |
| `agent.rs` | `Agent`: routes to ReAct (OpenAI) or plain chat (Gallium), manages memory |
| `lib.rs` | Library root: UniFFI scaffolding, `Agent`/`CloudAgentConfig`/`AgentResponse` public types |
| `main.rs` | Rust REPL CLI with `/reset`, `/help`, `/quit` commands |
| `skill.rs` | `SkillRegistry`: loads SKILL.md files from `~/.config/gallium/skills/` and `.gallium/skills/` |
| `session.rs` | JSONL session persistence in `.gallium/sessions/<id>.jsonl` |
| `mcp_client.rs` | JSON-RPC 2.0 MCP client: spawns server subprocess, discovers tools, wraps as `ToolHandler` |

**Built-in tools** (registered in `tool.rs`): `read`, `glob`, `write`, `edit`, `tasks`, `bash`, `web_fetch`, `lookup_skill`

### Swift Frontend (`swift/`)

| Target | Responsibility |
|--------|---------------|
| `GalliumCLI` | Main executable: text REPL (libedit) + voice REPL |
| `AgentBridge` | UniFFI-generated Swift bindings + generated `gallium_agent.swift` (symlinked from `vendor/`) |
| `AgentBridgeFFI` | System library: `module.modulemap` + `gallium_agentFFI.h` bridging to Rust static lib |
| `Util` | `Logger`, `Config` (YAML via Yams) |
| `TTS` | `TextToSpeech`: `AVSpeechSynthesizer` wrapper with `speakAsync()` |
| `Audio` | `AudioCapture`: on-device STT via `SpeechTranscriber` (macOS 26+) |
| `CEditline` | System library wrapper for libedit |

**UniFFI build flow:**
1. `cargo build --release` produces `target/release/libgallium_agent.a`
2. `make gen-uniffi` runs `uniffi-bindgen generate --language swift` → `swift/vendor/uniffi-swift/{gallium_agent.swift,gallium_agentFFI.h}`
3. `swift build` links `libgallium_agent.a` via `AgentBridge`'s linker flags

**Provider routing:**
- Gallium provider → `supports_tools() = false` → plain `chat()` (full history re-prefilled each turn)
- OpenAI provider → `supports_tools() = true` + tools registered → ReAct loop with `read`/`glob`/`tasks`

**Protocol adapters** — `ModelProtocol` has two methods:
- `format_prompt(&[ChatMessage]) -> String` — renders history to model-specific token string
- `parse_response(&str) -> String` — extracts user-facing reply from raw decoded output

| Protocol | Model | Notes |
|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + channel instructions; extracts `final` channel from output |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template |
| `QwenProtocol` | Qwen 3.5 | ChatML `<\|im_start\|>role` template |

**Harmony system prompt** (injected by `HarmonyProtocol::format_prompt`):
```
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: YYYY-MM-DD

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
```

## gallium-agent Flags

| Flag | Description |
|------|-------------|
| `--provider` | `gallium` (default) or `openai` |
| `--arch` | Model architecture (required for gallium): `gpt-oss`, `qwen35`, `gemma4` |
| `--format` | `safetensors` (default) or `gguf` |
| `--model` | Local path to model dir or GGUF file |
| `--hf-repo` / `--hf-file` / `--hf-tokenizer-repo` | HuggingFace download |
| `--dtype` | Weight dtype for safetensors (default: `f16`) |
| `--openai-model` | OpenAI model name (default: `gpt-5.4-mini`) |
| `--openai-api-key` | OpenAI API key (or `OPENAI_API_KEY` env var) |
| `--reasoning-effort` | For OpenAI reasoning models: `low`, `medium`, `high` |
| `--system-prompt` | System prompt injected before every turn |
| `--working-dir` | Root directory for `read`/`glob` tools (default: cwd) |
| `--max-tokens` | Max new tokens per turn (default: 512) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--context-window` | Context window size for memory compaction (default: 32000) |

## Common Pitfalls

- Tensor arithmetic (`+`, `*`, `-`) returns `Result<Tensor>`, not `Tensor`. Always `?` the result.
- `Linear::forward()` requires `use candle_core::Module;` in scope.
- Candle's `rope()` expects input shape `(batch, n_heads, seq_len, head_dim)` and cos/sin shape `(1, seq_len, head_dim/2)`.
- After `transpose()`, tensors are non-contiguous. Call `.contiguous()?` before passing to `rope.apply()`.
- After `expand()`, tensors are non-contiguous. Call `.contiguous()?` before `cat()` or any op that requires contiguous input.
- GPT-OSS uses MXFP4 E2M1 quantization for MoE expert weights in safetensors. Loading requires a separate `DType::U8` VarBuilder opened over the same files.
- `.i()` tensor indexing requires `use candle_core::IndexOp;` in scope.
- Always use `uv run python` instead of `python3`
- **Before implementing a new model or FFN variant, read the reference `modeling_*.py` in `references/transformers/`.** Activation functions, gate/up split ordering, and normalization order often differ from what the paper or config.json implies. GPT-OSS's FFN activation (`gate * sigmoid(gate * 1.702)`, interleaved split, `(up+1)` shift) looked like SwiGLU but was entirely different.
