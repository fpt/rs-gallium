# CLAUDE.md

## Project Overview

rs-gallium is a simple, paper-friendly LLM inference framework in Rust. It provides composable building blocks (attention, FFN, RoPE, normalization) that researchers can wire together to implement new model architectures quickly.

Target models: GPT-OSS, Qwen 3.5, Gemma 4. Also includes `gallium-agent`, an interactive ReAct agent backed by local gallium models or OpenAI.

## Essential Commands

```bash
# Build
cargo build --release

# Check (fast compile check)
cargo check --workspace

# Run tests
cargo test --workspace

# Format
cargo fmt --all

# Clippy
cargo clippy --workspace

# Run GPT-OSS 20B (downloads from HF, cached)
make run-gpt-oss PROMPT="Hello"

# Run CLI (GGUF, local file)
cargo run -p gallium-cli -- --arch gpt-oss --format gguf --model /path/to/model.gguf --prompt "Hello"

# Run CLI (safetensors, local dir)
cargo run -p gallium-cli -- --arch gpt-oss --format safetensors --model /path/to/model-dir/ --prompt "Hello"

# Run CLI (download from HuggingFace)
cargo run -p gallium-cli -- --arch gpt-oss --format safetensors --hf-repo openai/gpt-oss-20b --dtype f16 --chat --prompt "Hello"

# Run interactive agent (local GPT-OSS, plain chat)
cargo run -p gallium-agent -- --arch gpt-oss --hf-repo openai/gpt-oss-20b --dtype f16

# Run interactive agent (OpenAI, full ReAct with tools)
cargo run -p gallium-agent -- --provider openai --openai-model gpt-4o-mini
```

## Architecture

### Workspace Layout

- `crates/gallium-core/` — All reusable building blocks. Zero model-specific code.
- `crates/gallium-models/` — Concrete model implementations using gallium-core blocks.
- `crates/gallium-cli/` — Thin CLI binary for one-shot inference.
- `crates/gallium-agent/` — Interactive ReAct agent (multi-turn, tool calling).
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
4. Add `pub mod your_model;` to `lib.rs` and a CLI variant
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
| `main.rs` | REPL CLI with `/reset`, `/help`, `/quit` commands |

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

## CLI Flags

| Flag | Description |
|------|-------------|
| `--arch` | Model architecture: `gpt-oss`, `qwen35`, `gemma4` |
| `--format` | `safetensors` (default) or `gguf` |
| `--model` | Local path to model dir (safetensors) or `.gguf` file |
| `--hf-repo` | HuggingFace repo ID to download from (e.g. `openai/gpt-oss-20b`). Cached in `~/.cache/huggingface/hub/` |
| `--hf-file` | Filename within `--hf-repo` (required for GGUF, e.g. `model-q4_k_m.gguf`) |
| `--hf-tokenizer-repo` | Separate repo for `tokenizer.json` (for GGUF repos that omit it) |
| `--dtype` | `f32`, `f16`, `bf16` (safetensors only; use `f16` on Apple Silicon — BF16 matmul not supported) |
| `--chat` | Apply the GPT-OSS chat template to the prompt before tokenization |
| `--prompt` | Input text |
| `--max-tokens` | Max new tokens (default: 256) |
| `--temperature` | Sampling temperature (default: 0.7; 0.0 = greedy) |
| `--top-k` / `--top-p` | Sampling parameters |

## gallium-agent Flags

| Flag | Description |
|------|-------------|
| `--provider` | `gallium` (default) or `openai` |
| `--arch` | Model architecture (required for gallium): `gpt-oss`, `qwen35`, `gemma4` |
| `--format` | `safetensors` (default) or `gguf` |
| `--model` | Local path to model dir or GGUF file |
| `--hf-repo` / `--hf-file` / `--hf-tokenizer-repo` | HuggingFace download (same as gallium-cli) |
| `--dtype` | Weight dtype for safetensors (default: `f16`) |
| `--openai-model` | OpenAI model name (default: `gpt-4o-mini`) |
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
