# rs-gallium

A simple, paper-friendly LLM inference framework in Rust, with an interactive ReAct agent.

rs-gallium provides composable building blocks that map directly to how research papers describe transformer architectures. When a new paper proposes a novel attention mechanism, FFN variant, or position encoding, you can implement and test it with minimal boilerplate.

## Target Models

- **GPT-OSS** (OpenAI) — alternating full/sliding-window attention + MoE with SwiGLU
- **Qwen 3.5** (Alibaba) — hybrid Gated DeltaNet (linear attention) + full attention
- **Gemma 4** (Google) — dual RoPE, shared K=V, per-layer embeddings, logit softcapping

## Structure

```
crates/
  gallium-core/     # Composable building blocks + generation
  gallium-models/   # Model implementations (GPT-OSS, Qwen 3.5, Gemma 4)
  gallium-cli/      # One-shot inference CLI
  gallium-agent/    # Interactive ReAct agent (local model or OpenAI)
docs/               # Documentation
testsuite/          # Agent integration tests (runner + testcases)
```

## Design

- **Simple**: ~1400 lines of core framework code. Each model definition is ~150-200 lines.
- **Composable**: mix and match attention (MHA/GQA/MQA/DeltaNet), FFN (SwiGLU/GeGLU/MoE), position encoding (RoPE with various scalings), and normalization (RMSNorm/LayerNorm).
- **Per-layer heterogeneous**: first-class support for architectures where different layers use different attention types, RoPE configs, or FFN types.
- **Candle backend**: uses [candle](https://github.com/huggingface/candle) for tensor operations, giving CPU/CUDA/Metal support.

## Quick Start — One-shot Inference

```bash
# Build
cargo build --release

# Run GPT-OSS 20B (downloads from HuggingFace automatically, cached for subsequent runs)
make run-gpt-oss PROMPT="What is the capital of France?"

# Run Qwen 3.5 9B (GGUF, quantized)
make run-qwen35-gguf PROMPT="The capital of Japan is Tokyo. The capital of France is"

# Run Gemma 4 (GGUF, quantized)
make run-gemma4-gguf PROMPT="The capital of France is"
```

The `--hf-repo` flag downloads model files into `~/.cache/huggingface/hub/` and reuses them on subsequent runs. The `--chat` flag applies the GPT-OSS chat template.

### Running with a Local Model

```bash
# Safetensors (full precision)
cargo run --release -p gallium-cli -- \
  --arch gpt-oss --format safetensors \
  --model /path/to/model-dir/ --dtype f16 \
  --chat --prompt "Hello!"

# GGUF (quantized)
cargo run --release -p gallium-cli -- \
  --arch gpt-oss --format gguf \
  --model /path/to/model.gguf \
  --prompt "Hello!"
```

## Interactive Agent

`gallium-agent` is a multi-turn ReAct agent with tool calling. It supports two backends:

- **Gallium** — runs a local gallium model via a per-architecture protocol adapter:
  - **GPT-OSS** — full ReAct loop with tools (read, glob, write, edit, tasks) via the [Harmony protocol](https://github.com/openai/harmony)
  - **Gemma 4** — plain chat (Gemini `<start_of_turn>` template)
  - **Qwen 3.5** — plain chat (ChatML `<|im_start|>` template)
- **OpenAI** — calls the OpenAI Responses API with full ReAct loop + tools (read, glob, write, edit, tasks)

```bash
# Local GPT-OSS agent with tool calling (downloads on first run)
cargo run --release -p gallium-agent -- \
  --arch gpt-oss --hf-repo openai/gpt-oss-20b --dtype f16

# Local GPT-OSS GGUF (quantized, faster)
cargo run --release -p gallium-agent -- \
  --arch gpt-oss --format gguf \
  --hf-repo unsloth/gpt-oss-20b-GGUF \
  --hf-file gpt-oss-20b-Q4_K_M.gguf \
  --hf-tokenizer-repo openai/gpt-oss-20b

# OpenAI agent with ReAct + tools
cargo run --release -p gallium-agent -- \
  --provider openai --openai-model gpt-4o-mini

# With a system prompt
cargo run --release -p gallium-agent -- \
  --provider openai \
  --system-prompt "You are a helpful coding assistant."

# Batch mode: process a multi-turn prompt file (turns split by ----)
cargo run --release -p gallium-agent -- \
  --provider openai --file prompts.txt
```

Session commands: `/reset` (clear history), `/help`, `/quit`.

### Available Tools

| Tool | Description |
|------|-------------|
| `read` | Read a file from the working directory |
| `glob` | List files matching a pattern |
| `write` | Create or overwrite a file |
| `edit` | Replace an exact string in a file |
| `tasks` | Create and track tasks |

## Running with Docker

Build the image once, then run on any machine by mounting your local HuggingFace cache:

```bash
make docker-build

# Run with HuggingFace download
make docker-run \
  ARCH=gemma4 FORMAT=gguf \
  HF_REPO=unsloth/gemma-4-E4B-it-GGUF \
  HF_FILE=gemma-4-E4B-it-Q4_K_M.gguf \
  HF_TOKENIZER_REPO=google/gemma-4-E4B \
  PROMPT="The capital of France is"
```

The host's `~/.cache/huggingface` is mounted into the container, so downloads are shared.

## Integration Tests

### Model Inference Tests

End-to-end tests load real model weights and verify that inference produces correct output. Tests skip automatically when model files are not present.

```bash
cargo test -p gallium-models --test integration -- --nocapture
```

### Agent Integration Tests

The `testsuite/` directory contains end-to-end tests that run `gallium-agent` in an isolated working directory and verify the agent's output against expected behavior.

```bash
# Run a single testcase against a specific model config
AGENT=./target/release/gallium-agent \
  ./testsuite/runner.sh coding gpt-oss-gguf

# Run all testcases × all active models (prints a pass/fail table)
AGENT=./target/release/gallium-agent ./testsuite/matrix_runner.sh
```

**Testcase layout**: each testcase directory contains:
- `prompt.txt` — agent prompt (turns separated by `----` lines)
- `check.sh` — receives the agent output file; exits 0 for PASS, 1 for FAIL

**Model configs**: `testsuite/models/*.sh` — each file exports `AGENT_FLAGS` for one model variant (e.g. `gpt-oss-gguf.sh`, `openai.sh`).

Built-in testcases:

| Testcase | What it checks |
|----------|---------------|
| `coding` | Agent creates a working Go "Hello, World!" program |
| `refactoring` | Agent refactors Go code from global variable to a struct |
| `memory_state` | Agent correctly tracks context across two conversation turns |

## Building Blocks

| Module | What it does |
|--------|-------------|
| `Attention` | MHA/GQA/MQA with optional sliding window, logit softcapping, shared K=V, Q-norm |
| `GatedDeltaNet` | O(n) linear attention with delta update rule (Qwen 3.5) |
| `GatedFFN` | SwiGLU/GeGLU with optional clamp |
| `MoEFFN` | Mixture of Experts with top-k routing and optional shared expert |
| `RoPE` | Rotary embeddings with YaRN/Linear/Llama3/NTK scaling, partial rotary, freq factors |
| `TransformerBlock` | Pre-norm → attn → residual → post-norm → ffn → residual |
| `ModelCache` | Per-layer KV cache, recurrent state, or cross-layer sharing |

## Adding a New Model

See [docs/adding-models.md](docs/adding-models.md). The short version:

1. Define a config struct (deserializes from HuggingFace `config.json`)
2. Wire together gallium-core blocks in a `load()` function
3. Implement `CausalLM` (forward + reset)
4. Add to `gallium-models/src/lib.rs` and the CLI

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Adding Models Guide](docs/adding-models.md)
- [Building Blocks Reference](docs/building-blocks.md)
- [Target Model Notes](docs/target-models.md)
- [GPT-OSS Notes](docs/gpt-oss.md)
- [Qwen 3.5 Notes](docs/qwen35.md)
- [Gemma 4 Notes](docs/gemma4.md)

## License

MIT
