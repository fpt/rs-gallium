# GaLLiuM inference framework in Rust

A simple, paper-friendly LLM inference framework in Rust, with an agent (`gallium`) built on top of it.

rs-gallium provides composable building blocks that map directly to how research papers describe transformer architectures. When a new paper proposes a novel attention mechanism, FFN variant, or position encoding, you can implement and test it with minimal boilerplate.

## Target Models

- **GPT-OSS** (OpenAI) — alternating full/sliding-window attention + MoE
- **Qwen 3.5** (Alibaba) — hybrid Gated DeltaNet (linear attention) + full attention
- **Gemma 4** (Google) — dual RoPE, shared K=V, per-layer embeddings, logit softcapping
- **LFM2.5** (LiquidAI) — hybrid short-conv + GQA MoE (GGUF only)

## Structure

```
crates/
  gallium-core/     # Composable building blocks + generation
  gallium-models/   # Model implementations (GPT-OSS, Qwen 3.5, Gemma 4, LFM2.5)
  gallium-agent/    # The `gallium` binary: ReAct agent REPL + app-server
configs/            # TOML configs for the agent (--config)
docs/               # Documentation
testsuite/          # Agent capability tests (runner + backends + testcases)
```

## The `gallium` binary

One binary, two modes. It takes **no model flags** — settings come from environment
variables layered over an optional TOML `--config` file (env > config > default),
and prompts arrive on **stdin**, one line per turn.

```bash
make build          # cargo build --release
make install        # copy target/release/gallium to ~/bin (override with PREFIX=)
```

### REPL mode (default)

```bash
# Cloud (OpenAI Responses API)
OPENAI_API_KEY=sk-... gallium --config configs/openai.toml

# Local GGUF via the in-process llama.cpp backend (the default engine)
gallium --config configs/qwen3.6.toml

# Local model straight from the environment, no config file
MODEL_PATH=/path/to/model.gguf gallium

# `hf:ORG/REPO[@REV]/file.gguf` downloads into ~/.cache/huggingface on first use
MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf gallium

# One-shot: pipe a prompt instead of typing it
echo "Read Cargo.toml and summarize it" | MODEL_PATH=... gallium
```

Replies are printed to stdout prefixed `Assistant: `; diagnostics go to stderr.
REPL commands: `/reset` (clear history, keep the system prompt), `/quit` / `/exit`.

### app-server mode

```bash
gallium app-server --config configs/openai.toml
```

Serves the agent as a **whole-turn backend** over line-delimited JSON-RPC on stdio:
the client hands over an entire conversation turn and gets back the final text,
while gallium runs its own ReAct loop, tools, and MCP connections inside that turn.
Method set: `initialize` (capability negotiation), `initialized`, `thread/start`,
`turn/start`, `account/read`, with `item/*` / `turn/completed` / `turn/failed`
updates and `item/fileChange/requestApproval` approval round-trips flowing back out.
Clients may inject their own tools via `dynamicTools` on `thread/start`.

This is deliberately the same wire protocol codex's app-server presents — the
subset that `../rs-kessel` and `../klein-cli` refer to as "ACP". It is **not** the
agentclientprotocol.com standard (`session/new` / `session/prompt`); adopting that
was considered and declined (issue #15), so the surface here stays small.

In this mode stdout carries the JSON-RPC stream, so all logging is redirected to
stderr. Anything else writing to stdout will corrupt the protocol.

## Inference engines

`inferenceEngine` (or `INFERENCE_ENGINE`) selects the local backend:

| Engine | Value | Notes |
|---|---|---|
| llama.cpp, in-process | `llamacpp` *(default)* | GGUF only; renders the GGUF's embedded jinja chat template |
| native candle | `gallium` | GGUF + safetensors; arch auto-detected; needs a `tokenizer.json` |

Both are on by default as cargo features (`local`, `gallium`). macOS builds enable
Metal automatically; CUDA and Vulkan are opt-in (`--features cuda` / `vulkan`)
because they depend on host toolkits that `cfg()` cannot detect.

## Configuration

```toml
[llm]
baseURL = "https://api.openai.com/v1"  # note the uppercase URL
model = "gpt-5.6-luna"
apiKey = ""                            # empty → read OPENAI_API_KEY
modelPath = "hf:ORG/REPO/file.gguf"    # local model; presence selects local over cloud
inferenceEngine = "llamacpp"           # or "gallium"
temperature = 0.7
maxTokens = 4096
reasoningEffort = "medium"             # low | medium | high

[agent]
systemPromptPath = "system-prompt.md"  # relative to the config file's dir
maxTurns = 50                          # max ReAct iterations per turn
skillPaths = ["../skills"]             # SKILL.md dirs

[[mcpServers]]
command = "godevmcp"                   # stdio transport
args = ["serve"]

[[mcpServers]]
url = "http://127.0.0.1:27182/mcp"     # streamable HTTP transport
```

Ready-made configs live in `configs/`. Environment overrides:

| Variable | Overrides |
|---|---|
| `MODEL_PATH` | `llm.modelPath` |
| `LLM_BASE_URL` / `LLM_MODEL` / `OPENAI_API_KEY` | the `[llm]` cloud fields |
| `LLM_TEMPERATURE` / `MAX_TOKENS` / `REASONING_EFFORT` | sampling + budget |
| `INFERENCE_ENGINE` | `llm.inferenceEngine` |
| `MAX_REACT_ITERATIONS` | `agent.maxTurns` |
| `WORKING_DIR` | tool root (default: cwd) |
| `MCP_SERVERS` | extra stdio servers, `"cmd arg1,cmd2 arg1"` |
| `KESSEL_GALLIUM_DTYPE` / `KESSEL_GALLIUM_TOKENIZER_REPO` | native candle backend dtype / tokenizer source |
| `KESSEL_GPU_LAYERS` | llama.cpp GPU offload (`0` = CPU) |
| `KESSEL_AUTO_APPROVE=1` | approve mutating tools non-interactively (CI/tests) |
| `KESSEL_BASH_ALLOW` | extra allowed `bash` commands |

## Tools

Registered by default for every provider:

| Tool | Description |
|------|-------------|
| `read` | Read a file |
| `write` | Create or overwrite a file *(requires approval)* |
| `edit` | Replace an exact string in a file *(requires approval)* |
| `multi_edit` | Apply several edits to one file *(requires approval)* |
| `glob` | List files matching a pattern |
| `ls` | List a directory |
| `grep` | Search file contents |
| `bash` | Run a shell command *(requires approval)* |
| `tasks` | Create and track tasks |
| `lookup_skill` | Load a SKILL.md by name |
| `read_situation_messages` | Read pending situation messages |

MCP servers from the config (or `MCP_SERVERS`) register their tools alongside these.
Mutating tools prompt for approval on a TTY; in app-server mode the request is
routed to the client, honoring its `approvalPolicy`.

## Design

- **Simple**: each model definition is ~150-200 lines on top of the core blocks.
- **Composable**: mix and match attention (MHA/GQA/MQA/DeltaNet), FFN (SwiGLU/GeGLU/MoE), position encoding (RoPE with various scalings), and normalization (RMSNorm/LayerNorm).
- **Per-layer heterogeneous**: first-class support for architectures where different layers use different attention types, RoPE configs, or FFN types.
- **Candle backend**: the native engine uses [candle](https://github.com/huggingface/candle) for tensor operations, giving CPU/CUDA/Metal support.

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

## Tests

```bash
cargo test --workspace
```

**Model inference tests** load real weights and check that generation is correct.
They skip automatically when the model files are not cached:

```bash
cargo test -p gallium-models --test integration -- --nocapture
```

**Agent capability tests** run the `gallium` binary in an isolated temp dir against
one TOML backend config per model, and check the assistant's replies (plus any
files it wrote):

```bash
make testsuite                       # full matrix, all available backends
make testsuite-local                 # local backends only (no API key needed)

bash testsuite/runner.sh capital gemma4          # one testcase × one backend
BACKENDS="gemma4,gpt-oss" bash testsuite/matrix_runner.sh
TESTS="coding,refactoring"  bash testsuite/matrix_runner.sh
```

Backends are `testsuite/backends/*.toml`; testcases are `testsuite/testcases/*/`
with a `prompt.txt` and a `check.sh`. See [testsuite/README.md](testsuite/README.md).

## Docker

`Dockerfile.integration` builds a Linux image that runs the agent testsuite with
the host's HuggingFace cache mounted:

```bash
make docker-build-integration
make docker-run-integration ARGS="capital gemma4"
```

The top-level `Dockerfile` still builds the removed `gallium-cli` crate and does
not work — see issue #3.

## Adding a New Model

See [docs/adding-models.md](docs/adding-models.md). The short version:

1. Define a config struct (deserializes from HuggingFace `config.json`)
2. Wire together gallium-core blocks in a `load()` function
3. Implement `CausalLM` (forward + reset)
4. Add it to `gallium-models/src/lib.rs`, and to `Arch` in `gallium-agent/src/llm_gallium.rs`

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Adding Models Guide](docs/adding-models.md)
- [Building Blocks Reference](docs/building-blocks.md)
- [Target Model Notes](docs/target-models.md)
- [GPT-OSS Notes](docs/gpt-oss.md)
- [Qwen 3.5 Notes](docs/qwen35.md)
- [Gemma 4 Notes](docs/gemma4.md)
- [Test Suite](testsuite/README.md)

## License

MIT
