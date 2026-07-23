# CLAUDE.md

## Project Overview

rs-gallium is a simple, paper-friendly LLM inference framework in Rust. It provides composable building blocks (attention, FFN, RoPE, normalization) that researchers can wire together to implement new model architectures quickly.

Target models: GPT-OSS, Qwen 3.5, Gemma 4, LFM2.5. The workspace also ships `gallium`, a ReAct agent binary that runs those models locally (or OpenAI in the cloud) as a REPL or as a JSON-RPC whole-turn backend for other agents.

## Essential Commands

```bash
# Build (release) / install the binary to ~/bin
make build
make install

# Check (fast compile check)
cargo check --workspace

# Run tests
cargo test --workspace

# Format / lint
cargo fmt --all
cargo clippy --workspace

# Agent capability tests (matrix of testcases × backend configs)
make testsuite                  # all available backends
make testsuite-local            # local backends only (no OPENAI_API_KEY needed)
bash testsuite/runner.sh capital gemma4        # one testcase × one backend

# Run the agent (settings come from env vars over an optional TOML --config)
make run CONFIG=configs/qwen3.6.toml
OPENAI_API_KEY=sk-... gallium --config configs/openai.toml
MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf gallium
```

## Architecture

### Workspace Layout

- `crates/gallium-core/` — All reusable building blocks. Zero model-specific code.
- `crates/gallium-models/` — Concrete model implementations using gallium-core blocks.
- `crates/gallium-agent/` — The `gallium` binary: ReAct agent REPL + app-server, tools, MCP, skills, providers.
- `configs/` — TOML configs for the agent (`--config`).
- `testsuite/` — Agent capability tests: `runner.sh`, `matrix_runner.sh`, `backends/*.toml`, `testcases/*/`.
- `docs/` — Documentation.
- `references/` — Reference implementations (transformers, llama.cpp, vllm, mistral.rs). Cloned via `bash references/setup.sh`. Gitignored, not built by cargo.

### Key Design Decisions

- **Concrete structs + enum dispatch** over traits. Only one trait in the core: `CausalLM`.
- **Per-layer heterogeneous config**: layers can have different attention types, RoPE, FFN.
- **candle-core/candle-nn** as tensor backend for the native engine (git dependency pinned to rev 097655a2).
- **Two local inference engines**: in-process llama.cpp (`local` feature, the default) and native candle (`gallium` feature). Both on by default; Metal is automatic on macOS, CUDA/Vulkan opt-in.

### Core Modules (gallium-core)

| File | Responsibility |
|------|---------------|
| `attention.rs` | Standard attention (MHA/GQA/MQA), sliding window via mask, logit softcapping, shared K=V, Q-norm |
| `linear_attn.rs` | Gated DeltaNet linear attention with recurrent state |
| `ffn.rs` | GatedFFN (SwiGLU/GeGLU + clamp), MoEFFN (top-k routing + shared expert) |
| `quantized.rs` | GGUF loading: `QVarBuilder`, `QLinear`, `QNorm`, `GgufMetadata` |
| `turbo_quant.rs` | TurboQuant: vector quantization (MSE + InnerProduct modes) — experimental, see docs/TODO.md §2 |
| `turbo_kv_cache.rs` | TurboKvCache: KV cache with TurboQuant compression — experimental, no model uses it yet |
| `block.rs` | TransformerBlock combinator |
| `pos_enc.rs` | RoPE with scaling variants (YaRN, Linear, Llama3, NTK), partial rotary, freq factors |
| `norm.rs` | RMSNorm, LayerNorm wrappers around candle-nn |
| `kv_cache.rs` | KV cache, RecurrentState, cross-layer sharing |
| `mask.rs` | Causal and sliding-window mask builders |
| `sampling.rs` | Greedy, top-k, top-p, temperature sampling |
| `model.rs` | `CausalLM` trait, `generate()` with streaming callback |
| `kernels/` | Hand-written SIMD kernels — currently unreferenced, see docs/TODO.md §3.3 |

### Model Files (gallium-models)

| File | Model |
|------|-------|
| `gpt_oss.rs` | GPT-OSS (safetensors): alternating full/SW attn, MoE, YaRN RoPE |
| `gpt_oss_q.rs` | GPT-OSS (GGUF): quantized variant using QLinear |
| `qwen35.rs` | Qwen 3.5 (safetensors): hybrid DeltaNet + full attn |
| `qwen35_q.rs` | Qwen 3.5 (GGUF): quantized variant |
| `gemma4.rs` | Gemma 4 (safetensors): dual RoPE, shared K=V, PLE, softcapping |
| `gemma4_q.rs` | Gemma 4 (GGUF): quantized variant |
| `gemma4_vision.rs` | Gemma 4 vision tower — compiles and is exported, but nothing calls it |
| `lfm2moe_q.rs` | LFM2.5 (GGUF only): hybrid short-conv + GQA MoE |
| `loader.rs` | safetensors loading via VarBuilder |

### Adding a New Model

1. Add `your_model.rs` in `crates/gallium-models/src/`
2. Define config struct (serde deserialize from HuggingFace `config.json`)
3. Wire gallium-core blocks in `load()`, implement `CausalLM`
4. Add `pub mod your_model;` to `lib.rs`
5. Add an `Arch` variant in `gallium-agent/src/llm_gallium.rs` — wire `from_hint()` (GGUF `general.architecture` / safetensors `model_type`), the load `match`, and a `ModelProtocol`
6. Verify `vb.pp()` paths match safetensors weight names

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
| `main.rs` | The `gallium` binary: mode selection (REPL vs `app-server`), env/config resolution, REPL loop |
| `config.rs` | TOML `--config` schema (`[llm]`, `[agent]`, `[[mcpServers]]`) and `--config` flag parsing |
| `lib.rs` | Library root: `Agent`, `create_provider`, `ChatMessage`, `ConversationMemory` re-exports |
| `llm.rs` | `LlmProvider` trait, `OpenAiProvider` (Responses API), `InferenceEngine` selection |
| `llm_local.rs` | In-process llama.cpp backend (`local` feature); renders the GGUF's jinja chat template via minijinja |
| `llm_gallium.rs` | Native candle backend (`gallium` feature); `Arch` detection, model load, protocol dispatch |
| `protocol.rs` | `ModelProtocol` trait + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol`, `Lfm2Protocol` (candle backend only) |
| `harmony.rs` | Harmony chat template rendering |
| `gemma.rs` | Shared Gemma native tool-call parsing, used by both local backends |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `tool.rs` | `ToolHandler` trait, `ToolRegistry`, `ApprovalSink`, and the built-in tools |
| `memory.rs` | `ConversationMemory`: multi-turn history with compaction |
| `skill.rs` | `SkillRegistry`: loads SKILL.md files |
| `situation.rs` | Situation messages surfaced to the model between turns |
| `state_updater.rs` | `BackchannelDetector` for conversational state |
| `github.rs` | GitHub issue/project tools |
| `model_downloader.rs` | Resolves `hf:ORG/REPO[@REV]/file.gguf` into the HF cache (transactional, resumable) |
| `mcp_client.rs` / `mcp_client_http.rs` | MCP clients (stdio / streamable HTTP) wrapping remote tools as `ToolHandler` |
| `mcp_server.rs` / `mcp_server_http.rs` | MCP servers exposing gallium's own tools |
| `mcp.rs` | Shared MCP types |
| `appserver/` | JSON-RPC whole-turn agent backend (`mod.rs`, `rpc.rs`, `server.rs`, `tools.rs`) |

**Built-in tools** (registered in `create_default_registry`): `read`, `glob`, `ls`, `grep`, `write`, `edit`, `multi_edit`, `bash`, `tasks`, `lookup_skill`, `read_situation_messages`

`write` / `edit` / `multi_edit` / `bash` route through `ApprovalSink` before mutating.
On a TTY that prompts the user; in app-server mode it becomes an
`item/fileChange/requestApproval` request to the client, honoring its `approvalPolicy`.
`KESSEL_AUTO_APPROVE=1` is the non-interactive escape hatch for CI and tests.

**Provider routing:** every provider — OpenAI, llama.cpp, native candle — runs the
same ReAct loop in `react.rs`. There is no plain-chat path any more.

**Protocol adapters** apply to the **native candle backend only**; the llama.cpp
backend uses the chat template embedded in the GGUF instead. `ModelProtocol` has:

- `format_prompt(&[ChatMessage]) -> String` — renders history to model-specific token string
- `parse_response(&str) -> String` — extracts user-facing reply from raw decoded output

| Protocol | Model | Notes |
|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + channel instructions; extracts `final` channel |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template |
| `QwenProtocol` | Qwen 3.5 | ChatML `<\|im_start\|>role` template |
| `Lfm2Protocol` | LFM2.5 | Reasoning model — emits a `<think>` block before the answer |

### CLI surface

The binary parses exactly one flag, `--config <path>` (also `-c` / `--config=`), plus
an optional leading `app-server` positional. **Everything else is env vars or config
file keys** — there are no `--arch` / `--model` / `--dtype` / `--provider` flags.
Precedence is env > config file > built-in default. See README.md for the full table.

### app-server protocol

`gallium app-server` speaks line-delimited JSON-RPC on stdio: `initialize` (with
`experimentalApi` capability negotiation), `initialized`, `thread/start` (accepts
client `dynamicTools`), `turn/start`, `account/read`; outbound `item/*`,
`turn/completed`, `turn/failed`, and approval requests.

This is deliberately the same wire protocol codex's app-server presents, and is what
`../rs-kessel` and `../klein-cli` call "ACP". It is **not** the agentclientprotocol.com
standard (`session/new` / `session/prompt`) — adopting that was declined in issue #15.
When touching this area, keep the two senses of "ACP" distinct.

**stdout is the JSON-RPC stream in this mode.** Logging is redirected to stderr in
`main.rs`; anything that prints to stdout will corrupt the protocol.

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
