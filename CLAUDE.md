# CLAUDE.md

## Project Overview

rs-gallium is a simple, paper-friendly LLM inference framework in Rust. It provides composable building blocks (attention, FFN, RoPE, normalization) that researchers can wire together to implement new model architectures quickly.

Target models: GPT-OSS, Qwen 3.5, Gemma 4, LFM2.5. The workspace also ships `gallium`, a ReAct agent binary that runs those models locally (or OpenAI in the cloud) as a REPL or as a JSON-RPC whole-turn backend for other agents.

## Essential Commands

# Windows builds have extra toolchain requirements (MSVC cargo, Ninja, /MD CRT) —
# `make build` handles them; see docs/DEVELOPMENT.md for the why.

```bash
# Build (release) / install the binary to ~/bin
make build
make install

# Check (fast compile check)
cargo check --workspace

# Run tests (model integration tests are #[ignore]d — they need multi-GB models)
cargo test --workspace
make test-models                # opt in: skips whichever models are not cached

# Format / lint
cargo fmt --all
cargo clippy --workspace

# Sweep one mechanical edit across many sites (see .claude/skills/sweep-edit)
uv run python scripts/sweep.py --dry-run < edits.json
uv run python scripts/test_sweep.py

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
- **Device is a runtime choice, capability a build-time one**: macOS compiles candle's Metal backend in (per-target features on `gallium-core`, which cargo unifies across the workspace — and `candle-nn/metal` is required alongside `candle-core/metal`, or `softmax_last_dim`/`silu`/`sigmoid`/`rope`/`rms_norm` error on a Metal tensor). `gallium_core::resolve_device` then honors `GALLIUM_DEVICE` (`auto`/`cpu`/`metal`/`cuda`), so one binary benchmarks both. Naming an absent device is an error, never a silent CPU run.

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
| `device.rs` | `resolve_device` / `device_name` — `GALLIUM_DEVICE` parsing, accelerator-or-CPU fallback |
| `gqa.rs` | `gqa_scores` / `gqa_weighted_sum` — the two attention products with grouped Q, so K/V are never expanded to `h` heads |
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
| `lib.rs` | Library root: `AgentError`, `McpServerConfig`, and the crate's re-exports |
| `llm.rs` | `LlmProvider` trait, `OpenAiProvider` (Responses API), `InferenceEngine` selection |
| `llm_local.rs` | In-process llama.cpp backend (`local` feature); renders the GGUF's jinja chat template via minijinja |
| `llm_gallium.rs` | Native candle backend (`gallium` feature); `Arch` detection, model load, protocol dispatch |
| `protocol.rs` | `ModelProtocol` trait + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol`, `Lfm2Protocol` (candle backend only) |
| `gemma.rs` | Shared Gemma native tool-call parsing, used by both local backends |
| `event.rs` | `AgentEvent` / `AgentObserver` — the one progress stream every frontend renders from |
| `cancel.rs` | `CancellationToken` / `TurnContext` — how a running turn is stopped, plus `wait_cancellable` for blocking peers |
| `runtime.rs` | `run_turn` — the one turn path: compact → prompt → skill catalog → ReAct → reply. Used by the REPL and every app-server thread |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `tool.rs` | `Tool` trait, `ToolDescriptor`/`ToolSource`/`ToolAnnotations`, `ToolRegistry` (the capability catalog), `ApprovalSink`, `ToolResult` (model/display split), and the built-in tools |
| `trace.rs` | `TurnTrace` — one turn recorded whole, written per turn when asked for; `to_script()`/`diff()` make a recorded turn replayable |
| `memory.rs` | The compaction policy (`compaction_target` / `compact_messages`), applied by `runtime::run_turn` |
| `skill.rs` | `SkillRegistry`: loads skills, both `*.md` and `<name>/SKILL.md`, from `.claude`/`.agents`/`.gallium` skill dirs |
| `project.rs` | `find_context_file`: the project's own `AGENTS.md`/`CLAUDE.md`, injected as a second system message by the REPL |
| `github.rs` | GitHub issue/project tools |
| `model_downloader.rs` | Resolves `hf:ORG/REPO[@REV]/file.gguf` into the HF cache (transactional, resumable); the **only** HTTP client that talks to the hub — `ensure_repo_file`/`list_repo_files` serve the candle backend's `tokenizer.json`, `config.json`, and safetensors shards too, so `SSL_CERT_FILE` (a corporate TLS-intercept proxy) is honored everywhere |
| `mcp_client.rs` / `mcp_client_http.rs` | MCP clients (stdio / streamable HTTP) wrapping remote tools as `Tool`, carrying the server's annotation hints through |
| `mcp_server.rs` / `mcp_server_http.rs` | MCP servers exposing gallium's own tools |
| `mcp.rs` | Shared MCP types |
| `appserver/` | JSON-RPC whole-turn agent backend (`mod.rs`, `rpc.rs`, `server.rs`, `tools.rs`) |

**Built-in tools** (registered in `create_default_registry`): `Read`, `Glob`, `LS`, `Grep`, `Write`, `Edit`, `MultiEdit`, `Bash`, `Tasks`, `LookupSkill` — PascalCase, matching Claude Code and klein-cli, since gallium's app-server reports these names to those clients. The registry resolves a call whose case or underscores differ, so a model that emits `read` or `multi_edit` still lands.

**Approvals** (`approval.rs`): every mutating call is sorted into a `RiskLevel` —
`ReadOnly`, `WorkspaceWrite`, `ExternalSideEffect`, `Destructive` — and an
`ApprovalBroker` applies the `ApprovalPolicy` rule for that tier (`allow` / `ask` /
`deny`, from `[agent.approvals]`). The tier is a property of the *call*, not the
tool: a write inside the workspace root is `WorkspaceWrite`, the same write to a
path outside it is `Destructive`. Defaults: workspace writes proceed, everything
else asks.

`ask` consults the `ApprovalSink` if one is installed (app-server →
`item/fileChange/requestApproval`, honoring the client's `approvalPolicy`), else
prompts on a TTY, else **refuses** and names the config key that would have
allowed it. A session grant ("yes to all") is per tier and is never given for
`Destructive`. The app-server deliberately runs `ApprovalPolicy::CAUTIOUS` so
every mutation still reaches the client.

There is no `GALLIUM_AUTO_APPROVE` any more — `ApprovalPolicy::PERMISSIVE` is the
deliberate, config-only equivalent. `testsuite/runner.sh` needs nothing: its temp
dir is the workspace root, so its writes are the tier that is allowed.

**The tool catalog:** `ToolRegistry::register` computes a `ToolDescriptor` once
per tool (name, description, schema, `ToolAnnotations`, `ToolSource`) and keeps
it; `ToolAccess::descriptors()` is the catalog, and `get_definitions()` is the
projection of it the providers see. A new tool implements `Tool` and overrides
`annotations()` when it is not an ordinary workspace write — the default is
"mutates, recoverably", so forgetting gives the cautious answer. `source()`
defaults to `Builtin`; the MCP and `dynamicTools` wrappers override it. MCP's
`readOnlyHint` / `destructiveHint` / `openWorldHint` map onto `ToolAnnotations`
in both directions, so hints survive a round trip through gallium.

**Cancellation:** a turn carries a `TurnContext` (`TurnSetup::context`); `None`
means a turn nobody can stop. The token is checked at every loop boundary in
`react.rs`, between sampled tokens in both local backends
(`chat_with_tools_cancellable`, and `gallium_core::generate`'s `ControlFlow`
callback), and between polls of `Bash`'s child, whose whole process group is killed —
a shell forks for pipelines and background jobs, and those children would
otherwise outlive the turn. Calls into a
peer we do not control — MCP, `dynamicTools` — cannot be interrupted, so
`cancel::wait_cancellable` stops *waiting* and lets the abandoned worker finish.
An OpenAI round trip has no interruption point at all and completes. Neither
frontend cancels yet: the app-server surface is #28.

**Traces** (`trace.rs`): off unless `[agent.trace] dir`, `GALLIUM_TRACE=1`, or
`GALLIUM_TRACE_DIR` says otherwise (`GALLIUM_TRACE=0` wins over a config). When
on, `runtime::run_turn` mints a `TurnRecorder`, hangs it on the `TurnContext`,
and writes one JSON file per turn — every ending, cancellation included, since a
stopped turn rolls its history back and would otherwise leave nothing behind.

The recorder collects the first prompt and the tool catalog before the loop, the
response and usage per iteration, and each tool call's arguments, result, and
duration. Approval decisions come from `ApprovalBroker::take_journal`, drained
after each call and attributed to it: a decision is made below `Tool::call`,
which takes no `TurnContext`, and threading one through twenty tool impls to
carry a record out would be a large change for a small fact.

A trace **is a script**: `TurnTrace::to_script()` produces what
`INFERENCE_ENGINE=scripted` (`llm_scripted.rs`) replays, so a recorded turn
re-runs through the real loop with real tools and `TurnTrace::diff` reports where
it diverged — tool calls, arguments, approval outcomes, and the final text, not
timings or result bodies. That is the replay-based test in `trace.rs`.

Not recorded: the model's pre-parse output (providers parse tool calls before the
loop sees them) and the prompts of iterations after the first (they are the first
prompt plus the transcript the trace already holds).

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

With no `--config`, `config::default_config_path` loads `~/.config/gallium/config.toml`
— the directory global skills already load from. Every relative path *inside* a
config — `systemPromptPath`, `skillPaths`, `modelPath`, `tokenizerPath` — resolves
against that config file's own directory, never the cwd, which is what lets one
user-level config work from every directory.

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
