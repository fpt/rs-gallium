# CLAUDE.md

## Project Overview

rs-gallium is a simple, paper-friendly LLM inference framework in Rust. It provides composable building blocks (attention, FFN, RoPE, normalization) that researchers can wire together to implement new model architectures quickly.

Target models: GPT-OSS, Qwen3.8, Gemma 4, LFM2.5. The workspace also ships `gallium`, a ReAct agent binary that runs those models locally (or OpenAI in the cloud) as a REPL or as a JSON-RPC whole-turn backend for other agents.

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
# Run one at a time — each local backend loads a multi-GB model into GPU/CPU
# memory, so overlapping runner.sh/matrix_runner.sh invocations (even against
# different backends) compete for the same VRAM. `make testsuite`/`testsuite-local`
# do NOT build first (a plain invocation would silently drop CARGO_FEATURES like
# `cuda` from whatever's on disk) — run `make build` yourself first.
make testsuite                  # all available backends
make testsuite-local            # local backends only (no OPENAI_API_KEY needed)
bash testsuite/runner.sh capital gemma4        # one testcase × one backend

# Run the agent (settings come from env vars over an optional TOML --config)
make run CONFIG=configs/qwen3.8.toml
OPENAI_API_KEY=sk-... gallium --config configs/openai.toml
MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf gallium
```

## Architecture

### Workspace Layout

- `crates/gallium-core/` — All reusable building blocks. Zero model-specific code.
- `crates/gallium-models/` — Concrete model implementations using gallium-core blocks.
- `crates/gallium-agent/` — The `gallium` binary: ReAct agent REPL + app-server, tools, MCP, skills, providers.
- `configs/` — TOML configs for the agent (`--config`).
- `testsuite/` — Agent capability tests: `runner.sh`, `matrix_runner.sh`, `backends.txt` (which `configs/*.toml` to test, resolved directly — no separate testsuite-only config copies), `testcases/*/`, `fixtures/make_fixtures.py`.
- `docs/` — Documentation.
- `references/` — Reference implementations (transformers, llama.cpp, vllm, mistral.rs). Cloned via `bash references/setup.sh`. Gitignored, not built by cargo.

### Key Design Decisions

- **Concrete structs + enum dispatch** over traits. Only one trait in the core: `CausalLM`.
- **Per-layer heterogeneous config**: layers can have different attention types, RoPE, FFN.
- **candle-core/candle-nn** as tensor backend for the native engine (git dependency pinned to rev 097655a2).
- **Two local inference engines**: in-process llama.cpp (`local` feature, the default) and native candle (`candle` feature). Both on by default; Metal is automatic on macOS for both. CUDA is opt-in for both (`--features cuda`, one flag covering llama.cpp and candle together — see `gallium-agent/Cargo.toml`'s `cuda` feature, which reaches candle via `gallium-core?/cuda`); Vulkan is opt-in and llama.cpp-only, since candle has no Vulkan backend.
- **Device is a runtime choice, capability a build-time one**: macOS compiles candle's Metal backend in unconditionally (per-target features on `gallium-core`, which cargo unifies across the workspace — and `candle-nn/metal` is required alongside `candle-core/metal`, or `softmax_last_dim`/`silu`/`sigmoid`/`rope`/`rms_norm` error on a Metal tensor); `--features cuda` compiles candle's CUDA backend in the same way, needing `candle-nn/cuda` for the identical reason. `gallium_core::resolve_device` then honors `GALLIUM_DEVICE` (`auto`/`cpu`/`metal`/`cuda`), so one binary benchmarks whichever accelerators it was built with. Naming an absent device is an error, never a silent CPU run.

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
| `device.rs` | `resolve_device` / `device_name` — `GALLIUM_DEVICE` parsing, accelerator-or-CPU fallback; `par_map_on_cpu` — rayon fan-out on the CPU, serial on an accelerator (candle's Metal queue is not thread-safe) |
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
| `qwen35.rs` | Qwen 3.6 (safetensors): hybrid DeltaNet + full attn — module/file names stay `qwen35`, the underlying `qwen35moe` architecture family Qwen 3.6 shares with older Qwen 3.5 checkpoints |
| `qwen35_q.rs` | Qwen 3.6 (GGUF): quantized variant |
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
5. Add an `Arch` variant in `gallium-agent/src/llm_candle.rs` — wire `from_hint()` (GGUF `general.architecture` / safetensors `model_type`), the load `match`, and a `ModelProtocol`
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
| `llm_candle.rs` | Native candle backend (`candle` feature); `Arch` detection, model load, protocol dispatch |
| `protocol.rs` | `ModelProtocol` trait + `HarmonyProtocol`, `GemmaProtocol`, `QwenProtocol`, `Lfm2Protocol` (candle backend only) |
| `profile/` | `ModelProfile` — one model family's wire behavior (tool-call parsing, reply cleaning, stop markers): `gpt_oss`, `gemma4`, `qwen3`, `lfm2`, `minimax`, `deepseek`, and `generic`; `profile/wire/` is one module per wire format |
| `gemma.rs` | Shared Gemma native tool-call parsing, used by both local backends |
| `input.rs` | `UserInput` — the text *and attachments* a frontend hands a turn; `@image:` parsing for the REPL, data-URL parsing for the app-server |
| `event.rs` | `AgentEvent` / `AgentObserver` — the one progress stream every frontend renders from |
| `cancel.rs` | `CancellationToken` / `TurnContext` — how a running turn is stopped, plus `wait_cancellable` for blocking peers |
| `runtime.rs` | `run_turn` — the one turn path: compact → prompt → skill catalog → ReAct → reply. Used by the REPL and every app-server thread |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `tool.rs` | `Tool` trait, `ToolDescriptor`/`ToolSource`/`ToolAnnotations`, `ToolRegistry` (the capability catalog), `ApprovalSink`, `ToolResult` (model/display split), and the built-in tools |
| `trace.rs` | `TurnTrace` — one turn recorded whole, written per turn when asked for; `to_script()`/`diff()` make a recorded turn replayable |
| `memory.rs` | The compaction policy (`compaction_target` / `compact_messages`), applied by `runtime::run_turn`; `resolve_context_window` settles which window both it and a client's gauge use |
| `skill.rs` | `SkillRegistry`: loads skills, both `*.md` and `<name>/SKILL.md`, from `.claude`/`.agents` skill dirs |
| `project.rs` | `find_context_file`: the project's own `AGENTS.md`/`CLAUDE.md`, injected as a second system message by the REPL |
| `github.rs` | GitHub issue/project tools |
| `model_downloader.rs` | Resolves `hf:ORG/REPO[@REV]/file.gguf` into the HF cache (transactional, resumable — retries automatically on a dropped connection, and fetches every shard of a split `-NNNNN-of-MMMMM.gguf` file, not just the one named); the **only** HTTP client that talks to the hub — `ensure_repo_file`/`list_repo_files` serve the candle backend's `tokenizer.json`, `config.json`, and safetensors shards too, so `SSL_CERT_FILE` (a corporate TLS-intercept proxy) is honored everywhere |
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

**The four decisions a client may answer with** are codex's, in codex's
spelling — `accept`, `acceptForSession`, `decline`, `cancel` — decoded in
`appserver/tools.rs`. They were matched in `snake_case` once, so
`acceptForSession` fell through to a refusal and "yes to all" silently meant
"no"; an answer that decodes to nothing is now refused *and logged*, since a
client and a server disagreeing about the protocol should not look like a user
saying no.

`cancel` is refuse **and stop the turn**, and is deliberately not an
`ApprovalDecision` variant: that enum answers "may this action proceed", which
`cancel` answers exactly as `decline` does. The second half is a property of the
turn, so it rides the turn's existing stop switch — `Thread::current_cancel`, a
shared cell `run_turn` fills and the ending clears, since the sink is built at
`thread/start` and has no other way to reach the turn it is approving for. The
token is fired *before* the refusal is returned, or the ReAct loop could pass its
next boundary and get one more model call away.

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

**Steering:** the other thing a frontend can say to a running turn, and it rides
the same `TurnContext`. `cancel::SteerInbox` is a shared cell the app-server's
`turn/steer` pushes user text into; `react.rs` drains it at two boundaries —
before each model call, and again when the model returns text, where pending
steering makes the turn *continue* instead of ending (that superseded answer is
emitted as `AgentEvent::AgentMessage`, or it would vanish into history unseen).
Delivery is therefore at an iteration boundary and no finer: a model mid-sentence
cannot be handed a new instruction. The REPL never pushes — it reads one line at
a time and has nothing to say while a turn runs. (Codex is no finer either: its
`steer_input` queues onto the same kind of pending-input cell and its turn loop
drains it between model calls. What codex has and gallium does not is *streaming*
— `item/agentMessage/delta` — which is what gives a user the chance to interject
at all.)

The inbox **closes** when the turn stops reading it, and `push` refuses once it
has. That is what lets `turn/steer` answer honestly: the loop's decision to end
is `SteerInbox::finish` — check-and-close as one step — so a steer either lands
before it and carries the turn on, or is refused. A check followed by a separate
decision to return would leave a gap in which a steer is accepted, acknowledged,
and read by nobody.

The endings the turn did not choose close it too, and *where* they close it is
the point: an ending that announces itself first — a provider failure, running
out of iterations — closes before the `emit`, because emitting means taking the
app-server connection's lock and writing a notification, which is a wide enough
window to lose a steer in. `run_observed` closing on the way out is the backstop
for the exits that return immediately. The invariant is not "accepted means
delivered" — a turn can always fail after accepting — but the narrower one that
matters: a steer accepted by a turn that goes on to **complete** is one the model
saw, and anything accepted then dropped arrives with a `turn/completed` whose
status is `failed` or `interrupted`, naming that turn.

A steered continuation is **not charged** against `max_iterations`: that budget
bounds the model's own tool-calling loop, and charging a round the user asked for
means a steer arriving on the last iteration converts an answer that was already
produced into a failed turn — which `runtime::run_turn` then rolls the history
back over. `react.rs` counts `charged` (the budget) and `calls` (every model
call, what traces and logs number) separately.

Not yet recorded in traces. `trace.record_prompt` runs once, on the premise that
later prompts are the first plus the transcript; a steer breaks that, so a
steered turn replays without it. Fixing that means a `record_steer` and a script
step — see below.

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
loop sees them), the prompts of iterations after the first (they are the first
prompt plus the transcript the trace already holds), and text added mid-turn by
`turn/steer` — which is the one of the three that makes a trace *wrong* rather
than merely partial, since the replayed turn is missing input the recorded one
had.

[ADR 0004](docs/adr/0004-execution-traces-as-training-data.md) settles what a
trace is *for* beyond debugging: it is the source of truth a training-data or
eval export is derived from, never itself a training format. That is what makes
the three omissions above, plus the 16 KiB `capture()` truncation, the list of
things a full-fidelity mode has to close — and why an outcome label is a sidecar
keyed by session id rather than a field on the record.

**Context window** (`memory::resolve_context_window`): settled per thread from
three sources, in order — the user's explicit `contextWindow`, then
`LlmProvider::context_window()` (llama.cpp reports the GGUF's `n_ctx_train`;
candle reports `<arch>.context_length` or `max_position_embeddings` from the
model it loaded; OpenAI reports nothing), then a fallback of
`LOCAL_CONTEXT_WINDOW` / `DEFAULT_CONTEXT_WINDOW` by where the model runs.

It yields two values, and the split is the point. `effective` is what compaction
measures against and is always a number — a policy has to have a threshold.
`known` is `Some` only for the first two sources, and is what a gauge is allowed
to show: a percentage of a fallback would dress a guess as a measurement, which
is what `fpt/voice-agent#18` deleted its gauge to stop doing.

Zero is handled on both inputs, differently. A *configured* zero switches
compaction off and is honored as `effective: 0`, but it is not a window, so
`known` is `None` — everything downstream divides by it. A *reported* zero is a
model file that said nothing useful, and is discarded before it can switch
compaction off by accident. `known` is therefore never `Some(0)`.

Note that the provider's answer now also drives *compaction*, not just display —
a 32k local model no longer gets trimmed at the old 8192 guess.

Deliberately not `n_ctx`, the size llama.cpp actually builds a context at:
`llm_local.rs` opens each one at `n_ctx.max(n_prompt + max_tokens)` so a long
prompt is never refused, and a gauge against that denominator would grow to meet
its own numerator and never fill.

**Speed** (`llm::Timing`, hung off `TokenUsage::timing`): a model call is timed
in two halves — `prefill` (call start → first sampled token) and `decode` (first
token → last) — because they scale differently and a combined average hides
which one a change moved. Both local backends fill it in; OpenAI leaves it
`None`, since a blocking API cannot say when generation started, and reporting
the round trip as prefill would be a measurement of the network.

Prefill is timed from the *provider's* entry, not from the first `llama_decode`:
`llm_local` tokenizes and finds or builds a context there, and all of it is part
of the wait.

`TokenUsage::timed_partial_prefill` exists because of the KV cache below:
`input_tokens` is the whole prompt, which is what a context gauge measures,
while `Timing::prefill_tokens` is what was actually **evaluated**. Dividing the
whole prompt by the time to evaluate a hundred tokens would report a throughput
the hardware never reached, and it would climb the better the cache worked.
`(evaluated N)` on the usage log line is the gap.

Two rules make a turn's aggregate honest. `ttft` is the **first** call's prefill
and is never summed — it is the wait before the turn showed any sign of life,
and a sum is a latency nobody experienced. And `Timing` carries its **own**
token counts (`prefill_tokens`, `decode_tokens`) rather than reading them off
the `TokenUsage` it hangs on: `decode_tokens` drops each call's first token,
which its prefill produced, and keeping the counts inside means a total covering
both timed and untimed calls still divides timed durations by timed tokens.
Taking them from the usage would price another provider's output against this
one's clock — wrong in the flattering direction, and invisible. Hence
`TokenUsage::timed` takes the two `Duration`s and builds the `Timing` itself,
so the counts cannot drift from the ones reported beside them. An unmeasurable
rate renders as `n/a`, never `0.0`.

**Prompt purity** ([ADR 0001](docs/adr/0001-prompt-purity-and-explicit-context.md)):
the reuse below is worth only as much as the prompt is a *prefix* of the next
one, so gallium does not silently inject volatile state (time, cwd, git status)
into it — a changed token near the front discards the whole transcript behind
it. Volatile facts arrive as tool results or as client-supplied turn input,
where they also arrive *fresh* rather than as a turn-start snapshot the model
reads as current ten iterations later. The one known violation is
`HarmonyProtocol`'s `Current date:` line. When editing prompt construction, the
property to protect is that the same logical conversation yields the same token
prefix.

**KV cache reuse across ReAct iterations** (`llm_local`, issue #86): iteration
*N*'s prompt is a prefix of iteration *N+1*'s, so re-evaluating the whole
transcript each time was ~97% of an agent turn. `llm_local` now retains
`LlamaContext`s in a **slot pool** and evaluates only the suffix — measured
11.62s → 0.16s on the second iteration of a 2.3k-token turn.

Matching is on **token ids**, never on an assumption that the prompt was
appended to: the next prompt is re-rendered through the jinja template, and the
assistant turn inside it is gallium's `serialize_tool_calls` output rather than
the tokens the model emitted. A divergence anywhere just shortens the reuse.

Four things hold it together:

- **One token is always left to evaluate.** The sampler reads the logits of the
  last position *evaluated*; a fully-cached prompt evaluates nothing and would
  sample from whatever the previous call left behind.
- **The record may lag the cache, never lead it.** `Slot::tokens` is what the
  cache is *known* to hold; everything past it is cleared before the next call
  reads it. That is what makes a cancelled or failed generation leave a stale
  slot rather than a wrong one.
- **Slot sizes are rounded up to 4096.** Sized exactly, a slot built for an
  11 836-token prompt cannot hold the 11 882-token prompt of the next
  iteration, and llama.cpp cannot grow a context in place — so every iteration
  would rebuild and re-prefill, which is the cost being removed.
- **Slots are checked out, not held under the pool lock.** Locking across a
  decode would serialize the app-server's concurrent turns, which run in
  parallel today. A turn that finds every slot busy uses a throwaway context and
  is exactly as fast as before.

`GALLIUM_KV_CACHE_SLOTS` sets the pool size; `0` switches reuse off, which is
how the two paths get compared on one turn. Default **1** — a slot is a whole KV
cache, so the second costs as much memory as the first. Right for the REPL; an
app-server interleaving conversations wants one per conversation and pays for
them. The media path never uses a slot.

The retained contexts are why `LlamaLocalProvider` boxes its model and holds
`LlamaContext<'static>`: a context borrows the model, so the struct is
self-referential. The box gives a stable address, `slots` is declared **before**
`model` so it drops first, and `Drop` clears the pool explicitly.

The REPL prints per call (`⏱`, from `AgentEvent::Usage`) and per turn (on the
📊 line), and traces carry `usage.timing` as `TimingRecord` in milliseconds.
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) has the rest: which llama.cpp knobs
are reachable today (`GALLIUM_GPU_LAYERS` and little else), which are fixed in
code and what would expose them, and the plan for searching them.

**Multimodal input** (`input.rs`) — full reference in
[docs/MULTIMODAL.md](docs/MULTIMODAL.md), including the projector table, token
costs, and refusal meanings. In short: a turn is text *plus attachments*.
`runtime::run_turn` takes an `impl Into<UserInput>`, so a caller with nothing to
attach still passes a `String`, and `ChatMessage::user_with_media` puts the
attachments on the message the provider sees.

Two frontends fill it from two shapes. The REPL gets one line of stdin, so it
parses `@image:<path>` and `@audio:<path>` markers out of it — recognized only
at a whitespace boundary (`user@image:host` is an email), relative to the
agent's working dir, quoted for paths with spaces. One scanner handles both and
pushes onto **one ordered `Vec<MediaContent>`**, so the order markers appear in
is the order the model receives — `@audio:note.wav @image:shot.png` is not the
same prompt as its reverse, and a vec-per-modality could not say which was
written. The app-server is handed structured items and reads the `image` ones, accepting only
base64 `data:image/…` URLs; a remote URL would mean this process fetching
something a client chose, which is a different decision. `prompt_input` is
shared by `turn/start` and `turn/steer` so there is one set of parsing rules,
but a steer carrying an image is **refused**: `SteerInbox` carries a `String`
and there is no image on that path.

Nothing is ever dropped quietly. A `@image:`/`@audio:` that will not load fails
the turn; an app-server image item that cannot be read is counted and logged;
and any provider that cannot carry an attachment **refuses** rather than
building its prompt without it — candle via `llm::reject_media`, OpenAI via
`reject_audio` (it does carry images), llama.cpp via `refuse_unsupported_media`,
which names *which* piece is missing: no projector configured, versus a
projector with no encoder for that modality. The reason is the same everywhere:
an attachment nobody looked at produces a confident answer about something the
model never received, indistinguishable from a model that cannot perceive.
`gemma4_vision.rs` is still wired to nothing — the native candle backend has no
multimodal path.

**The llama.cpp backend does image and audio, through `mtmd`** — llama.cpp's
multimodal front end, driven by a projector (`mmproj-*.gguf`) named by
`[llm] mmprojPath` / `MMPROJ_PATH`. The `llama-cpp-2` `mtmd` feature is on
unconditionally: the cost is build time only, since an `MtmdContext` exists only
when a projector is configured, and a text turn never touches it.

`llm_local` has two prompt paths as a result, chosen by whether the turn carries
media at all. Text takes the path it always did — tokenize, one batch, decode.
Media goes through `stage_media` → `build_prompt` → `generate_with_media`:
`<__media__>` markers are injected into message content, `MtmdBitmap::from_buffer`
takes the raw file bytes (llama.cpp links stb_image and miniaudio and sniffs the
format, so gallium decodes neither), `tokenize` splits the prompt into text and
media chunks, and `eval_chunks` runs the projector and the decode in order.
After that both paths share `sample_until_done`.

Markers and bytes are produced by **one pass**, deliberately: mtmd pairs them
positionally, so building them separately could hand the model the wrong picture
— an error nothing downstream could detect.

**Which models can do it.** Every Gemma 4 (E2B/E4B, 12B, 26B-A4B) and
Qwen3.8-27B handles text and image, and their GGUF repos publish the
projector beside the model; GPT-OSS and LFM2.5 do not. Qwen3.8-27B's
projector is vision-only (`clip.has_vision_encoder` with no audio-encoder
field at all) — confirmed by loading it, not assumed from the model card. Audio is E2B/E4B/12B only — 26B-A4B's
projector (`mmproj-BF16.gguf`, ~550M vision params, 1.19GB) reports
`audio=false` at startup (the "Projector supports: vision=.., audio=.." log
line), matching its model card's supported modalities (text, image). The two
Gemma generations that *do* have audio differ in a way that shows up in
results, not just headers:

| | E4B | 12B Unified |
|---|---|---|
| Design | dedicated encoders | encoder-free |
| Projector | 1411 tensors, 478M, `vision.block_count=16`, `audio.block_count=12` | 11 tensors, 52M, `block_count=0` both |
| Download | 946 MB | 167 MB |
| Types | `gemma4v` / `gemma4a` | `gemma4uv` / `gemma4ua` |
| Transcription | exact | *"the secret code word is zuki"* |

So the small model's projector is the *larger* download, and it is the one that
transcribes correctly. Encoder-free buys size at the cost of audio fidelity;
vision is fine on both.

**Audio is user input only.** `AudioContent` and `@audio:` exist, but there is
still no `ToolContent::Audio`, so no tool can produce a clip, and the app-server
has no audio item (codex defines `imageUrl`, nothing for sound). It reaches a
model only on llama.cpp with a projector that has an audio encoder — OpenAI and
candle refuse the turn (`llm::reject_audio` / `reject_media`) rather than
answering about a clip they never received.

**Provider routing:** every provider — OpenAI, llama.cpp, native candle — runs the
same ReAct loop in `react.rs`. There is no plain-chat path any more.

**Model profiles** (`profile/`, [ADR 0003](docs/adr/0003-model-profiles.md)): what
gallium knows about one model *family*'s wire behavior — how it writes a tool
call, how it marks its reasoning, where generation stops, whether its chat
template declares tools natively. `ModelProfile`'s **default method bodies are
the generic behavior**, so a concrete profile is a unit struct overriding only
what its family does differently; profiles are compiled in and a config selects
one by name (it cannot define one, since the parsers are algorithms with boundary
rules, not patterns). Selection is `GALLIUM_PROFILE` > `[llm] profile` >
detection from what the model reports (`general.architecture`, the embedded
template) > `Generic`. **Naming a profile that does not exist fails the load**,
listing the valid names — the same rule `resolve_device` follows for an absent
device, and for the same reason: asking for a profile and silently getting the
generic one shows up only as a model that answers badly.

Six families plus the fallback: `gpt-oss` (Harmony), `gemma4`
(`<|tool_call>call:…`, the thought channel, and the only family with generation
stop markers), `minimax-m2`, `deepseek-v4` (DSML), and `qwen3` / `lfm2`, which
claim **no** native format — Qwen wraps a JSON object in `<tool_call>` tags and
the balanced-span scan reads it out of the middle, so the prose protocol is
already its actual path. Their files say so explicitly, because "this family has
no override" and "nobody has looked" are different states and #116 was the
second one.

For LFM2 that has now been measured rather than assumed (see `lfm2.rs`): its
template *does* declare tools, and claiming them changes nothing — 5 of 7
testcases pass either way. Its `<|tool_*|>` markers are **control** tokens
decoded with `special=false`, so a native call reaches the parser as a bare
`[Read(file_path="a.txt")]` — which is why `wire::python` exists at all, and why
every profile keeps it in `fallback_calls`. Its two failures (`coding`,
`refactoring`) are a *third* thing: the model answers a write with
`{"Write": {…}}`, a shape `wire::json` does not accept, so the call is printed to
the user as text. Gemma 4 E4B passes the same write case through its own native
format, which quotes with `<|"|>` and carries code intact.

The point is scope. Every wire parser used to run against every model's output
in one lenient cascade, so each new family put a new parser in front of all the
existing ones — the bug class behind the `to=`/`string=`/`</think>` fixes. That
cascade now lives in `Generic` alone, where it is the honest answer for a GGUF
nothing is known about; a recognized family reads only its own formats, which
`profile::tests` pins as a matrix (each family's sample parsed by every profile,
exactly one of which may find a call).

Two rules hold across every family, and both live in the provided
`parse_tool_calls` so a profile cannot forget them. **Reasoning is stripped once,
before any format is tried** — a model reasoning about a call it decided against
has not made one, and `strip_think_blocks` is not idempotent, so per-branch
stripping would cut twice. And **the native format is tried before the JSON
scan**: a native call's argument may itself be JSON carrying a `name` key, which
the balanced-span scan would return as a call to that name. `Generic` keeps the
old json-first order deliberately — improving it would change behavior for
exactly the models nobody has run.

Detection matches the architecture names llama.cpp registers in its own
`llama-arch.cpp`, exactly rather than by prefix where a sibling generation
exists: `gemma4`/`gemma4-assistant` but not `gemma3`, `deepseek4` but not
`deepseek2`, `minimax-m2` but not `minimax-m3`, `qwen3*` but not `qwen2*`. A GGUF
llama.cpp can load must report one of those names, since that is how it picks its
own loader — so the exposure is narrow, and an unrecognized one is logged with the
arch it did not match.

**Two passes, and the split is load-bearing**: every profile is asked
`matches_arch` before any is asked `matches_template`. Architecture names are
exact; template literals are whatever a format happens to spell, and Gemma 4's
`declaration:` is an ordinary word with a colon. In one pass that loose template
hit — from a profile early in `PROFILES` — outranks an exact arch hit from a
profile later in it, so a DeepSeek-V4 model whose template contains "declaration:"
parses as a Gemma. The template pass is only the rescue for an arch nobody here
knows (a fork, or a llama.cpp rename).

The testsuite configs are deliberately **not** pinned to a profile: an explicit
name overrides detection rather than checking it, and the testsuite is the only
place real GGUFs get loaded. Diagnosis comes from the startup log line, which says
which profile was chosen and on which signal.

`profile/wire/` is one module per format (`json`, `python`, `minimax`, `dsml`,
`tags`, `think`), plus `fallback_calls` — the JSON protocol gallium asks for and
the Python-ish call list some models substitute, the two formats that belong to no
family, which every profile falls back to. `crate::harmony` and `crate::gemma` are
the same layer, left at the crate root while `protocol.rs` still shares them.

**Protocol adapters** apply to the **native candle backend only**; the llama.cpp
backend uses the chat template embedded in the GGUF instead. `ModelProtocol` has:

- `format_prompt(&[ChatMessage]) -> String` — renders history to model-specific token string
- `parse_response(&str) -> String` — extracts user-facing reply from raw decoded output

| Protocol | Model | Notes |
|---|---|---|
| `HarmonyProtocol` | GPT-OSS | Injects canonical system prompt with date + channel instructions; extracts `final` channel |
| `GemmaProtocol` | Gemma 4 | `<start_of_turn>user/model` template |
| `QwenProtocol` | Qwen 3.6 | ChatML `<\|im_start\|>role` template |
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
client `dynamicTools` and `skillPaths`), `turn/start`, `turn/steer`,
`turn/interrupt`, `account/read`; outbound `item/*`, `turn/completed`,
`thread/tokenUsage/updated`, and approval requests.

**Every ending is a `turn/completed`** — `completed`, `interrupted`, or
`failed`, with the reason in `turn.error.message` on the last. There is no
`turn/failed`: gallium used to send one, a method codex does not define, so a
codex-native client watched a failed turn simply never end.

The switch had to be made *with* the client, not before it, and the asymmetry is
worth remembering for the next change of this shape. A client reading the
**method** (klein's `classifyNote`) treated any `turn/completed` as success, so
flipping gallium first would have converted every failure into a silent one —
strictly worse than the divergence it fixed. Teaching the client to read the
status is backward-compatible on its own, so it lands first and the window never
exists.

**Responses carry codex's required fields**, so a client deserializing into
codex's Rust/TS types succeeds — the threshold #77 is about. What is *required*
is narrower than it looks: serde defaults an `Option<T>` to `None` even without
`#[serde(default)]`, so only the non-`Option`, non-`default` fields (and `Vec`s,
which have no implicit default) have to be there. That is why `Thread` needs
twelve fields and not the twenty-five it declares.

Where gallium has no concept, the answer is the honest one rather than a
plausible one: `ephemeral: true` (nothing is persisted, and no `path` is
claimed), `sandbox: danger-full-access` (there is no sandbox — the approval
tiers are the containment, and claiming otherwise is the answer here that could
get someone hurt), `sessionId` = the thread's own id (no session tree),
`source: appServer`, `codexHome` = `~/.config/gallium` (where this server
actually keeps its state). `approvalPolicy` reports the *resolved* policy, not
the requested one, so an absent `approvalPolicy` is answered `untrusted` rather
than echoed back as nothing.

`items` on a `Turn` is always empty, and `itemsView` says which silence it is:
`full` at `turn/start` (the turn genuinely has none yet) and `notLoaded` at
`turn/completed` (it had them; they went out as `item/*` notifications and are
not reassembled here).

Tool output is shaped per item variant, because the two disagree and codex
defines no `result` field on either: `mcpToolCall` gets
`result: {content: [...]}`, `dynamicToolCall` gets `contentItems` beside
`success` — the same `inputText` shape a `dynamicTools` client sends results
*back* in. `arguments` is repeated on the completed item so an item describes
itself rather than patching the `item/started` before it, which is why
`AgentEvent::ToolCompleted` carries it.

The mirror structs in `appserver/e2e_tests.rs` (`mod codex_shapes`) are the
test: they are transcribed from codex and let serde judge, because asserting on
hand-listed fields cannot catch a *missing required* one — which is the failure
that hid in `initialize` for the app-server's whole life. When they drift,
re-transcribe from codex; a mirror loosened until it passes proves nothing.

`thread/tokenUsage/updated` is what a client draws a context gauge from —
`{threadId, turnId, tokenUsage: {total, last, modelContextWindow}}`, codex's
shape, emitted as each model call reports what it cost. `last` is that call,
`total` is the thread's running sum. The three fields gallium does not track
(`cachedInputTokens`, `cacheWriteInputTokens`, `reasoningOutputTokens`) are sent
as zero so the arithmetic still works.

`modelContextWindow` is **null unless someone can vouch for it** — see
**Context window** below. A provider that reports no usage produces no
notification at all, rather than a zeroed one: `0%` of a real window is a claim
too.

`turn/start` input items may include `{"type": "image", "imageUrl":
"data:image/png;base64,…"}` — see **Multimodal input** below. `turn/steer` may
not, and says so rather than dropping them.

`turn/steer` adds user text to the turn already running: same turn id, nothing
rolled back, no second turn. `expectedTurnId` is a precondition — a steer aimed
at a turn that has ended is refused rather than delivered to the next one, and so
is one aimed at a turn that has stopped reading. The text reaches the model at
the next ReAct boundary (see **Steering** below). The accepted message is echoed
back as a `userMessage` item, `item/started` *and* `item/completed`, which is
codex's shape for a user message.

`skillPaths` is how a client gets its *own* skills in front of the model: a list
of skill directories or single `SKILL.md` files, relative to the thread's `cwd`
or absolute, loaded after the standard locations and the launch config's
`agent.skillPaths` so they win a name collision. Codex spells this
`skills/extraRoots/set`; gallium takes it at thread start instead, since a
thread's skills do not change under it. A path that loads nothing is logged as a
warning — the one outcome worth knowing about, and the response has nowhere
truthful to put it.

**`thread/start` answers codex's `ThreadStartResponse` and nothing beside it.**
A flat `threadId` and a `skillCount` used to ride along; both are gone. The id
is at `thread.id` in codex's response and nowhere else, so a second spelling
only let a client work against gallium in a way that would fail against
codex — the failure this surface exists to prevent, and a worse one than the
divergence it papered over, since it surfaces only on the switch. `skillCount`
answered a real question (did the client's `skillPaths` land) that no protocol
has a field for, which made reading it a dependency on gallium rather than on
the protocol. Keep new answers inside codex's shape: "additive and harmless"
is how the first divergence always argues for itself.

This is deliberately the same wire protocol codex's app-server presents. It is
**not** the agentclientprotocol.com standard (`session/new` / `session/prompt`):
adopting that was considered and declined in #15 — keep the surface small, since
no editor-integration requirement justified a second wire format. A second
transport translating the same `AgentEvent` stream stays open if one appears.

The one consumer today is `../klein-cli`, whose `pkg/agentserver` is a
standalone client for this protocol — it imports nothing else of klein's and
drives codex and `gallium app-server` interchangeably.

**Gallium serves no Chat Completions API**, and this is the surface that replaces
it — see [ADR 0002](docs/adr/0002-no-chat-completions-api.md). A stateless
`messages[]` endpoint has nowhere to say "this continues that conversation",
which is what thread/turn gives the KV cache. An OpenAI-compatible façade belongs
in a separate adapter process. (Gallium remains an OpenAI *client*; consuming
that protocol and serving it are unrelated.)

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
