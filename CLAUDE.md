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
- `testsuite/` — Agent capability tests: `runner.sh`, `matrix_runner.sh`, `backends/*.toml`, `testcases/*/`, `fixtures/make_fixtures.py`.
- `docs/` — Documentation.
- `references/` — Reference implementations (transformers, llama.cpp, vllm, mistral.rs). Cloned via `bash references/setup.sh`. Gitignored, not built by cargo.

### Key Design Decisions

- **Concrete structs + enum dispatch** over traits. Only one trait in the core: `CausalLM`.
- **Per-layer heterogeneous config**: layers can have different attention types, RoPE, FFN.
- **candle-core/candle-nn** as tensor backend for the native engine (git dependency pinned to rev 097655a2).
- **Two local inference engines**: in-process llama.cpp (`local` feature, the default) and native candle (`candle` feature). Both on by default; Metal is automatic on macOS, CUDA/Vulkan opt-in.
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
| `gemma.rs` | Shared Gemma native tool-call parsing, used by both local backends |
| `input.rs` | `UserInput` — the text *and attachments* a frontend hands a turn; `@image:` parsing for the REPL, data-URL parsing for the app-server |
| `event.rs` | `AgentEvent` / `AgentObserver` — the one progress stream every frontend renders from |
| `cancel.rs` | `CancellationToken` / `TurnContext` — how a running turn is stopped, plus `wait_cancellable` for blocking peers |
| `runtime.rs` | `run_turn` — the one turn path: compact → prompt → skill catalog → ReAct → reply. Used by the REPL and every app-server thread |
| `react.rs` | ReAct loop: call LLM → execute tool calls → repeat until text response |
| `tool.rs` | `Tool` trait, `ToolDescriptor`/`ToolSource`/`ToolAnnotations`, `ToolRegistry` (the capability catalog), `ApprovalSink`, `ToolResult` (model/display split), and the built-in tools |
| `trace.rs` | `TurnTrace` — one turn recorded whole, written per turn when asked for; `to_script()`/`diff()` make a recorded turn replayable |
| `memory.rs` | The compaction policy (`compaction_target` / `compact_messages`), applied by `runtime::run_turn`; `resolve_context_window` settles which window both it and a client's gauge use |
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
saw, and anything accepted then dropped arrives with a `turn/failed` or an
`interrupted` naming that turn.

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

**Which models can do it.** Every Gemma 4 (E2B/E4B, 12B, 26B) and Qwen 3.6
handles text, image and audio, and their GGUF repos publish the projector beside
the model. GPT-OSS and LFM2.5 do not. The two Gemma generations differ in a way
that shows up in results, not just headers:

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
client `dynamicTools` and `skillPaths`), `turn/start`, `turn/steer`,
`turn/interrupt`, `account/read`; outbound `item/*`, `turn/completed`,
`turn/failed`, `thread/tokenUsage/updated`, and approval requests.

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
thread's skills do not change under it. `thread/start` answers with `skillCount`
beside `threadId`, so a client can see whether its paths landed rather than
inferring it from a model that reports no skills.

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
