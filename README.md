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

**Attaching an image or audio clip.** A prompt line may carry `@image:<path>`
and `@audio:<path>` markers, which are lifted out of the text and sent alongside
it:

```
@image:screenshot.png What is wrong with this layout?
@audio:memo.wav Transcribe this audio exactly.
@image:"my shot.png" @image:other.jpg Compare these two.
```

Paths are relative to the working directory and quoted if they contain spaces.
`png` / `jpeg` / `gif` / `webp` for images, `wav` / `mp3` / `flac` for audio. The
marker is recognized only at a whitespace boundary, so `user@image:host` stays
text. A file that will not load ends the turn with an error rather than being
dropped silently.

**Local multimodal** runs through
[`mtmd`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
llama.cpp's multimodal front end, and needs a **projector** — set `mmprojPath`
(or `MMPROJ_PATH`) to the `mmproj-*.gguf` published beside the model:

```toml
[llm]
modelPath  = "hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
mmprojPath = "hf:unsloth/gemma-4-E4B-it-GGUF/mmproj-BF16.gguf"
```

Every Gemma 4 and Qwen 3.6 ships one, covering **image and audio**; GPT-OSS and
LFM2.5 are text-only. Without `mmprojPath` the backend is text-only too, and
says so rather than answering about media it never received.

The OpenAI backend carries images but not audio. The native candle backend
carries neither. Both refuse rather than dropping an attachment silently.

**Ctrl-C** cancels the turn in progress and returns you to the prompt with the
conversation intact — history rolls back to before the prompt, so the next turn
is unaffected. At an idle prompt it exits, as it always has. Cancelling is
prompt but not instantaneous: generation stops after the current token, `bash`
has its process group killed, an MCP call is abandoned rather than interrupted,
and a cloud round trip has no interruption point at all — so a second Ctrl-C
during the same turn exits. Piped input keeps the default behavior.

### app-server mode

```bash
gallium app-server --config configs/openai.toml
```

Serves the agent as a **whole-turn backend** over line-delimited JSON-RPC on stdio:
the client hands over an entire conversation turn, while gallium runs its own
ReAct loop, tools, and MCP connections inside that turn.
Method set: `initialize` (capability negotiation), `initialized`, `thread/start`,
`turn/start`, `turn/interrupt`, `account/read`, with `item/*` / `turn/completed`
updates (every ending is a `turn/completed`; its `status` says which) and
`item/fileChange/requestApproval` approval round-trips
flowing back out. Clients may inject their own tools via `dynamicTools` on
`thread/start`, and point at their own skills via `skillPaths` on the same call —
a list of skill directories or single `SKILL.md` files, absolute or relative to
the thread's `cwd`. They load on top of the standard locations below and win a
name collision; a path that loads no skills is logged as a warning. Without this
a client whose skills live outside the standard directories has none:
`LookupSkill` is still advertised to the model and answers empty.

`turn/start` answers as soon as the turn is accepted — `{turn: {id, status:
"inProgress"}}` — and the turn runs in the background, reporting through
notifications and ending with `turn/completed`. One turn at a time per thread; a
second is refused, naming the one in flight.

**`turn/interrupt`** (`{threadId, turnId}` → `{}`) stops the running turn, and is
answered only once it has actually stopped, so the response means the turn is
over rather than that the request was heard. The turn then ends as
`turn/completed` with `status: "interrupted"` — not a failure — and its history
rolls back to before the prompt, leaving the thread usable. Stopping is prompt
but not instantaneous: generation stops after the current token, `bash` has its
process group killed, an MCP call is abandoned rather than interrupted, and a
cloud round trip finishes before the turn notices.

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
| native candle | `candle` | GGUF + safetensors; arch auto-detected; needs a `tokenizer.json` (see `tokenizerPath`) |
| scripted (no model) | `scripted` | Replays a JSON script from `modelPath`. For testing, including from another process |

The first two are on by default as cargo features (`local`, `candle`). macOS
builds enable Metal automatically; CUDA and Vulkan are opt-in
(`--features cuda` / `vulkan`) because they depend on host toolkits that `cfg()`
cannot detect.

### The scripted engine

```bash
gallium app-server --config configs/scripted.toml   # or the REPL, same config
```

It answers from a fixed list of steps — one per model call — so a whole turn
including tool calls completes in milliseconds with no weights, no API key, and
no network:

```json
{
  "steps": [
    { "toolCalls": [{ "id": "c1", "name": "LS", "arguments": { "path": "." } }] },
    { "text": "I listed the working directory.", "inputTokens": 42 }
  ]
}
```

This exists because everything that is *not* sampling — the app-server wire
format, the ReAct loop's tool plumbing, approval routing — is what a client
integrates against, and it used to be untestable without a multi-GB download.
That is also why it is a real engine rather than a test fixture: a client's own
CI can drive `gallium app-server` with it and catch protocol drift on either
side. See `crates/gallium-agent/src/llm_scripted.rs` for the format, and
`configs/scripted.toml` for a runnable example.

It deliberately does not match on the prompt or branch: a script that reacted to
what the model was asked would drift from the thing under test.

### Where the native engine finds its tokenizer

Only the `candle` engine needs one — llama.cpp uses the tokenizer inside the
GGUF — and plenty of GGUF repos ship none. `llm.tokenizerPath` says where to get
it, next to `modelPath` in the same config:

```toml
tokenizerPath = "hf:unsloth/gemma-4-E4B-it"   # fetch it from this repo
tokenizerPath = "tokenizer.json"              # a file, relative to this config
tokenizerPath = "/models/gemma"               # a directory holding one
```

A bare `ORG/REPO` and a relative path look identical, so the rule is: an `hf:`
prefix always means a repo; otherwise a path that **exists** is that path; and
anything else is a repo id — which is what `GALLIUM_TOKENIZER_REPO` has always
meant, so a value moved out of that env var keeps working. The env var still
overrides the config, as everywhere else.

Without either, gallium looks for a `tokenizer.json` beside the model and then
in the model's own HuggingFace repo, so many models need no setting at all.

## Configuration

With no `--config`, gallium loads **`~/.config/gallium/config.toml`** if it exists
— the same directory global skills come from (`~/.config/gallium/skills`). That is
what makes `gallium` behave the same from any directory instead of only where a
project TOML happens to sit. `--config` still wins, and the startup banner names
whichever file was read.

Relative paths inside a config resolve against **that config file's directory**,
so a user config's `systemPromptPath = "system-prompt.md"` means
`~/.config/gallium/system-prompt.md` no matter where the agent was started:

```bash
mkdir -p ~/.config/gallium
cp configs/gemma4-12b.toml ~/.config/gallium/config.toml
cp configs/gemma4-system-prompt.md ~/.config/gallium/
# systemPromptPath = "gemma4-system-prompt.md" already points at the copy
```

```toml
[llm]
baseURL = "https://api.openai.com/v1"  # note the uppercase URL
model = "gpt-5.6-luna"
apiKey = ""                            # empty → read OPENAI_API_KEY
modelPath = "hf:ORG/REPO/file.gguf"    # local model; presence selects local over cloud
mmprojPath = "hf:ORG/REPO/mmproj.gguf" # multimodal projector; absent → text only
inferenceEngine = "llamacpp"           # or "candle"
tokenizerPath = "hf:ORG/REPO"          # where the "candle" engine finds tokenizer.json
temperature = 0.7
maxTokens = 4096
contextWindow = 128000                 # history compacts at 90% of this
reasoningEffort = "medium"             # low | medium | high

[agent]
systemPromptPath = "system-prompt.md"  # relative to the config file's dir
maxTurns = 50                          # max ReAct iterations per turn
skillPaths = ["../skills"]             # SKILL.md dirs

[agent.approvals]                      # allow | ask | deny, per risk tier
workspaceWrite = "allow"               # writing inside the workspace root
externalSideEffect = "ask"             # remote APIs — the GitHub tools, MCP
destructive = "ask"                    # unrecognized shell commands, writes that
                                       # leave the workspace

[agent.trace]
dir = ".gallium/traces"                # naming a directory turns tracing on

[[mcpServers]]
command = "godevmcp"                   # stdio transport
args = ["serve"]

[[mcpServers]]
url = "http://127.0.0.1:27182/mcp"     # streamable HTTP transport
```

### Approvals

Every mutating tool call is sorted into a risk tier and answered by the policy
for that tier. Reading is never asked about and has no knob.

| Tier | What lands here | Default |
|---|---|---|
| workspace write | `Write` / `Edit` / `MultiEdit` inside the workspace root | `allow` |
| external side effect | the GitHub tools, MCP servers — effects we cannot inspect or undo from here | `ask` |
| destructive | a `Bash` command that is not whitelisted, or a write that resolves outside the workspace root | `ask` |

`ask` means: ask the driving client if one is attached, else prompt on the
terminal, else **refuse** — and say which `[agent.approvals]` key would have
allowed it. Nothing is granted for want of someone to ask.

"Yes to all" grants a tier for the rest of the session, except the destructive
tier, which is confirmed every time: a blanket yes there is a promise about
commands nobody has read yet.

Under `gallium app-server` the driving client answers everything, including
workspace writes, so the `item/fileChange/requestApproval` round trip is
unchanged; its `approvalPolicy: "never"` still means "stop asking me".

The startup banner prints the policy in force. Set `workspaceWrite = "ask"` to
be prompted before every write.

### Turn traces

Off by default. With `[agent.trace] dir` set — or `GALLIUM_TRACE=1`, or
`GALLIUM_TRACE_DIR=<dir>` — every turn is written to its own JSON file:

```bash
GALLIUM_TRACE=1 gallium                  # → .gallium/traces/turn-<epoch-ms>-<id>.json
GALLIUM_TRACE_DIR=/tmp/traces gallium app-server
GALLIUM_TRACE=0 gallium                  # off for this run, whatever the config says
```

One file holds one turn: the prompt the model was given, the tools it was
offered, what it answered at each iteration, every tool call with its arguments
and result, the approval decision each call provoked, token usage, and how the
turn ended — including a turn that was stopped, which otherwise leaves no trace
of itself at all since its history is rolled back.

A trace is also a **script**: the model outputs it recorded are the format
`INFERENCE_ENGINE=scripted` replays, so a recorded turn can be run again through
the real binary with real tools and no model, and compared against the original.
That is what `TurnTrace::to_script` and `TurnTrace::diff` are for; the diff
covers the tool calls, their arguments, the approval outcomes, and the final
text, and deliberately ignores timings, token counts, and result bodies, which
differ between two runs for reasons that say nothing about the agent.

Two things a trace is not. It is **not** the model's pre-parse output — tool
calls are extracted inside the providers, so what is recorded is the parsed
response. And it is **not** shareable by default: it holds every byte the model
saw, including whatever the tools read out of the workspace. Traces belong next
to the workspace they came from.

Ready-made configs live in `configs/`. Environment overrides:

| Variable | Overrides |
|---|---|
| `MODEL_PATH` | `llm.modelPath` |
| `MMPROJ_PATH` | `llm.mmprojPath` — multimodal projector for the llama.cpp backend |
| `LLM_BASE_URL` / `LLM_MODEL` / `OPENAI_API_KEY` | the `[llm]` cloud fields |
| `LLM_TEMPERATURE` / `MAX_TOKENS` / `REASONING_EFFORT` | sampling + budget |
| `CONTEXT_WINDOW` | `llm.contextWindow` — compaction trigger (default 8192 local, 128000 cloud) |
| `INFERENCE_ENGINE` | `llm.inferenceEngine` |
| `MAX_REACT_ITERATIONS` | `agent.maxTurns` |
| `WORKING_DIR` | tool root (default: cwd) |
| `MCP_SERVERS` | extra stdio servers, `"cmd arg1,cmd2 arg1"` |
| `GALLIUM_DEVICE` | native candle backend device: `auto` (default), `cpu`, `metal`, `cuda`, or `metal:1` |
| `GALLIUM_DTYPE` | native candle backend dtype |
| `GALLIUM_TOKENIZER_REPO` | `llm.tokenizerPath` — the native candle backend's tokenizer source |
| `GALLIUM_GPU_LAYERS` | llama.cpp GPU offload (`0` = CPU) |
| `GALLIUM_KV_CACHE_SLOTS` | llama.cpp retained KV caches (default `1`, `0` disables prompt reuse) — each slot is a whole KV cache |
| `GALLIUM_BASH_ALLOW` | extra allowed `Bash` commands |
| `GALLIUM_TRACE` | `1` turns per-turn traces on (default dir), `0` turns them off whatever the config says |
| `GALLIUM_TRACE_DIR` | `agent.trace.dir` — where traces are written (setting it turns them on) |
| `GALLIUM_GH_ORG` / `GALLIUM_GH_PROJECT` / `GALLIUM_GH_REPO` | GitHub Projects tools (absent = tools not registered) |

> **Renamed from `KESSEL_*` (2026-07-25).** These carried the old repo's name. The
> old names are **no longer read at all** — a `KESSEL_*` variable is now silently
> ignored, so update any scripts and service configs that set one:
>
> | Old | New |
> |---|---|
> | `KESSEL_AUTO_APPROVE` | `GALLIUM_AUTO_APPROVE` |
> | `KESSEL_BASH_ALLOW` | `GALLIUM_BASH_ALLOW` |
> | `KESSEL_GPU_LAYERS` | `GALLIUM_GPU_LAYERS` |
> | `KESSEL_GALLIUM_DTYPE` | `GALLIUM_DTYPE` |
> | `KESSEL_GALLIUM_TOKENIZER_REPO` | `GALLIUM_TOKENIZER_REPO` |
> | `KESSEL_GALLIUM_THINKING` | `GALLIUM_THINKING` |
> | `KESSEL_GH_ORG` / `KESSEL_GH_PROJECT` / `KESSEL_GH_REPO` | `GALLIUM_GH_*` |
>
> Note the `KESSEL_GALLIUM_*` ones lost the doubled prefix rather than becoming
> `GALLIUM_GALLIUM_*`.

## Tools

Registered by default for every provider:

| Tool | Description |
|------|-------------|
| `Read` | Read a file |
| `Write` | Create or overwrite a file *(requires approval)* |
| `Edit` | Replace an exact string in a file *(requires approval)* |
| `MultiEdit` | Apply several edits to one file *(requires approval)* |
| `Glob` | List files matching a pattern |
| `LS` | List a directory |
| `Grep` | Search file contents |
| `Bash` | Run a shell command *(requires approval)* |
| `Tasks` | Create and track tasks |
| `LookupSkill` | Load a SKILL.md by name |

MCP servers from the config (or `MCP_SERVERS`) register their tools alongside these.
Mutating tools prompt for approval on a TTY; in app-server mode the request is
routed to the client, honoring its `approvalPolicy`.

## What the agent reads at startup

Besides the system prompt from `agent.systemPromptPath`, the working directory
is searched for two things — both optional, and the REPL prints what it found:

| Read from | What it is |
|---|---|
| `AGENTS.md`, else `CLAUDE.md` | The project's own instructions, injected as a second system message under `# Project Context`. First non-empty file wins; they are alternatives, not layers. Matches `klein-cli`. |
| `~/.config/gallium/skills/`, `.claude/skills/`, `.agents/skills/`, `.gallium/skills/` | Skills, plus any `agent.skillPaths` from the config. Both layouts load: `*.md` directly in the directory, and `<name>/SKILL.md` one level down. |

Skills are keyed by name and the list above is in increasing precedence, so a
`.gallium/skills` entry overrides a `.claude/skills` one of the same name, and
both override the global directory. In app-server mode a client's own
`skillPaths` load last of all, so they override every one of these.

```
$ gallium --config configs/gemma4.toml
Working dir: /Users/me/src/myproject
Context: CLAUDE.md (13.0 KB)
Skills: 3 from .claude/skills
```

A large context file is a real bite out of a small local model's window — the
size is printed for that reason.

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

The top-level `Dockerfile` builds the `gallium` agent binary itself. It keeps the
env-var + stdin interface, so pass settings as `-e` and mount the workspace:

```bash
make docker-build

# REPL against a local GGUF, downloaded into the mounted HF cache on first use
docker run --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -e MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf \
  gallium

# As a whole-turn backend (stdout is the JSON-RPC stream, so no -t)
docker run --rm -i -e OPENAI_API_KEY -v "$PWD:/workspace" \
  gallium app-server --config /app/configs/openai.toml
```

Writes inside the mounted workspace need no approval; anything outside it, an
unrecognized shell command, or a remote API call is refused without a TTY to ask
at — set `[agent.approvals]` in the config to change that. GPU backends are build
args:
`docker build --build-arg CARGO_FEATURES=cuda -t gallium .`

## Adding a New Model

See [docs/adding-models.md](docs/adding-models.md). The short version:

1. Define a config struct (deserializes from HuggingFace `config.json`)
2. Wire together gallium-core blocks in a `load()` function
3. Implement `CausalLM` (forward + reset)
4. Add it to `gallium-models/src/lib.rs`, and to `Arch` in `gallium-agent/src/llm_candle.rs`

## Documentation

- [Development Notes](docs/DEVELOPMENT.md) — building on Windows, toolchain gotchas
- [Architecture Overview](docs/architecture.md)
- [Candle Metal Backend](docs/CANDLE_METAL.md) — `GALLIUM_DEVICE`, GPU throughput, where decode time goes
- [Optimization](docs/OPTIMIZATION.md) — what a turn's ttft and tok/s mean, which llama.cpp knobs are reachable, and the plan for searching them
- [Multimodal Support](docs/MULTIMODAL.md) — images and audio: what works where, projectors, what media costs
- [Adding Models Guide](docs/adding-models.md)
- [Building Blocks Reference](docs/building-blocks.md)
- [Target Model Notes](docs/target-models.md)
- [GPT-OSS Notes](docs/gpt-oss.md)
- [Qwen 3.5 Notes](docs/qwen35.md)
- [Gemma 4 Notes](docs/gemma4.md)
- [Test Suite](testsuite/README.md)

## License

MIT
