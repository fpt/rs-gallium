# GaLLiuM inference framework in Rust

A simple, paper-friendly LLM inference framework in Rust, with an agent (`gallium`) built on top of it.

rs-gallium provides composable building blocks that map directly to how research papers describe transformer architectures. When a new paper proposes a novel attention mechanism, FFN variant, or position encoding, you can implement and test it with minimal boilerplate.

## Target Models

- **GPT-OSS** (OpenAI) — alternating full/sliding-window attention + MoE
- **Qwen3.8** (Alibaba) — hybrid Gated DeltaNet (linear attention) + full attention
- **Gemma 4** (Google) — dual RoPE, shared K=V, per-layer embeddings, logit softcapping
- **LFM2.5** (LiquidAI) — hybrid short-conv + GQA MoE (GGUF only)
- **DeepSeek-V4-Flash** (DeepSeek) — MoE, 256 routed experts + 1 shared, top-6 routing (llama.cpp backend only — no native candle implementation)
- **MiniMax-M2.7** (MiniMax) — MoE, 256 experts, top-8 routing (llama.cpp backend only — no native candle implementation)

## Structure

```
crates/
  gallium-core/     # Composable building blocks + generation
  gallium-models/   # Model implementations (GPT-OSS, Qwen3.8, Gemma 4, LFM2.5)
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

# CUDA GPU offload: on Linux/macOS a bare `make build` is CPU-only — pass the
# feature explicitly. (Windows defaults to it; `make build CARGO_FEATURES=`
# there for CPU-only. See docs/DEVELOPMENT.md for toolchain requirements.)
make build CARGO_FEATURES=cuda
```

### REPL mode (default)

```bash
# Cloud (OpenAI Responses API)
OPENAI_API_KEY=sk-... gallium --config configs/openai.toml

# Local GGUF via the in-process llama.cpp backend (the default engine)
gallium --config configs/qwen3.8.toml

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

Every Gemma 4 ships one, covering **image and audio**; Qwen3.8 ships one too,
but vision-only, with no audio encoder at all. GPT-OSS and LFM2.5 are
text-only and have none. Without `mmprojPath` the backend is text-only too, and
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

A client that dialed in over TCP is the exception: its `skillPaths` are ignored,
and so are the workspace's own skill directories, because both are paths the
client named in *its* filesystem — see [Whose machine the tools run
on](#whose-machine-the-tools-run-on).

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

This is deliberately the same wire protocol codex's app-server presents. It is
**not** the agentclientprotocol.com standard (`session/new` / `session/prompt`);
adopting that was considered and declined (issue #15), so the surface here stays
small. [klein-cli](https://github.com/fpt/klein-cli)'s `pkg/agentserver` is a
standalone client for this protocol, and drives `gallium app-server` or codex
interchangeably.

In this mode stdout carries the JSON-RPC stream, so all logging is redirected to
stderr. Anything else writing to stdout will corrupt the protocol.

#### Over TCP, when the client is on another machine

```bash
# on the GPU box
gallium app-server --listen 0.0.0.0:47821 --config configs/qwen3.8.toml
```

Same protocol, same methods — only the byte stream changes, from stdin/stdout to
a persistent TCP connection.

`--listen` is the **only** way to make an app-server listen: there is no env var
and no config key for it, and that is deliberate. Every other setting configures
the server a client *spawns*, and such a client wants stdio — so an address
arriving from the environment or from `~/.config/gallium/config.toml` could only
ever turn a spawned server into one that opens a socket and never reads the stdin
it was handed. The client is told nothing; it waits for a reply that is not
coming. Requiring the address to be typed for the run that wants it makes that
unrepresentable rather than merely documented.

This is what separates the agent's head from its hands. The model runs where the
GPU is, while the client's `dynamicTools` keep running on the machine the user is
sitting at — so a turn can drive an application that only exists on the laptop,
over the same connection that carries the turn. `item/tool/call` already goes
server→client and blocks awaiting the answer, which is why a stream transport fits
and a request/response one would have to reinvent the reverse direction.

#### Whose machine the tools run on

**A listening server has no tools of its own.** Not a setting: gallium's
built-ins run as the user *gallium* was started as, and this socket carries no
authentication or identity, so a configurable version would hand whoever reaches
the port a `Bash` with that user's privileges. Same machine is not the same user,
so loopback earns no exception.

A networked thread therefore keeps only the two tools that touch no machine —
in-memory task bookkeeping and skill lookup — and everything that reads, writes,
or executes arrives as the client's `dynamicTools`, dispatched back over the same
connection that carries the turn, running under whoever runs the client:

```
LLM → gallium ReAct → RemoteTool → item/tool/call → TCP → klein → the user's shell
```

The same rule covers everything else a client can name a path in, since taking
the tools away only shuts the front door.

**Skills**: a networked thread loads only what the operator chose —
`~/.config/gallium/skills` and the launch config's `agent.skillPaths` — and
ignores the client's `skillPaths` along with `<cwd>/.claude/skills` and
`<cwd>/.agents/skills`. Reading those would be this host opening files the client
named, with this user's privileges, and returning their contents through the
prompt and `LookupSkill`.

**MCP servers**: a networked thread registers none of the client's. A stdio MCP
server is a command line, and gallium would spawn it here, as this user — the
same door one rung worse, since the first reads files and this one runs programs.
An MCP server belongs to the machine whose files and processes it is for, so a
client runs one beside itself and sends its tools as `dynamicTools`.

Both refusals are logged, since the symptom otherwise is a model quietly missing
capabilities the client believes it has.

A client tool also **replaces** a built-in of the same name, which is what makes
`Bash` and friends reusable names over stdio too. It is the only way such a tool
is reachable: gallium resolves a call to the first exact name match, so a client
`Bash` registered behind the built-in one would never be called.

The client must actually send tools, then. One that sends none leaves the model
able to read nothing, write nothing and run nothing — logged as a warning at
`thread/start`, since it otherwise looks like a broken model rather than a
half-configured pair.

Over stdio nothing changes: the client spawned gallium, so its tools already run
with exactly the privileges the client has.

**One client at a time, and the newest one wins.** The limit is the llama.cpp KV
cache: the slot pool holds one context by default (`GALLIUM_KV_CACHE_SLOTS`), and
its value comes entirely from each turn's prompt being a prefix of the next one.
Two conversations interleaving on one slot are not prefixes of each other, so
each evicts the other's tokens and both pay a full re-prefill. A second
connection therefore displaces the first, which gets a clean EOF — rather than
being refused, because a link that died with a sleeping laptop is
indistinguishable from a live one until the OS gives up on it, and refusing would
lock you out of your own GPU box on the reconnect meant to fix it.

Displacing stops the old client's turn rather than merely closing its socket: a
turn runs on its own thread and would otherwise keep calling the model beside the
replacement's turn. The old turns are cancelled, the socket is shut down (which
releases anything blocked awaiting an answer from the client that left), and the
replacement is served only once those turns have actually stopped — bounded, like
`turn/interrupt`, by the slowest thing a turn is currently inside.

The model stays loaded across that reconnect, and with llama.cpp so do the warm
KV slots: the process outlives connections, so the client that comes back finds
its prefix still cached. What it does *not* inherit is the old connection's
threads — ids and history are per connection, and a `threadId` names a
conversation on the connection that created it and nowhere else.

**There is no authentication and no transport encryption.** Anything that can
reach the port can run tools with this process's privileges. Bind loopback (with
an SSH tunnel) or a private overlay address — Tailscale, WireGuard — and let the
overlay do the authenticating; binding anywhere else is logged as the warning it
is. A listener that cannot bind exits with the reason rather than falling back to
stdio, where nothing would ever connect.

## Inference engines

`inferenceEngine` (or `INFERENCE_ENGINE`) selects the local backend:

| Engine | Value | Notes |
|---|---|---|
| llama.cpp, in-process | `llamacpp` *(default)* | GGUF only; renders the GGUF's embedded jinja chat template |
| native candle | `candle` | GGUF + safetensors; arch auto-detected; needs a `tokenizer.json` (see `tokenizerPath`) |
| scripted (no model) | `scripted` | Replays a JSON script from `modelPath`. For testing, including from another process |

The first two are on by default as cargo features (`local`, `candle`). macOS
builds enable Metal automatically for both engines; CUDA is opt-in
(`--features cuda`) and, built that way, covers both engines too — one flag,
one `GALLIUM_DEVICE=cuda` picks the GPU for whichever is running.
`GALLIUM_GPU_LAYERS` is llama.cpp-only: candle has no partial-offload knob, it
runs entirely on whatever `GALLIUM_DEVICE` resolves to.

`GALLIUM_MAX_CTX` / `[llm] maxCtx` is the largest context llama.cpp will be asked
to build, and it is **also the window compaction measures against** — the two
have to be one number, or history is trimmed against a size the card cannot
allocate. It defaults to the model's trained window and is never allowed above
it. What it cannot do is make a card bigger: if the KV cache does not fit in
VRAM, the levers are `GALLIUM_GPU_LAYERS` (fewer layers on the GPU leaves more
room for the cache), `cpuMoe` for an MoE model, or a smaller quant. Raising
`maxCtx` past what fits only means paying one failed allocation.

gallium lowers the ceiling itself when an allocation fails — it retries a chunk
smaller and logs the value to pin in the config — so an unmeasured machine
self-corrects once rather than failing every turn at the same point. That
learned value lasts for the life of the process and only ever descends; it is
not written anywhere, which is why the log line names the number to set. Vulkan
(`--features vulkan`) is llama.cpp-only; candle has no Vulkan backend. GPU
features are opt-in rather than default because they depend on host toolkits
that `cfg()` cannot detect.

## Model profiles

Different model families write a tool call differently — GPT-OSS in Harmony,
Gemma 4 in `<|tool_call>call:NAME{…}`, MiniMax-M2.7 in
`<minimax:tool_call><invoke name="…">`, DeepSeek-V4 in DSML — and mark their
reasoning differently too. A **profile** is what gallium knows about one family's
wire behavior, compiled into the binary:

| Profile | Family | Tool format |
|---|---|---|
| `gpt-oss` | GPT-OSS 20b/120b | Harmony channels |
| `gemma4` | Gemma 4 (all sizes) | `<\|tool_call>call:NAME{…}<tool_call\|>` |
| `qwen3` | Qwen 3.6 and `qwen3*` | gallium's JSON protocol |
| `lfm2` | LFM2.5 | gallium's JSON protocol |
| `minimax-m2` | MiniMax-M2.7 | `<minimax:tool_call><invoke name="…">` |
| `deepseek-v4` | DeepSeek-V4-Flash | `<｜DSML｜tool_calls>` |
| `generic` | anything else | all of the above, tried in turn |

**You normally set nothing.** The profile is detected from what the model file
reports — `general.architecture`, which is also how llama.cpp picks its own
loader, plus the embedded chat template — and an unrecognized model gets
`generic`, which tries every format and is how gallium behaved before profiles
existed. The startup log says which profile was chosen and whether it was
detected or configured.

Set `profile` (or `GALLIUM_PROFILE`) for the two cases detection cannot serve: a
repackaged or mislabeled GGUF, and forcing `generic` to compare behavior. A name
no profile answers to is a startup error listing the real ones, never a silent
fallback. Profiles apply to the llama.cpp backend; the candle engine still uses
its own protocol adapters.

Why per-family rather than one lenient parser for everything: a parser that
accepts any format also accepts *another* family's format, and reading a known
model's output by the wrong family's rules is where several real bugs came from.
See [ADR 0003](docs/adr/0003-model-profiles.md).

The `scripted` engine (`INFERENCE_ENGINE=scripted`, `configs/scripted.toml`)
answers from a fixed list of steps instead of a model, for testing the
app-server wire format and ReAct/tool plumbing with no weights, no API key, and
no network — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#testing-without-a-model).

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
| `LLM_TEMPERATURE` / `LLM_TOP_P` / `LLM_TOP_K` / `MAX_TOKENS` / `REASONING_EFFORT` | sampling + budget — `LLM_TOP_P`/`LLM_TOP_K` are local-backend only |
| `CONTEXT_WINDOW` | `llm.contextWindow` — compaction trigger (default 8192 local, 128000 cloud) |
| `INFERENCE_ENGINE` | `llm.inferenceEngine` |
| `MAX_REACT_ITERATIONS` | `agent.maxTurns` |
| `WORKING_DIR` | tool root (default: cwd) |
| `MCP_SERVERS` | extra stdio servers, `"cmd arg1,cmd2 arg1"` |
| `GALLIUM_DEVICE` | native candle backend device: `auto` (default), `cpu`, `metal`, `cuda`, or `metal:1` |
| `GALLIUM_DTYPE` | native candle backend dtype |
| `GALLIUM_TOKENIZER_REPO` | `llm.tokenizerPath` — the native candle backend's tokenizer source |
| `GALLIUM_PROFILE` | `llm.profile` — which model profile reads the model's output |
| `GALLIUM_GPU_LAYERS` | llama.cpp GPU offload (`0` = CPU) |
| `GALLIUM_MAX_CTX` | `llm.maxCtx` — largest llama.cpp context, in tokens (default: the model's trained window, and never above it). Also the window compaction measures against |
| `GALLIUM_KV_CACHE_SLOTS` | llama.cpp retained KV caches (default `1`, `0` disables prompt reuse) — each slot is a whole KV cache |
| `GALLIUM_BASH_ALLOW` | extra allowed `Bash` commands |
| `GALLIUM_TRACE` | `1` turns per-turn traces on (default dir), `0` turns them off whatever the config says |
| `GALLIUM_TRACE_DIR` | `agent.trace.dir` — where traces are written (setting it turns them on) |
| `GALLIUM_GH_ORG` / `GALLIUM_GH_PROJECT` / `GALLIUM_GH_REPO` | GitHub Projects tools (absent = tools not registered) |

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
| `~/.config/gallium/skills/`, `.claude/skills/`, `.agents/skills/` | Skills, plus any `agent.skillPaths` from the config. Both layouts load: `*.md` directly in the directory, and `<name>/SKILL.md` one level down. |

Skills are keyed by name and the list above is in increasing precedence, so an
`.agents/skills` entry overrides a `.claude/skills` one of the same name, and
both override the global directory. There is no gallium-specific tier above
`.agents/skills` — a skill written for this project belongs there, the same
convention `AGENTS.md` already follows over `CLAUDE.md`. In app-server mode a
client's own `skillPaths` load last of all, so they override every one of
these.

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

See [docs/building-blocks.md](docs/building-blocks.md) for the composable
building blocks `gallium-core` provides (attention, FFN, RoPE, caching) and
what each one does.

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
BACKENDS="gemma4,gpt-oss-20b" bash testsuite/matrix_runner.sh
TESTS="coding,refactoring"  bash testsuite/matrix_runner.sh
```

Backends are `configs/*.toml` (which ones are listed in `testsuite/backends.txt`);
testcases are `testsuite/testcases/*/` with a `prompt.txt` and a `check.sh`. See
[testsuite/README.md](testsuite/README.md).

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

See [docs/models/adding-models.md](docs/models/adding-models.md). The short version:

1. Define a config struct (deserializes from HuggingFace `config.json`)
2. Wire together gallium-core blocks in a `load()` function
3. Implement `CausalLM` (forward + reset)
4. Add it to `gallium-models/src/lib.rs`, and to `Arch` in `gallium-agent/src/llm_candle.rs`

## Documentation

- [Architecture Decision Records](docs/adr/) — what gallium is responsible for, and what it declines to be
- [Development Notes](docs/DEVELOPMENT.md) — building on Windows, toolchain gotchas
- [Session Handoff](docs/DEVELOPMENT.md#session-handoff) — how unfinished work (and its verification) moves between sessions and machines
- [Architecture Overview](docs/architecture.md)
- [Candle Backend](docs/CANDLE_BACKEND.md) — `GALLIUM_DEVICE`, GPU throughput (Metal + CUDA), where decode time goes
- [Optimization](docs/OPTIMIZATION.md) — what a turn's ttft and tok/s mean, which llama.cpp knobs are reachable, and the plan for searching them
- [Multimodal Support](docs/MULTIMODAL.md) — images and audio: what works where, projectors, what media costs
- [Verification Status](docs/VERIFICATION_STATUS.md) — which prompt-affecting changes have been confirmed against a running model, on what hardware, and what came out
- [The Remote App-Server](docs/REMOTE-APP-SERVER.md) — running the model on another machine: the transport, the connection model, and which machine runs what
- [Building Blocks Reference](docs/building-blocks.md)
- [LLaMA CPU MoE](docs/LLAMA_CPU_MOE.md) — the `cpuMoe` knob: experts on CPU RAM, attention on the GPU
- [Test Suite](testsuite/README.md)

Model notes — [`docs/models/`](docs/models/):

- [Architectures](docs/models/architectures.md) — attention / FFN / RoPE per target family
- [Adding a Model](docs/models/adding-models.md)
- [GGUF Tensor Names](docs/models/gguf-tensor-names.md)
- [GPT-OSS](docs/models/gpt-oss.md), [Qwen 3.5](docs/models/qwen35.md), [Gemma 4](docs/models/gemma4.md) — per-model implementation notes

## License

MIT
