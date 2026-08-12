# gallium Test Suite

Capability tests for the `gallium` binary across multiple LLM backends, modeled
after `../klein-cli`'s testsuite (`runner.sh` + `matrix_runner.sh` + per-testcase
`prompt.txt`/`check.sh`).

The `gallium` binary reads a TOML `--config` (env vars still override individual
fields) and takes prompts from **stdin** (a REPL, one line per turn). The
`gallium_cli.sh` adapter just locates the binary and forwards `--config
<backend.toml>`, feeding prompts on stdin. Environment overrides the binary still
honors on top of the config: `MODEL_PATH`, `LLM_BASE_URL`, `LLM_MODEL`,
`OPENAI_API_KEY`, `LLM_TEMPERATURE`, `MAX_TOKENS`, `REASONING_EFFORT`,
`INFERENCE_ENGINE`, `MAX_REACT_ITERATIONS`, `MCP_SERVERS`. Tests validate the
assistant's **text responses** (`Assistant:` lines);
file-writing testcases additionally inspect files the agent produced in its cwd.

## Layout

```
testsuite/
├── runner.sh            # run one testcase × one backend
├── matrix_runner.sh     # run all (filterable) → PASS/FAIL matrix
├── extract_response.sh  # pull assistant text (optionally per-turn) from output
├── gallium_cli.sh       # adapter: forwards TOML --config to `gallium` (stdin)
├── backends.txt         # which ../configs/*.toml are testsuite backends, and why
├── fixtures/
│   └── make_fixtures.py # regenerates the multimodal fixtures
├── testcases/
│   ├── arithmetic/       # 17 × 23 = 391
│   ├── capital/          # capital of France = Paris
│   ├── file_read/        # use the `read` tool on codeword.txt
│   ├── memory_state/     # 2-turn: recall conversational context
│   ├── needle_in_haystack/ # long-context recall of a buried string
│   ├── coding/           # write hello.go (Go), must compile and print "Hello"
│   ├── refactoring/      # refactor counter.go to a struct; must still build
│   ├── multimodal_image/ # read "42" out of number.png — needs a projector
│   └── multimodal_audio/ # transcribe speech.wav — needs an audio projector
└── results/             # timestamped matrix logs (gitignored)
```

A `<backend>` name (e.g. `gemma4`, `qwen3.6-cuda-12gb`) resolves to
`../configs/<backend>.toml` — the same config a person actually runs the
agent with, not a separate copy, so `modelPath`/`cpuMoe`/`gpuLayers`/
`mmprojPath` tuning lives in exactly one file. `backends.txt` lists which of
`configs/*.toml` are meant to be exercised this way; not every file there is
(see that file for what's excluded and why).

`gallium_cli.sh` strips everything but that file's `[llm]` table before
handing it to the binary — `[agent]`'s `systemPromptPath`/`skillPaths`/
`mcpServers` are tuned for real use, not a capability test, and letting them
through changes what's under test: `gemma4-system-prompt.md` frames the
model as a coding agent and made Gemma 4 refuse the `capital` testcase
outright, a false negative about the test rather than the model. So a
testsuite turn is a plain model call with the real tuning knobs, not a
scoped persona.

## Multimodal testcases

`multimodal_image` and `multimodal_audio` test what the model can *perceive*, so
they need input the other testcases do not have: a file attached to the turn.

For the feature itself — configuring a projector, what media costs in tokens,
what each refusal means — see [docs/MULTIMODAL.md](../docs/MULTIMODAL.md). What
follows is what these two testcases need.

**Attaching a file.** A prompt line may carry `@image:<path>` and `@audio:<path>`
markers, which the REPL lifts out of the text and loads as attachments. Relative
paths resolve against the agent's working directory — the testcase's temp dir —
and a path with spaces goes in double quotes:

```
@image:number.png What is in this picture?
@audio:speech.wav Transcribe this audio exactly.
@image:"my shot.png" @image:other.jpg Compare these.
```

Recognized only at a whitespace boundary, so `user@image:host` stays text. A
marker whose file will not load **fails the turn** rather than being dropped: an
attachment that silently vanished is indistinguishable from a model that cannot
perceive, which is the one thing these tests exist to tell apart. `png`, `jpeg`,
`gif`, `webp` for images; `wav`, `mp3`, `flac` for audio — what llama.cpp's
bundled stb_image and miniaudio decode.

**What passes, and why each failure differs:**

| Backend | `multimodal_image` | `multimodal_audio` |
|---|---|---|
| `gemma4` (E4B) | PASS | PASS |
| `gemma4-12b` | PASS | FAIL — heard it, wrote *"zuki"* |
| `openai` | PASS | FAIL — refused, no audio path |
| `lfm2` | FAIL — no projector | FAIL — no projector |
| `lfm2-candle` | FAIL — candle has no mtmd | FAIL — candle has no mtmd |

Local multimodal runs through
[`mtmd`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
llama.cpp's multimodal front end, and needs a **projector** — `mmprojPath` in the
backend TOML, published beside the model in its GGUF repo. A backend without one
refuses the turn and names what is missing; it never answers about media it did
not receive. `check.sh` reports which cause applied.

### The projector is the whole story

Every Gemma 4 (E2B/E4B, 12B, 26B) and Qwen 3.6 handles text, image and audio;
`gpt-oss-20b` and `lfm2` are text-only and have no projector to fetch. The two
Gemma generations differ in a way the results make visible:

| | `gemma4` (E4B) | `gemma4-12b` |
|---|---|---|
| Design | [dedicated encoders](https://huggingface.co/google/gemma-4-E4B) | [encoder-free](https://huggingface.co/google/gemma-4-12B) |
| Projector | 1411 tensors, 478M, `vision.block_count=16`, `audio.block_count=12` | 11 tensors, 52M, `block_count=0` on both |
| Download | 946 MB | 167 MB |
| Types | `gemma4v` / `gemma4a` | `gemma4uv` / `gemma4ua` |
| Transcription | exact | *"the secret code word is zuki"* |

The small model's projector is the **larger** download, and it is the one that
transcribes correctly: E4B ships two transformer stacks, 12B ships linear
layers. Vision is fine on both. `multimodal_audio` treats heard-but-garbled as
its own outcome rather than lumping it in with deaf — 12B plainly received the
audio, it just spells badly.

The main model GGUFs carry none of this — `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
is 667 tensors of attention, FFN and norms — so a backend TOML without
`mmprojPath` is text-only however capable its model is.

**Both tests forbid tool use, and enforce it.** The fixture sits in the agent's
cwd, so a capable agent could decode `number.png` or run something over
`speech.wav` with Bash and answer correctly without perceiving anything. That is
a real capability but not the one under test, so `check.sh` fails the test if any
tool ran.

### Two fixture lessons worth keeping

Both testcases were wrong on the first try, in the same way: the task was
guessable, or impossible, rather than a measure of perception.

- **The image asks for a two-digit number.** A colour is a test a blind model
  passes one time in six.
- **The audio asks for a transcription, not a pitch.** The first fixture was a
  613 Hz tone (itself chosen after a 440 Hz one that `gpt-5.6-luna` "passed"
  without receiving a single sample — A4 is what everyone guesses from the word
  "tone"). But *nothing* passes a pitch test: Gemma 4 12B demonstrably heard the
  613 Hz tone and still answered "440". Pitch estimation is not what an audio
  LLM does; transcription is.
- **The audio prompt's wording is load-bearing and was measured.** "Transcribe
  the speech in this audio clip" makes E4B answer "you have not provided one" —
  3 runs of 3, with the audio demonstrably encoded into the prompt. Dropping
  "in this audio clip" transcribes correctly, also 3 of 3.

The fixtures are committed, so running the tests needs no Python.
`uv run python testsuite/fixtures/make_fixtures.py` regenerates them; the image
is standard-library-only, and the speech needs macOS `say`/`afconvert` (any
16 kHz mono WAV of the phrase works elsewhere).

## Usage

```bash
# Build the binary first
make build          # or: cargo build --release -p gallium-agent

# List testcases / backends
bash testsuite/runner.sh

# One testcase × one backend
bash testsuite/runner.sh capital gemma4

# Full matrix (all testcases × all available backends)
bash testsuite/matrix_runner.sh

# Filter (comma-separated)
BACKENDS="gemma4,gpt-oss-20b"  bash testsuite/matrix_runner.sh
TESTS="memory_state,file_read"   bash testsuite/matrix_runner.sh

# Pick the local inference engine (default llamacpp; the native candle backend
# needs a tokenizer.json — see GALLIUM_TOKENIZER_REPO in the backend TOMLs)
INFERENCE_ENGINE=candle    bash testsuite/matrix_runner.sh
```

- `CLI` overrides the driver (defaults to `gallium_cli.sh`); `GALLIUM_BIN`
  overrides the binary path (defaults to `target/release/gallium`).
- `OPENAI_API_KEY` is read from the environment or a project-root `.env`
  (gitignored). Cloud backends are auto-skipped when no key is available.
- Each test runs in an isolated temp dir (its cwd), so the `read`/`glob`/`write`
  tools only see the testcase's own fixtures. Failed runs leave the temp dir for
  debugging; passed runs clean up.

## Adding a testcase

1. `mkdir testsuite/testcases/my_test`
2. `prompt.txt` — one user turn per non-empty line (`#` lines are comments);
   a line may carry `@image:<path>` attachments, see **Multimodal testcases**
3. `check.sh` (executable) — args `$1`=output file, `$2`=error file; cwd is the
   temp dir, with `./extract_response.sh` available. Exit 0 = pass.
4. Add any fixture files the test needs (copied into the temp workdir).
