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
├── backends/            # one TOML config per model
│   ├── gemma4.toml       # local Gemma 4 E4B
│   ├── gemma4-26b.toml   # local Gemma 4 26B-A4B (MoE)
│   ├── gpt-oss.toml      # local GPT-OSS 20B (harmony)
│   ├── lfm2.toml         # local LiquidAI LFM2.5-8B-A1B (MoE)
│   └── gpt-5.6-luna.toml # cloud OpenAI (needs OPENAI_API_KEY)
├── fixtures/
│   └── make_fixtures.py # regenerates the multimodal fixtures (stdlib only)
├── testcases/
│   ├── arithmetic/       # 17 × 23 = 391
│   ├── capital/          # capital of France = Paris
│   ├── file_read/        # use the `read` tool on codeword.txt
│   ├── memory_state/     # 2-turn: recall conversational context
│   ├── needle_in_haystack/ # long-context recall of a buried string
│   ├── coding/           # write hello.go (Go), must compile and print "Hello"
│   ├── refactoring/      # refactor counter.go to a struct; must still build
│   ├── multimodal_image/ # read "42" out of number.png — needs a vision model
│   └── multimodal_audio/ # name a tone's pitch — EXPECTED TO FAIL, see below
└── results/             # timestamped matrix logs (gitignored)
```

## Multimodal testcases

`multimodal_image` and `multimodal_audio` test what the model can *perceive*, so
they need input the other testcases do not have: a file attached to the turn.

**Attaching a file.** A prompt line may carry `@image:<path>` markers, which the
REPL lifts out of the text and loads as attachments. Relative paths resolve
against the agent's working directory — the testcase's temp dir — and a path
with spaces goes in double quotes:

```
@image:number.png What is in this picture?
@image:"my shot.png" @image:other.jpg Compare these.
```

Recognized only at a whitespace boundary, so `user@image:host` stays text. A
marker whose file will not load **fails the turn** rather than being dropped: an
attachment that silently vanished is indistinguishable from a model that cannot
see, which is the one thing these tests exist to tell apart. `png`, `jpeg`,
`gif`, and `webp` are carried.

**What passes.** Only providers that accept images — the OpenAI backend today.
Both local backends refuse the turn outright (`the llama.cpp backend cannot see
images as built…`) instead of dropping the pixels and letting the model answer
confidently about a picture it never got. `check.sh` reports which of the two
happened.

"As built" is doing real work in that message. llama.cpp *does* have multimodal
support — [`mtmd`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
driven by an `--mmproj` projector file, covering image **and** audio — and the
`llama-cpp-2` crate gallium already depends on wraps it behind an `mtmd` feature
we do not enable. So these two testcases are not asking for something the local
engine cannot do; they are measuring a path nobody has wired up yet, which is
exactly what a capability test is for.

### The `gemma4-12b` backend can do both, and its GGUF says so

Gemma 4 12B Unified is
[encoder-free](https://huggingface.co/google/gemma-4-12B): it projects raw image
patches and audio waveforms straight into the embedding space through linear
layers, with no SigLIP/USM stack in front. The repo behind
`backends/gemma4-12b.toml` ships those projections as a separate
`mmproj-{BF16,F16,F32}.gguf`, which is llama.cpp's packaging convention rather
than a contradiction of "encoder-free" — the header proves the point:

```
clip.has_vision_encoder    = True     clip.has_audio_encoder    = True
clip.vision.projector_type = gemma4uv clip.audio.projector_type = gemma4ua
clip.vision.block_count    = 0        clip.audio.block_count    = 0
clip.audio.num_mel_bins    = 128      general.size_label        = 52M
```

`block_count = 0` on both is the encoder-free design in the file format: 11
tensors, 52 MB, no transformer blocks — just `v.patch_embd` / `v.position_embd`
for image patches and `mm.input_projection` for audio.

The main GGUF we download carries **none** of this — 667 tensors, all attention,
FFN and norms — which is why `gemma4-12b` refuses exactly like every other local
backend. It is a missing download and a missing feature flag, not a missing
capability. So `multimodal_audio` is not permanently red: it has a concrete
route to green on a model already in the cache.

**Both tests forbid tool use, and enforce it.** The fixture sits in the agent's
cwd, so a capable agent could decode `number.png` or measure `tone.wav` with
Bash and answer correctly without perceiving anything. That is a real capability
but not the one under test, so `check.sh` fails the test if any tool ran.

**`multimodal_audio` is expected to fail everywhere.** gallium carries no audio
at all: there is no `AudioContent`, no `ToolContent::Audio`, and no provider
wired to take one, so `@audio:` is not a marker gallium recognizes and the line
reaches the model as literal text. The testcase is the record of that gap — a
failing test that names the missing feature, rather than a comment in a file
nobody runs. It should start passing, unmodified, when audio input lands, and
`mtmd` is the nearest route to that (`MtmdBitmap::from_audio_data`,
`MtmdContext::support_audio`).

The fixtures are committed, so running the tests needs no Python. Regenerate
them with `uv run python testsuite/fixtures/make_fixtures.py` — the script is
standard-library-only and documents why each fixture is what it is. Note in
particular that the tone is **613 Hz, not 440**: an earlier 440 Hz version passed
against a model that never received a single sample, because A4 is the answer
everyone guesses from the word "tone".

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
BACKENDS="gemma4,gpt-oss"  bash testsuite/matrix_runner.sh
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
