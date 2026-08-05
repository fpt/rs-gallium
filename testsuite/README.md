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
│   ├── vision_image/     # read "42" out of number.png — needs a vision model
│   └── vision_audio/     # name a tone's pitch — EXPECTED TO FAIL, see below
└── results/             # timestamped matrix logs (gitignored)
```

## Multimodal testcases

`vision_image` and `vision_audio` test what the model can *perceive*, so they
need input the other testcases do not have: a file attached to the turn.

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
images…`) instead of dropping the pixels and letting the model answer confidently
about a picture it never got. `check.sh` reports which of the two happened.

**Both tests forbid tool use, and enforce it.** The fixture sits in the agent's
cwd, so a capable agent could decode `number.png` or measure `tone.wav` with
Bash and answer correctly without perceiving anything. That is a real capability
but not the one under test, so `check.sh` fails the test if any tool ran.

**`vision_audio` is expected to fail everywhere.** gallium carries no audio at
all: there is no `AudioContent`, no `ToolContent::Audio`, and no provider that
would take one, so `@audio:` is not a marker gallium recognizes and the line
reaches the model as literal text. The testcase is the record of that gap — a
failing test that names the missing feature, rather than a comment in a file
nobody runs. It should start passing, unmodified, when audio input lands.

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
