# gallium Test Suite

Capability tests for the `gallium` binary across multiple LLM backends, modeled
after `../klein-cli`'s testsuite (`runner.sh` + `matrix_runner.sh` + per-testcase
`prompt.txt`/`check.sh`).

The `gallium` binary is env-driven and reads prompts from **stdin** (a REPL, one
line per turn). The `gallium_cli.sh` adapter maps a YAML backend config to the
env vars the binary understands (`MODEL_PATH`, `LLM_BASE_URL`, `LLM_MODEL`,
`OPENAI_API_KEY`, `LLM_TEMPERATURE`, `MAX_TOKENS`, `REASONING_EFFORT`,
`INFERENCE_ENGINE`, `MAX_REACT_ITERATIONS`, `MCP_SERVERS`) and feeds prompts on
stdin. Tests validate the assistant's **text responses** (`Assistant:` lines);
file-writing testcases additionally inspect files the agent produced in its cwd.

## Layout

```
testsuite/
├── runner.sh            # run one testcase × one backend
├── matrix_runner.sh     # run all (filterable) → PASS/FAIL matrix
├── extract_response.sh  # pull assistant text (optionally per-turn) from output
├── gallium_cli.sh       # adapter: YAML --config → env vars → `gallium` (stdin)
├── backends/            # one YAML config per model
│   ├── gemma4.yaml       # local Gemma 4 E4B
│   ├── gemma4-26b.yaml   # local Gemma 4 26B-A4B (MoE)
│   ├── gpt-oss.yaml      # local GPT-OSS 20B (harmony)
│   ├── lfm2.yaml         # local LiquidAI LFM2.5-8B-A1B (MoE)
│   └── gpt-5.6-luna.yaml # cloud OpenAI (needs OPENAI_API_KEY)
├── testcases/
│   ├── arithmetic/       # 17 × 23 = 391
│   ├── capital/          # capital of France = Paris
│   ├── file_read/        # use the `read` tool on codeword.txt
│   ├── instruction/      # output exactly one given word
│   ├── memory/           # 2-turn: recall a fact from turn 1
│   ├── memory_state/     # 2-turn: recall conversational context
│   ├── needle_in_haystack/ # long-context recall of a buried string
│   ├── sw_boundary/      # recall a fact before the sliding-window boundary
│   ├── coding/           # write hello.go, must compile and print "Hello"
│   └── refactoring/      # refactor counter.go to a struct; must still build
└── results/             # timestamped matrix logs (gitignored)
```

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
TESTS="memory,file_read"   bash testsuite/matrix_runner.sh

# Pick the local inference engine (default llamacpp; the native candle backend
# needs a tokenizer.json — see KESSEL_GALLIUM_TOKENIZER_REPO in the backend YAMLs)
INFERENCE_ENGINE=gallium   bash testsuite/matrix_runner.sh
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
2. `prompt.txt` — one user turn per non-empty line (`#` lines are comments)
3. `check.sh` (executable) — args `$1`=output file, `$2`=error file; cwd is the
   temp dir, with `./extract_response.sh` available. Exit 0 = pass.
4. Add any fixture files the test needs (copied into the temp workdir).
