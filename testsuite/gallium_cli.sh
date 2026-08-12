#!/usr/bin/env bash
#
# Adapter that lets the testsuite drive the `gallium` binary (the same one a
# client like klein-cli spawns over the app-server protocol). The binary now parses a TOML
# `--config` natively, so this shim only locates the binary, checks it is built,
# and forwards `--config <backend.toml>`; prompts arrive on stdin.
#
# Precedence is env > config > default inside the binary, so the matrix runner's
# ambient overrides still apply: INFERENCE_ENGINE (per-engine runs) flows through
# when a backend omits it, and OPENAI_API_KEY is inherited when a cloud backend's
# apiKey is empty.
#
# Testsuite backends resolve straight to configs/*.toml (see backends.txt) — the
# same file a person runs the agent with, not a separate testsuite-only copy.
# But that file's `[agent]` table (systemPromptPath, skillPaths, mcpServers) is
# tuned for real use, not for a capability test: gemma4-system-prompt.md frames
# the model as "a coding agent working inside a local checkout" and told Gemma 4
# to refuse the `capital` testcase outright ("I can only interact with the
# codebase provided to me") — a false negative about the test, not about the
# model. So this shim strips everything but `[llm]` before handing the config to
# gallium: the tuned modelPath/cpuMoe/gpuLayers/mmprojPath survive (one file, no
# more testsuite-only duplicate to drift), but the turn a test sends is a plain
# model call, not a scoped coding-agent persona. `maxTurns` goes with the
# section — react.rs's own DEFAULT_MAX_ITERATIONS (30) takes over, more than
# enough for these testcases' one or two tool calls.
#
# Use it as the testsuite's CLI (it is also the runner's default):
#   CLI="$PWD/testsuite/gallium_cli.sh" BACKENDS=gemma4 bash testsuite/matrix_runner.sh
#
# Override the binary with GALLIUM_BIN (default: target/release/gallium).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${GALLIUM_BIN:-$script_dir/../target/release/gallium}"

# The runner invokes us as: gallium_cli.sh --config <backend.toml>  (prompt on stdin)
config=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config) config="${2:-}"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "$config" ] || [ ! -f "$config" ]; then
    echo "gallium_cli.sh: --config <file> required (got '$config')" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "gallium_cli.sh: Rust binary not found: $BIN (build: cargo build --release -p gallium-agent)" >&2
    exit 2
fi

# Keep only the `[llm]` table: from its header line up to (not including) the
# next line that opens a new table ([agent], [[mcpServers]], ...). Any header
# comment before `[llm]` is dropped too — doc prose, not config.
filtered="$(mktemp)"
trap 'rm -f "$filtered"' EXIT
awk '
    /^\[llm\]/ { in_llm = 1; print; next }
    /^\[/      { in_llm = 0 }
    in_llm     { print }
' "$config" > "$filtered"

# Not `exec`: the EXIT trap above must still fire to clean up $filtered, and
# that trap belongs to this process — exec would replace it before it could run.
"$BIN" --config "$filtered"
