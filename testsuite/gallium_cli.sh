#!/usr/bin/env bash
#
# Adapter that lets the testsuite drive the `gallium` binary (the same one a
# client like kessel/klein spawns over ACP). The binary now parses a TOML
# `--config` natively, so this shim only locates the binary, checks it is built,
# and forwards `--config <backend.toml>`; prompts arrive on stdin.
#
# Precedence is env > config > default inside the binary, so the matrix runner's
# ambient overrides still apply: INFERENCE_ENGINE (per-engine runs) flows through
# when a backend omits it, and OPENAI_API_KEY is inherited when a cloud backend's
# apiKey is empty.
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

exec "$BIN" --config "$config"
