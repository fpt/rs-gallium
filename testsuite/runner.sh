#!/usr/bin/env bash
# gallium-agent integration test runner.
#
# Usage:
#   AGENT=./target/release/gallium-agent ./testsuite/runner.sh <testcase> [model]
#
# Arguments:
#   testcase  - directory name under testsuite/testcases/
#   model     - model preset name under testsuite/models/ (default: gpt-oss-gguf)
#
# Environment:
#   AGENT     - path to gallium-agent binary (required)
#   MAX_TOKENS - max tokens per turn (default: 512)
#
# Examples:
#   AGENT=./target/release/gallium-agent ./testsuite/runner.sh memory_state
#   AGENT=./target/release/gallium-agent ./testsuite/runner.sh coding openai

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASE="${1:-}"
MODEL="${2:-gpt-oss-gguf}"
MAX_TOKENS="${MAX_TOKENS:-512}"

# ── Validate inputs ──────────────────────────────────────────────────────────

if [ -z "$TESTCASE" ]; then
    echo "Usage: AGENT=./target/release/gallium-agent $0 <testcase> [model]"
    echo ""
    echo "Available testcases:"
    ls "$SCRIPT_DIR/testcases/"
    echo ""
    echo "Available models:"
    ls "$SCRIPT_DIR/models/" | sed 's/\.sh$//'
    exit 1
fi

if [ -z "${AGENT:-}" ]; then
    echo "ERROR: AGENT env var must point to the gallium-agent binary."
    echo "  Example: AGENT=./target/release/gallium-agent $0 $TESTCASE"
    exit 1
fi

if [ ! -x "$AGENT" ]; then
    echo "ERROR: AGENT binary not found or not executable: $AGENT"
    exit 1
fi

TESTCASE_DIR="$SCRIPT_DIR/testcases/$TESTCASE"
if [ ! -d "$TESTCASE_DIR" ]; then
    echo "ERROR: Testcase not found: $TESTCASE_DIR"
    exit 1
fi

MODEL_FILE="$SCRIPT_DIR/models/${MODEL}.sh"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model config not found: $MODEL_FILE"
    echo "Available models:"
    ls "$SCRIPT_DIR/models/" | sed 's/\.sh$//'
    exit 1
fi

# ── Load model flags ─────────────────────────────────────────────────────────

AGENT_FLAGS=""
# shellcheck source=/dev/null
source "$MODEL_FILE"

# ── Create isolated working directory ─────────────────────────────────────────

WORK_DIR="$(mktemp -d)"
trap 'if [ $? -ne 0 ]; then echo "Work dir preserved for debugging: $WORK_DIR"; else rm -rf "$WORK_DIR"; fi' EXIT

# Copy testcase files into the work dir.
cp -r "$TESTCASE_DIR"/. "$WORK_DIR/"
# Copy extract_response.sh so check.sh can use it.
cp "$SCRIPT_DIR/extract_response.sh" "$WORK_DIR/"
chmod +x "$WORK_DIR/extract_response.sh" "$WORK_DIR/check.sh"

OUTPUT_FILE="$WORK_DIR/output.txt"
ERROR_FILE="$WORK_DIR/error.txt"
PROMPT_FILE="$WORK_DIR/prompt.txt"

# ── Run the agent ─────────────────────────────────────────────────────────────

echo "Running: $TESTCASE  (model: $MODEL)"

# shellcheck disable=SC2086
if ! "$AGENT" \
    $AGENT_FLAGS \
    --max-tokens "$MAX_TOKENS" \
    --working-dir "$WORK_DIR" \
    -f "$PROMPT_FILE" \
    >"$OUTPUT_FILE" 2>"$ERROR_FILE"; then
    echo "FAIL: agent exited with non-zero status"
    echo "--- stderr ---"
    cat "$ERROR_FILE"
    exit 1
fi

# ── Run check.sh ──────────────────────────────────────────────────────────────

if (cd "$WORK_DIR" && ./check.sh "$OUTPUT_FILE" "$ERROR_FILE"); then
    echo "PASS: $TESTCASE"
    exit 0
else
    echo "FAIL: $TESTCASE"
    echo "--- output ---"
    cat "$OUTPUT_FILE"
    echo "--- stderr ---"
    cat "$ERROR_FILE"
    exit 1
fi
