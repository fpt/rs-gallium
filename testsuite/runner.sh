#!/usr/bin/env bash
# gallium integration test runner.
#
# Usage:
#   AGENT=./target/release/gallium ./testsuite/runner.sh <testcase> [model]
#
# Arguments:
#   testcase  - directory name under testsuite/testcases/
#   model     - model preset name under testsuite/models/ (default: gpt-oss-gguf)
#
# Environment:
#   AGENT      - path to gallium binary (required)
#   MAX_TOKENS - max tokens per turn (default: 512)
#   PROFILE    - set to 1 to CPU-profile the agent via gperftools (libprofiler.so).
#                Does not require kernel perf; works in Docker on WSL2.
#                Output: /logs/callgraph-<testcase>-<model>.svg
#
# Examples:
#   AGENT=./target/release/gallium ./testsuite/runner.sh memory_state
#   AGENT=./target/release/gallium ./testsuite/runner.sh coding openai
#   docker run --rm -e PROFILE=1 \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     -v "/tmp:/logs" \
#     gallium-integration coding gpt-oss-gguf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASE="${1:-}"
MODEL="${2:-gpt-oss-gguf}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PROFILE="${PROFILE:-0}"

# ── Validate inputs ──────────────────────────────────────────────────────────

if [ -z "$TESTCASE" ]; then
    echo "Usage: AGENT=./target/release/gallium $0 <testcase> [model]"
    echo ""
    echo "Available testcases:"
    ls "$SCRIPT_DIR/testcases/"
    echo ""
    echo "Available models:"
    ls "$SCRIPT_DIR/models/" | sed 's/\.sh$//'
    exit 1
fi

if [ -z "${AGENT:-}" ]; then
    echo "ERROR: AGENT env var must point to the gallium binary."
    echo "  Example: AGENT=./target/release/gallium $0 $TESTCASE"
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

if [ "$PROFILE" = "1" ]; then
    # CPU-profile via gperftools libprofiler.so (SIGPROF-based; no kernel perf needed).
    # Locate the shared library from ldconfig so the path is not hardcoded.
    PROFILER_LIB="$(ldconfig -p | awk '/libprofiler\.so/{print $NF; exit}')"
    if [ -z "$PROFILER_LIB" ]; then
        echo "WARNING: libprofiler.so not found — falling back to unprofiled run"
        PROFILE=0
    else
        # Write directly to /logs so it survives WORK_DIR cleanup.
        CPU_PROF="/logs/cpu-${TESTCASE}-${MODEL}.prof"
        # shellcheck disable=SC2086
        if ! CPUPROFILE="$CPU_PROF" LD_PRELOAD="$PROFILER_LIB" \
            "$AGENT" \
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

        # Generate call-graph SVG and save to the host-mounted logs volume.
        SVG="/logs/callgraph-${TESTCASE}-${MODEL}.svg"
        PPROF_ERR="$WORK_DIR/pprof_err.txt"
        if google-pprof --svg "$AGENT" "$CPU_PROF" > "$SVG" 2>"$PPROF_ERR"; then
            echo "Call graph: $SVG"
        else
            echo "WARNING: google-pprof failed (see below) — raw profile at $CPU_PROF"
            cat "$PPROF_ERR"
        fi
    fi
fi

if [ "$PROFILE" != "1" ]; then
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
