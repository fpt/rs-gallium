#!/usr/bin/env bash
# Run all testcases × all models and print a summary table.
#
# Usage:
#   AGENT=./target/release/gallium ./testsuite/matrix_runner.sh
#
# Filters (optional env vars):
#   TESTS="memory_state,coding"
#   MODELS="gpt-oss-gguf,openai"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/runner.sh"

# ── Collect testcases ─────────────────────────────────────────────────────────

if [ -n "${TESTS:-}" ]; then
    IFS=',' read -r -a TESTCASES <<< "$TESTS"
else
    mapfile -t TESTCASES < <(ls "$SCRIPT_DIR/testcases/")
fi

# ── Collect models ────────────────────────────────────────────────────────────

if [ -n "${MODELS:-}" ]; then
    IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
else
    mapfile -t MODEL_LIST < <(ls "$SCRIPT_DIR/models/" | sed 's/\.sh$//')
fi

# Filter models that require missing API keys.
ACTIVE_MODELS=()
for m in "${MODEL_LIST[@]}"; do
    case "$m" in
        openai*)
            [ -n "${OPENAI_API_KEY:-}" ] && ACTIVE_MODELS+=("$m") || echo "Skipping $m (no OPENAI_API_KEY)";;
        *)
            ACTIVE_MODELS+=("$m");;
    esac
done

if [ ${#ACTIVE_MODELS[@]} -eq 0 ]; then
    echo "No active models. Set OPENAI_API_KEY or use a local model."
    exit 1
fi

# ── Run matrix ────────────────────────────────────────────────────────────────

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/test_results_$(date +%Y%m%d_%H%M%S).txt"

declare -A PASS_MAP

for model in "${ACTIVE_MODELS[@]}"; do
    for tc in "${TESTCASES[@]}"; do
        key="${model}::${tc}"
        echo "── $tc / $model ───────────────────────────────────"
        if AGENT="${AGENT:-}" bash "$RUNNER" "$tc" "$model" 2>&1; then
            PASS_MAP["$key"]="✅"
        else
            PASS_MAP["$key"]="❌"
        fi
        echo ""
    done
done

# ── Summary table ─────────────────────────────────────────────────────────────

{
    printf "\n%-20s" "testcase \\ model"
    for m in "${ACTIVE_MODELS[@]}"; do printf " %-16s" "$m"; done
    echo ""
    printf '%0.s─' {1..80}; echo ""

    total=0; passed=0
    for tc in "${TESTCASES[@]}"; do
        printf "%-20s" "$tc"
        for model in "${ACTIVE_MODELS[@]}"; do
            key="${model}::${tc}"
            result="${PASS_MAP[$key]:-?}"
            printf " %-16s" "$result"
            ((total++))
            [[ "$result" == "✅" ]] && ((passed++))
        done
        echo ""
    done

    printf '%0.s─' {1..80}; echo ""
    echo "Total: $passed/$total passed"
} | tee "$RESULTS_FILE"

echo ""
echo "Results saved to $RESULTS_FILE"
