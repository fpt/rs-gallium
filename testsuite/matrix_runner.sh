#!/bin/bash

# Run every testcase against every backend and print a PASS/FAIL matrix.
# Usage: CLI=path/to/gallium ./testsuite/matrix_runner.sh
#
# Optional comma-separated filters:
#   TESTS=capital,memory_state       run only matching testcases
#   BACKENDS=gemma4,openai           run only matching backends
#
# Backend names come from backends.txt (each maps to configs/<name>.toml —
# see that file for what's included/excluded and why).

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

script_dir="$(cd "$(dirname "$0")" && pwd)"
proj_root="$(cd "$script_dir/.." && pwd)"

if [ -z "$CLI" ]; then
    CLI="$script_dir/gallium_cli.sh"
fi
# Export so the per-case runner.sh children pick up this exact driver — including
# the default set just above, which otherwise wouldn't propagate.
export CLI
if [ ! -f "$CLI" ]; then
    echo "Error: CLI binary '$CLI' not found. Build with:"
    echo "  cargo build --release -p gallium-agent"
    exit 1
fi

[ -f "$proj_root/.env" ] && { set -a; . "$proj_root/.env"; set +a; }

timestamp="$(date +%Y%m%d_%H%M%S)"
results_dir="$script_dir/results"
result_file="$results_dir/test_results_${timestamp}.txt"
mkdir -p "$results_dir"; touch "$result_file"

log() { echo -e "$1" | tee -a "$result_file"; }

in_filter() {  # in_filter <name> <comma-list>; empty list matches all
    [ -z "$2" ] && return 0
    echo "$2" | tr ',' '\n' | grep -qx "$1"
}

# A backend is available unless it needs an API key that's missing.
backend_available() {
    local f="$proj_root/configs/$1.toml"
    # Local models (modelPath set) are always available; cloud needs a key.
    if grep -qE '^\s*modelPath\s*=' "$f"; then
        return 0
    fi
    if [ -n "$OPENAI_API_KEY" ] || grep -qE '^\s*apiKey\s*=\s*"\S' "$f"; then
        return 0
    fi
    log "${YELLOW}⚠️  Skipping $1: no OPENAI_API_KEY and no apiKey in config${NC}"
    return 1
}

# The multimodal_* testcases need a perception path the backend may not have.
# A backend that *architecturally* cannot carry that modality will *always*
# fail — not sometimes — so skip rather than fail: a permanent, known
# limitation isn't the same signal as a regression, and shouldn't make
# `make testsuite` exit non-zero forever for a backend passing everything it
# can. `testcase_modality` maps a testcase to image/audio/"" (none); the skip
# below consults `backend_can <backend> <modality>`.
testcase_modality() {
    case "$1" in
        multimodal_image) echo image ;;
        multimodal_audio) echo audio ;;
        *) echo "" ;;
    esac
}
# image: llama.cpp needs a projector (mmprojPath); the candle backend needs a
#   bare (non-.gguf) Gemma 4 safetensors modelPath — its own vision tower
#   (`gemma4_vision.rs`), no `mtmd`.
# audio: llama.cpp only. The candle Gemma 4 path has no audio tower at all
#   (`llm::reject_audio` refuses the turn), so `multimodal_audio` is skipped
#   there for the same reason a text-only backend skips `multimodal_image`.
#   (A vision-*only* llama.cpp projector still runs and fails audio — the
#   config can't say which projectors carry an audio encoder, and that's a
#   pre-existing wart, not this backend's.)
backend_can() {
    local f="$proj_root/configs/$1.toml" modality="$2"
    local is_candle_gemma4_st=false
    if grep -qE '^\s*inferenceEngine\s*=\s*"candle"' "$f" \
        && grep -qE '^\s*modelPath\s*=.*gemma-4' "$f" \
        && ! grep -qE '^\s*modelPath\s*=.*\.gguf' "$f"; then
        is_candle_gemma4_st=true
    fi
    case "$modality" in
        image) grep -qE '^\s*mmprojPath\s*=' "$f" || $is_candle_gemma4_st ;;
        audio) grep -qE '^\s*mmprojPath\s*=' "$f" ;;
        *) return 0 ;;
    esac
}

log "=== gallium Matrix Test Results ==="
log "Timestamp: $(date)"
log "Binary: $CLI"
log "TESTS filter:    ${TESTS:-(all)}"
log "BACKENDS filter: ${BACKENDS:-(all)}"
log ""

testcases=""
for d in $(find "$script_dir/testcases" -maxdepth 1 -mindepth 1 -type d | sort); do
    n="$(basename "$d")"
    in_filter "$n" "$TESTS" || continue
    [ -f "$d/prompt.txt" ] && [ -x "$d/check.sh" ] && testcases="$testcases $n"
done
testcases="${testcases# }"

backends=""
for n in $(grep -vE '^\s*#|^\s*$' "$script_dir/backends.txt" | sort); do
    in_filter "$n" "$BACKENDS" || continue
    backend_available "$n" || continue
    backends="$backends $n"
done
backends="${backends# }"

[ -z "$testcases" ] && { log "${YELLOW}No testcases matched.${NC}"; exit 0; }
[ -z "$backends" ]  && { log "${YELLOW}No backends matched/available.${NC}"; exit 0; }

log "${BLUE}📊 Matrix${NC}: [$(echo "$backends" | wc -w | tr -d ' ') backends × $(echo "$testcases" | wc -w | tr -d ' ') testcases]"
log "Testcases: $testcases"
log "Backends:  $backends"
log ""

entries=""; total=0; passed=0; failed=0; skipped=0
for b in $backends; do
    for t in $testcases; do
        log "${CYAN}▶ $t × $b${NC}"
        modality="$(testcase_modality "$t")"
        if [ -n "$modality" ] && ! backend_can "$b" "$modality"; then
            log "${YELLOW}  ⏭  SKIP — $b has no $modality path${NC}"
            skipped=$((skipped+1)); entries="$entries $b:$t:SKIP"
            continue
        fi
        total=$((total+1))
        if "$script_dir/runner.sh" "$t" "$b" > /tmp/va_matrix_out 2>&1; then
            log "${GREEN}  ✅ PASS${NC}"; passed=$((passed+1)); entries="$entries $b:$t:PASS"
        else
            log "${RED}  ❌ FAIL${NC}"; failed=$((failed+1)); entries="$entries $b:$t:FAIL"
            grep -a '^Assistant:' /tmp/va_matrix_out | sed 's/^/    /' >> "$result_file" 2>/dev/null || true
        fi
        rm -f /tmp/va_matrix_out
    done
done

# ── Matrix table ────────────────────────────────────────────────────────────────
log ""
log "${BLUE}📊 Result Matrix:${NC}"
col_w=4; for t in $testcases; do [ ${#t} -gt $col_w ] && col_w=${#t}; done; col_w=$((col_w+2))
lbl_w=8; for b in $backends; do [ ${#b} -gt $lbl_w ] && lbl_w=${#b}; done; lbl_w=$((lbl_w+2))

header="$(printf "%-${lbl_w}s" "")"
for t in $testcases; do header="$header$(printf "%-${col_w}s" "$t")"; done
log "$header"
log "$(printf '%*s' $((lbl_w + col_w * $(echo "$testcases" | wc -w))) '' | tr ' ' '-')"
for b in $backends; do
    row="$(printf "%-${lbl_w}s" "$b")"
    for t in $testcases; do
        r="?"
        for e in $entries; do
            [ "$e" = "$b:$t:PASS" ] && { r="PASS"; break; }
            [ "$e" = "$b:$t:FAIL" ] && { r="FAIL"; break; }
            [ "$e" = "$b:$t:SKIP" ] && { r="SKIP"; break; }
        done
        row="$row$(printf "%-${col_w}s" "$r")"
    done
    log "$row"
done

log ""
log "${BLUE}📊 Summary:${NC} Total: $total  Passed: $passed  Failed: $failed  Skipped: $skipped"
[ $total -gt 0 ] && log "Success rate: $(( passed * 100 / total ))%"
log "Results: $result_file"

[ $failed -eq 0 ]
