#!/bin/bash

# Run every testcase against every backend and print a PASS/FAIL matrix.
# Usage: CLI=path/to/gallium ./testsuite/matrix_runner.sh
#
# Optional comma-separated filters:
#   TESTS=capital,memory_state       run only matching testcases
#   BACKENDS=gemma4,gpt-5.6-luna     run only matching backends

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
    local f="$script_dir/backends/$1.toml"
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
for f in $(find "$script_dir/backends" -maxdepth 1 -name '*.toml' | sort); do
    n="$(basename "$f" .toml)"
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

entries=""; total=0; passed=0; failed=0
for b in $backends; do
    for t in $testcases; do
        total=$((total+1))
        log "${CYAN}▶ $t × $b${NC}"
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
        done
        row="$row$(printf "%-${col_w}s" "$r")"
    done
    log "$row"
done

log ""
log "${BLUE}📊 Summary:${NC} Total: $total  Passed: $passed  Failed: $failed"
[ $total -gt 0 ] && log "Success rate: $(( passed * 100 / total ))%"
log "Results: $result_file"

[ $failed -eq 0 ]
