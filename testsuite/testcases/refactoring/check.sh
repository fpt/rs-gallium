#!/usr/bin/env bash
# Verify counter.go was refactored: uses a struct, compiles, still prints 3.
set -euo pipefail
output="$1"

fail() { echo "FAIL: $1"; exit 1; }

[ -f "counter.go" ] || fail "counter.go missing"

grep -q "struct" counter.go \
    || fail "counter.go should define a struct"

# Global variable should be gone
grep -q "^var count" counter.go \
    && fail "global 'var count' should have been removed"

go build -o counter_bin ./counter.go 2>build_err.txt \
    || fail "compilation failed: $(cat build_err.txt)"

result=$(./counter_bin 2>/dev/null)
rm -f counter_bin build_err.txt

echo "$result" | grep -q "3" \
    || fail "expected output '3', got: $result"

echo "PASS"
