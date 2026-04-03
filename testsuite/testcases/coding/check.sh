#!/usr/bin/env bash
# Verify that the agent created hello.go, it compiles, and prints "Hello, World!".
set -euo pipefail
output="$1"

fail() { echo "FAIL: $1"; exit 1; }

[ -f "hello.go" ] || fail "hello.go was not created"
grep -q "package main" hello.go || fail "hello.go missing 'package main'"
grep -q "func main" hello.go   || fail "hello.go missing 'func main'"

go build -o hello_bin ./hello.go 2>build_err.txt \
    || fail "compilation failed: $(cat build_err.txt)"

result=$(./hello_bin 2>/dev/null)
rm -f hello_bin build_err.txt

echo "$result" | grep -qi "hello" \
    || fail "expected 'Hello, World!' output, got: $result"

rm -f hello.go
echo "PASS"
