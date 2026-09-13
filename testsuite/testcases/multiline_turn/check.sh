#!/usr/bin/env bash
# Verify the model saw the whole multi-line turn: the function's body is on a
# later line than the question's subject, so a model handed only one line of
# it cannot answer.
set -euo pipefail
output="$1"

fail() { echo "FAIL: $1"; exit 1; }

turn1=$(./extract_response.sh "$output" 1)
echo "$turn1" | grep -qE '(^|[^0-9])5([^0-9]|$)' \
    || fail "expected 5 (add(2, 3)), got: $turn1"
echo "PASS"
