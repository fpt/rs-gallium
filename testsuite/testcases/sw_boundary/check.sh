#!/usr/bin/env bash
# Verify the model correctly recalls a fact placed before the sliding-window
# boundary.  The prompt token count (with protocol overhead) exceeds GPT-OSS's
# sliding_window=128, so the first generated token is decoded at pos>128.
# A correct implementation attends to early context via full-attention layers;
# a broken decode-time SW mask produces garbage or a wrong token.
set -euo pipefail
output_file="$1"

fail() { echo "FAIL: $1"; exit 1; }

turn1=$(./extract_response.sh "$output_file" 1)

# The unique token that appears only at the very start of the prompt.
echo "$turn1" | grep -qi "IRON-CROW-4491" \
    || fail "Expected recovery token IRON-CROW-4491 in response. Got: $turn1"

echo "PASS"
