#!/usr/bin/env bash
# Verify the model correctly retrieves a unique string ("needle") buried roughly
# midway through a ~500-token document ("haystack").
#
# For GPT-OSS (sliding_window=128): the needle sits at absolute token position
# ~350 — past one full sliding-window span from the end.  Full-attention layers
# (odd indices 1, 3, ..., 23) must carry the needle across the sliding boundary
# to the final layer.  A broken layer_types mapping or a missing decode mask
# causes the model to hallucinate a code or produce incoherent output.
#
# For models with larger sliding windows (e.g. Gemma 4 at 1024) this is a
# straightforward long-context recall test.
set -euo pipefail
output_file="$1"

fail() { echo "FAIL: $1"; exit 1; }

turn1=$(./extract_response.sh "$output_file" 1)

echo "$turn1" | grep -qi "FALCON-RIDGE-7823" \
    || fail "Expected needle FALCON-RIDGE-7823 in response. Got: $turn1"

echo "PASS"
