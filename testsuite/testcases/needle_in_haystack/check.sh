#!/usr/bin/env bash
# Verify the model correctly retrieves a unique string ("needle") buried roughly
# midway through a ~500-token document ("haystack"), delivered as ONE turn.
#
# prompt.txt is deliberately a single physical line: runner.sh sends every
# non-empty line of prompt.txt as a separate REPL turn, so a paragraph-broken
# version of this report was actually ~24 separate turns — turn 1 was just
# the instruction sentence, and the needle could never appear in it. That
# tested nothing (the model spent minutes trying to `Read` a file that didn't
# exist) and always would have failed a strict check, since this test means
# to measure in-context attention over one long prompt, not multi-turn recall
# (that's memory_state's job). Keep it one line if you edit this fixture.
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

resp=$(./extract_response.sh "$output_file")

echo "$resp" | grep -qi "FALCON-RIDGE-7823" \
    || fail "Expected needle FALCON-RIDGE-7823 in response. Got: $resp"

echo "PASS"
