#!/usr/bin/env bash
# Verify multi-turn memory: frog legs (turn 1) → arms (turn 2 must recall context).
set -euo pipefail
output="$1"

fail() { echo "FAIL: $1"; exit 1; }

# Turn 1: should answer "4" (or "four")
turn1=$(./extract_response.sh "$output" 1)
echo "$turn1" | grep -qiE "4|four" \
    || fail "Turn 1 should say frogs have 4 legs. Got: $turn1"

# Turn 2: must retain context (mention frog/amphibian) and answer about limbs
turn2=$(./extract_response.sh "$output" 2)
echo "$turn2" | grep -qiE "frog|amphibian|forelimb" \
    || fail "Turn 2 should reference frogs (context memory). Got: $turn2"
echo "$turn2" | grep -qiE "2|two|no arm|don.t have arm|forelimb|zero" \
    || fail "Turn 2 should answer about arms/forelimbs. Got: $turn2"

echo "PASS"
