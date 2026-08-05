#!/usr/bin/env bash
# Audio: the reply must name the pitch of tone.wav (613 Hz).
#
# EXPECTED TO FAIL on every backend today — gallium carries no audio, so the
# model is answering about a clip it never received. This testcase is the record
# of that gap: a failing test that names the missing feature, rather than a
# comment in a file nobody runs. See testsuite/README.md.
#
# $1 = output file, $2 = error file. cwd = temp test dir.
set -uo pipefail

resp="$(./extract_response.sh "$1")"

# tone.wav is in the agent's cwd, so a pass could also come from Bash reading
# the samples rather than from the model hearing them. That is a real agent
# capability but it is not the one under test, so it does not count as a pass.
if grep -q 'ReAct iteration .*tool call' "$2"; then
    echo "✗ the agent used tools — tone.wav was analyzed, not heard."
    echo "  reply: $resp"
    exit 1
fi

# 605-620 Hz: room for a model reporting a pitch it measured, without letting a
# nearby round number through.
if grep -Eq '(^|[^0-9])6(0[5-9]|1[0-9]|20)([^0-9]|$)' <<<"$resp"; then
    echo "✓ named the tone's frequency (~613 Hz) — audio input is working"
    exit 0
fi

echo "✗ ~613 Hz not found — the model did not hear the clip."
echo "  reply: $resp"
echo "  expected: gallium has no audio input path, so this is the known gap"
echo "            this testcase exists to record."
exit 1
