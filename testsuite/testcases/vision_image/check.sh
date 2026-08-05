#!/usr/bin/env bash
# Vision: the reply must contain the number drawn in number.png (42).
# $1 = output file, $2 = error file. cwd = temp test dir.
set -uo pipefail

resp="$(./extract_response.sh "$1")"

# number.png is in the agent's cwd, so a pass could also come from Bash decoding
# the file rather than from the model seeing it. That is a real agent capability
# but it is not the one under test, so it does not count as a pass.
if grep -q 'ReAct iteration .*tool call' "$2"; then
    echo "✗ the agent used tools — number.png was decoded, not seen."
    echo "  reply: $resp"
    exit 1
fi

# Anchored on a digit boundary so "4" and "2" appearing separately, or a stray
# "142", do not count as having read the image.
if grep -Eq '(^|[^0-9])42([^0-9]|$)' <<<"$resp"; then
    echo "✓ read '42' out of the image"
    exit 0
fi

echo "✗ '42' not found — the model did not read the image."
echo "  reply: $resp"
# The backend refusing outright is the other expected outcome, and it is a
# different failure from a model that looked and got it wrong. Name which.
if grep -q "cannot see images" "$2"; then
    echo "  cause: this backend has no vision path (it said so, rather than"
    echo "         answering about an image it never received)."
fi
exit 1
