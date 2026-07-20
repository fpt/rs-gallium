#!/bin/bash
# Tool use: the model must call the `read` tool on codeword.txt and report it.
# $1 = output file, $2 = error file. cwd = temp test dir.
resp="$(./extract_response.sh "$1")"
if echo "$resp" | grep -iq "zucchini"; then
    echo "✓ Found 'ZUCCHINI' (read tool worked)"
    exit 0
fi
echo "✗ 'ZUCCHINI' not found — model likely didn't use the read tool:"; echo "$resp"
exit 1
