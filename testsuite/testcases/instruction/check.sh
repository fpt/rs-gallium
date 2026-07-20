#!/bin/bash
# $1 = output file, $2 = error file. cwd = temp test dir.
resp="$(./extract_response.sh "$1")"
if echo "$resp" | grep -iq "pineapple"; then
    echo "✓ Found 'PINEAPPLE'"
    exit 0
fi
echo "✗ 'PINEAPPLE' not found in response:"; echo "$resp"
exit 1
