#!/bin/bash
# $1 = output file, $2 = error file. cwd = temp test dir.
resp="$(./extract_response.sh "$1")"
if echo "$resp" | grep -q "391"; then
    echo "✓ Found 391"
    exit 0
fi
echo "✗ 391 not found in response:"; echo "$resp"
exit 1
