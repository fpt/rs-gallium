#!/usr/bin/env bash
# Extract the response for a specific turn from gallium batch output.
#
# Usage:
#   ./extract_response.sh <output_file> [turn_number]
#
# Turn headers look like:  "=== Turn N ==="
# Default turn_number: 1

output_file="${1:-}"
turn="${2:-1}"

if [ -z "$output_file" ]; then
    echo "Usage: $0 <output_file> [turn_number]" >&2
    exit 1
fi

awk \
    -v turn="$turn" \
    'BEGIN { found=0 }
     /^=== Turn / {
         n = 0
         match($0, /[0-9]+/)
         n = substr($0, RSTART, RLENGTH) + 0
         if (n == turn) { found=1; next }
         if (n != turn && found) { exit }
     }
     found { print }' \
    "$output_file"
