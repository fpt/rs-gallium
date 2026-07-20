#!/bin/bash

# Extract assistant response text from gallium output.
# Usage: extract_response.sh <output_file> [turn_number]
#
# The gallium REPL prints each reply as a line starting with "Assistant: ",
# optionally followed by continuation lines, ending before the next "You:" prompt.
#   - No turn number: prints all assistant text.
#   - Turn N: prints just the Nth assistant response block.

output_file="$1"
turn_number="$2"

if [ -z "$output_file" ] || [ ! -f "$output_file" ]; then
    echo "Usage: extract_response.sh <output_file> [turn_number]" >&2
    exit 1
fi

if [ -z "$turn_number" ]; then
    # All assistant blocks: from each "Assistant:" line to the next "You:".
    awk '
        /^Assistant:/ { p=1 }
        p && /^You:/   { p=0 }
        p             { print }
    ' "$output_file" | sed 's/^Assistant: //'
else
    awk -v t="$turn_number" '
        /^Assistant:/ { c++; if (c==t) p=1 }
        p && /^You:/   { p=0 }
        p             { print }
    ' "$output_file" | sed 's/^Assistant: //'
fi
