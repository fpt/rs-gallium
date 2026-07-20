#!/bin/bash
# Multi-turn memory: turn 2 must recall the color from turn 1.
# $1 = output file, $2 = error file. cwd = temp test dir.
turn2="$(./extract_response.sh "$1" 2)"
if echo "$turn2" | grep -iq "teal"; then
    echo "✓ Turn 2 recalled 'teal'"
    exit 0
fi
echo "✗ Turn 2 did not recall 'teal':"; echo "$turn2"
exit 1
