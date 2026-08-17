#!/usr/bin/env bash
#
# A small data-analysis task, scored by gates — see `gates.sh`.
#
# Real analysis work is rarely one computation: it is reading a file, noticing
# that some of it is unusable, aggregating what is left, formatting the result to
# a spec, and then saying what it means. Each of those fails differently, so each
# is its own gate. In particular the arithmetic and the *data-quality* judgement
# are separated: a model that sums correctly but treats an empty `units` field as
# zero gets one wrong region and every other one right, and that is a much more
# useful signal than a single red cell.
#
# The file is dirty on purpose, in the three ways a concatenated export usually
# is: an empty `units`, a non-numeric `units` (`N/A`), and the header line
# repeated partway down. Note what this does and does not test. For a *sum*,
# skipping a bad row and treating it as zero give the same number, so the trap is
# not arithmetic — it is that a naive parse (`float(row[2])` over every line)
# raises on all three and produces nothing at all. The gate measures whether the
# model noticed the file was not clean.
#
# The expected figures, computed independently with awk over the same file:
#
#     north=67.50  west=50.00  east=30.00  south=12.50
#
# Revenue order (north, west, east, south) is deliberately *not* alphabetical
# (east, north, south, west), so the ordering gate cannot be passed by sorting on
# the wrong key — the first draft of this fixture had them coincide.
#
# Deliberately no `set -e`: a failing gate must not end the run.

source "$TESTSUITE_DIR/gates.sh"

output_file="$1"
reply="$(./extract_response.sh "$output_file")"

# Normalized view of the produced file: leading and trailing whitespace trimmed,
# blank lines gone, so formatting slack (a trailing newline, an indented line) is
# not what fails a gate.
#
# Trimmed at the ends only, deliberately. Stripping *all* whitespace would repair
# the output as it read it — `north=6 7.50` would normalize to `north=67.50` and
# pass the totals gate, though the prompt asks for an exact form and no consumer
# of that file could parse it. Slack at the edges is a formatting nicety; a space
# inside the amount is a wrong answer.
norm() { sed 's/^[[:space:]]*//; s/[[:space:]]*$//' revenue.txt | grep -v '^$'; }

gate "revenue.txt was written" \
    '[ -s revenue.txt ]'

gate "one line per region, four regions, no duplicates" \
    '[ "$(norm | wc -l | tr -d " ")" = "4" ] &&
     [ "$(norm | cut -d= -f1 | sort -u | tr "\n" " ")" = "east north south west " ]'

# The three regions whose arithmetic involves no judgement call. Split from the
# `east` gate so "cannot multiply and sum" and "mishandled the dirty row" are
# separate findings.
gate "north, south and west totals are correct" \
    'norm | grep -qx "north=67.50" &&
     norm | grep -qx "south=12.50" &&
     norm | grep -qx "west=50.00"'

# East owns both unusable rows (the empty `units` and the `N/A`). 30.00 means
# they contributed nothing; any other value means they were coerced into the sum,
# and no line at all usually means the parse raised partway through the file.
gate "east is 30.00 — the unusable rows did not corrupt it" \
    'norm | grep -qx "east=30.00"'

gate "lines are sorted from highest revenue to lowest" \
    '[ "$(norm | cut -d= -f1 | tr "\n" " ")" = "north west east south " ]'

# The final goal, and the only gate answered in prose rather than in a file:
# having done the work, say what it means. `-iE` with word boundaries because
# "North", "the north region" and "north." are all the same answer, while
# "northbound" is not.
gate "the reply names north as the highest-revenue region" \
    'printf %s "$reply" | grep -qiE "\bnorth\b"'

gate_summary
