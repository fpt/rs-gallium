#!/usr/bin/env bash
#
# Run a command and sample its memory once a second.
#
# Written for one question: a Metal run of an MoE model shows a sawtooth in the
# system memory graph, which looks like something allocating and freeing large
# buffers in a tight loop. A graph is a macroscopic clue and not evidence, so
# this turns it into numbers — resident size over time, and the swing between
# consecutive samples, which is what a sawtooth actually is.
#
#   bash scripts/memwatch.sh -- ./target/release/gallium --config configs/lfm2-candle.toml
#   bash scripts/memwatch.sh -i 0.5 -o /tmp/mem.csv -- <command>
#
# Options:
#   -i SECONDS   sampling interval (default 1)
#   -o FILE      where the CSV goes (default: a temp file, printed at the end)
#   --           everything after it is the command to run
#
# The CSV is `elapsed_s,rss_mb,delta_mb`, one row per sample, so it can go
# straight into a plot. The summary at the end reports peak, mean, and the
# largest swing in both directions — a steady climb and a sawtooth look the same
# in peak RSS and completely different in the swings.
#
# Caveat worth keeping in mind while reading the output: RSS is what the OS has
# resident, not what the process asked for. An allocator that reuses freed pages
# hides churn from this view entirely, so a *flat* line here does not prove there
# is no allocation traffic — it only fails to show any. Rust's default allocator
# does return large blocks to the OS, which is why this is worth sampling at all,
# but a null result should send you to a profiler rather than to a conclusion.
set -euo pipefail

interval=1
out=""
while [ $# -gt 0 ]; do
    case "$1" in
        -i) interval="$2"; shift 2 ;;
        -o) out="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "memwatch: unexpected argument '$1' (did you forget --?)" >&2; exit 2 ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: memwatch.sh [-i SECONDS] [-o FILE] -- COMMAND [ARGS...]" >&2
    exit 2
fi

if [ -z "$out" ]; then
    out="$(mktemp -t memwatch)"
fi

"$@" &
pid=$!

echo "elapsed_s,rss_mb,delta_mb" > "$out"
started=$(date +%s)
prev=""

# The whole process *tree*, not just the pid we started: a command is usually
# wrapped in a shell (`sh -c 'cd … && …'`), and sampling the wrapper reports its
# own two megabytes while the model runs beside it — which is exactly the null
# result this script must not produce quietly.
#
# `ps` rather than anything richer, so this works on a stock macOS or Linux with
# no privileges: the point is the shape of the curve, not a breakdown.
tree_rss_kb() {
    ps -eo pid=,ppid=,rss= 2>/dev/null | awk -v root="$1" '
        { rss[$1] = $3; parent[$1] = $2; kids[$2] = kids[$2] " " $1 }
        END {
            n = 1; queue[1] = root; total = 0
            for (i = 1; i <= n; i++) {
                pid = queue[i]
                if (!(pid in rss)) continue
                total += rss[pid]
                split(kids[pid], c, " ")
                for (j in c) if (c[j] != "") queue[++n] = c[j]
            }
            print total
        }'
}

while kill -0 "$pid" 2>/dev/null; do
    rss_kb=$(tree_rss_kb "$pid")
    if [ -n "$rss_kb" ] && [ "$rss_kb" != "0" ]; then
        now=$(date +%s)
        awk -v t="$((now - started))" -v kb="$rss_kb" -v prev="$prev" \
            'BEGIN { mb = kb / 1024; d = (prev == "" ? 0 : mb - prev); printf "%d,%.1f,%+.1f\n", t, mb, d }' \
            >> "$out"
        prev=$(awk -v kb="$rss_kb" 'BEGIN { printf "%.1f", kb / 1024 }')
    fi
    sleep "$interval"
done

wait "$pid" 2>/dev/null && status=0 || status=$?
wall=$(( $(date +%s) - started ))

echo
echo "=== memwatch: $(($(wc -l < "$out") - 1)) samples every ${interval}s, wall ${wall}s, exit $status ==="
awk -F, 'NR > 1 {
    n++; sum += $2
    if ($2 > peak) peak = $2
    if (min == 0 || $2 < min) min = $2
    if ($3 > up) up = $3
    if ($3 < down) down = $3
    if ($3 > 0) { rises++; risen += $3 } else if ($3 < 0) { falls++; fell -= $3 }
}
END {
    if (n == 0) { print "no samples — the command exited too quickly"; exit }
    printf "rss    min %.0f MB   mean %.0f MB   peak %.0f MB\n", min, sum / n, peak
    printf "swing  largest rise %+.0f MB   largest fall %.0f MB\n", up, down
    printf "churn  %d samples rose (%.0f MB total), %d fell (%.0f MB total)\n", rises, risen, falls, fell
    if (falls > 0 && fell > peak - min) {
        printf "\nThe total fallen (%.0f MB) exceeds the whole range (%.0f MB): memory is\n", fell, peak - min
        print  "being returned and taken again rather than simply growing — which is the"
        print  "shape a per-step allocate/free loop makes."
    }
}' "$out"
echo "csv: $out"
exit "$status"
