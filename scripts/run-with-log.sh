#!/usr/bin/env bash
#
# Run a gallium invocation with stderr captured to a retained, timestamped log
# file — not just the terminal scrollback, which is what's lost when a crash
# (a hard llama.cpp/ggml SIGABRT with its own gdb backtrace, or an ordinary
# decode error) scrolls past or the pane closes before anyone reads it.
#
#   scripts/run-with-log.sh -- gallium app-server --listen 0.0.0.0:47821
#   scripts/run-with-log.sh -d ~/gallium-logs -- gallium app-server --listen 0.0.0.0:47821
#
# Only stderr is captured — where gallium's own tracing output and ggml's own
# crash handler (it execs gdb on SIGABRT/SIGSEGV, see ggml/src/ggml.c) both
# write. stdout is left untouched: an app-server run over stdio needs it as the
# raw JSON-RPC channel (CLAUDE.md's "stdout is the JSON-RPC stream in this
# mode"), and `--listen` doesn't use it either, so there's nothing to gain by
# touching it. Output still prints live to the terminal via `tee`.
#
# Options:
#   -d DIR   where log files go (default: ~/.config/gallium/logs)
#   --       everything after it is the command to run
set -euo pipefail

logdir="${GALLIUM_LOG_DIR:-$HOME/.config/gallium/logs}"
while [ $# -gt 0 ]; do
    case "$1" in
        -d) logdir="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "run-with-log: unexpected argument '$1' (did you forget --?)" >&2; exit 2 ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "usage: run-with-log.sh [-d LOGDIR] -- COMMAND [ARGS...]" >&2
    exit 2
fi

mkdir -p "$logdir"
logfile="$logdir/$(date +%Y%m%d-%H%M%S).log"
echo "logging stderr to $logfile"

"$@" 2> >(tee "$logfile" >&2)
status=$?

echo
echo "=== exited with status $status — log: $logfile ===" >&2
exit "$status"
