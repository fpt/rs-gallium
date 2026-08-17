#!/usr/bin/env bash
#
# Gate helpers for *scenario* testcases — the ones that ask for a piece of work
# with several requirements rather than a single answer.
#
# The older testcases are fail-fast: `check.sh` calls `fail` on the first
# problem and exits. That is right when there is one thing to verify (did it say
# "Paris"), and wrong for a scenario, where it throws away the measurement. A
# model that compiles the code, keeps the existing test honest, and then forgets
# one method is not the same as a model that produced nothing — but fail-fast
# reports both as one red cell.
#
# So a scenario declares **gates**, each checked independently, and the result is
# a score: `4/6 gates passed` plus exactly which one broke and why. That turns a
# binary matrix into a graded one, which is what makes two local models
# comparable on a task neither fully completes.
#
# Usage, from a testcase's check.sh (runner.sh exports TESTSUITE_DIR):
#
#     source "$TESTSUITE_DIR/gates.sh"
#     gate "package builds"       'go build ./...'
#     gate "existing test intact" 'grep -q "want 5" stock_test.go'
#     gate_summary
#
# `gate` takes a name and a shell snippet, evaluated with the testcase's cwd (the
# temp workdir). Non-zero exit fails the gate; its output is quoted into the
# report so a failure explains itself. `gate_summary` prints the score and exits
# 0 only if every gate passed — the runner still sees one pass/fail, so nothing
# about the matrix changes.

_gate_index=0
_gate_passed=0
_gate_failed_names=()

# gate <name> <shell snippet>
gate() {
    local name="$1"
    local snippet="$2"
    _gate_index=$((_gate_index + 1))

    local out status
    # `eval` because a gate is a shell expression — pipelines, globs, `grep -q`
    # — not an argv. The snippets are testcase source, not model output, so
    # there is nothing here for a model to inject into.
    #
    # Written as one `&&`/`||` compound rather than a bare assignment followed by
    # `$?`, so that a failing gate cannot trip `set -e` in a testcase that uses
    # it: errexit fires on a failed command substitution in a plain assignment,
    # and the whole point here is to keep going after a failure.
    out="$(eval "$snippet" 2>&1)" && status=0 || status=$?

    if [ $status -eq 0 ]; then
        _gate_passed=$((_gate_passed + 1))
        printf '  OK   gate %d - %s\n' "$_gate_index" "$name"
    else
        _gate_failed_names+=("gate $_gate_index - $name")
        printf '  MISS gate %d - %s\n' "$_gate_index" "$name"
        if [ -n "$out" ]; then
            # Bounded: a failing `go build` can print a screen of errors, and the
            # first few lines are what identifies the failure.
            printf '%s\n' "$out" | head -8 | sed 's/^/        /'
        fi
    fi
}

# Print the score and exit: 0 only when every gate passed.
gate_summary() {
    # A scenario with no gates is a broken testcase, not a passing one: `0/0`
    # would otherwise report PASS on the matrix, so a check.sh that sources this
    # file and then fails before declaring anything (a typo'd `gate` name, an
    # early `return`) looks exactly like a model that did the work.
    if [ "$_gate_index" -eq 0 ]; then
        echo "FAIL: no gates were declared — check.sh is misconfigured"
        return 1
    fi

    printf '\n  %d/%d gates passed\n' "$_gate_passed" "$_gate_index"
    if [ ${#_gate_failed_names[@]} -eq 0 ]; then
        echo "PASS"
        return 0
    fi
    printf '  failed:\n'
    printf '    %s\n' "${_gate_failed_names[@]}"
    # Named FAIL so a log scan finds it the same way the older testcases' output
    # is found.
    echo "FAIL: ${#_gate_failed_names[@]} gate(s) failed"
    return 1
}
