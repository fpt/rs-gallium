#!/usr/bin/env bash
# Self-test for pr-merged-guard.sh. Run: bash .claude/hooks/test-pr-merged-guard.sh
#
# The guard fails open by design, which means a broken guard is indistinguishable
# from a quiet one: it would simply stop catching the mistake and never say so.
# That is the whole reason this file exists.
#
# `git` and `gh` are stubbed rather than driven for real, so every branch is
# reachable — a live repo can only be in one state at a time.

set -u

guard="$(cd "$(dirname "$0")" && pwd)/pr-merged-guard.sh"
stub_dir=$(mktemp -d)
trap 'rm -rf "$stub_dir"' EXIT

cat > "$stub_dir/git" <<'STUB'
#!/bin/sh
[ "${FAKE_GIT_EXIT:-0}" != "0" ] && exit "$FAKE_GIT_EXIT"
printf '%s\n' "${FAKE_BRANCH:-main}"
STUB

cat > "$stub_dir/gh" <<'STUB'
#!/bin/sh
[ "${FAKE_GH_EXIT:-0}" != "0" ] && exit "$FAKE_GH_EXIT"
printf '%s\n' "${FAKE_PR_STATE:-OPEN}"
STUB

chmod +x "$stub_dir/git" "$stub_dir/gh"

failures=0

# run <description> <expectation: fires|silent> <command-string> [env assignments...]
run() {
    description=$1; expectation=$2; command=$3; shift 3
    output=$(
        env PATH="$stub_dir:$PATH" "$@" \
            bash "$guard" <<< "$(printf '{"tool_name":"Bash","tool_input":{"command":%s}}' "$(printf '%s' "$command" | jq -Rs .)")"
    )
    status=$?

    if [ "$status" != "0" ]; then
        printf 'FAIL: %s — exited %s; a hook that errors is noise on every call\n' "$description" "$status"
        failures=$((failures + 1))
        return
    fi
    case "$expectation" in
        fires)
            if printf '%s' "$output" | jq -e '.hookSpecificOutput.permissionDecision == "ask"' >/dev/null 2>&1; then
                printf 'ok: %s\n' "$description"
            else
                printf 'FAIL: %s — expected an ask, got: %s\n' "$description" "${output:-<nothing>}"
                failures=$((failures + 1))
            fi
            ;;
        silent)
            if [ -z "$output" ]; then
                printf 'ok: %s\n' "$description"
            else
                printf 'FAIL: %s — expected silence, got: %s\n' "$description" "$output"
                failures=$((failures + 1))
            fi
            ;;
    esac
}

run "a push to a merged branch asks" fires \
    "git push -q -u origin HEAD" FAKE_BRANCH=feat/x FAKE_PR_STATE=MERGED

run "a push buried in a compound command still asks" fires \
    "git add -A && git commit -q -m x && git push -q && git log -1" FAKE_BRANCH=feat/x FAKE_PR_STATE=MERGED

run "a push to a branch whose PR is open is silent" silent \
    "git push" FAKE_BRANCH=feat/x FAKE_PR_STATE=OPEN

run "a command that is not a push is silent" silent \
    "cargo test --workspace" FAKE_BRANCH=feat/x FAKE_PR_STATE=MERGED

run "a branch with no PR is silent" silent \
    "git push" FAKE_BRANCH=feat/x FAKE_GH_EXIT=1

run "a detached HEAD is silent" silent \
    "git push" FAKE_BRANCH=HEAD FAKE_PR_STATE=MERGED

run "somewhere that is not a git repo is silent" silent \
    "git push" FAKE_GIT_EXIT=128 FAKE_PR_STATE=MERGED

# Not JSON at all: the guard must not spray jq errors into every Bash call.
if output=$(env PATH="$stub_dir:$PATH" bash "$guard" <<< "not json" 2>&1) && [ -z "$output" ]; then
    printf 'ok: a malformed payload is silent\n'
else
    printf 'FAIL: a malformed payload produced: %s\n' "${output:-<error exit>}"
    failures=$((failures + 1))
fi

if [ "$failures" -eq 0 ]; then
    printf '\nall pr-merged-guard tests passed\n'
else
    printf '\n%s test(s) failed\n' "$failures"
    exit 1
fi
