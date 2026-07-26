#!/usr/bin/env bash
# Warn before pushing to a branch whose pull request has already merged.
#
# The mistake it catches: you open a PR, it gets merged, and you keep working on
# the same branch. The next push succeeds, GitHub shows nothing (a merged PR does
# not reopen), and the commits never reach main. It looks like the work landed.
#
# Wired as a PreToolUse hook on Bash in .claude/settings.json.
#
# Two deliberate choices:
#
#   - It ASKS rather than blocks. Pushing to a merged branch is occasionally
#     correct — restoring one to its merge state, for instance — and a hook that
#     forbids the repair for the same mistake it flags is worse than no hook.
#
#   - It matches "git push" anywhere in the command, not as a prefix. Pushes are
#     usually the tail of a compound: `git add -A && git commit -m … && git push`.
#     A prefix match would look installed and never fire.
#
# It fails open everywhere except the one case where it has positive evidence:
# no git, no gh, no PR, unauthenticated, offline, detached HEAD — all silent.
# The guard exists to catch a mistake, not to stand between you and a push.
#
# Test: .claude/hooks/test-pr-merged-guard.sh

set -u

payload=$(cat)
command=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null) || exit 0

case "$command" in
    *"git push"*) ;;
    *) exit 0 ;;
esac

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
# "HEAD" means detached: no branch, so no PR to ask about.
[ -n "$branch" ] && [ "$branch" != "HEAD" ] || exit 0

state=$(gh pr view "$branch" --json state -q .state 2>/dev/null) || exit 0
[ "$state" = "MERGED" ] || exit 0

# `ask` surfaces the reason and hands the decision back.
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"The PR for branch %s is already MERGED. Commits pushed here will not reach main — branch off main first, unless you are deliberately restoring this branch."}}' "$branch"
