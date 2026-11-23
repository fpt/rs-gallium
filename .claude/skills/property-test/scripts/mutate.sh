#!/usr/bin/env bash
# Plant bugs (mutations) one at a time and report which tests catch each one.
#
# usage: mutate.sh <mutations.tsv> <label>=<test command> [<label>=<test command> ...]
#
# mutations.tsv: one mutation per line, "name<TAB>file<TAB>perl substitution".
# Lines starting with # are ignored. perl runs with -0, so a substitution may
# span lines (\s*, \n). A test command that exits non-zero has caught the
# mutation; a compile error is reported separately.
#
# Works on a dirty tree: each target file is backed up before a mutation and
# restored after it (also on interrupt). proptest-regressions files are
# snapshotted first and restored at the end, so counterexamples found against
# mutated code never survive the run. Run from the repository root.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,14p' "$0"
  exit 2
fi
tsv="$1"; shift
labels=() commands=()
for spec in "$@"; do
  labels+=("${spec%%=*}")
  commands+=("${spec#*=}")
done

backup="$(mktemp)"
regressions="$(mktemp -d)"
find crates -path '*/proptest-regressions/*' -type f -print0 2>/dev/null |
  while IFS= read -r -d '' f; do mkdir -p "$regressions/$(dirname "$f")"; cp "$f" "$regressions/$f"; done

current=""
cleanup() {
  if [ -n "$current" ]; then cp "$backup" "$current"; fi
  rm -f "$backup"
  # Counterexamples written during the run describe the mutated code.
  find crates -path '*/proptest-regressions/*' -type f -delete 2>/dev/null || true
  (cd "$regressions" && find . -type f -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$OLDPWD/$(dirname "$f")"; cp "$f" "$OLDPWD/$f"; done) || true
  find crates -type d -name proptest-regressions -empty -delete 2>/dev/null || true
  rm -rf "$regressions"
}
trap cleanup EXIT INT TERM

row() {
  local name="$1" cell="$2"
  printf '| %s |' "$name"
  for _ in "${labels[@]}"; do printf ' %s |' "$cell"; done
  printf '\n'
}

printf '| mutation |'; for l in "${labels[@]}"; do printf ' %s |' "$l"; done; printf '\n'
printf '|---|'; for _ in "${labels[@]}"; do printf -- '---|'; done; printf '\n'

while IFS=$'\t' read -r name file subst; do
  [ -z "${name:-}" ] && continue
  case "$name" in \#*) continue ;; esac
  cp "$file" "$backup"
  current="$file"
  if ! perl -0pi -e "$subst" "$file" 2>/dev/null; then
    cp "$backup" "$file"; current=""
    row "$name" "bad substitution"
    continue
  fi
  if cmp -s "$file" "$backup"; then
    current=""
    row "$name" "not applied (no match)"
    continue
  fi
  printf '| %s |' "$name"
  for i in "${!commands[@]}"; do
    out="$(bash -c "${commands[$i]}" 2>&1)" && status=0 || status=$?
    if [ "$status" -eq 0 ]; then
      printf ' **missed** |'
    elif printf '%s' "$out" | grep -qE '^error\[E[0-9]+\]|could not compile'; then
      printf ' build failed |'
    else
      printf ' caught |'
    fi
  done
  printf '\n'
  cp "$backup" "$file"; current=""
done < "$tsv"
