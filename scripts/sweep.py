#!/usr/bin/env python3
"""Apply the same mechanical edit to many places at once, or refuse to.

A crate-wide rename or a trait method added to twenty impls is one intent and
twenty edits. Doing it by hand is twenty chances to quote the wrong context;
doing it with `sed` is one chance to match nothing at all and never find out
(BSD sed has no `\\b`, so `s/\\bOldName\\b/New/g` silently changes nothing on
macOS). This does the sweep in one pass and *fails* unless each edit hits
exactly as many places as you said it would.

Usage — a JSON list of edits on stdin:

    uv run python scripts/sweep.py <<'JSON'
    [
      {"file": "src/tool.rs", "find": "fn call(&self", "replace": "fn call_with(&self"},
      {"glob": "src/**/*.rs", "regex": "\\\\bToolHandler\\\\b", "replace": "Tool", "count": "*"}
    ]
    JSON

Edit fields:

    file / glob   where to look. `file` is one path; `glob` is a pattern, and
                  files that do not contain the pattern are skipped rather than
                  failing — but the edit still has to hit somewhere.
    find          literal text to replace. Exact, including indentation.
    regex         Python regex alternative to `find`.
    replace       what to put there. With `regex`, backreferences (\\1) work.
    count         expected hits: an int (default 1), or "*" for "one or more".
                  With `glob`, an int applies per file that has any hit.

Options:

    --dry-run     print a unified diff and write nothing
    --quiet       only report failures

Nothing is written unless every edit meets its expectation, so a spec that is
half right leaves the tree untouched instead of half edited.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


class SweepError(Exception):
    """A spec that cannot be applied as written."""


@dataclass
class Hit:
    """One file's before/after, so a caller can diff or write it."""

    path: Path
    before: str
    after: str


def _expected(spec: dict, index: int) -> int | str:
    count = spec.get("count", 1)
    if count == "*":
        return count
    if isinstance(count, int) and count > 0:
        return count
    raise SweepError(f"edit {index}: count must be a positive int or \"*\", got {count!r}")


def _apply_one(text: str, spec: dict, index: int) -> tuple[str, int]:
    """Return the rewritten text and how many places changed."""
    if "find" in spec and "regex" in spec:
        raise SweepError(f"edit {index}: use either `find` or `regex`, not both")
    if "replace" not in spec:
        raise SweepError(f"edit {index}: missing `replace`")

    if "regex" in spec:
        pattern = re.compile(spec["regex"], re.MULTILINE | re.DOTALL)
        new, n = pattern.subn(spec["replace"], text)
        return new, n
    if "find" in spec:
        needle = spec["find"]
        n = text.count(needle)
        return text.replace(needle, spec["replace"]), n
    raise SweepError(f"edit {index}: needs `find` or `regex`")


def _targets(spec: dict, index: int, root: Path) -> list[Path]:
    if "file" in spec and "glob" in spec:
        raise SweepError(f"edit {index}: use either `file` or `glob`, not both")
    if "file" in spec:
        path = root / spec["file"]
        if not path.is_file():
            raise SweepError(f"edit {index}: no such file: {spec['file']}")
        return [path]
    if "glob" in spec:
        found = sorted(p for p in root.glob(spec["glob"]) if p.is_file())
        if not found:
            raise SweepError(f"edit {index}: glob matched no files: {spec['glob']}")
        return found
    raise SweepError(f"edit {index}: needs `file` or `glob`")


def plan(specs: list[dict], root: Path) -> tuple[list[Hit], list[str]]:
    """Work out every change without writing any of it.

    Returns the hits and a per-edit report. Raises on the first edit that does
    not match its expectation — a wrong anchor is a bug in the spec, and
    applying the rest of it would leave the tree in a state nobody asked for.
    """
    pending: dict[Path, str] = {}
    hits: list[Hit] = []
    report: list[str] = []

    for index, spec in enumerate(specs):
        expected = _expected(spec, index)
        by_glob = "glob" in spec
        total = 0
        touched: list[str] = []

        for path in _targets(spec, index, root):
            original = pending.get(path)
            if original is None:
                original = path.read_text()
            new, n = _apply_one(original, spec, index)
            if n == 0:
                # Under a glob, a file that does not contain the pattern is
                # simply not one of the places this edit is about.
                if by_glob:
                    continue
                raise SweepError(
                    f"edit {index}: matched nothing in {path.relative_to(root)} "
                    f"(expected {expected})"
                )
            if expected != "*" and n != expected:
                where = path.relative_to(root)
                raise SweepError(
                    f"edit {index}: matched {n} times in {where}, expected {expected} — "
                    "the anchor is ambiguous or the file already changed"
                )
            pending[path] = new
            total += n
            touched.append(str(path.relative_to(root)))

        if total == 0:
            raise SweepError(f"edit {index}: matched nothing anywhere")
        report.append(f"edit {index}: {total} replacement(s) in {', '.join(touched)}")

    for path, after in pending.items():
        hits.append(Hit(path=path, before=path.read_text(), after=after))
    return hits, report


def diff(hits: list[Hit], root: Path) -> str:
    out: list[str] = []
    for hit in hits:
        name = str(hit.path.relative_to(root))
        out.extend(
            difflib.unified_diff(
                hit.before.splitlines(keepends=True),
                hit.after.splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    return "".join(out)


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    quiet = "--quiet" in argv
    root = Path.cwd()

    try:
        specs = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"sweep: stdin is not valid JSON: {e}", file=sys.stderr)
        return 2
    if not isinstance(specs, list) or not specs:
        print("sweep: expected a non-empty JSON list of edits", file=sys.stderr)
        return 2

    try:
        hits, report = plan(specs, root)
    except SweepError as e:
        print(f"sweep: {e}", file=sys.stderr)
        print("sweep: nothing was written", file=sys.stderr)
        return 1

    if dry_run:
        sys.stdout.write(diff(hits, root))
        if not quiet:
            print(f"sweep: dry run — {len(hits)} file(s) would change", file=sys.stderr)
        return 0

    for hit in hits:
        hit.path.write_text(hit.after)
    if not quiet:
        for line in report:
            print(f"sweep: {line}")
        print(f"sweep: wrote {len(hits)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
