#!/usr/bin/env python3
"""Self-test for sweep.py. Stdlib only: `uv run python scripts/test_sweep.py`.

Covers the behavior the tool's usefulness rests on — that it refuses a spec it
cannot apply exactly, and writes nothing when it refuses. A sweep that half
applies is worse than one that does not run.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sweep import SweepError, plan, read  # noqa: E402


def fixture(root: Path) -> None:
    (root / "src" / "sub").mkdir(parents=True)
    (root / "src" / "a.rs").write_text("trait ToolHandler {\n    fn call(&self) {}\n}\n")
    (root / "src" / "sub" / "b.rs").write_text("impl ToolHandler for X {}\n")
    (root / "src" / "sub" / "c.rs").write_text("nothing to see\n")
    (root / "src" / "dup.rs").write_text("let x = 1;\nlet x = 1;\n")


def check(name: str, condition: bool) -> None:
    if not condition:
        print(f"FAIL: {name}")
        raise SystemExit(1)
    print(f"ok: {name}")


def refuses(root: Path, specs: list[dict]) -> str:
    try:
        plan(specs, root)
    except SweepError as e:
        return str(e)
    raise AssertionError("expected the spec to be refused")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root)
        before = {p: p.read_text() for p in root.rglob("*.rs")}

        hits, _ = plan(
            root=root,
            specs=[
                {
                    "glob": "src/**/*.rs",
                    "regex": r"\bToolHandler\b",
                    "replace": "Tool",
                    "count": "*",
                }
            ],
        )
        changed = {h.name for h in hits}
        check("a glob edit skips files with no hit", changed == {"src/a.rs", "src/sub/b.rs"})

        hits, _ = plan(
            root=root,
            specs=[
                {"file": "src/a.rs", "find": "trait ToolHandler", "replace": "trait Tool"},
                {"file": "src/a.rs", "find": "trait Tool {", "replace": "trait Tool: Send {"},
            ],
        )
        check(
            "a later edit sees an earlier edit's result",
            hits[0].after.startswith("trait Tool: Send {"),
        )

        message = refuses(root, [{"file": "src/dup.rs", "find": "let x = 1;", "replace": "y"}])
        check("an ambiguous anchor is refused", "expected 1" in message)

        message = refuses(root, [{"file": "src/a.rs", "find": "absent", "replace": "y"}])
        check("a missing anchor is refused", "matched nothing" in message)

        message = refuses(
            root, [{"glob": "src/**/*.rs", "find": "absent anywhere", "replace": "y"}]
        )
        check("a glob edit that hits nothing at all is refused", "nothing anywhere" in message)

        refuses(
            root,
            [
                {"file": "src/sub/b.rs", "find": "impl ToolHandler", "replace": "impl Tool"},
                {"file": "src/a.rs", "find": "absent", "replace": "y"},
            ],
        )
        check(
            "nothing is written when any edit in the spec fails",
            all(p.read_text() == text for p, text in before.items()),
        )

        message = refuses(root, [{"file": "src/a.rs", "find": "x", "replace": "y", "count": 0}])
        check("a nonsense count is refused", "positive int" in message)

        # A spec is a small program running with the user's permissions, so the
        # repo boundary has to be enforced rather than documented.
        outside = Path(tmp).parent / "outside.txt"
        outside.write_text("do not touch\n")
        try:
            message = refuses(root, [{"file": "../outside.txt", "find": "do", "replace": "did"}])
            check("a `..` path out of the repo is refused", "outside the repo" in message)

            message = refuses(
                root, [{"file": str(outside), "find": "do", "replace": "did"}]
            )
            check("an absolute path out of the repo is refused", "outside the repo" in message)

            (root / "src" / "escape.rs").symlink_to(outside)
            message = refuses(
                root, [{"glob": "src/*.rs", "find": "do not touch", "replace": "x", "count": "*"}]
            )
            check("a symlink pointing out of the repo is refused", "outside the repo" in message)

            check("nothing outside the repo was written", outside.read_text() == "do not touch\n")
        finally:
            outside.unlink()
            (root / "src" / "escape.rs").unlink(missing_ok=True)

        # A one-line edit must not rewrite every line ending in the file.
        crlf = root / "src" / "crlf.rs"
        crlf.write_bytes(b"let a = 1;\r\nlet b = 2;\r\n")
        hits, _ = plan(root=root, specs=[{"file": "src/crlf.rs", "find": "let a", "replace": "let z"}])
        check("CRLF line endings survive an edit", hits[0].after == "let z = 1;\r\nlet b = 2;\r\n")
        check("the reader does not translate newlines", read(crlf) == "let a = 1;\r\nlet b = 2;\r\n")

    print("\nall sweep self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
