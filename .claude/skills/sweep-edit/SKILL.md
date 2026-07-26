---
name: sweep-edit
description: Apply one mechanical change across many files or many sites at once — a crate-wide rename, a trait method added to every impl, a new parameter threaded through every call site. Triggered when an edit is the same shape in roughly eight or more places, or when a rename spans multiple files. Not for ordinary edits.
argument-hint: "[what to change]  — e.g. rename ToolHandler to Tool, add annotations() to every tool impl"
allowed-tools: Bash, Read, Grep, Glob
---

# Sweeping mechanical edits

Some changes are one intent and twenty edits: renaming a trait across a crate,
adding a method to every `impl`, threading a parameter through every call site.
Doing those with twenty `Edit` calls costs twenty exact context quotes and a lot
of tokens. `scripts/sweep.py` does the whole sweep in one call and refuses to
write anything unless each edit hits exactly as many places as you claimed.

## When to use this, and when not to

| Situation | Use |
|---|---|
| Same edit in ~8+ places, or a rename spanning files | `sweep.py` |
| Fewer than that, or each site differs | `Edit` |
| Anything the user will want to review closely | `Edit` — it renders as a diff |
| Whole-file rewrite | `Write` |

`Edit` is the default. The sweep is for when the repetition itself is the cost.
It shows up in the user's terminal as a bash command, not as a reviewable diff,
so prefer `--dry-run` first on anything non-obvious and paste the diff.

**Never use `sed` for this.** BSD sed (macOS) has no `\b`, so
`sed -i '' 's/\bOldName\b/New/g'` matches nothing and reports success — a silent
no-op you only notice later with `grep`.

## How to run it

Per CLAUDE.md, always `uv run python`, never `python3`. The spec is a JSON list
of edits on stdin:

```bash
uv run python scripts/sweep.py --dry-run <<'JSON'
[
  {"glob": "crates/gallium-agent/src/**/*.rs", "regex": "\\bToolHandler\\b",
   "replace": "Tool", "count": "*"},
  {"file": "crates/gallium-agent/src/tool.rs",
   "find": "    fn call(&self, args: serde_json::Value)",
   "replace": "    fn call_with(&self, ctx: &TurnContext, args: serde_json::Value)"}
]
JSON
```

Drop `--dry-run` to write. Edits apply in order and later edits see earlier
results, so a rename followed by an insertion into the renamed text works.

| Field | Meaning |
|---|---|
| `file` | one path, relative to the repo root |
| `glob` | a pattern; files without a hit are skipped, but the edit must hit somewhere |
| `find` | literal text, exact including indentation |
| `regex` | Python regex instead of `find`; `\1` backreferences work in `replace` |
| `replace` | the replacement |
| `count` | expected hits, default `1`; `"*"` means one or more; with `glob`, per file |

## Boundaries

Every path must resolve to somewhere under the directory sweep runs from. A
`..` path, an absolute path, or a symlink pointing out of the tree is refused —
a spec is a small program running with your permissions, so the repo edge is
enforced rather than documented.

Line endings are preserved byte for byte. The consequence to remember: an
anchor written with `\n` will not match a CRLF file, so such an edit is refused
rather than quietly rewriting every line in it.

## The guarantee, and why it matters

Nothing is written unless **every** edit meets its expectation. A wrong anchor
is a bug in the spec, and applying the rest would leave the tree in a state
nobody asked for. The default `count: 1` is what makes an ambiguous anchor an
error rather than a surprise:

```
sweep: edit 0: matched 2 times in src/tool.rs, expected 1 — the anchor is
ambiguous or the file already changed
sweep: nothing was written
```

This is the same guarantee `Edit` gives, kept rather than traded away for speed.

## After a sweep

A sweep is not verified until the compiler says so. Always follow with:

```bash
cargo fmt --all && cargo check --workspace --all-targets
```

`cargo fmt` matters especially here: an inserted method arrives with whatever
indentation the spec had, and rustfmt will not reformat the inside of a `json!`
or other macro body, so check anything inserted near one by eye.

## Writing good anchors

1. **Anchor on something unique.** A whole signature line beats a fragment. If
   `count` fails with "matched N times", lengthen the anchor rather than raising
   the count.
2. **Keep indentation in the anchor** — it is part of the literal.
3. **Prefer `find` over `regex`.** Regexes match things you did not picture;
   literals do not.
4. **For inserts, anchor on the line before and repeat it in `replace`.**
   Replacing `X` with `X\n<new>` is how you insert after `X`.
5. **Leave macro bodies alone.** Do not reindent a `json!` block to fit a new
   wrapper; bind it to a `let` first so the body keeps its indentation.

## Self-test

`scripts/test_sweep.py` covers the behavior the guarantee rests on — ambiguous
anchors refused, missing anchors refused, nothing written when any edit fails,
glob skipping files without a hit. Run it after changing the script:

```bash
uv run python scripts/test_sweep.py
```
