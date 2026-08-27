# Session handoff

How unfinished work moves between working sessions — often between machines,
since gallium is developed against models that only fit on particular hardware.

A **handover document is a work order, not a reference.** It exists to be
discharged and deleted, not to accumulate. The steady state of the repo is
**zero** `HANDOVER-*.md` files.

## The ending session writes it

`docs/HANDOVER-<YYYY-MM-DD>.md`, dated in the filename so a stale one is
obvious. It carries only what the next session needs and cannot reconstruct:

- **changes that landed but were not checked against a running model** — what
  each one alters in the prompt or the wire format, and the exact
  `make testsuite BACKENDS="…"` (or other) command that would verify it
- **open questions that need the other machine or context** — phrased as a
  question with the command that answers it
- **forward-looking blockers** — an upstream PR, a dependency bump, a filed
  issue the work waits on
- **working-tree state not in git** — an uncommitted experiment, a local-only
  commit, a config the reader would otherwise be surprised by

Do not put in it anything already true in the repo: merged PRs (git history),
how a subsystem works (`docs/*.md`, CLAUDE.md), or a design decision
(`docs/adr/`).

Never overwrite or "update" a previous session's handover. If one is still in
the repo, the session that was meant to discharge it did not finish — that is a
signal, not a document to edit.

## The session that picks it up owns discharging it

Reading the handover and carrying on is not enough. The takeover session's work
is not done until the file is gone, and every item in it has landed somewhere
durable:

| Item in the handover | Where it goes |
|---|---|
| Needs tracking beyond this session | a **GitHub issue** |
| A bug with a fix you made | a **PR** |
| "Confirmed against a running model" — outcome, hardware, what came out | [`docs/VERIFICATION_STATUS.md`](VERIFICATION_STATUS.md) |
| How something works, now that it's settled | the relevant `docs/*.md` |
| A responsibility boundary or design decision | CLAUDE.md or `docs/adr/` |
| Already covered by one of the above | nothing — just confirm it and move on |

Then `git rm docs/HANDOVER-<date>.md`, in whichever PR is the natural home for
it (or its own small PR). The commit message says which issues were filed and
which PRs discharged which items, so the deletion is auditable.

If you are a fresh session and a `HANDOVER-*.md` exists, discharging it comes
**before** new work — it is the previous session's unfinished business and it
decays.

## Worked example

The 2026-08-27 handover moved the template-fixture batch (#181, #183–#187,
#189) to the CUDA box for verification against weights. It was discharged by:
issues #118 / #172 comments (findings that needed tracking), PRs #191 (unsloth's
patched Qwen3.8 template) and #192 (#172 verbatim replay), a new
`docs/VERIFICATION_STATUS.md` (the "checked against a running model" outcomes),
and its own deletion in the PR that added that file.
