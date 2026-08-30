---
name: eval-improve
description: Run gallium's testsuite, root-cause each failure from its turn trace and raw model output, and produce a filed issue plus a verified fix plus a PR for the ones that are wire-layer (model profile) bugs or system-prompt gaps — naming, not fixing, failures that are out of scope (model/candle numerics, hardware-specific capacity, an already-documented limitation). Triggered when the user asks to "run eval-improve", "improve profiles from testsuite failures", "run the eval loop", "find and fix testsuite failures", or when invoked on a schedule via /loop.
argument-hint: "[backend,backend,...]  — optional comma list to scope one cycle; default is the full portable local matrix (see Phase 1)"
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, ScheduleWakeup
---

`Bash` covers every `git`/`gh` call in Phases 0, 3, and 6 (branch, push, `gh
issue`/`pr` create and list) — there's no separate GitHub-integration tool to
declare. `ScheduleWakeup` is declared for the `/loop` pacing described below;
outside `/loop` this skill never calls it.

# Eval-improve

Runs the loop this repo's own history already did by hand, six times, across
`docs/adr/0003-model-profiles.md`'s landing order: run the testsuite, find a
failure, read its raw model output and reason out whether it's a bug in a
shared parser, a gap in a system prompt, or neither — then, only for the
first two, file an issue, fix it, re-verify against the full matrix, and open
a PR that references the issue. `gallium#118`, `#123`, `#124` are what "named
but not fixed" looks like when the finding is real but out of scope; `#119`
→ PR #125, `#120` → PR #126, `#121` → PR #127 are what the fixed half looks
like, issue to merged PR. Read one of each before running this the first
time — they're the calibration for both halves of the classification in
Phase 2.

**The shape is Issue → Fix → Verify → PR, in that order, every time.** The
issue is the checkpoint a human can veto *before* any code changes exist —
skipping straight to a fix because the diagnosis feels obvious is exactly the
failure mode this skill exists to avoid, and it's proven to matter more than
once: the LFM2 `wire::json` gap looked simple until reproducing it showed the
proposed fix in ADR 0003 itself wouldn't have covered either real failure
(`docs/adr/0003-model-profiles.md`'s step-3a amendment). Diagnose fully,
write it down as an issue, *then* touch code.

## Running standalone vs. under /loop

Standalone: run once, do what Phase 1 finds, stop.

Under `/loop` (no fixed interval — self-paced, since a cycle's length varies
from "zero failures, nothing to do" to "a 95GB model's full matrix plus a
fix"): every wake starts at Phase 0. Pace the next `ScheduleWakeup` by what
happened this cycle:

- A fix just landed and needs its regression gate re-checked, or the matrix
  is mid-run: short delay (a few minutes), `noop: false`.
- An issue was filed but the PR isn't ready yet within this cycle's budget:
  short delay, continue next wake.
- A PR from this skill is open and awaiting human review (Phase 0 found
  one): long delay (30–60 min) — nothing productive happens until a human
  acts, and re-checking every few minutes is just noise. `noop: true` if nothing
  else changed this cycle.
- Zero failures found in Phase 1: long delay, `noop: true`.

## Phase 0 — Don't duplicate work already in flight

Every branch and PR this skill creates is named `eval-improve/<slug>` —
check that prefix, not every open PR (a human's unrelated WIP isn't yours to
block on or touch):

```bash
gh pr list --state open --json headRefName,number,title \
  -q '.[] | select(.headRefName | startswith("eval-improve/"))'
```

If one exists: **stop here for this cycle.** Report its number and status
(pass/fail on its own CI or review state if visible) and end the turn — don't
start a second investigation while the first is awaiting review. This is the
loop's own idempotency gate; skipping it is how a looped run ends up filing
the same finding twice or opening two PRs that fight over the same file.

Then, before Phase 3 later files anything new, check existing issues so a
re-run doesn't refile a known gap:

```bash
gh issue list --state open --limit 50
```

## Phase 1 — Triage: run the matrix

The regression gate is **llama.cpp only, whatever accelerator this host
has** — not CUDA-specific, since this may run on a Mac (Metal, automatic) or
this project's Linux dev box (CUDA, needs `--features cuda` — see the
Guardrails below, this exact silent-CPU-fallback trap has bitten this
project's own sessions before). Excluded:

- **candle backends** (`gemma4-candle`, `lfm2-candle`) — correct to run
  during a *targeted* investigation of a candle-specific finding, but too
  slow (candle is CPU-only in this repo's build, ~5 tok/s) to sit in a loop's
  regression gate.
- **`openai`** — cloud, not a local wire-layer/prompt target this skill can
  act on.

The local configs are each tuned for one of this project's two reference
machines (RTX 4070 12GB, or a 24GB M3 — see
`.claude/skills/model-viability/SKILL.md`'s Step 6 and
`docs/VERIFICATION_STATUS.md`). On unfamiliar hardware a `gpuLayers` /
`cpuMoe` value may be wrong — treat a context-creation failure there as a
hardware-capacity result, not a bug (see Phase 1's failure buckets).

```bash
BACKENDS=$(grep -vE '^\s*#|^\s*$' testsuite/backends.txt | while read -r b; do
    case "$b" in
        *-candle|openai) continue ;;
    esac
    echo "$b"
done | paste -sd,)

# Confirm the binary actually matches this host's accelerator before trusting
# a single result from this run — see Guardrails.
CLI="$(pwd)/testsuite/gallium_cli.sh" BACKENDS="$BACKENDS" bash testsuite/matrix_runner.sh
```

This is the **full** matrix per that filtered list — large models included
(`gpt-oss-120b`, `minimax-m2`, `deepseek-v4-flash`), so budget real time for
it. That's why pacing is self-paced under `/loop` rather than a fixed short
interval.

Read the result file's matrix table. Zero failures: nothing to do this
cycle — report and stop (or, under `/loop`, schedule a long wakeup). One or
more failures: record each `(testcase, backend)` pair and move to Phase 2.

## Phase 2 — Root-cause each failure

The matrix run's own log only shows the first line of the model's reply —
not enough to diagnose. Re-run each failing pair **individually**, with
tracing on, exactly the way this repo's own sessions have done it by hand:

```bash
mkdir -p /tmp/eval-improve/<testcase>-<backend>
cp testsuite/testcases/<testcase>/* /tmp/eval-improve/<testcase>-<backend>/
cd /tmp/eval-improve/<testcase>-<backend>
prompt_stream="$(grep -vE '^\s*#' prompt.txt | grep -vE '^\s*$'; echo '/quit')"
echo "$prompt_stream" | RUST_LOG=debug GALLIUM_TRACE_DIR=./trace \
    <repo>/target/release/gallium --config <repo>/configs/<backend>.toml \
    > out.txt 2> err.txt
```

Read `out.txt`/`err.txt` for the raw decoded model output (`CandleProvider
raw output:` / the equivalent llama.cpp debug line) and `./trace/*.json` for
the full turn — prompt, every tool call attempted, every result. This is the
evidence a filed issue must cite; don't guess from the truncated matrix log.

**Classify against three buckets — this is the judgment call the whole skill
exists to make carefully, so don't skip straight to "looks like a bug":**

1. **Profile bug** — the model's raw output is well-formed *for its own
   family's wire format*, but the shared parser (`crate::gemma`,
   `crate::harmony`, `profile::wire::*`) misreads it. The tell: you can point
   at a specific function and a specific input where its own stated logic
   does the wrong thing. Reference case: `gallium#123` — an ordinary `"`
   where Gemma's `<|"|>` was expected made `scan_call_body`'s deliberate
   "unterminated quote consumes the remainder" rule (correct for a
   genuinely cut-off value) swallow the call's own closing brace instead.
2. **System-prompt gap** — the model's output is syntactically fine and
   parses correctly, but is the *wrong* content: refuses a task it
   shouldn't, calls the wrong tool, ignores an instruction the system prompt
   should have made unambiguous. Fixable by editing the prompt text
   (`configs/*-system-prompt.md`), not by touching a parser.
3. **Out of scope** — neither of the above. Concretely:
   - **Model/engine numerics** — the model (or `gallium-models`' candle
     implementation) samples something outright wrong that no text-level
     fix addresses. Reference case: `gallium#124`, one character wrong
     (`782}` vs `7823`) with nothing in the decode/clean-reply path that
     could produce that from a correct generation.
   - **Hardware capacity** — an OOM or a context-creation failure tied to
     *this machine's* VRAM/RAM, not to gallium's code. Don't file this as a
     bug; note it and move on. Configs are tuned for a 12GB CUDA card or a
     24GB M3; on anything else a `gpuLayers` / `cpuMoe` value may not fit.
   - **Already documented** — grep `CLAUDE.md` and the open-issues list from
     Phase 0 before concluding something is new. The gemma4-12b
     `multimodal_audio` "Zukiu" mistranscription and the 26B-A4B projector's
     `audio=false` are both *expected*, documented in CLAUDE.md's Multimodal
     input section — filing either as a fresh finding would be re-discovering
     a known fact, not root-causing anything.

If a failure doesn't clearly sort into one bucket, don't force it — report
your uncertainty and ask rather than guessing (see Guardrails).

## Phase 3 — File the issue (buckets 1–2 only)

Skip filing if Phase 0 already found an equivalent open issue — reference it
in Phase 6's PR instead. Otherwise, file with the same evidence bar as the
reference issues above:

```
## The problem
<one or two sentences>

## Repro
<exact command from Phase 2, exact raw output/trace excerpt>

## Root cause
<file:line, and what its logic does on this input>

## Why not <the other plausible bucket>
<the reasoning that ruled it out — this is what makes the classification
checkable by someone else, not just asserted>

## Reference
<ADR/CLAUDE.md section, related issue numbers, testcase/backend>
```

Label `bug` for profile bugs, leave unlabeled or use whatever this repo's
convention is for prompt-only gaps (check `gh label list` — don't invent a
new label without checking first).

## Phase 4 — Fix

Branch: `eval-improve/<short-slug-of-the-issue>`.

- **Profile bug**: edit the Rust in `crates/gallium-agent/src/profile/` (or
  the shared `crate::gemma`/`crate::harmony` modules) and add a unit test
  reproducing the exact input from Phase 2's repro — every existing profile
  fix in this repo's history did this, and it's what makes the fix
  independently checkable later. Re-read `CLAUDE.md`'s "Model profiles"
  section before touching anything shared: reasoning is stripped **once**,
  before any format is tried; the native format is tried **before** the JSON
  scan; a family reads only its own formats. These are invariants a
  seemingly-local fix can violate by accident.
- **System-prompt gap**: edit the relevant `configs/*-system-prompt.md`.
  There's no compiler for prose — Phase 5's full-matrix re-run is the only
  thing that catches a wording change fixing one testcase while breaking
  another on the same model, so don't skip or shrink it for this fix type.

One root cause per PR — don't bundle an unrelated second fix just because
you're already in the file.

## Phase 5 — Verify

1. Re-run the **specific** failing `(testcase, backend)` pair from Phase 2 —
   fast signal, confirms the fix actually does something.
2. Re-run Phase 1's **full filtered matrix** again.
3. Compare against Phase 1's own baseline results, pair by pair — the bar is
   **no new failures**, not 100% green. A fix is not required to also
   resolve unrelated pre-existing out-of-scope failures it wasn't targeting
   (that would silently expand scope past what the issue described).

If step 2 shows a new failure anywhere else, the fix touched a shared parser
too broadly — go back to Phase 4, not around this step.

## Phase 6 — PR

Commit message explains *why*, not *what* (the diff already shows what).
Reference the issue with `Fixes #N`. PR body's test plan lists exactly what
was re-run in Phase 5, backend by backend — not just "tests pass."

```bash
git push -u origin eval-improve/<slug>
gh pr create --title "..." --body "$(cat <<'EOF'
## Summary
...

## Test plan
- [x] <specific testcase/backend re-run from Phase 5, with result>
- [x] Full filtered matrix re-run: no new failures vs. baseline

Fixes #<N>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Guardrails

- **Serial only.** Never run two `matrix_runner.sh`/`runner.sh` invocations
  concurrently — they compete for the same GPU/CPU memory loading multi-GB
  models. Check for a stray `gallium`/`runner.sh`/`matrix_runner.sh` process
  before starting *any* phase that runs the binary:
  `ps -ef --forest | grep -E "matrix_runner|runner.sh|gallium" | grep -v grep`.
  If the user or a prior cycle left one running, wait for it — don't kill
  something you didn't start without checking first.
- **The binary silently loses CUDA on a bare `make build`.** On Linux,
  `CARGO_FEATURES` only defaults to `cuda` on Windows; a plain `make build`
  after a previous `--features cuda` build quietly relinks a CPU-only
  binary with no error. Before trusting Phase 1's results, confirm the
  binary matches expectations: `ldd target/release/gallium | grep cuda` on
  Linux (non-empty = CUDA linked), or just always rebuild explicitly —
  `make build CARGO_FEATURES=cuda` on this project's Linux box, plain
  `make build` on macOS (Metal is automatic there, no flag needed).
- **A classification you're not confident about is not a filed issue.**
  Report the uncertainty and ask, the same way this skill's own design
  discussion resolved open questions rather than guessing at them. Guessing
  wrong here means either a false "bug" report against a model-quality
  issue, or a real bug quietly waved through as "out of scope."
- **Never fix a shared parser without the full Phase 5 gate.**
  `crate::gemma`, `crate::harmony`, `profile::wire::*` are read by every
  family that shares them — this is the exact failure class ADR 0003 exists
  to prevent, and skipping the regression gate on a "small" shared-code
  change is how it comes back.
- **Never bypass Phase 0's dedup check**, even for a standalone
  (non-`/loop`) invocation — a human may have started a cycle by hand and
  left a PR open.
