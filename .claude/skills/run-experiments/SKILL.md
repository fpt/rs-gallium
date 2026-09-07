---
name: run-experiments
description: Run a measurement sweep or A/B for the ../gallium-research paper — a cache-budget sweep, a policy-grid cell (cpuMoe, fused on/off, device), or any other mechanism gallium-research's notes call for — verify the result is trustworthy before trusting it, and land it in ../gallium-research following that repo's own evidence-recording rules. Triggered when the user asks to "run experiments", "run a sweep", "collect evidence for the paper", "run the run brief", "measure <mechanism>", "fill the policy grid", "close a data gap", or names a task from gallium-research's notes/04-experiment-plan.md or notes/06-run-brief-*.md.
argument-hint: "[task] — e.g. 'R4' or 'T0.4' (from the current run brief / experiment plan), or a free-form spec like 'GALLIUM_EXPERT_CACHE_BYTES sweep 0-6GiB on gemma4-26b-candle, refactoring'. Defaults to the next open item in gallium-research's current notes/06-run-brief-*.md for this host, or notes/04-experiment-plan.md if no run brief names one."
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# Run-experiments

`../gallium-research` (sibling repo, private) is a systems paper drawn from
this codebase — see this repo's own `CLAUDE.md` under "Research findings live
in `../gallium-research/`". This skill is the procedure that repo's own
`CLAUDE.md` describes under "Adding a result", made concrete: run something
under `notes/02-measurement-protocol.md`'s rules, check the number is real
before writing it down, then write it down the way that repo requires.

**This is not `eval-improve`** (testsuite correctness → filed issues → PRs
against this repo) **and not `verify-preamble`** (one specific before/after on
one profile method). This skill measures a *mechanism* — a cache, a routing
policy, a fused kernel, whatever gallium-research's notes are asking for next —
and the destination for what it finds is the other repo, not this one.

**Two measurement shapes**, matching what already exists:

- **Shape A — fixed-shape microbenchmark sweep.** An `#[ignore]`d integration
  test with its own timing (`Instant`, printed decode tok/s) driven by env
  vars, run once per sweep point. `gemma4_26b_gguf_fused_decode_speed` is the
  only one that exists today (`crates/gallium-models/tests/integration.rs`) —
  grep for `#[ignore` near `speed`/`matvec`/`decode` before assuming a given
  model has one; if it doesn't, the sweep has to go through Shape B instead,
  reading decode rate from the trace rather than a test's own `eprintln!`.
- **Shape B — agent-workload A/B.** Toggle one env var, run
  `testsuite/runner.sh <testcase> <backend>` (or the matrix runner for more
  than one case), and read wall time, pass/fail, and `GALLIUM_TRACE_DIR`'s
  per-turn timing. This is how `cpuMoe` cells and anything without a dedicated
  speed test get measured.

Both need VRAM sampled alongside them — gallium is right not to instrument
this itself (host facts, not model facts), so this skill's wrapper scripts do
it with `nvidia-smi` running in the background for the run's duration.

## Preconditions

1. `../gallium-research` exists and is a git repo with a clean-enough working
   tree that this session's writes won't get tangled with someone else's
   uncommitted work — `git -C ../gallium-research status --short`. If it's
   dirty with unrelated changes, ask before proceeding rather than committing
   over them.
2. **Read `../gallium-research/CLAUDE.md` and
   `../gallium-research/notes/02-measurement-protocol.md` in full before doing
   anything else.** They are authoritative; if either has changed since this
   skill was written, the notes win and this skill is stale. Everything below
   operationalizes M1–M5 of the protocol — it does not restate them in full.
3. **Serial only.** Check for a stray runner/model process before starting —
   this competes for the same GPU/CPU memory as any other testsuite or
   integration-test run: `ps -ef --forest | grep -E "matrix_runner|runner.sh|gallium|cargo test" | grep -v grep`.
   If one is running (yours or someone else's), wait — don't kill it without
   checking first.
4. The model(s) involved are multi-GB downloads — if not already cached
   (`~/.cache/huggingface/hub/`), confirm with the user before triggering an
   unplanned fetch.

## Step 0 — Resolve what to measure

If the user gave a task id (an `R<n>` from a `notes/06-run-brief-*.md`, or a
`T0.x`/`T1.x` from `notes/04-experiment-plan.md`) or a free-form spec, use it.
Otherwise: read the current run brief for this host if one exists
(`notes/06-run-brief-<host>.md` — `4070` for this box's RTX 4070, check for
others if this runs elsewhere) and take its next item not yet marked done;
if no run brief covers this host, fall back to `notes/04-experiment-plan.md`'s
T0 items in order, then T1.

**Whatever is chosen must trace back to a row in `notes/00-claims.md`'s
claim → evidence table or a `\gap{}` in `paper/sections/*.tex`.** A
measurement that doesn't close or advance one of those is not what this
skill is for — flag it to the user as a possible new claim/task to add to
the experiment plan first, rather than taking a one-off reading gallium-
research's own structure has nowhere to put.

Identify which shape (A or B) applies: grep
`crates/gallium-models/tests/integration.rs` for an `#[ignore]`d test whose
name matches the model and mechanism; if none exists, it's Shape B.

## Step 1 — Host and build stamps

```bash
lscpu | grep -E "Model name|CPU\(s\):" | head -2
free -h | grep Mem
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git log -1 --format='%H'
grep 'candle-core' Cargo.toml | grep rev
grep 'llama-cpp-2' crates/gallium-agent/Cargo.toml | head -1
```

Write or update `../gallium-research/data/host_<id>.json` and
`../gallium-research/data/build_<date>.json` (schema: whatever fields
`02-measurement-protocol.md` §M1's `host:`/`build:` rows list). Reuse an
existing stamp file from today if the build hasn't changed since — don't
create a duplicate per sweep.

## Step 2 — Build and confirm the accelerator is linked

**The binary silently loses CUDA/Metal on a bare `make build`** — on Linux,
`CARGO_FEATURES` only defaults to `cuda` on Windows, so a plain `make build`
after a previous `--features cuda` build quietly relinks a CPU-only binary
with no error, turning every number in the sweep into a CPU measurement
mislabeled as a CUDA one.

```bash
make build CARGO_FEATURES=cuda   # Linux; plain `make build` on macOS (Metal is automatic)
ldd target/release/gallium | grep -i cuda   # non-empty on Linux = confirmed linked
```

For Shape A, also confirm the test compiles with the accelerator feature:

```bash
cargo test --release -p gallium-models --test integration \
    --features gallium-core/cuda <test_name> -- --ignored --nocapture --list
```

## Step 3 — Warm state, confirm idle

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should be near 0 before you start
```

Warm the filesystem cache for every GGUF the sweep touches, once, before
timing anything (M2 §1 — the "second arm reads a file the first paged in"
trap is the same whether the two arms are two env-var values in one loop or
two whole process invocations):

```bash
cat <path-to-gguf> > /dev/null
```

## Step 4 — Reproduce a known number before sweeping a new one

**Do this before varying anything.** Find the closest existing measurement in
`../gallium-research/notes/01-evidence.md` for the same model/mechanism and
reproduce it exactly — same prompt length, same generation length, same
temperature, same cache budget if one is baked into the config. If it
reproduces within a few percent, proceed to Step 5. If it does not:

**Stop and diff every parameter the existing entry's provenance line names
against what you just ran, before writing the gap down as drift.** This is
not a formality — the first time this skill's procedure was run by hand
(2026-09-07), the first sweep reading looked like an 11% regression against
the documented headline, and the actual cause was a generation-length
mismatch (`GALLIUM_KVTEST_GEN=96` documented, `48` the test's own default,
never set explicitly) — a one-line setup difference that a "must be noise"
shrug would have buried in the dataset instead of fixing. Check, in order:
prompt length / filler reps, generation length, temperature and seed, which
arm ran first (cold-cache order), and whether the config bakes in a second
policy knob (e.g. `expertCacheBytes` alongside `cpuMoe`) that the mechanism
under test doesn't actually consult — confirm that last one from the *source*
(grep where the env var is read and what gates it), not from assumption.

Only once every documented parameter matches and a gap still exists is it a
real finding — and a real finding is exactly what `../gallium-research`'s
`CLAUDE.md` says to record as both numbers with what differed, never to
silently overwrite.

## Step 5 — Run the sweep or grid

**One process per point — never a loop inside one process.** Two conditions
sharing a process share whatever the first one paged into the filesystem
cache or warmed on the GPU, which is exactly M2 §1's trap. A test that itself
runs two arms internally (like `gemma4_26b_gguf_fused_decode_speed`'s
fused-off-then-on) is fine to use as one *point*, since both its arms are
part of the documented method — just don't also loop *that* across budgets in
one process.

**Shape A wrapper** (fixed-shape microbenchmark, one point per invocation):

```bash
cat > /tmp/run-experiments-point-a.sh <<'SCRIPT'
#!/bin/bash
# Usage: run-experiments-point-a.sh <test_name> <label> <outdir> [ENV=val ...]
set -uo pipefail
TEST="$1"; LABEL="$2"; OUTDIR="$3"; shift 3
VRAM_LOG="$OUTDIR/vram_${LABEL}.csv"; : > "$VRAM_LOG"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG"; sleep 0.2; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT
env "$@" GALLIUM_DEVICE=cuda \
  cargo test --release -p gallium-models --test integration --features gallium-core/cuda \
  "$TEST" -- --ignored --nocapture > "$OUTDIR/log_${LABEL}.txt" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null || true; trap - EXIT
PEAK=$(sort -n "$VRAM_LOG" | tail -1)
echo "peak_vram_mib=$PEAK rc=$RC" >> "$OUTDIR/log_${LABEL}.txt"
grep -E "tok/s|fused|decode" "$OUTDIR/log_${LABEL}.txt" || true
if [ "$RC" -ne 0 ]; then
  echo "  FAILED (rc=$RC) — see $OUTDIR/log_${LABEL}.txt; this point is not usable evidence" >&2
else
  echo "  peak VRAM: ${PEAK} MiB"
fi
exit "$RC"
SCRIPT
chmod +x /tmp/run-experiments-point-a.sh

mkdir -p /tmp/run-experiments/sweep
# The exit code now reflects the point's own pass/fail (Step 6's correctness
# gate) — chain with && / check $? rather than only reading the printed line.
bash /tmp/run-experiments-point-a.sh <test_name> <label> /tmp/run-experiments/sweep <ENV_VAR>=<value>
# repeat per sweep point, e.g.:
#   bash ... p0 ... GALLIUM_EXPERT_CACHE_BYTES=0
#   bash ... p1 ... GALLIUM_EXPERT_CACHE_BYTES=1073741824
```

**Shape B wrapper** (agent-workload A/B via the testsuite):

```bash
cat > /tmp/run-experiments-point-b.sh <<'SCRIPT'
#!/bin/bash
# Usage: run-experiments-point-b.sh <testcase> <backend> <label> <outdir> [ENV=val ...]
set -uo pipefail
CASE="$1"; BACKEND="$2"; LABEL="$3"; OUTDIR="$4"; shift 4
TRACE_DIR="$OUTDIR/trace_${LABEL}"; mkdir -p "$TRACE_DIR"
VRAM_LOG="$OUTDIR/vram_${LABEL}.csv"; : > "$VRAM_LOG"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG"; sleep 0.3; done ) &
SAMPLER=$!
START=$(date +%s.%N)
env "$@" GALLIUM_TRACE_DIR="$TRACE_DIR" \
  bash testsuite/runner.sh "$CASE" "$BACKEND" > "$OUTDIR/log_${LABEL}.txt" 2>&1
RC=$?
END=$(date +%s.%N)
kill $SAMPLER 2>/dev/null || true; wait $SAMPLER 2>/dev/null || true
PEAK=$(sort -n "$VRAM_LOG" | tail -1)
WALL=$(echo "$END - $START" | bc)
RESULT=$(grep -oE '(PASS|FAIL): [a-z_]+ × [a-z0-9._-]+' "$OUTDIR/log_${LABEL}.txt" | tail -1)
echo "[$LABEL] $CASE x $BACKEND -- $RESULT -- wall ${WALL}s -- peak VRAM ${PEAK} MiB (rc=$RC)"
echo "wall_s=$WALL peak_vram_mib=$PEAK rc=$RC result=$RESULT" >> "$OUTDIR/log_${LABEL}.txt"
# Propagate failure: runner.sh's own exit code (a CLI crash) OR a result line
# that says FAIL (the testcase's check.sh rejected the output) both mean this
# point is not usable evidence (Step 6) -- a caller chaining with && must see
# that, not just a logged "result=FAIL" a later step could skim past.
if [ "$RC" -ne 0 ] || [[ "$RESULT" == FAIL:* ]]; then
  echo "  FAILED -- see $OUTDIR/log_${LABEL}.txt; this point is not usable evidence" >&2
  exit 1
fi
exit 0
SCRIPT
chmod +x /tmp/run-experiments-point-b.sh

mkdir -p /tmp/run-experiments/grid
# Always LLM_TEMPERATURE=0 for a speed/exactness A-B (M2 §2) unless the
# question is explicitly about robustness under sampling. Exit code reflects
# pass/fail -- chain with && / check $? rather than only reading the printed
# line.
bash /tmp/run-experiments-point-b.sh <testcase> <backend> <label> /tmp/run-experiments/grid \
  LLM_TEMPERATURE=0 <ENV_VAR>=<value>
```

**Verify the trace was actually produced**, not just that the exit code was
0 — `ls "$TRACE_DIR"` should show at least one `turn-*.json`. An empty trace
directory after a PASS means tracing wasn't wired up for this invocation
(a stale `GALLIUM_TRACE=0` in the environment beats `GALLIUM_TRACE_DIR`, per
`trace.rs`'s own precedence — see its doc comment), not that there's nothing
to show.

Run large-model points in the background (`run_in_background: true`) and wait
for the completion notification rather than polling with `sleep`.

## Step 6 — Verify before writing anything down

- **Repeat at least one point** in a fresh process and compare — M2 §9 asks
  for periodic baseline re-runs; a whole new sweep doesn't need every point
  repeated, but zero repeats means the spread is simply unknown. Report the
  spread, don't just assert it's fine.
- **A Shape B point that FAILed its `check.sh` is not usable evidence** (M5) —
  a fast-and-wrong configuration doesn't get a speed number, it gets
  investigated. Don't average or report timing from a failing run.
- **An "n/a" or "no effect" reading needs a source-level reason, not just a
  null measurement.** If a sweep parameter appears to do nothing (a budget
  that never changes VRAM, a flag that never changes output), find *why* in
  the code — a wired-off path (grep where the env var is read and what
  gates it) is a different, stronger finding than "measured no difference
  and moved on." Confirmed-by-type beats confirmed-by-absence.
- Confirm the GPU is idle and no process was left running:
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` should read
  near 0 once the sweep finishes.

## Step 7 — Land it in `../gallium-research`

Follow that repo's `CLAUDE.md` "Adding a result" procedure exactly:

1. Confirm it was taken under the protocol (Steps 1–6 above); if a deviation
   was unavoidable, say so in the entry rather than hiding it.
2. Append a new `E<n>` heading to `notes/01-evidence.md`: what question it
   answers, the table, a provenance line (host, build sha, cargo features,
   config, protocol deviations), and — explicitly — **what it does not
   show**. If Step 4 caught a real discrepancy against an existing entry,
   record both numbers and what differed; do not overwrite the old one.
3. Update the claim → evidence table in `notes/00-claims.md`.
4. If it closes a `\gap{}`, remove the marker in the relevant
   `paper/sections/*.tex` and write the sentence it was standing in for; if
   it only partly closes one, narrow the `\gap{}`'s text to what's actually
   still missing rather than deleting it.
5. Update `notes/04-experiment-plan.md`'s task status and, if a
   `notes/06-run-brief-*.md` named this task, mark it done there with a
   pointer to the new `E<n>` section — and add a short "not done this
   session" note for whatever else that brief still has open, so the next
   session doesn't have to re-derive it.

Raw logs, per-run VRAM samples, and trace JSON go to
`data/raw/<date>-<host>/` (gitignored, per `data/README.md`); the reduced
table(s) a figure would be built from go to `data/*.csv` (committed) with a
`trial_id` per row traceable back to the raw files.

## Step 8 — Commit

```bash
cd ../gallium-research
git add -A
git commit -m "$(cat <<'EOF'
<one line: what was measured and which task it closes>

<what was run, what it found, what it does/doesn't close — mirror the E<n>
section's own summary rather than restating the diff>

EOF
)"
```

Use whatever commit-message attribution trailer this session's own
instructions specify (it varies by model/session — don't hardcode one here).

**Do not push without asking** — same rule as this repo's own git
conventions (`git remote -v` will show `gallium-research` has its own
`origin`; a commit here is a separate decision from a push, same as
anywhere else).

## Guardrails

- **Serial only** — never run two `runner.sh`/`matrix_runner.sh`/`cargo test
  --test integration` invocations concurrently; they compete for the same
  GPU/CPU memory (same reasoning as `eval-improve`'s identical guardrail).
- **One process per sweep point.** A loop that varies an env var inside one
  `cargo test` invocation or one `runner.sh` call shares warm state across
  points and silently biases every point after the first.
- **Never trust a discrepancy against existing evidence without diffing the
  setup first** (Step 4). "Session-to-session variance" is a real category
  in `02-measurement-protocol.md` §M2 rule 9, but it is the conclusion you
  reach *after* checking prompt length, generation length, temperature,
  seed, arm order, and baked-in config knobs all match — not the first
  explanation reached for.
- **Never write a number you did not measure or transcribe** — this repo's
  `../gallium-research/CLAUDE.md` rule, restated because it is the one this
  skill exists to make easy to follow under time pressure: an interpolated,
  estimated, or "should be about" figure is a `\gap{}`, not a table entry.
- **A result that contradicts `01-evidence.md` is itself the finding.**
  Record both numbers and what differed; overwriting the old one destroys
  the evidence that something changed (a regression, a fixed bug, a
  methodology correction) rather than reporting it.
- **Correctness gates every Shape B point** — a configuration that answers
  fast and wrong is not a result (M5); if a point fails, that's a Phase-2-
  style investigation (see `eval-improve`), not a number to report with an
  asterisk.
- **This skill measures; it doesn't argue.** Whether a measured effect is
  large enough to matter for the paper's claims is `00-claims.md`'s call —
  update the table's status honestly (`have` / `partial` / `missing`) and
  let the writing-plan review passes in `notes/05-writing-plan.md` decide
  what it means for the draft.

## Output format

End with:

```
## What was measured
<task id / free-form spec, shape A or B, mechanism, model(s), testcase(s)>

## Result
<the table, or a pointer to the new E<n> section>

## Verification
<repeat-run spread; correctness gate result for Shape B; any discrepancy
against existing evidence and how it was resolved or flagged>

## Landed in ../gallium-research
<files touched: 01-evidence.md §E<n>, 00-claims.md, 04-experiment-plan.md,
run-brief status, paper/sections/*.tex \gap{} changes, data/*.csv, commit sha>

## Still open
<what this task's brief/plan entry still doesn't close, so the next session
doesn't have to re-derive it>
```
