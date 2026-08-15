---
name: verify-preamble
description: Measure whether a ModelProfile's agent_preamble_suffix actually changes model behavior, by running the same testsuite cases through the same backend with and without it and comparing pass/fail, model-call count, and tool-call count per case from the turn trace. Triggered when the user asks to "verify the preamble", "test the preamble", "measure preamble effect", "compare with and without the preamble", "does the preamble help", "run verify-preamble", or after adding/editing a ModelProfile::agent_preamble_suffix override, before trusting it.
argument-hint: "[profile] [backend] [testcase,testcase,...]  — e.g. gpt-oss gpt-oss-120b arithmetic,capital,coding,file_read,memory_state,needle_in_haystack,refactoring. backend defaults to the testsuite/backends.txt entry whose configs/<backend>.toml resolves to that profile; the testcase list defaults to every testsuite/testcases/* except multimodal_* when that config has no mmprojPath."
allowed-tools: Bash, Read, Edit, Grep, Glob
---

# Verify-preamble

`ModelProfile::agent_preamble_suffix` (`crates/gallium-agent/src/profile/mod.rs`)
is meant to be earned by a *measured* effect on gallium's own testsuite, not by
a guess at what should help — the doc comment on `BASE_AGENT_PREAMBLE` says so
directly, and the GPT-OSS suffix's own doc comment names this exact procedure.
This skill is that procedure: same testcases, same backend, same everything
except the one profile method under test, before vs. after.

**What this measures**: pass/fail, model-call count (ReAct iterations), and
tool-call count per testcase, pulled from `GALLIUM_TRACE_DIR` JSON rather than
eyeballing transcripts — the same fields `docs/adr/0003-model-profiles.md`'s
evidence bar expects.

**What this does not measure**: reply *quality* on a passing case (a plan
reminder could make prose worse while call counts stay identical), and it
does not prove an effect from a single run — see the Guardrails on
non-determinism before reporting a verdict.

## Preconditions

1. **Serial only.** Check for a stray runner/model process before starting —
   this competes for the same GPU/CPU memory as any other testsuite run:
   `ps -ef --forest | grep -E "matrix_runner|runner.sh|gallium" | grep -v grep`.
   If one is running (yours or someone else's), wait for it — don't kill it
   without checking first.
2. The profile under test already has, or you are about to add, an
   `agent_preamble_suffix` override in
   `crates/gallium-agent/src/profile/<family>.rs`. If it doesn't exist yet,
   add it first (with its own doc comment citing what observed behavior it's
   meant to correct — see `deepseek.rs`'s and `gpt_oss.rs`'s for the bar),
   then run this skill to check whether it does what it's meant to.
3. `git status` on that profile file. If it has changes beyond the suffix
   override (something else you're mid-editing), don't stash the whole file
   in Step 4 — hand-edit the suffix out and back instead, so unrelated work
   isn't touched.
4. The backend's model is a multi-GB download — if it isn't already in the HF
   cache, confirm with the user before triggering an unplanned fetch.

## Step 0 — Resolve inputs

If the user didn't name a backend, find one whose `configs/<name>.toml`
resolves to the profile under test — either an explicit `profile = "<name>"`
key, or check the config's `modelPath`/arch against
`crates/gallium-agent/src/profile/<family>.rs`'s `matches_arch`. Prefer the
smallest cached model for the family (faster iteration) unless the user wants
a specific size.

## Step 1 — Pick testcases

```bash
ls testsuite/testcases/
```

Exclude `multimodal_audio`/`multimodal_image` unless the backend's config has
`mmprojPath` set — a family with no projector fails those the same way with
or without the preamble, so they add runtime without adding signal. Everything
else is fair game; the ones that actually exercise tool calls
(`coding`, `file_read`, `refactoring`, `memory_state`) are where a suffix
about plan-following or over-exploration is most likely to show a difference —
don't drop those to save time even if the user's list is short.

## Step 2 — Build and confirm the accelerator is linked

**The binary silently loses CUDA/Metal on a bare `make build`** — on Linux,
`CARGO_FEATURES` only defaults to `cuda` on Windows, so a plain `make build`
after a previous `--features cuda` build quietly relinks a CPU-only binary
with no error, and a CPU-only 100B+-class MoE model turns a 10-second case
into a 100-second one, silently invalidating any timing comparison.

```bash
make build CARGO_FEATURES=cuda   # Linux; plain `make build` on macOS (Metal is automatic)
ldd target/release/gallium | grep -i cuda   # non-empty on Linux = confirmed linked
```

## Step 3 — Write the runner, then run "after" (current code)

One script, reused for both labels so the only thing that differs between
runs is the profile file:

```bash
cat > /tmp/verify-preamble-run.sh <<'SCRIPT'
#!/bin/bash
# Usage: verify-preamble-run.sh <label> <backend> <case1,case2,...>
set -u
label="$1"; backend="$2"; IFS=',' read -ra cases <<< "$3"
scratch="/tmp/verify-preamble"
summary="$scratch/summary-$label.txt"
mkdir -p "$scratch"; : > "$summary"

for tc in "${cases[@]}"; do
    tracedir="$scratch/trace-$label/$tc"
    rm -rf "$tracedir"; mkdir -p "$tracedir"
    export GALLIUM_TRACE_DIR="$tracedir"
    start=$(date +%s)
    if bash testsuite/runner.sh "$tc" "$backend" > "$scratch/log-$label-$tc.txt" 2>&1; then
        result="PASS"
    else
        result="FAIL"
    fi
    dur=$(( $(date +%s) - start ))

    calls=0; tools=0; n_turns=0
    if ls "$tracedir"/*.json >/dev/null 2>&1; then
        for f in "$tracedir"/*.json; do
            n_turns=$((n_turns + 1))
            calls=$((calls + $(jq '.steps | length' "$f")))
            tools=$((tools + $(jq '[.steps[].calls | length] | add // 0' "$f")))
        done
    fi
    echo "$tc result=$result duration_s=$dur turns=$n_turns model_calls=$calls tool_calls=$tools" | tee -a "$summary"
done
SCRIPT
chmod +x /tmp/verify-preamble-run.sh

bash /tmp/verify-preamble-run.sh after <backend> <comma-separated-cases>
```

Run this in the background (`run_in_background: true`) if the model is large
— a 100B+-class MoE case can take tens of seconds each — and wait for the
completion notification rather than polling with `sleep`.

## Step 4 — Get the baseline: temporarily remove the suffix

```bash
git diff crates/gallium-agent/src/profile/<family>.rs   # confirm it's *only* the suffix change
git stash push -- crates/gallium-agent/src/profile/<family>.rs
```

If Step 0's `git status` check found unrelated changes in that file, don't
stash — instead `Edit` the `agent_preamble_suffix` override out (delete the
method or make it return `None`) and remember to `Edit` it back in Step 5.

Rebuild and confirm the accelerator is still linked (Step 2), then:

```bash
bash /tmp/verify-preamble-run.sh before <backend> <same-comma-separated-cases>
```

## Step 5 — Restore

```bash
git stash pop   # or the manual Edit-back, if you didn't stash
make build CARGO_FEATURES=cuda
cargo test --workspace 2>&1 | grep -E "FAILED|test result:|^error"
```

**Do not end this skill with the repo in the "before" state.** The suffix
being tested is the code the user is asking about; leaving it reverted after
a comparison run is the one outcome this skill must never produce by
accident. Confirm the full test suite is green before reporting anything.

## Step 6 — Compare and report

Read both summary files and build one table:

```bash
paste /tmp/verify-preamble/summary-before.txt /tmp/verify-preamble/summary-after.txt
```

| testcase | before | after |
|---|---|---|
| ... | PASS · N calls · M tools · Ts | PASS · N calls · M tools · Ts |

## Guardrails

- **Non-determinism means one run per condition is weak evidence.** Most
  `configs/*.toml` sample at `temperature > 0`. A difference on one testcase
  after a single run each is a candidate, not a finding — flag it and offer
  to repeat that specific case (both labels) 2-3× more before calling it a
  real effect either way. A difference that shows up consistently across
  repeats is real; one that doesn't reproduce is noise, and reporting it as
  the preamble's effect would be exactly the "guess at what should help" the
  suffix hook's own doc comment warns against.
- **Never leave the repo reverted.** Step 5 is not optional cleanup — it's
  the point where a comparison run stops being a comparison run and starts
  being an accidental regression if skipped.
- **Never stash a file with unrelated uncommitted work.** Check first; hand-edit
  instead when the file isn't clean.
- **Don't generalize a result to other families.** A suffix's effect is
  entangled with that family's own template, quantization, and sampling
  settings — a measured effect on `gpt-oss-120b` says nothing about whether
  the same suffix text would do anything on `gemma4` or `deepseek-v4-flash`.
  Each family earns its own suffix through its own run of this skill.
- **Serial only** — same reasoning as every other testsuite invocation in
  this repo (see `eval-improve`'s identical guardrail).
- This skill reports; it doesn't decide. Whether an observed difference is
  an improvement, a regression, or noise is the user's call — present the
  numbers and, where relevant, note which reading you'd lean toward and why,
  but don't unilaterally add, remove, or edit the suffix based on the result
  without asking.

## Output format

End with the comparison table from Step 6, plus:

```
## Verdict

<one sentence: no observed difference / a difference of N calls on case X that
needs repeat runs to confirm / a consistent difference across R repeats on case X>

## Repo state

Restored to the "after" (suffix present) state — confirmed via
`cargo test --workspace` (N passed, 0 failed).
```
