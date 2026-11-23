---
name: property-test
description: Write property-based tests (proptest) for gallium's pure logic and prove they catch bugs by planting mutations. Triggered when the user asks to "write a property test", "add PBT", "test this with proptest", "check this invariant on random inputs", "run the mutation check", or to pick up the next candidate in docs/FORMAL_FIT.md.
argument-hint: "[target]  — e.g. profile/wire/python.rs, StreamingReply::advance"
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
---

# Property tests for gallium

Write property-based tests (PBT) for gallium's pure logic with
[proptest](https://github.com/proptest-rs/proptest), then prove the tests can
catch bugs by planting mutations. No Lean, no model checking: gallium's pure
targets are string-structural or bookkeeping, and a property against a
reference model gets nearly all of the value.

## Where it fits

Use PBT on code that is **pure and has a statable invariant** — wire parsers,
reply cleaning and streaming, compaction, budget ledgers, small resolvers.
[docs/FORMAL_FIT.md](../../../docs/FORMAL_FIT.md) is the ranked list of
candidates with their evidence; start there and update it as targets are done.

Do not force PBT onto:

- **Concurrency and reconnect protocols** — `SteerInbox` close/push,
  app-server displacement (`cancel_turns` / `adopt_predecessor`), the
  responses-api generation lock. Function-level tests do not find design errors
  there; that is where gallium's ordering bugs (#166) came from. Record them in
  `FORMAL_FIT.md` as model-checking (TLA+ / loom) candidates and move on.
- **Numerics against a reference** — candle kernels, MXFP4. Those are compared
  against llama.cpp / ollama / an f32 reference (`ollama-narrow`,
  `scripts/layer_diff.py`), not against a property.
- **Code where most of the work is I/O or FFI** (`llm_local.rs`'s KV reuse is
  tangled with llama.cpp contexts) — propose extracting a pure core first.

To find new candidates, read fix history by path. Most gallium fixes are titled
`<area>: <what went wrong>` without the word "fix", so read subjects rather than
counting keywords:

```bash
git log --format='%h %ad %s' --date=short -- <path>
```

## Steps

1. **State the properties first.** One sentence each, numbered per target with
   a letter (`C1`, `A1`, `V1`, …). The same number goes in the test's doc
   comment, in the `prop_assert!` message (`"C2"`), and in `FORMAL_FIT.md` — a
   failure then names the rule it broke. If you cannot write a property, the
   spec is unclear: go to step 6 before writing code.

2. **Write the generator.**
   - **Location:** a `#[cfg(test)] mod pbt` at the end of the target's own file,
     beside its existing `mod tests`. Most targets are private or `pub(crate)`
     (`memory` is a private module), so an integration test in `tests/` cannot
     reach them. Generators live in the same `mod pbt`.
   - **Shape inputs the way production produces them**, and cite the producer:
     a history is built the way `react.rs` appends it (user, assistant with
     calls, one result per call, steers as user messages mid-turn).
   - **Raise the odds of boundaries on purpose** with weighted `prop_oneof!` —
     uniform random almost never hits them: `0`, `u32::MAX`, exactly the
     threshold and one under it, an empty history, a system message mid-history.
   - **Boundaries that depend on state** (exactly what is left in a budget)
     cannot be drawn up front. Generate a *relative* size and resolve it when
     the step runs — `Size::Available(-2..=2)` in `vram::pbt` hits "exactly
     fits" and "one byte over" on every run.
   - **Make values traceable.** Tag every generated message body uniquely
     (`#7:xxx`) so a property can follow a message through the code by content.
   - List the boundaries the generator targets in its doc comment.

3. **Write the properties** inside `proptest! { … }`.
   - **No self-reference.** Never compute the expected value with the function
     under test. Write the rule directly from the doc comment — a reference
     model (`vram::pbt::Model`), a pairing checker, an integer formula
     (`w * 9 / 10` for the 90% trigger rather than reusing the `f64` constant).
   - **Stateful code:** replay a random `Vec<Op>` against the real object and
     the reference model, and compare *every observable* after *every step*.
   - **Order-independence:** `Just(v).prop_shuffle()` and compare results.
   - Inputs outside a precondition: `prop_assume!`; if many are rejected, fix
     the generator instead. A property vacuous for an input should just skip
     that assertion (`if let … { prop_assert!(…) }`).
   - Behavior nobody has decided on is **excluded** from the property and pinned
     in a separate test (step 6).

4. **Run it.** Default 256 cases, then many more:

   ```bash
   cargo test -q -p <crate> --lib <module>::pbt
   PROPTEST_CASES=20000 cargo test -q -p <crate> --lib <module>::pbt
   ```

   On failure proptest prints a shrunk counterexample and writes it to
   `crates/<crate>/proptest-regressions/<module>.txt`. **Commit that file when
   the failure was real** — it replays first on every later run (the opposite
   of Go's rapid). Delete it when the failure was a bug in your own model.

   **When a property fails, first ask which side is wrong.** In `vram::pbt` the
   first failure was the reference model raising headroom on a release, which
   the ledger never does — but tracing it is what exposed #370. Then show the
   user the minimal counterexample before changing production code.

5. **Test the tests with mutations.** Plant bugs and confirm some property
   catches each one. Keep mutations next to this skill as
   `mutations/<target>.tsv` (one per line: `name<TAB>file<TAB>perl s///`,
   applied with `perl -0pi`; use `\Q…\E` for literal code and `\s*` between
   lines). Compare against the crate's **whole** existing suite, not just the
   target's `mod tests` — gallium often tests a function from another module
   (`react::tests` catches compaction mutations):

   ```bash
   .claude/skills/property-test/scripts/mutate.sh \
     .claude/skills/property-test/mutations/compaction.tsv \
     "PBT=cargo test -q -p gallium-agent --lib memory::pbt" \
     "existing=cargo test -q -p gallium-agent --lib -- --skip pbt"
   ```

   Pick mutations from: the reverse of a past fix commit (most convincing —
   f26a37f for compaction), off-by-one (`<`↔`<=`, `>=`↔`>`), dropping one side
   of a condition, `max` replaced by last-wins, a field update forgotten in one
   of two code paths (`book` vs `resize`), and disabling the branch the
   generator's boundary is meant to reach. Every mutation should be caught by
   some property; explain any miss. The "existing" column is the evidence for
   the PR: mutations only PBT catches are what the tests add.

   Check each substitution applies before a long run (the script reports
   `not applied`), and mind disk space: each mutation rebuilds the crate's test
   binary, and the workspace's `target/` is large.

6. **Pin undecided behavior; don't fix it.** Writing a property often forces a
   question the code never answered (should phase 2 of `compact_active_turn`
   stop at a system message? should a live `Passing` reservation count toward
   headroom?). Then:
   - Look for an answer first: `docs/`, `CLAUDE.md`, doc comments, the commit
     that introduced the behavior. Look for code that *relies* on an answer —
     `runtime.rs`'s catalog test relies on "`compact_active_turn` never drops
     system messages", which is how #369 was found. If decided, cite it and
     write the property with that exception.
   - Otherwise write a plain `#[test]` that pins what the code does today, with
     a doc comment starting **"Pins current behaviour, not a decided rule."**
   - **Say whether it can happen in practice** by tracing the callers (#370:
     `model::try_forward` books one transient per forward, so it cannot today).
     If you could not check, say "unverified".
   - List it under "Team decisions" in `FORMAL_FIT.md`, ask the user, and file
     an issue only when asked (`gh-fpt issue create --label question`).

7. **Verify like CI.**
   - `cargo clippy -p <crate> --all-targets` — no new warnings in touched files.
   - `rustfmt --edition 2021 --check <touched files>` — not `cargo fmt --all`,
     which also rewrites unrelated files that are not formatted on `main`.
   - `cargo test -p <crate>` — the existing suite still passes.
   - `proptest = "1"` sits in the crate's `[dev-dependencies]` beside
     `tempfile`.

## Done when

- [ ] Each property is numbered and stated in one line in its doc comment.
- [ ] The generator targets named boundaries, and a mutation confirmed the
      boundary branch is reached.
- [ ] Mutation results (PBT vs the whole existing suite) are in a table;
      misses are explained.
- [ ] Undecided behavior is pinned in its own test and listed in
      `FORMAL_FIT.md`, not silently fixed.
- [ ] Passes at `PROPTEST_CASES=20000`; real-failure regression files are
      committed; clippy and existing tests unaffected.
