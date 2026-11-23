# Property-test candidates

Where property-based tests (proptest) pay off in gallium, ranked, with the
evidence for each and what is already done. The how-to is the `property-test`
skill (`.claude/skills/property-test/`).

**Why PBT and not proofs:** gallium has no money/date/permission arithmetic,
and its pure logic is string-structural (wire parsers, reply cleaning) or
bookkeeping (compaction, VRAM ledger). A property against a reference model gets
nearly all of the value there. The recurring bug class is the wire/reply layer:
#362, #337, #325, #234, #199, #194.

## Done

| Target | Module | Properties | Mutations caught (PBT / existing suite) |
|---|---|---|---|
| `resolve_context_window` | `memory::pbt` | R1–R4 | 2 / 2 |
| `compaction_target` | `memory::pbt` | T1–T4 (T2 is the f26a37f invariant) | 2 / 2 |
| `compact_messages`, `compact_active_turn` | `memory::pbt` | C1–C5, A1–A6 | 5 / 2 |
| `VramLedger` accounting | `vram::pbt` | V1–V8 (state machine vs a reference model) | 8 / 5 |
| `VramLedger::prefill_booking` | `vram::pbt` | P1–P5 | 2 / 0 |

Mutation lists: `.claude/skills/property-test/mutations/{compaction,vram_ledger}.tsv`.
"Existing suite" is the crate's whole library suite with the `pbt` modules
skipped.

## Next

### 1. `StreamingReply::advance` (`crates/gallium-agent/src/streaming.rs`) — S

- **Properties:** never panics for any sequence of `visible` inputs, monotone
  or not; the concatenated output is always a prefix of the latest `visible`;
  marker-free text fed as growing prefixes then `done` reassembles exactly
  unless frozen; output never contains a `FREEZE_MARKERS` entry or splits a
  UTF-8 character.
- **Evidence:** b6c41d4 (#233/#234), a `clamp(min > max)` panic that killed
  the turn thread on an app-server. A scratch run of the first three
  properties re-finds that panic as soon as the b6c41d4 guard is removed.

### 2. `profile/wire/python.rs`, then `json.rs`, `think.rs`, `qwen_xml.rs`, `dsml.rs` — S each

- **Properties:** round-trip (`parse_calls("[F(k='" + esc(s) + "')]")` yields
  `k == s` for any `s`); never panics; any strict prefix of a call list yields
  `[]`; list/dict literals of literals round-trip, and a container with any
  non-literal element stays a string. `strip_think_blocks` is documented as
  not idempotent, which is itself a property worth pinning.
- **Evidence:** c4fb3bb (#362), the regex-truncation history in the module doc,
  #194. The first two properties hold over 5 000 cases in a scratch run.

## Team decisions

Undecided behaviour found while writing properties. Each is pinned by a test,
not fixed.

| Issue | Behaviour | Pinned by | Reachable today? |
|---|---|---|---|
| #369 | `compact_active_turn` phase 2 drops a system message that follows the task | `memory::pbt::a_system_message_after_the_task_is_dropped_in_phase_2` | No — the skill catalog goes before the first non-system message |
| #370 | A `Headroom` booking made while a `Passing` reservation is live keeps the passing bytes as headroom for good | `vram::pbt::a_passing_reservation_live_during_a_decode_booking_becomes_headroom` | No — `model::try_forward` books one transient per forward |
| #371 | Unquoted `nan` / `inf` / `infinity` argument values parse to `null` in the python wire format; other bare words stay strings | — (no in-tree test yet) | Unlikely |

## Not PBT: model-checking candidates

These have the highest bug density in the history, but they are interleaving
arguments, and function-level tests do not reach them. They are candidates for
TLA+ or loom, not proptest.

- **`SteerInbox` close/push** (`cancel.rs`, `react.rs`): "a steer accepted by a
  turn that completes was seen by the model".
- **App-server displacement** (`appserver/tcp.rs`, `AppServer::cancel_turns` /
  `adopt_predecessor`, the `accepting_turns` gate, and the lock order
  `Predecessor` → `accepting_turns` → `threads` → `active_turn`): #166 was an
  ordering bug.
- **responses-api generation lock**: 70a1df0, 8ab2ff8, 4c5fefb (panic or
  deadlock under concurrent requests).

## Not a fit

- **Candle kernels, MXFP4:** compared against references instead
  (`mxfp4_gpu.rs`'s CPU↔CUDA test, `ollama-narrow`, `scripts/layer_diff.py`).
- **`llm_local.rs` KV reuse:** entangled with llama.cpp contexts. Extracting the
  common-prefix and checkpoint-gate decisions into a pure function would make
  it a candidate.
