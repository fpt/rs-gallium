# 0004 — Execution traces are the source of truth; training data is an export

**Status:** Accepted, 2026-08-17
**Related:** #71 (a steered turn's trace is wrong), [ADR 0001](0001-prompt-purity-and-explicit-context.md), [ADR 0003](0003-model-profiles.md)

## Context

Gallium runs six model families through one ReAct loop, and every turn it runs is
a demonstration of tool use by whichever model ran it. Wanting to fine-tune on
those turns — SFT from the ones that worked, DPO from a task one backend solved
and another did not — is the obvious next use of the testsuite, which already
runs the same testcases across every configured backend.

The tempting shortcut is to write SFT JSONL directly: teach the agent to emit
`{"messages": [...]}` per turn and be done. That fixes the training format at the
moment of capture, and the format is the part most likely to change — the moment
a second trainer, a preference-pair format, or an eval harness appears, the
capture path has to grow a second writer, and the turns already recorded cannot
be re-exported into the new shape because the information the new shape needs was
never written down.

Gallium already has the thing that avoids this. [`trace.rs`](../../crates/gallium-agent/src/trace.rs)
records one turn whole — prompt, catalog, every model call, every tool call with
its arguments, result and approval decisions, usage with a prefill/decode timing
split, and how the turn ended — and `TurnTrace::to_script()` /
`TurnTrace::diff()` make a recorded turn replayable through the real loop with
real tools against `INFERENCE_ENGINE=scripted`. That is an execution trace, not a
log, and it is already most of what a training-data pipeline needs.

What it is not is *lossless*, and it was right not to be. A trace is a debugging
artifact that lives beside the workspace it describes, so it truncates every
captured text at 16 KiB (`MAX_CAPTURED_CHARS`), records the model's output only
*after* the profile parsed it, drops attachment payloads to counts, and holds no
identity beyond a turn id. Each of those is a deliberate trade for a file a
person reads. Each of them is a defect in a training example:

| Trace's choice | Right for debugging | Wrong for training |
|---|---|---|
| 16 KiB truncation | a `Read` of a large file would dwarf the turn describing it | the model's next action was conditioned on bytes the record doesn't have, so the example teaches a decision from evidence that isn't in its own prompt |
| post-parse model output | the loop's view is what the loop's bugs are about | Gemma 4 emitted `<\|tool_call>call:…` with `<\|"\|>` quoting and LFM2 emitted a bare `[Read(file_path="a.txt")]`; training on gallium's normalized `ToolCallInfo` teaches a wire format neither model speaks |
| steering not recorded (#71) | a known gap in replay | the exported conversation is missing a user message, so the assistant's next turn reads as unmotivated — the model learns to volunteer what it was actually asked for |
| no session identity | one turn per file is what you want to read | multi-turn SFT needs the conversation, and `startedAtEpochMs` cannot tell two concurrent app-server threads apart |
| no outcome | a trace records what happened, not whether it was good | SFT needs "this one worked", and nothing in the turn knows |

The last one is the one that must not be fixed the obvious way. Whether a turn
was *good* is not known when it ends — the testsuite's gates run afterwards, and
human judgement later still. Writing an outcome field at capture time means
either leaving it null forever or re-writing records after the fact.

## Decision

**The trace is the source of truth. Training data is a downstream, offline export
that no turn ever executes.** Concretely, four commitments.

### 1. Fidelity becomes a mode, not a rewrite

`capture()`'s truncation stays the default. A `GALLIUM_TRACE_FIDELITY=full` mode
spills any text over the threshold to a content-addressed blob —
`.gallium/traces/objects/sha256-<hex>` — and stores `{"ref": "sha256:…", "len":
N, "preview": "…"}` in its place. Debugging traces stay small and readable;
training traces are lossless; both are the same schema, so one reader serves
both. This is a breaking read change (`String` becomes a `Captured` enum on
`MessageRecord`, `ToolResultRecord` and `ModelOutput`), so `TRACE_FORMAT_VERSION`
goes to 2, with `#[serde(untagged)]` so a bare string still reads as inline and
today's files stay loadable.

The same mechanism carries the two payloads the trace deliberately drops to
counts today — attachment bytes and tool-result images — under full fidelity
only, for the same reason it carries file contents: a turn that answered a
question about a screenshot is not a training example without the screenshot.

### 2. The raw model output is recorded

`LlmResponse` gains the pre-parse string, filled by each provider before
`profile::parse_tool_calls` runs, and `ModelOutput` records it. `trace.rs`
deferred this to "the day a tool-call parse bug costs an afternoon"; training is
that day, for a sharper reason than parse bugs. ADR 0003 established that the
wire format is a property of the *family* — so the target a fine-tune should
learn is the family's own format, and gallium's normalized form is a translation
of it. Recording only the translation makes the family-specific half of what the
model did unrecoverable.

**Landed 2026-09-01** (docs/TODO.md §9.1, forensics rather than training the
trigger). `LlmResponse::{Text,ToolCalls}` carry `Option<RawGeneration>`; it is
recorded as `TraceStep::raw` rather than inside `ModelOutput` (orthogonal to the
text/tool-call split), which bumped `TRACE_FORMAT_VERSION` to 2 on its own —
ahead of the `Captured` enum from §1, which will not need a further bump. Text
only so far: `RawGeneration::token_ids` is reserved and unfilled.

The rendered prompt string is deliberately **not** recorded the same way. On
llama.cpp it is the GGUF's jinja render, on candle it is
`PromptRenderer::format_prompt` — both deterministic functions of the messages
plus the model, and ADR 0001's prompt-purity rule is what keeps that
reconstruction reliable. A `promptRenderSha256` (with the blob under full
fidelity) records enough to *verify* a reconstruction without making every trace
carry a second copy of its own largest field.

**Landed 2026-09-01** (docs/TODO.md §9.2, forensics the trigger): as
`UsageRecord.promptSha256` per model call, filled by both local backends, plus
`UsageRecord.kv` (fresh / slot-reuse / checkpoint-restore / reset, with the
reused and evaluated token counts) — `TRACE_FORMAT_VERSION` → 3. The digest
only, not the full render or a running hash *chain*: those need full fidelity
(§1) and are still open.

### 3. Identity and conditions live on the trace; outcome lives beside it

Additive fields, no version bump of their own: `sessionId` (minted per
`TraceSession`; the app-server passes its thread id, which it already has in
scope) and `turnIndex`, so turns regroup into conversations. `TraceMeta` grows
the inference conditions a re-run needs — selected `profile` and the signal it
was detected from, `reasoningEffort`, sampling (`temperature` / `topP` /
`maxTokens`), `contextWindow`, gallium's version, and the workspace's git
revision, resolved once per session rather than per turn.

Reasoning gains a `visibility: full | summary | none`. This is a real distinction
in gallium, not a hypothetical one: `OpenAiProvider` requests
`summary: "detailed"` and can only ever receive a summary, while the local
profiles' thought channels are the full chain. A dataset that treats the two as
the same field trains on a mixture of a model's reasoning and a description of
it.

**Outcome is a sidecar keyed by `sessionId`, never a field.** The producer is the
testsuite, which already computes exactly the right thing: `gates.sh` scores a
scenario gate by gate, and `runner.sh` — which currently exports no `GALLIUM_*`
at all — sets `GALLIUM_TRACE_DIR` and writes `labels/<sessionId>.json` beside the
traces. A label can then be added, corrected, or added by a human months later
without rewriting the record it describes.

This is also what makes preference pairs nearly free: `matrix_runner.sh` already
runs the same testcase across every backend, so identical-prompt success/failure
pairs exist the moment labels do.

### 4. The exporter is a separate binary

A new `crates/gallium-trace/` reads `TurnTrace` (already public, already
versioned) and emits `messages` JSONL, ShareGPT/TRL, DPO pairs joined on
`(testcase, promptSha)`, and eval bundles. It is not a subcommand of `gallium`
and not a feature of `gallium-agent`: export is offline batch work, and nothing
that shapes training data should be able to fail, slow, or alter a turn.

### Landing order

Each step is useful on its own, and only step 2 has real blast radius.

0. **No code.** Run the testsuite with `GALLIUM_TRACE=1` and read the files. How
   often the 16 KiB cut actually fires, and how large a full-fidelity trace
   really is, are the numbers that set the blob threshold — guessing them first
   would be designing the storage layer against an imagined workload.
1. **Identity and labels.** `sessionId` / `turnIndex`, the extended `TraceMeta`,
   reasoning `visibility`, and the testsuite writing label sidecars. Purely
   additive; makes today's traces groupable and scored.
2. **Fidelity.** The object store and `GALLIUM_TRACE_FIDELITY`, raw model output
   through `LlmResponse`, and steer recording (#71). Touches every provider and
   every trace consumer — including `diff`, whose `Captured` comparison has to
   normalize inline-versus-ref or a replay will report divergence where two runs
   merely spilled differently.
3. **The exporter**, `messages` format only.
4. **DPO pairing and eval bundles.**

## Consequences

**Good.** A format that does not exist yet can be produced from turns already
recorded, because the record holds the execution rather than a projection of it.
The replay machinery is unchanged and gains a second use: an eval bundle is a
trace plus a workspace snapshot, which is what `to_script`/`diff` already
consume. Labels are correctable, so a mislabelled run is a one-line fix rather
than a re-run. And the debugging artifact stays the small, readable thing it is —
full fidelity is opt-in, so nobody pays for training storage while chasing a bug.

**Bad.** Full fidelity means traces that hold entire file contents, entire
prompts, and attachment payloads. `trace.rs` already warns that a trace "belongs
next to the workspace it came from, not in a bug report"; this makes that
strictly more true, and the object store inherits the rule — local, gitignored,
never attached to an issue. Anything exported for training needs a scrub pass for
absolute paths and secrets, and that pass belongs in the exporter, where it can
be tested, rather than in the capture path, where a mistake is unrecoverable.
Recording the raw model output widens `LlmResponse`, which every provider
implements. And two representations of a captured text (inline and by reference)
is a comparison hazard anywhere traces are diffed.

**Boundary.** This ADR governs *capture and export*. It takes no position on what
to train, on which trainer, or on whether a given fine-tune is a good idea. It
also does not make gallium a training tool: the exporter reads files and writes
files, and the agent is unaware it exists.

**Legal, and worth stating once.** Outputs from the OpenAI backend are subject to
that provider's terms, which generally prohibit using them to train competing
models. Local-backend traces carry no such restriction. The `engine` field is
already on every trace, so the exporter can filter on it — but the decision is
the operator's, and the record is deliberately neutral about it.

## Alternatives considered

**Write SFT JSONL directly from the agent.** The shortcut this ADR exists to
decline. Rejected because it fixes the training format at capture time, which is
the part most likely to change, and because the information a *second* format
needs (approval outcomes for a safety fine-tune, timings for a latency study, the
raw wire text for a format fine-tune) would have to be predicted in advance.
Every one of those is already in a trace incidentally.

**Emit a flat event log (`run_start` / `model_request` / `tool_call` / …) as a new
capture path.** The shape most trajectory-recording projects converge on, and the
one this design was proposed as. Rejected as a rewrite of something that already
exists: `TurnTrace` *is* that event stream, structured rather than flattened —
`steps[].calls[]` carries the same information as paired `tool_call` /
`tool_result` events with strictly less room for a call and its result to
disagree, since they are one record. A flat log is a better *export* target than
an internal model, and the exporter can produce one.

**Record every iteration's full prompt** rather than the first plus the
transcript. Rejected for the reason `trace.rs` already gives: later prompts are
the first prompt plus the tool transcript the trace already holds, so recording
each whole multiplies the largest field by the iteration count to say nothing
new. ADR 0001's prefix property is what makes the reconstruction sound, which is
a second reason to keep protecting it.

**Make full fidelity the default and let the exporter truncate.** Simpler — one
mode, one code path. Rejected because it makes every debugging session pay for
training storage, and because the failure is silent: a `.gallium/traces` that
quietly grows to the size of the workspace's read history is discovered late and
by accident. The blob store is what makes the choice cheap enough to make twice.

**A `outcome` field on the trace, filled by the testsuite after the run.**
Rejected because it means rewriting a record after it is written, and a record
that gets rewritten is one whose earlier copy cannot be trusted. It also cannot
represent the case that matters most — a human disagreeing with an automated
gate — without a second field, at which point it is a sidecar with extra steps.

**Export from the app-server's event stream** (`item/*` notifications) instead of
from traces. Attractive because a client already receives it. Rejected: the
stream is a projection for display, it omits the prompt and the catalog
entirely, and it exists only when a client is connected — the REPL and the
testsuite, which is where the interesting turns actually happen, produce no
stream at all.
