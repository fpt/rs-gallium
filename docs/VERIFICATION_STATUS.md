# Verification status

gallium's chat-template and wire-protocol changes are tested at the **template
level** — rendered through the real `chat_env` / `render_native_prompt`, no
weights (`crates/gallium-agent/src/llm_local_templates.rs`). That catches
whether a prompt is *built* right, not whether a real model *answers* right.

This document records which of those changes have additionally been confirmed
against a **running model**, on what hardware, and what came out. Append a row
when a prompt-affecting change is verified; the template tests stay the first
line of defence.

## Reference hardware

| Box | Accelerator | Notes |
|---|---|---|
| CUDA dev box | RTX 4070 12GB, 128GB RAM | `--features cuda`; the `configs/*.toml` `gpuLayers` / `cpuMoe` values are tuned for this card |
| Mac | M3, 24GB, Metal (automatic) | the other reference machine the configs must run on |

`make testsuite BACKENDS="<name>"` is the check; results land in
`testsuite/results/test_results_*.txt` (gitignored).

## Prompt-affecting changes and their verification

The 2026-08 template-fixture work (#181, #183–#187, #189) changed what several
models put in front of the model. Verified 2026-08-27 on the CUDA box:

| Change | Models affected | Status |
|---|---|---|
| #181 chat-template fixture harness | — (test infra) | n/a |
| #183 LFM2 renders its own template (was manual ChatML) | LFM2.5 | **verified** — see LFM2 below |
| #184 system messages merged when a template admits one | Qwen3.8 (and any single-system template) | **verified clean** — Qwen3.8 matrix unchanged vs. the 2026-08-15 baseline |
| #185 prior-turn `reasoning_content` carried forward | Gemma 4, Qwen3.8, LFM2 | **verified** — no regression on Gemma 4 or Qwen3.8. On LFM2 it never became effective *through the template*: that family's `preserve_prior_reasoning` is `Some(false)`, its own template's default. The one thing that put prior reasoning back in front of it was #192's verbatim replay, which #196 removed — see LFM2 below |
| #186 `reasoningEffort` projected onto the qwen3 family's accepted set | Qwen3.8 | **verified clean** — see Q1 |
| #187 `qwen4exp` arch, protocol-downgrade `warn`, multi-block XML parsing | Qwen3.8-Flash-Next (unloadable), Qwen3.8-27B | template-level only; Flash-Next is blocked on [llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742) |
| #189 each family states its own prior-reasoning policy | all | **verified clean** — behaviour unchanged, matrices stable |

### Qwen3.8-27B (`qwen3.8`, Q3_K_XL)

2026-08-27: **10 / 11 pass**, only `multimodal_audio` failing — which is the
documented limitation, not a regression: Qwen3.8-27B's projector is vision-only
(`clip.has_vision_encoder`, no audio-encoder field). Matches the 2026-08-15
baseline (8/9, same one failure; `data_analysis` and `spec_discovery` are newer
cases and both pass). The four stacked changes (#184/#185/#186/#189) did not
regress it.

**Performance re-tuned 2026-08-30 (RTX 4070).** Two config changes —
`gpuLayers 30 → 42` and `reasoningEffort` unset → `"high"`. Full testsuite on
the new config: **10 / 11**, same `multimodal_audio`-only failure — quality
unchanged, speed not:

| | before (gl30, xhigh default) | after (gl42, high) |
|---|---|---|
| `capital`, one call | 41 s | 10 s |
| `memory_state`, thinking off (isolates the offload) | 57 s | 6 s |
| `needle_in_haystack`, thinking off | 7.6 s | 1.5 s |

- **gpuLayers.** The old 30 (and the note that the projector forced it down
  from a text-only 40) predated flash attention being enabled in the vendored
  llama.cpp — with 4 KV heads that roughly halves the KV cache, the most likely
  reason the ceiling moved up ~15 layers. Re-bisected with the projector loaded,
  3× `coding` per value: 42/44/46 load and run (46 peaks ~9.2 GB of 12), **50
  fails at context creation**. 42 keeps headroom for a long conversation's KV
  growth and a project's AGENTS.md/CLAUDE.md, neither of which the testsuite
  exercises. Isolated with thinking off (fixed output length), the +12 GPU
  layers are ~9× on a decode-bound turn.
- **reasoningEffort.** The template turns thinking on by default and then
  defaults `reasoning_effort` to `xhigh`; on a 27B with ~23 CPU layers that
  budget dominates every turn. gallium's `"high"` (→ template `medium`) keeps
  every multi-step testcase passing and cuts the one-shot turns ~4×.
- **Q3 vs Q4.** Measured at their respective projector-loaded ceilings (Q3 ~46,
  Q4 ~36 GPU layers). Testsuite: **10 / 10 runnable each**, indistinguishable;
  wall time within noise. Q3 keeps ~10 more layers on the GPU for this card, so
  it is the pick. (The cross-platform Q4 config was removed 2026-08-30 — no
  hardware here fits it whole.)

**`refactoring` wire-shape gap (fixed).** The model reasoned correctly and wrote
correct replacement code but emitted its tool call as
`{"function": "Edit", "file_path": …, "new_string": …, "old_string": …}` — tool
name as a flat string under `"function"`, arguments as sibling keys rather than
nested under `"arguments"`, a shape `profile::wire::json` does not recognise.
Fixed by not needing that parser: `profile::Qwen3::template_formats_tools_natively`
matches this model's embedded template, which declares a native
`<tool_call><function=NAME><parameter=K>value</parameter></function>` format, so
llama.cpp renders tools through it instead of asking for JSON prose the model
produced less reliably.

**Sampling — `temperature 1.0`, `topP 0.95`, `topK 20`.** This model card's
recommended "Thinking Mode" set (which applies since thinking defaults on). The
card's `min_p` / `presence_penalty` have no gallium sampler stage yet, so the
match is partial. Adopted after a klein report of malformed tool-call JSON deep
in a ReAct loop (an unclosed `<tool_call>` trailing into a duplicated
`<tool_call>`/`</function>` — a blend of gallium's JSON-prose shape and the
model's native XML tool format) under the earlier unspecified-`topP`/`topK`,
`temperature 0.7` setting. Re-verified against the full local testsuite with no
malformed output observed, including the multi-iteration cases — not a guarantee
against the ~8-iteration, ~17K-token conversation that first surfaced it, but a
measured improvement.

The template-patch details (`unsloth/Qwen3.8-27B-GGUF` maps `high` → `xhigh`
before the membership check; the Hub template raises instead) are in
`crates/gallium-agent/tests/fixtures/chat_templates/README.md`.

### Gemma 4 E4B (`gemma4`, Q4_K_M + projector)

2026-08-27: **10 / 11 pass**, only `data_analysis` failing. **Not a #185
regression** — the testcase was added 2026-08-17 and never ran against `gemma4`
in any saved matrix, and the failure is model capability, not wire-layer: the
trace shows E4B parsing the `Read` tool's line-number prefix as a CSV column,
writing `---RESULTS---` markers into the output file, then a fallback Python
script that ends up writing a literal `\n`. The 10 passing cases include the
`<|channel>thought` reasoning path #185 touched.

**`enable_thinking` on E4B — measured 2026-08-29, not adopted.** The 12B and
26B-A4B below both use Gemma 4's thought channel productively; the 4B E4B is the
one size that does not — so the split is **capacity, not dense-vs-MoE**. E4B with
`reasoningEffort = "medium"` (→ `enable_thinking = true`): `data_analysis` 1/6
gates vs 2/6 off — both fail, marginally *worse* with it; `refactoring`,
`arithmetic`, `multimodal_image` all still pass. It rescues nothing on the
testsuite and adds ~18k tokens/turn, so `configs/gemma4.toml` stays greedy with
no effort set. The docs/models/gemma4.md thinking-loop warning did not reproduce at
this small n, but there is no upside to weigh against it either.

### Gemma 4 12B (`gemma4-12b`, Q4_K_XL + encoder-free projector)

Dense 12B, between E4B and the 26B-A4B in size. **`reasoningEffort = "medium"` —
adopted**, measured 2026-08-29:

| case | thinking off | thinking on |
|---|---|---|
| `data_analysis` | FAIL 0/6 gates, 14-iter | **PASS 6/6**, 12 calls, ~106s |
| `spec_discovery` | FAIL (2026-08-28 matrix) | **PASS**, 8 calls, ~50s |
| `refactoring` | PASS | **PASS**, 4 calls |
| `arithmetic` | PASS | **PASS**, 1 call |

Two failures flip, nothing breaks. Full matrix with the flag, 2026-08-29 CUDA
box: **10 / 11 pass**, only `multimodal_audio` failing — up from 8/11. That
projector is encoder-free (`clip.vision.block_count = 0`, 167 MB) and transcribes
audio inexactly (*"zuki"*), independent of thinking.

**`gpuLayers = 24`.** Measured 2026-08-30 (RTX 4070, 11902 MiB), text-only turn
with the system prompt, skill catalog and tool schemas. At full offload (999)
`llama_new_context_with_model` returns NULL on the first turn ("Failed to create
context: null reference from llama.cpp") — the 6.7 GB Q4_K_XL weights fit alone,
but plus the vision + audio CLIP contexts (mmproj, also on the GPU) plus an
8192-token, `n_ctx`-batch context is over the card. Bisected: `20`/`26`/`30`
hold (peak VRAM 8947 / 9473 / 10847 MiB), `40` and full offload fail at context
creation. Ships `24` — a margin below the moving edge, not the tightest passing
value (issue #92). A 24GB M3 fits far more; raise `gpuLayers` or let
`GALLIUM_GPU_LAYERS` override. Confirmed end to end over the app-server dialed
from klein: multi-turn agent loop with tool calls, no failure, ~7 GB peak.

### Gemma 4 26B-A4B (`gemma4-26b`, Q4_K_XL, `cpuMoe`, `gpuLayers = 20`)

Two 2026-08 findings, both landed together:

**The thought-channel leak — fixed.** 2026-08-28 matrix: `data_analysis` and
`refactoring` failed with the raw string `<|channel>thought<tool_call|>` shown to
the user as the reply. `crate::gemma::strip_thinking_blocks` knew only Gemma's
*closed* channel form (`… <channel|>`), so an unclosed `<|channel>thought` — the
model opens the thought channel and then produces a bare `<tool_call|>` and EOS —
passed straight through, exactly as an unclosed `<|think|>` used to before that
case was handled. Now an unclosed `<|channel>thought` is stripped as thinking
(and `thinking_content` claims it, keeping the two inverses in agreement), and
`Gemma4::clean_reply` also trims a trailing bare `<|tool_call>` / `<tool_call|>`
delimiter.

**`reasoningEffort = "medium"` — adopted here, unlike E4B.** The bare-marker
reply was the *symptom* of a deeper problem: with thinking off, the 26B-A4B
looped to `max_iterations` on `data_analysis` and diverged on `refactoring`.
Turn its thought channel on and it plans the multi-step task instead. Measured
2026-08-29 on this config (leak fix in tree):

| case | thinking off | thinking on (`reasoningEffort = "medium"`) |
|---|---|---|
| `data_analysis` | FAIL 0/6 gates, 30-iter loop | **PASS 6/6**, 8 ReAct calls, ~80s |
| `refactoring` | FAIL (diverged) / PASS with leak fix | **PASS**, 4 calls, ~31s |

So the MoE *does* use the scratchpad the dense E4B could not. Full matrix with
the flag set, 2026-08-29 CUDA box: **10 / 11 pass**, only `multimodal_audio`
failing (26B-A4B's `mmproj-BF16.gguf` reports `audio=false` — its model card
lists text + image only). Up from 8/11 in the 2026-08-28 matrix, where
`data_analysis` and `refactoring` both failed (`refactoring` via the leaked
marker).

### DeepSeek-V4-Flash (`deepseek-v4-flash`, UD-IQ3_XXS, `cpuMoe`)

284B / 13B-active MoE, `general.architecture = deepseek4`. Text-only (no
projector), so `multimodal_*` skip. The config comment recorded **7/7** on the
seven tool cases that existed 2026-08-15; the **2026-08-28** full matrix found
`file_read` and `data_analysis` failing, and the investigation split three ways.

- **Not #196 (`Slot::checkpoint`).** DeepSeek-V4 hits the refused-partial-rollback
  path — `llama_kv_cache_dsv4::seq_rm` refuses every real partial rollback, which
  is why the reuse gate latches on an observed refusal rather than on
  `llm_arch_is_hybrid` (which omits `deepseek4`). It *used to* then take a
  `Slot::checkpoint`; as of the safe half of #209 it does not (see "The
  checkpoint state does not round-trip" below), so a refusal is a full
  re-prefill. Either way `file_read` fails identically with
  `GALLIUM_KV_CACHE_SLOTS=0` — slot pool and checkpoint code fully bypassed, 3/3
  — and the failure is on ReAct iteration 1, before any reuse could apply.
  Verbatim replay (#172 / #192) is not in the tree. The checkpoint work did not
  cause *these* failures — though it is, separately, **not exact on this cache**.

- **The real cause: reasoning was off.** DeepSeek-V4-Flash reasons only when
  asked (HF card: `reasoning_effort`, *"not enabled by default"*), and the config
  set no `reasoningEffort`, so the GGUF template rendered `thinking = false` and
  the model was prompted to act with no scratchpad. Symptom: one intent sentence
  + EOS on iteration 1, no DSML call (*"I'll read the file for you."* /
  *"I'll analyze the data."*), deterministic 3/3 — the same "announced intent and
  stopped" the config's sampling history chased, which was never sampling.
  `reasoningEffort = "medium"` (→ `thinking = true`, no forced effort text) fixes
  both: `data_analysis` 0/6 → **6/6 gates**, `file_read` reads the file and
  answers. `"high"` also passes both but prepends a long "absolute maximum, no
  shortcuts" instruction block that is overkill for routine agent turns; `"low"`
  is `thinking = false` and does not help. Cost is real: thinking adds ~500–900
  output tokens per call at ~9 tok/s decode on this quant, so `data_analysis`
  runs 7–9 ReAct iterations over several minutes.

- **Secondary, independent: `ModelProfile::agent_preamble` (2b39c51) regressed
  `file_read` on its own.** A/B on the thinking-off binary, `DeepSeekV4`'s
  `agent_preamble_suffix` forced to `None` (so `agent_preamble()` is `None`
  entirely — no `BASE_AGENT_PREAMBLE`, no DSML reminder): `file_read` went
  0/3 → 2/2. Thinking on absorbs this — both A/B arms pass with it — so the fix
  is `reasoningEffort` alone and the preamble is left in place, but a
  thinking-off run of this model would hit it again.

Full matrix with `reasoningEffort = "medium"`, 2026-08-28 CUDA box: **9 / 9
runnable pass** (2 `multimodal_*` skipped, no projector) — every case, up from
7/9 with the two failures above. `data_analysis` and `refactoring` run 7–9 ReAct
iterations each; the whole matrix took ~40 min.

**The checkpoint state does not round-trip — measured 2026-08-29.**
`llama_kv_cache_dsv4` (`kv_raw` plus three compressed sub-caches at different
position scales) was the one shipping checkpoint-path cache with no
`tests/kv_state_spike.rs`-style logit check. Run against `UD-IQ3_XXS` on the CUDA
box (`GALLIUM_KV_SPIKE_CPU_MOE=1`), deterministic across two runs to the last
digit: **`test result: FAILED. 3 passed; 3 failed`**.

| test | Δ top-1 logit |
|---|---|
| `a_restore_with_nothing_in_between_is_the_same_state` | **1.689527** (1681 tokens, no generation between snapshot and restore) |
| `a_sequence_snapshot_..._costs_less_than_a_prefill` | **0.808103** (12-token continuation agrees; logits do not) |
| `an_on_device_checkpoint_restores_an_equivalent_state` | **1.689527** (identical to the host path) |

For comparison the same test is **0.000000** on LFM2 and **0.021921** (cell
placement) on Gemma 4 E4B. Cost is fine (snapshot 28.3 MiB, get 7.2 ms /
set 4.2 ms, restore+suffix 0.1× the prefill; on-device handle 27 KiB) — it is
*equivalence* that fails. This is **not** what caused the 2026-08-28 `file_read`
/ `data_analysis` failures (those were reasoning-off, unchanged with
`GALLIUM_KV_CACHE_SLOTS=0`), but a 1.69 Δlogit is the silent-corruption shape
#196's gate exists to prevent. `docs/OPTIMIZATION.md` §3.5 has the cross-model
table.

**Resolved (safe half of #209): `deepseek4` no longer takes a checkpoint.**
`arch_checkpoint_state_round_trips` in `llm_local.rs` returns `false` for
`general.architecture == "deepseek4"`, so `take_checkpoint` stores nothing and a
refused partial trim falls through to a full re-prefill — slower, but not running
from an almost-equal state with only a `debug!` line to say so. The gate logic is
unit-tested (`only_deepseek4_disables_state_checkpoints`); the round-trip itself
is still broken and `kv_state_spike` still fails 3/3 on this model — fixing
`llama_kv_cache_dsv4`'s `state_write`/`state_read` (vendored or upstream) is the
hard half and remains open on #209.

### LFM2.5-8B-A1B (`lfm2`, Q4_K_M)

Hybrid short-conv + GQA-MoE. After #183 it renders its own template; after #196
a `Slot::checkpoint` restores the sequence state llama.cpp will not partially roll
back, so the KV cache survives a ReAct turn. (#192 answered that same question
with verbatim assistant-turn replay and has been removed — see below.)

**CUDA box, 2026-08-27, post-#192: 6 / 11 pass** (`arithmetic`, `capital`,
`file_read`, `memory_state`, `needle_in_haystack`, `refactoring`), 3 fail
(`coding`, `data_analysis`, `spec_discovery`), 2 skip (text-only, no projector).
Up from 5/11 pre-#192 — `refactoring` flipped to pass once the model saw its own
prior reasoning and calls instead of gallium's reserialization. The three
failures were read as #118: the model answers a write with a name-less
`{"file_path": …, "content": …}` object that no wire parser claimed.

**Mac (M3, Metal), 2026-08-27/28.** The same GGUF blob and the same fixed sampler
seed produce a *different wire shape* here, which is the finding worth keeping:

| case | CUDA | Metal |
|---|---|---|
| `coding` | name-less `{file_path, content}` (#194) | `{"Write": {…}}` — already parsed |
| `refactoring` | passed | `{"MultiEdit": [ {…} ]}` — name-keyed, value an unwrapped array |

So #194's fix is unverifiable here (its shape never appears), and #194's scope
note — attributing `refactoring` to the candle path — was incomplete: on Metal
llama.cpp fails it on a third shape. Deterministic, 5/5 identical runs.

Fixed in three steps, each measured on this box:

1. `arguments_for` binds a shape-1 array value to the tool's one array
   parameter → `refactoring` passes on llama.cpp, matrix 5/11 → 6/11.
2. `wire::python` rewritten as a structural parser (quotes, escapes, nesting)
   → a code payload survives the Python-ish format instead of truncating at its
   first `)`.
3. `Lfm2::template_formats_tools_natively` → llama.cpp asks this model for its
   own format, as candle always had. `coding` joins `refactoring` in passing.
4. Both `lfm2` configs set to `temperature = 0.0` — see below; this is what makes
   candle's `refactoring` reliable rather than a coin flip.

**`coding` and `refactoring` pass on both engines**, and the matrix (llama.cpp) is
**8 / 11**, up from 6/11 this morning — `data_analysis` is the only failure left,
`spec_discovery` having come along with greedy.

Rates, since one of them was not a rate you would want to quote as "passes":

| | llama.cpp @0.3 | candle @0.3 | llama.cpp greedy | candle greedy |
|---|---|---|---|---|
| `coding` | 3/3 | 2/2 | 3/3 | 1/1 |
| `refactoring` | 4/4 | **2/5** | 3/3 | 2/2 |

The candle failures are not wire failures, and not one failure repeated — each
run breaks differently, and one is not Go at all: a duplicated `func main()` left
behind by an edit; a file rewritten without its `import "fmt"`; a "refactoring"
containing `new Counter` in three places, which is Java's syntax. That looked
like the candle engine's generation, so it was checked rather than assumed:
**at `temperature = 0.0` both engines write the correct file** — valid Go, import
intact, prints 3, candle reaching it with `Edit` rather than a whole-file
rewrite.

So the residue is **sampling**, not the backend. These configs ran at
`temperature = 0.3` with no `topK`/`topP`, so one unlucky token in a code payload
is a file that does not compile, and a coding agent has no reason to want that
draw: an agent's tool calls and the code inside them are not a place for
diversity. Both `lfm2` configs are greedy now, which also makes a testsuite run
reproducible instead of a coin flip. Whether candle is *more* temperature-
sensitive than llama.cpp on the same weights (4/4 against 2/5 at 0.3) is a
smaller question left open at n=9; `docs/CANDLE_BACKEND.md` item 6 is where it would
start.

`spec_discovery` is worth flagging as *borderline* rather than failing: it passed
in one of the three matrices run today (the one carrying the reverted native-path
replay) and failed in the other two, with no wire-layer difference between them.
`data_analysis` failed in all three.

`coding` needed step 3 specifically, and the reason generalizes: on the JSON
prose protocol this model writes a code payload's newlines as `\\n`, and nothing
downstream may repair that — JSON says `\\n` *is* a backslash and an `n`, and
code legitimately contains one. In the native format `\n` in a quoted literal is
a newline by the format's own rule, so the payload arrives intact. Instructing
the model instead made it worse: an `agent_preamble_suffix` about escaping (three
runs) left `coding` uniformly double-escaped and flipped `refactoring` to fail by
sending `edits` as a string. Reverted; recorded in `profile/lfm2.rs`.

Switching to the native path cost #192 at first, and the fix was not the one that
looked obvious. `build_prompt` returned early for a native render and the sentinel
staging lived on the prose path, so iteration 2 evaluated 1827 of 1827 tokens
(prefill 2.62s) where the prose path evaluated 118 of 1890 (0.26s).

Extending the staging to `render_native` was written and measured, and **it was
reverted**: it restores the cache exactly as expected — iteration 2 evaluating
**115 of 1828** (prefill 0.25s), iteration 3 **62 of 2324** — and costs the
`refactoring` testcase, **1 of 7 runs passing against 4 of 4 without it**. The
failure is not a wire failure: the model writes a `counter.go` with no
`import "fmt"`, so `go build` reports `undefined: fmt`. The one passing run
noticed it itself with `go run` and rewrote the file.

**A `Slot::checkpoint` gets the reuse back without that cost** — see
`docs/OPTIMIZATION.md` §3.5 for the primitive and its price. Restoring a state
captured at the end of an earlier prompt asks nothing of the model's output, so
the prompt stays the template's render and the `<think>` block never returns.
Measured on the same turn: iteration 2 evaluates **156 of 1754** and iteration 3
**862 of 2460**, taking the turn's prefill from 9.1s to 4.2s, with `coding` and
`refactoring` both still 3/3.

**And it retired #172's replay entirely.** Head to head on one turn, with LFM2
forced onto the prose path so both mechanisms applied:

| | prompt | evaluated | prefill |
|---|---|---|---|
| verbatim replay (+ checkpoint) | 1864 | 118 | 0.26 s |
| checkpoint alone | 1796 | **131** | 0.29 s |

Thirteen tokens and 30 ms — and the checkpoint's prompt is 68 tokens *shorter*,
since replay puts the model's `<think>` block back in front of it. Nothing was
reaching the replay either: it needs a recurrent/hybrid model on the **prose**
path, and every such model here renders tools through its own template
(LFM2 as of this work, Qwen 3.6 hybrid and Qwen3.8-Flash-Next through
`<function=`, whose templates are byte-identical). Checkpoints sit below
`build_prompt` and cover both paths regardless. Removed, along with
`replay_text`, `ChatMessage::raw_generation` and `LlmResponse::ToolCalls::raw`.

That tension was forced by the mechanism, and it is what retiring it dissolved.
Replaying the model's own bytes replayed the `<think>` block this template
*drops* for a prior turn (`preserve_thinking` defaults false, gated on
`loop.index0 > last_user_index`), and byte-exactness was not optional — llama.cpp
refuses a partial rollback of recurrent state, so trimming the reasoning out of
the replay would have returned the reuse to zero. Cache reuse and answer quality
were in direct tension for as long as reuse was defined in terms of the model's
output. A checkpoint asks nothing of that output, so the question does not arise.

Worth noting the sign flip while the replay existed: on the **prose** path it
*helped* `refactoring` (that is what #192's 5/11 → 6/11 was), and on the native
path it hurt it. One model, one mechanism, opposite outcomes on the two prompts —
which is a reason to measure a cache change against the testsuite and not only
against `evaluated`.

And it was not the first sighting in this repo: `docs/models/gemma4.md` recorded
`GemmaProtocol::format_prompt_with_tools` replaying prior assistant turns
verbatim as a likely cause of the thinking loops seen with `--thinking` on E4B
(since fixed — it strips thinking from history). Same shape — a model handed its
own earlier reasoning behaves worse — on a different family and a different
backend.

**Candle on Metal, 2026-08-29: the MoE forward stopped dequantizing.** The
expert weights were expanded to f32 inside `forward`, once per active expert per
token — 288 expansions of 14.7 MB each, 4.23 GB allocated and freed per token,
decode at 1.1 tok/s. `QExperts::qmatmuls` multiplies against the quantized weight
instead: **37 ms/token, 26.9 tok/s**, and the matrix runs in 218 s instead of
3774 s. `docs/CANDLE_BACKEND.md` §6 has the table and the two alternatives that
were measured and rejected.

**#204 (`2a2dec3`) corrected #203's explanation** — not 8-bit activations
(llama.cpp multiplies quantized against the same GGUF and passes) but candle's
*two-implementation* quantized multiply: exact at one row, ~1.4e-3 off at two or
more, so prefill drifted and decode did not. #204 routes the expert matmul by
shape (quantized for one row, expanded for many); on Metal that restores
`refactoring`, matrix **6/11 → 7/11** at 26.6 tok/s. Full mechanism, the CUDA
re-run, and the device cross-comparison are in `docs/CANDLE_BACKEND.md` §6/§6b/§6c
— this section keeps only the pass/fail.

**CUDA box, 2026-08-29 — the candle stack (#201 seeded sampling + every system
message, #202/`009439b` KV-cache reuse, #203/#204 MoE matmul) run against weights
for the first time on this backend** (only ever verified on Metal before):

| backend | device | result |
|---|---|---|
| `lfm2-candle` | CUDA, post-#204 | **6 / 9 runnable** — fails `data_analysis`, `refactoring`, `spec_discovery`; 2 `multimodal_*` skip |
| `gemma4-candle` | CPU (E4B; CUDA OOMs — PLE table ~10.5 GB dequantized) | **9 / 9 runnable, 100%**; 2 `multimodal_*` skip |

- `lfm2-candle` on CUDA is **unchanged by #204** — identical 6/9 to the pre-#204
  `quantize-always` run on this box, where Metal goes to 7/11. `refactoring` still
  fails; `data_analysis` output shifted (gave-up → wrong-ordering), so the routing
  does move CUDA numerics, just not across a grading line. `docs/CANDLE_BACKEND.md`
  §6c has the `engine_logits` table and the 2×2 device comparison: candle's CUDA
  prefill is the outlier of {candle,llama}×{CPU,CUDA} by ~0.27 logit, dequant
  (bit-identical CPU/CUDA) and TF32 (candle uses `CUBLAS_COMPUTE_32F`) are ruled
  out, and the residue — two honest-fp32 matmuls ~1e-2 apart — is unexplained
  pending a per-layer comparison.
- `gemma4-candle`'s clean sweep is the first against-weights exercise of #202's
  KV-reuse path for a **pure-attention** model on candle (positional truncate, no
  checkpoint); it includes `data_analysis`, which fails on E4B via llama.cpp.
- `lfm2_gguf_reuse_matches_a_cold_cache` (ignored integration test, the one
  `609cbcd` cites) **passes**, 9.25 s on CPU: a warm hybrid cache rewound to an
  earlier prompt continues identically to a cold one. `gallium-core`'s
  `a_seeded_draw_repeats` / `a_seeded_draw_still_varies_by_position` (#201) and
  `a_quantized_matmul_matches_dequantize_then_matmul` also pass.

**M3/24 GB Mac, 2026-08-30 — the four #219–#222 changes (host-resident PLE,
preallocated KV + `slice_set`, sliding-window K/V narrowing, strided-view
matmul) exercised on Metal for the first time** — they were developed and
measured on the CUDA box (CPU/CUDA), and `slice_set`, strided `matmul` and the
host→device gather are separate implementations per backend:

| check | result |
|---|---|
| `gemma4_gguf` (E4B end to end, `GALLIUM_DEVICE=metal`) | **pass**, 61.8 s total |
| `gemma4_gguf_kv_narrowing_is_exact_and_faster` on Metal | **pass** — identical greedy stream at 2220 prompt tokens, 64 gen |
| `gallium-core` unit tests (54, includes the gqa strided-vs-expanded equivalence on CPU) | **pass** |

Decode absolute on this box: ~11–16 tok/s (E4B, 2220-token context). **The A/B's
*timing* print is unusable here, and the way it fails is worth stating precisely,
because the obvious summary of it is wrong.** Four runs of the same unchanged
test on this Mac:

| run | prefill (off→on) | decode | decode ratio |
|---|---|---|---|
| first pair | 21.2 → 45.6 s | — | — |
| same, flags swapped | 20.3 → 43.5 s | — | — |
| both integration tests in parallel | 39.7 → 27.8 s | 35.2 → 4.6 s | **7.66×** |
| serial, this test alone | 21.2 → 25.3 s | 5.4 → 5.3 s | **1.02×** |

The first two look like a clean order effect — the *second* arm ~2.2× slower
whichever flag it carried, the second model load paying the first one's memory
pressure on 24 GB — and that was the reading recorded here first. It does not
survive the other two: run 4 is the same order and the same flags with the
second arm only 1.19× slower, and run 3 has the second arm *faster*. The first
arm's prefill is reproducible (21.2, 20.3, 21.2 s); nothing about the second one
is.

So the honest statement is not "the second arm is slower" but **the timing print
on a 24 GB box is a measurement of ambient memory pressure, not of the change**:
the decode ratio alone spans 1.02× to 7.66× across runs that differ only in what
else was resident. Quote neither. The **1.24× decode from the 128 GB CUDA box is
the only speedup figure with a machine behind it**, and the narrowing's speedup
remains unmeasured on Metal — not because one confound was identified, but
because this host cannot hold the two loads steady enough to measure anything.

What *is* durable on Metal is the exactness assertion, which passed on every one
of the four runs: identical greedy streams at 2220 prompt tokens, 64 generated.
That is the property #221 needed verified on this backend, and it is verified.

### Gemma 4 E4B safetensors + vision on candle (`gemma4-vision-candle`)

2026-09-02, PR #249 — the config that exercises the candle image path
(`gemma4_vision.rs` + `gemma4_image.rs`): `hf:unsloth/gemma-4-E4B-it`
safetensors, `inferenceEngine = "candle"`, run `GALLIUM_DEVICE=cpu` on the
128 GB CUDA box (the safetensors `Gemma4::load` keeps E4B's ~7 GB of PLE +
embedding tables on-device, so the 4070's 12 GB OOMs; the 24 GB M3 fits it on
Metal). ~28 tok/s prefill, ~3.4 tok/s decode on CPU — an order of magnitude
slower than the GGUF configs, so a full matrix runs in hours, not minutes.

| testcase | result | note |
|---|---|---|
| arithmetic, capital, coding, file_read, memory_state, needle_in_haystack, refactoring, spec_discovery | **PASS** (8/8) | at `temperature = 0.7` |
| multimodal_image | **PASS** | the point of the config — candle's first image path; caption verified bit-exact against a `transformers` reference (`encode_image` Δ 8e-5) |
| multimodal_audio | n/a | no audio tower on the candle Gemma 4 path (`llm::reject_audio` refuses the turn); `matrix_runner.sh` now **skips** it here, the same way a text-only backend skips `multimodal_image` |
| data_analysis | **FAIL** | E4B loops retrying `awk … > file` Bash calls that the approval broker refuses as `[destructive]` non-interactively, until it hits the time cap. Model-capability, not vision/candle: `data_analysis` is the known-hard E4B case (`gemma4` llama.cpp fails it too, for a related reason — the Read line-number prefix read as a CSV column). Independent of temperature — same loop at 0.0 and 0.7. |

So **9 / 10 runnable pass**, `data_analysis` the sole failure and it is E4B's,
not the backend's. `temperature = 0.7` in the config matches `gemma4-candle.toml`
and `gemma4.toml`: at the `0.0` first cut, Gemma 4 ran away under greedy decode
on `memory_state` (a repetition loop to `maxTokens`, the candle `GemmaProtocol`
stop detection not catching the malformed `<turn|>` marker it emitted) — 0.7
fixes it.

### Gemma 4 E4B GGUF + mmproj vision on candle (`gemma4-candle`)

2026-09-03 — the candle image path from the **GGUF** distribution:
`gemma-4-E4B-it-Q4_K_M.gguf` text + the `mmproj-BF16.gguf` vision tower
(`Gemma4Multimodal::load_gguf`), the same two files `gemma4.toml` feeds to
llama.cpp. Measured on the 24 GB M3 (Metal):

- The mmproj→HF tensor rename table was verified **bit-exact** against
  `unsloth/gemma-4-E4B-it`'s safetensors before any Rust was written (ranged
  HTTP fetch of just the tensors under test — norms carry no +1 offset,
  `v.patch_embd` is the patchify Linear as conv `permute(0,2,3,1)`, and every
  ClippedLinear clamp bound is present and identical).
- `multimodal_image` **PASS** (the "42" read correctly), `capital` **PASS**
  (the text path through the refactored `Gemma4Q` split unchanged),
  `multimodal_audio` **SKIP** (no audio tower on candle; `backend_can` now
  refuses audio for any candle config, mmprojPath or not).
- `gemma4_gguf_mmproj_vision_tower` (`make test-models`) loads both files and
  encodes a synthetic image on CPU in ~82 s.

This is also the only candle image path that fits the 12 GB reference card:
~5.5 GB of files against the safetensors checkpoint's ~16 GB, with the text
half running as the ordinary quantized `Gemma4Q` (host-resident PLE — see the
`gemma4-candle` entry above).

### The PLE row-gather made greedy decode non-reproducible (fixed)

2026-09-03, 24 GB M3. A full `gemma4-candle` matrix came back **6 / 10** where
the same binary had been 8 / 9 the night before, and the first suspicion was
the day's three commits — the mmproj load in particular, since it swaps
`Gemma4Q` for `Gemma4Multimodal`. It was neither the vision tower nor the
model: **`temperature = 0.0` was not reproducible**, so a testsuite run was a
draw even at greedy.

One prompt, one binary, identical `promptSha256`, greedy (no RNG on that path
at all):

| build | runs | distinct token streams |
|---|---|---|
| `6eb11cf` (before the row-gather) | 4 | **1** |
| `31a8d9b` | 8 | **4** — plus one CPU run that collapsed into token 262143 × 343 |
| `31a8d9b` + the fix | 5 (4 Metal, 1 CPU) | **1**, and equal to `6eb11cf`'s token for token |

The cause is a use-after-free in `QExperts::gather_rows`. It passed its
freshly-built `Vec` to `QStorage::from_data` as a `Cow::Owned`; candle's
`as_t_slice` takes that `Cow` **by value** and returns a slice borrowed from
it, so the `Vec` is dropped before the copy that follows reads it — `.to_vec()`
on the CPU, the buffer upload on Metal. Whether the stale read comes back
intact is the allocator's business, which is what made the model *sometimes*
wrong instead of broken. It is also why the corruption was device-independent:
the gather always dequantizes on the CPU and only then moves to the device.

Three things worth carrying forward:

- **The whole-table dequantization it replaced was never compared against.**
  The commit asserted "bit-identical to a whole-table dequantization" in prose;
  `gemma4_gguf_ple_gather_matches_per_row_dequantize` now checks it against
  `dequantize_expert`, row by row.
- **A value test does not guard this class of bug.** With the `Cow::Owned`
  reintroduced that test passes at both 30 KB and 10 MB. What caught it
  deterministically was a probe where the replacement allocation landed on the
  just-freed block and tripped std's `copy_nonoverlapping` overlap check. A
  permanent guard needs a sanitizer or Miri, not another assertion.
- **"Unexplained candle-Metal behavior" was this bug.** `docs/TODO.md` §3
  carried a `token_embd` gather that produced reproducible NaN logits on Metal
  and was filed against the backend. Same use-after-free, bigger buffer:
  ~2.7 GB is served by `mmap` and genuinely unmapped on free, so it failed
  every time where the PLE's few MB usually survived. That retry is open again.

Methodological note, since this cost the first two hours: the run that started
the investigation was read as a regression, and it was — but the four failing
testcases were not the evidence for it. `coding` passed on the very next run.
What actually localized the bug was fixing the sampler (greedy), holding the
prompt constant, and diffing token ids across repeats and across commits. A
pass count at `temperature = 0.7` cannot distinguish a regression from a draw,
and this backend's configs are still at 0.7.


## Settled questions

| Question | Answer | Landed in |
|---|---|---|
| Is `unsloth/Qwen3.8-27B-GGUF`'s embedded template the same as `Qwen/Qwen3.8-27B`'s on the Hub? | **No** — unsloth patched it ("developer role, merged system messages, tool calling"): it remaps `reasoning_effort = 'high'` → `'xhigh'` instead of raising, and merges leading system/developer messages instead of `raise_exception`. All cached snapshots and both quants carry identical bytes. | #191 — fixture replaced with the GGUF's bytes; two declared gaps closed (#175, part of #176); notes in `crates/gallium-agent/tests/fixtures/chat_templates/README.md` |
| Does #172 (KV cache defeated by recurrent-state rollback refusal) reproduce on LFM2? | **Yes** — `evaluated == input` on iteration 2, same signature as Qwen3.8. So the fix could be developed against the 4.9GB model. | #192 — verbatim assistant-turn replay for `is_recurrent() \|\| is_hybrid()` on the prose tool path; LFM2 iter 2 `evaluated 1767 → 34`. Since **replaced** by `Slot::checkpoint`, which reaches both render paths and does not put the model's own reasoning back in the prompt — see LFM2 above for the head-to-head |

## Still unverified against weights

- **The checkpoint path's *equivalence* on the Qwen 3.6 hybrid GGUFs.** They
  latch the same rollback-refusal gate as LFM2 and DeepSeek-V4 but have not been
  run through `tests/kv_state_spike.rs` at all. DeepSeek-V4 **has** now been
  measured, and it fails — restore-only Δlogit 1.69 — recorded in the
  DeepSeek-V4-Flash section above and `docs/OPTIMIZATION.md` §3.5. LFM2 is exact
  (0.000000). Qwen3.8-27B, Gemma 4, GPT-OSS and MiniMax are pure attention, take
  llama.cpp's partial trim, and never latch the gate or take a checkpoint at all.
  (The native-render-path staging for #172 is no longer a pending item: it was
  measured, declined, and then removed with the replay itself in #196.)
- **The candle backend's prior-reasoning path.** Its renderers drop prior-turn
  reasoning unconditionally and no `PromptRenderer` emits
  `ChatMessage::reasoning` (documented on `ModelProfile::preserve_prior_reasoning`).
  No config or testsuite backend exercises the Qwen candle path, so #185's
  candle half is template-tested only.
