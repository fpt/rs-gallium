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
| CUDA dev box | RTX 4070 12GB, 128GB RAM | `--features cuda`; runs the `*-cuda-12gb` configs with partial offload |
| Mac | Metal (automatic) | runs the small models; 27B-class only at low quant |

`make testsuite BACKENDS="<name>"` is the check; results land in
`testsuite/results/test_results_*.txt` (gitignored).

## Prompt-affecting changes and their verification

The 2026-08 template-fixture work (#181, #183–#187, #189) changed what several
models put in front of the model. Verified 2026-08-27 on the CUDA box:

| Change | Models affected | Status |
|---|---|---|
| #181 chat-template fixture harness | — (test infra) | n/a |
| #183 LFM2 renders its own template (was manual ChatML) | LFM2.5 | **verified** — see LFM2 below |
| #184 system messages merged when a template admits one | Qwen3.8 (and any single-system template) | **verified clean** — Qwen3.8-cuda-12gb matrix unchanged vs. the 2026-08-15 baseline |
| #185 prior-turn `reasoning_content` carried forward | Gemma 4, Qwen3.8, LFM2 | **verified** — no regression on Gemma 4 or Qwen3.8. On LFM2 it never became effective *through the template*: that family's `preserve_prior_reasoning` is `Some(false)`, its own template's default. The one thing that put prior reasoning back in front of it was #192's verbatim replay, which #196 removed — see LFM2 below |
| #186 `reasoningEffort` projected onto the qwen3 family's accepted set | Qwen3.8 | **verified clean** — see Q1 |
| #187 `qwen4exp` arch, protocol-downgrade `warn`, multi-block XML parsing | Qwen3.8-Flash-Next (unloadable), Qwen3.8-27B | template-level only; Flash-Next is blocked on [llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742) |
| #189 each family states its own prior-reasoning policy | all | **verified clean** — behaviour unchanged, matrices stable |

### Qwen3.8-27B (`qwen3.8-cuda-12gb`, Q3_K_XL, `gpuLayers = 30`)

2026-08-27: **10 / 11 pass**, only `multimodal_audio` failing — which is the
documented limitation, not a regression: Qwen3.8-27B's projector is vision-only
(`clip.has_vision_encoder`, no audio-encoder field). Matches the 2026-08-15
baseline (8/9, same one failure; `data_analysis` and `spec_discovery` are newer
cases and both pass). The four stacked changes (#184/#185/#186/#189) did not
regress it.

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
no effort set. The docs/gemma4.md thinking-loop warning did not reproduce at
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

### Gemma 4 26B-A4B (`gemma4-26b-cuda-12gb`, Q4_K_XL, `cpuMoe`, `gpuLayers = 20`)

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

- **Not #196 (`Slot::checkpoint`).** DeepSeek-V4 *does* run the checkpoint path —
  `llama_kv_cache_dsv4::seq_rm` refuses every real partial rollback, which is why
  the gate latches on an observed refusal rather than on `llm_arch_is_hybrid`
  (which omits `deepseek4`). But `file_read` fails identically with
  `GALLIUM_KV_CACHE_SLOTS=0` — slot pool and checkpoint code fully bypassed, 3/3
  — and the failure is on ReAct iteration 1, before any reuse could apply.
  Verbatim replay (#172 / #192) is not in the tree. The checkpoint work is clear.

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
smaller question left open at n=9; `docs/CANDLE_METAL.md` item 6 is where it would
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

And it was not the first sighting in this repo: `docs/gemma4.md` recorded
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
3774 s. `docs/CANDLE_METAL.md` §6 has the table and the two alternatives that
were measured and rejected.

It is **not free**, and the cost is on the side that matters here: `QMatMul` runs
ggml's kernel, which quantizes the activations to 8 bits, so the arithmetic
changed — and the `lfm2-candle` matrix went **7/11 → 6/11**, losing
`refactoring`. That is the same arithmetic llama.cpp performs for this GGUF, and
`refactoring` passes there, so this reads as the case sitting near a threshold
rather than as the quantized path being wrong; it is not re-diagnosed here.
A 100× decode speedup is what makes the case re-runnable at all.


## Settled questions

| Question | Answer | Landed in |
|---|---|---|
| Is `unsloth/Qwen3.8-27B-GGUF`'s embedded template the same as `Qwen/Qwen3.8-27B`'s on the Hub? | **No** — unsloth patched it ("developer role, merged system messages, tool calling"): it remaps `reasoning_effort = 'high'` → `'xhigh'` instead of raising, and merges leading system/developer messages instead of `raise_exception`. All cached snapshots and both quants carry identical bytes. | #191 — fixture replaced with the GGUF's bytes; two declared gaps closed (#175, part of #176); notes in `configs/qwen3.8.toml`, `fixtures/chat_templates/README.md` |
| Does #172 (KV cache defeated by recurrent-state rollback refusal) reproduce on LFM2? | **Yes** — `evaluated == input` on iteration 2, same signature as Qwen3.8. So the fix could be developed against the 4.9GB model. | #192 — verbatim assistant-turn replay for `is_recurrent() \|\| is_hybrid()` on the prose tool path; LFM2 iter 2 `evaluated 1767 → 34`. Since **replaced** by `Slot::checkpoint`, which reaches both render paths and does not put the model's own reasoning back in the prompt — see LFM2 above for the head-to-head |

## Still unverified against weights

- **The checkpoint path's *equivalence* on a second model that needs it.** LFM2
  is the only rollback-refusing model with a `tests/kv_state_spike.rs`-style
  logit-vector check. DeepSeek-V4 runs the same code — `llama_kv_cache_dsv4::seq_rm`
  refuses every real partial rollback while `llm_arch_is_hybrid` omits the arch,
  which is why the gate latches on an observed refusal rather than on that list —
  and it *was* run against weights on 2026-08-28 (full matrix, plus a
  `GALLIUM_KV_CACHE_SLOTS=0` A/B for the `file_read` regression): no
  checkpoint-shaped failure surfaced, and disabling the slot pool changed
  nothing, so the path is at least not *breaking* this model. What is still
  unmeasured is a restored-vs-fresh logit comparison on the dsv4 cache
  specifically. The Qwen 3.6 hybrid GGUFs have not been run at all. Qwen3.8-27B,
  Gemma 4, GPT-OSS and MiniMax are pure attention, take llama.cpp's partial trim,
  and never latch the gate or take a checkpoint at all. (The native-render-path
  staging for #172 is no longer a pending item: it was measured, declined, and
  then removed with the replay itself in #196.)
- **The candle backend's prior-reasoning path.** Its renderers drop prior-turn
  reasoning unconditionally and no `PromptRenderer` emits
  `ChatMessage::reasoning` (documented on `ModelProfile::preserve_prior_reasoning`).
  No config or testsuite backend exercises the Qwen candle path, so #185's
  candle half is template-tested only.
