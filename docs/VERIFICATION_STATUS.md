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
| #185 prior-turn `reasoning_content` carried forward | Gemma 4, Qwen3.8, LFM2 | **verified** — no regression on Gemma 4 or Qwen3.8; on LFM2 it only became effective with #192 (see below) |
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

### LFM2.5-8B-A1B (`lfm2`, Q4_K_M)

Hybrid short-conv + GQA-MoE. After #183 it renders its own template; after #192
its prior assistant turns replay verbatim so the KV cache survives a ReAct turn.

- 2026-08-27, post-#192: **6 / 11 pass** (`arithmetic`, `capital`, `file_read`,
  `memory_state`, `needle_in_haystack`, `refactoring`), 3 fail (`coding`,
  `data_analysis`, `spec_discovery`), 2 skip (text-only, no projector). Up from
  5/11 pre-#192 — `refactoring` flipped to pass once the model saw its own prior
  reasoning and calls instead of gallium's reserialization.
- `coding` / `data_analysis` / `spec_discovery` fail on **#118**, not on
  anything the 2026-08 work changed: the model answers a write with a name-less
  `{"file_path": …, "content": …}` object that no wire parser claims. Re-ran the
  "claim LFM2's native tool format" experiment (which #182 had invalidated):
  routing LFM2 onto its `List of tools:` declaration changes nothing — 3/3 runs
  emit the same name-less object. See the #118 thread.

## Settled questions

| Question | Answer | Landed in |
|---|---|---|
| Is `unsloth/Qwen3.8-27B-GGUF`'s embedded template the same as `Qwen/Qwen3.8-27B`'s on the Hub? | **No** — unsloth patched it ("developer role, merged system messages, tool calling"): it remaps `reasoning_effort = 'high'` → `'xhigh'` instead of raising, and merges leading system/developer messages instead of `raise_exception`. All cached snapshots and both quants carry identical bytes. | #191 — fixture replaced with the GGUF's bytes; two declared gaps closed (#175, part of #176); notes in `configs/qwen3.8.toml`, `fixtures/chat_templates/README.md` |
| Does #172 (KV cache defeated by recurrent-state rollback refusal) reproduce on LFM2? | **Yes** — `evaluated == input` on iteration 2, same signature as Qwen3.8. So the fix could be developed against the 4.9GB model. | #192 — verbatim assistant-turn replay for `is_recurrent() \|\| is_hybrid()` on the prose tool path; LFM2 iter 2 `evaluated 1767 → 34` |

## Still unverified against weights

- **The native-tools render path for #172.** #192 fixes the prose path only.
  Qwen3.8, Gemma 4, GPT-OSS, MiniMax, DeepSeek go through `render_native`, which
  the sentinel replay does not touch. The #172 thread argues Qwen3.8's
  round-trip is close to lossless once #175/#185 are in, so it may need less
  than LFM2 did — but it has not been measured.
- **The candle backend's prior-reasoning path.** Its renderers drop prior-turn
  reasoning unconditionally and no `PromptRenderer` emits
  `ChatMessage::reasoning` (documented on `ModelProfile::preserve_prior_reasoning`).
  No config or testsuite backend exercises the Qwen candle path, so #185's
  candle half is template-tested only.
