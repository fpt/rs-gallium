# Optimization

Tuning the **llama.cpp backend for agent workloads** — a search over placement,
memory and execution settings, scored by how fast gallium actually completes
turns, with gallium itself as the harness.

This is a different question from the two performance documents already here,
and the three should not be confused:

| Document | Subject |
|---|---|
| [docs/performance.md](performance.md) | CPU profiling of the **candle** GGUF path — where the samples went, what SIMD and expert batching bought |
| [docs/CANDLE_METAL.md](CANDLE_METAL.md) | **candle** on Metal — throughput, and a per-step breakdown of decode |
| **this file** | **llama.cpp** settings, searched automatically, scored on agent turns |

Status: **measurement is in place; one structural fix has landed (KV cache
reuse, §2.1); the knobs are still not.** Everything under
[Roadmap](#roadmap) is planned, not built.

---

## 1. What is measured today

A model call is timed in two halves, `llm::Timing` hanging off `TokenUsage`:

| Field | Meaning |
|---|---|
| `prefill` | provider entry → first sampled token — tokenization, context setup, prompt eval, first sample |
| `decode` | first token → last token |
| `ttft` | the **first** call's prefill, never summed |
| `prefill_tokens` / `decode_tokens` | the tokens those two durations actually cover |
| `calls` | how many model calls the record covers |

Two halves rather than one total because they scale differently: prefill is one
forward over the whole prompt, decode is one forward per token. A setting that
doubles prefill throughput and costs 10% of decode looks like a wash in a
combined number, which is the exact confusion this search has to avoid.

Prefill is timed from the *provider's* entry — before tokenization, on both
local backends. `llm_local` also finds or builds a `LlamaContext` there, and all
of it is part of the wait; excluding any of it would report a TTFT nobody
experiences.

With a warm cache, `prefill_tokens` is what was **evaluated**, not the whole
prompt — see §2.1. Pricing the whole prompt against the time to evaluate its
suffix would report a throughput the hardware never reached, rising with how
well the cache worked.

Two rules keep a turn's aggregate honest:

- **`ttft` is the first call's and is never summed.** It is the wait before the
  turn showed any sign of life. A sum of every call's prefill is a latency no
  one sat through.
- **The timing carries its own token counts.** `decode_tokens` excludes each
  call's first token, which its prefill produced — `output_tokens - 1` over a
  five-call turn overstates decode throughput. Holding the counts inside the
  timing rather than reading them off the enclosing `TokenUsage` also means a
  total covering both timed and untimed calls divides timed durations by timed
  tokens only; the alternative prices another provider's output against this
  one's clock, which is wrong in the flattering direction and invisible.

An unmeasurable rate renders `n/a`, never `0.0` — a rate of zero and an absent
measurement must not look alike on a line someone is reading to compare two
configurations.

### Where the numbers surface

| Surface | What it shows |
|---|---|
| REPL, per model call | `⏱  ttft 95.45s · prefill 11843 tok 124.1 tok/s · decode 11 tok 13.8 tok/s` |
| REPL, per turn (📊 line) | first-call ttft, summed prefill and decode rates |
| `tracing` at INFO (`llm_local`, `llm_candle`) | the same split, plus candle's median/slowest per-token times |
| Trace JSON (`GALLIUM_TRACE=1`) | `usage.timing` → `TimingRecord` in ms, per step *and* per turn |

`TimingRecord` carries `ttftMs`, `prefillMs`, `decodeMs`, `prefillTokens`,
`decodeTokens`, and `calls`, so both rates can be computed from the record alone
— a trial's numbers are already machine-readable.

Providers that cannot measure leave `timing` absent rather than reporting
something plausible: OpenAI's blocking Responses API cannot say when generation
started, and charging the round trip to prefill would be a measurement of the
network.

### What is **not** measured

Peak VRAM, peak RAM, GPU/CPU utilization, memory-controller traffic, page
faults, disk reads, power. None of it is collected by gallium and none of it
should be — these are host facts, and the trial harness is the right place to
sample them (`nvidia-smi`, `rocm-smi`, `powermetrics`, `/proc`) alongside the
turn it is timing. The constraint set in [§5](#5-objective-and-constraints)
depends on them.

---

## 2. The first measurement, and what it changes

`gemma-4-E4B-it-Q4_K_M`, Metal, M3, a two-iteration agent turn (one `Bash` call):

```
⏱  ttft 95.45s · prefill 11843 tok 124.1 tok/s · decode 11 tok 13.8 tok/s
⏱  ttft 97.75s · prefill 11882 tok 121.6 tok/s · decode 30 tok 10.0 tok/s
📊 tokens: in=23725, out=43, total=23768 · 9% of 131072
   · ttft 95.45s · prefill 122.8 tok/s · decode 10.8 tok/s
```

**Prefill was 97% of the turn.** 193 seconds of prompt evaluation against 3
seconds of generation — and the second call re-prefilled the same 11.8k
transcript the first one had just processed, because `llm_local::generate` built
a new `LlamaContext` per call and dropped it.

Three consequences for how this search should be run:

1. **A `llama-bench` optimum is not the agent's optimum.** `pp512/tg128`
   weights decode heavily; a real agent turn is prefill-bound by an order of
   magnitude. Whatever wins the microbenchmark has to be re-scored on the macro
   workload before it means anything.
2. **The prompt is large before the user says anything interesting.** 11.8k
   tokens here is the system prompt plus `CLAUDE.md` (33 KB) plus the skill
   catalog plus the tool schemas. Prefill throughput is therefore the dominant
   term in the objective, and anything that shrinks the prompt competes
   directly with anything that speeds it up.
3. **KV reuse moves the optimum, not just the score** — which is why it was
   fixed first (issue #86) rather than tuned around. See below.

### 2.1 What KV reuse changed

`llm_local` now retains contexts in a slot pool and evaluates only the suffix of
each prompt. The same two-iteration workload, reuse off versus on, same machine,
same model, **identical answer and identical token counts**:

| | reuse off | reuse on |
|---|---|---|
| iteration 1 prefill | 16.93 s (2295 tok evaluated) | 11.62 s (2295 tok) |
| iteration 2 prefill | 15.91 s (2336 tok evaluated) | **0.16 s (29 tok)** |
| turn prefill | 32.8 s | 11.8 s |

The second iteration is the whole story: 2307 of 2336 prompt tokens came from
the cache, so 29 were evaluated instead of 2336.

Two things follow for the search. **Iteration count is no longer a prefill
multiplier** — a five-iteration turn now costs roughly one prefill, not five, so
workloads should be re-timed before any conclusions drawn from the numbers above
§2 are reused. And **the first prefill is now the whole prefill**, which makes
the fixed prompt (system prompt, `AGENTS.md`/`CLAUDE.md`, skill catalog, tool
schemas) a one-off cost per conversation rather than a per-iteration one — the
argument for trimming it is correspondingly weaker.

`GALLIUM_KV_CACHE_SLOTS=0` restores the old behaviour, which is how the table
above was produced and how any trial can be A/B'd.

---

## 3. Current knobs

### Reachable today

| Knob | How it is set | Notes |
|---|---|---|
| `n_gpu_layers` | `GALLIUM_GPU_LAYERS` / `[llm] gpuLayers` | default `999` = offload everything; `0` = CPU |
| `temperature` | `[llm] temperature` / `LLM_TEMPERATURE` | default 0.7 |
| `top_p` | `[llm] topP` / `LLM_TOP_P` | default unset — the sampler chain skips the stage entirely rather than running a `1.0` no-op |
| `max_tokens` | `[llm] maxTokens` / `MAX_TOKENS` | per-call generation budget |
| `mmproj` | `[llm] mmprojPath` / `MMPROJ_PATH` | multimodal only; follows the model's GPU decision |
| KV cache slots | `GALLIUM_KV_CACHE_SLOTS` (env only) | default `1`; `0` disables reuse. Each slot is a whole KV cache |
| MoE experts on CPU | `GALLIUM_CPU_MOE` / `[llm] cpuMoe` | default off; see [docs/LLAMA_CPU_MOE.md](LLAMA_CPU_MOE.md) — moves the `gpuLayers` ceiling, doesn't replace tuning it |
| build-time backend | cargo features `metal` / `cuda` / `vulkan` | Metal automatic on macOS |

That is the whole list. `[llm] contextWindow` is **not** one of them — it drives
compaction and the client's gauge, and never reaches llama.cpp.

### Fixed in code, and what would expose each

`llm_local.rs` sizes a context at `n_ctx.max(n_prompt + max_tokens)` rounded up
to 4096, from a hardcoded `LOCAL_CONTEXT_WINDOW` (8192), and sets `n_batch` to
that same size. Everything else is llama.cpp's default. All of the following are
one-line builder calls on `llama-cpp-2` 0.1.151, so the work is config plumbing,
not FFI:

| Knob | API | Today |
|---|---|---|
| `n_ctx` | `LlamaContextParams::with_n_ctx` | derived from prompt length, rounded to 4096 |
| `n_batch` | `with_n_batch` | set equal to `n_ctx` |
| `n_ubatch` | `with_n_ubatch` | llama.cpp default (512) |
| `n_threads` / `n_threads_batch` | `with_n_threads` / `with_n_threads_batch` | llama.cpp default |
| flash attention | `with_flash_attention_policy` | llama.cpp default (auto) |
| `cache_type_k` / `cache_type_v` | `with_type_k` / `with_type_v` (`KvCacheType`) | f16 |
| KV offload | `with_offload_kqv` | on |
| SWA full / unified KV | `with_swa_full` / `with_kv_unified` | defaults |
| mmap / mlock | `LlamaModelParams::with_use_mmap` / `with_use_mlock` | mmap on, mlock off |

MoE-experts-on-CPU is now wired up as `cpuMoe` (see
[docs/LLAMA_CPU_MOE.md](LLAMA_CPU_MOE.md)), through `add_cpu_moe_override` —
all-or-nothing, every layer's experts move together. llama.cpp's own
`--n-cpu-moe N` is graduated (only the first *N* layers' experts move), a
tensor buffer-type override matching `blk.<0..N-1>.ffn_(up|down|gate)_exps`;
`add_cpu_buft_override` takes exactly that kind of regex, so a layer-graduated
`nCpuMoe` integer knob is a regex-builder away, not new FFI, if the
all-or-nothing version turns out too coarse for some model/card pairing.

### The largest single lever, now pulled

**A `LlamaContext` per call, discarded afterwards** was the biggest cost in §2,
and it is gone: contexts are retained in a slot pool and only the divergent
suffix of a prompt is evaluated. See §2.1 for the measurement and CLAUDE.md for
the invariants that keep it correct.

`n_ctx` deserves a second look because of it. A slot is allocated once and
cannot grow, so a conversation that outgrows its slot rebuilds and loses its
cache — the 4096 rounding buys many iterations between rebuilds, but a
configured `n_ctx` that actually reached llama.cpp would let a long session be
sized once, up front. That plumbing is still Phase 0 work.

---

## 4. Determinism

A search needs trials that differ because of the settings, not because of the
sampler.

- The sampler seed is **fixed at 1234** (`llm_local::sample_until_done`), so a
  fixed prompt at a fixed temperature produces the same token stream every run.
  Token counts are therefore comparable across trials without averaging.
- `temperature = 0` for benchmark configs removes the remaining sensitivity to
  any sampler change.
- The *transcript* is the hazard, not the sampler. A tool whose output varies —
  a timestamp, a directory listing that a previous trial wrote into, `git
  status` — changes the second iteration's prompt and therefore its token count.
  Benchmark workloads must use tools with stable output in a fresh workspace,
  which is what `testsuite/runner.sh` already provides (isolated temp dir per
  run, and it is the approval workspace root).
- `INFERENCE_ENGINE=scripted` replays a recorded turn's *tool* calls. It is not
  a benchmark of inference — it replaces the model. Useful for validating the
  harness, useless for timing it.

---

## Roadmap

### Phase 0 — make the knobs settable

The blocker for everything else: you cannot search what you cannot set. Add an
`[llm.llamacpp]` table (env overrides in the usual precedence) covering the
table in §3, and record the resolved values somewhere a trial can read back —
the startup banner and, ideally, the trace, which today records what a turn
*cost* but nothing about the settings that produced it. Until then the trial
harness has to keep the knob values itself, and a mislabeled trial is a silently
poisoned dataset.

Deliverable: every row of §3 reachable from a config file, plus a
`configs/bench.toml` with `temperature = 0`.

### Phase 1 — microbenchmark

`llama-bench` over the placement knobs, for orientation and for a sanity check
that the numbers gallium reports agree with llama.cpp's own. Note that gallium
links llama.cpp through `llama-cpp-2` and builds no CLI tools, so this needs a
separate llama.cpp checkout — `references/setup.sh` already clones one
(gitignored, not built by cargo).

Expect the result to be *wrong for the agent* in the way §2 describes. That is
part of the finding, not a failure of the phase.

### Phase 2 — macrobenchmark

A fixed set of agent workloads, run end to end, scored on wall clock. The
material exists: `testsuite/` has nine testcases, per-backend TOMLs, an isolated
temp workspace per run, and `matrix_runner.sh` for testcase × backend. What it
needs is a timing-aware result record — turn on `GALLIUM_TRACE_DIR` and keep the
per-turn JSON.

Pick workloads that span the shape space rather than the difficulty space: a
short single-call turn (`capital`), a long-prompt tool turn (`file_read`,
`coding`), and a multi-iteration turn (`refactoring`). Prefill-bound and
decode-bound workloads will not agree on the best settings, and knowing where
they disagree is more useful than one blended score.

### Phase 3 — search

Optuna, **one objective plus constraints** rather than many objectives — with
several objectives most trials end up non-dominated and the study stops
discriminating.

Search hierarchically, because the interactions are strongest within groups and
weakest across them: placement (`n_gpu_layers`, `n_cpu_moe`) → memory (KV types,
`n_ctx`, offload) → execution (threads, batch, ubatch, flash attention). Fixing
the earlier groups while searching the later ones also makes the result
explainable, which is the point of Phase 5.

### Phase 4 — local sensitivity

Around the best point, vary one parameter at a time and record the delta. A
ranked list of "+1 gpu layer → +0.3 tok/s, +4 cpu_moe → −0.9 tok/s" is what
turns a winning configuration into an understood one, and it is cheap next to
the search that produced the point.

### Phase 5 — explanation

Have a model explain *why* a configuration wins, **constrained to the observed
quantities** — the utilization, memory-traffic and VRAM deltas between the two
configurations, not the parameter values alone. An explanation generated from
the parameters is a plausible story; one generated from the measurements is a
reading of the data. Record which observations were in the prompt.

### Phase 6 — cross-model reproduction

Re-run the best point and the sensitivity sweep on a second architecture. The
interesting outcome is a *disagreement* — MoE placement paying off on one model
and not another — because the explanation then has to reach architecture facts
(active experts vs. total, expert-weight share of the model) rather than
restating the tuning result.

---

## 5. Objective and constraints

Proposed, for Phase 3:

```
maximize   completed_turns / wall_clock_seconds     (over the fixed workload set)

subject to VRAM_peak  <= budget            (11.5 GB on a 12 GB card)
           RAM_peak   <= budget
           TTFT       <= ceiling
           no OOM
           no silent CPU fallback
           every workload still passes its testcase assertion
```

Notes on the shape:

- **The last constraint is not optional.** A KV cache quantized far enough down
  will be fast and wrong, and a throughput score cannot tell the difference. The
  testsuite assertions are what keep the search inside the set of configurations
  that still work.
- **"No silent CPU fallback"** needs an explicit check. It is the failure that
  looks like a legitimate slow trial, and a search that cannot see it will spend
  its budget mapping the performance of a device nobody meant to use.
- Turn completion, not tokens, is the numerator: a configuration that generates
  quickly but drives the model into extra ReAct iterations is not faster at the
  job.

---

## 6. What to record per trial

| Field | Source |
|---|---|
| every knob's resolved value | Phase 0 — harness today |
| model, quantization, engine | trace `model` / `engine` |
| prompt / generated tokens, per call and per turn | trace `usage`, `steps[].usage` |
| ttft, prefill, decode (ms) + the tokens each covers | trace `usage.timing` |
| turn wall clock | trace `duration_ms` |
| ReAct iterations, tool calls | trace `steps` |
| ending (completed / failed / interrupted) | trace `ending` |
| testcase pass/fail | `testsuite/runner.sh` |
| peak VRAM / RAM, GPU & CPU utilization, power | **not collected — harness must sample** |
| host identity, build features, llama.cpp revision | **not recorded — harness must stamp** |

The last two rows are the gap. Everything above them a trace already holds.

---

## Open questions

- **Does KV reuse land before or after the search?** It removes most of what is
  currently being measured. Tuning first means tuning a system that is about to
  change shape.
- **Is per-call context construction a measurable share of prefill?** It is
  inside the `prefill` number today and cannot be separated from prompt eval
  without a finer split. On a large model with a small prompt it may dominate.
- **How much does `n_ctx` sizing cost?** The context is currently sized per call
  from the prompt, so consecutive calls with growing transcripts allocate
  different-sized contexts. A fixed, larger `n_ctx` trades memory for allocator
  churn, and nothing has measured which way that goes.
- **Thermal drift over a long study.** Hundreds of trials on a warm machine will
  show a downward trend that has nothing to do with the parameters. Randomize
  trial order, and re-run the baseline periodically.
