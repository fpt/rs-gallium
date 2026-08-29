# Candle Backend

GPU support for the native candle engine (`INFERENCE_ENGINE=candle`): what enabling
it took, what it bought, and where decode time still goes.

Most of this document is about **Metal** — the backend candle has had longest and
the machine of record below is a Mac. **CUDA** support (`--features cuda`, an
RTX 4070) is newer; where it behaves differently the section says so, and §6c
collects the CUDA-specific numerics findings.

The in-process llama.cpp engine has always used Metal on macOS and is unaffected by
any of this — it offloads through `llama-cpp-2`'s own `metal` feature.

**Machine and model of record for every number below:** Apple M3 (19 GB recommended
working set), `unsloth/gemma-4-12B-it-qat-GGUF` at `UD-Q4_K_XL`, release build,
prompt 1577 tokens. Read them as order-of-magnitude guides, not benchmarks: one
machine, one model, and sampling at `temperature 0.7` means two runs generate
different token counts.

**Second machine, written `M3/24 GB` below:** Apple M3, 24 GB, macOS 26.5.2, same
model, same prompt, same recipe. It exists in this document because the end-to-end
run the machine of record could never finish does finish there, so every claim that
was previously microbenchmark-only now has an end-to-end A/B behind it. Where both
machines have a number, both are given rather than one overwriting the other.

---

## Using it

```bash
# Metal on macOS, automatically
INFERENCE_ENGINE=candle gallium

# Pin the device — the same binary runs either, so a comparison is one env var
GALLIUM_DEVICE=cpu   INFERENCE_ENGINE=candle gallium
GALLIUM_DEVICE=metal INFERENCE_ENGINE=candle gallium
```

| `GALLIUM_DEVICE` | Meaning |
|---|---|
| unset / `auto` | best accelerator this build has and this machine can create, else CPU |
| `cpu` | CPU |
| `metal` / `metal:N` | Metal, ordinal `N` (also accepts `mps`) |
| `cuda` / `cuda:N` | CUDA, ordinal `N` (also accepts `gpu`) |

Naming a device that is not there is an **error**, not a silent CPU run — a
benchmark that quietly measured the wrong device is worse than one that refuses.
`auto` is the forgiving spelling and falls back with a warning.

Every run logs the device it resolved and its throughput, so a result can always be
attributed:

```
INFO gallium_agent::llm_candle: Candle device: metal
INFO gallium_agent::llm_candle: CandleProvider: metal — prefill 1577 tok in 19.29s
     (81.8 tok/s), decode 22 tok in 88.94s (0.2 tok/s)
INFO gallium_agent::llm_candle: per-token median … ms, slowest … ms
```

Prefill and decode are reported separately because they scale differently and a
single average hides which one a change moved. The per-token median/slowest line
separates "uniformly slow" from "one long stall".

### How the capability is compiled in

Metal support is a **build-time** fact; choosing it is a **runtime** one. macOS
builds get candle's Metal backend from per-target features on `gallium-core`:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { workspace = true, features = ["metal"] }
candle-nn = { workspace = true, features = ["metal"] }
```

Three things about that block are load-bearing:

- **`candle-nn/metal` is required, not optional.** The `metal_fwd` of
  `softmax_last_dim`, `silu`, `sigmoid`, `rotary_emb::rope`, and `rms_norm` — all of
  which gallium-core calls — sits behind candle-nn's own `metal` feature. With only
  `candle-core/metal`, those custom ops *error* on a Metal tensor rather than
  falling back to CPU.
- **One crate is enough.** Cargo unifies features per package, so enabling it on the
  lowest crate that touches candle covers every dependent, including
  `cargo test -p gallium-models`.
- **No `cfg` gating in our code.** Candle keeps a dummy Metal backend when the
  feature is off, so `Device::new_metal` always compiles and returns
  `NotCompiledWithMetalSupport`. `gallium_core::resolve_device` turns that into a
  fallback or an error, with no conditional compilation of our own.

Note that `gallium-agent`'s `metal` feature means something different: llama.cpp's
Metal, via `llama-cpp-2/metal`. The candle side has no feature flag at all.

---

## What had to change

**F64 RoPE tables — the load blocker.** `RoPE::new` built the position/frequency
outer product as F64 tensors and matmul'd them *on the device*. Metal has no F64
matmul, so **every** model failed at load with `Metal error mlx matmul doesn't
support F64`. The CPU had silently tolerated it for as long as the code existed.

The table is now built in F32. That is not a Metal concession — it is what the
references do: transformers keeps `inv_freq` in float32 and matmuls there, and
llama.cpp's rope carries a float theta. The scaling math (YaRN, Llama3, NTK) still
runs in f64 on the host, where the precision matters, and the table is cast to the
model dtype immediately afterwards.

**Metal constraints worth knowing before touching the models:**

- `qmatmul` requires **contiguous** input, rank ≥ 2, and produces F32. The GGUF path
  is F32 throughout, so no dtype change was needed — but a missed `.contiguous()`
  that CPU tolerated becomes a hard error.
- No F64 anywhere in a device-side matmul.

---

## Measured: prefill flies, decode does not

| | prefill (1577 tok) | decode | model load |
|---|---|---|---|
| CPU | 397.5 s (4.0 tok/s) | 1.27 s/token (0.8 tok/s) | seconds (mmap) |
| Metal | **19.3 s (81.8 tok/s)** | **4.04 s/token (0.2 tok/s)** | ~3.2 min |

Metal prefill is **~20× faster**. Metal decode is **~3× slower than CPU**, and model
load regressed badly (weights are copied into Metal buffers per tensor instead of
being mmap'd).

For a typical agent turn — large prompt, short reply — Metal still wins overall: it
saves ~378 s of prefill and loses ~2.8 s per generated token. For long generations,
`GALLIUM_DEVICE=cpu` is currently faster.

**That table predates `gqa.rs` and is the pre-change baseline.** Its CPU decode
figure was independently re-measured on `M3/24 GB` at 1.328 s/token against a
binary built at `44211ac`, which is the commit before the GQA change — near enough
to 1.27 s to treat the whole row as a baseline reading rather than a current one.
The post-change numbers are below.

> **The machine of record could not finish this run**, and that has not changed: a
> later attempt there managed 39 s to load, then over nine minutes without
> completing prefill plus eight decode tokens, where the table predicts about a
> minute. It **does** finish on `M3/24 GB` — the same recipe, cold, first try, in
> about 65 s of prefill-plus-decode. The most likely difference is headroom (19 GB
> recommended working set versus 24 GB of RAM, against a 6.3 GB model plus per-step
> temporaries) now that the GQA change has removed ~2.9 GB of those temporaries, but
> thermal and contention were never ruled out on the original box and still are not.
> Re-measure on the machine you care about. [Verifying a change end to
> end](#3-verifying-a-change-end-to-end) is the recipe, and it is an A/B against a
> baseline binary precisely because absolute numbers here have not proven portable.

### End to end, after `gqa.rs`

Measured on `M3/24 GB`. Same binary pair, same box, back to back: `44211ac`
(expanding K/V) against `478341a` (grouping Q). Metal is 7 pairs run in **both
orderings**; CPU is one pair, which its much tighter spread makes sufficient.

| | baseline | grouping Q | change |
|---|---|---|---|
| Metal decode | 4557 ms/tok (median of 7) | **3738 ms/tok** | **1.22×**, ~820 ms/step |
| CPU decode | 1328 ms/tok | **751 ms/tok** | **1.77×**, ~577 ms/step |
| Metal prefill | 38.4 tok/s (median) | 43.2 tok/s | indistinguishable from noise |
| CPU prefill | 431 s | 440 s | indistinguishable from noise |

Both decode savings land close to what the microbenchmark predicted in isolation —
820 ms observed against ~900 ms predicted on Metal, 577 against ~527 on the CPU —
which is the result that was missing when the change was first made. Prefill moves
in neither direction on either device, as the code implies: at prefill the two forms
issue the same arithmetic, and the expansion the change removes was amortised over
1577 rows rather than one.

**How much to trust the Metal row:** 6 of the 7 pairs favour grouping Q, which is
p≈0.13 by sign test alone — suggestive, not conclusive, on its own. It is the
agreement with the microbenchmark in *both* direction and magnitude that carries it.
The CPU row needs no such hedge. The one dissenting Metal pair is the one where the
baseline ran first after an idle gap, which is the ordering effect described under
[gotchas](#3-verifying-a-change-end-to-end).

---

## Where a decode step goes

Per decode step at context 1577, measured with `crates/gallium-core/tests/device_bench.rs`
at the model's real shapes (48 layers = 40 sliding-window with 8 KV heads and
head_dim 256, plus 8 global with **1** KV head and head_dim 512; 16 query heads):

| Cost | Metal | CPU |
|---|---|---|
| GQA `expand` + `contiguous` (2.89 GB materialised) | **700 ms** | 73 ms |
| `Kᵀ.contiguous()` before the scores matmul | 281 ms | **732 ms** |
| KV cache `Tensor::cat` | 30 ms | 32 ms |
| ~336 quantized matvecs (estimated from 0.71/0.64 ms per projection) | ~150 ms | ~215 ms |
| **accounted for** | ~1.16 s | ~1.05 s |
| **actually measured** | **4.04 s** | **1.27 s** |

The CPU column essentially adds up. The Metal column leaves ~2.9 s unexplained;
the working hypothesis is allocator and buffer-pool churn from ~4 GB of temporaries
created and dropped every step (candle's Metal pool calls `drop_unused_buffers` on
flush). That is the same volume the first two rows create, so the fix for them
should test the hypothesis.

The 1-KV-head global layers are the expensive ones: they expand 16× to reach 16
query heads.

### Fixed: the expansion is gone (`gqa.rs`)

The first two rows are one problem, not two, and both come from making K/V match Q's
head count. `gallium_core::gqa` does the two attention products the other way round:
it folds each group of `rep = h / h_kv` query heads into the *row* axis, which makes
the batch dims agree without touching K/V.

```text
Q  (b, h, s, d)      -> (b, h_kv, rep*s, d)   ⎫ batch dims agree, so this is one
Kᵀ (b, h_kv, d, t)                            ⎬ plain matmul: no expansion, no
=> (b, h_kv, rep*s, t) -> (b, h, s, t)        ⎭ broadcast, nothing materialised
```

Both reshapes are metadata-only, because a group's query heads are already adjacent:
head `i = j*rep + r` sits at row `(j*rep + r)*s + u`, which is row `r*s + u` of group
`j`. It also gives the GEMM a better shape — at decode `s == 1`, so the old form asked
for `h` matmuls of a single row each and this one asks for `h_kv` of `rep` rows.

Measured as a pair (`device_bench.rs::attention_products_per_step`), because the
change drops the K/V copy, shrinks `Kᵀ`'s copy to `h_kv` heads, and reshapes the GEMM
all at once — attributing it to any one of those would be arbitrary:

| 48 layers, ctx 1577 | Metal | CPU |
|---|---|---|
| expanding K/V (both matmuls) | 999 ms | 833 ms |
| grouping Q (both matmuls) | **105 ms** | **319 ms** |

A 9.5x cut on Metal and 2.6x on the CPU, and the 2.89 GB of per-step temporaries are
gone. All six attention sites use it — `attention.rs` ×2, `gemma4_q.rs` ×2,
`gpt_oss_q.rs`, `qwen35_q.rs`, `lfm2moe_q.rs` — so it is not a Gemma-only fix. The
`qwen35_q.rs` DeltaNet expansion is a *different*, tiled layout feeding a recurrence
rather than a matmul, and is deliberately left alone.

**Scope of that claim:** it is a microbenchmark of the two products in isolation, and
an A/B inside one process, which is the part it is trustworthy about. It says these
two matmuls got ~9.5× cheaper on Metal; on its own it does not say how much of a
whole decode step that was, because the run that would have shown it did not complete
on the machine of record. `M3/24 GB` has since supplied that half — **1.22× on Metal
and 1.77× on the CPU end to end**, saving 820 and 577 ms per step against the ~900
and ~527 ms this microbenchmark predicts. See [End to end, after
`gqa.rs`](#end-to-end-after-gqars). Correctness is separately covered: `gqa.rs`'s
unit tests check both products against the expanding form they replace across the
`(16,8)`, `(16,1)`, `(8,8)` and `(4,2)` groupings, and the bench asserts the two agree at the
model's real shapes on the real device before it times anything.

This also settles docs/TODO.md §1.10 (two GQA paths disagreeing about
`.contiguous()`) by removing the expanded tensor rather than by picking a spelling.

### Ruled out by measurement

Three plausible-sounding culprits are not the problem, and each was measured before
being dismissed:

| Suspect | Measurement | Verdict |
|---|---|---|
| Per-op dispatch overhead | 5 µs/op → ~3.5 ms for a 700-op step | not it |
| Quantized matvec kernel being slow at one row | Metal 0.71 ms vs CPU 0.64 ms per projection | not it — but note **no Metal advantage** either |
| `QMatMul::forward_via_f16` as a remedy | 64 ms/call on Metal, 67 on the CPU: it dequantizes the whole weight every call | dead end unless the dequantized copy is cached |

The `forward_via_f16` row read **3551 ms/call** until `M3/24 GB` measured 64 ms on
Metal and 67 on the CPU — a ~55× discrepancy that has no explanation, on a bench
that reproduces every other row to within a few percent. Treat the original figure
as an error. The verdict is unchanged, and for the same reason: 64 ms against the
0.67 ms of a direct matvec is still ~95× worse.

The matvec result is the crisp explanation of the whole table: at one row the
projections are memory-bandwidth-bound, and on unified memory the GPU has no
bandwidth advantage. Prefill's 256-row shape is compute-bound and Metal is ~30×
faster there (0.05 vs 1.52 ms/row). Decode does not gain because *matmul is not what
decode spends its time on* — copies are.

---

## Reproducing

Three levels, cheapest first. The first two are self-contained; the third needs a
multi-GB model and a machine with headroom.

### 1. Correctness (seconds, no model)

```bash
cargo test -p gallium-core gqa            # products vs the expanding form they replace
cargo test --workspace                    # everything
```

### 2. Per-step attribution (minutes, no model)

Synthetic tensors at the real model's shapes. `#[ignore]`d because it allocates GBs.

```bash
cargo test --release -p gallium-core --test device_bench -- --ignored --nocapture
GALLIUM_DEVICE=cpu \
  cargo test --release -p gallium-core --test device_bench -- --ignored --nocapture
```

`attention_products_per_step` is the A/B for the GQA change — it runs both forms in
one process and prints a line each, so the comparison is immune to machine-to-machine
drift. The others (`gqa_expand_per_step`, `kt_contiguous_per_step`,
`kv_cache_cat_per_step`, `kv_cache_slice_set_per_step`, `qmatmul_decode_vs_prefill_shape`,
`tiny_op_dispatch_cost`) attribute the remaining costs.

**Run one device at a time.** Two of these in parallel contend and both sets of
numbers become meaningless. The same goes for anything else heavy on the box.

### 3. Verifying a change end to end

This is what did not complete on the machine of record; it wants a box with real
memory headroom, and it ran cold and first-try on `M3/24 GB`. Everything below is
exact, so it can be run cold on another machine.

**Prerequisites.** The model of record (~6.3 GB) and a tokenizer, both fetched into
the HF cache on first use — from a network that can reach huggingface.co:

```bash
cargo build --release
mkdir -p /tmp/gallium-bench/run && cd /tmp/gallium-bench
```

**`bench.toml`** — deliberately no `systemPromptPath` and no `skillPaths`, so the
prompt is identical across runs:

```toml
[llm]
modelPath = "hf:unsloth/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
tokenizerPath = "unsloth/gemma-4-12b-it"   # this GGUF repo ships no tokenizer.json
temperature = 0.3
maxTokens = 8                              # raise once you know the per-token cost

[agent]
maxTurns = 1
```

**Run — one device at a time, from the empty directory:**

```bash
cd /tmp/gallium-bench/run
echo "Name three primary colors." | GALLIUM_DEVICE=metal INFERENCE_ENGINE=candle \
  RUST_LOG=info ../../../target/release/gallium --config ../bench.toml 2>&1 \
  | tee ../metal.log

# same again with GALLIUM_DEVICE=cpu, into cpu.log
```

Use `tee` (or redirect) rather than piping into `grep`: grep block-buffers, so a run
in progress looks silent and you cannot tell loading from hanging.

**What to record** — three lines, all at `RUST_LOG=info`:

```
Candle device: metal                             <- confirms no silent CPU fallback
CandleProvider: prompt_tokens=1577
CandleProvider: metal — prefill 1577 tok in 19.29s (81.8 tok/s),
                 decode 7 tok in 28.3s (0.2 tok/s)
CandleProvider: per-token median … ms, slowest … ms
```

**A/B against a baseline**, which is the part that actually answers "did this change
help": absolute numbers here have not proven portable, so compare two binaries on the
same box, back to back.

```bash
git stash push -u                 # or: git checkout <baseline-ref>
cargo build --release && cp target/release/gallium /tmp/gallium-bench/gallium_base
git stash pop                     # or: git checkout -
cargo build --release && cp target/release/gallium /tmp/gallium-bench/gallium_new
# then run each against the same bench.toml, same device, same directory
```

**Gotchas, all of them hit at least once:**

- **The 1577-token prompt is the tool catalog plus system prompt**, not the question.
  That is why a one-line question still measures a realistic agent turn — and why
  `maxTurns`/`maxTokens` are the only knobs that change the shape of the run.
- **Cold vs warm page cache dominates load time**: the same 6.3 GB model took ~3.2 min
  cold and 39–52 s warm (37–39 s on `M3/24 GB`). Discard the first run after a reboot.
- **The first Metal run after an idle gap has fast prefill, and it is not the binary.**
  On `M3/24 GB` every first-after-idle run measured 63–64 tok/s and every run that
  followed one measured 36–46, *regardless of which binary it was*. The machine of
  record's unreproducible 81.8 tok/s is most likely this. Prefill figures are only
  comparable between runs at the same position after an idle gap — and this is the
  single easiest way to manufacture a fake result here. It first appeared as an
  apparent 60% prefill regression that was entirely an artifact of run order.
- **Alternate the binaries *and* run both orderings.** Whichever binary runs second
  meets a hotter GPU. Running `new` then `base` four times in a row made baseline
  decode climb monotonically (3880 → 4403 → 4557 → 5120 ms) from thermal drift alone,
  which flatters `new` by roughly the size of the effect being measured. Only after
  the reverse ordering agreed was the 1.22× above worth reporting.
- **`maxTokens` is the decode budget.** Start at 8. At the worst rate seen (~4 s/token)
  32 tokens is over two minutes of decode on top of load and prefill.
- **Watch memory.** With the expansion gone the per-step temporaries are ~2.9 GB
  smaller. This did complete on `M3/24 GB` where it used to stall, which is itself
  part of the result — though that box also has 5 GB more RAM than the machine of
  record, so the two explanations are not separated.
- Sampling is at `temperature 0.3`, so runs generate different token counts; compare
  the per-token rates, never the wall clock.

---

## Remaining work

In priority order, by measured impact. These are model/core changes, not device
plumbing.

1. ~~**Stop materialising GQA.**~~ **Done and confirmed end to end** — `gqa.rs`,
   above. 999 → 105 ms/step on Metal and 833 → 319 ms on the CPU in the
   microbenchmark; 1.22× and 1.77× on whole decode steps in the `M3/24 GB` A/B.
   docs/TODO.md §1.10 settled with it.
2. **Cache K pre-transposed** so the scores matmul needs no `Kᵀ.contiguous()`. Note
   the payoff shrank with item 1: that copy now moves `h_kv` heads rather than `h`,
   so the 732 ms/step CPU figure in the table above no longer applies — re-measure
   before doing the work. Still the largest single copy left on the CPU.
3. **Preallocate the KV cache** and write with `slice_set` instead of `Tensor::cat`:
   30 ms → 0.7 ms measured. Unaffected by item 1, and now a larger share of what is
   left.
4. **Model load on Metal**: per-tensor buffer copies. Worth looking at whether the
   GGUF mmap can back Metal buffers directly — though see the note on the load figure
   below; it is a smaller problem than first recorded.
5. **Fixing the sliding-window mask at decode** (docs/TODO.md §1.1) would cut what
   attention touches at all: 40 of 48 layers would read 1024 positions instead of the
   full context. With the expansion gone this is now the biggest structural win left.
6. **The MoE path fanned out with rayon over one Metal command queue, and that was
   a correctness bug, not just a slow one.** ~~Untested on Metal~~ — LFM2.5 tested it:
   the model loaded, prefilled, produced **NaN logits**, and panicked in `sampling.rs`.
   Every closure in those expert loops dequantizes, allocates and enqueues on the same
   `Device`, whose Metal command buffer and buffer pool are not built for concurrent
   use from several threads. **Fixed** in `lfm2moe_q.rs`, `gpt_oss_q.rs` and
   `gemma4_q.rs::QGemmaMoe`, which now all call `gallium_core::par_map_on_cpu` instead
   of `par_iter()` directly: it fans out on the CPU and runs serially on an
   accelerator, which costs nothing real — the GPU serialises that work anyway. The
   rule lives in one place deliberately, so a fourth MoE model gets it by construction
   rather than by the author remembering. Only LFM2.5's failure was observed directly;
   the other two were the identical code shape and were fixed on that basis, not
   re-observed. (`qwen35_q.rs` was listed here in error: it has no MoE.)

   **The performance half is now measured, and it did dominate.** On
   LFM2.5-8B-A1B (24 blocks, 32 experts, top-4, `d_ff` 1792, hidden 2048) the
   forward expanded 24 × 4 × 3 = **288 expert weights per token**, 14.7 MB each —
   4.23 GB allocated and freed per token, which `scripts/memwatch.sh` sampled as
   resident memory cycling 8 GB → 12 GB → 8 GB, 47 GB taken and 42 GB returned
   over a run whose whole range is 12 GB. Decode ran at 1.1 tok/s.

   `QExperts::qmatmuls` builds one `QMatMul` per expert at load and multiplies
   against the quantized weight instead. Measured on the same prompt and machine:

   | expert weights | per token | decode |
   |---|---|---|
   | expand per token, re-uploading each time (what this was) | 940 ms | 1.1 tok/s |
   | expand per token, quantized bytes already resident | 1451 ms | 0.7 tok/s |
   | expand per token to f16, already resident | 1615 ms | 0.6 tok/s |
   | **multiply quantized (`QMatMul`)** | **37 ms** | **26.9 tok/s** |

   The middle two rows are the ones worth keeping: **the cost is the expansion,
   not the upload**, so nothing is recovered by keeping the quantized bytes on
   the device, and f16 is *worse* because the result is converted back.

   Caching the expansions does not fit and cannot be made to fit by holding only
   the hot ones. All 2304 expert-projections are 16.9 GB in f16 (33.8 in f32)
   against a 19 GB working set, and the routing is not concentrated: traced over
   296 decode steps, **every layer used all 32 of its experts**. A cache of 24 of
   32 per layer — 12.7 GB — would still miss 15% of selections, which at ~5 ms
   per expansion is 223 ms/token, six times worse than not expanding at all.
   (`gallium::moe=trace` logs each layer's selection, which is how that was
   measured and how it would be re-measured for another model.)

   **Correction — "the arithmetic is now what llama.cpp performs" was wrong, and
   the way it was wrong is the useful part.** Multiplying quantized cost the
   `refactoring` testcase (matrix 7/11 → 6/11) and the explanation offered was
   8-bit activations. Feeding both engines the *same token ids* and comparing
   logits (`gallium-agent/tests/engine_logits.rs`) says otherwise:

   | rows of the input | candle vs llama.cpp, top-1 logit |
   |---|---|
   | 1 — a decode step | 10.895035 vs 10.895246 (**0.0002**) |
   | 2 | 37.38982 vs 37.661972 (0.27) |
   | 14 — a prefill | 31.36 vs 30.84 (0.5) |

   candle's quantized multiply is **two implementations**: one row takes the
   matvec kernel ported from ggml and agrees to four decimals, more than one row
   takes a different path and drifts. So decode was never degraded — *prefill*
   was, and a prompt evaluated 0.5 of a logit off picks a different first token.

   What ggml does on Metal **for a single row** is better than either option
   candle offers: `kernel_mul_mv_q4_K_f32` reads activations as
   `device const float *`, unpacks the weight block in registers and accumulates
   in f32 — no expansion, no activation quantization. (Its *CPU* path does
   quantize activations, which is why llama.cpp on CPU and llama.cpp on Metal
   disagree with each other by 0.6 on this input — the accurate one is the GPU
   kernel, not ggml in general.)

   So `expert_matmul` routes by shape: quantized for one row, expanded for many.
   Expanding for a prefill costs one expansion per active expert-projection *per
   prefill* rather than per token, which is the distinction the first pass
   missed by treating this as one choice for the whole model.

   | expert path | matrix | score | decode |
   |---|---|---|---|
   | expand always (was) | 3774 s | 7/11 | 0.7 tok/s |
   | quantize always | 218 s | **6/11** | 26.9 tok/s |
   | **route by shape** | **820 s** | **7/11** | **26.6 tok/s** |

   Only `lfm2moe_q.rs` is changed. `gpt_oss_q.rs` and `gemma4_q.rs::QGemmaMoe`
   have the identical shape and neither model is cached on the machine that
   measured this, so they were left rather than changed unmeasured.

   docs/TODO.md §3.1, §3.2 cover the same code on the CPU side.

### 6b. What is *not* understood about the many-row path

The paragraph that used to end §6 proposed writing a gallium-side Metal kernel
modelled on ggml's, on the premise that "candle has no many-row equivalent". **That
premise is false and the plan is withdrawn.** `candle-metal-kernels` ships
`kernel_mul_mm_q4_K_f32`, generated from the same template ggml uses, dispatched
by `call_quantized_matmul_mm_t` whenever `dim(-2) != 1` — the same
`simdgroup_half8x8` tiles, the same shape. There is no missing kernel to write.

What differs is *when* each project reaches for it, and how accurate it is.

**The kernel's error, isolated from any model** — `cargo run -p gallium-core
--release --example qmm_shape`, one Q4_K weight at an LFM2 expert projection's
shape (1792×2048), against dequantize-then-matmul in f32 on the same bytes:

| rows | max abs delta | relative |
|---|---|---|
| 1 | 0.000009 | 1.9e-6 |
| 2 | 0.006691 | 1.2e-3 |
| 4 | 0.008257 | 1.4e-3 |
| 8 … 200 | 0.009298 | ~1.5e-3 |

One row is exact to six digits; two or more is ~1.4e-3 and then **flat**. Flat is
the informative part: an error that does not grow with the number of rows is not
accumulation, it is a rounding step — the mm kernel stages the dequantized weight
in `half` tiles, and 1e-3 is f16's relative precision. **ggml makes the same
choice in the same kernel**, so this is inherited design, not a defect in the
port.

**ggml switches on token count, candle on row count.** ggml has two thresholds in
`ggml-metal.m`: `ne11_mm_min = 8` for `mul_mat`, and `ne21_mm_id_min = 32` for
`mul_mat_id` — the MoE dispatch, keyed on the *total* tokens in the batch, not
per expert. So llama.cpp uses its f32-accumulating matvec below 32 tokens and its
f16-tile matmul above. candle's boundary is `dim(-2) == 1`. A comparison made at
a short prompt therefore measures a different pair of kernels than a comparison
made at a real prefill, which is what `GALLIUM_LOGITS_REPEAT` in
`gallium-agent/tests/engine_logits.rs` exists to cross.

**Crossing it inverts the picture.** Same token ids, candle on the shipped
shape-routed path (so candle is *exact* at both lengths):

| prompt | llama.cpp top-5 | candle top-5 |
|---|---|---|
| 16 tokens (ggml: `mul_mv_id`) | 25.791 … | 25.788 … — agree to **0.006** |
| 91 tokens (ggml: `mul_mm_id`) | 22.352, 22.241, 22.157, 21.949, 21.443 | 22.131, **21.700, 21.645**, 21.326, 21.116 |

At 91 tokens the engines are 0.22 apart on top-1 and **ranks 2 and 3 have
swapped** — and the side that moved is llama.cpp, which crossed its own threshold
into f16 tiles while candle's *expert* matmuls stayed exact. At a real prefill
length, gallium's candle path is the *more* accurate of the two.

(Two corrections from §6d, which was measured later. "candle stayed exact" is too
strong: `expert_matmul` routes only the MoE experts, so the attention
projections, the dense blocks' FFN and `lm_head` still take the f16-tile `mul_mm`
path at a prefill. And the hypothesis this section leaves open — that the f16
tiles are what costs `refactoring` — is contradicted there: the same divergence
is *larger* at one token, where that kernel does not run at all.)

**So the mechanism behind the testsuite result is still open.** The story that
would close it — "candle's mm path is inaccurate, llama.cpp's is exact, hence
`refactoring`" — does not survive the measurement above: at a prefill both
engines run an f16-tile MoE kernel, and llama.cpp's departure from exact is the
same order as candle's was. Two facts are solid and one link between them is not:

- Solid: candle's mm kernel carries ~1.4e-3 relative error at ≥2 rows.
- Solid, and reproducible over the whole matrix: routing by shape recovers
  `refactoring` (6/11 → 7/11) at 97% of the quantized path's speed.
- **Not established**: that the first *causes* the second. Both engines perturb a
  prefill by roughly this much; they perturb it *differently*, and `refactoring`
  evidently sits close enough to a decision boundary to flip on which
  perturbation it gets. A case that flips on 1e-3 is not a measurement of
  accuracy, and treating one run of it as one would be the same mistake this
  section already made once.

The shape routing therefore stands on the matrix score and the decode rate, not
on a settled mechanism. What would actually settle it is a per-layer comparison
against an f32 reference — where the divergence enters the residual stream, and
whether it is a few tokens near a tie or a broad drift — and that has not been
done. Until it is, the honest statement is that the fast path is fast, the routed
path scores better, and nobody here can yet say why 1e-3 decides a testcase.

### 6c. The same question on CUDA

Everything in §6 and §6b was measured on Metal. Re-run on an RTX 4070
(`--features cuda`, `GALLIUM_DEVICE=cuda`, `GALLIUM_LLAMA_GPU_LAYERS=999`), post-#204,
2026-08-29.

**#204's shape routing does not move the CUDA testsuite score.** `lfm2-candle` on
CUDA is **6 / 9 runnable** both before and after #204 — `arithmetic`, `capital`,
`coding`, `file_read`, `memory_state`, `needle_in_haystack` pass; `data_analysis`,
`refactoring`, `spec_discovery` fail. On Metal the same commit takes the matrix
6/11 → 7/11 by recovering `refactoring`. `data_analysis`'s *output* did change
across #204 on CUDA (pre: gave up parsing the CSV; post: computed a wrong
ordering), so the routing is moving CUDA numerics — just not across a grading
line.

**`engine_logits` on CUDA** — candle vs llama.cpp top-1 logit on the same token
ids, both fully GPU-resident:

| shape | tokens | Δ top-1 | top-5 overlap | argmax |
|---|---|---|---|---|
| decode | 1 | **0.012** | 5/5 | agree |
| prefill | 16 | 0.25 | 3/5 | agree |
| prefill | 121 | 0.29 | 4/5 | agree |
| prefill | 301 | 0.21 | 4/5 | agree |

Same *shape* as the Metal table in §6b — the 1-row path is tight, many-row
prefill drifts ~0.2–0.3 and swaps one or two of the top-5, argmax still agrees so
the test passes — but the reason is **not** the f16-tile `mul_mm` kernel #204
routed around. On CUDA the many-row path already goes through `dequantize` +
cuBLAS, and it still lands here.

**The 2×2 device cross-comparison** localises it. Top-1 logit for the argmax
token (440) at a 121-token prefill, one prompt, four implementations:

| implementation | logit(440) | vs candle CPU | vs llama CPU |
|---|---|---|---|
| candle CPU | 23.283 | — | 0.215 |
| llama.cpp CPU | 23.068 | 0.215 | — |
| llama.cpp CUDA | 23.256 | 0.027 | 0.188 |
| **candle CUDA** | **23.550** | **0.267** | **0.482** |

candle's CUDA forward is the **outlier of the four** — furthest from every other
implementation, including candle's own CPU reference. (**That reading does not
survive §6d**: candle's CPU forward is not a reference. It is left here because
the numbers are right and only the yardstick was wrong.) But llama.cpp's own CPU and
CUDA paths are already 0.19 apart on this input, so ~0.2 of cross-device drift at
a 121-token Q4_K prefill is not unique to candle; candle's is ~0.08 wider and in
its own direction.

**Ruled out on CUDA:**

- **Q4_K dequantization.** `cargo run -p gallium-core --release --features cuda
  --example qk_dequant_device` quantizes the same CPU f32 weight on each device
  and expands it: `cuda vs cpu quant` is **0.000000** — bit-identical. The
  dequant kernel is not the source.
- **TF32.** candle's f32 cuBLAS path (`gemm_strided_batched_f32`) uses
  `CUBLAS_COMPUTE_32F`; `MM_F32_REDUCED_PRECISION` defaults to `false`
  ("similar to pytorch"), so `CUBLAS_COMPUTE_32F_FAST_TF32` is never selected
  unless a caller opts in, and nothing here does. The matmul is honest fp32.
- **candle's CUDA q-matvec kernel** (the 1-row decode path) *is* lossy —
  `qmm_shape` on CUDA shows 3.1e-3 relative at 1 row and 5.3e-3 at ≥16, where the
  Metal kernel is 1.9e-6 at 1 row — but decode agrees with llama.cpp at the model
  level (Δ0.012 above), so this is not what fails the testsuite.

**What was thought to be left** — see §6d, which retracts the premise: two
*honest-fp32* matmuls — candle's `gemm` crate on CPU vs
cuBLAS SGEMM on CUDA, multiplying a **bit-identical** dequantized weight — produce
model-level logits 0.27 apart. fp32 GEMMs normally agree to ~1e-5 relative; this
is ~1e-2. Either a per-layer ~1e-4 is compounding through 24 residual layers into
0.27 at the head, or another candle CUDA operator outside the MoE (GQA attention,
short-conv, the DeltaNet recurrence, RMSNorm, `silu`) carries the difference and
the MoE matmul is a red herring. The `engine_logits` harness compares only the
final logits, so it cannot tell these apart. Same answer as §6b: it needs a
per-layer CPU-vs-CUDA hidden-state comparison, which has not been done.

### 6d. Correction: candle's CPU forward is not a reference

§6c makes candle CPU the yardstick for the other three implementations. It is
the **noisiest** candle path measured in this document.

`qmm_shape` on each device — the same Q4_K weight, against dequantize-then-matmul
in f32 *on that same device*, so nothing here is a cross-device comparison:

| rows | Metal | CPU | CUDA |
|---|---|---|---|
| 1 | **1.9e-6** | 3.8e-3 | 3.1e-3 |
| 2 | 1.2e-3 | 3.4e-3 | — |
| ≥ 8 | 1.5e-3 | 4.6e-3 | 5.3e-3 |

The mechanism is in candle's source rather than inferred from the numbers:
`BlockQ4K::VecDotType = BlockQ8K`, and the CPU matmul calls
`VecDotType::from_float` on the **left-hand side** — it quantizes the
*activations* to Q8_K, at every shape and every row count. §6 already recorded
that ggml's CPU path does this, which is why llama.cpp's own CPU and Metal
forwards sit 0.6 apart on one input. What was missed is that it is equally true
of candle's CPU path, which §6c then used as ground truth.

So "candle CUDA is the outlier of the four" does not follow. **Metal is the
accurate one** — by three orders of magnitude at one row — and CPU and CUDA sit
together in the same ~4e-3 band. Re-read against Metal instead, the 2×2 table
says something much duller: three of the four implementations quantize
something, and they land ~0.2–0.5 of a logit apart at the head.

It also dissolves §6c's closing puzzle. "Two honest-fp32 matmuls … multiplying a
bit-identical dequantized weight" describes only the **expert** matmuls, which
#204's `expert_matmul` routes to `dequantize` + `matmul` above one row. Every
other quantized linear in the model — the attention projections, the dense
blocks' FFN, `lm_head` — still goes through `QMatMul::forward`, which on CPU is
the Q8_K path. The CPU forward was never honest fp32, so there was no paradox to
explain.

**Per-layer, at last** — the measurement §6b and §6c both end on as missing.
`gallium_core::probe` fingerprints each stage of a forward pass onto the
`gallium::layers` trace target; `gallium-models/examples/lfm2_layers.rs` runs one
prefill of fixed synthetic ids so two runs cannot differ by their input; and
`scripts/layer_diff.py` aligns two logs. The logs are text and CPU is a target
every box has, so a Metal machine and a CUDA machine can be compared without
being the same machine — which here they are not.

CPU vs Metal, 121 tokens, one prefill, disagreement as a fraction of the signal:

| stage | | |
|---|---|---|
| 0 — embedding lookup | **0.00e+00** | the control |
| 1 — after block 0 | 2.2e-3 | already the size of a kernel error |
| 8 | 2.4e-2 | |
| 16 | 2.3e-2 | stopped climbing |
| 24 — into `lm_head` | 6.9e-3 | |

Stage 0 is exactly zero, which is what makes the rest readable: same ids, same
embedding rows, so everything below that line is arithmetic. The difference
enters immediately — in block 0, not anywhere specific to the MoE — reaches
~2.5e-2 by the middle of the stack and then stops growing. **No stage multiplies
it by more than 2.8×**, so it is not one operator: it is every quantized linear
adding its own, and the residual stream carrying the sum.

The shape dependence is the *opposite* of the f16-tile story. At **one** token
the same comparison ends eight times worse on the pointwise measure and twenty
times worse on rms (5.5e-2 and 1.1e-1, against 6.9e-3 and 5.6e-3 at 121). That is
exactly what the table above predicts — at one row Metal is exact and CPU is not,
so the gap is the whole of CPU's Q8_K error, while at 121 rows both are lossy and
partly in the same direction. It is not what "the many-row f16 kernel is the
problem" predicts, which is the hypothesis §6b was still holding open.

**CPU vs CUDA, on the CUDA box, same run.** The comparison above needs a CUDA
machine for its other half; here it is, `chan/signal` at 121 tokens:

| stage | CPU vs Metal | CPU vs CUDA |
|---|---|---|
| 0 — embedding | 0.00e+00 | 0.00e+00 |
| 1 — after block 0 | 2.2e-3 | 3.6e-3 |
| 8 | 2.4e-2 | 1.8e-2 |
| 14–16 | 2.3e-2 | 3.9e-2 (peak) |
| 24 — into `lm_head` | 6.9e-3 | 1.7e-2 |

Same signature: enters at block 0, ramps, plateaus by mid-stack, **largest
single-stage step 3.5×** (no operator singled out). The one-token run amplifies
it the same way — `chan/signal` 0.15–0.23 mid-stack against ~0.04 at 121, the
same "worse at one token" that rules the f16 kernel out — with one 4.9× step at
block 10 that appears *only* at one token, where CPU's own Q8_K error dominates
every stage and a single bad row is not a CUDA fact. At the 121-token prefill,
CUDA's per-stage shape is Metal's, not its own.

**What this settles and what it does not.** Settled: the divergence compounds
rather than stepping, it enters at the first block, and CUDA carries the same
per-stage shape as Metal rather than a step of its own — so "candle CUDA is
*unusually* wrong" has no support at the layer level, only the retracted
CPU-yardstick reading of §6c ever suggested it. Not settled: which of CPU and
CUDA is *closer* to truth, because nothing in this document has yet been compared
against an actual f32 reference — dequantizing the whole model and running it
unquantized. Until that exists, a ~0.2 logit spread between two Q4_K
implementations at a prefill is the expected size of the effect and not evidence
about any one backend.
