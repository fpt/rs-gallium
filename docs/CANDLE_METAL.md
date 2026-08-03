# Candle Metal Backend

GPU support for the native candle engine (`INFERENCE_ENGINE=gallium`): what enabling
it took, what it bought, and where decode time still goes.

The in-process llama.cpp engine has always used Metal on macOS and is unaffected by
any of this — it offloads through `llama-cpp-2`'s own `metal` feature.

**Machine and model of record for every number below:** Apple M3 (19 GB recommended
working set), `unsloth/gemma-4-12B-it-qat-GGUF` at `UD-Q4_K_XL`, release build,
prompt 1577 tokens. Read them as order-of-magnitude guides, not benchmarks: one
machine, one model, and sampling at `temperature 0.7` means two runs generate
different token counts.

---

## Using it

```bash
# Metal on macOS, automatically
INFERENCE_ENGINE=gallium gallium

# Pin the device — the same binary runs either, so a comparison is one env var
GALLIUM_DEVICE=cpu   INFERENCE_ENGINE=gallium gallium
GALLIUM_DEVICE=metal INFERENCE_ENGINE=gallium gallium
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
INFO gallium_agent::llm_gallium: Gallium device: metal
INFO gallium_agent::llm_gallium: GalliumProvider: metal — prefill 1577 tok in 19.29s
     (81.8 tok/s), decode 22 tok in 88.94s (0.2 tok/s)
INFO gallium_agent::llm_gallium: per-token median … ms, slowest … ms
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

> **These end-to-end figures are from one machine on one day and have not been
> reproduced since.** A later attempt on the same M3 could not finish the same run at
> all — 39 s to load, then over nine minutes without completing prefill plus eight
> decode tokens, where these numbers predict about a minute. Whether that is memory
> pressure (19 GB recommended working set against a 6.3 GB model plus per-step
> temporaries), thermal, or contention was never established, so treat the table as
> indicative and re-measure on the machine you care about before drawing conclusions
> from it. [Verifying a change end to end](#verifying-a-change-end-to-end) is the
> recipe, and it is an A/B against a baseline binary precisely because absolute
> numbers here have not proven portable.

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
an A/B inside one process, which is the part it is trustworthy about. It is *not* an
end-to-end tok/s measurement — the run that would have produced one did not complete
on the machine of record (see the note above). What the numbers below say is that
these two matmuls got ~9.5x cheaper on Metal; what they do not say is how much of a
whole decode step that was. Correctness is separately covered: `gqa.rs`'s unit tests
check both products against the expanding form they replace across the `(16,8)`,
`(16,1)`, `(8,8)` and `(4,2)` groupings, and the bench asserts the two agree at the
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
| `QMatMul::forward_via_f16` as a remedy | 3551 ms/call: it dequantizes the whole weight every call | dead end unless the dequantized copy is cached |

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
memory headroom. Everything below is exact, so it can be run cold on another machine.

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
echo "Name three primary colors." | GALLIUM_DEVICE=metal INFERENCE_ENGINE=gallium \
  RUST_LOG=info ../../../target/release/gallium --config ../bench.toml 2>&1 \
  | tee ../metal.log

# same again with GALLIUM_DEVICE=cpu, into cpu.log
```

Use `tee` (or redirect) rather than piping into `grep`: grep block-buffers, so a run
in progress looks silent and you cannot tell loading from hanging.

**What to record** — three lines, all at `RUST_LOG=info`:

```
Gallium device: metal                             <- confirms no silent CPU fallback
GalliumProvider: prompt_tokens=1577
GalliumProvider: metal — prefill 1577 tok in 19.29s (81.8 tok/s),
                 decode 7 tok in 28.3s (0.2 tok/s)
GalliumProvider: per-token median … ms, slowest … ms
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
  cold and 39–52 s warm. Discard the first run after a reboot.
- **`maxTokens` is the decode budget.** Start at 8. At the worst rate seen (~4 s/token)
  32 tokens is over two minutes of decode on top of load and prefill.
- **Watch memory.** With the expansion gone the per-step temporaries should be ~2.9 GB
  smaller, so if this now completes where it used to stall, that is itself the result.
- Sampling is at `temperature 0.3`, so runs generate different token counts; compare
  the per-token rates, never the wall clock.

---

## Remaining work

In priority order, by measured impact. These are model/core changes, not device
plumbing.

1. ~~**Stop materialising GQA.**~~ **Done** — `gqa.rs`, above. 999 → 105 ms/step on
   Metal, 833 → 319 ms on the CPU, and docs/TODO.md §1.10 settled with it.
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
6. **The MoE path is untested on Metal.** The dense 12B model does not reach it, but
   `gemma4_q.rs::QGemmaMoe`, `gpt_oss_q.rs`, `qwen35_q.rs`, and `lfm2moe_q.rs`
   dequantize expert weights *inside* forward and fan out with rayon over a single
   Metal command queue. Expect that to dominate for those models (docs/TODO.md §3.1,
   §3.2 cover the same code on the CPU side).
