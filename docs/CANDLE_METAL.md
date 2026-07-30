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

```bash
cargo build --release

# End-to-end, one device each. Use a config with no systemPromptPath/skillPaths and
# run from an empty directory so the prompt is small and identical across runs.
echo "Name three primary colors." | GALLIUM_DEVICE=metal INFERENCE_ENGINE=gallium \
  ./target/release/gallium --config /path/to/bench.toml

# Per-step attribution (allocates GBs; #[ignore]d so it never runs in CI)
cargo test --release -p gallium-core --test device_bench -- --ignored --nocapture
GALLIUM_DEVICE=cpu cargo test --release -p gallium-core --test device_bench -- --ignored --nocapture
```

Run one device at a time. Two of these in parallel contend and both sets of numbers
become meaningless.

---

## Remaining work

In priority order, by measured impact. Items 1–3 are model/core changes, not device
plumbing, and 2 helps the CPU more than the GPU:

1. **Stop materialising GQA.** `gemma4_q.rs::expand_gqa` does `expand` +
   `contiguous`; use a broadcast matmul over `(b, h_kv, rep, s, d)` instead. Removes
   2.89 GB/step. See also docs/TODO.md §1.10, which flags the two GQA paths in
   `attention.rs` as inconsistent about `.contiguous()` — worth settling together.
2. **Cache K pre-transposed** so the scores matmul needs no `Kᵀ.contiguous()`.
   Biggest single CPU win available (732 ms/step).
3. **Preallocate the KV cache** and write with `slice_set` instead of `Tensor::cat`:
   30 ms → 0.7 ms measured.
4. **Model load on Metal** (~3.2 min): per-tensor buffer copies. Worth looking at
   whether the GGUF mmap can back Metal buffers directly.
5. **Fixing the sliding-window mask at decode** (docs/TODO.md §1.1) would also cut
   the expansion volume: 40 of 48 layers would hold 1024 tokens instead of the full
   context.
6. **The MoE path is untested on Metal.** The dense 12B model does not reach it, but
   `gemma4_q.rs::QGemmaMoe`, `gpt_oss_q.rs`, `qwen35_q.rs`, and `lfm2moe_q.rs`
   dequantize expert weights *inside* forward and fan out with rayon over a single
   Metal command queue. Expect that to dominate for those models (docs/TODO.md §3.1,
   §3.2 cover the same code on the CPU side).

Not yet measured: the per-token median/slowest distribution on an idle machine, which
would confirm whether the unexplained ~2.9 s is uniform or a stall.
