# Performance Notes

CPU profiling results and optimizations for the GGUF inference path (GPT-OSS Q4_K_M on x86-64).

Profiling uses gperftools (`libprofiler.so`, SIGPROF-based) inside the integration Docker image.
Run with `PROFILE=1` — output lands at `%TEMP%\callgraph-<testcase>-<model>.svg`.

---

## Optimization history

All measurements use total SIGPROF samples as a proxy for wall-clock time.
Baseline is the state after the initial Docker integration work (before any perf work).

| Step | Total samples | vs baseline |
|------|--------------|-------------|
| Baseline | 712,172 | 1× |
| AVX2 candle k-quant | ~660,000 | 1.08× |
| `from_vec` (zero-copy tensor) | 602,573 | 1.18× |
| Rayon parallel experts | 404,893 | 1.76× |
| Expert batching (prefill) | **100,691** | **7.1×** |

---

## Techniques

### 1. AVX2 for candle k-quant (`RUSTFLAGS`)

**Problem:** `candle_core::quantized::k_quant::vec_dot_unopt` was appearing at 4%+ in the profile — the scalar fallback for Q4_K dot products.

**Cause:** Candle's k-quant uses `#[cfg(target_feature = "avx2")]` at **compile time**, not runtime detection. Without `-C target-feature=+avx2,+fma` in `RUSTFLAGS`, even an AVX2-capable CPU falls through to the scalar path.

**Fix:** Added to `Dockerfile.integration` builder step:
```dockerfile
RUSTFLAGS="-C target-feature=+avx2,+fma -C force-frame-pointers=yes"
```
`force-frame-pointers=yes` is also required for gperftools to unwind stacks correctly through SIMD code.

---

### 2. AVX2 MXFP4 dequantization (`dequantize_mxfp4_avx2`)

**Problem:** `dequantize_mxfp4_into` was the top self-time node at ~39% of total. The scalar loop does a table lookup + f32 multiply per nibble.

**Fix:** `gallium_core::quantized::dequantize_mxfp4_avx2` — runtime-dispatched via `is_x86_feature_detected!("avx2")`. Processes one 32-element MXFP4 block per iteration:
- `pshufb` (`_mm_shuffle_epi8`) as a 16-entry in-register LUT: nibble → E2M1 i8 value
- `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`: i8 → i32 → f32 widening
- `_mm256_mul_ps` with broadcast scale
- Four `_mm256_storeu_ps` writes cover all 32 output floats

---

### 3. Zero-copy tensor creation (`Tensor::from_vec`)

**Problem:** The scratch-buffer approach (reuse `Vec<f32>`, pass to `Tensor::from_slice`) added a 40% memcpy overhead: `from_slice` allocates a new candle buffer and copies the entire expert weight matrix into it.

**Fix:** Use `Tensor::from_vec` which takes ownership of the `Vec` and hands its memory directly to candle — no copy. One fresh `Vec` per dequant call is cheaper than a reused buffer + memcpy for matrices this large (~128 KB per expert weight).

Also removed the redundant zero-initialization in `dequantize_mxfp4`:
```rust
// Before: vec![0f32; n_elems]  — allocates + memsets, then overwrites
// After:
let mut out = Vec::with_capacity(n_elems);
unsafe { out.set_len(n_elems) }; // dequantize_mxfp4_into writes every element
```

---

### 4. Rayon parallel experts

**Problem:** The 4 active MoE experts per token were processed sequentially, leaving ~82% of cores idle (CPU usage ~18%).

**Fix:** `gpt_oss_q.rs QMoEFFN::forward` — replaced the sequential expert loop with `par_iter()` from Rayon. Each expert's dequant + matmul is fully independent:
```rust
let expert_outs: Vec<Tensor> = indexed
    .par_iter()
    .map(|(expert_idx, weight)| { /* dequant + forward */ })
    .collect::<Result<Vec<_>>>()?;
```
Result: 1.76× speedup; CPU usage rose from ~18% to ~35%.

---

### 5. Expert batching (gather–compute–scatter)

**Problem:** Even with parallel experts, each expert still processed one token at a time — resulting in `num_tokens × k` dequantize calls and `num_tokens × k` GEMV operations (batch size = 1) per MoE layer. BLAS GEMV uses far fewer cores than GEMM.

**Fix:** Restructured `QMoEFFN::forward` to use a gather–compute–scatter pattern:

1. **Route:** build `expert_tokens[e] = [(token_idx, norm_weight), …]` across all tokens.
2. **Gather + compute** (parallel across experts): for each active expert, stack its assigned tokens into a `(n_e, hidden)` batch, dequantize the expert weights **once**, run one `(n_e, H) × (H, F)` GEMM.
3. **Scatter:** write weighted expert outputs back to per-token accumulators.

Key points:
- `broadcast_add` / `broadcast_mul` required for bias and weight scaling since candle's `+`/`*` operators do not auto-broadcast.
- Expert count drops from `num_tokens × k` to `n_active_experts` (≤ `n_experts`).
- During prefill, prompt tokens tend to concentrate on a small set of popular experts, so batches are large and GEMM efficiency is high.
- During generation (1 token), batches are size 1 — no regression over Rayon-only approach.

Result: **7.1× total speedup** from baseline; dominant cost is now `dequantize_expert` at 32.8% and `matmul` at 20.8%.

---

## Profile hot spots (final state)

| Function | Self % | Notes |
|---|---|---|
| `Tq2Tensor::dequantize_expert` | 32.8% | AVX2 MXFP4 → f32, one call per active expert per layer |
| `candle matmul` | 20.8% | Batched GEMM, now BLAS-parallelised |
| attention + norms + routing | ~46% | Previously buried under MoE overhead |

The remaining `dequantize_expert` cost is the fundamental work of reading ~17 bytes/block from mmap and writing 128 bytes/block of f32 — at or near memory bandwidth limits.

---

## Profiling setup

The integration Docker image includes gperftools and `google-pprof`. Run from PowerShell:

```powershell
docker run --rm -e PROFILE=1 `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  -v "${env:TEMP}:/logs" `
  gallium-integration coding gpt-oss-gguf
```

Output: `%TEMP%\callgraph-coding-gpt-oss-gguf.svg` (call graph) and `%TEMP%\cpu-coding-gpt-oss-gguf.prof` (raw profile).

Note: run from PowerShell, not Git Bash — Git Bash mangles the volume bind paths (`${TEMP}` expands to a Git Bash internal path that Docker Desktop does not bind to the Windows filesystem).
