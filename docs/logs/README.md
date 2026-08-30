# `gallium::layers` probe logs — CANDLE_BACKEND §6e

Committed so a **Metal** box can diff against them without a 128 GB machine (the
f32 reference) or an RTX 4070 (the CUDA forward). See `docs/CANDLE_BACKEND.md`
§6e for what they measure and #214 / #212 for what is still open.

## What is here

`gallium-models/examples/lfm2_layers.rs` run against `LiquidAI/LFM2.5-8B-A1B-GGUF`
`LFM2.5-8B-A1B-Q4_K_M.gguf`, one prefill of fixed synthetic ids, `RUST_LOG=gallium::layers=trace`,
`NO_COLOR=1`, on the CUDA dev box (RTX 4070, 128 GB RAM), 2026-08-30. Every
forward is bit-identical across repeat runs.

| file | `GALLIUM_DEVICE` | `GALLIUM_LAYERS_F32_REF` | what it is |
|---|---|---|---|
| `ref_<N>.log`  | `cpu`  | `1` | **f32 reference** — Q4_K weights dequantized to f32, plain f32 matmul, no Q8_K on activations |
| `cpu_<N>.log`  | `cpu`  | —   | candle CPU, quantized (Q4_K weight · Q8_K activations) |
| `cuda_<N>.log` | `cuda` | —   | candle CUDA, quantized |

`<N>` ∈ {1, 48, 121, 160, 301} — the prompt length in tokens (`GALLIUM_LAYERS_TOKENS`).
`stdout.log` is every run's `device=… tokens=… f32_ref=…` line plus its top-5.

Each `.log` is 25 `stage=` lines: stage 0 is the embedding lookup (the control,
always identical for identical ids), stages 1–24 are after each block.

## Using them from a Mac

```bash
# produce the Metal halves (needs the model in the HF cache)
for N in 1 48 121 160 301; do
  GALLIUM_DEVICE=metal GALLIUM_LAYERS_TOKENS=$N NO_COLOR=1 \
    RUST_LOG=gallium::layers=trace \
    cargo run -p gallium-models --release --example lfm2_layers 2> metal_$N.log
done

# the comparison §6e could not do: Metal vs the f32 reference (#212)
for N in 1 48 121 160 301; do
  echo "== $N =="; uv run python scripts/layer_diff.py docs/logs/ref_$N.log metal_$N.log
done

# and Metal vs candle CUDA directly, the #214 question
for N in 1 48 121 160 301; do
  echo "== $N =="; uv run python scripts/layer_diff.py docs/logs/cuda_$N.log metal_$N.log
done
```

`layer_diff.py`'s first argument is the reference. It reads only the first
forward pass in each file.

The Mac halves (`metal_<N>.log`, and an arm-CPU forward) are **regenerate-only,
not committed** — the Mac that produced them for §6f cannot push (work-account
credentials, proxy-broken SSH) and the log files could not be transferred. §6f in
`docs/CANDLE_BACKEND.md` carries the resulting numbers; re-run the block above to
reproduce them.

## Regenerating the CUDA-box halves

```bash
cargo build -p gallium-models --example lfm2_layers --features gallium-core/cuda --release
B=./target/release/examples/lfm2_layers
for N in 1 48 121 160 301; do
  GALLIUM_DEVICE=cpu  GALLIUM_LAYERS_F32_REF=1 GALLIUM_LAYERS_TOKENS=$N NO_COLOR=1 \
    RUST_LOG=gallium::layers=trace $B 2> docs/logs/ref_$N.log
  GALLIUM_DEVICE=cpu  GALLIUM_LAYERS_TOKENS=$N NO_COLOR=1 \
    RUST_LOG=gallium::layers=trace $B 2> docs/logs/cpu_$N.log
  GALLIUM_DEVICE=cuda GALLIUM_LAYERS_TOKENS=$N NO_COLOR=1 \
    RUST_LOG=gallium::layers=trace $B 2> docs/logs/cuda_$N.log
done
```

The f32 reference needs ~32 GB for the dequantized 8B weights, so it is CPU-only —
the same flag OOMs a 12 GB card.
