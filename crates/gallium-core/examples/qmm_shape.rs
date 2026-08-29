//! How accurate is candle's quantized matmul, as a function of the input's shape?
//!
//! The question has a habit of coming back: a quantized multiply is not one
//! implementation, and which one runs depends on how many rows you hand it. This
//! isolates that from any model — one weight, one input, both paths, no
//! attention or residual stream to launder the answer through.
//!
//!   cargo run -p gallium-core --release --example qmm_shape
//!   GALLIUM_DEVICE=cpu cargo run -p gallium-core --release --example qmm_shape
//!
//! What it printed on Metal (M3) at an LFM2 expert projection's shape, and what
//! `docs/CANDLE_BACKEND.md` §6 is built on:
//!
//! ```text
//!   rows   max |delta|         scale    relative
//!      1      0.000009         4.965     1.87e-6
//!      2      0.006691         5.484     1.22e-3
//!      4      0.008257         6.000     1.38e-3
//!    200      0.009298         6.369     1.46e-3
//! ```
//!
//! One row is exact to six digits; two or more is ~1.4e-3 and **flat** from
//! there. Flat is the informative part: an error that does not grow with the
//! reduction length is not accumulation, it is a rounding step — the many-row
//! kernel dequantizes the weight into `half` threadgroup tiles, and 1e-3 is
//! f16's relative precision. ggml's kernel makes the same choice, so this is a
//! design decision inherited from it rather than a defect in the port.
//!
//! On **CUDA** (`--features cuda`, RTX 4070) the shape is the same but the 1-row
//! path is *not* exact: ~3.1e-3 at 1 row, ~5.3e-3 at ≥16. See
//! `docs/CANDLE_BACKEND.md` §6c — that kernel is the decode path, which still
//! agrees with llama.cpp at the model level, so it is not the CUDA prefill
//! divergence that section is about.
//!
//! The comparison is against dequantize-then-matmul in f32, which is the exact
//! answer for these weights: both sides read the same quantized bytes, so
//! nothing here measures quantization error.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Module, Result, Tensor};

/// The shape of one LFM2.5-8B-A1B expert projection — `d_ff` 1792, hidden 2048.
const OUT_DIM: usize = 1792;
const IN_DIM: usize = 2048;

fn main() -> Result<()> {
    let device = gallium_core::resolve_device(std::env::var("GALLIUM_DEVICE").ok().as_deref())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!(
        "device: {}   weight: {OUT_DIM}x{IN_DIM} Q4_K",
        gallium_core::device_name(&device)
    );

    // Structured rather than random, so a transposed or misindexed kernel shows
    // up as a large error instead of hiding in noise.
    let w: Vec<f32> = (0..OUT_DIM * IN_DIM)
        .map(|i| ((i % 61) as f32 - 30.0) / 31.0)
        .collect();
    let w = Tensor::from_vec(w, (OUT_DIM, IN_DIM), &device)?;
    let qt = QTensor::quantize(&w, GgmlDType::Q4K)?;

    // The exact answer: the same quantized bytes, expanded, multiplied in f32.
    let expanded = qt.dequantize(&device)?.t()?.contiguous()?;
    let quantized = QMatMul::from_qtensor(qt)?;

    println!(
        "{:>6}  {:>12}  {:>12}  {:>10}",
        "rows", "max |delta|", "scale", "relative"
    );
    for rows in [1usize, 2, 4, 8, 16, 64, 200] {
        let x: Vec<f32> = (0..rows * IN_DIM)
            .map(|i| ((i % 17) as f32 - 8.0) / 9.0)
            .collect();
        let x = Tensor::from_vec(x, (rows, IN_DIM), &device)?;

        let by_kernel = quantized.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let by_hand = x.matmul(&expanded)?.flatten_all()?.to_vec1::<f32>()?;

        let worst = by_kernel
            .iter()
            .zip(&by_hand)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = by_hand
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        println!(
            "{rows:>6}  {worst:>12.6}  {scale:>12.3}  {:>10.2e}",
            worst / scale
        );
    }
    Ok(())
}
