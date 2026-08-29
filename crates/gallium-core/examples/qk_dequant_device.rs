//! Does candle round-trip a Q4_K weight to the *same* floats on CPU and on the
//! accelerator?
//!
//! `qmm_shape.rs` measures candle's quantized *matmul* against dequantize-then-
//! matmul on one device. This asks the question one layer down: the model-level
//! comparison in `gallium-agent/tests/engine_logits.rs` finds candle's CUDA
//! prefill ~0.27 of a logit away from candle's *own* CPU prefill on the same
//! tokens, and the many-row expert path (`lfm2moe_q::expert_matmul`) is
//! `dequantize` + `matmul` — so either the quant/dequant kernels or the cuBLAS
//! matmul is the outlier. This isolates the quant+dequant half.
//!
//!   cargo run -p gallium-core --release --features cuda --example qk_dequant_device
//!
//! The same CPU f32 weight is quantized on each device and expanded there; the
//! deltas are against the original f32, plus the cross-device disagreement.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

const OUT_DIM: usize = 1792;
const IN_DIM: usize = 2048;

fn roundtrip(w_cpu: &Tensor, dev: &Device) -> candle_core::Result<Vec<f32>> {
    let qt = QTensor::quantize_onto(w_cpu, GgmlDType::Q4K, dev)?;
    qt.dequantize(dev)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn main() -> candle_core::Result<()> {
    let w: Vec<f32> = (0..OUT_DIM * IN_DIM)
        .map(|i| ((i % 61) as f32 - 30.0) / 31.0)
        .collect();
    let w_cpu = Tensor::from_vec(w.clone(), (OUT_DIM, IN_DIM), &Device::Cpu)?;
    let scale = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);

    let cpu = roundtrip(&w_cpu, &Device::Cpu)?;
    let l2 = |v: &[f32], u: &[f32]| -> f32 {
        (v.iter().zip(u).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / v.len() as f32).sqrt()
    };
    let linf = |v: &[f32], u: &[f32]| {
        v.iter()
            .zip(u)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    };

    println!("Q4_K {OUT_DIM}x{IN_DIM}   scale {scale:.4}");
    println!(
        "{:>18}  {:>12}  {:>12}  {:>10}",
        "", "rms |delta|", "max |delta|", "rel(max)"
    );
    println!(
        "{:>18}  {:>12.6}  {:>12.6}  {:>10.2e}",
        "cpu quant vs f32",
        l2(&cpu, &w),
        linf(&cpu, &w),
        linf(&cpu, &w) / scale
    );

    #[cfg(feature = "cuda")]
    {
        let dev = Device::new_cuda(0)?;
        let cuda = roundtrip(&w_cpu, &dev)?;
        println!(
            "{:>18}  {:>12.6}  {:>12.6}  {:>10.2e}",
            "cuda quant vs f32",
            l2(&cuda, &w),
            linf(&cuda, &w),
            linf(&cuda, &w) / scale
        );
        println!(
            "{:>18}  {:>12.6}  {:>12.6}  {:>10.2e}",
            "cuda vs cpu quant",
            l2(&cuda, &cpu),
            linf(&cuda, &cpu),
            linf(&cuda, &cpu) / scale
        );
    }
    Ok(())
}
