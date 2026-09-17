//! Metal fused attention (`candle_nn::ops::sdpa`) against the matmul path
//! (`gqa_scores` → f32 softmax → `gqa_weighted_sum`) at Gemma 4 E4B shapes —
//! the issue #308 experiment, measured on the attention op alone so the
//! number is not swamped by model load, FS cache state or thermal drift
//! between whole-model runs.
//!
//! Both paths are spelled exactly as `gemma4_q.rs::QAttention` spells them.
//! K/V are narrowed views into a larger buffer with a non-zero start offset,
//! as they are when read out of a `KvCache` — the layout candle's Metal sdpa
//! read wrongly before huggingface/candle 0c5895368 (#3599, element offset
//! set as a byte offset), which is what the `max|Δ|` column guards.
//!
//! `cargo test -p gallium-core --release --test metal_sdpa_bench -- --ignored --nocapture`

use candle_core::{DType, Device, Result, Tensor};
use gallium_core::{gqa_scores, gqa_weighted_sum};
use std::time::Instant;

fn matmul_path(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let mut scores = gqa_scores(q, k)?.to_dtype(DType::F32)?;
    if let Some(mask) = mask {
        scores =
            scores.broadcast_add(&mask.to_dtype(scores.dtype())?.unsqueeze(0)?.unsqueeze(0)?)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
    gqa_weighted_sum(&probs, v)
}

fn sdpa_path(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let (k, v) = (k.clone(), v.clone());
    let (b, h, s, _) = q.dims4()?;
    let t = k.dim(2)?;
    match mask {
        None => candle_nn::ops::sdpa(q, &k, &v, None, false, 1.0, 1.0),
        Some(m) => {
            let m = m
                .to_dtype(q.dtype())?
                .reshape((1, 1, s, t))?
                .broadcast_as((b, h, s, t))?;
            candle_nn::ops::sdpa(q, &k, &v, Some(&m), false, 1.0, 1.0)
        }
    }
}

/// `(b, h_kv, t, d)` view with a non-zero start offset inside a `cap`-long buffer.
fn kv_view(
    dev: &Device,
    h_kv: usize,
    cap: usize,
    off: usize,
    t: usize,
    d: usize,
) -> Result<Tensor> {
    let buf = Tensor::randn(0f32, 1.0, (1, h_kv, cap, d), dev)?.to_dtype(DType::F16)?;
    buf.narrow(2, off, t)
}

fn causal_mask(s: usize, t: usize, dev: &Device) -> Result<Tensor> {
    // Query i sits at absolute position (t - s + i); key j is visible iff j <= that.
    let mut m = vec![0f32; s * t];
    for i in 0..s {
        for j in 0..t {
            if j > t - s + i {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (s, t), dev)
}

fn time(dev: &Device, iters: usize, f: &dyn Fn() -> Result<Tensor>) -> Result<f64> {
    for _ in 0..5 {
        f()?;
    }
    dev.synchronize()?;
    let start = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(f()?);
    }
    // Force completion of the whole queue before reading the clock.
    let _ = last
        .unwrap()
        .sum_all()?
        .to_dtype(DType::F32)?
        .to_vec0::<f32>()?;
    dev.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

#[test]
#[ignore = "Metal microbenchmark; run with --ignored --nocapture"]
fn metal_sdpa_vs_matmul_gemma4_shapes() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        eprintln!("SKIP: no Metal device");
        return Ok(());
    };
    // E4B: 8 q heads, 2 kv heads; sliding d=256 (window 512), global d=512.
    let (h, h_kv) = (8usize, 2usize);
    struct Case {
        name: &'static str,
        s: usize,
        t: usize,
        d: usize,
        masked: bool,
    }
    let cases = [
        Case {
            name: "decode  sliding  (s=1,  t=512,  d=256)",
            s: 1,
            t: 512,
            d: 256,
            masked: false,
        },
        Case {
            name: "decode  global   (s=1,  t=2220, d=512)",
            s: 1,
            t: 2220,
            d: 512,
            masked: false,
        },
        Case {
            name: "decode  global   (s=1,  t=8192, d=512)",
            s: 1,
            t: 8192,
            d: 512,
            masked: false,
        },
        Case {
            name: "prefill sliding  (s=512,t=1023, d=256)",
            s: 512,
            t: 1023,
            d: 256,
            masked: true,
        },
        Case {
            name: "prefill global   (s=512,t=2732, d=512)",
            s: 512,
            t: 2732,
            d: 512,
            masked: true,
        },
        Case {
            name: "prefill global   (s=512,t=8192, d=512)",
            s: 512,
            t: 8192,
            d: 512,
            masked: true,
        },
    ];
    eprintln!(
        "{:<42} {:>10} {:>10} {:>8} {:>8}",
        "case", "matmul ms", "sdpa ms", "ratio", "max|Δ|"
    );
    for c in &cases {
        let q = Tensor::randn(0f32, 1.0, (1, h, c.s, c.d), &dev)?.to_dtype(DType::F16)?;
        let cap = c.t + 1024;
        let k = kv_view(&dev, h_kv, cap, 300, c.t, c.d)?;
        let v = kv_view(&dev, h_kv, cap, 300, c.t, c.d)?;
        let mask = if c.masked {
            Some(causal_mask(c.s, c.t, &dev)?)
        } else {
            None
        };
        let a = matmul_path(&q, &k, &v, mask.as_ref())?.to_dtype(DType::F32)?;
        let b = sdpa_path(&q, &k, &v, mask.as_ref())?.to_dtype(DType::F32)?;
        let delta = (a - b)?.abs()?.max_all()?.to_vec0::<f32>()?;
        let iters = if c.s == 1 { 200 } else { 20 };
        let mm = time(&dev, iters, &|| matmul_path(&q, &k, &v, mask.as_ref()))?;
        let sd = time(&dev, iters, &|| sdpa_path(&q, &k, &v, mask.as_ref()))?;
        eprintln!(
            "{:<42} {mm:>10.3} {sd:>10.3} {:>7.2}x {delta:>8.4}",
            c.name,
            mm / sd
        );
        // randn inputs at d=512 give logits ~20σ wide and a near-one-hot
        // softmax, where f16 rounding shows up as ~1e-1 on single outputs; the
        // real-model bar (max |Δlogit| 0.027, identical greedy stream) is the
        // integration test. Here only a gross mismatch is an error.
        assert!(
            delta < 0.5,
            "{}: fused and matmul disagree by {delta}",
            c.name
        );
    }
    Ok(())
}
