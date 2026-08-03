//! Where a decode step's time goes, per device.
//!
//! `#[ignore]`d like the model tests: these allocate GBs and exist to answer "why
//! is decode slow on this device", not to gate CI.
//!
//! Run with:
//!   cargo test --release -p gallium-core --test device_bench -- --ignored --nocapture
//!   GALLIUM_DEVICE=cpu cargo test --release … (same, on the CPU)
//!
//! Shapes are gemma-4-12B-it-qat's, read from its GGUF metadata: 48 layers, 16
//! query heads, and a heterogeneous attention stack — 40 sliding-window layers
//! (8 KV heads, head_dim 256) and 8 global layers (1 KV head, head_dim 512). The
//! 1-KV-head layers are the expensive ones: they expand 16×.
//!
//! The sliding-window layers are modelled at full context rather than at the
//! 1024-token window because that is what the decode path currently does (see
//! docs/TODO.md §1.1, sliding-window mask skipped at decode).

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

/// (kv_heads, head_dim, layer_count) per attention flavour in the real model.
const FLAVOURS: [(usize, usize, usize); 2] = [(8, 256, 40), (1, 512, 8)];
const Q_HEADS: usize = 16;
const CONTEXT: usize = 1577;
const STEPS: usize = 8;

fn device() -> Device {
    let spec = std::env::var("GALLIUM_DEVICE").ok();
    let d = gallium_core::resolve_device(spec.as_deref()).expect("device");
    eprintln!("device: {}", gallium_core::device_name(&d));
    d
}

fn sync(t: &Tensor) {
    t.sum_all().unwrap().to_scalar::<f32>().unwrap();
}

/// GQA head expansion as the models do it (`gemma4_q.rs::expand_gqa`): `expand`
/// then `contiguous`, materialising a full copy of K and V over the whole context
/// — per layer, per decode step. Cost grows with context length.
#[test]
#[ignore]
fn gqa_expand_per_step() {
    let dev = device();
    let mut bytes = 0usize;
    let mut last = None;

    // Allocate outside the timed region: the cache already exists in a real step.
    let caches: Vec<(Tensor, Tensor, usize, usize, usize)> = FLAVOURS
        .iter()
        .map(|&(h_kv, d, layers)| {
            let k = Tensor::zeros((1, h_kv, CONTEXT, d), DType::F32, &dev).unwrap();
            let v = Tensor::zeros((1, h_kv, CONTEXT, d), DType::F32, &dev).unwrap();
            sync(&k);
            (k, v, h_kv, d, layers)
        })
        .collect();

    let expand = |t: &Tensor, h_kv: usize, d: usize| {
        t.unsqueeze(2)
            .unwrap()
            .expand((1, h_kv, Q_HEADS / h_kv, CONTEXT, d))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((1, Q_HEADS, CONTEXT, d))
            .unwrap()
    };
    // Warm every kernel variant before timing.
    for (k, _, h_kv, d, _) in &caches {
        sync(&expand(k, *h_kv, *d));
    }

    let started = Instant::now();
    for (k, v, h_kv, d, layers) in &caches {
        for _ in 0..*layers {
            let _ke = expand(k, *h_kv, *d);
            last = Some(expand(v, *h_kv, *d));
            bytes += 2 * Q_HEADS * CONTEXT * d * 4;
        }
    }
    sync(&last.unwrap());
    eprintln!(
        "GQA expand, 48 layers × K+V at ctx {CONTEXT}: {:.0} ms/decode step \
         ({:.2} GB materialised)",
        started.elapsed().as_secs_f64() * 1000.0,
        bytes as f64 / 1e9,
    );
}

/// The two attention products end to end, both ways, at one decode step: the
/// expanding form the models used to run, against `gqa_scores`/`gqa_weighted_sum`.
///
/// Measured as a pair rather than as an expansion in isolation, because grouping Q
/// changes three things at once — it drops the K/V copy, shrinks `Kᵀ`'s copy to
/// `h_kv` heads, and hands the GEMM `rep` rows instead of one. Timing only the
/// expansion would credit it for the first and miss the rest.
#[test]
#[ignore]
fn attention_products_per_step() {
    use candle_core::D;

    let dev = device();
    let inputs: Vec<(Tensor, Tensor, Tensor, usize, usize)> = FLAVOURS
        .iter()
        .map(|&(h_kv, d, layers)| {
            // Decode: one query row against the whole cached context.
            let q = Tensor::randn(0f32, 1.0, (1, Q_HEADS, 1, d), &dev).unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, h_kv, CONTEXT, d), &dev).unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, h_kv, CONTEXT, d), &dev).unwrap();
            sync(&k);
            (q, k, v, h_kv, layers)
        })
        .collect();

    let expanding = |q: &Tensor, k: &Tensor, v: &Tensor, h_kv: usize| {
        let d = k.dim(3).unwrap();
        let expand = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((1, h_kv, Q_HEADS / h_kv, CONTEXT, d))
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((1, Q_HEADS, CONTEXT, d))
                .unwrap()
        };
        let (ke, ve) = (expand(k), expand(v));
        let scores = q
            .matmul(
                &ke.transpose(D::Minus2, D::Minus1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        candle_nn::ops::softmax_last_dim(&scores)
            .unwrap()
            .matmul(&ve)
            .unwrap()
    };
    let grouped = |q: &Tensor, k: &Tensor, v: &Tensor| {
        let scores = gallium_core::gqa_scores(q, k).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        gallium_core::gqa_weighted_sum(&probs, v).unwrap()
    };

    // The fast path is worth nothing if it computes something else, so check the
    // two agree here too — at real shapes, in real dtype, on the real device.
    for (q, k, v, h_kv, _) in &inputs {
        let want = expanding(q, k, v, *h_kv);
        let got = grouped(q, k, v);
        assert_eq!(want.dims(), got.dims());
        let diff = (&want - &got)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-3, "{h_kv} KV heads: paths differ by {diff}");
    }

    for (label, run) in [
        (
            "expanding K/V",
            &expanding as &dyn Fn(&Tensor, &Tensor, &Tensor, usize) -> Tensor,
        ),
        ("grouping Q", &|q, k, v, _| grouped(q, k, v)),
    ] {
        let mut last = None;
        let started = Instant::now();
        for (q, k, v, h_kv, layers) in &inputs {
            for _ in 0..*layers {
                last = Some(run(q, k, v, *h_kv));
            }
        }
        sync(&last.unwrap());
        eprintln!(
            "attention products, 48 layers at ctx {CONTEXT} — {label}: {:.0} ms/decode step",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// The copy that follows the expansion: `k.transpose(-2, -1).contiguous()` before
/// the scores matmul. Same volume again, but strided rather than a straight blit.
#[test]
#[ignore]
fn kt_contiguous_per_step() {
    use candle_core::D;

    let dev = device();
    let expanded: Vec<(Tensor, usize)> = FLAVOURS
        .iter()
        .map(|&(_, d, layers)| {
            let k = Tensor::zeros((1, Q_HEADS, CONTEXT, d), DType::F32, &dev).unwrap();
            sync(&k);
            (k, layers)
        })
        .collect();
    let transpose = |k: &Tensor| {
        k.transpose(D::Minus2, D::Minus1)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    for (k, _) in &expanded {
        sync(&transpose(k));
    }

    let mut last = None;
    let started = Instant::now();
    for (k, layers) in &expanded {
        for _ in 0..*layers {
            last = Some(transpose(k));
        }
    }
    sync(&last.unwrap());
    eprintln!(
        "Kᵀ contiguous, 48 layers at ctx {CONTEXT}: {:.0} ms/decode step",
        started.elapsed().as_secs_f64() * 1000.0
    );
}

/// What the KV cache does today (`kv_cache.rs::append`): `Tensor::cat` the whole
/// cache with the new step, per layer, per token.
#[test]
#[ignore]
fn kv_cache_cat_per_step() {
    let dev = device();
    let mut caches: Vec<(Tensor, Tensor, Tensor, usize)> = FLAVOURS
        .iter()
        .map(|&(h_kv, d, layers)| {
            let k = Tensor::zeros((1, h_kv, CONTEXT, d), DType::F32, &dev).unwrap();
            let v = Tensor::zeros((1, h_kv, CONTEXT, d), DType::F32, &dev).unwrap();
            let step = Tensor::zeros((1, h_kv, 1, d), DType::F32, &dev).unwrap();
            sync(&k);
            (k, v, step, layers)
        })
        .collect();

    let started = Instant::now();
    for _ in 0..STEPS {
        for (k, v, step, layers) in caches.iter_mut() {
            for _ in 0..*layers {
                *k = Tensor::cat(&[&*k, &*step], 2).unwrap();
                *v = Tensor::cat(&[&*v, &*step], 2).unwrap();
            }
        }
    }
    sync(&caches[0].0);
    eprintln!(
        "KV cat: {:.0} ms/decode step",
        started.elapsed().as_secs_f64() / STEPS as f64 * 1000.0
    );
}

/// The fix for the above: allocate the cache once at max length and write each
/// step into it in place.
#[test]
#[ignore]
fn kv_cache_slice_set_per_step() {
    let dev = device();
    let max_len = CONTEXT + STEPS + 1;
    let caches: Vec<(Tensor, Tensor, Tensor, usize)> = FLAVOURS
        .iter()
        .map(|&(h_kv, d, layers)| {
            let k = Tensor::zeros((1, h_kv, max_len, d), DType::F32, &dev).unwrap();
            let v = Tensor::zeros((1, h_kv, max_len, d), DType::F32, &dev).unwrap();
            let step = Tensor::zeros((1, h_kv, 1, d), DType::F32, &dev).unwrap();
            sync(&k);
            (k, v, step, layers)
        })
        .collect();

    let started = Instant::now();
    for s in 0..STEPS {
        for (k, v, step, layers) in caches.iter() {
            for _ in 0..*layers {
                k.slice_set(step, 2, CONTEXT + s).unwrap();
                v.slice_set(step, 2, CONTEXT + s).unwrap();
                // The forward pass reads only the filled prefix.
                let _used = k.narrow(2, 0, CONTEXT + s + 1).unwrap();
            }
        }
    }
    sync(&caches[0].0);
    eprintln!(
        "KV slice_set: {:.1} ms/decode step",
        started.elapsed().as_secs_f64() / STEPS as f64 * 1000.0
    );
}

/// Quantized matmul at decode shape (one row) versus prefill shape (many), the one
/// thing that structurally differs between the phases: candle dispatches `fwd_mv`
/// for a single row and `fwd_mm` above that.
///
/// Shape is one gemma-4-12B FFN projection: 3840 → 15360.
#[test]
#[ignore]
fn qmatmul_decode_vs_prefill_shape() {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::Module;

    let dev = device();
    let (hidden, out) = (3840usize, 15360usize);
    let w = Tensor::randn(0f32, 1.0, (out, hidden), &dev).unwrap();
    let qw = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
    let mm = QMatMul::from_qtensor(qw).unwrap();

    for (label, seq) in [("decode (1 row)", 1usize), ("prefill (256 rows)", 256)] {
        let x = Tensor::randn(0f32, 1.0, (seq, hidden), &dev).unwrap();
        sync(&mm.forward(&x).unwrap());

        // One sync at the end, not per call: a decode step issues hundreds of
        // matmuls before reading anything back, and a per-call readback measures
        // the round trip instead of the matmul.
        let iters = 50;
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(mm.forward(&x).unwrap());
        }
        sync(&last.unwrap());
        let per = started.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "{label}: {:.2} ms/call, {:.2} ms per row",
            per * 1000.0,
            per * 1000.0 / seq as f64
        );

        // candle's alternative, measured once: it dequantizes the whole weight per
        // call, so it only helps if the dequantized copy is cached.
        let started = Instant::now();
        sync(&mm.forward_via_f16(&x).unwrap());
        eprintln!(
            "{label} via f16 (dequantizes every call): {:.0} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Fixed per-op cost, independent of how much data the op touches. A decode step
/// is a few hundred small ops; if dispatch dominated, per-token time would be set
/// by this number times that count.
#[test]
#[ignore]
fn tiny_op_dispatch_cost() {
    let dev = device();
    let ops = 700;
    let mut x = Tensor::zeros((1, 256), DType::F32, &dev).unwrap();
    for _ in 0..20 {
        x = (&x + 1.0).unwrap();
    }
    sync(&x);

    let started = Instant::now();
    for _ in 0..ops {
        x = (&x + 1.0).unwrap();
    }
    sync(&x);
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "{ops} tiny ops: {:.0} ms total, {:.3} ms/op",
        elapsed * 1000.0,
        elapsed * 1000.0 / ops as f64
    );
}
