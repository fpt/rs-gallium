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

/// `docs/CANDLE_BACKEND.md` item 2: `gqa_scores` ends its one remaining copy with
/// `k.transpose(-2, -1).contiguous()` before the scores matmul. candle's matmul
/// forwards strides to gemm/cuBLAS, so the copy may be droppable — this A/Bs
/// `q · Kᵀ.contiguous()` against `q · Kᵀ` (strided) at real decode shape, on 48
/// layers, checking the two agree first. Post-GQA the transpose moves `h_kv`
/// heads, not `h`.
#[test]
#[ignore]
fn kt_contiguous_vs_strided_matmul_per_step() {
    use candle_core::D;

    let dev = device();
    let inputs: Vec<(Tensor, Tensor, usize)> = FLAVOURS
        .iter()
        .map(|&(h_kv, d, layers)| {
            // One query row per KV head against the whole cached context.
            let q = Tensor::randn(0f32, 1.0, (1, h_kv, 1, d), &dev).unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, h_kv, CONTEXT, d), &dev).unwrap();
            sync(&k);
            (q, k, layers)
        })
        .collect();

    let copy = |q: &Tensor, k: &Tensor| {
        q.matmul(
            &k.transpose(D::Minus2, D::Minus1)
                .unwrap()
                .contiguous()
                .unwrap(),
        )
        .unwrap()
    };
    let strided = |q: &Tensor, k: &Tensor| {
        q.matmul(&k.transpose(D::Minus2, D::Minus1).unwrap())
            .unwrap()
    };

    for (q, k, _) in &inputs {
        let d = (&copy(q, k) - &strided(q, k))
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-3, "copy vs strided differ by {d}");
    }

    for (label, run) in [
        (
            "Kᵀ copy + matmul",
            &copy as &dyn Fn(&Tensor, &Tensor) -> Tensor,
        ),
        ("Kᵀ strided matmul", &strided),
    ] {
        let mut last = None;
        let started = Instant::now();
        for (q, k, layers) in &inputs {
            for _ in 0..*layers {
                last = Some(run(q, k));
            }
        }
        sync(&last.unwrap());
        eprintln!(
            "{label}, 48 layers at ctx {CONTEXT}: {:.1} ms/decode step",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// What `kv_cache.rs::append` used to do before the preallocated buffer:
/// `Tensor::cat` the whole cache with the new step, per layer, per token. Kept
/// as the baseline half of the A/B.
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

/// The quantized projections of one Qwen3.8-9B layer at a 512-row prefill
/// chunk, Q4_K, Metal many-row kernel (`kernel_mul_mm_q4_K_f32`): the matmul
/// floor a prefill cannot go below, to compare against a measured forward.
/// Shapes from the 9B's config: hidden 4096, FFN 12288, DeltaNet qkv 8192 /
/// z 4096 / out 4096, attention q 8192 / k,v 1024 / o 4096.
#[test]
#[ignore]
fn qwen35_9b_projections_per_prefill_chunk() {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::Module;

    let dev = device();
    let rows = 512usize;
    // (label, out, in, count per layer, layers)
    let shapes: [(&str, usize, usize, usize, usize); 6] = [
        ("ffn gate/up 4096->12288", 12288, 4096, 2, 32),
        ("ffn down 12288->4096", 4096, 12288, 1, 32),
        ("deltanet qkv 4096->8192", 8192, 4096, 1, 24),
        ("deltanet z / out 4096->4096", 4096, 4096, 2, 24),
        ("attn q 4096->8192", 8192, 4096, 1, 8),
        ("attn k,v 4096->1024 + o 4096->4096", 4096, 4096, 3, 8),
    ];
    let mut total = 0.0;
    for (label, out, inp, per_layer, layers) in shapes {
        let w = Tensor::randn(0f32, 1.0, (out, inp), &dev).unwrap();
        let mm = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q4K).unwrap()).unwrap();
        let x = Tensor::randn(0f32, 1.0, (rows, inp), &dev).unwrap();
        sync(&mm.forward(&x).unwrap());
        let iters = 10;
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(mm.forward(&x).unwrap());
        }
        sync(&last.unwrap());
        let per = started.elapsed().as_secs_f64() / iters as f64;
        let model = per * (per_layer * layers) as f64;
        total += model;
        let gflops = 2.0 * rows as f64 * out as f64 * inp as f64 / per / 1e9;
        eprintln!(
            "{label}: {:.2} ms/call ({gflops:.0} GFLOP/s) x {per_layer} x {layers} layers = {:.0} ms per 512-row forward",
            per * 1000.0,
            model * 1000.0
        );
    }
    eprintln!(
        "all projections: {:.2} s per 512-row forward -> {:.1} s for a 2217-token prompt (5 forwards)",
        total,
        total * 5.0
    );
}

/// The elementwise ops a DeltaNet layer runs over its (512, 8192) conv
/// activation, per dispatch: is a `broadcast_mul` against a (1, c) row as
/// cheap as a same-shape `mul`, and what does one such op cost against the
/// device's bandwidth? Answers whether `causal_conv1d`'s k shifted
/// broadcast multiply-adds are paying for the broadcast or for the dispatch.
#[test]
#[ignore]
fn deltanet_conv_elementwise_per_op() {
    let dev = device();
    let (s, c) = (512usize, 8192usize);
    let x = Tensor::randn(0f32, 1.0, (1, s, c), &dev).unwrap();
    let w_row = Tensor::randn(0f32, 1.0, (1, c), &dev).unwrap();
    let w_full = w_row.broadcast_as((1, s, c)).unwrap().contiguous().unwrap();
    sync(&x);
    let bytes = (s * c * 4 * 2) as f64; // read + write
    let iters = 40;
    let cases: Vec<(&str, Box<dyn Fn() -> Tensor>)> = vec![
        (
            "broadcast_mul (1,c) row",
            Box::new(|| x.broadcast_mul(&w_row).unwrap()),
        ),
        ("mul same shape", Box::new(|| (&x * &w_full).unwrap())),
        ("add same shape", Box::new(|| (&x + &w_full).unwrap())),
        (
            "narrow(1, 1, s-1) + contiguous",
            Box::new(|| x.narrow(1, 1, s - 1).unwrap().contiguous().unwrap()),
        ),
        ("silu", Box::new(|| candle_nn::ops::silu(&x).unwrap())),
        (
            "cat along seq (k-1 rows + x)",
            Box::new(|| {
                let pad = x.narrow(1, 0, 3).unwrap();
                Tensor::cat(&[&pad, &x], 1).unwrap()
            }),
        ),
    ];
    for (label, f) in cases {
        sync(&f());
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(f());
        }
        sync(&last.unwrap());
        let per = started.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "{label}: {:.2} ms/op ({:.0} GB/s effective)",
            per * 1000.0,
            bytes / per / 1e9
        );
    }
}

/// The strided-kernel costs the chunked DeltaNet pays per layer, isolated: a
/// row-scaling broadcast over the last dim, a head/seq transpose copy, a
/// broadcast-then-contiguous tile, and the diagonal-matmul alternative to the
/// row scaling.
#[test]
#[ignore]
fn deltanet_chunk_strided_ops_per_op() {
    let dev = device();
    let (s, h, d, c) = (512usize, 32usize, 128usize, 64usize);
    let n = s / c;
    let k = Tensor::randn(0f32, 1.0, (1, h, n, c, d), &dev).unwrap(); // 8 MB
    let col = Tensor::randn(0f32, 1.0, (1, h, n, c, 1), &dev).unwrap();
    let x_sh = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev).unwrap();
    let w_row = Tensor::randn(0f32, 1.0, (1, 8192), &dev).unwrap();
    let eye = Tensor::eye(c, DType::F32, &dev).unwrap();
    sync(&k);
    let iters = 40;
    let cases: Vec<(&str, Box<dyn Fn() -> Tensor>)> = vec![
        (
            "row-scale: k.broadcast_mul((..,c,1))",
            Box::new(|| k.broadcast_mul(&col).unwrap()),
        ),
        (
            "diag(col) as (..,c,c) via broadcast_mul(eye)",
            Box::new(|| col.broadcast_mul(&eye).unwrap()),
        ),
        ("row-scale via bmm(diag, k) (diag prebuilt)", {
            let diag = col
                .broadcast_mul(&eye)
                .unwrap()
                .reshape((h * n, c, c))
                .unwrap();
            let k3 = k.reshape((h * n, c, d)).unwrap();
            Box::new(move || diag.matmul(&k3).unwrap())
        }),
        (
            "transpose(1,2).contiguous() (1,s,h,d)->(1,h,s,d)",
            Box::new(|| x_sh.transpose(1, 2).unwrap().contiguous().unwrap()),
        ),
        (
            "transpose(3,4).contiguous() k -> kT",
            Box::new(|| k.transpose(3, 4).unwrap().contiguous().unwrap()),
        ),
        (
            "tile (1,c)->(1,s,c) broadcast_as+contiguous",
            Box::new(|| {
                w_row
                    .broadcast_as((1, s, 8192))
                    .unwrap()
                    .contiguous()
                    .unwrap()
            }),
        ),
        (
            "same-shape mul (1,h,n,c,d)",
            Box::new(|| (&k * &k).unwrap()),
        ),
    ];
    for (label, f) in cases {
        sync(&f());
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(f());
        }
        sync(&last.unwrap());
        eprintln!(
            "{label}: {:.2} ms/op",
            started.elapsed().as_secs_f64() / iters as f64 * 1000.0
        );
    }
}

/// Cheaper spellings of the chunked DeltaNet's strided ops: a diagonal matrix
/// from an outer-product matmul plus a same-shape mask, a transposed matmul
/// operand handed to `matmul` as a strided view, and the cost of expanding
/// the (c, c) constants once.
#[test]
#[ignore]
fn deltanet_chunk_strided_alternatives() {
    let dev = device();
    let (s, h, d, c) = (512usize, 32usize, 128usize, 64usize);
    let n = s / c;
    let bt = h * n;
    let k3 = Tensor::randn(0f32, 1.0, (bt, c, d), &dev).unwrap();
    let col = Tensor::randn(0f32, 1.0, (bt, c, 1), &dev).unwrap();
    let ones_row = Tensor::ones((bt, 1, c), DType::F32, &dev).unwrap();
    let eye_full = Tensor::eye(c, DType::F32, &dev)
        .unwrap()
        .broadcast_as((bt, c, c))
        .unwrap()
        .contiguous()
        .unwrap();
    sync(&eye_full);
    let iters = 40;
    let time = |label: &str, f: &dyn Fn() -> Tensor| {
        sync(&f());
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(f());
        }
        sync(&last.unwrap());
        eprintln!(
            "{label}: {:.2} ms/op",
            started.elapsed().as_secs_f64() / iters as f64 * 1000.0
        );
    };
    time("diag via bmm(col, ones_row) * eye_full", &|| {
        (col.matmul(&ones_row).unwrap() * &eye_full).unwrap()
    });
    time("outer bmm(col, ones_row) alone", &|| {
        col.matmul(&ones_row).unwrap()
    });
    time("expand eye (c,c) -> (bt,c,c) contiguous", &|| {
        Tensor::eye(c, DType::F32, &dev)
            .unwrap()
            .broadcast_as((bt, c, c))
            .unwrap()
            .contiguous()
            .unwrap()
    });
    time("k @ k^T with contiguous transpose", &|| {
        k3.matmul(&k3.transpose(1, 2).unwrap().contiguous().unwrap())
            .unwrap()
    });
    match k3.matmul(&k3.transpose(1, 2).unwrap()) {
        Ok(_) => time("k @ k^T with strided transpose (no copy)", &|| {
            k3.matmul(&k3.transpose(1, 2).unwrap()).unwrap()
        }),
        Err(e) => eprintln!("strided-transpose matmul refused: {e}"),
    }
    let cum = Tensor::randn(0f32, 1.0, (bt, c, 1), &dev).unwrap();
    time("pairwise cum_i - cum_j via two outer bmm + sub", &|| {
        let a = cum.matmul(&ones_row).unwrap();
        let b = ones_row
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .matmul(&cum.transpose(1, 2).unwrap().contiguous().unwrap())
            .unwrap();
        (a - b).unwrap()
    });
    time("pairwise via broadcast_sub", &|| {
        cum.broadcast_sub(&cum.transpose(1, 2).unwrap()).unwrap()
    });
}

/// Decode-shape A/B for the DeltaNet spellings that were changed for the
/// prefill's sake: at one token the tensors are tiny and a fused custom op or
/// an expand-then-multiply may cost more than the naive formula it replaced.
/// Shapes are Qwen3.8-9B's: 32 heads, head dim 128, state (1, 32, 128, 128).
#[test]
#[ignore]
fn deltanet_decode_shape_ab() {
    let dev = device();
    let (h, d) = (32usize, 128usize);
    let iters = 200;
    let time = |label: &str, f: &dyn Fn() -> Tensor| {
        sync(&f());
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            last = Some(f());
        }
        sync(&last.unwrap());
        eprintln!(
            "{label}: {:.3} ms/op",
            started.elapsed().as_secs_f64() / iters as f64 * 1000.0
        );
    };
    for (label, s) in [("decode", 1usize), ("prefill", 512usize)] {
        let x = Tensor::randn(0f32, 1.0, (1, s, h, d), &dev).unwrap();
        let alpha = Tensor::full((d as f32).powf(-0.5), d, &dev).unwrap();
        time(
            &format!("{label} l2norm naive (sqr,sum,add,sqrt,broadcast_div)"),
            &|| {
                let n = (x.sqr().unwrap().sum_keepdim(3).unwrap() + 1e-6)
                    .unwrap()
                    .sqrt()
                    .unwrap();
                x.broadcast_div(&n).unwrap()
            },
        );
        time(
            &format!("{label} l2norm via rms_norm (alpha prebuilt)"),
            &|| candle_nn::ops::rms_norm(&x, &alpha, 1e-6 / d as f32).unwrap(),
        );
        time(
            &format!("{label} l2norm via rms_norm (alpha built per call)"),
            &|| {
                let a = Tensor::full((d as f32).powf(-0.5), d, &dev).unwrap();
                candle_nn::ops::rms_norm(&x, &a, 1e-6 / d as f32).unwrap()
            },
        );
    }
    let st = Tensor::randn(0f32, 1.0, (1, h, d, d), &dev).unwrap();
    let g = Tensor::randn(0f32, 0.1, (1, h, 1, 1), &dev)
        .unwrap()
        .neg()
        .unwrap()
        .exp()
        .unwrap();
    time("state decay: broadcast_mul (1,h,1,1)", &|| {
        st.broadcast_mul(&g).unwrap()
    });
    time("state decay: broadcast_as+contiguous then mul", &|| {
        (&st * g.broadcast_as(st.shape()).unwrap().contiguous().unwrap()).unwrap()
    });
    let k = Tensor::randn(0f32, 1.0, (1, h, d), &dev).unwrap();
    time("read kT S: broadcast_mul + sum", &|| {
        st.broadcast_mul(&k.unsqueeze(3).unwrap())
            .unwrap()
            .sum(2)
            .unwrap()
    });
    time("read kT S: matmul (1,h,1,d)@(1,h,d,d)", &|| {
        k.unsqueeze(2).unwrap().matmul(&st).unwrap()
    });
}
