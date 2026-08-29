#![cfg(all(feature = "local", feature = "candle"))]
//! How far apart are the two local engines on the *same* tokens?
//!
//! The question this answers came from a real disagreement: `lfm2-candle` loses
//! the `refactoring` testcase when its MoE multiplies against the quantized
//! expert weight instead of expanding it, while `lfm2` on llama.cpp — which also
//! multiplies quantized, against the same GGUF — passes. The tempting
//! explanation was "8-bit activations cost accuracy", and it does not survive
//! that comparison: if it were the arithmetic, llama.cpp would lose the case
//! too.
//!
//! Everything else about the two paths differs — the prompt is rendered by a
//! jinja template on one side and hand-written on the other, and every operator
//! is a separate implementation — so a testsuite result cannot say where a
//! divergence comes from. Feeding both engines the **same token ids** removes
//! the prompt from the question and leaves the numerics.
//!
//! Ignored (loads a multi-GB GGUF twice) and serial:
//!   cargo test -p gallium-agent --test engine_logits -- --ignored --nocapture --test-threads=1
//!
//! `GALLIUM_LFM2_GGUF_PATH` overrides the model; without it the HuggingFace
//! cache is searched and the test skips when the file is not there.

use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

const REPO: &str = "LiquidAI/LFM2.5-8B-A1B-GGUF";
const FILE: &str = "LFM2.5-8B-A1B-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GALLIUM_LFM2_GGUF_PATH") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", REPO.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(FILE))
        .find(|p| p.exists())
}

/// The `k` highest logits, as `(token id, value)`, highest first.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .into_iter()
        .take(k)
        .map(|(i, v)| (i as u32, v))
        .collect()
}

#[test]
#[ignore = "loads a 4.9GB GGUF twice"]
fn the_two_engines_agree_on_the_same_tokens() {
    let Some(path) = gguf_path() else {
        eprintln!("SKIP: {REPO}/{FILE} is not in the HuggingFace cache");
        return;
    };

    // ---- llama.cpp: tokenize, forward, read the last position's logits ------
    let backend = LlamaBackend::init().expect("backend");
    let gpu_layers: u32 = std::env::var("GALLIUM_LLAMA_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(999);
    eprintln!("llama.cpp gpu_layers={gpu_layers}");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &path, &model_params).expect("llama model");

    // `GALLIUM_LOGITS_REPEAT` lengthens the prompt, which matters because ggml
    // switches kernels by *token count*: `mul_mat_id` uses its matrix-vector
    // path below 32 tokens (`ne21_mm_id_min`) and the matrix-matrix one above,
    // so a short prompt and a long one exercise different halves of llama.cpp.
    let repeat: usize = std::env::var("GALLIUM_LOGITS_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let sentence = "The capital of France is Paris, and the capital of Japan is Tokyo. ";
    let text = sentence.repeat(repeat);
    let text = text.trim_end();
    let mut tokens = model.str_to_token(text, AddBos::Always).expect("tokenize");
    // `GALLIUM_LOGITS_TOKENS=1` forwards a single token, which is the shape a
    // *decode* step has. It matters because a quantized matmul can take a
    // different kernel for one row than for many — matvec versus matmul — and a
    // comparison made at one shape says nothing about the other.
    if let Some(n) = std::env::var("GALLIUM_LOGITS_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        tokens.truncate(n.max(1));
    }
    eprintln!("\n{} tokens: {:?}", tokens.len(), tokens);

    let n_ctx = 2048u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("llama context");

    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len() - 1;
    for (i, tok) in tokens.iter().copied().enumerate() {
        batch
            .add(tok, i as i32, &[0], i == last)
            .expect("batch add");
    }
    ctx.decode(&mut batch).expect("decode");
    let llama_logits: Vec<f32> = ctx.get_logits_ith(last as i32).to_vec();
    let ids: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
    drop(ctx);
    drop(model);

    // ---- candle: the same ids, through gallium's own implementation ---------
    let device = gallium_core::resolve_device(std::env::var("GALLIUM_DEVICE").ok().as_deref())
        .expect("device");
    let (metadata, vb) = gallium_core::load_gguf(&path, &device).expect("load gguf");
    let mut candle_model =
        gallium_models::lfm2moe_q::Lfm2MoeQ::load(&metadata, &vb, &device).expect("candle model");

    let input = candle_core::Tensor::from_vec(ids.clone(), (1, ids.len()), &device)
        .expect("input tensor")
        .to_dtype(candle_core::DType::U32)
        .expect("u32");
    let out = gallium_core::CausalLM::forward(&mut candle_model, &input, 0).expect("forward");
    let candle_logits: Vec<f32> = out
        .squeeze(0)
        .expect("squeeze")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .to_vec1()
        .expect("logits");

    // ---- compare ------------------------------------------------------------
    assert_eq!(
        llama_logits.len(),
        candle_logits.len(),
        "the two engines disagree about the vocabulary size"
    );

    let a = top_k(&llama_logits, 5);
    let b = top_k(&candle_logits, 5);
    eprintln!("llama.cpp top-5: {a:?}");
    eprintln!("candle    top-5: {b:?}");

    // Absolute logit values are not comparable across implementations — a
    // constant offset changes nothing about the distribution — so compare what
    // sampling actually reads: the ranking, and the gaps between the leaders.
    let spread = |t: &[(u32, f32)]| t[0].1 - t[4].1;
    eprintln!(
        "argmax: llama.cpp {} vs candle {} | top-1 minus top-5 spread: {:.3} vs {:.3}",
        a[0].0,
        b[0].0,
        spread(&a),
        spread(&b),
    );

    let shared = a
        .iter()
        .filter(|(id, _)| b.iter().any(|(other, _)| other == id))
        .count();
    eprintln!("shared members of the two top-5 sets: {shared}/5");

    assert_eq!(
        a[0].0, b[0].0,
        "the engines pick different next tokens for the same input — with greedy \
         sampling that is two different conversations from the first step, and no \
         testsuite comparison between them means anything until it is explained"
    );
}
