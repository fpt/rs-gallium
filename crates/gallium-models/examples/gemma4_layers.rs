//! One Gemma 4 prefill, fingerprinted stage by stage, on whichever device is
//! asked for — the Gemma counterpart of `lfm2_layers.rs`.
//!
//! Its first job is a load test: E4B's PLE table (`per_layer_token_embd.weight`,
//! `[vocab, n_layers·ple_dim]`, ~11 GB dequantized) used to be dequantized
//! straight onto the compute device, which OOMs a 12 GB card before the first
//! block. It is host-resident and gathered per token now; this example on
//! `GALLIUM_DEVICE=cuda` is how that is checked.
//!
//!   GALLIUM_DEVICE=cuda RUST_LOG=gallium::layers=trace \
//!     cargo run -p gallium-models --release --features gallium-core/cuda \
//!       --example gemma4_layers 2> cuda.log
//!   GALLIUM_DEVICE=cpu  RUST_LOG=gallium::layers=trace \
//!     cargo run -p gallium-models --release --example gemma4_layers 2> cpu.log
//!   uv run python scripts/layer_diff.py cpu.log cuda.log
//!
//! `GALLIUM_GEMMA4_GGUF_PATH` overrides the model (default: E4B Q4_K_M in the HF
//! cache), `GALLIUM_LAYERS_TOKENS` the prompt length (default 121, matching
//! `lfm2_layers` and `docs/CANDLE_BACKEND.md` §6). `GALLIUM_LAYERS_F32_REF=1`
//! sets candle's `CANDLE_DEQUANTIZE_ALL` for an f32 reference forward, as in
//! `lfm2_layers` — see §6e.
//!
//! Synthetic, fixed token ids: both runs must receive identical input, which
//! arithmetic-generated ids guarantee without a tokenizer download.

use std::path::PathBuf;

use candle_core::{DType, Tensor};
use gallium_core::CausalLM;

const REPO: &str = "unsloth/gemma-4-E4B-it-GGUF";
const FILE: &str = "gemma-4-E4B-it-Q4_K_M.gguf";

fn gguf_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GALLIUM_GEMMA4_GGUF_PATH") {
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

fn env_flag(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn main() -> anyhow::Result<()> {
    let f32_ref = env_flag("GALLIUM_LAYERS_F32_REF");
    if f32_ref {
        std::env::set_var("CANDLE_DEQUANTIZE_ALL", "1");
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gallium::layers=trace".into()),
        )
        .with_writer(std::io::stderr)
        .without_time()
        .with_target(false)
        .init();

    let Some(path) = gguf_path() else {
        eprintln!("{REPO}/{FILE} is not in the HuggingFace cache");
        return Ok(());
    };
    let device = gallium_core::resolve_device(std::env::var("GALLIUM_DEVICE").ok().as_deref())?;
    let tokens: usize = std::env::var("GALLIUM_LAYERS_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(121);
    println!(
        "device={} tokens={tokens} f32_ref={f32_ref} model={}",
        gallium_core::device_name(&device),
        path.display()
    );

    let (metadata, vb) = gallium_core::load_gguf(&path, &device)?;
    let mut model = gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device)?;

    let ids: Vec<u32> = (0..tokens as u32)
        .map(|i| (i * 977 + 101) % 30_000)
        .collect();
    let input = Tensor::from_vec(ids, (1, tokens), &device)?.to_dtype(DType::U32)?;

    let logits: Vec<f32> = model.forward(&input, 0)?.squeeze(0)?.to_vec1()?;
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top-5: {:?}", &ranked[..5.min(ranked.len())]);
    Ok(())
}
