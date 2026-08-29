//! One LFM2 prefill, fingerprinted stage by stage, on whichever device is asked
//! for.
//!
//! This is the producer half of the cross-device comparison `layer_diff.py`
//! consumes; see `gallium_core::probe` for what the two halves are for.
//! Deliberately not the agent binary: a turn's prompt depends on the skill
//! catalog and the working directory, and two runs that differ by one token are
//! not a numerics comparison at all.
//!
//!   GALLIUM_DEVICE=cpu   RUST_LOG=gallium::layers=trace \
//!     cargo run -p gallium-models --release --example lfm2_layers 2> cpu.log
//!   GALLIUM_DEVICE=metal RUST_LOG=gallium::layers=trace \
//!     cargo run -p gallium-models --release --example lfm2_layers 2> metal.log
//!   uv run python scripts/layer_diff.py cpu.log metal.log
//!
//! `GALLIUM_LFM2_GGUF_PATH` overrides the model, `GALLIUM_LAYERS_TOKENS` the
//! prompt length. The default of 121 is chosen to match the prefill in
//! `docs/CANDLE_BACKEND.md` §6c, and to sit above ggml's `ne21_mm_id_min = 32`
//! so this exercises the same kernels a real prefill does.
//!
//! The token ids are synthetic and fixed. Real text would need a tokenizer
//! download to say the same thing, and what matters here is only that both runs
//! receive **identical** ids — which arithmetic-generated ones guarantee by
//! construction rather than by two downloads agreeing.

use std::path::PathBuf;

use candle_core::{DType, Tensor};
use gallium_core::CausalLM;

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

fn main() -> anyhow::Result<()> {
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
        "device={} tokens={tokens} model={}",
        gallium_core::device_name(&device),
        path.display()
    );

    let (metadata, vb) = gallium_core::load_gguf(&path, &device)?;
    let mut model = gallium_models::lfm2moe_q::Lfm2MoeQ::load(&metadata, &vb, &device)?;

    // Spread across the vocabulary rather than clustered, so the MoE router is
    // given something to disagree about: a prompt that routes every token to
    // one expert would compare a narrow slice of the model and read as
    // agreement.
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
