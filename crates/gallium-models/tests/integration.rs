//! Integration tests for all model variants (safetensors + GGUF).
//!
//! Each test skips gracefully if the required model files are not found in the
//! HuggingFace cache. Run with:
//!   cargo test -p gallium-models --test integration -- --nocapture
//!
//! Override model paths via environment variables:
//!   GALLIUM_GEMMA4_SAFETENSORS_DIR    (default: HF cache google/gemma-4-E4B)
//!   GALLIUM_GEMMA4_GGUF_PATH          (default: HF cache unsloth/gemma-4-E4B-it-GGUF)
//!   GALLIUM_GEMMA4_12B_GGUF_PATH      (default: HF cache unsloth/gemma-4-12B-it-GGUF)
//!   GALLIUM_GPT_OSS_SAFETENSORS_DIR   (default: HF cache openai/gpt-oss-20b)
//!   GALLIUM_GPT_OSS_GGUF_PATH         (no default; must be set explicitly)
//!   GALLIUM_QWEN35_SAFETENSORS_DIR    (default: HF cache Qwen/Qwen3.5-9B)

use candle_core::{DType, Device, IndexOp};
use gallium_core::{generate, load_gguf, CausalLM, SamplingParams};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the first snapshot directory for a HuggingFace repo, or None.
fn hf_snapshot(repo_id: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let sanitized = repo_id.replace('/', "--");
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{sanitized}"))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.file_type().ok().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
}

/// Return the path to a specific file inside a HF repo snapshot, or None.
fn hf_file(repo_id: &str, filename: &str) -> Option<PathBuf> {
    let p = hf_snapshot(repo_id)?.join(filename);
    p.exists().then_some(p)
}

/// Load a tokenizer from a directory that contains tokenizer.json.
fn load_tokenizer(dir: &Path) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))
}

/// Greedy sampling params.
fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: Some(1),
        ..Default::default()
    }
}

/// Run `generate()` and return the decoded text of newly generated tokens only.
fn run_inference(
    model: &mut dyn CausalLM,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<String> {
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    let eos: Vec<u32> = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .iter()
        .filter(|(k, _)| k.contains("eos") || k.contains("<|end") || k.contains("</s>"))
        .map(|(_, &v)| v)
        .collect();

    let mut generated: Vec<u32> = Vec::new();
    generate(model, &prompt_ids, &greedy(), max_tokens, &eos, |id| {
        generated.push(id);
    })?;

    tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode error: {e}"))
}

// ---------------------------------------------------------------------------
// Gemma 4 — safetensors
// ---------------------------------------------------------------------------

#[test]
fn gemma4_safetensors() {
    let dir = std::env::var("GALLIUM_GEMMA4_SAFETENSORS_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_snapshot("google/gemma-4-E4B"));

    let dir = match dir {
        Some(d) => d,
        None => {
            eprintln!("SKIP gemma4_safetensors: model not found (set GALLIUM_GEMMA4_SAFETENSORS_DIR or cache google/gemma-4-E4B)");
            return;
        }
    };

    let device = Device::Cpu;
    let safetensors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    assert!(!safetensors.is_empty(), "no .safetensors files in {:?}", dir);

    let config_path = dir.join("config.json");
    let vb = gallium_models::loader::load_safetensors(&safetensors, DType::F16, &device)
        .expect("load vb");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let full: serde_json::Value =
        gallium_models::loader::load_config(&config_path).expect("config");
    let text_cfg = full.get("text_config").unwrap_or(&full).clone();
    let cfg: gallium_models::gemma4::Gemma4Config =
        serde_json::from_value(text_cfg).expect("parse gemma4 config");

    let mut model = gallium_models::gemma4::Gemma4::load(&cfg, vb, &device).expect("load model");

    // Parallel-structure prompt strongly biases base models toward "Paris".
    let output = run_inference(
        &mut model,
        &tokenizer,
        "The capital of Japan is Tokyo. The capital of France is",
        8,
    )
    .expect("inference");
    eprintln!("gemma4_safetensors output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Gemma 4 — GGUF
// ---------------------------------------------------------------------------

#[test]
fn gemma4_gguf() {
    let gguf_path = std::env::var("GALLIUM_GEMMA4_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) => p,
        None => {
            eprintln!("SKIP gemma4_gguf: model not found (set GALLIUM_GEMMA4_GGUF_PATH or cache unsloth/gemma-4-E4B-it-GGUF)");
            return;
        }
    };

    let device = Device::Cpu;
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .expect("tokenizer");

    let mut model =
        gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device).expect("load model");

    let output = run_inference(&mut model, &tokenizer, "The capital of France is", 8)
        .expect("inference");
    eprintln!("gemma4_gguf output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Gemma 4 12B — GGUF
// ---------------------------------------------------------------------------

#[test]
fn gemma4_12b_gguf() {
    let gguf_path = std::env::var("GALLIUM_GEMMA4_12B_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-12B-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!("SKIP gemma4_12b_gguf: set GALLIUM_GEMMA4_12B_GGUF_PATH or cache unsloth/gemma-4-12B-it-GGUF");
            return;
        }
    };

    let device = Device::Cpu;
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    // tokenizer.json is saved alongside the GGUF by the agent downloader
    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .expect("tokenizer")
    } else if let Some(snap) = hf_snapshot("google/gemma-4-12B-it") {
        load_tokenizer(&snap).expect("tokenizer from google/gemma-4-12B-it snapshot")
    } else {
        eprintln!("SKIP gemma4_12b_gguf: no tokenizer found");
        return;
    };

    let mut model =
        gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device).expect("load model");

    // Gemma 4 12B is heavily IT-finetuned and requires the chat template.
    // Use the Gemma 4 chat format (same as GemmaProtocol in the agent).
    // The tokenizer for this model adds BOS automatically via the template.
    let chat_prompt = "<start_of_turn>user\nWhat is the capital of France? Answer in one word.<end_of_turn>\n<start_of_turn>model\n";
    let enc = tokenizer.encode(chat_prompt, true)
        .map_err(|e| anyhow::anyhow!("{e}")).expect("encode chat prompt");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    let mut generated: Vec<u32> = Vec::new();
    let eos = vec![1u32, 107u32];  // <eos>, <end_of_turn>
    generate(
        &mut model,
        &prompt_ids,
        &SamplingParams { temperature: 0.0, top_k: Some(1), ..Default::default() },
        20,
        &eos,
        |id| generated.push(id),
    ).expect("generate");
    let output = tokenizer.decode(&generated, true).expect("decode");
    eprintln!("gemma4_12b_gguf output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// GPT-OSS — safetensors
// ---------------------------------------------------------------------------

#[test]
fn gpt_oss_safetensors() {
    let dir = std::env::var("GALLIUM_GPT_OSS_SAFETENSORS_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_snapshot("openai/gpt-oss-20b"));

    let dir = match dir {
        Some(d) => d,
        None => {
            eprintln!("SKIP gpt_oss_safetensors: model not found (set GALLIUM_GPT_OSS_SAFETENSORS_DIR or cache openai/gpt-oss-20b)");
            return;
        }
    };

    let device = Device::Cpu;
    let safetensors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    assert!(!safetensors.is_empty(), "no .safetensors files in {:?}", dir);

    let config_path = dir.join("config.json");
    let vb = gallium_models::loader::load_safetensors(&safetensors, DType::BF16, &device)
        .expect("load vb");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let cfg: gallium_models::gpt_oss::GptOssConfig =
        gallium_models::loader::load_config(&config_path).expect("config");
    let mut model =
        gallium_models::gpt_oss::GptOss::load(&cfg, vb, &safetensors, &device).expect("load model");

    // GPT-OSS uses a chat template; wrap the prompt.
    let prompt = "<|start|>system<|message|>You are a helpful assistant.<|end|>\
                  <|start|>user<|message|>What is the capital of France?<|end|>\
                  <|start|>assistant\n";
    let output = run_inference(&mut model, &tokenizer, prompt, 20).expect("inference");
    eprintln!("gpt_oss_safetensors output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// GPT-OSS — GGUF
// ---------------------------------------------------------------------------

#[test]
fn gpt_oss_gguf() {
    let gguf_path = std::env::var("GALLIUM_GPT_OSS_GGUF_PATH")
        .ok()
        .map(PathBuf::from);

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        Some(p) => {
            eprintln!("SKIP gpt_oss_gguf: path {:?} does not exist", p);
            return;
        }
        None => {
            eprintln!("SKIP gpt_oss_gguf: set GALLIUM_GPT_OSS_GGUF_PATH to the .gguf file");
            return;
        }
    };

    let device = Device::Cpu;
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .expect("tokenizer");

    let mut model =
        gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device).expect("load model");

    let prompt = "<|start|>system<|message|>You are a helpful assistant.<|end|>\
                  <|start|>user<|message|>What is the capital of France?<|end|>\
                  <|start|>assistant\n";
    let output = run_inference(&mut model, &tokenizer, prompt, 20).expect("inference");
    eprintln!("gpt_oss_gguf output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Qwen 3.5 — safetensors
// ---------------------------------------------------------------------------

#[test]
fn qwen35_safetensors() {
    let dir = std::env::var("GALLIUM_QWEN35_SAFETENSORS_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_snapshot("Qwen/Qwen3.5-9B"));

    let dir = match dir {
        Some(d) => d,
        None => {
            eprintln!("SKIP qwen35_safetensors: model not found (set GALLIUM_QWEN35_SAFETENSORS_DIR or cache Qwen/Qwen3.5-9B)");
            return;
        }
    };

    let device = Device::Cpu;
    let safetensors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    assert!(!safetensors.is_empty(), "no .safetensors files in {:?}", dir);

    let config_path = dir.join("config.json");
    let vb = gallium_models::loader::load_safetensors(&safetensors, DType::F16, &device)
        .expect("load vb");
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let full: serde_json::Value =
        gallium_models::loader::load_config(&config_path).expect("config");
    let text_cfg = full.get("text_config").unwrap_or(&full).clone();
    let cfg: gallium_models::qwen35::Qwen35Config =
        serde_json::from_value(text_cfg).expect("parse qwen35 config");

    let mut model = gallium_models::qwen35::Qwen35::load(&cfg, vb, &device).expect("load model");

    let prompt = "The capital of Japan is Tokyo. The capital of France is";
    let enc = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("{e}")).expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let input = candle_core::Tensor::new(prompt_ids.as_slice(), &device)
        .expect("tensor").unsqueeze(0).expect("unsqueeze");
    let logits = model.forward(&input, 0).expect("forward");
    let top5 = top_k_logits(&logits.i(0).expect("batch"), 5).expect("top5");
    eprintln!("qwen35_safetensors top-5 first token:");
    for (id, logit) in &top5 {
        let tok = tokenizer.decode(&[*id], true).unwrap_or_default();
        eprintln!("  id={} {:?} logit={:.3}", id, tok, logit);
    }
    model.reset();

    let output = run_inference(
        &mut model,
        &tokenizer,
        prompt,
        8,
    )
    .expect("inference");
    eprintln!("qwen35_safetensors output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Qwen 3.5 — GGUF
// ---------------------------------------------------------------------------

/// Return top-k (index, logit) pairs from a 1D logit tensor.
fn top_k_logits(logits: &candle_core::Tensor, k: usize) -> anyhow::Result<Vec<(u32, f32)>> {
    let vals: Vec<f32> = logits.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = vals.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(indexed[..k.min(indexed.len())]
        .iter()
        .map(|&(i, v)| (i as u32, v))
        .collect())
}

#[test]
fn qwen35_gguf() {
    let gguf_path = std::env::var("GALLIUM_QWEN35_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!("SKIP qwen35_gguf: set GALLIUM_QWEN35_GGUF_PATH or cache unsloth/Qwen3.5-9B-GGUF");
            return;
        }
    };

    // Try to find tokenizer from a sibling snapshot
    let tokenizer = {
        let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
        if tok_path.exists() {
            Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .expect("tokenizer")
        } else if let Some(snap) = hf_snapshot("Qwen/Qwen3.5-9B") {
            load_tokenizer(&snap).expect("tokenizer from Qwen/Qwen3.5-9B snapshot")
        } else {
            eprintln!("SKIP qwen35_gguf: no tokenizer found");
            return;
        }
    };

    let device = Device::Cpu;
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    let mut model =
        gallium_models::qwen35_q::Qwen35Q::load(&metadata, &vb, &device).expect("load model");

    let prompt = "The capital of Japan is Tokyo. The capital of France is";

    // Print top-5 logits from the first forward pass for diagnostics.
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))
        .expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let input = candle_core::Tensor::new(
        prompt_ids.as_slice(),
        &device,
    )
    .expect("tensor")
    .unsqueeze(0)
    .expect("unsqueeze");

    let logits = model.forward(&input, 0).expect("forward");
    let top5 = top_k_logits(&logits.i(0).expect("batch"), 10).expect("top10");
    eprintln!("qwen35_gguf top-10 first token:");
    for (id, logit) in &top5 {
        let tok = tokenizer.decode(&[*id], true).unwrap_or_default();
        eprintln!("  id={} {:?} logit={:.3}", id, tok, logit);
    }
    // Also find rank of " Paris"
    {
        let paris_enc = tokenizer.encode(" Paris", false).expect("encode paris");
        if let Some(&paris_id) = paris_enc.get_ids().first() {
            let vals: Vec<f32> = logits.i(0).expect("batch").to_vec1().expect("vec1");
            let paris_logit = vals[paris_id as usize];
            let rank = vals.iter().filter(|&&v| v > paris_logit).count() + 1;
            eprintln!("  ' Paris' (id={}) logit={:.3} rank={}", paris_id, paris_logit, rank);
        }
    }

    model.reset();
    let output = run_inference(&mut model, &tokenizer, prompt, 8).expect("inference");
    eprintln!("qwen35_gguf output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}
