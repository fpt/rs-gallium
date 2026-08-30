//! Integration tests for all model variants (safetensors + GGUF).
//!
//! Every test here loads a multi-GB model from the HuggingFace cache, so they
//! are all `#[ignore]`d: `cargo test` stays fast and deterministic on a machine
//! that has no models (or has some but not others). Run them with:
//!   make test-models
//!   cargo test -p gallium-models --test integration -- --ignored --nocapture
//!
//! Each one still skips gracefully when its own model is missing, so running the
//! set on a partially populated cache exercises whatever is there.
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
use std::ops::ControlFlow;
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

/// `GALLIUM_DEVICE` if set, else CPU. Most tests here pin CPU for determinism;
/// `gemma4_gguf` uses this so the same assertion can be run against the
/// accelerator whose load path it exercises (E4B's PLE table — see `gemma4_q`).
fn test_device() -> Device {
    match std::env::var("GALLIUM_DEVICE").ok().as_deref() {
        None | Some("") | Some("cpu") => Device::Cpu,
        other => gallium_core::resolve_device(other).expect("resolve GALLIUM_DEVICE"),
    }
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

    // Stop tokens, by name, across the model families this file exercises.
    //
    // `<turn|>` is Gemma 4's end-of-turn and is why it is here: a Gemma model
    // ends its answer with that and *not* with `<eos>`, so a set built only
    // from "eos"-ish names ran straight past the reply and appended noise —
    // `"The capital of France is Paris.ayım"` is what that looks like.
    let eos: Vec<u32> = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .iter()
        .filter(|(k, _)| {
            k.contains("eos")
                || k.contains("<|end")
                || k.contains("</s>")
                || k.as_str() == "<turn|>"
        })
        .map(|(_, &v)| v)
        .collect();

    let mut generated: Vec<u32> = Vec::new();
    generate(model, &prompt_ids, &greedy(), max_tokens, &eos, |id| {
        generated.push(id);
        ControlFlow::Continue(())
    })?;

    tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decode error: {e}"))
}

// ---------------------------------------------------------------------------
// Gemma 4 — safetensors
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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
    if safetensors.is_empty() {
        eprintln!(
            "SKIP: no .safetensors weight files in {:?} (metadata-only cache)",
            dir
        );
        return;
    }

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

    // A completion prompt, not the chat template: this is `gemma-4-E4B`, the
    // base model, which was never tuned on turns. The parallel structure biases
    // it toward "Paris" rather than some other continuation.
    //
    // `<bos>` regardless, though — a base model wants it as much as an
    // instruction-tuned one, and this tokenizer adds none (see #30, where its
    // absence turned an instruction-tuned Gemma into an echo loop).
    let output = run_inference(
        &mut model,
        &tokenizer,
        "<bos>The capital of Japan is Tokyo. The capital of France is",
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
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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

    let device = test_device();
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    // tokenizer.json is saved alongside the GGUF by the agent downloader; a
    // cache populated some other way may not have it.
    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .expect("tokenizer")
    } else if let Some(snap) = hf_snapshot("unsloth/gemma-4-E4B-it") {
        load_tokenizer(&snap).expect("tokenizer from unsloth/gemma-4-E4B-it snapshot")
    } else {
        eprintln!("SKIP gemma4_gguf: no tokenizer found next to the GGUF or in the cache");
        return;
    };

    let mut model =
        gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device).expect("load model");

    // A well-formed Gemma 4 prompt: `<bos>` then the documented turn structure
    // (https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4, and the
    // thinking page for the `<bos>`), which is what `GemmaProtocol` builds and
    // what the GGUF's own chat template renders under llama.cpp.
    //
    // This test used to send the bare completion prompt `"The capital of France
    // is"` and assert on the continuation. That is not how an instruction-tuned
    // Gemma is addressed, and without a `<bos>` it does not merely answer badly
    // — it degenerates into echoing its own input (`" France is France is
    // France is"`), which is what #30 recorded as a `gemma4_q` inference bug.
    // The model and the loader were fine; the prompt was not.
    let prompt = "<bos><|turn>user\nWhat is the capital of France? \
                  Answer in one short sentence.<turn|>\n<|turn>model\n";
    let output = run_inference(&mut model, &tokenizer, prompt, 16).expect("inference");
    eprintln!("gemma4_gguf output: {:?}", output);
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

/// Sliding-window K/V narrowing (`gemma4_q.rs`) is meant to be *exact*: the
/// positions it drops before the scores matmul are the ones the mask sets to
/// `-inf`, which softmax weights at zero. This drives the cache past the window
/// (E4B 512, 12B 1024) so the narrowed path actually engages, runs it with
/// narrowing on and off (`GALLIUM_GEMMA4_KV_NARROW`), and asserts the two greedy
/// token streams are identical. It also splits the wall time into prefill (to
/// first token) and decode (the rest) — decode is where a long-context turn
/// spends its time and where the narrowing pays off.
///
/// `GALLIUM_KVTEST_FILLER` (default 220) and `GALLIUM_KVTEST_GEN` (default 64)
/// size the prompt and generation; the 12B on CPU wants smaller.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gemma4_gguf_kv_narrowing_is_exact_and_faster() {
    use std::time::Instant;

    let gguf_path = std::env::var("GALLIUM_GEMMA4_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gemma4_gguf_kv_narrowing: model not found");
        return;
    };
    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .unwrap()
    } else if let Some(snap) = hf_snapshot("unsloth/gemma-4-E4B-it") {
        load_tokenizer(&snap).unwrap()
    } else {
        eprintln!("SKIP gemma4_gguf_kv_narrowing: no tokenizer");
        return;
    };
    let device = test_device();

    // A long prompt so every sliding-layer *decode* step runs against a cache
    // that dwarfs the window — that gap on 35–40 of 42–48 layers is the target,
    // not the ~10% at the tail of a short turn.
    let reps: usize = std::env::var("GALLIUM_KVTEST_FILLER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(220);
    let n_gen: usize = std::env::var("GALLIUM_KVTEST_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(reps);
    let prompt = format!(
        "<bos><|turn>user\n{filler}\nIn one sentence, what animal is mentioned?<turn|>\n<|turn>model\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt_ids.len() > 1100, "prompt must dwarf the window");

    // Restore the caller's `GALLIUM_GEMMA4_KV_NARROW` on the way out, panic or
    // not, rather than leaving the process env mutated for other tests.
    struct RestoreEnv(&'static str, Option<String>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }
    let _guard = RestoreEnv(
        "GALLIUM_GEMMA4_KV_NARROW",
        std::env::var("GALLIUM_GEMMA4_KV_NARROW").ok(),
    );

    // (ids, prefill_s, decode_s)
    let run = |narrow: bool| -> (Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GEMMA4_KV_NARROW", if narrow { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device).expect("load model");
        let mut ids = Vec::new();
        let start = Instant::now();
        let mut first_tok: Option<f64> = None;
        generate(&mut model, &prompt_ids, &greedy(), n_gen, &[], |id| {
            first_tok.get_or_insert_with(|| start.elapsed().as_secs_f64());
            ids.push(id);
            ControlFlow::Continue(())
        })
        .expect("generate");
        let total = start.elapsed().as_secs_f64();
        let prefill = first_tok.unwrap_or(total);
        (ids, prefill, total - prefill)
    };

    // The timing print below measures ambient memory pressure, not this flag,
    // on any machine where two E4B loads do not fit comfortably. Four runs on an
    // M3/24 GB put the decode ratio anywhere from 1.02× to 7.66× with nothing
    // changed but what else was resident, and the second arm's prefill was
    // 2.2× slower, 1.19× slower, or *faster* depending on the run; only the
    // first arm reproduces (21.2, 20.3, 21.2 s). Trust the exactness assert
    // everywhere — it held on all four. Trust the speedup figure only where the
    // two loads fit (the RTX 4070 box measured 1.24× decode); on a tight box it
    // is noise, in both directions, and a single run of it will look conclusive.
    let (off_ids, off_pre, off_dec) = run(false);
    let (on_ids, on_pre, on_dec) = run(true);

    let dec_per = |s: f64| (n_gen.saturating_sub(1)) as f64 / s;
    eprintln!(
        "kv-narrow ({} prompt tok, {n_gen} gen): prefill {off_pre:.1}s→{on_pre:.1}s | \
         decode {off_dec:.1}s→{on_dec:.1}s ({:.1}→{:.1} tok/s, {:.2}x)",
        prompt_ids.len(),
        dec_per(off_dec),
        dec_per(on_dec),
        off_dec / on_dec.max(1e-6),
    );
    assert_eq!(
        on_ids, off_ids,
        "narrowed K/V must produce the identical greedy stream"
    );
    assert_eq!(on_ids.len(), n_gen);
}

// ---------------------------------------------------------------------------
// Gemma 4 12B — GGUF
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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

    // Gemma 4 12B uses a Harmony-style channel chat format — NOT the classic Gemma
    // <start_of_turn> template. Turns are <|turn>role ... <turn|> and the generation
    // prompt opens an empty "thought" channel that is immediately closed, so the
    // model emits its final answer as the very next token. (Special tokens:
    // <|turn>=105, <turn|>=106, <|channel>=100, <channel|>=101; the tokenizer does
    // NOT auto-prepend <bos>, so we include it literally.)
    //
    // A single prefill of a 12B Q4_K_M model on CPU is minutes-long, so we assert on
    // the first predicted token rather than running a multi-token decode.
    let chat_prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    let enc = tokenizer
        .encode(chat_prompt, false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .expect("encode chat prompt");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let input = candle_core::Tensor::new(prompt_ids.as_slice(), &device)
        .expect("tensor")
        .unsqueeze(0)
        .expect("unsqueeze");

    let logits = model.forward(&input, 0).expect("forward");
    let top5 = top_k_logits(&logits.i(0).expect("batch"), 5).expect("top5");
    eprintln!("gemma4_12b_gguf top-5 first token:");
    for (id, logit) in &top5 {
        let tok = tokenizer.decode(&[*id], false).unwrap_or_default();
        eprintln!("  id={} {:?} logit={:.3}", id, tok, logit);
    }
    let top_tok = tokenizer.decode(&[top5[0].0], false).unwrap_or_default();
    assert!(
        top_tok.to_lowercase().contains("paris"),
        "expected top token to be 'Paris', got: {:?}",
        top_tok
    );
}

// ---------------------------------------------------------------------------
// GPT-OSS — safetensors
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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
    if safetensors.is_empty() {
        eprintln!(
            "SKIP: no .safetensors weight files in {:?} (metadata-only cache)",
            dir
        );
        return;
    }

    let config_path = dir.join("config.json");
    // F16, not the checkpoint's native BF16: candle's CPU backend has no BF16
    // matmul, and `llm_candle` loads this model as F16 too (GALLIUM_DTYPE
    // defaults to "f16").
    let vb = gallium_models::loader::load_safetensors(&safetensors, DType::F16, &device)
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
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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

    let device = test_device();
    let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");

    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .expect("tokenizer")
    } else if let Some(snap) = hf_snapshot("openai/gpt-oss-20b") {
        load_tokenizer(&snap).expect("tokenizer from openai/gpt-oss-20b snapshot")
    } else {
        eprintln!("SKIP gpt_oss_gguf: no tokenizer found next to the GGUF or in the cache");
        return;
    };

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
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
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
    if safetensors.is_empty() {
        eprintln!(
            "SKIP: no .safetensors weight files in {:?} (metadata-only cache)",
            dir
        );
        return;
    }

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
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .expect("encode");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    let input = candle_core::Tensor::new(prompt_ids.as_slice(), &device)
        .expect("tensor")
        .unsqueeze(0)
        .expect("unsqueeze");
    let logits = model.forward(&input, 0).expect("forward");
    let top5 = top_k_logits(&logits.i(0).expect("batch"), 5).expect("top5");
    eprintln!("qwen35_safetensors top-5 first token:");
    for (id, logit) in &top5 {
        let tok = tokenizer.decode(&[*id], true).unwrap_or_default();
        eprintln!("  id={} {:?} logit={:.3}", id, tok, logit);
    }
    model.reset();

    let output = run_inference(&mut model, &tokenizer, prompt, 8).expect("inference");
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
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn qwen35_gguf() {
    let gguf_path = std::env::var("GALLIUM_QWEN35_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!(
                "SKIP qwen35_gguf: set GALLIUM_QWEN35_GGUF_PATH or cache unsloth/Qwen3.5-9B-GGUF"
            );
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
    let input = candle_core::Tensor::new(prompt_ids.as_slice(), &device)
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
            eprintln!(
                "  ' Paris' (id={}) logit={:.3} rank={}",
                paris_id, paris_logit, rank
            );
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

/// KV reuse across calls must produce the *same state* as never having reused —
/// the property the whole optimisation rests on, and the one whose failure
/// nothing downstream can detect.
///
/// LFM2 is the interesting case: short-conv + GQA, so a rewind is a positional
/// truncate on the attention layers and a snapshot restore on the recurrent
/// ones, in one operation. (llama.cpp will not do that pair —
/// `llama_memory_hybrid::seq_rm` tries the recurrent half first and refuses the
/// whole thing — which is why the llama.cpp backend snapshots the entire
/// sequence instead. Owning the cache is what makes the cheap version possible.)
///
/// The observable is a 12-token greedy continuation, not one argmax:
/// `crates/gallium-agent/tests/kv_state_spike.rs` records why — a single argmax
/// compared equal on a state that was demonstrably wrong. Twelve tokens compound
/// a divergence rather than hiding it, and unlike the spike, which owns the
/// forward pass and can read logits directly, this goes through
/// `generate_reusing` and sees only what it emits.
#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn lfm2_gguf_reuse_matches_a_cold_cache() {
    use gallium_core::generate_reusing;

    let gguf_path = std::env::var("GALLIUM_LFM2_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("LiquidAI/LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP lfm2_gguf_reuse_matches_a_cold_cache: model not in the cache");
        return;
    };
    let Some(snap) = hf_snapshot("LiquidAI/LFM2.5-8B-A1B") else {
        eprintln!(
            "SKIP lfm2_gguf_reuse_matches_a_cold_cache: no tokenizer (LiquidAI/LFM2.5-8B-A1B)"
        );
        return;
    };
    let tokenizer = load_tokenizer(&snap).expect("tokenizer");

    let device = Device::Cpu;
    let load = || {
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        gallium_models::lfm2moe_q::Lfm2MoeQ::load(&metadata, &vb, &device).expect("model")
    };
    let ids = |text: &str| {
        tokenizer
            .encode(text, true)
            .expect("tokenize")
            .get_ids()
            .to_vec()
    };

    // Iteration N's prompt, then iteration N+1's — the second extends the first,
    // which is what an agent turn does.
    let prompt = ids("A counter keeps a private value. Refactoring means replacing the package-level variable with a struct.");
    let mut extended = prompt.clone();
    extended.extend_from_slice(&ids(" Then main uses the struct and still prints three."));

    let params = greedy();
    let peek = |model: &mut dyn CausalLM, tokens: &[u32], reuse: usize| -> Vec<u32> {
        let (out, _) = generate_reusing(model, tokens, reuse, &params, 12, &[], |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("generate");
        out
    };

    // Cold: a model that has never seen the shorter prompt.
    let mut cold = load();
    let expected = peek(&mut cold, &extended, 0);
    drop(cold);

    // Warm: the first prompt, some generation on top of it, then a rewind to the
    // end of that prompt and the extended one evaluated from there.
    let mut warm = load();
    let (_, checkpoint) = generate_reusing(&mut warm, &prompt, 0, &params, 20, &[], |_| {
        std::ops::ControlFlow::Continue(())
    })
    .expect("first call");
    assert!(
        checkpoint.is_some(),
        "LFM2 is hybrid: a rewind needs a checkpoint, so one must have been taken"
    );
    let rewound = warm
        .cache()
        .expect("this model exposes its cache")
        .rewind(prompt.len(), checkpoint.as_ref())
        .expect("rewind");
    assert!(rewound, "the hybrid rewind was refused");
    let reused = peek(&mut warm, &extended, prompt.len());

    assert_eq!(
        reused, expected,
        "a reused cache continued differently from a cold one — reuse is not \
         equivalent, which is the failure a cache must never have"
    );
}
