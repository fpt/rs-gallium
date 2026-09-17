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
//!   GALLIUM_GEMMA4_26B_GGUF_PATH      (default: HF cache unsloth/gemma-4-26B-A4B-it-qat-GGUF)
//!   GALLIUM_GPT_OSS_SAFETENSORS_DIR   (default: HF cache openai/gpt-oss-20b)
//!   GALLIUM_GPT_OSS_GGUF_PATH         (default: HF cache unsloth/gpt-oss-20b-GGUF)
//!   GALLIUM_GPT_OSS_120B_GGUF_PATH    (default: HF cache unsloth/gpt-oss-120b-GGUF,
//!                                      shard 1 — load_gguf discovers the rest)

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
        gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, false)
            .expect("load model");

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

/// Shared plumbing for the `*_KV_NARROW` A/B tests — `GALLIUM_GEMMA4_KV_NARROW`
/// (GGUF and safetensors) and `GALLIUM_GPT_OSS_KV_NARROW` (same, #232). Each
/// var is process-wide and these tests flip it, so every test — regardless of
/// which var it flips — takes one shared lock and restores its own var's
/// value on drop. One lock covering every var (rather than one per var) is
/// deliberate: these tests never need to run two at once, and a single lock
/// means a Gemma 4 narrowing test and a GPT-OSS one can't interleave their
/// env-var flips even though the vars themselves are independent.
mod kv_narrow {
    use gallium_core::{generate, CausalLM, SamplingParams};
    use std::ops::ControlFlow;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    pub struct EnvGuard(
        #[allow(dead_code)] MutexGuard<'static, ()>,
        &'static str,
        Option<String>,
    );
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.2 {
                Some(v) => std::env::set_var(self.1, v),
                None => std::env::remove_var(self.1),
            }
        }
    }

    /// Hold the shared A/B lock until the returned guard drops, then restore
    /// `var`'s value.
    pub fn lock_and_restore(var: &'static str) -> EnvGuard {
        let g = lock().lock().unwrap_or_else(|e| e.into_inner());
        EnvGuard(g, var, std::env::var(var).ok())
    }

    /// Greedy-decode `n_gen` tokens with `var` off, then on, reloading the
    /// model each time (the flag is read at load). Returns `(off_ids, on_ids)`.
    pub fn greedy_ab<M: CausalLM>(
        var: &'static str,
        mut load: impl FnMut() -> M,
        prompt_ids: &[u32],
        n_gen: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let run = |narrow: bool, load: &mut dyn FnMut() -> M| -> Vec<u32> {
            std::env::set_var(var, if narrow { "1" } else { "0" });
            let mut model = load();
            let mut ids = Vec::new();
            let params = SamplingParams {
                temperature: 0.0,
                top_k: Some(1),
                ..Default::default()
            };
            generate(&mut model, prompt_ids, &params, n_gen, &[], |id| {
                ids.push(id);
                ControlFlow::Continue(())
            })
            .expect("generate");
            ids
        };
        let off = run(false, &mut load);
        let on = run(true, &mut load);
        (off, on)
    }
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

    // `GALLIUM_GEMMA4_KV_NARROW` is process-wide; hold the lock so a concurrent
    // model-loading test can't observe this test's temporary `0`/`1`, and
    // restore the caller's value on the way out.
    let _env = kv_narrow::lock_and_restore("GALLIUM_GEMMA4_KV_NARROW");

    // (ids, prefill_s, decode_s)
    let run = |narrow: bool| -> (Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GEMMA4_KV_NARROW", if narrow { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, false)
                .expect("load model");
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

/// Fused Metal attention (`GALLIUM_GEMMA4_SDPA=1`, issue #308) against the
/// matmul path, same model and greedy prompt, both read at load. A fused
/// kernel rounds differently, so the bar is not a byte-identical stream: it
/// is the prefill's last-position logits (max |Δ| over the vocab, argmax
/// equal) — the same observable `tests/kv_state_spike.rs` uses — plus a
/// report of how far the two greedy streams agree and the prefill/decode
/// split for each arm. Metal only; skips elsewhere.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gemma4_gguf_metal_sdpa_matches_matmul() {
    use std::time::Instant;

    let device = test_device();
    if !device.is_metal() {
        eprintln!("SKIP gemma4_gguf_metal_sdpa: Metal only (GALLIUM_DEVICE=metal)");
        return;
    }
    let gguf_path = std::env::var("GALLIUM_GEMMA4_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gemma4_gguf_metal_sdpa: model not found");
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
        eprintln!("SKIP gemma4_gguf_metal_sdpa: no tokenizer");
        return;
    };

    // Past the window (E4B 512), so the sliding layers run narrowed and the
    // decode steps hit the vector kernel against a cache wider than the window.
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

    let _env = kv_narrow::lock_and_restore("GALLIUM_GEMMA4_SDPA");

    // (last-position logits of a one-shot prefill, greedy ids, prefill_s, decode_s)
    let run = |fused: bool| -> (Vec<f32>, Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GEMMA4_SDPA", if fused { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, true)
                .expect("load model");
        let input =
            candle_core::Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device)
                .unwrap();
        let logits: Vec<f32> = model
            .forward(&input, 0)
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        model.reset();
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
        (logits, ids, prefill, total - prefill)
    };

    // `GALLIUM_KVTEST_ARM=off|on` runs a single arm and reports its timing —
    // for a speed measurement, where two E4B loads in one process make the
    // second arm's numbers noise on a 24 GB Mac (see the narrowing test).
    if let Ok(arm) = std::env::var("GALLIUM_KVTEST_ARM") {
        let fused = match arm.as_str() {
            "on" => true,
            "off" => false,
            other => panic!("GALLIUM_KVTEST_ARM must be on|off, got {other:?}"),
        };
        let (_, ids, pre, dec) = run(fused);
        eprintln!(
            "metal-sdpa arm={arm} ({} prompt tok, {n_gen} gen): prefill {pre:.2}s ({:.0} tok/s) | \
             decode {dec:.2}s ({:.1} tok/s) | {:?}",
            prompt_ids.len(),
            prompt_ids.len() as f64 / pre,
            (n_gen.saturating_sub(1)) as f64 / dec,
            tokenizer.decode(&ids, true).unwrap_or_default(),
        );
        return;
    }

    let (off_logits, off_ids, off_pre, off_dec) = run(false);
    let (on_logits, on_ids, on_pre, on_dec) = run(true);

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    let max_delta = off_logits
        .iter()
        .zip(&on_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let agree = off_ids
        .iter()
        .zip(&on_ids)
        .take_while(|(a, b)| a == b)
        .count();
    let dec_per = |s: f64| (n_gen.saturating_sub(1)) as f64 / s;
    eprintln!(
        "metal-sdpa ({} prompt tok, {n_gen} gen): max|Δlogit| {max_delta:.4}, argmax {} vs {} | \
         greedy streams agree on {agree}/{n_gen} | prefill {off_pre:.2}s→{on_pre:.2}s ({:.0}→{:.0} tok/s) | \
         decode {off_dec:.2}s→{on_dec:.2}s ({:.1}→{:.1} tok/s)",
        prompt_ids.len(),
        argmax(&off_logits),
        argmax(&on_logits),
        prompt_ids.len() as f64 / off_pre,
        prompt_ids.len() as f64 / on_pre,
        dec_per(off_dec),
        dec_per(on_dec),
    );
    eprintln!(
        "  matmul: {:?}\n  sdpa:   {:?}",
        tokenizer.decode(&off_ids, true).unwrap_or_default(),
        tokenizer.decode(&on_ids, true).unwrap_or_default()
    );
    assert_eq!(
        argmax(&off_logits),
        argmax(&on_logits),
        "fused and matmul attention disagree on the prefill's next token"
    );
    assert!(
        max_delta < 0.25,
        "max |Δlogit| {max_delta} between fused and matmul attention is too large"
    );
    assert_eq!(on_ids.len(), n_gen);
}

/// Fused CUDA *prefill* attention (`candle-flash-attn`, `GALLIUM_GEMMA4_FLASH_ATTN=1`,
/// issue #308), sliding layers only, checked against an **f32 baseline**
/// (`kv_f16 = false`, every tensor full precision) rather than against the
/// matmul-f16 path — the matmul-f16 path is not a trustworthy reference on
/// CUDA. Measured on this prompt: matmul-f16 vs f32 has max |Δlogit| **2.06**
/// with **21,765 of 262,144** vocab positions off by more than 1.0; flash-f16
/// vs f32 has max |Δlogit| **1.25** with only **17** such positions — flash
/// is *closer* to ground truth than the already-shipped, default-on
/// `gemma4KvF16` matmul path is, on this machine. (A prior version of this
/// test compared flash-f16 against matmul-f16 directly, measured "2.0–2.2,
/// wider than the Metal sdpa test's 0.25" and read that as a flash-attn
/// concern; it wasn't — matmul-f16's own drift from f32 accounts for nearly
/// all of it. The likely mechanism is cuBLAS's f16×f16 GEMM defaulting to
/// f16 accumulation unless told otherwise, against FA2's f32-internal
/// online-softmax; unconfirmed, and diagnosing candle's CUDA matmul path is
/// out of scope here — issue #305/#307's default-on `gemma4KvF16` predates
/// this PR and is unaffected by it either way.) Global/head_dim-512 is cut
/// entirely — see `QAttention::flash_attention`'s doc comment for that
/// measurement. Decode always stays on the matmul path on both arms. CUDA
/// only; skips elsewhere.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gemma4_gguf_flash_attn_matches_matmul() {
    use std::time::Instant;

    let device = test_device();
    if !device.is_cuda() {
        eprintln!("SKIP gemma4_gguf_flash_attn: CUDA only (GALLIUM_DEVICE=cuda)");
        return;
    }
    let gguf_path = std::env::var("GALLIUM_GEMMA4_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gemma4_gguf_flash_attn: model not found");
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
        eprintln!("SKIP gemma4_gguf_flash_attn: no tokenizer");
        return;
    };

    // Past the window (E4B 512), so the sliding layers' prefill windowing
    // math runs at a non-zero `pos` too, not just the trivial pos=0 case.
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

    let _env = kv_narrow::lock_and_restore("GALLIUM_GEMMA4_FLASH_ATTN");
    let input =
        candle_core::Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device).unwrap();

    // Last-position logits of a one-shot prefill. `kv_f16 = false` (f32) is
    // the ground-truth arm; `flash` only takes effect when `kv_f16 = true`
    // (see `Gemma4Q::load`'s gate), so the f32 arm ignores `GALLIUM_GEMMA4_FLASH_ATTN`
    // regardless of its value.
    let prefill_logits = |kv_f16: bool, flash: bool| -> Vec<f32> {
        std::env::set_var("GALLIUM_GEMMA4_FLASH_ATTN", if flash { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, kv_f16)
                .expect("load model");
        model
            .forward(&input, 0)
            .expect("forward")
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let f32_logits = prefill_logits(false, false);
    let matmul_f16_logits = prefill_logits(true, false);
    let flash_f16_logits = prefill_logits(true, true);

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    let max_delta = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let big_deltas = |a: &[f32], b: &[f32], thresh: f32| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| (**x - **y).abs() > thresh)
            .count()
    };
    let matmul_vs_f32 = max_delta(&f32_logits, &matmul_f16_logits);
    let flash_vs_f32 = max_delta(&f32_logits, &flash_f16_logits);
    eprintln!(
        "prefill logits ({} prompt tok, vocab {}): argmax f32={} matmul-f16={} flash-f16={}",
        prompt_ids.len(),
        f32_logits.len(),
        argmax(&f32_logits),
        argmax(&matmul_f16_logits),
        argmax(&flash_f16_logits)
    );
    eprintln!(
        "vs f32: matmul-f16 max|Δ| {matmul_vs_f32:.4} ({} positions |Δ|>1.0) | \
         flash-f16 max|Δ| {flash_vs_f32:.4} ({} positions |Δ|>1.0)",
        big_deltas(&f32_logits, &matmul_f16_logits, 1.0),
        big_deltas(&f32_logits, &flash_f16_logits, 1.0),
    );
    assert_eq!(
        argmax(&f32_logits),
        argmax(&flash_f16_logits),
        "flash-attn disagrees with the f32 baseline on the prefill's next token"
    );
    // 2.0, informed by the measured 1.25 plus margin — this is flash-f16
    // against the trustworthy (f32) reference, not against matmul-f16.
    assert!(
        flash_vs_f32 < 2.0,
        "max |Δlogit| {flash_vs_f32} between flash-attn and the f32 baseline is too large"
    );

    // Secondary check: the two f16 arms' own greedy streams, for a read on
    // whether flash-attn's prefill difference propagates into generation.
    let run = |flash: bool| -> (Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GEMMA4_FLASH_ATTN", if flash { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, true)
                .expect("load model");
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
    let (off_ids, off_pre, off_dec) = run(false);
    let (on_ids, on_pre, on_dec) = run(true);
    let agree = off_ids
        .iter()
        .zip(&on_ids)
        .take_while(|(a, b)| a == b)
        .count();
    let dec_per = |s: f64| (n_gen.saturating_sub(1)) as f64 / s;
    eprintln!(
        "greedy streams agree on {agree}/{n_gen} | prefill {off_pre:.2}s→{on_pre:.2}s \
         ({:.0}→{:.0} tok/s) | decode {off_dec:.2}s→{on_dec:.2}s ({:.1}→{:.1} tok/s)",
        prompt_ids.len() as f64 / off_pre,
        prompt_ids.len() as f64 / on_pre,
        dec_per(off_dec),
        dec_per(on_dec),
    );
    eprintln!(
        "  matmul: {:?}\n  flash:  {:?}",
        tokenizer.decode(&off_ids, true).unwrap_or_default(),
        tokenizer.decode(&on_ids, true).unwrap_or_default()
    );
    assert_eq!(on_ids.len(), n_gen);
}

/// Isolates whether `candle-flash-attn`'s head_dim-512 **causal** kernel is
/// itself wrong, independent of Gemma 4, gallium's call-site code, or any
/// GGUF weights — random Q/K/V, no model, no GQA (`h == h_kv`, since GQA
/// mapping is not in question here — see the review that asked for this
/// test). `d = 512`, `s = 512` (one prefill chunk), `t = 2732` (a KV cache
/// past the window, so this exercises the same `seqlen_q < seqlen_k`
/// bottom-right causal alignment the model path relies on) — the shapes
/// `QAttention::flash_attention`'s doc comment cites the ~22 max |Δlogit|
/// measurement for.
///
/// Reference is a plain causal-masked `softmax(QK^T·scale)V` computed
/// **entirely in f32** (`Tensor::matmul` + `candle_nn::ops::softmax_last_dim`),
/// independent of `gqa_scores`/`gqa_weighted_sum` and of the matmul-f16 path
/// — `gemma4_gguf_flash_attn_matches_matmul` already established that
/// matmul-f16 is not a trustworthy reference on this machine's CUDA build,
/// so this test doesn't lean on it either. `q`/`k`/`v` are cast to f16 before
/// the flash-attn call, matching production; the reference stays f32
/// throughout.
///
/// CUDA + the `flash-attn` cargo feature only (doesn't need a cached model).
#[cfg(feature = "flash-attn")]
#[test]
#[ignore = "needs CUDA; run with `cargo test --features cuda,flash-attn -- --ignored`"]
fn flash_attn_hdim512_causal_probe() {
    let device = test_device();
    if !device.is_cuda() {
        eprintln!("SKIP flash_attn_hdim512_causal_probe: CUDA only (GALLIUM_DEVICE=cuda)");
        return;
    }

    let (b, h, d, s, t) = (1usize, 8usize, 512usize, 512usize, 2732usize);
    // Deterministic pseudo-random fill (no `rand` dependency needed) — a
    // standard hash-noise formula, not true randomness, but decorrelated
    // enough to stress the kernel across the full value range.
    let noise = |i: usize| -> f32 {
        let x = (i as f32) * 12.9898;
        (x.sin() * 43758.5).fract()
    };
    let make = |n: usize, off: usize| -> Vec<f32> { (0..n).map(|i| noise(i + off)).collect() };

    let q = candle_core::Tensor::from_vec(make(b * s * h * d, 0), (b, s, h, d), &device).unwrap();
    let k = candle_core::Tensor::from_vec(make(b * t * h * d, 1_000_000), (b, t, h, d), &device)
        .unwrap();
    let v = candle_core::Tensor::from_vec(make(b * t * h * d, 2_000_000), (b, t, h, d), &device)
        .unwrap();

    let scale = 1.0 / (d as f64).sqrt();

    // Reference: f32 throughout, bottom-right causal (query row i <=> key
    // col <= t - s + i), the same alignment `flash_attn(..., causal = true)`
    // uses for `seqlen_q < seqlen_k`.
    let q_bhsd = q.transpose(1, 2).unwrap().contiguous().unwrap(); // (b, h, s, d)
    let k_bhtd = k.transpose(1, 2).unwrap().contiguous().unwrap(); // (b, h, t, d)
    let v_bhtd = v.transpose(1, 2).unwrap().contiguous().unwrap();
    let scores = q_bhsd
        .matmul(&k_bhtd.transpose(2, 3).unwrap().contiguous().unwrap())
        .unwrap()
        * scale;
    let scores = scores.unwrap();
    let mut mask_data = vec![0f32; s * t];
    for i in 0..s {
        for j in 0..t {
            if j > t - s + i {
                mask_data[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = candle_core::Tensor::from_vec(mask_data, (1, 1, s, t), &device).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    let ref_out = probs.matmul(&v_bhtd).unwrap(); // (b, h, s, d)
    let ref_out: Vec<f32> = ref_out
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // FA: f16 in, causal = true, same (b, s, h, d) layout the model call
    // site feeds it (there, via a metadata-only transpose of a (b, h, s, d)
    // buffer; here, natively).
    let q16 = q.to_dtype(DType::F16).unwrap();
    let k16 = k.to_dtype(DType::F16).unwrap();
    let v16 = v.to_dtype(DType::F16).unwrap();
    let fa_out =
        gallium_models::candle_flash_attn::flash_attn(&q16, &k16, &v16, scale as f32, true)
            .expect("flash_attn");
    let fa_out: Vec<f32> = fa_out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let max_delta = ref_out
        .iter()
        .zip(&fa_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let big = ref_out
        .iter()
        .zip(&fa_out)
        .filter(|(a, b)| (**a - **b).abs() > 1.0)
        .count();
    eprintln!(
        "flash-attn hdim512 causal probe (b={b} h={h} d={d} s={s} t={t}): \
         max|Δ| {max_delta:.4}, {big}/{} positions off by >1.0",
        ref_out.len()
    );
    assert!(
        max_delta < 2.0,
        "flash-attn's head_dim-512 causal kernel disagrees with an independent f32 \
         reference by {max_delta} on random q/k/v — this is the kernel itself, not \
         gallium's call site (no model, no GGUF weights, no GQA)"
    );
}

/// Same probe as [`flash_attn_hdim512_causal_probe`], but with K/V fed as a
/// **strided view into an oversized buffer**, matching `KvCache`'s actual
/// layout instead of a tight contiguous one — `KvCache::plan_capacity` rounds
/// a fresh cache up to the next power of two, so a 2220-token single-shot
/// prefill (this test's own `gemma4_gguf_flash_attn_matches_matmul` prompt
/// length) gets `capacity = 4096`: the buffer is `[b, h, capacity, d]`, and
/// after `.narrow(2, 0, t).transpose(1, 2)` (exactly what `QAttention::forward`
/// does) K/V's head stride is `capacity·d`, not `t·d`. Measured max |Δ| in
/// the 1–1.5 range here (some run-to-run float noise at this head width,
/// unlike the stable figures below) against well under 1 for the
/// clean-contiguous probe — elevated, but nowhere near the real model's ~22.
/// The stride gap contributes; it isn't the whole story — see
/// `flash_attn_hdim256_vs_hdim512_at_matched_magnitude` for what is.
#[cfg(feature = "flash-attn")]
#[test]
#[ignore = "needs CUDA; run with `cargo test --features cuda,flash-attn -- --ignored"]
fn flash_attn_hdim512_causal_probe_strided_cache() {
    let device = test_device();
    if !device.is_cuda() {
        eprintln!(
            "SKIP flash_attn_hdim512_causal_probe_strided_cache: CUDA only (GALLIUM_DEVICE=cuda)"
        );
        return;
    }

    let (b, h, d, t, capacity) = (1usize, 8usize, 512usize, 2220usize, 4096usize);
    let noise = |i: usize| -> f32 {
        let x = (i as f32) * 12.9898;
        (x.sin() * 43758.5).fract()
    };
    let make = |n: usize, off: usize| -> Vec<f32> { (0..n).map(|i| noise(i + off)).collect() };

    // Q: fresh each call, always tightly contiguous — unaffected by the cache.
    let q = candle_core::Tensor::from_vec(make(b * t * h * d, 0), (b, t, h, d), &device).unwrap();

    // K/V: the cache's own `[b, h, capacity, d]` buffer shape, live data in
    // the first `t` of `capacity` — everything beyond `t` is left as zeros,
    // matching `KvCache`'s own scratch tail (never read once narrowed, but
    // zeros rather than uninitialized memory keeps this test's own math
    // simple if a stride bug ever *does* read past the narrow).
    let k_buf = candle_core::Tensor::zeros((b, h, capacity, d), DType::F32, &device).unwrap();
    let v_buf = candle_core::Tensor::zeros((b, h, capacity, d), DType::F32, &device).unwrap();
    let k_live =
        candle_core::Tensor::from_vec(make(b * h * t * d, 1_000_000), (b, h, t, d), &device)
            .unwrap();
    let v_live =
        candle_core::Tensor::from_vec(make(b * h * t * d, 2_000_000), (b, h, t, d), &device)
            .unwrap();
    let k_buf = k_buf
        .slice_assign(&[0..b, 0..h, 0..t, 0..d], &k_live)
        .unwrap();
    let v_buf = v_buf
        .slice_assign(&[0..b, 0..h, 0..t, 0..d], &v_live)
        .unwrap();
    // Narrow to the live length, then transpose to (b, seq, h, d) — exactly
    // `QAttention::forward`'s own `narrow_kv_to_mask` + `flash_attention`
    // sequence. Strides are preserved by both ops, so `k`/`v` keep head
    // stride `capacity·d`, not `t·d`, same as production.
    let k = k_buf.narrow(2, 0, t).unwrap().transpose(1, 2).unwrap(); // (b, t, h, d), strided
    let v = v_buf.narrow(2, 0, t).unwrap().transpose(1, 2).unwrap();

    let scale = 1.0 / (d as f64).sqrt();

    // Reference: plain causal softmax(QK^T)V in f32, computed from the same
    // *values* but via ordinary contiguous tensors (k_live/v_live directly,
    // sidestepping the stride question for the reference itself).
    let q_bhtd = q.transpose(1, 2).unwrap().contiguous().unwrap(); // (b, h, t, d)
    let scores = q_bhtd
        .matmul(&k_live.transpose(2, 3).unwrap().contiguous().unwrap())
        .unwrap()
        * scale;
    let scores = scores.unwrap();
    let mut mask_data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask_data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mask = candle_core::Tensor::from_vec(mask_data, (1, 1, t, t), &device).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    let ref_out = probs.matmul(&v_live).unwrap(); // (b, h, t, d)
    let ref_out: Vec<f32> = ref_out
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let q16 = q.to_dtype(DType::F16).unwrap();
    let k16 = k.to_dtype(DType::F16).unwrap();
    let v16 = v.to_dtype(DType::F16).unwrap();
    let fa_out =
        gallium_models::candle_flash_attn::flash_attn(&q16, &k16, &v16, scale as f32, true)
            .expect("flash_attn");
    let fa_out: Vec<f32> = fa_out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let max_delta = ref_out
        .iter()
        .zip(&fa_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let big = ref_out
        .iter()
        .zip(&fa_out)
        .filter(|(a, b)| (**a - **b).abs() > 1.0)
        .count();
    eprintln!(
        "flash-attn hdim512 causal probe, strided cache (b={b} h={h} d={d} t={t}, \
         capacity={capacity}): max|Δ| {max_delta:.4}, {big}/{} positions off by >1.0",
        ref_out.len()
    );
    assert!(
        max_delta < 2.0,
        "flash-attn's head_dim-512 causal kernel disagrees with an independent f32 \
         reference by {max_delta} when K/V are a strided KvCache-shaped view — this \
         reproduces (or rules out) the stride gap as the cause of the ~22 max |Δlogit| \
         seen through the real model"
    );
}

/// Same as [`flash_attn_hdim512_causal_probe_strided_cache`], plus E4B's own
/// GQA ratio (`n_q = 8`, `n_kv = 2`, `rep = 4`) — the one production
/// difference the strided-cache probe still didn't have (E4B's global layers
/// all carry `attn_v.weight`, so shared-K=V is not in play; confirmed
/// against the real GGUF, not assumed). Stays in the same 1–1.5-ish range as
/// the strided-cache probe alone, nowhere near the ~22 seen through the real
/// model — this test is what's left once head_dim 512, the stride gap, and
/// GQA are all reproduced together outside the model, and it still doesn't
/// explain the real failure on its own.
#[cfg(feature = "flash-attn")]
#[test]
#[ignore = "needs CUDA; run with `cargo test --features cuda,flash-attn -- --ignored"]
fn flash_attn_hdim512_causal_probe_strided_cache_gqa() {
    let device = test_device();
    if !device.is_cuda() {
        eprintln!(
            "SKIP flash_attn_hdim512_causal_probe_strided_cache_gqa: CUDA only (GALLIUM_DEVICE=cuda)"
        );
        return;
    }

    let (b, h, h_kv, d, t, capacity) = (1usize, 8usize, 2usize, 512usize, 2220usize, 4096usize);
    let rep = h / h_kv;
    let noise = |i: usize| -> f32 {
        let x = (i as f32) * 12.9898;
        (x.sin() * 43758.5).fract()
    };
    let make = |n: usize, off: usize| -> Vec<f32> { (0..n).map(|i| noise(i + off)).collect() };

    let q = candle_core::Tensor::from_vec(make(b * t * h * d, 0), (b, t, h, d), &device).unwrap();

    let k_buf = candle_core::Tensor::zeros((b, h_kv, capacity, d), DType::F32, &device).unwrap();
    let v_buf = candle_core::Tensor::zeros((b, h_kv, capacity, d), DType::F32, &device).unwrap();
    let k_live =
        candle_core::Tensor::from_vec(make(b * h_kv * t * d, 1_000_000), (b, h_kv, t, d), &device)
            .unwrap();
    let v_live =
        candle_core::Tensor::from_vec(make(b * h_kv * t * d, 2_000_000), (b, h_kv, t, d), &device)
            .unwrap();
    let k_buf = k_buf
        .slice_assign(&[0..b, 0..h_kv, 0..t, 0..d], &k_live)
        .unwrap();
    let v_buf = v_buf
        .slice_assign(&[0..b, 0..h_kv, 0..t, 0..d], &v_live)
        .unwrap();
    // (b, h_kv, t, d), head stride capacity·d — flash-attn's own GQA support
    // reads this directly, no repeat needed on this side.
    let k = k_buf.narrow(2, 0, t).unwrap().transpose(1, 2).unwrap(); // (b, t, h_kv, d)
    let v = v_buf.narrow(2, 0, t).unwrap().transpose(1, 2).unwrap();

    let scale = 1.0 / (d as f64).sqrt();

    // Reference: GQA-expand K/V to `h` heads (repeat_interleave by `rep`)
    // before the matmul, since `Tensor::matmul` has no native GQA.
    let expand = |x: &candle_core::Tensor| -> candle_core::Tensor {
        x.unsqueeze(2)
            .unwrap()
            .broadcast_as((b, h_kv, rep, t, d))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, h, t, d))
            .unwrap()
    };
    let k_live_expanded = expand(&k_live);
    let v_live_expanded = expand(&v_live);
    let q_bhtd = q.transpose(1, 2).unwrap().contiguous().unwrap(); // (b, h, t, d)
    let scores = q_bhtd
        .matmul(
            &k_live_expanded
                .transpose(2, 3)
                .unwrap()
                .contiguous()
                .unwrap(),
        )
        .unwrap()
        * scale;
    let scores = scores.unwrap();
    let mut mask_data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask_data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    let mask = candle_core::Tensor::from_vec(mask_data, (1, 1, t, t), &device).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
    let ref_out = probs.matmul(&v_live_expanded).unwrap(); // (b, h, t, d)
    let ref_out: Vec<f32> = ref_out
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let q16 = q.to_dtype(DType::F16).unwrap();
    let k16 = k.to_dtype(DType::F16).unwrap();
    let v16 = v.to_dtype(DType::F16).unwrap();
    let fa_out =
        gallium_models::candle_flash_attn::flash_attn(&q16, &k16, &v16, scale as f32, true)
            .expect("flash_attn");
    let fa_out: Vec<f32> = fa_out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let max_delta = ref_out
        .iter()
        .zip(&fa_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let big = ref_out
        .iter()
        .zip(&fa_out)
        .filter(|(a, b)| (**a - **b).abs() > 1.0)
        .count();
    eprintln!(
        "flash-attn hdim512 causal probe, strided cache + GQA (h={h} h_kv={h_kv} d={d} t={t}, \
         capacity={capacity}): max|Δ| {max_delta:.4}, {big}/{} positions off by >1.0",
        ref_out.len()
    );
    assert!(
        max_delta < 2.0,
        "flash-attn's head_dim-512 causal kernel disagrees with an independent f32 \
         reference by {max_delta} with K/V strided AND GQA — this is the closest \
         reproduction of the real model's shapes outside the model itself, and by \
         itself still doesn't reproduce the ~22 seen through the real model; see \
         flash_attn_hdim256_vs_hdim512_at_matched_magnitude for what does"
    );
}

/// **The decisive isolation.** Neither the stride gap nor GQA (the two probes
/// above) reproduces the real model's ~22 max |Δlogit| on their own — both
/// stayed under 1.6 at the same noise scale those probes use. What does: the
/// *value magnitude*. Same clean, unstrided, non-GQA setup as
/// `flash_attn_hdim512_causal_probe`, at 10× that test's noise amplitude
/// (still a fixed, deterministic multiplier — not tuned to make an assertion
/// pass, chosen because it's what reproduces the real failure's magnitude):
///
/// | head_dim | max &#124;Δ&#124; at 1× (single sample) | max &#124;Δ&#124; at 10× (stable across reruns) |
/// |---|---|---|
/// | 256 | ~0.0001 | 0.11 |
/// | 512 | ~0.2 | ~19.9 (matches the ~22 measured through the real model) |
///
/// head_dim 256 stays tight at both scales; head_dim 512 breaks specifically
/// at the larger one, with the same shape of degradation (not a handful of
/// outlier positions — over 30% of the output tensor off by more than 1.0).
/// That head_dim, not the stride gap, not GQA, not Gemma 4's own weights, is
/// what the shipped default's sliding-layer-only scope
/// (`FUSED_PREFILL_MAX_HEAD_DIM`-style, `QAttention::flash_attention`) is
/// actually excluding — and confirms that scope is a real safety margin, not
/// luck: 256 held at 10× the amplitude the real model's own activations
/// produce.
#[cfg(feature = "flash-attn")]
#[test]
#[ignore = "needs CUDA; run with `cargo test --features cuda,flash-attn -- --ignored`"]
fn flash_attn_hdim256_vs_hdim512_at_matched_magnitude() {
    let device = test_device();
    if !device.is_cuda() {
        eprintln!(
            "SKIP flash_attn_hdim256_vs_hdim512_at_matched_magnitude: CUDA only \
             (GALLIUM_DEVICE=cuda)"
        );
        return;
    }

    let (b, h, s, t) = (1usize, 8usize, 512usize, 2732usize);
    const NOISE_SCALE: f32 = 10.0;

    let probe = |d: usize| -> f32 {
        let noise = |i: usize| -> f32 {
            let x = (i as f32) * 12.9898;
            (x.sin() * 43758.5).fract() * NOISE_SCALE
        };
        let make = |n: usize, off: usize| -> Vec<f32> { (0..n).map(|i| noise(i + off)).collect() };

        let q =
            candle_core::Tensor::from_vec(make(b * s * h * d, 0), (b, s, h, d), &device).unwrap();
        let k =
            candle_core::Tensor::from_vec(make(b * t * h * d, 1_000_000), (b, t, h, d), &device)
                .unwrap();
        let v =
            candle_core::Tensor::from_vec(make(b * t * h * d, 2_000_000), (b, t, h, d), &device)
                .unwrap();
        let scale = 1.0 / (d as f64).sqrt();

        let q_bhsd = q.transpose(1, 2).unwrap().contiguous().unwrap();
        let k_bhtd = k.transpose(1, 2).unwrap().contiguous().unwrap();
        let v_bhtd = v.transpose(1, 2).unwrap().contiguous().unwrap();
        let scores = q_bhsd
            .matmul(&k_bhtd.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap()
            * scale;
        let scores = scores.unwrap();
        let mut mask_data = vec![0f32; s * t];
        for i in 0..s {
            for j in 0..t {
                if j > t - s + i {
                    mask_data[i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = candle_core::Tensor::from_vec(mask_data, (1, 1, s, t), &device).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let ref_out = probs.matmul(&v_bhtd).unwrap();
        let ref_out: Vec<f32> = ref_out
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let q16 = q.to_dtype(DType::F16).unwrap();
        let k16 = k.to_dtype(DType::F16).unwrap();
        let v16 = v.to_dtype(DType::F16).unwrap();
        let fa_out =
            gallium_models::candle_flash_attn::flash_attn(&q16, &k16, &v16, scale as f32, true)
                .expect("flash_attn");
        let fa_out: Vec<f32> = fa_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        ref_out
            .iter()
            .zip(&fa_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };

    let delta_256 = probe(256);
    let delta_512 = probe(512);
    eprintln!(
        "flash-attn magnitude probe (noise ×{NOISE_SCALE}): head_dim 256 max|Δ| {delta_256:.4} | \
         head_dim 512 max|Δ| {delta_512:.4}"
    );
    assert!(
        delta_256 < 1.0,
        "head_dim 256 (the shipped sliding-layer path) disagreed with the f32 reference \
         by {delta_256} at 10× realistic magnitude — the safety margin this test exists \
         to confirm didn't hold"
    );
    assert!(
        delta_512 > 10.0,
        "head_dim 512 no longer reproduces its own known breakage ({delta_512} < 10) — \
         if candle-flash-attn was updated, this is good news: revisit whether global \
         layers can use flash-attn too (issue #308)"
    );
}

/// Chunked prefill (`generate_reusing` feeding the prompt to `forward` in
/// `GALLIUM_PREFILL_CHUNK`-token windows instead of one shot — the fix for the
/// GPU OOM on a ~20k-token prompt) must be **exact**: the KV cache carries
/// context across windows, so the greedy stream is byte-identical to a single
/// prefill. The prompt here spans the E4B sliding window (512) so the run also
/// crosses that boundary mid-prefill, exercising the narrowed-K/V mask path at
/// a non-zero `pos`.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gemma4_gguf_chunked_prefill_matches_single() {
    let gguf_path = std::env::var("GALLIUM_GEMMA4_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gemma4_gguf_chunked_prefill: model not found");
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
        eprintln!("SKIP gemma4_gguf_chunked_prefill: no tokenizer");
        return;
    };
    let device = test_device();

    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(70);
    let prompt = format!(
        "<bos><|turn>user\n{filler}\nIn one sentence, what animal is mentioned?<turn|>\n<|turn>model\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(
        prompt_ids.len() > 600,
        "prompt must span several chunks and the 512 window"
    );

    // `GALLIUM_PREFILL_CHUNK` is process-wide; share the env A/B lock so a
    // concurrent test can't observe this one's temporary value.
    let _env = kv_narrow::lock_and_restore("GALLIUM_PREFILL_CHUNK");
    let n_gen = 24;
    let run = |chunk: &str| -> Vec<u32> {
        std::env::set_var("GALLIUM_PREFILL_CHUNK", chunk);
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, false)
                .expect("load model");
        let mut ids = Vec::new();
        generate(&mut model, &prompt_ids, &greedy(), n_gen, &[], |id| {
            ids.push(id);
            ControlFlow::Continue(())
        })
        .expect("generate");
        ids
    };

    let single = run("0"); // chunking disabled — one forward over the whole prompt
    let chunked = run("128"); // ~5 windows, crossing the 512 window boundary
    assert_eq!(
        chunked, single,
        "chunked prefill must produce the identical greedy stream"
    );
    assert_eq!(chunked.len(), n_gen);
}

/// The same exactness contract for the **safetensors** Gemma 4 path
/// (`gemma4.rs` + `gallium_core::Attention`), where #232 moved
/// `narrow_kv_to_mask` and the sliding branch now picks
/// `build_sliding_window_mask_narrowed`. Greedy stream must be byte-identical
/// with narrowing on vs off. Runs on CPU (safetensors E4B keeps ~7 GB of PLE +
/// embeddings on-device — it OOMs a 12 GB card) and is slow; `#[ignore]`d.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gemma4_safetensors_kv_narrowing_is_exact() {
    // Prefer the text-only base checkpoint (what `gemma4_safetensors` uses);
    // fall back to the E4B-it multimodal one, whose text half loads the same.
    let dir = std::env::var("GALLIUM_GEMMA4_SAFETENSORS_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_snapshot("google/gemma-4-E4B"))
        .or_else(|| hf_snapshot("unsloth/gemma-4-E4B-it"));
    let Some(dir) = dir else {
        eprintln!("SKIP gemma4_safetensors_kv_narrowing: model not found");
        return;
    };
    let safetensors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    if safetensors.is_empty() {
        eprintln!("SKIP gemma4_safetensors_kv_narrowing: no .safetensors in {dir:?}");
        return;
    }
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    // Prompt must comfortably exceed the 512 window so every sliding-layer
    // decode step runs against a cache the narrowing actually shortens. Kept
    // modest — safetensors E4B on CPU (the only device that fits it) is slow.
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(80);
    let prompt = format!("<bos>{filler}\nThe animal in the sentences above is the");
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt_ids.len() > 650, "prompt must dwarf the 512 window");

    let _env = kv_narrow::lock_and_restore("GALLIUM_GEMMA4_KV_NARROW");

    let full: serde_json::Value =
        gallium_models::loader::load_config(&dir.join("config.json")).expect("config");
    let text_cfg = full.get("text_config").unwrap_or(&full).clone();
    let cfg: gallium_models::gemma4::Gemma4Config =
        serde_json::from_value(text_cfg).expect("parse gemma4 config");

    let (off_ids, on_ids) = kv_narrow::greedy_ab(
        "GALLIUM_GEMMA4_KV_NARROW",
        || {
            let vb =
                gallium_models::loader::load_safetensors(&safetensors, DType::F16, &Device::Cpu)
                    .expect("load vb");
            gallium_models::gemma4::Gemma4::load(&cfg, vb, &Device::Cpu).expect("load model")
        },
        &prompt_ids,
        16,
    );

    assert_eq!(
        on_ids, off_ids,
        "narrowed K/V must produce the identical greedy stream (safetensors)"
    );
    assert_eq!(on_ids.len(), 16);
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
        gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, false)
            .expect("load model");

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

/// The same exactness contract as `gemma4_safetensors_kv_narrowing_is_exact`,
/// for `gpt_oss.rs` — issue #232's other half. GPT-OSS's window (128) is a
/// quarter of Gemma 4 E4B's (512), so it needs a much shorter filler to dwarf
/// it, and its attention sink (appended to scores *after* the mask — see
/// `gallium_core::attention::narrow_kv_to_mask`'s doc comment) is the reason
/// this needed checking at all rather than assuming the Gemma result carries
/// over. Greedy stream must be byte-identical with narrowing on vs off.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gpt_oss_safetensors_kv_narrowing_is_exact() {
    let dir = std::env::var("GALLIUM_GPT_OSS_SAFETENSORS_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_snapshot("openai/gpt-oss-20b"));
    let Some(dir) = dir else {
        eprintln!("SKIP gpt_oss_safetensors_kv_narrowing: model not found");
        return;
    };
    let safetensors: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read model dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    if safetensors.is_empty() {
        eprintln!("SKIP gpt_oss_safetensors_kv_narrowing: no .safetensors in {dir:?}");
        return;
    }
    let tokenizer = load_tokenizer(&dir).expect("tokenizer");

    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(60);
    let prompt = format!(
        "<|start|>system<|message|>You are a helpful assistant.<|end|>\
         <|start|>user<|message|>{filler}\nIn one sentence, what animal is mentioned?<|end|>\
         <|start|>assistant\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt_ids.len() > 300, "prompt must dwarf the 128 window");

    let _env = kv_narrow::lock_and_restore("GALLIUM_GPT_OSS_KV_NARROW");

    let config_path = dir.join("config.json");
    let cfg: gallium_models::gpt_oss::GptOssConfig =
        gallium_models::loader::load_config(&config_path).expect("config");

    let (off_ids, on_ids) = kv_narrow::greedy_ab(
        "GALLIUM_GPT_OSS_KV_NARROW",
        || {
            let vb =
                gallium_models::loader::load_safetensors(&safetensors, DType::F16, &Device::Cpu)
                    .expect("load vb");
            gallium_models::gpt_oss::GptOss::load(&cfg, vb, &safetensors, &Device::Cpu)
                .expect("load model")
        },
        &prompt_ids,
        16,
    );

    assert_eq!(
        on_ids, off_ids,
        "narrowed K/V must produce the identical greedy stream (gpt-oss safetensors)"
    );
    assert_eq!(on_ids.len(), 16);
}

// ---------------------------------------------------------------------------
// GPT-OSS — GGUF
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gpt_oss_gguf() {
    let gguf_path = std::env::var("GALLIUM_GPT_OSS_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        Some(p) => {
            eprintln!("SKIP gpt_oss_gguf: path {:?} does not exist", p);
            return;
        }
        None => {
            eprintln!("SKIP gpt_oss_gguf: set GALLIUM_GPT_OSS_GGUF_PATH or cache unsloth/gpt-oss-20b-GGUF");
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

    let mut model = gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device, &device)
        .expect("load model");

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

/// The same exactness-and-speed contract as
/// `gemma4_gguf_kv_narrowing_is_exact_and_faster`, for `gpt_oss_q.rs` —
/// issue #232's GGUF half. GPT-OSS's window (128) is a quarter of Gemma 4
/// E4B's (512), so the win-per-decode-step ratio should be the largest of any
/// model this repo runs on candle — a much shorter filler already dwarfs the
/// window, since the win is proportional to `total_len / window`.
///
/// `GALLIUM_KVTEST_FILLER` (default 60) and `GALLIUM_KVTEST_GEN` (default 64)
/// size the prompt and generation, same knobs as the Gemma 4 test — sized
/// smaller by default since GPT-OSS's window needs far less filler to dwarf.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gpt_oss_gguf_kv_narrowing_is_exact_and_faster() {
    use std::time::Instant;

    let gguf_path = std::env::var("GALLIUM_GPT_OSS_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gpt_oss_gguf_kv_narrowing: model not found");
        return;
    };
    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .unwrap()
    } else if let Some(snap) = hf_snapshot("openai/gpt-oss-20b") {
        load_tokenizer(&snap).unwrap()
    } else {
        eprintln!("SKIP gpt_oss_gguf_kv_narrowing: no tokenizer");
        return;
    };
    let device = test_device();

    let reps: usize = std::env::var("GALLIUM_KVTEST_FILLER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let n_gen: usize = std::env::var("GALLIUM_KVTEST_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(reps);
    let prompt = format!(
        "<|start|>system<|message|>You are a helpful assistant.<|end|>\
         <|start|>user<|message|>{filler}\nIn one sentence, what animal is mentioned?<|end|>\
         <|start|>assistant\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(prompt_ids.len() > 300, "prompt must dwarf the 128 window");

    let _env = kv_narrow::lock_and_restore("GALLIUM_GPT_OSS_KV_NARROW");

    let run = |narrow: bool| -> (Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GPT_OSS_KV_NARROW", if narrow { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model = gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device, &device)
            .expect("load model");
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

    let (off_ids, off_pre, off_dec) = run(false);
    let (on_ids, on_pre, on_dec) = run(true);

    let dec_per = |s: f64| (n_gen.saturating_sub(1)) as f64 / s;
    eprintln!(
        "gpt-oss kv-narrow ({} prompt tok, {n_gen} gen): prefill {off_pre:.1}s→{on_pre:.1}s | \
         decode {off_dec:.1}s→{on_dec:.1}s ({:.1}→{:.1} tok/s, {:.2}x)",
        prompt_ids.len(),
        dec_per(off_dec),
        dec_per(on_dec),
        off_dec / on_dec.max(1e-6),
    );
    let first_diff = off_ids.iter().zip(&on_ids).position(|(a, b)| a != b);
    eprintln!("  narrow on vs off: first token divergence at {first_diff:?} of {n_gen}");
    assert_eq!(
        on_ids, off_ids,
        "narrowed K/V must produce the identical greedy stream"
    );
    assert_eq!(on_ids.len(), n_gen);
}

/// Is the fused MXFP4 decode path (`gpt_oss_q.rs`, on by default) deterministic
/// run to run? It fans out across experts *and* across each expert's output
/// rows (`Tq2Tensor::matvec_expert`'s `par_iter`), so a non-order-preserving
/// collect or a shared-state bug would make greedy decode a coin flip.
///
/// Two fresh loads, same prompt, greedy. **Short context on purpose**
/// (`GALLIUM_KVTEST_FILLER` default 20): candle's own CPU GEMM in the attention
/// path is not bit-reproducible at long context — its tiled reduction order
/// depends on rayon pool state — so a 2000-token prompt makes *any* GPT-OSS
/// decode flaky (~1 run in 3), fused kernel or not. This test isolates the
/// fused row-gather, which is a pure per-row map and stays exact.
#[test]
#[ignore = "needs a local model in the HF cache; run with `make test-models`"]
fn gpt_oss_gguf_fused_decode_is_deterministic() {
    let gguf_path = std::env::var("GALLIUM_GPT_OSS_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf"));
    let Some(gguf_path) = gguf_path else {
        eprintln!("SKIP gpt_oss_gguf_fused_decode_is_deterministic: model not found");
        return;
    };
    let tok_path = gguf_path.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .unwrap()
    } else if let Some(snap) = hf_snapshot("openai/gpt-oss-20b") {
        load_tokenizer(&snap).unwrap()
    } else {
        eprintln!("SKIP gpt_oss_gguf_fused_decode_is_deterministic: no tokenizer");
        return;
    };
    let device = test_device();
    let reps: usize = std::env::var("GALLIUM_KVTEST_FILLER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let n_gen: usize = std::env::var("GALLIUM_KVTEST_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(reps);
    let prompt = format!(
        "<|start|>system<|message|>You are a helpful assistant.<|end|>\
         <|start|>user<|message|>{filler}\nIn one sentence, what animal is mentioned?<|end|>\
         <|start|>assistant\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();

    let run = || -> Vec<u32> {
        let (metadata, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
        let mut model = gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device, &device)
            .expect("load model");
        let mut ids = Vec::new();
        generate(&mut model, &prompt_ids, &greedy(), n_gen, &[], |id| {
            ids.push(id);
            ControlFlow::Continue(())
        })
        .expect("generate");
        ids
    };

    let a = run();
    let b = run();
    let first_diff = a.iter().zip(&b).position(|(x, y)| x != y);
    eprintln!(
        "fused decode determinism ({} prompt tok, {n_gen} gen): first divergence at {:?}",
        prompt_ids.len(),
        first_diff
    );
    assert_eq!(a, b, "fused MXFP4 decode must be deterministic run to run");
}

/// The same bit-equality contract as
/// `gemma4_gguf_token_embd_gather_matches_whole_dequantize`, for `gpt_oss_q.rs`
/// (issue #255, mirroring #252) — the ~2.3 GB (20B, Q5_0 → f32) `token_embd`
/// table `GptOssQ` used to dequantize whole onto the device at load, row-gathered
/// per forward instead.
#[test]
#[ignore]
fn gpt_oss_gguf_token_embd_gather_matches_whole_dequantize() {
    let Some(gguf) = hf_file("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf") else {
        eprintln!("SKIP: gpt-oss-20b GGUF not in the HF cache");
        return;
    };
    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf, &device).expect("load gpt-oss-20b GGUF");
    let table = vb
        .get_experts("token_embd.weight")
        .expect("GGUF carries token_embd");

    // Spread of ids with a repeat (id order must be preserved, a repeated id
    // must produce two identical rows); includes the last valid row
    // (vocab_size 201088).
    let ids: Vec<u32> = vec![0, 1, 42, 100000, 42, 201087];
    let gathered = table.gather_rows(&ids, &device).expect("gather_rows");
    for (i, &id) in ids.iter().enumerate() {
        let want: Vec<f32> = table
            .dequantize_expert(id as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = gathered.i(i).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "row {i} (id {id}) differs from its own per-row dequantization"
        );
    }

    // Prefill-sized gather — past the point `free` unmaps the allocation,
    // which is the half that would catch a `Cow::Owned` regression rather
    // than describe it.
    let many: Vec<u32> = (0..2048u32).map(|i| i * 7 % 201_088).collect();
    let bulk = table
        .gather_rows(&many, &device)
        .expect("prefill-sized gather");
    for probe in [0usize, 1, 1023, 2047] {
        let want: Vec<f32> = table
            .dequantize_expert(many[probe] as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = bulk.i(probe).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "bulk row {probe} (id {}) differs from its own per-row dequantization",
            many[probe]
        );
    }
}

/// `Tq2Tensor::matvec_expert` (the fused MXFP4 stream-and-dot path used for a
/// single-token decode in `gpt_oss_q.rs`) must track the expand-then-`matmul`
/// path closely — **not** bit-exactly: the reduction order differs, which is
/// exactly why `gpt_oss_q.rs` gates it behind `GALLIUM_GPT_OSS_FUSED_MXFP4`
/// and an A/B testsuite run. This just catches a gross decode/index bug; the
/// real check is the testsuite comparison in docs/VERIFICATION_STATUS.md.
#[test]
#[ignore]
fn gpt_oss_gguf_mxfp4_matvec_tracks_dequantize_matmul() {
    use gallium_core::KernelSet;

    let Some(gguf) = hf_file("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q4_K_M.gguf") else {
        eprintln!("SKIP: gpt-oss-20b GGUF not in the HF cache");
        return;
    };
    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf, &device).expect("load gpt-oss-20b GGUF");
    let kernels = KernelSet::detect();

    for name in [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ] {
        let t = vb.pp("blk.0").get_tq2(name).expect("expert tensor");
        let d_in = *t.dims.last().unwrap();
        let mut lcg: u64 = 0x9e37_79b9_7f4a_7c15;
        let x: Vec<f32> = (0..d_in)
            .map(|_| {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                (lcg >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect();
        let x_t = candle_core::Tensor::from_slice(&x, (1, d_in), &device).unwrap();

        for expert in [0usize, 7, 31] {
            let fused = t
                .matvec_expert(expert, &x, &kernels)
                .expect("matvec_expert");
            let w = t
                .dequantize_expert(expert, &device)
                .expect("dequantize_expert");
            let reference: Vec<f32> = x_t
                .matmul(&w.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(fused.len(), reference.len());
            let max_rel = fused
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs() / (b.abs() + 1e-3))
                .fold(0f32, f32::max);
            assert!(
                max_rel < 2e-3,
                "{name} expert {expert}: max relative diff {max_rel} between fused matvec and dequantize+matmul"
            );
        }
    }
}

/// The reason `load_gguf` learned to merge split-GGUF shards: every quantized
/// 120B GGUF on the hub is a 2-shard split (`gallium-core/src/quantized.rs`'s
/// `split_shard_paths` / `load_gguf_shards`), ~63 GB total — too big to fit a
/// 12 GB or 24 GB reference card, so this forces `Device::Cpu` regardless of
/// `GALLIUM_DEVICE` rather than risking an accelerator OOM. Layer 28's
/// experts straddle the shard boundary (`blk.28.ffn_down_exps.weight` ends
/// shard 1, `blk.28.ffn_gate_exps.weight` opens shard 2 — verified by reading
/// both shards' headers with the `gguf` python library), so a real forward
/// pass through that layer exercises tensors from both mmaps in one op, not
/// just the header-merge `split_gguf_tests` in `quantized.rs` already covers
/// without any real weights.
#[test]
#[ignore = "needs the 120b split GGUF in the HF cache (~63 GB, 2 shards); CPU-only, budget minutes"]
fn gpt_oss_120b_gguf_split() {
    let shard1 = std::env::var("GALLIUM_GPT_OSS_120B_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let snap = hf_snapshot("unsloth/gpt-oss-120b-GGUF")?;
            std::fs::read_dir(&snap)
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .flat_map(|d| std::fs::read_dir(&d).into_iter().flatten())
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.contains("-00001-of-") && n.ends_with(".gguf"))
                })
        });

    let shard1 = match shard1 {
        Some(p) if p.exists() => p,
        Some(p) => {
            eprintln!("SKIP gpt_oss_120b_gguf_split: path {:?} does not exist", p);
            return;
        }
        None => {
            eprintln!(
                "SKIP gpt_oss_120b_gguf_split: no split GGUF found (set \
                 GALLIUM_GPT_OSS_120B_GGUF_PATH to shard 1, or cache \
                 unsloth/gpt-oss-120b-GGUF)"
            );
            return;
        }
    };

    let device = Device::Cpu;
    let load_start = std::time::Instant::now();
    let (metadata, vb) = load_gguf(&shard1, &device).expect("load split gguf");
    eprintln!(
        "gpt_oss_120b_gguf_split: header merge across shards in {:?}, {} tensors",
        load_start.elapsed(),
        vb.tensor_names().len()
    );
    assert_eq!(metadata.get_str("general.architecture").unwrap(), "gpt-oss");

    let tokenizer = if let Some(snap) = hf_snapshot("openai/gpt-oss-120b") {
        load_tokenizer(&snap).expect("tokenizer from openai/gpt-oss-120b snapshot")
    } else {
        eprintln!("SKIP gpt_oss_120b_gguf_split: no tokenizer cached (openai/gpt-oss-120b)");
        return;
    };

    let mut model = gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, &device, &device)
        .expect("load model");

    let prompt = "<|start|>system<|message|>You are a helpful assistant.<|end|>\
                  <|start|>user<|message|>What is the capital of France?<|end|>\
                  <|start|>assistant\n";
    // GPT-OSS answers through Harmony's `analysis` channel before `final` —
    // `run_inference` decodes raw tokens with no channel stripping, so the
    // budget has to cover the reasoning preamble too, not just the answer.
    let infer_start = std::time::Instant::now();
    let output = run_inference(&mut model, &tokenizer, prompt, 80).expect("inference");
    eprintln!(
        "gpt_oss_120b_gguf_split output ({:?}): {:?}",
        infer_start.elapsed(),
        output
    );
    assert!(
        output.to_lowercase().contains("paris"),
        "expected 'Paris' in output, got: {:?}",
        output
    );
}

// ---------------------------------------------------------------------------
// Qwen 3.5 — GGUF
//
// Safetensors (`qwen35.rs`, `qwen35_safetensors`) was dropped for maintenance
// cost — GGUF only now. See docs/models/qwen35.md and CLAUDE.md's model-files
// table.
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

/// The same bit-equality contract as
/// `gpt_oss_gguf_token_embd_gather_matches_whole_dequantize` /
/// `gemma4_gguf_token_embd_gather_matches_whole_dequantize`, for `qwen35_q.rs`
/// — `Qwen35Q` used to dequantize `token_embd` whole onto the device at load
/// (via `Embedding::new`), row-gathered per forward instead.
#[test]
#[ignore]
fn qwen35_gguf_token_embd_gather_matches_whole_dequantize() {
    let gguf_path = std::env::var("GALLIUM_QWEN35_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!(
                "SKIP qwen35_gguf_token_embd_gather_matches_whole_dequantize: set \
                 GALLIUM_QWEN35_GGUF_PATH or cache unsloth/Qwen3.5-9B-GGUF"
            );
            return;
        }
    };

    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
    let table = vb
        .get_experts("token_embd.weight")
        .expect("GGUF carries token_embd");
    let vocab = table.n_experts();

    // Spread of ids with a repeat (id order must be preserved, a repeated id
    // must produce two identical rows); includes the last valid row.
    let ids: Vec<u32> = vec![0, 1, 42, 100000 % vocab as u32, 42, (vocab - 1) as u32];
    let gathered = table.gather_rows(&ids, &device).expect("gather_rows");
    for (i, &id) in ids.iter().enumerate() {
        let want: Vec<f32> = table
            .dequantize_expert(id as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = gathered.i(i).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "row {i} (id {id}) differs from its own per-row dequantization"
        );
    }

    // Prefill-sized gather — past the point `free` unmaps the allocation,
    // which is the half that would catch a `Cow::Owned` regression rather
    // than describe it.
    let many: Vec<u32> = (0..2048u32).map(|i| i * 7 % vocab as u32).collect();
    let bulk = table
        .gather_rows(&many, &device)
        .expect("prefill-sized gather");
    for probe in [0usize, 1, 1023, 2047] {
        let want: Vec<f32> = table
            .dequantize_expert(many[probe] as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = bulk.i(probe).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "bulk row {probe} (id {}) differs from its own per-row dequantization",
            many[probe]
        );
    }
}

/// The same bit-equality contract as
/// `qwen35_gguf_token_embd_gather_matches_whole_dequantize`, for `lfm2moe_q.rs`
/// — `Lfm2MoeQ` used to dequantize `token_embd` whole onto the device at load
/// (via `Embedding::new`), row-gathered per forward instead.
#[test]
#[ignore]
fn lfm2_gguf_token_embd_gather_matches_whole_dequantize() {
    let gguf_path = std::env::var("GALLIUM_LFM2_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_file("LiquidAI/LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q4_K_M.gguf"));

    let gguf_path = match gguf_path {
        Some(p) if p.exists() => p,
        _ => {
            eprintln!(
                "SKIP lfm2_gguf_token_embd_gather_matches_whole_dequantize: set \
                 GALLIUM_LFM2_GGUF_PATH or cache LiquidAI/LFM2.5-8B-A1B-GGUF"
            );
            return;
        }
    };

    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf_path, &device).expect("load gguf");
    let table = vb
        .get_experts("token_embd.weight")
        .expect("GGUF carries token_embd");
    let vocab = table.n_experts();

    // Spread of ids with a repeat (id order must be preserved, a repeated id
    // must produce two identical rows); includes the last valid row.
    let ids: Vec<u32> = vec![0, 1, 42, 100000 % vocab as u32, 42, (vocab - 1) as u32];
    let gathered = table.gather_rows(&ids, &device).expect("gather_rows");
    for (i, &id) in ids.iter().enumerate() {
        let want: Vec<f32> = table
            .dequantize_expert(id as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = gathered.i(i).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "row {i} (id {id}) differs from its own per-row dequantization"
        );
    }

    // Prefill-sized gather — past the point `free` unmaps the allocation,
    // which is the half that would catch a `Cow::Owned` regression rather
    // than describe it.
    let many: Vec<u32> = (0..2048u32).map(|i| i * 7 % vocab as u32).collect();
    let bulk = table
        .gather_rows(&many, &device)
        .expect("prefill-sized gather");
    for probe in [0usize, 1, 1023, 2047] {
        let want: Vec<f32> = table
            .dequantize_expert(many[probe] as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = bulk.i(probe).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "bulk row {probe} (id {}) differs from its own per-row dequantization",
            many[probe]
        );
    }
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
        gallium_models::lfm2moe_q::Lfm2MoeQ::load(&metadata, &vb, &device, &device).expect("model")
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

/// Candle GGUF vision: `Gemma4Multimodal::load_gguf` builds the tower from the
/// mmproj (llama.cpp's `v.blk.*`/`mm.*` names, renamed to the safetensors
/// paths) and a synthetic image encodes to well-formed soft tokens. The rename
/// table itself was verified bit-exact against the safetensors originals; this
/// guards the load path end to end on the two files `make testsuite` already
/// caches. Ignored: needs the multi-GB E4B GGUF + mmproj in the HF cache.
#[test]
#[ignore]
fn gemma4_gguf_mmproj_vision_tower() {
    let repo = "unsloth/gemma-4-E4B-it-GGUF";
    let (Some(gguf), Some(mmproj)) = (
        hf_file(repo, "gemma-4-E4B-it-Q4_K_M.gguf"),
        hf_file(repo, "mmproj-BF16.gguf"),
    ) else {
        eprintln!("SKIP: {repo} GGUF/mmproj not in the HF cache");
        return;
    };

    let device = test_device();
    let (metadata, vb) = load_gguf(&gguf, &device).expect("text GGUF");
    let (model, vc) = gallium_models::gemma4_vision::Gemma4Multimodal::load_gguf(
        &metadata, &vb, &mmproj, &device, false,
    )
    .expect("mmproj tower load");

    // clip.vision.* metadata → the processor's config.
    assert_eq!(
        (
            vc.hidden_size,
            vc.num_hidden_layers,
            vc.patch_size,
            vc.pooling_kernel_size
        ),
        (768, 16, 16, 3),
    );

    // A 12×12-patch gradient (192×192 px): 144 patches pool 3×3 → 16 soft tokens.
    let (nph, npw, ps) = (12usize, 12usize, vc.patch_size);
    let n = nph * npw;
    let plen = 3 * ps * ps;
    let pixels: Vec<f32> = (0..n * plen)
        .map(|i| ((i / plen) as f32) / (n as f32)) // constant per patch, gradient across
        .collect();
    let pv = candle_core::Tensor::from_vec(pixels, (1, n, plen), &device).unwrap();
    let mut pos = Vec::with_capacity(n * 2);
    for pr in 0..nph {
        for pc in 0..npw {
            pos.push(pc as i64);
            pos.push(pr as i64);
        }
    }
    let pos = candle_core::Tensor::from_vec(pos, (1, n, 2), &device).unwrap();

    let feats = model.encode_image(&pv, &pos).expect("encode_image");
    let dims = feats.dims2().expect("2-d soft tokens");
    assert_eq!(dims.0, n / 9, "144 patches pool to 16 soft tokens");

    let v: Vec<f32> = feats
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "no NaN/inf in soft tokens");
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    assert!(var > 1e-6, "soft tokens are not a constant (var={var})");
}

/// 26B-A4B's mmproj (unlike E4B/12B's) carries a `v.std_bias`/`v.std_scale`
/// QAT recalibration affine — llama.cpp's `gemma4v.cpp`: `hidden_states =
/// (hidden_states - std_bias) * std_scale`, applied after pooling and before
/// the projector's RMSNorm. Confirms both that the tensors load (`load_gguf`
/// used to bail with "unrecognized mmproj tensor name: v.std_bias") and that
/// applying them actually changes the encoded features —
/// `GALLIUM_ABLATE_STD_AFFINE=1` is the same switch used to verify this by
/// hand against the testsuite's `multimodal_image` case, which happens to
/// read the same either way (a plain digit is not a sensitive enough probe).
/// Ignored: needs the multi-GB 26B-A4B GGUF + mmproj in the HF cache.
#[test]
#[ignore]
fn gemma4_26b_gguf_mmproj_std_affine() {
    let repo = "unsloth/gemma-4-26B-A4B-it-qat-GGUF";
    let (Some(gguf), Some(mmproj)) = (
        hf_file(repo, "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"),
        hf_file(repo, "mmproj-BF16.gguf"),
    ) else {
        eprintln!("SKIP: {repo} GGUF/mmproj not in the HF cache");
        return;
    };

    let device = test_device();
    let (metadata, vb) = load_gguf(&gguf, &device).expect("text GGUF");
    let (model, vc) = gallium_models::gemma4_vision::Gemma4Multimodal::load_gguf(
        &metadata, &vb, &mmproj, &device, false,
    )
    .expect("mmproj tower load");

    let (nph, npw, ps) = (12usize, 12usize, vc.patch_size);
    let n = nph * npw;
    let plen = 3 * ps * ps;
    let pixels: Vec<f32> = (0..n * plen)
        .map(|i| ((i / plen) as f32) / (n as f32))
        .collect();
    let pv = candle_core::Tensor::from_vec(pixels, (1, n, plen), &device).unwrap();
    let mut pos = Vec::with_capacity(n * 2);
    for pr in 0..nph {
        for pc in 0..npw {
            pos.push(pc as i64);
            pos.push(pr as i64);
        }
    }
    let pos = candle_core::Tensor::from_vec(pos, (1, n, 2), &device).unwrap();

    let stats = |feats: &candle_core::Tensor| -> (f32, f32) {
        let v: Vec<f32> = feats
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        (mean, var)
    };

    std::env::remove_var("GALLIUM_ABLATE_STD_AFFINE");
    let with_affine = model.encode_image(&pv, &pos).expect("encode_image");
    let (mean_on, var_on) = stats(&with_affine);
    assert!(mean_on.is_finite() && var_on > 1e-6);

    std::env::set_var("GALLIUM_ABLATE_STD_AFFINE", "1");
    let without_affine = model.encode_image(&pv, &pos).expect("encode_image");
    std::env::remove_var("GALLIUM_ABLATE_STD_AFFINE");
    let (mean_off, var_off) = stats(&without_affine);
    assert!(mean_off.is_finite() && var_off > 1e-6);

    eprintln!("std_affine on:  mean={mean_on} var={var_on}");
    eprintln!("std_affine off: mean={mean_off} var={var_off}");
    assert!(
        (mean_on - mean_off).abs() > 1e-4 || (var_on - var_off).abs() > 1e-4,
        "the std_bias/std_scale affine made no measurable difference \
         (on: mean={mean_on} var={var_on}, off: mean={mean_off} var={var_off}) — \
         either the tensors didn't load or encode_image stopped applying them"
    );
}

/// `QExperts::gather_rows` must return exactly what dequantizing those rows
/// one at a time returns — the "bit-identical to a whole-table dequantization"
/// claim the PLE row-gather is built on, and the one nothing checked when it
/// landed.
///
/// Rows rather than the whole table on purpose: E4B's `per_layer_token_embd`
/// is ~11 GB dequantized, which is exactly why `gather_rows` exists.
/// `dequantize_expert` reads the same byte range through the borrowed path, so
/// comparing against it costs a few KB and still pins bit-equality.
///
/// **This pins the invariant; it is not a reliable guard against the bug that
/// broke it.** `gather_rows` handed its freshly-built `Vec` to
/// `QStorage::from_data` as a `Cow::Owned`, and candle's `as_t_slice` takes
/// that `Cow` by value and returns a slice borrowed from it — so the copy that
/// follows read freed heap, and the model was *sometimes* wrong: greedy decode
/// diverged in 4 of 8 runs of one fixed prompt, and one run collapsed into a
/// single repeated token. Whether a stale read comes back intact is the
/// allocator's business, and this test passes with the bug reintroduced at
/// both sizes below. What catches it deterministically is a sanitizer, Miri,
/// or the `owned_storage_regression` probe in `gallium-core`, where the
/// replacement allocation lands on the just-freed block and trips std's
/// `copy_nonoverlapping` overlap check. Read this test as documentation of
/// the contract, and keep `Cow::Borrowed` at the call site.
#[test]
#[ignore]
fn gemma4_gguf_ple_gather_matches_per_row_dequantize() {
    let Some(gguf) = hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf") else {
        eprintln!("SKIP: E4B GGUF not in the HF cache");
        return;
    };
    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf, &device).expect("load E4B GGUF");
    let table = vb
        .get_experts("per_layer_token_embd.weight")
        .expect("E4B carries a PLE table");

    // A spread of ids, with a repeat: the gather concatenates row bytes in id
    // order, so a repeated id must produce two identical rows and not shift
    // the ones after it.
    let ids: Vec<u32> = vec![0, 1, 9264, 258880, 9264, 262143];
    let gathered = table.gather_rows(&ids, &device).expect("gather_rows");

    for (i, &id) in ids.iter().enumerate() {
        let want: Vec<f32> = table
            .dequantize_expert(id as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = gathered.i(i).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "row {i} (id {id}) differs from its own per-row dequantization"
        );
    }

    // And again at the size a real prefill gathers. This half is what actually
    // catches the bug: a few rows is ~30 KB, which `malloc` serves from its
    // heap and does not hand back on free, so the stale read is still intact
    // and a small gather passes either way. A prefill's ~1600 rows is ~8 MB,
    // past the threshold where the allocation is its own `mmap` and `free`
    // unmaps it — the difference between a test that pins the invariant and
    // one that describes it.
    let many: Vec<u32> = (0..2048u32).map(|i| i * 7 % 262_144).collect();
    let bulk = table
        .gather_rows(&many, &device)
        .expect("prefill-sized gather");
    for probe in [0usize, 1, 1023, 2047] {
        let want: Vec<f32> = table
            .dequantize_expert(many[probe] as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = bulk.i(probe).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "bulk row {probe} (id {}) differs from its own per-row dequantization",
            many[probe]
        );
    }
}

/// The same bit-equality contract as the PLE test above, for the **main
/// `token_embd.weight`** — the ~2.7 GB (E4B, f32) table `Gemma4Q` used to
/// dequantize whole onto the device at load. It is now row-gathered per forward
/// (`embed_scaled`), the change that brings E4B's candle-CUDA footprint down to
/// llama.cpp's and lets the 12B GGUF fit a 12 GB card at all (docs/TODO.md §3).
///
/// `token_embd` is `[vocab, hidden]` — the degenerate `QExperts` the same as the
/// PLE table, just a smaller inner dim — so the gather goes through the identical
/// `Cow::Borrowed` path and this pins the same invariant: keep `Cow::Borrowed`
/// at the `QStorage::from_data` call site (see the PLE test's note on why this
/// documents rather than deterministically guards the use-after-free).
#[test]
#[ignore]
fn gemma4_gguf_token_embd_gather_matches_whole_dequantize() {
    let Some(gguf) = hf_file("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf") else {
        eprintln!("SKIP: E4B GGUF not in the HF cache");
        return;
    };
    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf, &device).expect("load E4B GGUF");
    let table = vb
        .get_experts("token_embd.weight")
        .expect("GGUF carries token_embd");

    // Spread of ids with a repeat (id order must be preserved, a repeated id
    // must produce two identical rows); includes the last valid row.
    let ids: Vec<u32> = vec![0, 1, 42, 258880, 42, 262143];
    let gathered = table.gather_rows(&ids, &device).expect("gather_rows");
    for (i, &id) in ids.iter().enumerate() {
        let want: Vec<f32> = table
            .dequantize_expert(id as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = gathered.i(i).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "row {i} (id {id}) differs from its own per-row dequantization"
        );
    }

    // Prefill-sized gather — ~1600 rows × 2560 f32 is ~16 MB, past the point
    // `free` unmaps the allocation, which is the half that would catch a
    // `Cow::Owned` regression rather than describe it.
    let many: Vec<u32> = (0..2048u32).map(|i| i * 7 % 262_144).collect();
    let bulk = table
        .gather_rows(&many, &device)
        .expect("prefill-sized gather");
    for probe in [0usize, 1, 1023, 2047] {
        let want: Vec<f32> = table
            .dequantize_expert(many[probe] as usize, &device)
            .expect("per-row dequantize")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let got: Vec<f32> = bulk.i(probe).unwrap().to_vec1().unwrap();
        assert_eq!(
            got, want,
            "bulk row {probe} (id {}) differs from its own per-row dequantization",
            many[probe]
        );
    }
}

/// `QExperts::matvec_expert` (candle's quantized matmul against one expert's
/// mmap-resident bytes — the single-token-decode fast path `QGemmaMoe` uses
/// behind `GALLIUM_GEMMA4_FUSED`) must track `dequantize_expert` + `matmul` on
/// a real merged expert tensor: right byte slice, right shape, right transpose.
///
/// **Not** bit-exact — `quantized.rs::qmatmul_equivalence` already records that
/// candle's ggml kernel quantizes the activations to 8 bits, so the two paths
/// differ by ~1% of the output scale — which is why the model gates the fast
/// path to one row and A/Bs the testsuite. Measured here relative to the
/// output's own magnitude (per-element relative is meaningless near a zero
/// crossing of a dot product). A transpose/slice bug would blow past this.
///
/// Needs the 26B-A4B MoE GGUF (the E4B/12B Gemma 4 are dense — no `QGemmaMoe`).
#[test]
#[ignore = "needs unsloth/gemma-4-26B-A4B-it-qat-GGUF in the HF cache"]
fn gemma4_26b_gguf_matvec_tracks_dequantize_matmul() {
    let gguf = std::env::var("GALLIUM_GEMMA4_26B_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            hf_file(
                "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
                "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
            )
        });
    let Some(gguf) = gguf.filter(|p| p.exists()) else {
        eprintln!("SKIP: gemma-4-26B-A4B GGUF not in the HF cache");
        return;
    };
    let device = Device::Cpu;
    let (_meta, vb) = load_gguf(&gguf, &device).expect("load gemma-4-26B GGUF");

    for name in ["ffn_gate_up_exps.weight", "ffn_down_exps.weight"] {
        let t = vb.pp("blk.0").get_experts(name).expect("expert tensor");
        let d_in = *t.expert_shape().last().unwrap();
        let mut lcg: u64 = 0x2545_f491_4f6c_dd1d;
        let x: Vec<f32> = (0..d_in)
            .map(|_| {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                (lcg >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect();
        let x_t = candle_core::Tensor::from_slice(&x, (1, d_in), &device).unwrap();

        for expert in [0usize, 5, 63] {
            let fused: Vec<f32> = t
                .matvec_expert(expert, &x_t, &device)
                .expect("matvec_expert")
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let w = t
                .dequantize_expert(expert, &device)
                .expect("dequantize_expert");
            let reference: Vec<f32> = x_t
                .matmul(&w.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(fused.len(), reference.len());
            let worst = fused
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = reference
                .iter()
                .map(|v| v.abs())
                .fold(0f32, f32::max)
                .max(1e-6);
            let dot: f32 = fused.iter().zip(&reference).map(|(a, b)| a * b).sum();
            let na = fused.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
            let cos = dot / (na * nb).max(1e-12);
            assert!(
                worst / scale < 3e-2 && cos > 0.999,
                "{name} expert {expert}: worst {worst} / scale {scale} = {} rel, cos {cos}",
                worst / scale
            );
        }
    }
}

/// Does the fused expert matvec (`GALLIUM_GEMMA4_FUSED`) actually speed up
/// `QGemmaMoe` **decode**? The testsuite quick cases are model-load-dominated
/// and can't show it; this times prefill and decode separately with the fast
/// path on vs off, and checks the greedy stream is unchanged for the first few
/// tokens (candle's quantized matmul is not bit-exact, so a late divergence is
/// expected and not asserted). Needs the 26B-A4B MoE GGUF.
#[test]
#[ignore = "needs unsloth/gemma-4-26B-A4B-it-qat-GGUF in the HF cache; slow"]
fn gemma4_26b_gguf_fused_decode_speed() {
    use std::time::Instant;

    let gguf = std::env::var("GALLIUM_GEMMA4_26B_GGUF_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            hf_file(
                "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
                "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
            )
        });
    let Some(gguf) = gguf.filter(|p| p.exists()) else {
        eprintln!("SKIP: gemma-4-26B-A4B GGUF not in the HF cache");
        return;
    };
    let tokenizer = match hf_snapshot("unsloth/gemma-4-26B-A4B-it") {
        Some(snap) => load_tokenizer(&snap).expect("tokenizer"),
        None => {
            eprintln!("SKIP gemma4_26b_gguf_fused_decode_speed: no tokenizer");
            return;
        }
    };
    let device = test_device();

    let reps: usize = std::env::var("GALLIUM_KVTEST_FILLER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let n_gen: usize = std::env::var("GALLIUM_KVTEST_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(reps);
    let prompt = format!(
        "<start_of_turn>user\n{filler}\nIn one sentence, what animal is mentioned?<end_of_turn>\n<start_of_turn>model\n"
    );
    let prompt_ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .unwrap()
        .get_ids()
        .to_vec();

    // Optionally exercise the resident expert cache (issue #253): set
    // `GALLIUM_EXPERT_CACHE_BYTES` to a budget. `test_device()` must be an
    // accelerator for it to attach — on CPU the bytes are already mmap-resident
    // and there is nothing to keep. Unlike `load_candle_provider`, this attaches
    // on Metal too: the agent refuses it there because it measured as a loss
    // (docs/VERIFICATION_STATUS.md "Resident expert cache on Metal"), and this
    // test is how that measurement is reproduced.
    let cache_bytes: Option<usize> = std::env::var("GALLIUM_EXPERT_CACHE_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|b| *b > 0 && !device.is_cpu());

    let _env = kv_narrow::lock_and_restore("GALLIUM_GEMMA4_FUSED");
    let run = |fused: bool| -> (Vec<u32>, f64, f64) {
        std::env::set_var("GALLIUM_GEMMA4_FUSED", if fused { "1" } else { "0" });
        let (metadata, vb) = load_gguf(&gguf, &device).expect("load gguf");
        let vb = match cache_bytes {
            Some(b) => vb.with_expert_cache(gallium_core::ExpertCache::new(b)),
            None => vb,
        };
        let mut model =
            gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, &device, &device, false)
                .expect("load model");
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

    let (off_ids, off_pre, off_dec) = run(false);
    let (on_ids, on_pre, on_dec) = run(true);
    let per = |s: f64| (n_gen.saturating_sub(1)) as f64 / s;
    eprintln!(
        "gemma4-26b fused ({} prompt tok, {n_gen} gen): prefill {off_pre:.1}s→{on_pre:.1}s | \
         decode {off_dec:.1}s→{on_dec:.1}s ({:.2}→{:.2} tok/s, {:.2}x)",
        prompt_ids.len(),
        per(off_dec),
        per(on_dec),
        off_dec / on_dec.max(1e-6),
    );
    let first_diff = off_ids.iter().zip(&on_ids).position(|(a, b)| a != b);
    eprintln!("  fused on vs off: first token divergence at {first_diff:?} of {n_gen}");
    assert_eq!(off_ids[..4.min(n_gen)], on_ids[..4.min(n_gen)]);
}
