#![cfg(feature = "local")]
//! Can a sequence's state be snapshotted and restored cheaply enough to reuse a
//! KV cache without replaying the model's own bytes?
//!
//! This is the measurement that settled it, recorded in
//! `docs/VERIFICATION_STATUS.md`. The answer #172 had at the time — replaying
//! each prior assistant turn verbatim — bought cache reuse for a
//! recurrent/hybrid model by requiring prompt *N+1* to be a byte-exact token
//! extension of what the cache held, and that requirement is what dragged the
//! model's own `<think>` block back into the prompt and cost the `refactoring`
//! testcase. A snapshot taken *before* generation has no such requirement:
//! restore it, evaluate whatever the template renders next, and the reuse
//! boundary is the prompt prefix rather than the model's output. That is what
//! `Slot::checkpoint` does today, and the replay is gone.
//!
//! Two things have to hold for that to be worth building:
//!
//! 1. **Equivalence.** A restored state plus a suffix must produce the same
//!    logits as a fresh context fed prompt + suffix. Anything less is a cache
//!    that returns plausible wrong answers, which is the one failure mode
//!    nothing downstream can detect.
//! 2. **Cost.** `get` + `set` must be far cheaper than re-evaluating the prompt,
//!    or the snapshot is just a slower prefill.
//!
//! Ignored, like every test that loads a multi-GB model, and **serial**: each
//! test initializes the llama.cpp backend and loads the model, and running them
//! concurrently fails on device init.
//!   cargo test -p gallium-agent --test kv_state_spike -- --ignored --nocapture --test-threads=1
//!
//! # Running it against another model
//!
//! The equivalence question is per **cache implementation**, not per model, and
//! LFM2's hybrid memory is not the only one gallium drives into the checkpoint
//! path. DeepSeek-V4 is the open case: `llama_kv_cache_dsv4` is `kv_raw` plus
//! three *compressed* sub-caches addressed at different position scales
//! (`p0/DSV4_CSA_RATIO`, `p0/DSV4_HCA_RATIO`), and a `state_seq_set` that
//! restored those inconsistently would produce plausible wrong logits — the one
//! failure nothing downstream can see. A passing testsuite does not rule it out;
//! that is what the argmax lesson below is about.
//!
//! So the model is a parameter:
//!
//! ```text
//! GALLIUM_KV_SPIKE_MODEL=unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
//! GALLIUM_KV_SPIKE_CPU_MOE=1 \
//!   cargo test -p gallium-agent --test kv_state_spike -- --ignored --nocapture --test-threads=1
//! ```
//!
//! | variable | default | what it is |
//! |---|---|---|
//! | `GALLIUM_KV_SPIKE_MODEL` | LFM2.5-8B-A1B Q4_K_M | an absolute path, or `ORG/REPO/FILE.gguf` looked up in the local HuggingFace cache — never downloaded, since these files are tens of GB |
//! | `GALLIUM_KV_SPIKE_GPU_LAYERS` | `999` | as `gpuLayers` in a config |
//! | `GALLIUM_KV_SPIKE_CPU_MOE` | off | `1` keeps MoE experts on the CPU, as `cpuMoe = true` does |
//! | `GALLIUM_KV_SPIKE_CTX` | `8192` | context size for every test here |
//!
//! Skips (rather than fails) when the model is not on this machine.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::session::LlamaStateSeqFlags;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

/// How far the logits may move across a restore before the state is not the same
/// state.
///
/// Not zero, and the reason is measured rather than assumed. A restore is
/// bit-exact on some caches and not on others:
///
/// | cache | restore-only delta |
/// |---|---|
/// | LFM2.5, unified attention + recurrent | **0.000000** |
/// | Gemma 4 E4B, iSWA | **0.021921** |
///
/// `a_restore_with_nothing_in_between_is_the_same_state` isolates that — no
/// generation between the snapshot and the restore — so the difference is the
/// restore's own doing, and cell placement is what it can be: cells landing in
/// different physical slots reorder attention's floating-point accumulation.
/// The *content* is intact, which is what the identical 16-token continuation
/// says.
///
/// So the continuation is the hard bar and this is the sensitivity check behind
/// it. `0.1` still catches real corruption by a wide margin: the recurrent-only
/// shortcut that silently drops attention context measures **4.83**.
const LOGIT_TOLERANCE: f32 = 0.1;

/// The default: the hybrid model this question was first asked about — short-conv
/// + GQA MoE, the family llama.cpp will not partially roll back.
const DEFAULT_MODEL: &str = "LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf";

/// What to load and how, from the environment. See the module doc for the
/// variables; the defaults reproduce the original LFM2 measurements.
struct ModelUnderTest {
    path: PathBuf,
    /// The spec as written, for the header line — a full snapshot path says
    /// nothing about which model it is.
    label: String,
    gpu_layers: u32,
    cpu_moe: bool,
    n_ctx: u32,
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Resolve `ORG/REPO/path/to/file.gguf` inside the local HuggingFace cache.
///
/// Cache-only on purpose: these files are tens of gigabytes and a test that
/// silently starts a download is a test nobody runs twice. A model that is not
/// here makes the tests skip.
fn hf_cached(spec: &str) -> Option<PathBuf> {
    let (org, rest) = spec.split_once('/')?;
    let (repo, file) = rest.split_once('/')?;
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(file))
        .find(|p| p.exists())
}

fn model_under_test() -> Option<ModelUnderTest> {
    let spec =
        std::env::var("GALLIUM_KV_SPIKE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let path = if spec.starts_with('/') {
        let p = PathBuf::from(&spec);
        p.exists().then_some(p)
    } else {
        hf_cached(&spec)
    }?;
    Some(ModelUnderTest {
        path,
        label: spec,
        gpu_layers: env_u32("GALLIUM_KV_SPIKE_GPU_LAYERS", 999),
        cpu_moe: std::env::var("GALLIUM_KV_SPIKE_CPU_MOE").is_ok_and(|v| v != "0"),
        n_ctx: env_u32("GALLIUM_KV_SPIKE_CTX", 8192),
    })
}

/// The model, plus the header line saying what was measured — every number
/// below is meaningless without it.
fn load(backend: &LlamaBackend, m: &ModelUnderTest) -> LlamaModel {
    eprintln!(
        "\nmodel: {} (gpu_layers={}, cpu_moe={}, n_ctx={})",
        m.label, m.gpu_layers, m.cpu_moe, m.n_ctx
    );
    let params = LlamaModelParams::default().with_n_gpu_layers(m.gpu_layers);
    // Pinned before `add_cpu_moe_override`, which stores a pointer into the
    // params' own buffer — the same dance `llm_local::new` documents.
    let mut params = std::pin::pin!(params);
    if m.cpu_moe {
        params.as_mut().add_cpu_moe_override();
    }
    LlamaModel::load_from_file(backend, &m.path, &params).expect("model")
}

/// Skip with a reason, or hand back the model and its settings.
macro_rules! model_or_skip {
    ($backend:expr) => {
        match model_under_test() {
            Some(m) => {
                let model = load($backend, &m);
                (m, model)
            }
            None => {
                eprintln!("SKIP: the model under test is not in this machine's HuggingFace cache");
                return;
            }
        }
    };
}

/// Evaluate `tokens` at `start`, asking for logits on the last one — the same
/// shape `llm_local::feed` uses, so the numbers here are comparable to the
/// `evaluated N` the provider logs.
fn feed(ctx: &mut LlamaContext, tokens: &[LlamaToken], start: i32) -> i32 {
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    let last = tokens.len() - 1;
    for (offset, token) in tokens.iter().copied().enumerate() {
        batch
            .add(token, start + offset as i32, &[0], offset == last)
            .expect("batch add");
    }
    ctx.decode(&mut batch).expect("decode");
    // The logits row is addressed by its index *within this batch*, not by a
    // position in the sequence.
    last as i32
}

/// A greedy continuation of `n` tokens, and the logits that produced the first
/// of them.
///
/// The equivalence check has to be *sensitive*, and a single argmax is not: an
/// earlier version of this file compared one token and passed against a state
/// that was demonstrably wrong — argmax over a repetitive prompt survives a
/// sizeable shift in the distribution. A continuation compounds any divergence
/// (each wrong token feeds the next), and the logit vector catches a shift too
/// small to change even that.
fn continuation(
    ctx: &mut LlamaContext,
    first_row: i32,
    start_pos: i32,
    n: usize,
) -> (Vec<f32>, Vec<LlamaToken>) {
    let logits = ctx.get_logits_ith(first_row).to_vec();
    let mut out = Vec::with_capacity(n);
    let mut row = first_row;
    let mut pos = start_pos;
    for _ in 0..n {
        let token = argmax(ctx, row);
        out.push(token);
        row = feed(ctx, &[token], pos);
        pos += 1;
    }
    (logits, out)
}

/// Largest absolute difference between two logit vectors.
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vocabulary size changed between contexts");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Greedy pick from the logits of batch row `row`.
fn argmax(ctx: &LlamaContext, row: i32) -> LlamaToken {
    let logits = ctx.get_logits_ith(row);
    let mut best = 0usize;
    for (i, l) in logits.iter().enumerate() {
        if *l > logits[best] {
            best = i;
        }
    }
    LlamaToken(best as i32)
}

#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn a_sequence_snapshot_restores_an_equivalent_state_and_costs_less_than_a_prefill() {
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    // A prompt in the size range a ReAct iteration actually reaches (the
    // `refactoring` turn's first prompt was 1598 tokens).
    let paragraph = "The counter type keeps a private value and exposes Increment and Value. \
                     Refactoring it means replacing the package-level variable with a struct, \
                     moving the mutation behind a pointer receiver, and leaving main printing 3. ";
    let prompt = paragraph.repeat(40);
    let suffix = "Now write the file and explain what changed in one sentence.";

    let n_ctx = settings.n_ctx;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .expect("tokenize prompt");
    let suffix_tokens = model
        .str_to_token(suffix, AddBos::Never)
        .expect("tokenize suffix");
    eprintln!(
        "prompt {} tokens, suffix {} tokens",
        prompt_tokens.len(),
        suffix_tokens.len()
    );

    // ---- control: one context that never snapshots, fed prompt + suffix ----
    let mut control = model
        .new_context(&backend, ctx_params.clone())
        .expect("control context");
    let t = Instant::now();
    feed(&mut control, &prompt_tokens, 0);
    let prefill = t.elapsed();
    let row = feed(&mut control, &suffix_tokens, prompt_tokens.len() as i32);
    let (expected_logits, expected) = continuation(
        &mut control,
        row,
        (prompt_tokens.len() + suffix_tokens.len()) as i32,
        16,
    );
    drop(control);

    // ---- snapshot path ----
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .expect("snapshot context");
    let row = feed(&mut ctx, &prompt_tokens, 0);

    // The *first* `state_seq_get` in a context costs ~390 ms regardless of what
    // it copies — see `where_the_first_get_spends_its_time`, which pins that as a
    // one-time per-context setup rather than a per-snapshot or per-byte cost. So
    // warm it up and time a representative one, or this reports the setup.
    let _warmup = ctx
        .state_seq_get(0, LlamaStateSeqFlags::empty())
        .expect("warm-up get");
    let t = Instant::now();
    let snapshot = ctx
        .state_seq_get(0, LlamaStateSeqFlags::empty())
        .expect("state_seq_get");
    let get = t.elapsed();

    // Dirty the state the way a turn does: generate past the snapshot point.
    // 120 tokens is the short end of what this model emits before a tool call.
    let mut pos = prompt_tokens.len() as i32;
    let mut next = argmax(&ctx, row);
    for _ in 0..120 {
        let row = feed(&mut ctx, &[next], pos);
        pos += 1;
        next = argmax(&ctx, row);
    }

    let t = Instant::now();
    ctx.state_seq_set(&snapshot, 0).expect("state_seq_set");
    let set = t.elapsed();

    let t = Instant::now();
    let row = feed(&mut ctx, &suffix_tokens, prompt_tokens.len() as i32);
    let suffix_eval = t.elapsed();
    let (restored_logits, restored) = continuation(
        &mut ctx,
        row,
        (prompt_tokens.len() + suffix_tokens.len()) as i32,
        16,
    );

    eprintln!(
        "\nsnapshot {:.1} MiB | get {:?} | set {:?}\n\
         prefill {:?} ({} tokens) | suffix after restore {:?} ({} tokens)\n\
         restore+suffix is {:.1}x the prefill's cost",
        snapshot.byte_len() as f64 / (1024.0 * 1024.0),
        get,
        set,
        prefill,
        prompt_tokens.len(),
        suffix_eval,
        suffix_tokens.len(),
        (set + suffix_eval).as_secs_f64() / prefill.as_secs_f64(),
    );

    let delta = max_delta(&restored_logits, &expected_logits);
    eprintln!("max logit delta {delta:.6}");
    assert_eq!(
        restored, expected,
        "a restored state continued differently than a fresh context fed \
         prompt + suffix — the snapshot is not equivalent, which is the failure \
         a cache must never have"
    );
    assert!(
        delta < LOGIT_TOLERANCE,
        "continuations agree but the logits differ by {delta} — the state is \
         close, not equal, and the gap will show on a harder prompt"
    );
}

/// What each flag set costs, which is what decides where a tier's boundary sits:
/// `empty` is the whole sequence in host memory (the warm tier), `ON_DEVICE`
/// keeps the copy in VRAM (a hot-tier checkpoint), and `PARTIAL_ONLY` is the
/// recurrent/SWA part alone — the part llama.cpp cannot roll back, and therefore
/// the only part a hybrid model actually needs snapshotted if the attention KV
/// can be trimmed the ordinary way.
#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn what_each_state_flag_costs() {
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    let paragraph = "The counter type keeps a private value and exposes Increment and Value. ";
    let n_ctx = settings.n_ctx;

    for repeats in [40usize, 160, 400] {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);
        let mut ctx = model.new_context(&backend, ctx_params).expect("context");
        let tokens = model
            .str_to_token(&paragraph.repeat(repeats), AddBos::Always)
            .expect("tokenize");
        if tokens.len() as u32 >= n_ctx {
            continue;
        }
        let t = Instant::now();
        feed(&mut ctx, &tokens, 0);
        let prefill = t.elapsed();

        eprintln!("\n=== {} tokens (prefill {:?}) ===", tokens.len(), prefill);
        // Warm-up: the first get in a context pays a one-time ~390 ms setup.
        let _ = ctx.state_seq_get(0, LlamaStateSeqFlags::empty());
        for (name, flags) in [
            ("empty (host)", LlamaStateSeqFlags::empty()),
            ("PARTIAL_ONLY", LlamaStateSeqFlags::PARTIAL_ONLY),
            ("ON_DEVICE", LlamaStateSeqFlags::ON_DEVICE),
        ] {
            let t = Instant::now();
            match ctx.state_seq_get(0, flags) {
                Ok(state) => {
                    let get = t.elapsed();
                    let t = Instant::now();
                    let set = match ctx.state_seq_set(&state, 0) {
                        Ok(()) => format!("{:?}", t.elapsed()),
                        Err(e) => format!("set failed: {e}"),
                    };
                    eprintln!(
                        "  {name:<14} {:>9.3} MiB | get {:>12?} | set {set}",
                        state.byte_len() as f64 / (1024.0 * 1024.0),
                        get,
                    );
                }
                Err(e) => eprintln!("  {name:<14} get failed: {e}"),
            }
        }
    }
}

/// Why the cheap path is **not** available, pinned rather than assumed.
///
/// The flag comparison shows a recurrent-only snapshot is constant-size
/// (0.28 MiB) and costs tens of microseconds at any context length — so if the
/// attention half could be trimmed the ordinary position-addressed way, a hybrid
/// model's rewind would be essentially free, with no host copy at all.
///
/// It cannot, and the reason is one line of llama.cpp:
///
/// ```text
/// bool llama_memory_hybrid::seq_rm(...) {
///     // Try removing from the recurrent cache first since it may fail. If it
///     // does fail, the cache will not have been mutated.
///     if (!mem_recr->seq_rm(seq_id, p0, p1)) { return false; }
///     return mem_attn->seq_rm(seq_id, p0, p1);
/// }
/// ```
///
/// The recurrent half refuses (that is #172's whole problem), so the attention
/// half is never reached and the generated cells stay. They are not harmless:
/// `apply_ubatch` only evicts cells it physically overwrites, and while the
/// cache has free space the new tokens land elsewhere — leaving the stale cells
/// in sequence 0 at positions the causal mask (`p0 <= p1`) happily keeps.
///
/// This test therefore asserts the *broken* state deliberately, so that the day
/// upstream changes either half of it, it fails and says so. It is also the
/// reason [`continuation`] exists: an earlier version compared a single argmax
/// here, got a match, and concluded the shortcut worked.
///
/// **Only for the default model.** Unlike the equivalence tests, this one pins a
/// specific `llama_memory_hybrid` behaviour and the exact shape of the damage it
/// leaves. Another cache refuses for its own reasons and corrupts differently —
/// DeepSeek-V4's `seq_rm` also returns false — so running this against one would
/// fail in a way that says nothing about that model.
#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn a_recurrent_only_snapshot_is_not_enough_because_the_attention_trim_is_refused() {
    if std::env::var_os("GALLIUM_KV_SPIKE_MODEL").is_some() {
        eprintln!("SKIP: this test pins the hybrid memory's behaviour, not the model under test");
        return;
    }
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    let paragraph = "The counter type keeps a private value and exposes Increment and Value. ";
    let prompt = paragraph.repeat(120);
    let suffix = "Now write the file and explain what changed in one sentence.";
    let n_ctx = settings.n_ctx;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .expect("tokenize prompt");
    let suffix_tokens = model
        .str_to_token(suffix, AddBos::Never)
        .expect("tokenize suffix");
    let n_prompt = prompt_tokens.len() as i32;
    let after_suffix = n_prompt + suffix_tokens.len() as i32;

    let mut control = model
        .new_context(&backend, ctx_params.clone())
        .expect("control context");
    feed(&mut control, &prompt_tokens, 0);
    let row = feed(&mut control, &suffix_tokens, n_prompt);
    let (expected_logits, expected) = continuation(&mut control, row, after_suffix, 16);
    drop(control);

    let mut ctx = model.new_context(&backend, ctx_params).expect("context");
    let row = feed(&mut ctx, &prompt_tokens, 0);
    let recurrent = ctx
        .state_seq_get(0, LlamaStateSeqFlags::PARTIAL_ONLY)
        .expect("recurrent snapshot");

    let mut pos = n_prompt;
    let mut next = argmax(&ctx, row);
    for _ in 0..120 {
        let row = feed(&mut ctx, &[next], pos);
        pos += 1;
        next = argmax(&ctx, row);
    }

    let trimmed = ctx
        .clear_kv_cache_seq(Some(0), Some(n_prompt as u32), None)
        .expect("seq_rm arguments");
    ctx.state_seq_set(&recurrent, 0).expect("recurrent restore");

    let row = feed(&mut ctx, &suffix_tokens, n_prompt);
    let (restored_logits, restored) = continuation(&mut ctx, row, after_suffix, 16);
    let delta = max_delta(&restored_logits, &expected_logits);

    eprintln!(
        "\nrecurrent-only snapshot {:.3} MiB | attention trim accepted: {trimmed} | \
         max logit delta {delta:.6}\n\
         rewound continuation {:?}\n\
         control  continuation {:?}",
        recurrent.byte_len() as f64 / (1024.0 * 1024.0),
        restored.iter().map(|t| t.0).collect::<Vec<_>>(),
        expected.iter().map(|t| t.0).collect::<Vec<_>>(),
    );

    assert!(
        !trimmed,
        "llama.cpp accepted a position-addressed trim on a hybrid sequence — the \
         constant-cost rewind (recurrent snapshot + attention trim) is available \
         now and is worth taking; see this test's docs"
    );
    assert_ne!(
        restored, expected,
        "the rewind produced the right continuation even though the trim was \
         refused — recheck why before relying on either fact"
    );
    assert!(
        delta > 1.0,
        "logits differ by only {delta} after a refused trim; the corruption this \
         test pins may have changed shape"
    );
}

/// The hot tier's mechanism: a checkpoint whose data stays in VRAM. The handle
/// that comes back is kilobytes, so this is cheap enough to take on *every*
/// iteration — which is what would let a live session rewind without ever
/// touching host memory.
///
/// Same equivalence bar as the full snapshot: a 16-token continuation and the
/// logit vector, because one argmax is not sensitive enough to catch a state
/// that is merely close (this file learned that the hard way — see
/// [`continuation`]).
#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn an_on_device_checkpoint_restores_an_equivalent_state() {
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    let paragraph = "The counter type keeps a private value and exposes Increment and Value. ";
    let prompt = paragraph.repeat(120);
    let suffix = "Now write the file and explain what changed in one sentence.";
    let n_ctx = settings.n_ctx;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .expect("tokenize prompt");
    let suffix_tokens = model
        .str_to_token(suffix, AddBos::Never)
        .expect("tokenize suffix");
    let n_prompt = prompt_tokens.len() as i32;
    let after_suffix = n_prompt + suffix_tokens.len() as i32;

    let mut control = model
        .new_context(&backend, ctx_params.clone())
        .expect("control context");
    feed(&mut control, &prompt_tokens, 0);
    let row = feed(&mut control, &suffix_tokens, n_prompt);
    let (expected_logits, expected) = continuation(&mut control, row, after_suffix, 16);
    drop(control);

    let mut ctx = model.new_context(&backend, ctx_params).expect("context");
    let row = feed(&mut ctx, &prompt_tokens, 0);

    // Warmed up for the same reason as the host snapshot above.
    let _warmup = ctx
        .state_seq_get(0, LlamaStateSeqFlags::ON_DEVICE)
        .expect("warm-up get");
    let t = Instant::now();
    let checkpoint = ctx
        .state_seq_get(0, LlamaStateSeqFlags::ON_DEVICE)
        .expect("on-device checkpoint");
    let get = t.elapsed();

    // Generate past the checkpoint, as a turn does.
    let mut pos = n_prompt;
    let mut next = argmax(&ctx, row);
    for _ in 0..120 {
        let row = feed(&mut ctx, &[next], pos);
        pos += 1;
        next = argmax(&ctx, row);
    }

    let t = Instant::now();
    ctx.state_seq_set(&checkpoint, 0)
        .expect("on-device restore");
    let set = t.elapsed();

    let row = feed(&mut ctx, &suffix_tokens, n_prompt);
    let (restored_logits, restored) = continuation(&mut ctx, row, after_suffix, 16);
    let delta = max_delta(&restored_logits, &expected_logits);

    eprintln!(
        "\non-device handle {:.3} KiB | get {get:?} | set {set:?} | max logit delta {delta:.6}",
        checkpoint.byte_len() as f64 / 1024.0,
    );

    assert_eq!(
        restored, expected,
        "an on-device checkpoint restored a state that continues differently \
         from a fresh context — the hot tier cannot be built on it"
    );
    assert!(
        delta < LOGIT_TOLERANCE,
        "logits differ by {delta} after an on-device restore"
    );
}

/// Where the ~380 ms in every `get` above actually goes.
///
/// It is the same figure for a 21 MiB host snapshot and for a 21 KiB on-device
/// handle, which rules out "copying the data". The candidates are a one-time
/// cost per context and a cost paid once per decode (a device sync), and they
/// have very different consequences: the first is amortized away by a long
/// session, the second is paid on every ReAct iteration.
#[test]
#[ignore = "loads a 4.9GB GGUF"]
fn where_the_first_get_spends_its_time() {
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    let paragraph = "The counter type keeps a private value and exposes Increment and Value. ";
    let n_ctx = settings.n_ctx;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model.new_context(&backend, ctx_params).expect("context");
    let tokens = model
        .str_to_token(&paragraph.repeat(120), AddBos::Always)
        .expect("tokenize");
    let row = feed(&mut ctx, &tokens, 0);

    let time = |ctx: &LlamaContext, flags| {
        let t = Instant::now();
        let state = ctx.state_seq_get(0, flags).expect("get");
        (t.elapsed(), state.byte_len())
    };

    let (first, n1) = time(&ctx, LlamaStateSeqFlags::ON_DEVICE);
    let (second, _) = time(&ctx, LlamaStateSeqFlags::ON_DEVICE);
    let (host, n2) = time(&ctx, LlamaStateSeqFlags::empty());
    let (host_again, _) = time(&ctx, LlamaStateSeqFlags::empty());

    // Now decode one token and ask again: if the cost is a per-decode sync, it
    // comes back.
    let next = argmax(&ctx, row);
    feed(&mut ctx, &[next], tokens.len() as i32);
    let (after_decode, _) = time(&ctx, LlamaStateSeqFlags::ON_DEVICE);
    let (after_decode_2, _) = time(&ctx, LlamaStateSeqFlags::ON_DEVICE);

    eprintln!(
        "\n{} tokens\n\
         on-device get: first {first:?}, again {second:?} ({n1} B)\n\
         host get:      first {host:?}, again {host_again:?} ({n2} B)\n\
         after one decode: {after_decode:?}, then {after_decode_2:?}",
        tokens.len(),
    );
}

/// Is a restore *itself* exact, or does the cycle around it perturb the state?
///
/// LFM2 restores to `max logit delta 0.000000`; Gemma 4's iSWA cache restores to
/// ~0.02 with an identical 16-token continuation. Two explanations fit that, and
/// they have different consequences: cells landing in different physical slots
/// after a restore reorders attention's floating-point accumulation (harmless,
/// and unavoidable), or the restore drops something (not harmless, and the fact
/// that a continuation survives it would only mean this prompt is forgiving).
///
/// This separates them by taking the generation out of the loop entirely:
/// snapshot, restore immediately, and compare against the same context that was
/// never restored. Anything non-zero here is the restore's own doing.
#[test]
#[ignore = "loads a multi-GB GGUF"]
fn a_restore_with_nothing_in_between_is_the_same_state() {
    let backend = LlamaBackend::init().expect("backend");
    let (settings, model) = model_or_skip!(&backend);

    let paragraph = "The counter type keeps a private value and exposes Increment and Value. ";
    let prompt = paragraph.repeat(120);
    let suffix = "Now write the file and explain what changed in one sentence.";
    let n_ctx = settings.n_ctx;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);

    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .expect("tokenize prompt");
    let suffix_tokens = model
        .str_to_token(suffix, AddBos::Never)
        .expect("tokenize suffix");
    let n_prompt = prompt_tokens.len() as i32;

    // Baseline: prompt, then suffix, no snapshot anywhere near it.
    let mut plain = model
        .new_context(&backend, ctx_params.clone())
        .expect("plain context");
    feed(&mut plain, &prompt_tokens, 0);
    let row = feed(&mut plain, &suffix_tokens, n_prompt);
    let plain_logits = plain.get_logits_ith(row).to_vec();
    drop(plain);

    // Same thing, with a snapshot taken and put straight back before the suffix.
    let mut ctx = model.new_context(&backend, ctx_params).expect("context");
    feed(&mut ctx, &prompt_tokens, 0);
    let snapshot = ctx
        .state_seq_get(0, LlamaStateSeqFlags::empty())
        .expect("state_seq_get");
    ctx.state_seq_set(&snapshot, 0).expect("state_seq_set");
    let row = feed(&mut ctx, &suffix_tokens, n_prompt);
    let restored_logits = ctx.get_logits_ith(row).to_vec();

    let delta = max_delta(&restored_logits, &plain_logits);
    eprintln!("\nrestore-only max logit delta {delta:.6} ({n_prompt} tokens)");
    assert!(
        delta < LOGIT_TOLERANCE,
        "a restore with no generation in between moved the logits by {delta} — \
         more than cell placement explains"
    );
}
