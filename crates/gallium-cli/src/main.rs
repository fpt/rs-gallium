use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use gallium_core::{generate, load_gguf, CausalLM, SamplingParams};
use gallium_models::loader;

#[derive(Debug, Clone, clap::ValueEnum)]
enum ModelArch {
    GptOss,
    Qwen35,
    Gemma4,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum ModelFormat {
    /// SafeTensors format (HuggingFace)
    Safetensors,
    /// GGUF format (llama.cpp / ollama compatible)
    Gguf,
}

#[derive(Parser, Debug)]
#[command(name = "gallium", about = "Simple LLM inference")]
struct Args {
    /// Model architecture.
    #[arg(long)]
    arch: ModelArch,

    /// Model format.
    #[arg(long, default_value = "safetensors")]
    format: ModelFormat,

    /// Path to model directory (safetensors) or GGUF file.
    /// Required unless --hf-repo is specified.
    #[arg(long, required_unless_present = "hf_repo")]
    model: Option<PathBuf>,

    /// HuggingFace repo ID to download from (e.g. "Qwen/Qwen3-8B").
    /// Files are cached in ~/.cache/huggingface/hub/ and reused on subsequent runs.
    #[arg(long)]
    hf_repo: Option<String>,

    /// File to download from --hf-repo. Required for GGUF format (e.g. "model-q4_k_m.gguf").
    /// Ignored for safetensors (all shards are downloaded automatically).
    #[arg(long)]
    hf_file: Option<String>,

    /// HuggingFace repo to download the tokenizer from (defaults to --hf-repo).
    /// Useful when the GGUF repo does not include tokenizer.json (e.g. --hf-tokenizer-repo unsloth/gpt-oss-20b).
    #[arg(long)]
    hf_tokenizer_repo: Option<String>,

    /// Prompt text.
    #[arg(long)]
    prompt: String,

    /// Maximum new tokens to generate.
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Top-k sampling.
    #[arg(long)]
    top_k: Option<usize>,

    /// Top-p (nucleus) sampling.
    #[arg(long)]
    top_p: Option<f32>,

    /// Repetition penalty (1.0 = none, >1.0 penalizes repeated tokens).
    #[arg(long)]
    repetition_penalty: Option<f32>,

    /// Data type for safetensors (f32, f16, bf16). Ignored for GGUF.
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Apply the arch-appropriate chat template to the prompt before generation.
    /// GPT-OSS: Harmony system/user/assistant turns.
    /// Gemma 4:  <start_of_turn>user/model.
    /// Qwen 3.5: <|im_start|>user/assistant (ChatML).
    #[arg(long)]
    chat: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let device = Device::Cpu; // TODO: auto-detect CUDA/Metal

    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        ..Default::default()
    };

    // Resolve local model path, downloading from HF if needed.
    let model_path = match &args.hf_repo {
        Some(repo_id) => download_from_hub(
            repo_id,
            &args.format,
            args.hf_file.as_deref(),
            args.hf_tokenizer_repo.as_deref(),
        )?,
        None => args.model.clone().expect("--model is required when --hf-repo is not set"),
    };

    let (mut model, tokenizer): (Box<dyn CausalLM>, tokenizers::Tokenizer) = match args.format {
        ModelFormat::Gguf => load_gguf_model(&args.arch, &model_path, &device)?,
        ModelFormat::Safetensors => load_safetensors_model(&args.arch, &model_path, &args.dtype, &device)?,
    };

    // Apply chat template if requested
    let prompt_text = if args.chat {
        match args.arch {
            ModelArch::GptOss  => format_gpt_oss_chat(&args.prompt),
            ModelArch::Gemma4  => format_gemma4_chat(&args.prompt),
            ModelArch::Qwen35  => format_qwen35_chat(&args.prompt),
        }
    } else {
        args.prompt.clone()
    };

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    // EOS tokens — collect all common stop tokens across architectures.
    let eos_tokens: Vec<u32> = tokenizer
        .get_added_vocabulary()
        .get_vocab()
        .iter()
        .filter(|(k, _)| {
            k.contains("eos")
                || k.contains("<|end")      // <|end|>, <|endoftext|>
                || k.contains("</s>")
                || k.contains("<end_of_turn>") // Gemma
                || k.contains("<|im_end|>")    // Qwen ChatML
                || *k == "<|call|>"            // Harmony tool call
        })
        .map(|(_, &v)| v)
        .collect();

    // Generate
    if args.chat {
        // In chat mode, don't echo the full (templated) prompt.
    } else {
        print!("{}", args.prompt);
    }
    let mut generated_ids: Vec<u32> = Vec::new();
    let _generated = generate(
        model.as_mut(),
        &prompt_tokens,
        &params,
        args.max_tokens,
        &eos_tokens,
        |token_id| {
            generated_ids.push(token_id);
            // Decode the full sequence so far so subword merging works correctly.
            if let Ok(text) = tokenizer.decode(&generated_ids, true) {
                // Print only the newly added characters.
                let prev = if generated_ids.len() > 1 {
                    tokenizer.decode(&generated_ids[..generated_ids.len() - 1], true)
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                print!("{}", &text[prev.len()..]);
                let _ = std::io::stdout().flush();
            }
        },
    )?;
    println!();

    Ok(())
}

/// Format a prompt using the GPT-OSS chat template.
/// Produces the exact token sequence the model was trained with for single-turn chat.
fn format_gpt_oss_chat(user_prompt: &str) -> String {
    // Compute current date as YYYY-MM-DD from system clock.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_since_epoch = secs / 86400;
    let date = epoch_days_to_ymd(days_since_epoch);

    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Knowledge cutoff: 2024-06\n\
         Current date: {date}\n\
         \n\
         Reasoning: medium\n\
         \n\
         # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\
         <|start|>user<|message|>{user_prompt}<|end|>\
         <|start|>assistant\n"
    )
}

/// Format a prompt using the Gemma 4 chat template (`<start_of_turn>user/model`).
fn format_gemma4_chat(user_prompt: &str) -> String {
    format!(
        "<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
}

/// Format a prompt using the Qwen 3.5 ChatML template (`<|im_start|>role`).
fn format_qwen35_chat(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Convert days since Unix epoch (1970-01-01) to a YYYY-MM-DD string.
fn epoch_days_to_ymd(mut days: u64) -> String {
    let mut year = 1970u32;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    format!("{year:04}-{month:02}-{:02}", days + 1)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Download model files from HuggingFace Hub into the local cache.
/// Returns the local path to use as the model path (directory for safetensors, file for GGUF).
fn download_from_hub(
    repo_id: &str,
    format: &ModelFormat,
    hf_file: Option<&str>,
    hf_tokenizer_repo: Option<&str>,
) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;

    eprintln!("Fetching from HuggingFace: {repo_id}");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    match format {
        ModelFormat::Safetensors => {
            // List all repo files and download every safetensors shard.
            let info = repo.info()?;
            let shards: Vec<_> = info
                .siblings
                .iter()
                .map(|s| s.rfilename.clone())
                .filter(|name| name.ends_with(".safetensors"))
                .collect();

            if shards.is_empty() {
                anyhow::bail!("no .safetensors files found in repo {repo_id}");
            }

            eprintln!("  Downloading config.json and tokenizer.json");
            repo.get("config.json")?;
            let tok_repo = hf_tokenizer_repo.unwrap_or(repo_id);
            api.model(tok_repo.to_string()).get("tokenizer.json")?;

            for shard in &shards {
                eprintln!("  Downloading {shard}");
                repo.get(shard)?;
            }

            // All files land in the same snapshot directory; return it.
            let config_local = repo.get("config.json")?;
            Ok(config_local.parent().unwrap().to_path_buf())
        }
        ModelFormat::Gguf => {
            let filename = hf_file.ok_or_else(|| {
                anyhow::anyhow!("--hf-file is required when using --hf-repo with GGUF format\nExample: --hf-file model-q4_k_m.gguf")
            })?;

            let tok_repo = hf_tokenizer_repo.unwrap_or(repo_id);
            eprintln!("  Downloading tokenizer.json from {tok_repo}");
            let tok_local = api.model(tok_repo.to_string()).get("tokenizer.json")?;

            eprintln!("  Downloading {filename}");
            let gguf_local = repo.get(filename)?;

            // The GGUF loader looks for tokenizer.json next to the model file.
            // If they landed in different cache dirs, symlink tokenizer into the GGUF dir.
            let gguf_dir = gguf_local.parent().unwrap();
            let tok_dest = gguf_dir.join("tokenizer.json");
            if !tok_dest.exists() {
                std::fs::copy(&tok_local, &tok_dest)?;
            }

            Ok(gguf_local)
        }
    }
}

fn load_gguf_model(
    arch: &ModelArch,
    path: &PathBuf,
    device: &Device,
) -> Result<(Box<dyn CausalLM>, tokenizers::Tokenizer)> {
    eprintln!("Loading GGUF model from {:?}...", path);

    let (metadata, vb) = load_gguf(path, device)?;

    // Try to load tokenizer from the same directory
    let dir = path.parent().unwrap_or(std::path::Path::new("."));
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!(
            "failed to load tokenizer from {:?}: {e}\n\
             Hint: download tokenizer.json from the HuggingFace model page",
            tokenizer_path
        ))?;

    let model: Box<dyn CausalLM> = match arch {
        ModelArch::GptOss => {
            Box::new(gallium_models::gpt_oss_q::GptOssQ::load(&metadata, &vb, device)?)
        }
        ModelArch::Qwen35 => {
            Box::new(gallium_models::qwen35_q::Qwen35Q::load(&metadata, &vb, device)?)
        }
        ModelArch::Gemma4 => {
            Box::new(gallium_models::gemma4_q::Gemma4Q::load(&metadata, &vb, device)?)
        }
    };

    eprintln!("Model loaded.");
    Ok((model, tokenizer))
}

fn load_safetensors_model(
    arch: &ModelArch,
    model_dir: &PathBuf,
    dtype_str: &str,
    device: &Device,
) -> Result<(Box<dyn CausalLM>, tokenizers::Tokenizer)> {
    let dtype = match dtype_str {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        other => anyhow::bail!("unsupported dtype: {other}"),
    };

    let safetensors: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "safetensors").unwrap_or(false))
        .collect();

    if safetensors.is_empty() {
        anyhow::bail!("no .safetensors files found in {:?}", model_dir);
    }

    let config_path = model_dir.join("config.json");
    let vb = loader::load_safetensors(&safetensors, dtype, device)?;

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let model: Box<dyn CausalLM> = match arch {
        ModelArch::GptOss => {
            let cfg: gallium_models::gpt_oss::GptOssConfig = loader::load_config(&config_path)?;
            Box::new(gallium_models::gpt_oss::GptOss::load(&cfg, vb, &safetensors, device)?)
        }
        ModelArch::Qwen35 => {
            // Qwen3.5 wraps text config under "text_config" (multimodal model).
            let full: serde_json::Value = loader::load_config(&config_path)?;
            let text = full.get("text_config").unwrap_or(&full);
            let cfg: gallium_models::qwen35::Qwen35Config = serde_json::from_value(text.clone())
                .map_err(|e| anyhow::anyhow!("Qwen35 config error: {e}"))?;
            Box::new(gallium_models::qwen35::Qwen35::load(&cfg, vb, device)?)
        }
        ModelArch::Gemma4 => {
            // Gemma 4 config.json wraps text config under "text_config".
            let full: serde_json::Value = loader::load_config(&config_path)?;
            let text = full.get("text_config").unwrap_or(&full);
            let cfg: gallium_models::gemma4::Gemma4Config = serde_json::from_value(text.clone())
                .map_err(|e| anyhow::anyhow!("Gemma4 config error: {e}"))?;
            Box::new(gallium_models::gemma4::Gemma4::load(&cfg, vb, device)?)
        }
    };

    Ok((model, tokenizer))
}
