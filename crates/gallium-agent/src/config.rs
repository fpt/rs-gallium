//! TOML config file support for the `gallium` CLI (`--config <path>`).
//!
//! Mirrors the schema the Swift/C# frontends used to parse (`configs/*.toml`):
//! an `[llm]` block, an `[agent]` block, and a `[[mcpServers]]` array. Voice-only
//! sections (`tts`, `stt`, `ambient`) are ignored — this is a headless CLI.
//!
//! Precedence for every field is: environment variable > config file > built-in
//! default, so a config file sets the baseline and env vars still override it at
//! runtime (matching the old frontend behavior for `INFERENCE_ENGINE` etc.).

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmConfig {
    /// Note the uppercase key: `baseURL`, not the camelCase default `baseUrl`.
    #[serde(rename = "baseURL")]
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Local GGUF path, or an `hf:ORG/REPO[@REV]/file.gguf` spec the model
    /// downloader resolves into the HF cache.
    pub model_path: Option<String>,
    pub reasoning_effort: Option<String>,
    /// Local backend for `model_path`: "llamacpp" (default) or "gallium".
    pub inference_engine: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentConfig {
    /// Path to a system-prompt file, resolved relative to the config file's dir.
    pub system_prompt_path: Option<String>,
    /// Max ReAct iterations per turn (the config's `maxTurns`).
    pub max_turns: Option<u32>,
    pub language: Option<String>,
    /// SKILL.md dirs, resolved relative to the config file's dir.
    #[serde(default)]
    pub skill_paths: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    /// stdio transport: the binary to spawn. Absent for an HTTP (`url`) server.
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Streamable HTTP transport. Absent means stdio.
    pub url: Option<String>,
}

impl FileConfig {
    /// Parse a TOML config file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read config '{}': {}", path.display(), e))?;
        toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("failed to parse config '{}': {}", path.display(), e))
    }
}

/// Resolve a config-relative path against the config file's directory. Absolute
/// paths pass through unchanged.
pub fn resolve_relative(config_dir: Option<&Path>, p: &str) -> PathBuf {
    let path = Path::new(p);
    match config_dir {
        Some(dir) if path.is_relative() => dir.join(path),
        _ => path.to_path_buf(),
    }
}

/// Extract `--config <path>` / `-c <path>` / `--config=<path>` from argv,
/// returning the path (if any). Other args are left for the caller.
pub fn parse_config_flag(args: &[String]) -> Option<String> {
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(val) = arg.strip_prefix("--config=") {
            return Some(val.to_string());
        }
        if arg == "--config" || arg == "-c" {
            return it.next().cloned();
        }
    }
    None
}
