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
    /// Model context window in tokens. Drives history compaction — set it to
    /// what the model actually has, or a long session compacts too late.
    pub context_window: Option<u32>,
    /// Local GGUF path, or an `hf:ORG/REPO[@REV]/file.gguf` spec the model
    /// downloader resolves into the HF cache.
    pub model_path: Option<String>,
    pub reasoning_effort: Option<String>,
    /// Local backend for `model_path`: "llamacpp" (default) or "gallium".
    pub inference_engine: Option<String>,
    /// Where the native candle backend finds its `tokenizer.json`, for the
    /// common case of a GGUF repo that ships none. Ignored by the llama.cpp
    /// backend, which uses the tokenizer inside the GGUF.
    ///
    /// Takes the same two shapes as `model_path`: a local path, or a
    /// HuggingFace repo to fetch it from. See [`resolve_tokenizer_path`] for
    /// how the two are told apart.
    pub tokenizer_path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentConfig {
    /// Path to a system-prompt file, resolved relative to the config file's dir.
    pub system_prompt_path: Option<String>,
    /// Max ReAct iterations per turn (the config's `maxTurns`).
    pub max_turns: Option<u32>,
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

/// The config to load when `--config` is absent: `~/.config/gallium/config.toml`,
/// or `None` when the user has written none.
///
/// One location, and it is the one global skills already load from
/// (`~/.config/gallium/skills`, see [`crate::skill::load_skills`]) — gallium's
/// own things belong in one directory rather than split across two.
///
/// Every relative path *inside* that file still resolves against the file's own
/// directory, so `systemPromptPath = "system-prompt.md"` next to the config
/// means the same thing from every working directory — which is the point of
/// having a user-level config at all.
pub fn default_config_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let path = Path::new(&home)
        .join(".config")
        .join("gallium")
        .join("config.toml");
    path.is_file().then_some(path)
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

/// Resolve a config file's `tokenizerPath`, which is either a place on disk or
/// a HuggingFace repo to fetch `tokenizer.json` from.
///
/// Telling those apart needs a rule, because a bare repo id (`unsloth/gemma-4`)
/// and a relative path look identical:
///
/// 1. An `hf:` prefix means a repo, and is stripped. Say this when you mean it.
/// 2. Otherwise, if the path exists — resolved against the config's directory,
///    like every other path in the file — it is that file or directory.
/// 3. Otherwise it is a repo id, which is what `GALLIUM_TOKENIZER_REPO` has
///    always meant, so a value moved from the env var keeps working.
///
/// Rule 2 is the only one that touches the filesystem, and it is what makes
/// `tokenizerPath = "tokenizer.json"` next to the config do the obvious thing.
pub fn resolve_tokenizer_path(config_dir: Option<&Path>, spec: String) -> String {
    if let Some(repo) = spec.strip_prefix("hf:") {
        return repo.to_string();
    }
    let resolved = resolve_relative(config_dir, &spec);
    if resolved.exists() {
        return resolved.to_string_lossy().into_owned();
    }
    spec
}

/// Resolve a config file's `modelPath`. An `hf:ORG/REPO/file.gguf` download spec
/// is returned untouched (it only *looks* relative); a filesystem path is
/// resolved relative to the config file's directory like the other paths.
pub fn resolve_model_path(config_dir: Option<&Path>, spec: String) -> String {
    if spec.starts_with("hf:") {
        return spec;
    }
    resolve_relative(config_dir, &spec)
        .to_string_lossy()
        .into_owned()
}

/// Extract `--config <path>` / `-c <path>` / `--config=<path>` from argv.
/// `Ok(None)` means the flag is absent; `Err` means it was given without a path
/// (a usage error the caller should report rather than silently ignore).
pub fn parse_config_flag(args: &[String]) -> Result<Option<String>, String> {
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(val) = arg.strip_prefix("--config=") {
            return if val.is_empty() {
                Err("--config= requires a path".to_string())
            } else {
                Ok(Some(val.to_string()))
            };
        }
        if arg == "--config" || arg == "-c" {
            return match it.next() {
                Some(val) => Ok(Some(val.clone())),
                None => Err(format!("{} requires a path argument", arg)),
            };
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hf:` is the way to say "repo" out loud, and it survives even when
    /// something of that name happens to exist on disk.
    #[test]
    fn an_hf_prefixed_tokenizer_is_a_repo() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("unsloth")).unwrap();
        std::fs::write(dir.path().join("unsloth/gemma"), "decoy").unwrap();

        assert_eq!(
            resolve_tokenizer_path(Some(dir.path()), "hf:unsloth/gemma".to_string()),
            "unsloth/gemma"
        );
    }

    /// A relative path that exists is that path, made absolute against the
    /// config's directory — the same treatment `modelPath` and `skillPaths` get.
    #[test]
    fn a_tokenizer_path_that_exists_is_resolved_against_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, "{}").unwrap();

        let resolved = resolve_tokenizer_path(Some(dir.path()), "tokenizer.json".to_string());

        assert_eq!(resolved, file.to_string_lossy());
    }

    /// A directory is accepted whole: people point at the model directory more
    /// often than at the file inside it.
    #[test]
    fn a_tokenizer_directory_resolves_too() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("weights");
        std::fs::create_dir(&sub).unwrap();

        let resolved = resolve_tokenizer_path(Some(dir.path()), "weights".to_string());

        assert_eq!(resolved, sub.to_string_lossy());
    }

    /// Nothing of that name on disk, so it is a repo id — which is what
    /// `GALLIUM_TOKENIZER_REPO` has always meant, so a value moved from the env
    /// var into the config keeps working.
    #[test]
    fn a_tokenizer_spec_that_is_not_a_path_stays_a_repo_id() {
        let dir = tempfile::tempdir().unwrap();

        assert_eq!(
            resolve_tokenizer_path(Some(dir.path()), "unsloth/gemma-4-E4B-it".to_string()),
            "unsloth/gemma-4-E4B-it"
        );
    }

    /// The default config search reads `$HOME`, which is process-global, so
    /// these tests take a lock rather than racing each other over it.
    static HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with `$HOME` pointed at `home`, restoring it afterwards.
    fn with_home<T>(home: &Path, f: impl FnOnce() -> T) -> T {
        let _guard = HOME_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var_os("HOME");
        // Safety: single-threaded within the lock, and restored before release.
        unsafe { std::env::set_var("HOME", home) };
        let out = f();
        match saved {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        out
    }

    /// Beside `~/.config/gallium/skills`, which is where the global skills the
    /// same run loads already come from.
    #[test]
    fn the_user_config_is_found_without_a_config_flag() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join(".config").join("gallium");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "").unwrap();

        assert_eq!(
            with_home(home.path(), default_config_path),
            Some(dir.join("config.toml"))
        );
    }

    /// A directory of that name is not a config file, and must not be loaded as
    /// one — `~/.config/gallium/` itself is a directory people already have.
    #[test]
    fn a_directory_named_config_toml_is_not_a_config() {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join(".config/gallium/config.toml")).unwrap();

        assert_eq!(with_home(home.path(), default_config_path), None);
    }

    /// Having written no config at all is the ordinary case, not an error: the
    /// binary falls back to env vars and its built-in defaults.
    #[test]
    fn no_user_config_is_not_an_error() {
        let home = tempfile::tempdir().unwrap();

        assert_eq!(with_home(home.path(), default_config_path), None);
    }

    /// A `systemPromptPath` in the user config points beside the config, not at
    /// whatever directory the agent was started in — the reason a user-level
    /// config is worth having.
    #[test]
    fn a_user_configs_relative_paths_follow_the_config_not_the_cwd() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join(".config").join("gallium");
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(
            resolve_relative(Some(&dir), "system-prompt.md"),
            dir.join("system-prompt.md")
        );
    }

    /// With no config file there is nothing to resolve against, and a bare repo
    /// id must not become a path relative to the cwd.
    #[test]
    fn a_repo_id_survives_having_no_config_directory() {
        assert_eq!(
            resolve_tokenizer_path(None, "unsloth/gemma-4-E4B-it".to_string()),
            "unsloth/gemma-4-E4B-it"
        );
    }
}
