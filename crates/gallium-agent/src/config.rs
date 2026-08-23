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

use gallium_agent::approval::{ApprovalPolicy, ApprovalRule};
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
    /// Nucleus sampling threshold for the llama.cpp backend only — candle has
    /// no top_p sampler stage yet. `None` means llama.cpp's sampler chain
    /// skips the stage entirely (unrestricted), not "1.0" (an explicit no-op
    /// that still runs the stage).
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff for the llama.cpp backend only — candle has no
    /// top_k sampler stage yet. `None` means llama.cpp's sampler chain skips
    /// the stage entirely, not "vocab size" (an explicit no-op that still
    /// runs the stage).
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    /// Model context window in tokens. Drives history compaction — set it to
    /// what the model actually has, or a long session compacts too late.
    pub context_window: Option<u32>,
    /// Local GGUF path, or an `hf:ORG/REPO[@REV]/file.gguf` spec the model
    /// downloader resolves into the HF cache.
    pub model_path: Option<String>,
    /// Multimodal projector (`mmproj-*.gguf`) for the llama.cpp backend: what
    /// turns an image or an audio clip into embeddings the text model can read.
    ///
    /// Takes the same two shapes as `model_path` — a local path, or an
    /// `hf:ORG/REPO[@REV]/file.gguf` spec — and is almost always the latter,
    /// since a GGUF repo publishes the projector beside the model it belongs to.
    ///
    /// Absent means text only. It is deliberately not derived from `model_path`
    /// by guessing at a sibling filename: a projector that silently failed to
    /// match its model produces garbage embeddings rather than an error, and
    /// naming it is one line.
    pub mmproj_path: Option<String>,
    pub reasoning_effort: Option<String>,
    /// Which model profile reads this model's output — the compiled-in name
    /// (`"gpt-oss"`, `"gemma4"`, `"qwen3"`, `"lfm2"`, `"minimax-m2"`,
    /// `"deepseek-v4"`, `"generic"`). See `gallium_agent::profile` and
    /// docs/adr/0003-model-profiles.md.
    ///
    /// Absent means detect it from what the model file reports, which is right
    /// almost always. Naming one here is for the two cases detection cannot
    /// serve: a repackaged or mislabeled GGUF, and pinning a testsuite backend so
    /// a detection regression fails as "wrong profile" instead of as flaky tool
    /// calls. A name no profile answers to is a startup error listing the real
    /// ones — never a silent fall back to `generic`.
    pub profile: Option<String>,
    /// Local backend for `model_path`: "llamacpp" (default) or "candle".
    pub inference_engine: Option<String>,
    /// Where the native candle backend finds its `tokenizer.json`, for the
    /// common case of a GGUF repo that ships none. Ignored by the llama.cpp
    /// backend, which uses the tokenizer inside the GGUF.
    ///
    /// Takes the same two shapes as `model_path`: a local path, or a
    /// HuggingFace repo to fetch it from. See [`resolve_tokenizer_path`] for
    /// how the two are told apart.
    pub tokenizer_path: Option<String>,
    /// Layers to offload to the GPU for the llama.cpp backend. `None` means
    /// llama.cpp's default (offload everything); a full model that does not
    /// fit a smaller card's VRAM fails to load rather than falling back to a
    /// partial offload, so this is how a config pins the number that is known
    /// to work on the machine it targets rather than restating
    /// `GALLIUM_GPU_LAYERS` in every shell that runs it. The env var still
    /// wins when both are set — same precedence as every other setting here.
    pub gpu_layers: Option<u32>,
    /// Move every MoE expert tensor to CPU RAM for the llama.cpp backend,
    /// keeping attention and the KV cache on the GPU (mirrors llama.cpp's
    /// `--n-cpu-moe` in spirit — see the long comment in `llm_local.rs` for
    /// why it's all-or-nothing here rather than layer-graduated). For a
    /// sparse MoE this trades a slower per-token CPU hop for the experts
    /// actually routed to against a much smaller VRAM footprint, since the
    /// expert tensors are most of the file but only a few are read per
    /// token. `false` (the default) offloads experts the same as everything
    /// else, per `gpuLayers`.
    #[serde(default)]
    pub cpu_moe: bool,
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
    /// Per-risk-tier approval rules. Absent means the built-in policy.
    #[serde(default)]
    pub approvals: ApprovalsConfig,
    /// Where per-turn traces are written. Absent means none are.
    #[serde(default)]
    pub trace: TraceConfig,
    /// **Removed.** Kept only so a config that still names it can be told, since
    /// serde ignores unknown fields and the symptom is otherwise a GPU box that
    /// quietly speaks stdio to nobody. Use `--listen` — see `parse_listen_flag`
    /// for why the address is typed rather than configured.
    pub listen: Option<String>,
}

/// The `[agent.trace]` table. Naming a directory turns tracing on; the
/// `GALLIUM_TRACE` / `GALLIUM_TRACE_DIR` env vars override it either way, so a
/// single run can turn it on without a config or off without editing one.
///
/// There is no `enabled` key. A directory is the only setting tracing has, and
/// two ways to say "on" that can disagree is one too many.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceConfig {
    pub dir: Option<String>,
}

/// The `[agent.approvals]` table: one rule per risk tier, each `"allow"`,
/// `"ask"`, or `"deny"`. Absent keys keep their default.
///
/// There is deliberately no key for the read-only tier. A configuration that
/// can make reading a file prompt is a configuration someone will write by
/// accident, and no useful surface wants it.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalsConfig {
    pub workspace_write: Option<String>,
    pub external_side_effect: Option<String>,
    pub destructive: Option<String>,
}

impl ApprovalsConfig {
    /// Layer these keys over the default policy. An unrecognized value is
    /// reported and the default kept: guessing which rule someone meant by
    /// `"maybe"` is how a config silently grants more than it says.
    pub fn resolve(&self) -> ApprovalPolicy {
        let mut policy = ApprovalPolicy::default();
        let apply = |key: &str, value: &Option<String>, slot: &mut ApprovalRule| {
            let Some(raw) = value else { return };
            match ApprovalRule::parse(raw) {
                Some(rule) => *slot = rule,
                None => eprintln!(
                    "Warning: [agent.approvals] {key} = \"{raw}\" is not one of \
                     allow/ask/deny — keeping the default"
                ),
            }
        };
        apply(
            "workspaceWrite",
            &self.workspace_write,
            &mut policy.workspace_write,
        );
        apply(
            "externalSideEffect",
            &self.external_side_effect,
            &mut policy.external_side_effect,
        );
        apply("destructive", &self.destructive, &mut policy.destructive);
        policy
    }
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
    match parse_flag(args, "--config", Some("-c"), "a path argument")? {
        // A config file at "" is nothing anyone meant, so it stays a usage error
        // — unlike `--listen=`, where empty is a thing to say.
        Some(val) if val.is_empty() => Err("--config= requires a path".to_string()),
        other => Ok(other),
    }
}

/// Extract `--listen <host:port>` / `--listen=<host:port>` from argv.
///
/// This is the **only** way to make an app-server listen. There is deliberately
/// no env var and no config key: every other setting configures the server a
/// client *spawns*, and such a client wants stdio — so an address arriving from
/// the environment or from `~/.config/gallium/config.toml` could only ever turn
/// a spawned server into one that opens a socket and never reads the stdin it
/// was handed. The client is told nothing and waits for a reply that is not
/// coming. Requiring the address to be typed for the run that wants it makes
/// that unrepresentable rather than merely documented.
///
/// An empty value is not an error, unlike `--config=`; it simply names no
/// address, which is stdio.
pub fn parse_listen_flag(args: &[String]) -> Result<Option<String>, String> {
    parse_flag(args, "--listen", None, "a host:port address")
}

/// The shared shape of gallium's value flags: `--name value`, `--name=value`,
/// and an optional short alias. The first occurrence wins; a `--name` with
/// nothing after it is a usage error rather than a silently ignored flag.
///
/// `what` names what was expected, for that error message.
fn parse_flag(
    args: &[String],
    long: &str,
    short: Option<&str>,
    what: &str,
) -> Result<Option<String>, String> {
    let inline = format!("{long}=");
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if let Some(val) = arg.strip_prefix(&inline) {
            return Ok(Some(val.to_string()));
        }
        if arg == long || short.is_some_and(|s| arg == s) {
            return match it.next() {
                Some(val) => Ok(Some(val.clone())),
                None => Err(format!("{arg} requires {what}")),
            };
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| a.to_string()).collect()
    }

    /// Both spellings, and the two ways of saying nothing. An empty `--listen=`
    /// is not an error the way `--config=` is: it names no address, which is
    /// stdio.
    #[test]
    fn the_listen_flag_takes_an_address_either_way_round() {
        assert_eq!(
            parse_listen_flag(&argv(&["gallium", "app-server", "--listen", "1.2.3.4:5"])),
            Ok(Some("1.2.3.4:5".to_string()))
        );
        assert_eq!(
            parse_listen_flag(&argv(&["gallium", "app-server", "--listen=1.2.3.4:5"])),
            Ok(Some("1.2.3.4:5".to_string()))
        );
        assert_eq!(
            parse_listen_flag(&argv(&["gallium", "app-server"])),
            Ok(None)
        );
        assert_eq!(
            parse_listen_flag(&argv(&["gallium", "app-server", "--listen="])),
            Ok(Some(String::new()))
        );
    }

    /// A flag with nothing after it is a usage error, not a silently ignored
    /// flag: someone who typed `--listen` meant to start a server.
    #[test]
    fn a_listen_flag_without_an_address_is_a_usage_error() {
        let err = parse_listen_flag(&argv(&["gallium", "app-server", "--listen"])).unwrap_err();
        assert!(err.contains("--listen"), "{err}");
    }

    /// The `--config` messages are unchanged by sharing a parser with `--listen`.
    #[test]
    fn config_still_refuses_an_empty_or_missing_path() {
        assert_eq!(
            parse_config_flag(&argv(&["gallium", "--config="])),
            Err("--config= requires a path".to_string())
        );
        assert_eq!(
            parse_config_flag(&argv(&["gallium", "-c"])),
            Err("-c requires a path argument".to_string())
        );
        assert_eq!(
            parse_config_flag(&argv(&["gallium", "-c", "a.toml"])),
            Ok(Some("a.toml".to_string()))
        );
    }

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

    /// An absent `[agent.approvals]` is the built-in policy — a config that says
    /// nothing about approvals must not change them.
    #[test]
    fn no_approvals_table_keeps_the_default_policy() {
        let file: FileConfig = toml::from_str("[agent]\nmaxTurns = 5\n").unwrap();

        assert_eq!(file.agent.approvals.resolve(), ApprovalPolicy::default());
    }

    #[test]
    fn approval_rules_come_from_the_config() {
        let file: FileConfig =
            toml::from_str("[agent.approvals]\nworkspaceWrite = \"ask\"\ndestructive = \"deny\"\n")
                .unwrap();

        let policy = file.agent.approvals.resolve();

        assert_eq!(policy.workspace_write, ApprovalRule::Ask);
        assert_eq!(policy.destructive, ApprovalRule::Deny);
        // Untouched keys keep their default rather than being reset.
        assert_eq!(
            policy.external_side_effect,
            ApprovalPolicy::default().external_side_effect
        );
    }

    /// A value nobody recognizes keeps the default. Guessing which rule someone
    /// meant is how a config grants more than it says.
    #[test]
    fn an_unreadable_rule_falls_back_rather_than_guessing() {
        let file: FileConfig =
            toml::from_str("[agent.approvals]\ndestructive = \"whenever\"\n").unwrap();

        assert_eq!(
            file.agent.approvals.resolve().destructive,
            ApprovalPolicy::default().destructive
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
