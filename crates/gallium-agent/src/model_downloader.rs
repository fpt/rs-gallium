//! HuggingFace model downloader.
//!
//! Resolves an `hf:` model spec to a local GGUF path, downloading it into the
//! standard HuggingFace hub cache (`~/.cache/huggingface/hub/models--org--name/`
//! with `blobs/`, `snapshots/<commit>/`, `refs/`) if missing.
//!
//! Downloads are **transactional**: bytes stream into `blobs/<etag>.incomplete`
//! and the file is atomically renamed to `blobs/<etag>` only on success. If an
//! `.incomplete` file already exists the download **resumes** from its size via
//! an HTTP `Range` request — automatically, up to a few times, if the
//! connection drops mid-stream (common on very large files; see
//! [`download_blob_with_retry`]).
//!
//! **Split files**: a spec naming one shard of a `<name>-<NNNNN>-of-<MMMMM>.gguf`
//! set (common once a quantized model exceeds ~50GB) fetches every sibling
//! shard too, not just the one named — see [`split_shard_filenames`]. llama.cpp
//! needs all of them present on disk, named by the same convention, to
//! auto-discover the split when opening shard 1.
//!
//! Spec format: `hf:ORG/REPO[@REVISION]/path/to/file.gguf`
//!   e.g. `hf:LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf`

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};

/// Resolve a model spec to a local file path, downloading if necessary.
///
/// - A plain path is returned as-is if it exists (else an error).
/// - An `hf:` spec is downloaded into the HF hub cache and the snapshot path
///   is returned.
pub fn ensure_model(spec: &str) -> Result<PathBuf> {
    if let Some((repo, revision, file)) = parse_hf_spec(spec) {
        return download_to_cache(&repo, &revision, &file);
    }
    let path = PathBuf::from(spec);
    if path.exists() {
        Ok(path)
    } else {
        bail!("Model file not found: {spec}")
    }
}

/// Parse `hf:ORG/REPO[@REV]/path/file` → (repo="ORG/REPO", revision, file).
fn parse_hf_spec(spec: &str) -> Option<(String, String, String)> {
    let rest = spec
        .strip_prefix("hf://")
        .or_else(|| spec.strip_prefix("hf:"))?;
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() < 3 || parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let org = parts[0];
    let (repo_name, revision) = match parts[1].split_once('@') {
        Some((name, rev)) => (name, rev.to_string()),
        None => (parts[1], "main".to_string()),
    };
    Some((format!("{org}/{repo_name}"), revision, parts[2].to_string()))
}

/// HuggingFace hub cache root, honoring the standard env vars.
fn hub_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let base = home_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join(".cache").join("huggingface").join("hub")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn hf_token() -> Option<String> {
    let from_env = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty());
    from_env.or_else(token_file)
}

/// The token `huggingface-cli login` writes. Read because this downloader is now
/// the only HTTP client reaching the hub, and it is where people expect their
/// login to apply.
fn token_file() -> Option<String> {
    let path = match std::env::var("HF_HOME") {
        Ok(home) => PathBuf::from(home).join("token"),
        Err(_) => home_dir()?.join(".cache").join("huggingface").join("token"),
    };
    let token = fs::read_to_string(path).ok()?.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// The token value the hub last rejected, so the remaining requests of a run —
/// every shard of a safetensors repo — skip the attempt that will fail instead of
/// paying a 401 each.
///
/// Keyed on the token rather than latched on/off, because the process outlives
/// any one credential: the REPL and the app-server both run for hours, and a
/// `hf auth login` or a fresh `HF_TOKEN` part-way through must get its own
/// chance. A one-way flag would have disabled authentication — and so every
/// gated repo — for the rest of the process.
static REJECTED_TOKEN: Mutex<Option<String>> = Mutex::new(None);

/// Lock the rejected-token slot, ignoring poisoning: a panic elsewhere while
/// holding it says nothing about the token, and wedging every later download
/// would be a worse answer than reading it anyway.
fn rejected_token() -> std::sync::MutexGuard<'static, Option<String>> {
    REJECTED_TOKEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Send a hub request, authenticated if we hold a usable token, and retry
/// anonymously if the hub rejects it.
///
/// A token the hub refuses is common and not always visible locally: an OAuth
/// token whose signature no longer verifies is still an unexpired entry in
/// `~/.cache/huggingface/token`, and `hf auth list` reports it as current. The
/// hub answers 401 for it where it would have served the same public file to
/// nobody in particular, so a token we cannot use must not be worse than no
/// token.
// `ureq::Error` is a large enum, but it is what the call sites already match on
// to tell a status code from a transport failure. Boxing it here would buy a
// smaller Result at the cost of a deref in every one of them.
#[allow(clippy::result_large_err)]
fn call_hub(req: ureq::Request) -> Result<ureq::Response, ureq::Error> {
    let Some(token) = hf_token() else {
        return req.call();
    };
    // Re-read every call, so replacing the token is picked up rather than
    // shadowed by the last one's rejection.
    if rejected_token().as_deref() == Some(token.as_str()) {
        return req.call();
    }
    let authed = req
        .clone()
        .set("Authorization", &format!("Bearer {token}"))
        .call();
    match authed {
        Err(ureq::Error::Status(401 | 403, _)) => {
            let mut rejected = rejected_token();
            // Warn once per token, not once per request: a repo is many shards.
            if rejected.as_deref() != Some(token.as_str()) {
                tracing::warn!(
                    "HuggingFace rejected the stored token; continuing anonymously. \
                     `hf auth login --force` restores it (needed only for gated repos)"
                );
                *rejected = Some(token);
            }
            drop(rejected);
            req.call()
        }
        other => other,
    }
}

/// Split an `[hf:]ORG/NAME[@REVISION]` repo spec into repo id and revision.
///
/// The `hf:` prefix is accepted so a `tokenizerPath` reads the same as a
/// `modelPath`, and dropped because the hub API wants neither it nor a trailing
/// slash.
fn split_revision(repo: &str) -> (&str, &str) {
    let repo = repo
        .strip_prefix("hf://")
        .or_else(|| repo.strip_prefix("hf:"))
        .unwrap_or(repo)
        .trim_end_matches('/');
    match repo.split_once('@') {
        Some((name, rev)) => (name, rev),
        None => (repo, "main"),
    }
}

/// Resolve one file in a HuggingFace repo to a local path, downloading it into
/// the hub cache if missing.
///
/// Everything the hub is asked for goes through here — GGUFs, `tokenizer.json`,
/// `config.json`, safetensors shards — so a corporate TLS-intercepting proxy is
/// configured in exactly one place (`SSL_CERT_FILE`, honored by
/// [`crate::llm::http_agent_with_ca`]) rather than once per HTTP client.
pub fn ensure_repo_file(repo: &str, file: &str) -> Result<PathBuf> {
    let (repo, revision) = split_revision(repo);
    download_to_cache(repo, revision, file)
}

/// The file names a HuggingFace repo holds. Needed because safetensors shard
/// names vary per repo, so they cannot be guessed.
///
/// Falls back to the names already in the cache when the hub cannot be reached,
/// for the same reason [`download_to_cache`] does: this is the *first* call a
/// safetensors load makes, so without the fallback here a fully cached repo would
/// still fail offline before reaching the per-file one.
pub fn list_repo_files(repo: &str) -> Result<Vec<String>> {
    let (repo, revision) = split_revision(repo);
    match fetch_repo_listing(repo, revision) {
        Ok(files) => Ok(files),
        Err(e) => {
            let repo_dir = repo_cache_dir(repo);
            match cached_repo_listing(&repo_dir, revision) {
                Some(files) => {
                    tracing::warn!(
                        "Listing {} cached file(s) for {repo} (could not reach HuggingFace: {e})",
                        files.len()
                    );
                    Ok(files)
                }
                None => Err(e),
            }
        }
    }
}

/// The repo listing from the hub API.
fn fetch_repo_listing(repo: &str, revision: &str) -> Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(serde::Deserialize)]
    struct RepoInfo {
        #[serde(default)]
        siblings: Vec<Sibling>,
    }

    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let agent = crate::llm::http_agent_with_ca(None);
    let resp = match call_hub(agent.get(&url).set("User-Agent", "gallium")) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            bail!("HuggingFace returned {code} for {url}: {}", r.status_text());
        }
        Err(e) => return Err(anyhow!("GET {url} failed: {e}")),
    };
    let info: RepoInfo = resp
        .into_json()
        .with_context(|| format!("could not parse repo listing from {url}"))?;
    Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

/// Metadata for a repo file, read from the `resolve` endpoint's headers.
struct FileMeta {
    commit: String,
    etag: String,
    size: Option<u64>,
    /// Final download URL (CDN location for LFS files, else the resolve URL).
    url: String,
}

fn fetch_meta(repo: &str, revision: &str, file: &str) -> Result<FileMeta> {
    let resolve_url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");

    // Don't follow redirects: the 302 carries X-Repo-Commit / X-Linked-Etag and
    // the CDN Location, which we'd otherwise lose. Honor SSL_CERT_FILE so HF
    // downloads work behind a corporate TLS-intercept proxy (e.g. Zscaler).
    let agent = crate::llm::http_agent_with_ca(Some(0));
    let resp = match call_hub(agent.head(&resolve_url).set("User-Agent", "gallium")) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            bail!(
                "HuggingFace returned {code} for {resolve_url}: {}",
                r.status_text()
            );
        }
        Err(e) => return Err(anyhow!("HEAD {resolve_url} failed: {e}")),
    };

    let status = resp.status();
    let commit = resp
        .header("X-Repo-Commit")
        .map(str::to_string)
        .unwrap_or_else(|| revision.to_string());
    let etag_raw = resp
        .header("X-Linked-Etag")
        .or_else(|| resp.header("ETag"))
        .ok_or_else(|| anyhow!("No ETag header for {resolve_url}"))?;
    let etag = etag_raw
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_string();
    // `X-Linked-Size` is the real file size (LFS files). `Content-Length` is
    // only the file size on a 200 — on a 3xx it describes the redirect body, not
    // the target, so a non-LFS file (config.json, tokenizer.json) would get a
    // bogus small size and then fail the post-download `got != total` check.
    let size = resp
        .header("X-Linked-Size")
        .or_else(|| {
            (status == 200)
                .then(|| resp.header("Content-Length"))
                .flatten()
        })
        .and_then(|s| s.parse::<u64>().ok());

    let url = if (300..400).contains(&status) {
        let loc = resp
            .header("Location")
            .ok_or_else(|| anyhow!("Redirect without Location for {resolve_url}"))?;
        // LFS files redirect to an absolute CDN URL; small non-LFS files
        // (config.json, tokenizer.json, …) redirect to a *relative*
        // `/api/resolve-cache/...` path, which is not a URL on its own — resolve
        // it against the hub origin.
        if loc.starts_with("http://") || loc.starts_with("https://") {
            loc.to_string()
        } else if let Some(path) = loc.strip_prefix('/') {
            format!("https://huggingface.co/{path}")
        } else {
            format!("https://huggingface.co/{repo}/resolve/{revision}/{loc}")
        }
    } else {
        resolve_url
    };

    Ok(FileMeta {
        commit,
        etag,
        size,
        url,
    })
}

/// Where the hub cache keeps one repo: `models--org--name` under the cache root.
fn repo_cache_dir(repo: &str) -> PathBuf {
    hub_cache_dir().join(format!("models--{}", repo.replace('/', "--")))
}

/// Matches the GGUF/safetensors split convention: `<stem>-<idx>-of-<count>.<ext>`,
/// e.g. `MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf`. `idx` and `count` are
/// captured as their original digit strings so the zero-padding width is
/// preserved when reconstructing sibling names, rather than assumed to be 5.
fn split_pattern() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"^(.+)-(\d+)-of-(\d+)(\.[^./]+)$").unwrap())
}

/// If `file` names one shard of a split GGUF/safetensors file, every shard's
/// repo-relative path in order (1..=count), same directory and zero-padding
/// width as `file` itself. `None` if `file` isn't part of a split set — most
/// files, so this is cheap to call unconditionally.
///
/// This reads the shard count out of the filename alone. No repo listing is
/// needed (unlike the safetensors whole-repo path in `llm_candle.rs`, which
/// downloads every shard because it doesn't know a priori how many there
/// are) — a GGUF spec already commits to exactly one quant variant, and that
/// variant's own name says how many parts it comes in.
fn split_shard_filenames(file: &str) -> Option<Vec<String>> {
    let (dir, base) = match file.rsplit_once('/') {
        Some((d, b)) => (format!("{d}/"), b),
        None => (String::new(), file),
    };
    let caps = split_pattern().captures(base)?;
    let stem = &caps[1];
    let idx_str = &caps[2];
    let count_str = &caps[3];
    let ext = &caps[4];
    let count: usize = count_str.parse().ok()?;
    let width = idx_str.len();
    if count == 0 {
        return None;
    }
    Some(
        (1..=count)
            .map(|i| format!("{dir}{stem}-{i:0width$}-of-{count_str}{ext}"))
            .collect(),
    )
}

/// Resolve one file, downloading every sibling shard first if it's split.
///
/// Split shards are fetched **in order starting from shard 1**, regardless
/// of which shard `file` itself names: llama.cpp's split loader has to be
/// pointed at the first shard specifically to auto-discover the rest by the
/// same naming pattern, so that's the path this returns even if the caller
/// asked for a later one. "In order" governs which path is *returned*, not
/// the order shards finish downloading in — see [`download_shards_in_parallel`].
fn download_to_cache(repo: &str, revision: &str, file: &str) -> Result<PathBuf> {
    match split_shard_filenames(file) {
        Some(shards) if shards.len() > 1 => {
            tracing::info!(
                "{file} is part of a {}-shard split file; fetching every shard \
                 in parallel (llama.cpp needs them all present on disk to \
                 auto-discover the split)",
                shards.len()
            );
            download_shards_in_parallel(repo, revision, &shards)
        }
        _ => download_one_file(repo, revision, file, Progress::Solo),
    }
}

/// How many shards to fetch at once. A hub CDN comfortably serves this many
/// concurrent connections from one client (this is well below what a plain
/// browser opens per origin), and it bounds the downside for a split with an
/// unusually large shard count — this repo has not seen one past 4, but
/// nothing stops a future model from shipping more.
const MAX_PARALLEL_SHARD_DOWNLOADS: usize = 4;

/// Fetch every shard in `shards`, up to [`MAX_PARALLEL_SHARD_DOWNLOADS`] at
/// once, and return shard 1's path (see [`download_to_cache`] for why that's
/// the one that matters).
///
/// Batched rather than one thread per shard unconditionally: an unbounded
/// spawn is fine for this repo's usual 2-4 shards but would open as many
/// concurrent multi-GB streams as a future split has shards, with no cap.
/// Each batch fully joins (success or error) before the next one starts —
/// downloads can't be cancelled mid-stream once spawned, so a batch that's
/// already running finishes rather than being abandoned, but a batch that
/// hasn't started yet is skipped once an earlier one has failed.
fn download_shards_in_parallel(repo: &str, revision: &str, shards: &[String]) -> Result<PathBuf> {
    let mut first_path = None;
    for batch in shards.chunks(MAX_PARALLEL_SHARD_DOWNLOADS) {
        let results: Vec<Result<PathBuf>> = std::thread::scope(|scope| {
            let handles: Vec<_> = batch
                .iter()
                .map(|shard| {
                    scope.spawn(move || {
                        download_one_file(repo, revision, shard, Progress::Concurrent)
                            .with_context(|| format!("failed to fetch split shard {shard}"))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| bail!("download thread panicked"))
                })
                .collect()
        });
        for path in results {
            first_path.get_or_insert(path?);
        }
    }
    Ok(first_path.expect("shards is non-empty"))
}

fn download_one_file(
    repo: &str,
    revision: &str,
    file: &str,
    progress: Progress,
) -> Result<PathBuf> {
    let repo_dir = repo_cache_dir(repo);
    let meta = match fetch_meta(repo, revision, file) {
        Ok(meta) => meta,
        Err(e) => {
            // The network was needed only to learn the current commit. A file
            // already in the cache is still the file, so an offline (or
            // firewalled) run should use it rather than fail.
            if let Some(cached) = cached_snapshot(&repo_dir, revision, file) {
                tracing::warn!(
                    "Using cached {} (could not reach HuggingFace: {e})",
                    cached.display()
                );
                return Ok(cached);
            }
            return Err(e);
        }
    };

    let snapshot_file = repo_dir.join("snapshots").join(&meta.commit).join(file);
    if snapshot_file.exists() {
        tracing::info!("Already cached: {}", snapshot_file.display());
        return Ok(snapshot_file);
    }

    let blob_path = repo_dir.join("blobs").join(&meta.etag);
    if !blob_path.exists() {
        let display_name = Path::new(file)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(file);
        download_blob_with_retry(&meta, &blob_path, display_name, progress)?;
    }

    link_snapshot(&blob_path, &snapshot_file)?;

    // Record the branch → commit mapping like huggingface_hub does. Every
    // shard of a split file shares the same `repo`/`revision` and so writes
    // the identical `refs_path` — since `download_shards_in_parallel` runs
    // several of these concurrently, this is now more than one thread
    // targeting the same file. All writers agree on the content (the same
    // commit), so a torn write couldn't leave the wrong value, but relying on
    // `fs::write`'s truncate-then-write being atomic enough in practice isn't
    // a guarantee the standard library makes. Write-to-temp-then-rename
    // instead — the same pattern `download_blob` already uses for the blob
    // itself — so a reader (a later, separate process) always sees either the
    // old or the complete new content, never a partial one, regardless of
    // how the writers interleave. The temp name is unique per call, not
    // shared, so concurrent writers never collide on *that* either.
    let refs_dir = repo_dir.join("refs");
    let _ = fs::create_dir_all(&refs_dir);
    let refs_path = refs_dir.join(revision);
    let refs_tmp = refs_dir.join(format!(
        "{revision}.tmp.{}.{}",
        std::process::id(),
        REFS_TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    if fs::write(&refs_tmp, &meta.commit).is_ok() {
        let _ = fs::rename(&refs_tmp, &refs_path);
    }

    Ok(snapshot_file)
}

/// Disambiguates concurrent shard-download threads' refs temp files within
/// one process — see the write site above. `process::id()` alone isn't
/// enough since several shards of one download share a process.
static REFS_TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A previously downloaded copy of `file`, if the cache has one.
fn cached_snapshot(repo_dir: &Path, revision: &str, file: &str) -> Option<PathBuf> {
    cached_snapshot_dirs(repo_dir, revision)
        .into_iter()
        .map(|dir| dir.join(file))
        .find(|path| path.exists())
}

/// The cache directories that may hold a snapshot of `revision`, most direct
/// first: `revision` is either a commit naming a snapshot outright, or a branch
/// this repo has a `refs/` entry for.
fn cached_snapshot_dirs(repo_dir: &Path, revision: &str) -> Vec<PathBuf> {
    let snapshots = repo_dir.join("snapshots");
    let mut dirs = vec![snapshots.join(revision)];
    if let Ok(commit) = fs::read_to_string(repo_dir.join("refs").join(revision)) {
        dirs.push(snapshots.join(commit.trim()));
    }
    dirs.retain(|dir| dir.is_dir());
    dirs
}

/// The repo-relative names of every file in a cached snapshot — what
/// [`fetch_repo_listing`] would have returned, read off the disk instead.
fn cached_repo_listing(repo_dir: &Path, revision: &str) -> Option<Vec<String>> {
    cached_snapshot_dirs(repo_dir, revision)
        .into_iter()
        .find_map(|dir| {
            let mut files = Vec::new();
            collect_snapshot_files(&dir, "", &mut files);
            (!files.is_empty()).then_some(files)
        })
}

/// Walk a snapshot directory into hub-style `dir/file` relative names.
///
/// Entries are symlinks into `blobs/`, so what matters is what they point at:
/// `is_dir` follows the link where `file_type` would just say "symlink".
fn collect_snapshot_files(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let path = entry.path();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        if path.is_dir() {
            collect_snapshot_files(&path, &rel, out);
        } else {
            out.push(rel);
        }
    }
}

/// Retries [`download_blob`] on a connection dropping mid-stream, which a very
/// large file (tens of GB, common once a model needs splitting at all — see
/// [`split_shard_filenames`]) hits often enough in practice that a human
/// re-running the command by hand shouldn't be the retry mechanism. Every
/// retry resumes from `.incomplete`'s current size via `download_blob`'s own
/// Range request, so it costs only the bytes lost since the last flush, not
/// the whole file.
///
/// Only errors `download_blob` marks `"transient network error: "` are
/// retried — a bad HTTP status (404, 403) is not, since it will fail
/// identically every time and five delayed identical failures is a worse
/// error experience than one immediate one.
fn download_blob_with_retry(
    meta: &FileMeta,
    blob_path: &Path,
    display_name: &str,
    progress: Progress,
) -> Result<()> {
    const MAX_ATTEMPTS: u32 = 6;
    for attempt in 1..=MAX_ATTEMPTS {
        match download_blob(meta, blob_path, display_name, progress) {
            Ok(()) => return Ok(()),
            Err(e)
                if attempt < MAX_ATTEMPTS && e.to_string().contains("transient network error") =>
            {
                let backoff = Duration::from_secs(2u64.saturating_pow(attempt).min(30));
                tracing::warn!(
                    "{display_name}: download interrupted (attempt {attempt}/{MAX_ATTEMPTS}): \
                     {e} — resuming in {}s",
                    backoff.as_secs()
                );
                std::thread::sleep(backoff);
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("loop always returns by MAX_ATTEMPTS")
}

/// Stream the blob to `<blob>.incomplete`, resuming if it already exists, then
/// atomically rename to the final blob path.
fn download_blob(
    meta: &FileMeta,
    blob_path: &Path,
    display_name: &str,
    progress: Progress,
) -> Result<()> {
    if let Some(parent) = blob_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let incomplete = blob_path.with_extension("incomplete");

    let mut already: u64 = fs::metadata(&incomplete).map(|m| m.len()).unwrap_or(0);
    if let Some(total) = meta.size {
        if already > total {
            // Corrupt/oversized partial — start over.
            already = 0;
            let _ = fs::remove_file(&incomplete);
        }
    }

    // Follows redirects (CDN may redirect again); honors SSL_CERT_FILE.
    let agent = crate::llm::http_agent_with_ca(None);
    let mut req = agent.get(&meta.url).set("User-Agent", "gallium");
    if already > 0 {
        tracing::info!("Resuming download from byte {already}");
        req = req.set("Range", &format!("bytes={already}-"));
    }

    let resp = match call_hub(req) {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            bail!("Download failed ({code}): {}", r.status_text());
        }
        Err(e) => {
            return Err(anyhow!(
                "transient network error: GET {} failed: {e}",
                meta.url
            ))
        }
    };

    // If we asked for a range but the server sent the whole file (200), restart.
    let append = already > 0 && resp.status() == 206;
    if !append {
        already = 0;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .open(&incomplete)
        .with_context(|| format!("Failed to open {}", incomplete.display()))?;
    if !append {
        file.set_len(0).ok();
    }

    let total = meta.size;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 16];
    let mut downloaded = already;
    let mut last_report = already;
    // `Solo` redraws one line, so 8MB granularity reads as smooth progress.
    // `Concurrent` prints a new line per update (see `Progress`'s doc for
    // why), so the same granularity across several 50GB shards downloading
    // at once would scroll-flood the terminal — coarsen it by 32x.
    let report_every: u64 = match progress {
        Progress::Solo => 8 * 1024 * 1024,
        Progress::Concurrent => 256 * 1024 * 1024,
    };

    loop {
        let n = reader
            .read(&mut buf)
            .context("transient network error: read from network")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write to disk")?;
        downloaded += n as u64;
        if downloaded - last_report >= report_every {
            last_report = downloaded;
            report_progress(display_name, downloaded, total, progress);
        }
    }
    file.flush().ok();
    drop(file);
    report_progress(display_name, downloaded, total, progress);
    if progress == Progress::Solo {
        // `Concurrent` already ends every line with its own `\n`; this is
        // only to move off a `\r`-redrawn line before the next thing prints.
        eprintln!();
    }

    if let Some(total) = total {
        let got = fs::metadata(&incomplete).map(|m| m.len()).unwrap_or(0);
        if got != total {
            bail!(
                "transient network error: incomplete download, got {got} of {total} bytes \
                 (partial kept for resume)"
            );
        }
    }

    fs::rename(&incomplete, blob_path)
        .with_context(|| format!("Failed to finalize {}", blob_path.display()))?;
    Ok(())
}

/// How [`report_progress`] prints — see [`download_shards_in_parallel`] for
/// why one download in flight and several need different treatment.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Progress {
    /// One line, redrawn in place via `\r`. Correct only when exactly one
    /// download is writing progress to stderr at a time — two threads
    /// redrawing the same line at once interleaves into unreadable garbage.
    Solo,
    /// One line **per update**, newline-terminated and prefixed by `name` —
    /// safe with any number of concurrent writers, since each update is a
    /// complete, independent line rather than an in-place redraw.
    Concurrent,
}

fn report_progress(name: &str, downloaded: u64, total: Option<u64>, progress: Progress) {
    let mb = downloaded as f64 / 1_000_000.0;
    let line = match total {
        Some(t) if t > 0 => {
            let pct = (downloaded as f64 / t as f64 * 100.0) as u32;
            format!(
                "Downloading {name}: {:.0}/{:.0} MB ({pct}%)",
                mb,
                t as f64 / 1_000_000.0
            )
        }
        _ => format!("Downloading {name}: {mb:.0} MB"),
    };
    match progress {
        Progress::Solo => eprint!("\r{line}"),
        Progress::Concurrent => eprintln!("{line}"),
    }
    let _ = std::io::stderr().flush();
}

/// Point the snapshot path at the blob. Tries a hard link (works on Windows and
/// Unix without privilege, same volume) and falls back to a full copy.
fn link_snapshot(blob_path: &Path, snapshot_file: &Path) -> Result<()> {
    if let Some(parent) = snapshot_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    if snapshot_file.exists() {
        let _ = fs::remove_file(snapshot_file);
    }
    match fs::hard_link(blob_path, snapshot_file) {
        Ok(()) => Ok(()),
        Err(_) => fs::copy(blob_path, snapshot_file)
            .map(|_| ())
            .with_context(|| {
                format!(
                    "Failed to link/copy blob {} -> {}",
                    blob_path.display(),
                    snapshot_file.display()
                )
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case that prompted this: a 3-shard split GGUF in a quant subdirectory.
    #[test]
    fn split_shards_preserve_directory_and_width() {
        let shards =
            split_shard_filenames("UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf")
                .unwrap();
        assert_eq!(
            shards,
            [
                "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf",
                "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf",
                "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00003-of-00003.gguf",
            ]
        );
    }

    /// Naming a later shard still yields every shard, shard 1 first — the
    /// path llama.cpp's split loader actually needs to be pointed at, not
    /// necessarily the one a config happened to name.
    #[test]
    fn split_shards_start_from_one_even_if_a_later_shard_was_named() {
        let shards = split_shard_filenames("model-00002-of-00004.gguf").unwrap();
        assert_eq!(shards[0], "model-00001-of-00004.gguf");
        assert_eq!(shards.len(), 4);
    }

    /// A different zero-padding width (e.g. safetensors' usual 5-digit vs. a
    /// hypothetical shorter one) round-trips rather than being hardcoded to 5.
    #[test]
    fn split_shards_use_the_source_files_own_padding_width() {
        let shards = split_shard_filenames("model-001-of-010.safetensors").unwrap();
        assert_eq!(shards[0], "model-001-of-010.safetensors");
        assert_eq!(shards[9], "model-010-of-010.safetensors");
    }

    /// An ordinary, non-split filename — most files — isn't mistaken for one.
    #[test]
    fn a_non_split_filename_is_not_a_split_file() {
        assert!(split_shard_filenames("gemma-4-12B-it-qat-UD-Q4_K_XL.gguf").is_none());
    }

    /// [`download_shards_in_parallel`]'s batching: at most
    /// `MAX_PARALLEL_SHARD_DOWNLOADS` shards run at once, and a shard count
    /// that doesn't divide evenly still covers every shard exactly once
    /// (the last batch is simply smaller, not padded or dropped).
    #[test]
    fn shard_batches_respect_the_parallel_cap_and_cover_every_shard() {
        for n in [1, 2, 3, 4, 5, 9, 13] {
            let shards: Vec<String> = (1..=n).map(|i| format!("shard{i}")).collect();
            let batches: Vec<&[String]> = shards.chunks(MAX_PARALLEL_SHARD_DOWNLOADS).collect();

            let covered: Vec<&String> = batches.iter().flat_map(|b| b.iter()).collect();
            assert_eq!(covered, shards.iter().collect::<Vec<_>>(), "n={n}");

            for batch in &batches {
                assert!(
                    batch.len() <= MAX_PARALLEL_SHARD_DOWNLOADS,
                    "n={n}: batch of {} exceeds the cap",
                    batch.len()
                );
                assert!(!batch.is_empty(), "n={n}: empty batch");
            }
        }
    }

    /// "-of-" appearing in a model name for unrelated reasons (no digits
    /// around it) must not be mistaken for the split marker.
    #[test]
    fn a_name_containing_of_without_digits_is_not_a_split_file() {
        assert!(split_shard_filenames("state-of-the-art-model.gguf").is_none());
    }

    #[test]
    fn parse_basic() {
        let (repo, rev, file) =
            parse_hf_spec("hf:LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf").unwrap();
        assert_eq!(repo, "LiquidAI/LFM2.5-8B-A1B-GGUF");
        assert_eq!(rev, "main");
        assert_eq!(file, "LFM2.5-8B-A1B-Q4_K_M.gguf");
    }

    #[test]
    fn parse_revision_and_subdir() {
        let (repo, rev, file) = parse_hf_spec("hf://org/repo@abc123/sub/model.gguf").unwrap();
        assert_eq!(repo, "org/repo");
        assert_eq!(rev, "abc123");
        assert_eq!(file, "sub/model.gguf");
    }

    #[test]
    fn non_hf_spec_is_none() {
        assert!(parse_hf_spec("/models/foo.gguf").is_none());
        assert!(parse_hf_spec("hf:org/repo").is_none()); // no file part
    }

    /// A repo spec is what `tokenizerPath` holds, so all three spellings people
    /// write it in have to mean the same repo.
    #[test]
    fn repo_spec_revision_and_prefix() {
        assert_eq!(
            split_revision("unsloth/gemma-4-12b-it"),
            ("unsloth/gemma-4-12b-it", "main")
        );
        assert_eq!(
            split_revision("hf:unsloth/gemma-4-12b-it/"),
            ("unsloth/gemma-4-12b-it", "main")
        );
        assert_eq!(
            split_revision("hf://org/repo@abc123"),
            ("org/repo", "abc123")
        );
    }

    /// The offline fallback: a file already in the cache is usable without the
    /// network, reached either through `refs/<branch>` or by naming the commit.
    #[test]
    fn cached_snapshot_found_via_ref_and_commit() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir.path().join("models--org--repo");
        let snap = repo_dir.join("snapshots").join("deadbeef");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("tokenizer.json"), "{}").unwrap();
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::write(repo_dir.join("refs").join("main"), "deadbeef\n").unwrap();

        assert!(cached_snapshot(&repo_dir, "main", "tokenizer.json").is_some());
        assert!(cached_snapshot(&repo_dir, "deadbeef", "tokenizer.json").is_some());
        assert!(cached_snapshot(&repo_dir, "main", "config.json").is_none());
    }

    /// A safetensors load lists the repo before it fetches anything, so the
    /// listing needs the same offline fallback the per-file path has — otherwise a
    /// fully cached repo still fails without the network.
    #[test]
    fn cached_repo_listing_covers_a_whole_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let repo_dir = dir.path().join("models--org--repo");
        let snap = repo_dir.join("snapshots").join("deadbeef");
        fs::create_dir_all(snap.join("nested")).unwrap();
        fs::write(snap.join("config.json"), "{}").unwrap();
        fs::write(snap.join("model-00001-of-00002.safetensors"), "x").unwrap();
        fs::write(snap.join("nested").join("extra.json"), "{}").unwrap();
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::write(repo_dir.join("refs").join("main"), "deadbeef\n").unwrap();

        let mut via_ref = cached_repo_listing(&repo_dir, "main").unwrap();
        via_ref.sort();
        assert_eq!(
            via_ref,
            [
                "config.json",
                "model-00001-of-00002.safetensors",
                // Nested files come back repo-relative with `/`, like the API's.
                "nested/extra.json",
            ]
        );
        assert!(cached_repo_listing(&repo_dir, "deadbeef").is_some());
        assert!(cached_repo_listing(&repo_dir, "no-such-rev").is_none());
    }

    /// The rejection is remembered per token, not as a one-way switch: a replaced
    /// token has to get its own attempt, or one stale credential would disable
    /// authentication — and every gated repo — for the rest of the process.
    #[test]
    fn a_replaced_token_is_tried_again() {
        let is_rejected = |t: &str| rejected_token().as_deref() == Some(t);

        *rejected_token() = None;
        assert!(!is_rejected("stale"));

        *rejected_token() = Some("stale".to_string());
        assert!(is_rejected("stale"));
        assert!(!is_rejected("freshly-logged-in"));

        *rejected_token() = None;
    }

    /// Regression: a small non-LFS file (`config.json`) from a bare safetensors
    /// repo. HuggingFace redirects those to a *relative* `/api/resolve-cache/…`
    /// path, which `fetch_meta` used to hand to the downloader verbatim — "Bad
    /// URL: relative URL without a base". Needs the network, so `#[ignore]`.
    #[test]
    #[ignore = "hits huggingface.co"]
    fn a_small_file_from_a_bare_repo_downloads() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", tmp.path());
        let got = ensure_repo_file("openai/gpt-oss-20b", "config.json")
            .expect("config.json from a bare repo must download");
        let body = std::fs::read_to_string(&got).unwrap();
        assert!(
            body.contains("gpt_oss") || body.contains("num_hidden_layers"),
            "{body:.200}"
        );
        std::env::remove_var("HF_HUB_CACHE");
    }
}
