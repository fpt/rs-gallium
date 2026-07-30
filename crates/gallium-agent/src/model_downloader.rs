//! HuggingFace model downloader.
//!
//! Resolves an `hf:` model spec to a local GGUF path, downloading it into the
//! standard HuggingFace hub cache (`~/.cache/huggingface/hub/models--org--name/`
//! with `blobs/`, `snapshots/<commit>/`, `refs/`) if missing.
//!
//! Downloads are **transactional**: bytes stream into `blobs/<etag>.incomplete`
//! and the file is atomically renamed to `blobs/<etag>` only on success. If an
//! `.incomplete` file already exists the download **resumes** from its size via
//! an HTTP `Range` request.
//!
//! Spec format: `hf:ORG/REPO[@REVISION]/path/to/file.gguf`
//!   e.g. `hf:LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf`

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

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

/// Set once the hub has rejected the token we hold, so the remaining requests of
/// a run — every shard of a safetensors repo — skip the attempt that will fail
/// instead of paying a 401 each.
static TOKEN_REJECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
    use std::sync::atomic::Ordering;

    if TOKEN_REJECTED.load(Ordering::Relaxed) {
        return req.call();
    }
    let Some(token) = hf_token() else {
        return req.call();
    };
    let authed = req
        .clone()
        .set("Authorization", &format!("Bearer {token}"))
        .call();
    match authed {
        Err(ureq::Error::Status(401 | 403, _)) => {
            if !TOKEN_REJECTED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "HuggingFace rejected the stored token; continuing anonymously. \
                     `hf auth login --force` restores it (needed only for gated repos)"
                );
            }
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

/// The file names a HuggingFace repo holds, from the hub API. Needed because
/// safetensors shard names vary per repo, so they cannot be guessed.
pub fn list_repo_files(repo: &str) -> Result<Vec<String>> {
    #[derive(serde::Deserialize)]
    struct Sibling {
        rfilename: String,
    }
    #[derive(serde::Deserialize)]
    struct RepoInfo {
        #[serde(default)]
        siblings: Vec<Sibling>,
    }

    let (repo, revision) = split_revision(repo);
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
    let size = resp
        .header("X-Linked-Size")
        .or_else(|| resp.header("Content-Length"))
        .and_then(|s| s.parse::<u64>().ok());

    let url = if (300..400).contains(&status) {
        resp.header("Location")
            .map(str::to_string)
            .ok_or_else(|| anyhow!("Redirect without Location for {resolve_url}"))?
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

fn download_to_cache(repo: &str, revision: &str, file: &str) -> Result<PathBuf> {
    let cache = hub_cache_dir();
    let repo_dir = cache.join(format!("models--{}", repo.replace('/', "--")));
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
        download_blob(&meta, &blob_path, display_name)?;
    }

    link_snapshot(&blob_path, &snapshot_file)?;

    // Record the branch → commit mapping like huggingface_hub does.
    let refs_path = repo_dir.join("refs").join(revision);
    if let Some(parent) = refs_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&refs_path, &meta.commit);

    Ok(snapshot_file)
}

/// A previously downloaded copy of `file`, if the cache has one: `revision` is
/// either a branch this repo has a `refs/` entry for, or a commit naming a
/// snapshot directly.
fn cached_snapshot(repo_dir: &Path, revision: &str, file: &str) -> Option<PathBuf> {
    let snapshots = repo_dir.join("snapshots");
    let direct = snapshots.join(revision).join(file);
    if direct.exists() {
        return Some(direct);
    }
    let commit = fs::read_to_string(repo_dir.join("refs").join(revision)).ok()?;
    let path = snapshots.join(commit.trim()).join(file);
    path.exists().then_some(path)
}

/// Stream the blob to `<blob>.incomplete`, resuming if it already exists, then
/// atomically rename to the final blob path.
fn download_blob(meta: &FileMeta, blob_path: &Path, display_name: &str) -> Result<()> {
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
        Err(e) => return Err(anyhow!("GET {} failed: {e}", meta.url)),
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

    loop {
        let n = reader.read(&mut buf).context("read from network")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write to disk")?;
        downloaded += n as u64;
        if downloaded - last_report >= 8 * 1024 * 1024 {
            last_report = downloaded;
            report_progress(display_name, downloaded, total);
        }
    }
    file.flush().ok();
    drop(file);
    report_progress(display_name, downloaded, total);
    eprintln!();

    if let Some(total) = total {
        let got = fs::metadata(&incomplete).map(|m| m.len()).unwrap_or(0);
        if got != total {
            bail!("Incomplete download: got {got} of {total} bytes (partial kept for resume)");
        }
    }

    fs::rename(&incomplete, blob_path)
        .with_context(|| format!("Failed to finalize {}", blob_path.display()))?;
    Ok(())
}

fn report_progress(name: &str, downloaded: u64, total: Option<u64>) {
    let mb = downloaded as f64 / 1_000_000.0;
    match total {
        Some(t) if t > 0 => {
            let pct = (downloaded as f64 / t as f64 * 100.0) as u32;
            eprint!(
                "\rDownloading {name}: {:.0}/{:.0} MB ({pct}%)",
                mb,
                t as f64 / 1_000_000.0
            );
        }
        _ => eprint!("\rDownloading {name}: {:.0} MB", mb),
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
}
