//! What a project tells an agent about itself.
//!
//! A checkout usually already carries instructions for coding agents —
//! `AGENTS.md` or `CLAUDE.md` at its root, holding the build commands, the
//! layout, and the traps. Without them the model rediscovers that by reading
//! `README.md` every session, or asks. Reading the file the project already
//! wrote is cheaper than either.
//!
//! `../klein-cli` does the same thing (`InjectContextFile`), and this matches
//! its precedence deliberately: the two tools run in the same checkouts, and a
//! project that tuned its `AGENTS.md` for one should get the same behavior from
//! the other.

use std::path::{Path, PathBuf};

/// The candidates, in order. The first that exists and has content wins —
/// they are alternatives, not layers, so nothing is concatenated.
const CANDIDATES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];

/// A project instruction file that was found and read.
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

impl ContextFile {
    /// The system message this becomes. Headed, because it arrives as a second
    /// system message and the model should be able to tell the operator's
    /// instructions from the project's.
    pub fn as_system_message(&self) -> String {
        format!("# Project Context\n\n{}", self.content)
    }
}

/// Find the project's instruction file in `working_dir`.
///
/// Only the working directory is searched — no walk up to a repo root. A parent
/// directory's instructions are a guess about what the user meant; the
/// directory they launched in is not.
///
/// A file that exists but is blank is treated as absent, and the next candidate
/// is still tried: an empty `AGENTS.md` should not shadow a real `CLAUDE.md`.
pub fn find_context_file(working_dir: &Path) -> Option<ContextFile> {
    for name in CANDIDATES {
        let path = working_dir.join(name);
        match std::fs::read_to_string(&path) {
            Ok(content) if !content.trim().is_empty() => {
                return Some(ContextFile { path, content })
            }
            Ok(_) => tracing::debug!("ignoring empty context file {:?}", path),
            Err(_) => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn a_project_with_neither_file_has_no_context() {
        let dir = dir_with(&[("README.md", "not an instruction file")]);
        assert!(find_context_file(dir.path()).is_none());
    }

    #[test]
    fn claude_md_is_used_when_it_is_the_only_one() {
        let dir = dir_with(&[("CLAUDE.md", "build with make")]);
        let found = find_context_file(dir.path()).unwrap();
        assert_eq!(found.content, "build with make");
        assert!(found.path.ends_with("CLAUDE.md"));
    }

    /// The two are alternatives. A project carrying both means one tool wrote
    /// each, and `AGENTS.md` is the one this agrees with klein-cli on.
    #[test]
    fn agents_md_wins_when_both_exist() {
        let dir = dir_with(&[
            ("AGENTS.md", "the agents one"),
            ("CLAUDE.md", "the claude one"),
        ]);
        let found = find_context_file(dir.path()).unwrap();
        assert_eq!(found.content, "the agents one");
    }

    /// An empty `AGENTS.md` — a placeholder someone committed and never filled
    /// in — must not shadow a `CLAUDE.md` that says something.
    #[test]
    fn an_empty_candidate_does_not_shadow_a_real_one() {
        let dir = dir_with(&[
            ("AGENTS.md", "\n  \n"),
            ("CLAUDE.md", "the real instructions"),
        ]);
        let found = find_context_file(dir.path()).unwrap();
        assert_eq!(found.content, "the real instructions");
    }

    #[test]
    fn the_system_message_is_labelled_as_project_context() {
        let dir = dir_with(&[("AGENTS.md", "run the tests")]);
        let message = find_context_file(dir.path()).unwrap().as_system_message();
        assert!(message.starts_with("# Project Context\n\n"));
        assert!(message.contains("run the tests"));
    }
}
