//! What a frontend hands a turn.
//!
//! A user turn is not always text. `UserInput` is the whole of what a frontend
//! collected — the prompt plus whatever the user attached to it — and it is what
//! [`crate::runtime::run_turn`] pushes onto history. `From<String>` keeps the
//! text-only case a caller can still write as a string literal, which is what
//! most of them are.
//!
//! Two frontends fill it, from two different shapes:
//!
//! - the REPL reads one line and parses `@image:<path>` markers out of it
//!   ([`parse_line`]), because a line of stdin is all it gets;
//! - the app-server is handed structured items and reads the `image` ones
//!   ([`image_from_data_url`]).
//!
//! **Audio is absent, but not out of reach.** There is no `AudioContent`, no
//! `ToolContent::Audio`, and no provider wired to accept one, so a marker for it
//! would be a promise nothing keeps. The route exists though: llama.cpp's `mtmd`
//! carries audio as well as images, and `llama-cpp-2` already wraps it
//! (`MtmdBitmap::from_audio_data`, `MtmdContext::support_audio`) behind a
//! feature gallium does not enable. `testsuite/testcases/multimodal_audio`
//! records the gap as a failing test rather than as a comment.

use std::path::{Path, PathBuf};

use base64::Engine;

use crate::llm::ImageContent;
use crate::AgentError;

/// The attachment marker a REPL line uses: `@image:<path>`.
///
/// Recognized only at a whitespace boundary, so `user@image:host` — or any
/// other mid-word occurrence — stays the text the user typed.
pub const IMAGE_MARKER: &str = "@image:";

/// One user turn: the text, and the images attached to it.
#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub text: String,
    pub images: Vec<ImageContent>,
}

impl UserInput {
    /// A turn with no attachments.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    /// Nothing to send: no text *and* no attachments. An image with no caption
    /// is still a turn, which is why this is not just `text.is_empty()`.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.images.is_empty()
    }
}

impl From<String> for UserInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for UserInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// Parse one line of REPL input, loading any `@image:<path>` attachments.
///
/// Relative paths resolve against `base` — the agent's working directory — not
/// the process cwd, so the same line means the same thing however the binary
/// was launched.
///
/// A path containing spaces is written in double quotes: `@image:"my shot.png"`.
///
/// Fails rather than dropping the marker: an attachment that silently did not
/// load looks exactly like a model that cannot see, which is the one thing this
/// is here to tell apart.
pub fn parse_line(line: &str, base: &Path) -> Result<UserInput, AgentError> {
    let mut text = String::new();
    let mut images = Vec::new();
    let mut rest = line;

    while let Some(at) = rest.find(IMAGE_MARKER) {
        // Mid-word: not a marker. Keep it, and resume scanning after it so the
        // same occurrence is not found forever.
        if at > 0 && !rest[..at].ends_with(char::is_whitespace) {
            let split = at + IMAGE_MARKER.len();
            text.push_str(&rest[..split]);
            rest = &rest[split..];
            continue;
        }

        text.push_str(&rest[..at]);
        let (spec, tail) = take_path(&rest[at + IMAGE_MARKER.len()..]);
        rest = tail;

        if spec.is_empty() {
            return Err(AgentError::InvalidInput(format!(
                "{IMAGE_MARKER} needs a path, e.g. {IMAGE_MARKER}shot.png"
            )));
        }
        images.push(load_image(&resolve(base, spec))?);
    }
    text.push_str(rest);

    Ok(UserInput {
        text: text.trim().to_string(),
        images,
    })
}

/// Split the path off the front of `s`, returning it and what follows.
///
/// A leading `"` runs to the closing quote; otherwise the path ends at the
/// first whitespace.
fn take_path(s: &str) -> (&str, &str) {
    match s.strip_prefix('"') {
        Some(quoted) => match quoted.find('"') {
            Some(end) => (&quoted[..end], &quoted[end + 1..]),
            // Unterminated quote: take the rest of the line rather than
            // guessing where the user meant it to end.
            None => (quoted, ""),
        },
        None => match s.find(char::is_whitespace) {
            Some(end) => (&s[..end], &s[end..]),
            None => (s, ""),
        },
    }
}

fn resolve(base: &Path, spec: &str) -> PathBuf {
    let path = Path::new(spec);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// Read an image off disk into the base64 an LLM request carries.
pub fn load_image(path: &Path) -> Result<ImageContent, AgentError> {
    let media_type = media_type_for(path).ok_or_else(|| {
        AgentError::InvalidInput(format!(
            "{}: unsupported image type (png, jpeg, gif, webp)",
            path.display()
        ))
    })?;
    let bytes = std::fs::read(path)
        .map_err(|e| AgentError::InvalidInput(format!("{}: {e}", path.display())))?;
    Ok(ImageContent {
        base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        media_type: media_type.to_string(),
    })
}

/// The media type an extension implies, or `None` for one we do not carry.
///
/// Deliberately a short list: these are the four every vision provider in reach
/// accepts, and guessing wrong sends bytes a model will reject with a message
/// far less clear than this one.
pub fn media_type_for(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    }
}

/// Parse `data:image/png;base64,<payload>` — the shape an app-server client
/// sends an image in.
///
/// Only `base64` data URLs are accepted. A remote `https://` image would mean
/// this process fetching a URL a client chose, which is a different decision
/// than carrying bytes the client already had.
pub fn image_from_data_url(url: &str) -> Option<ImageContent> {
    let spec = url.strip_prefix("data:")?;
    let (meta, payload) = spec.split_once(',')?;
    let media_type = meta.strip_suffix(";base64")?;
    if !media_type.starts_with("image/") {
        return None;
    }
    Some(ImageContent {
        base64: payload.to_string(),
        media_type: media_type.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1×1 PNG, the smallest thing that is really a file on disk.
    const PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    fn fixture(name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, PNG).unwrap();
        (dir, path)
    }

    #[test]
    fn a_plain_line_is_text_with_no_attachments() {
        let input = parse_line("what is 2 + 2?", Path::new("/tmp")).unwrap();
        assert_eq!(input.text, "what is 2 + 2?");
        assert!(input.images.is_empty());
    }

    #[test]
    fn a_marker_is_lifted_out_of_the_text() {
        let (dir, _) = fixture("shot.png");
        let input = parse_line("@image:shot.png what is this?", dir.path()).unwrap();
        assert_eq!(input.text, "what is this?");
        assert_eq!(input.images.len(), 1);
        assert_eq!(input.images[0].media_type, "image/png");
        assert!(!input.images[0].base64.is_empty());
    }

    #[test]
    fn a_marker_may_come_after_the_text() {
        let (dir, _) = fixture("shot.png");
        let input = parse_line("describe @image:shot.png please", dir.path()).unwrap();
        assert_eq!(input.text, "describe  please");
        assert_eq!(input.images.len(), 1);
    }

    #[test]
    fn several_markers_attach_several_images() {
        let (dir, _) = fixture("a.png");
        std::fs::write(dir.path().join("b.jpg"), PNG).unwrap();
        let input = parse_line("@image:a.png @image:b.jpg compare", dir.path()).unwrap();
        assert_eq!(input.text, "compare");
        assert_eq!(input.images.len(), 2);
        assert_eq!(input.images[1].media_type, "image/jpeg");
    }

    #[test]
    fn a_quoted_path_may_contain_spaces() {
        let (dir, _) = fixture("my shot.png");
        let input = parse_line("@image:\"my shot.png\" hello", dir.path()).unwrap();
        assert_eq!(input.text, "hello");
        assert_eq!(input.images.len(), 1);
    }

    /// The whole point of requiring a whitespace boundary.
    #[test]
    fn a_mid_word_marker_is_left_as_text() {
        let input = parse_line("mail user@image:host about it", Path::new("/tmp")).unwrap();
        assert_eq!(input.text, "mail user@image:host about it");
        assert!(input.images.is_empty());
    }

    #[test]
    fn an_absolute_path_ignores_the_base() {
        let (_dir, path) = fixture("shot.png");
        let line = format!("@image:{} look", path.display());
        let input = parse_line(&line, Path::new("/nonexistent")).unwrap();
        assert_eq!(input.images.len(), 1);
    }

    /// Loud, not silent: a dropped attachment is indistinguishable from a model
    /// that cannot see.
    #[test]
    fn a_missing_file_is_an_error() {
        let err = parse_line("@image:gone.png hi", Path::new("/tmp")).unwrap_err();
        assert!(matches!(err, AgentError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn an_unsupported_extension_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("clip.wav"), b"RIFF").unwrap();
        let err = parse_line("@image:clip.wav hi", dir.path()).unwrap_err();
        assert!(matches!(err, AgentError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn a_marker_with_no_path_is_an_error() {
        let err = parse_line("@image: hi", Path::new("/tmp")).unwrap_err();
        assert!(matches!(err, AgentError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn a_data_url_round_trips() {
        let img = image_from_data_url("data:image/png;base64,AAAA").unwrap();
        assert_eq!(img.media_type, "image/png");
        assert_eq!(img.base64, "AAAA");
    }

    #[test]
    fn a_non_image_data_url_is_refused() {
        assert!(image_from_data_url("data:audio/wav;base64,AAAA").is_none());
        assert!(image_from_data_url("https://example.com/a.png").is_none());
        assert!(image_from_data_url("data:image/png,notbase64").is_none());
    }

    #[test]
    fn an_image_alone_is_not_an_empty_turn() {
        let (dir, _) = fixture("shot.png");
        let input = parse_line("@image:shot.png", dir.path()).unwrap();
        assert!(input.text.is_empty());
        assert!(!input.is_empty());
    }
}
