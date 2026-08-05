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
//! Audio works the same way, through `@audio:`. It reaches a model only on the
//! llama.cpp backend with a projector that has an audio encoder — which every
//! Gemma 4 ships. There is still no `ToolContent::Audio`, so a *tool* cannot
//! produce a clip; this is user input only.

use std::path::{Path, PathBuf};

use base64::Engine;

use crate::llm::{AudioContent, ImageContent, MediaContent};
use crate::AgentError;

/// The attachment markers a REPL line uses: `@image:<path>` and `@audio:<path>`.
///
/// Recognized only at a whitespace boundary, so `user@image:host` — or any
/// other mid-word occurrence — stays the text the user typed.
pub const IMAGE_MARKER: &str = "@image:";
pub const AUDIO_MARKER: &str = "@audio:";

/// One user turn: the text, and whatever was attached to it.
///
/// `media` is a single ordered list rather than a vec per modality, so
/// `@audio:note.wav @image:shot.png` reaches the model in the order it was
/// typed. Two vecs would force a reassembly order downstream, silently
/// rewriting a prompt whose sequence may be the point.
#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub text: String,
    pub media: Vec<MediaContent>,
}

impl UserInput {
    /// A turn with no attachments.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            media: Vec::new(),
        }
    }

    /// Nothing to send: no text *and* no attachments. An image with no caption
    /// is still a turn, which is why this is not just `text.is_empty()`.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.media.is_empty()
    }

    /// How many attachments of any kind this turn carries.
    pub fn media_count(&self) -> usize {
        self.media.len()
    }

    /// The images among them, in order.
    pub fn images(&self) -> impl Iterator<Item = &ImageContent> {
        self.media.iter().filter_map(|m| match m {
            MediaContent::Image(i) => Some(i),
            MediaContent::Audio(_) => None,
        })
    }

    /// The audio clips among them, in order.
    pub fn audio(&self) -> impl Iterator<Item = &AudioContent> {
        self.media.iter().filter_map(|m| match m {
            MediaContent::Audio(a) => Some(a),
            MediaContent::Image(_) => None,
        })
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
    let mut media = Vec::new();
    let mut rest = line;

    // Scan for whichever marker comes first, so `@image:a.png @audio:b.wav` and
    // the reverse both work and each attachment keeps its place in the line.
    while let Some((at, marker)) = next_marker(rest) {
        // Mid-word: not a marker. Keep it, and resume scanning after it so the
        // same occurrence is not found forever.
        if at > 0 && !rest[..at].ends_with(char::is_whitespace) {
            let split = at + marker.len();
            text.push_str(&rest[..split]);
            rest = &rest[split..];
            continue;
        }

        text.push_str(&rest[..at]);
        let (spec, tail) = take_path(&rest[at + marker.len()..]);
        rest = tail;

        if spec.is_empty() {
            return Err(AgentError::InvalidInput(format!(
                "{marker} needs a path, e.g. {marker}{}",
                if marker == IMAGE_MARKER {
                    "shot.png"
                } else {
                    "clip.wav"
                }
            )));
        }
        // Pushed onto one list as the scan reaches it, so the order in the
        // line survives to the model.
        let path = resolve(base, spec);
        media.push(if marker == IMAGE_MARKER {
            MediaContent::Image(load_image(&path)?)
        } else {
            MediaContent::Audio(load_audio(&path)?)
        });
    }
    text.push_str(rest);

    Ok(UserInput {
        text: text.trim().to_string(),
        media,
    })
}

/// The earliest attachment marker in `s`, and which one it is.
fn next_marker(s: &str) -> Option<(usize, &'static str)> {
    let image = s.find(IMAGE_MARKER).map(|at| (at, IMAGE_MARKER));
    let audio = s.find(AUDIO_MARKER).map(|at| (at, AUDIO_MARKER));
    match (image, audio) {
        (Some(i), Some(a)) => Some(if i.0 <= a.0 { i } else { a }),
        (found, None) | (None, found) => found,
    }
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

/// Read an audio clip off disk into the base64 an LLM request carries.
pub fn load_audio(path: &Path) -> Result<AudioContent, AgentError> {
    let media_type = audio_type_for(path).ok_or_else(|| {
        AgentError::InvalidInput(format!(
            "{}: unsupported audio type (wav, mp3, flac)",
            path.display()
        ))
    })?;
    let bytes = std::fs::read(path)
        .map_err(|e| AgentError::InvalidInput(format!("{}: {e}", path.display())))?;
    Ok(AudioContent {
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

/// The audio media type an extension implies, or `None` for one we do not carry.
///
/// The three llama.cpp's bundled miniaudio decodes. Anything else would reach
/// the projector as bytes it cannot read, and fail further from the cause.
pub fn audio_type_for(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "wav" => Some("audio/wav"),
        "mp3" => Some("audio/mpeg"),
        "flac" => Some("audio/flac"),
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
        assert!(input.images().next().is_none());
    }

    #[test]
    fn a_marker_is_lifted_out_of_the_text() {
        let (dir, _) = fixture("shot.png");
        let input = parse_line("@image:shot.png what is this?", dir.path()).unwrap();
        assert_eq!(input.text, "what is this?");
        assert_eq!(input.images().count(), 1);
        assert_eq!(input.images().next().unwrap().media_type, "image/png");
        assert!(!input.images().next().unwrap().base64.is_empty());
    }

    #[test]
    fn a_marker_may_come_after_the_text() {
        let (dir, _) = fixture("shot.png");
        let input = parse_line("describe @image:shot.png please", dir.path()).unwrap();
        assert_eq!(input.text, "describe  please");
        assert_eq!(input.images().count(), 1);
    }

    #[test]
    fn several_markers_attach_several_images() {
        let (dir, _) = fixture("a.png");
        std::fs::write(dir.path().join("b.jpg"), PNG).unwrap();
        let input = parse_line("@image:a.png @image:b.jpg compare", dir.path()).unwrap();
        assert_eq!(input.text, "compare");
        assert_eq!(input.images().count(), 2);
        assert_eq!(input.images().nth(1).unwrap().media_type, "image/jpeg");
    }

    #[test]
    fn a_quoted_path_may_contain_spaces() {
        let (dir, _) = fixture("my shot.png");
        let input = parse_line("@image:\"my shot.png\" hello", dir.path()).unwrap();
        assert_eq!(input.text, "hello");
        assert_eq!(input.images().count(), 1);
    }

    /// The whole point of requiring a whitespace boundary.
    #[test]
    fn a_mid_word_marker_is_left_as_text() {
        let input = parse_line("mail user@image:host about it", Path::new("/tmp")).unwrap();
        assert_eq!(input.text, "mail user@image:host about it");
        assert!(input.images().next().is_none());
    }

    #[test]
    fn an_absolute_path_ignores_the_base() {
        let (_dir, path) = fixture("shot.png");
        let line = format!("@image:{} look", path.display());
        let input = parse_line(&line, Path::new("/nonexistent")).unwrap();
        assert_eq!(input.images().count(), 1);
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

    /// A minimal RIFF/WAVE header — enough that the file is what its extension
    /// claims. gallium never decodes it; llama.cpp does.
    const WAV: &[u8] = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00";

    #[test]
    fn an_audio_marker_attaches_a_clip() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("clip.wav"), WAV).unwrap();
        let input = parse_line("@audio:clip.wav transcribe this", dir.path()).unwrap();
        assert_eq!(input.text, "transcribe this");
        assert!(input.images().next().is_none());
        assert_eq!(input.audio().count(), 1);
        assert_eq!(input.audio().next().unwrap().media_type, "audio/wav");
    }

    /// The two markers share one scanner, so their order in the line has to be
    /// the order they are collected in — mtmd pairs media with markers
    /// positionally, so a reordering here reaches the model as a different
    /// prompt.
    ///
    /// Asserted in **both** directions: an implementation that keeps images and
    /// audio in separate vecs passes one and fails the other, which is exactly
    /// the bug this shape was chosen to make impossible.
    #[test]
    fn markers_keep_the_order_they_were_written_in() {
        let (dir, _) = fixture("shot.png");
        std::fs::write(dir.path().join("clip.wav"), WAV).unwrap();

        let audio_first = parse_line("@audio:clip.wav @image:shot.png both", dir.path()).unwrap();
        assert_eq!(audio_first.text, "both");
        assert_eq!(
            audio_first
                .media
                .iter()
                .map(|m| m.kind())
                .collect::<Vec<_>>(),
            vec!["audio", "image"]
        );

        let image_first = parse_line("@image:shot.png @audio:clip.wav both", dir.path()).unwrap();
        assert_eq!(
            image_first
                .media
                .iter()
                .map(|m| m.kind())
                .collect::<Vec<_>>(),
            vec!["image", "audio"]
        );

        assert_eq!(audio_first.media_count(), 2);
    }

    /// Interleaved with text, and more than two: position is the only thing
    /// tying an attachment to its marker.
    #[test]
    fn many_mixed_markers_keep_their_sequence() {
        let (dir, _) = fixture("a.png");
        std::fs::write(dir.path().join("b.wav"), WAV).unwrap();
        std::fs::write(dir.path().join("c.jpg"), PNG).unwrap();
        let input = parse_line(
            "look @audio:b.wav then @image:a.png and @image:c.jpg done",
            dir.path(),
        )
        .unwrap();
        assert_eq!(
            input.media.iter().map(|m| m.kind()).collect::<Vec<_>>(),
            vec!["audio", "image", "image"]
        );
        assert_eq!(
            input
                .media
                .iter()
                .map(|m| m.media_type())
                .collect::<Vec<_>>(),
            vec!["audio/wav", "image/png", "image/jpeg"]
        );
    }

    #[test]
    fn an_unsupported_audio_extension_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("clip.ogg"), WAV).unwrap();
        let err = parse_line("@audio:clip.ogg hi", dir.path()).unwrap_err();
        assert!(matches!(err, AgentError::InvalidInput(_)), "{err:?}");
    }

    /// An image path handed to `@audio:` is caught by extension, before any
    /// backend has to puzzle over the bytes.
    #[test]
    fn an_image_given_to_the_audio_marker_is_refused() {
        let (dir, _) = fixture("shot.png");
        let err = parse_line("@audio:shot.png hi", dir.path()).unwrap_err();
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
