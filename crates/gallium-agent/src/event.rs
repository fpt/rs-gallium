//! The event stream every frontend consumes.
//!
//! Before this, the only progress type was `ReactEvent`, defined inside the
//! ReAct loop with a single implementor in the app-server; the REPL rendered
//! nothing until a turn had finished. Each frontend translating agent progress
//! its own way is what this replaces: the app-server's `item/*` notifications,
//! the CLI's live output, and later a TTS renderer are all translations of this
//! one stream.
//!
//! Events borrow rather than own. They are emitted on the turn's own thread and
//! every consumer today formats them immediately, so allocating a `String` per
//! tool call would be pure waste. A consumer that needs to outlive the call —
//! queueing for playback, say — copies what it needs.

use serde_json::Value;

use crate::llm::TokenUsage;
use crate::tool::ToolResult;

/// Something that happened partway through a turn, reported as it occurs.
#[derive(Debug)]
pub enum AgentEvent<'a> {
    /// A fragment of the answer the model is producing *right now*, streamed as
    /// it decodes. Only the native candle backend emits these today
    /// (`LlmProvider::chat_with_tools_streaming`); the OpenAI blocking API and
    /// the llama.cpp backend still deliver their answer in one piece, so a
    /// consumer must treat deltas as best-effort — the authoritative text is
    /// always the [`TurnCompleted`](Self::TurnCompleted) / steered
    /// [`AgentMessage`](Self::AgentMessage) that follows, which a renderer
    /// should let *replace* whatever it accumulated from deltas. Reasoning and
    /// half-formed tool-call syntax are filtered out before a fragment is
    /// emitted; when the filter cannot be sure, it stops streaming for the rest
    /// of that model call rather than risk leaking either.
    MessageDelta { text: &'a str },
    /// The model asked to call a tool.
    ToolStarted {
        call_id: &'a str,
        name: &'a str,
        arguments: &'a Value,
    },
    /// A tool returned. The whole result is passed so a consumer can choose
    /// between the model's text and the display form, and can see `is_error`.
    ///
    /// `arguments` repeats what [`ToolStarted`](Self::ToolStarted) already
    /// carried, because a consumer that renders one item per call needs the
    /// finished item to describe itself: the app-server's protocol requires
    /// `arguments` on the completed item, and reconstructing it would mean
    /// every observer keeping its own table of calls in flight.
    ToolCompleted {
        call_id: &'a str,
        name: &'a str,
        arguments: &'a Value,
        result: &'a ToolResult,
    },
    /// An LLM call reported token usage.
    Usage { usage: &'a TokenUsage },
    /// The model produced text that is *not* the end of the turn: it had
    /// finished answering, and steering arrived to carry the turn on. Without
    /// this the answer would go straight into history unseen, which is the one
    /// case where steering would silently cost the user a reply.
    AgentMessage { text: &'a str },
    /// The turn produced its final text.
    TurnCompleted { text: &'a str },
    /// The turn failed.
    Error { message: &'a str },
}

/// Notified as a turn progresses.
///
/// Called on the turn's own thread, so an implementation that blocks stalls the
/// turn.
pub trait AgentObserver {
    fn on_event(&self, event: AgentEvent<'_>);
}

/// Convenience for the common `Option<&dyn AgentObserver>` plumbing.
pub(crate) fn emit(observer: Option<&dyn AgentObserver>, event: AgentEvent<'_>) {
    if let Some(observer) = observer {
        observer.on_event(event);
    }
}
