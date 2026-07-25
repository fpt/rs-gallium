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
///
/// Deliberately limited to what the agent actually emits today. Token deltas
/// are the obvious omission: no provider streams yet — the OpenAI path uses the
/// blocking Responses API and both local backends return a finished string — so
/// a `MessageDelta` variant would be a promise nothing keeps. It belongs here
/// once a provider can produce it.
#[derive(Debug)]
pub enum AgentEvent<'a> {
    /// The model asked to call a tool.
    ToolStarted {
        call_id: &'a str,
        name: &'a str,
        arguments: &'a Value,
    },
    /// A tool returned. The whole result is passed so a consumer can choose
    /// between the model's text and the display form, and can see `is_error`.
    ToolCompleted {
        call_id: &'a str,
        name: &'a str,
        result: &'a ToolResult,
    },
    /// An LLM call reported token usage.
    Usage { usage: &'a TokenUsage },
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
