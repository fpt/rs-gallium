//! Stopping a turn that is already running.
//!
//! A turn is a stack of blocking calls — generation, a shell child, an MCP
//! round trip — so there is no single place to interrupt. What there is instead
//! is a flag every one of those places already returns to often enough to
//! notice: between sampled tokens, between polls of a child process, between
//! ReAct iterations. [`CancellationToken`] is that flag, and [`TurnContext`] is
//! how it reaches the code that has to check it.
//!
//! Cancelling is therefore prompt, not instantaneous, and what "prompt" means
//! depends on where the turn is:
//!
//! | Where the turn is | What cancelling does |
//! |---|---|
//! | Between ReAct iterations | stops before the next model call |
//! | Generating, native candle or llama.cpp | stops after the current token |
//! | Running `bash` | kills the command's process group within a poll interval |
//! | Waiting on an MCP tool | stops waiting; the request itself finishes unread |
//! | Inside an OpenAI HTTP round trip | nothing until the response lands |
//!
//! The last two are honest limits rather than oversights. A blocking `ureq`
//! request has no interruption point, and an MCP server's stdio pipe cannot be
//! read with a deadline portably; in both cases the turn stops at the next
//! check instead of the call being torn out from under the peer.
//!
//! [`SteerInbox`] is here for the same reason and reaches the turn the same way:
//! it is the other thing a frontend can say to a turn already in flight, and
//! saying it means writing to a cell the loop reads at those same boundaries.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::llm::ChatMessage;
use crate::AgentError;

/// A shared "stop now" flag for one turn.
///
/// Cloning shares the flag rather than copying it, so a frontend can hold one
/// clone to cancel with while the turn holds another to check.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the turn to stop. Idempotent; safe from any thread.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// `Err(AgentError::Cancelled)` once cancelled, so a check is one `?`.
    pub fn check(&self) -> Result<(), AgentError> {
        if self.is_cancelled() {
            return Err(AgentError::Cancelled);
        }
        Ok(())
    }
}

/// Text the user added while the turn was already running, waiting to be handed
/// to the model.
///
/// The mirror image of [`CancellationToken`]: the same shared cell reached from
/// the same two sides, saying "here is more to go on" instead of "stop". It has
/// to be a cell rather than a direct write into the turn's history because the
/// turn holds that history for its whole duration — the app-server's `run_turn`
/// locks `Thread::messages` and does not let go until the turn ends, so a
/// steering request arriving mid-turn has nowhere else to put its text.
///
/// Steering is therefore prompt in the same qualified sense cancelling is: the
/// text lands at the next ReAct boundary, once the current generation and the
/// tool calls it asked for are done. Unlike cancelling, it cannot be finer than
/// that — a model is mid-sentence and there is no way to hand it a new
/// instruction except by ending the sentence and asking again.
///
/// The inbox closes when the turn stops reading it, and a push to a closed inbox
/// is refused rather than accepted and dropped. That is what makes the app-server
/// able to answer `turn/steer` honestly: the alternative is a check ("is anything
/// pending?") and a decision ("then this turn is over") made as two steps, with a
/// steer able to land in between — acknowledged to the client, and read by nobody.
#[derive(Debug, Clone, Default)]
pub struct SteerInbox {
    inner: Arc<Mutex<Inbox>>,
}

#[derive(Debug, Default)]
struct Inbox {
    pending: Vec<String>,
    /// Set once the turn has stopped reading. One-way: a turn that has finished
    /// with its inbox never goes back to it.
    closed: bool,
}

impl SteerInbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue text for the running turn. Idempotent in no sense at all: two
    /// pushes are two messages, in the order they arrived.
    ///
    /// `false` means the turn has already stopped reading and the text was *not*
    /// queued. A caller holding a client's request must refuse it rather than
    /// acknowledge it — the whole point of the flag is that "accepted" should
    /// mean the model will see it.
    #[must_use]
    pub fn push(&self, text: String) -> bool {
        let mut inbox = self.inner.lock();
        if inbox.closed {
            return false;
        }
        inbox.pending.push(text);
        true
    }

    pub fn has_pending(&self) -> bool {
        !self.inner.lock().pending.is_empty()
    }

    /// Stop reading, unless something is waiting to be read.
    ///
    /// `false` means the caller may not stop: a steer arrived and has not been
    /// delivered. This is the check-and-decide the ReAct loop makes when a model
    /// response would end the turn, as one atomic step — `has_pending` followed
    /// by a decision to return would leave exactly the gap this closes.
    pub fn finish(&self) -> bool {
        let mut inbox = self.inner.lock();
        if !inbox.pending.is_empty() {
            return false;
        }
        inbox.closed = true;
        true
    }

    /// Stop reading whatever is in the inbox.
    ///
    /// For the endings that are not the turn choosing to stop — a failure, a
    /// cancellation, a loop out of iterations. Anything pending is discarded
    /// with the rest of the turn, which is rolled back to before it started.
    pub fn close(&self) {
        self.inner.lock().closed = true;
    }

    /// Take everything queued so far, leaving the inbox empty.
    ///
    /// Draining rather than peeking is what keeps a steer from being delivered
    /// twice: the loop reads at more than one boundary, and a message the model
    /// has already been given is settled conversation.
    pub fn drain(&self) -> Vec<ChatMessage> {
        std::mem::take(&mut self.inner.lock().pending)
            .into_iter()
            .map(ChatMessage::user)
            .collect()
    }
}

/// What a running turn carries with it, beyond its messages and tools.
///
/// It exists as a struct rather than as a bare token so that what a turn needs
/// can grow without every call site it threads through changing again — which
/// is what the trace recorder did: it reaches the ReAct loop through here, and
/// no signature moved to let it.
#[derive(Debug, Clone, Default)]
pub struct TurnContext {
    pub cancellation: CancellationToken,
    /// Text the user added mid-turn, drained by the ReAct loop at each boundary.
    /// A frontend with no way to speak mid-turn simply never pushes to it.
    pub steer: SteerInbox,
    /// Records what the turn does, when someone asked for a trace. `None` — the
    /// default — records nothing and costs one branch per step.
    pub trace: Option<Arc<crate::trace::TurnRecorder>>,
}

impl TurnContext {
    /// A context for a turn nobody can stop: tests, `mcp_server`, and any
    /// caller that has no cancel button to offer.
    pub fn detached() -> Self {
        Self::default()
    }

    pub fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation,
            steer: SteerInbox::new(),
            trace: None,
        }
    }

    /// The same context, steered through `inbox`. The app-server hands in the
    /// inbox `turn/steer` pushes to; a caller that has no way to speak mid-turn
    /// leaves the default, which nothing ever writes to.
    pub fn with_steering(mut self, inbox: SteerInbox) -> Self {
        self.steer = inbox;
        self
    }

    /// The same context, recording into `recorder`. `runtime::run_turn` attaches
    /// it, so a frontend that wants traces configures a destination rather than
    /// assembling a recorder itself.
    pub fn with_trace(mut self, recorder: Arc<crate::trace::TurnRecorder>) -> Self {
        self.trace = Some(recorder);
        self
    }

    /// Shorthand for the check that appears at every loop boundary.
    pub fn check(&self) -> Result<(), AgentError> {
        self.cancellation.check()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

/// How often a blocking wait comes up for air to check the token.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// Run a blocking call on a worker thread, and stop *waiting* for it when the
/// turn is cancelled.
///
/// This is for calls into a peer we do not control — an MCP server's stdio
/// pipe, an HTTP round trip, a request back to the driving client. None of them
/// can be interrupted: there is no portable way to read a pipe with a deadline,
/// and tearing a request out from under a peer mid-flight would leave it
/// answering into a socket nobody is reading.
///
/// So the call is not cancelled — the *wait* is. On cancellation the worker is
/// left to finish and its answer is dropped. That costs one abandoned thread
/// and, for MCP, a transport lock held until the server replies, which delays
/// the next call to *that server* rather than the turn. The alternative is a
/// turn that cannot be stopped while a slow server thinks about it.
pub(crate) fn wait_cancellable<T: Send + 'static>(
    ctx: &TurnContext,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, AgentError> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // A send failure means the waiter gave up; the result is simply dropped.
        let _ = tx.send(work());
    });

    loop {
        match rx.recv_timeout(POLL_INTERVAL) {
            Ok(value) => return Ok(value),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => ctx.check()?,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(AgentError::InternalError(
                    "tool worker stopped without answering".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clone_shares_the_flag_rather_than_copying_it() {
        let token = CancellationToken::new();
        let held_by_the_turn = token.clone();
        assert!(held_by_the_turn.check().is_ok());

        token.cancel();

        assert!(held_by_the_turn.is_cancelled());
        assert!(matches!(
            held_by_the_turn.check(),
            Err(AgentError::Cancelled)
        ));
    }

    #[test]
    fn cancelling_twice_is_the_same_as_cancelling_once() {
        let token = CancellationToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn a_clone_of_the_inbox_delivers_to_the_same_turn() {
        let inbox = SteerInbox::new();
        let held_by_the_turn = inbox.clone();

        assert!(inbox.push("actually, use tabs".to_string()));

        assert!(held_by_the_turn.has_pending());
        let drained = held_by_the_turn.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].role, crate::llm::ChatRole::User);
        assert_eq!(drained[0].content, "actually, use tabs");
    }

    /// Two boundaries read the inbox, so a message the model has already been
    /// given must not come back at the next one.
    #[test]
    fn draining_empties_the_inbox() {
        let inbox = SteerInbox::new();
        assert!(inbox.push("first".to_string()));
        assert!(inbox.push("second".to_string()));

        let drained = inbox.drain();

        assert_eq!(drained.len(), 2, "in the order they arrived");
        assert_eq!(drained[0].content, "first");
        assert_eq!(drained[1].content, "second");
        assert!(!inbox.has_pending());
        assert!(inbox.drain().is_empty());
    }

    /// The race the closed flag exists for, from the inbox's side: a turn only
    /// gets to stop reading when there is nothing left to read.
    #[test]
    fn an_inbox_with_a_message_in_it_refuses_to_stop_reading() {
        let inbox = SteerInbox::new();
        assert!(inbox.push("wait".to_string()));

        assert!(
            !inbox.finish(),
            "a turn must not stop reading over a message nobody has been given"
        );

        assert_eq!(inbox.drain().len(), 1);
        assert!(inbox.finish(), "and may stop once the inbox is empty");
    }

    /// The other side of it: once the turn has stopped reading, a steer is
    /// refused rather than accepted into a cell nobody will look at again.
    #[test]
    fn a_closed_inbox_refuses_a_push() {
        let inbox = SteerInbox::new();
        assert!(inbox.finish());

        assert!(!inbox.push("too late".to_string()));
        assert!(!inbox.has_pending());
    }

    /// A turn that failed or was cancelled stops reading whatever is in the
    /// inbox — its history is rolled back, so there is nothing to deliver into.
    #[test]
    fn closing_stops_reading_even_with_something_pending() {
        let inbox = SteerInbox::new();
        assert!(inbox.push("never delivered".to_string()));

        inbox.close();

        assert!(!inbox.push("nor this".to_string()));
    }

    #[test]
    fn a_detached_context_never_stops_a_turn() {
        let ctx = TurnContext::detached();
        assert!(ctx.check().is_ok());
        assert!(!ctx.is_cancelled());
    }

    #[test]
    fn a_wait_that_finishes_returns_its_value() {
        let ctx = TurnContext::detached();
        assert_eq!(wait_cancellable(&ctx, || 7).unwrap(), 7);
    }

    /// The peer is still working when the turn is cancelled. The waiter has to
    /// come back promptly rather than sit on a call it no longer wants.
    #[test]
    fn a_cancelled_wait_gives_up_without_waiting_for_the_peer() {
        let ctx = TurnContext::new(CancellationToken::new());
        let token = ctx.cancellation.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(60));
            token.cancel();
        });

        let started = std::time::Instant::now();
        let result = wait_cancellable(&ctx, || {
            std::thread::sleep(std::time::Duration::from_secs(30));
            "far too late"
        });

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "gave up after {:?}, which is not promptly",
            started.elapsed()
        );
    }
}
