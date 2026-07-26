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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// What a running turn carries with it, beyond its messages and tools.
///
/// One field today. It exists as a struct rather than as a bare token because
/// the approval policy and workspace scope of #14 belong here next, and every
/// call site this threads through would otherwise have to change again to
/// gain them.
#[derive(Debug, Clone, Default)]
pub struct TurnContext {
    pub cancellation: CancellationToken,
}

impl TurnContext {
    /// A context for a turn nobody can stop: tests, `mcp_server`, and any
    /// caller that has no cancel button to offer.
    pub fn detached() -> Self {
        Self::default()
    }

    pub fn new(cancellation: CancellationToken) -> Self {
        Self { cancellation }
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
