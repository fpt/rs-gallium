//! One turn, run the same way for every frontend.
//!
//! Each frontend used to assemble a turn by hand — compact, push the prompt,
//! decide what context to inject, call the ReAct loop, record usage, append the
//! reply. They drifted, as duplicated sequences do: the app-server never
//! injected the skill catalog and built its `SkillRegistry` empty, so
//! `lookup_skill` was advertised to the model and could never find anything.
//!
//! This is the single place a turn happens, so cancellation, approval policy,
//! and tracing land here once instead of once per frontend.

use std::sync::Arc;

use crate::cancel::TurnContext;
use crate::event::AgentObserver;
use crate::input::UserInput;
use crate::llm::{ChatMessage, ChatRole, LlmProvider, TokenUsage};
use crate::memory;
use crate::react;
use crate::skill::SkillRegistry;
use crate::tool::ToolAccess;
use crate::trace::{TraceSession, TurnEnding};
use crate::AgentError;

/// What a turn needs beyond the history it runs against.
pub struct TurnSetup<'a> {
    pub provider: &'a dyn LlmProvider,
    pub tools: &'a dyn ToolAccess,
    /// Catalogued into the prompt so the model knows which skills exist.
    /// `None` disables it.
    pub skills: Option<&'a SkillRegistry>,
    /// Max ReAct iterations; `None` uses the library default.
    pub max_iterations: Option<u32>,
    /// Compaction trigger, in tokens. `0` disables compaction.
    pub context_window: u32,
    pub observer: Option<&'a dyn AgentObserver>,
    /// The turn's cancellation. `None` is a turn nobody can stop, which is what
    /// a caller with no cancel button to offer should say.
    pub context: Option<&'a TurnContext>,
    /// Where to write a record of this turn. `None` writes none, which is the
    /// default: a trace holds everything the model saw, so it is something an
    /// operator asks for rather than something that happens to them.
    pub trace: Option<&'a TraceSession>,
    /// The caller's name for this turn, used in the trace and its filename. The
    /// app-server has one worth keeping; the REPL passes `None` and gets a
    /// sequence number.
    pub turn_id: Option<&'a str>,
}

/// What a finished turn reports back.
pub struct TurnOutcome {
    pub text: String,
    pub reasoning: Option<String>,
    pub usage: TokenUsage,
    /// Messages compaction dropped before the turn ran.
    pub compacted: usize,
}

/// Run one turn against `history`.
///
/// `history` is mutated in place and ends up holding the user message, the tool
/// transcript, and the final reply. Persisting the tool transcript is
/// deliberate: the next turn's model can then see what it already read, and
/// compaction is what keeps the cost bounded.
///
/// `last_input_tokens` is the previous turn's peak prompt, which decides whether
/// this one starts by compacting. `0` before any turn has reported usage.
///
/// `user_input` is a [`UserInput`] — text plus whatever the user attached — and
/// `From<String>` means a caller with nothing to attach still passes a string.
pub fn run_turn(
    setup: &TurnSetup<'_>,
    history: &mut Vec<ChatMessage>,
    last_input_tokens: u64,
    user_input: impl Into<UserInput>,
) -> Result<TurnOutcome, AgentError> {
    let user_input = user_input.into();
    // A turn lands whole or not at all. If it fails partway, history goes back
    // to what it was — rather than being left holding a prompt and a
    // half-finished tool transcript with no reply, which the next turn would
    // send to the model as if it were settled conversation.
    //
    // Compaction counts as part of the turn for this purpose: it drops settled
    // messages from the caller's history, so a failure has to put them back.
    // Only the compacting path pays for a snapshot — compaction is rare, and
    // cloning history on every turn to guard against a rare failure would cost
    // far more than it saves.
    let before_turn = history.len();
    let mut dropped_by_compaction = None;

    // Compact before the prompt goes in, so the turn starts inside the window.
    let compacted = match memory::compaction_target(
        last_input_tokens,
        memory::estimate_messages_tokens(history),
        setup.context_window,
    ) {
        Some(target) => {
            let snapshot = history.clone();
            let dropped = memory::compact_messages(history, target);
            if dropped > 0 {
                dropped_by_compaction = Some(snapshot);
            }
            dropped
        }
        None => 0,
    };

    // The recorder is minted here rather than by the frontends: a turn is what a
    // trace is *of*, and this is the only place that sees one whole — including
    // the compaction that happened before the prompt went in.
    let recorder = setup.trace.map(|session| session.recorder(setup.turn_id));
    if let Some(recorder) = &recorder {
        // The text only. An attachment's base64 is megabytes of payload that
        // would dwarf everything else in the file, and a trace replays as a
        // script of tool calls, which no image participates in.
        recorder.record_input(&user_input.text, compacted);
    }

    history.push(ChatMessage::user_with_images(
        user_input.text,
        user_input.images,
    ));

    // The catalog is injected for this turn only: skills can be added between
    // turns, and a stale copy accumulating in history would be both wrong and
    // expensive. It goes after any existing system messages rather than at the
    // end, because a chat template that expects system-first will not tolerate
    // one appearing mid-conversation.
    let catalog_at = setup
        .skills
        .and_then(|skills| skills.catalog())
        .map(|catalog| {
            let at = history
                .iter()
                .position(|m| m.role != ChatRole::System)
                .unwrap_or(history.len());
            history.insert(at, ChatMessage::system(catalog));
            at
        });

    // Cloning shares the caller's cancellation flag rather than copying it, so
    // the context the turn runs under is still the one the frontend can stop.
    let mut ctx = setup.context.cloned().unwrap_or_default();
    if let Some(recorder) = &recorder {
        ctx = ctx.with_trace(Arc::clone(recorder));
    }

    let result = react::run_observed(
        setup.provider,
        history,
        setup.tools,
        setup.max_iterations,
        setup.observer,
        &ctx,
    );

    // Lift the catalog back out whether or not the turn succeeded.
    if let Some(at) = catalog_at {
        history.remove(at);
    }

    let (text, reasoning, usage) = match result {
        Ok(turn) => turn,
        Err(e) => {
            match dropped_by_compaction {
                // Compaction ran, so the pre-turn history has to be restored
                // wholesale — the dropped messages came out of the middle and
                // cannot be recovered by truncating.
                Some(original) => *history = original,
                None => history.truncate(before_turn),
            }
            // A turn that did not finish is the one worth having a record of.
            // Cancellation especially: it leaves nothing behind in history by
            // design, so without this it would leave nothing behind at all.
            write_trace(
                setup,
                recorder.as_deref(),
                match e {
                    AgentError::Cancelled => TurnEnding::Cancelled,
                    ref e => TurnEnding::Failed {
                        message: e.to_string(),
                    },
                },
                &TokenUsage::default(),
            );
            return Err(e);
        }
    };
    history.push(ChatMessage::assistant(text.clone()));

    write_trace(
        setup,
        recorder.as_deref(),
        TurnEnding::Completed {
            text: text.clone(),
            reasoning: reasoning.clone(),
        },
        &usage,
    );

    Ok(TurnOutcome {
        text,
        reasoning,
        usage,
        compacted,
    })
}

/// Close the recording and put it on disk, if there was one.
fn write_trace(
    setup: &TurnSetup<'_>,
    recorder: Option<&crate::trace::TurnRecorder>,
    ending: TurnEnding,
    usage: &TokenUsage,
) {
    let (Some(session), Some(recorder)) = (setup.trace, recorder) else {
        return;
    };
    session.write(&recorder.finish(ending, usage));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmResponse, ToolDefinition};
    use crate::tool::ToolRegistry;

    /// Replies with fixed text, recording the prompt it was handed.
    struct Recorder {
        seen: std::sync::Mutex<Vec<Vec<ChatMessage>>>,
    }

    impl LlmProvider for Recorder {
        fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }
        fn supports_tools(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> anyhow::Result<LlmResponse> {
            self.seen.lock().unwrap().push(messages.to_vec());
            Ok(LlmResponse::Text {
                content: "ok".to_string(),
                reasoning: None,
                usage: Some(TokenUsage::single(5, 1, 6)),
            })
        }
    }

    fn recorder() -> Recorder {
        Recorder {
            seen: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn setup<'a>(provider: &'a Recorder, tools: &'a ToolRegistry) -> TurnSetup<'a> {
        TurnSetup {
            provider,
            tools,
            skills: None,
            max_iterations: Some(5),
            context_window: memory::DEFAULT_CONTEXT_WINDOW,
            observer: None,
            context: None,
            trace: None,
            turn_id: None,
        }
    }

    #[test]
    fn a_turn_appends_the_prompt_and_the_reply_to_history() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let mut history = vec![ChatMessage::system("sys".to_string())];

        let outcome =
            run_turn(&setup(&provider, &tools), &mut history, 0, "hi".to_string()).unwrap();

        assert_eq!(outcome.text, "ok");
        assert_eq!(outcome.compacted, 0);
        let roles: Vec<_> = history.iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            roles,
            vec![ChatRole::System, ChatRole::User, ChatRole::Assistant]
        );
    }

    /// The whole point of `UserInput`: an attachment the frontend collected has
    /// to be on the message the provider is handed, not just on the way in.
    #[test]
    fn a_turns_attachments_reach_the_provider() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let mut history = vec![ChatMessage::system("sys".to_string())];

        let input = UserInput {
            text: "what is this?".to_string(),
            images: vec![crate::llm::ImageContent {
                base64: "AAAA".to_string(),
                media_type: "image/png".to_string(),
            }],
        };
        run_turn(&setup(&provider, &tools), &mut history, 0, input).unwrap();

        let seen = provider.seen.lock().unwrap();
        let user = seen[0]
            .iter()
            .find(|m| m.role == ChatRole::User)
            .expect("the prompt reached the provider");
        assert_eq!(user.content, "what is this?");
        assert_eq!(user.images.len(), 1);
        assert_eq!(user.images[0].base64, "AAAA");
    }

    #[test]
    fn the_skill_catalog_reaches_the_model_but_not_the_history() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let skills = SkillRegistry::new();
        skills.add(
            "deploy".to_string(),
            "How to deploy".to_string(),
            "steps".to_string(),
        );

        let mut history = vec![ChatMessage::system("sys".to_string())];
        let mut s = setup(&provider, &tools);
        s.skills = Some(&skills);

        run_turn(&s, &mut history, 0, "hi".to_string()).unwrap();

        let seen = provider.seen.lock().unwrap();
        let prompt = &seen[0];
        assert!(
            prompt.iter().any(|m| m.content.contains("deploy")),
            "the model must be told which skills exist"
        );
        assert!(
            !history.iter().any(|m| m.content.contains("deploy")),
            "the catalog must not accumulate in history — it changes between turns"
        );
    }

    #[test]
    fn the_catalog_sits_with_the_system_messages_not_mid_conversation() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let skills = SkillRegistry::new();
        skills.add(
            "deploy".to_string(),
            "How to deploy".to_string(),
            "steps".to_string(),
        );

        let mut history = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user("earlier".to_string()),
            ChatMessage::assistant("reply".to_string()),
        ];
        let mut s = setup(&provider, &tools);
        s.skills = Some(&skills);

        run_turn(&s, &mut history, 0, "hi".to_string()).unwrap();

        let seen = provider.seen.lock().unwrap();
        let first_non_system = seen[0]
            .iter()
            .position(|m| m.role != ChatRole::System)
            .unwrap();
        assert!(
            seen[0][..first_non_system]
                .iter()
                .any(|m| m.content.contains("deploy")),
            "a template expecting system-first must not meet a system message mid-conversation"
        );
    }

    /// Runs one tool round and then fails, leaving a half-finished transcript.
    struct FailsAfterOneToolCall;

    impl LlmProvider for FailsAfterOneToolCall {
        fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }
        fn supports_tools(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> anyhow::Result<LlmResponse> {
            // Second call (the transcript is already in `messages`) blows up.
            if messages.iter().any(|m| m.role == ChatRole::Tool) {
                anyhow::bail!("provider exploded");
            }
            Ok(LlmResponse::ToolCalls(
                vec![crate::llm::ToolCallInfo {
                    id: "c1".to_string(),
                    name: "nope".to_string(),
                    arguments: serde_json::json!({}),
                }],
                None,
            ))
        }
    }

    #[test]
    fn a_failed_turn_leaves_history_exactly_as_it_found_it() {
        let tools = ToolRegistry::new();
        let mut history = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user("earlier".to_string()),
            ChatMessage::assistant("reply".to_string()),
        ];
        let before = history.clone();

        let setup = TurnSetup {
            provider: &FailsAfterOneToolCall,
            tools: &tools,
            skills: None,
            max_iterations: Some(5),
            context_window: memory::DEFAULT_CONTEXT_WINDOW,
            observer: None,
            context: None,
            trace: None,
            turn_id: None,
        };

        let result = run_turn(&setup, &mut history, 0, "hi".to_string());
        assert!(result.is_err(), "the provider was set up to fail");

        let roles: Vec<_> = history.iter().map(|m| m.role.clone()).collect();
        let before_roles: Vec<_> = before.iter().map(|m| m.role.clone()).collect();
        assert_eq!(
            roles, before_roles,
            "a failed turn must not leave a prompt or a dangling tool transcript behind"
        );
        assert_eq!(
            history.last().unwrap().content,
            "reply",
            "the last settled exchange should still be the last thing in history"
        );
    }

    #[test]
    fn a_failed_turn_also_puts_back_what_compaction_dropped() {
        // Compaction mutates the caller's history before the turn runs. If the
        // turn then fails, those settled messages have to come back, or "the
        // turn did not happen" quietly costs the user their history.
        let tools = ToolRegistry::new();
        let bulky = "x".repeat(4000);
        let mut history = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user(bulky.clone()),
            ChatMessage::assistant("old reply".to_string()),
        ];
        let before = history.clone();

        let setup = TurnSetup {
            provider: &FailsAfterOneToolCall,
            tools: &tools,
            skills: None,
            max_iterations: Some(5),
            // Small enough that the 950-token prior turn triggers compaction.
            context_window: 1000,
            observer: None,
            context: None,
            trace: None,
            turn_id: None,
        };

        let result = run_turn(&setup, &mut history, 950, "hi".to_string());
        assert!(result.is_err(), "the provider was set up to fail");

        let contents: Vec<_> = history.iter().map(|m| m.content.clone()).collect();
        let before_contents: Vec<_> = before.iter().map(|m| m.content.clone()).collect();
        assert_eq!(
            contents, before_contents,
            "a failed turn must not silently cost the user compacted history"
        );
    }

    /// Cancelling is a way for a turn not to happen, so it has to leave history
    /// exactly as a failure would: no prompt, no half-finished transcript.
    #[test]
    fn a_cancelled_turn_leaves_history_exactly_as_it_found_it() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let ctx = crate::cancel::TurnContext::new(crate::cancel::CancellationToken::new());
        ctx.cancellation.cancel();

        let mut history = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user("earlier".to_string()),
            ChatMessage::assistant("reply".to_string()),
        ];
        let before = history.clone();

        let mut s = setup(&provider, &tools);
        s.context = Some(&ctx);

        let result = run_turn(&s, &mut history, 0, "hi".to_string());

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert!(
            provider.seen.lock().unwrap().is_empty(),
            "a turn cancelled before it started must not reach the model"
        );
        let contents: Vec<_> = history.iter().map(|m| m.content.clone()).collect();
        let before_contents: Vec<_> = before.iter().map(|m| m.content.clone()).collect();
        assert_eq!(contents, before_contents);
    }

    #[test]
    fn a_turn_compacts_before_it_starts_when_the_last_one_neared_the_window() {
        let provider = recorder();
        let tools = ToolRegistry::new();
        let mut s = setup(&provider, &tools);
        s.context_window = 1000;

        let bulky = "x".repeat(4000);
        let mut history = vec![
            ChatMessage::system("sys".to_string()),
            ChatMessage::user(bulky.clone()),
            ChatMessage::assistant("old".to_string()),
        ];

        // The previous turn peaked at 950 of a 1000-token window.
        let outcome = run_turn(&s, &mut history, 950, "hi".to_string()).unwrap();

        assert!(outcome.compacted > 0, "history should have been trimmed");
        assert!(
            !history.iter().any(|m| m.content == bulky),
            "the bulky exchange should be gone"
        );
    }
}
