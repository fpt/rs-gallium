//! A provider that replays a canned script instead of running a model.
//!
//! Everything else in this crate needs multi-GB weights or a network key to
//! answer a single prompt, which is why the model tests are `#[ignore]`d and the
//! agent testsuite is opt-in. That cost lands hardest on the things that have
//! nothing to do with sampling: the app-server wire format, the ReAct loop's
//! tool plumbing, approval routing. A scripted provider makes those testable in
//! milliseconds, and — the reason it exists as a real engine rather than a test
//! fixture — testable *from outside the process*, by a client driving
//! `gallium app-server` in its own CI.
//!
//! ```bash
//! INFERENCE_ENGINE=scripted MODEL_PATH=script.json gallium app-server
//! ```
//!
//! The script is a list of steps, consumed one per model call:
//!
//! ```json
//! {
//!   "steps": [
//!     { "toolCalls": [{ "id": "c1", "name": "Read", "arguments": { "path": "Cargo.toml" } }] },
//!     { "text": "It is a manifest.", "reasoning": "the user asked about a file" }
//!   ]
//! }
//! ```
//!
//! It is deliberately not a mock framework: no matching on the prompt, no
//! conditionals. A script that depended on what the model was asked would drift
//! from the thing under test in exactly the way a fixed sequence cannot.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::llm::{ChatMessage, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo, ToolDefinition};

/// One scripted model turn: text, or a batch of tool calls.
///
/// Untagged rather than a `type` discriminator so the common case reads as
/// `{"text": "..."}`. `toolCalls` wins if a step somehow carries both, since a
/// step with tool calls is the one that keeps the loop going.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScriptStep {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ScriptToolCall>,
    /// Reported as this step's usage. Absent means the provider reports none,
    /// which is what the native candle backend does and what the compaction
    /// path has to cope with.
    #[serde(default)]
    pub input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScriptToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Script {
    pub steps: Vec<ScriptStep>,
}

impl Script {
    pub fn parse(json: &str) -> Result<Self> {
        let script: Script =
            serde_json::from_str(json).context("parsing the scripted provider's script")?;
        if script.steps.is_empty() {
            anyhow::bail!("a script needs at least one step; this one has none");
        }
        Ok(script)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("reading script '{}'", path.display()))?;
        Self::parse(&json)
    }
}

/// Replays a [`Script`], one step per model call.
pub struct ScriptedProvider {
    script: Script,
    next: AtomicUsize,
}

impl ScriptedProvider {
    pub fn new(script: Script) -> Self {
        Self {
            script,
            next: AtomicUsize::new(0),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self::new(Script::load(path)?))
    }

    /// Take the next step, or explain that the script ran dry.
    ///
    /// Running out is an error rather than a repeat of the last step: a loop
    /// that asked for more turns than the script describes has diverged from
    /// what the test meant, and silently answering again would hide it behind
    /// an iteration-budget failure instead.
    fn step(&self) -> Result<ScriptStep> {
        let i = self.next.fetch_add(1, Ordering::SeqCst);
        self.script.steps.get(i).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "scripted provider exhausted: the turn asked for step {} of a {}-step script",
                i + 1,
                self.script.steps.len()
            )
        })
    }
}

impl LlmProvider for ScriptedProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> Result<String> {
        let step = self.step()?;
        Ok(step.text.unwrap_or_default())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let step = self.step()?;
        let usage = step
            .input_tokens
            .map(|input| TokenUsage::single(input, 1, input + 1));

        if !step.tool_calls.is_empty() {
            let calls = step
                .tool_calls
                .into_iter()
                .map(|c| ToolCallInfo {
                    id: c.id,
                    name: c.name,
                    arguments: c.arguments,
                })
                .collect();
            return Ok(LlmResponse::ToolCalls {
                calls,
                usage,
                reasoning: None,
            });
        }

        Ok(LlmResponse::Text {
            content: step.text.unwrap_or_default(),
            reasoning: step.reasoning,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_STEPS: &str = r#"{
        "steps": [
            { "toolCalls": [{ "id": "c1", "name": "Read", "arguments": { "path": "x" } }] },
            { "text": "done", "reasoning": "read it first" }
        ]
    }"#;

    #[test]
    fn steps_are_replayed_in_order() {
        let provider = ScriptedProvider::new(Script::parse(TWO_STEPS).unwrap());

        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "c1");
                assert_eq!(calls[0].name, "Read");
                assert_eq!(calls[0].arguments["path"], "x");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }

        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::Text {
                content, reasoning, ..
            } => {
                assert_eq!(content, "done");
                assert_eq!(reasoning.as_deref(), Some("read it first"));
            }
            other => panic!("expected text, got {other:?}"),
        }
    }

    /// Running dry has to say which step was asked for, because the usual cause
    /// is a loop taking more turns than the script's author expected.
    #[test]
    fn running_out_of_steps_is_an_error_that_says_so() {
        let provider =
            ScriptedProvider::new(Script::parse(r#"{"steps":[{"text":"one"}]}"#).unwrap());
        provider.chat_with_tools(&[], &[]).unwrap();

        let err = provider.chat_with_tools(&[], &[]).unwrap_err().to_string();

        assert!(err.contains("exhausted"), "{err}");
        assert!(err.contains("step 2"), "{err}");
        assert!(err.contains("1-step"), "{err}");
    }

    /// No `inputTokens` means no usage at all — the native candle backend
    /// reports none, and the compaction path has to survive that.
    #[test]
    fn usage_is_absent_unless_the_step_gives_one() {
        let provider = ScriptedProvider::new(
            Script::parse(r#"{"steps":[{"text":"hi"},{"text":"yo","inputTokens":40}]}"#).unwrap(),
        );

        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::Text { usage, .. } => assert!(usage.is_none()),
            other => panic!("expected text, got {other:?}"),
        }
        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::Text { usage, .. } => {
                let usage = usage.expect("a step with inputTokens reports usage");
                assert_eq!(usage.input_tokens, 40);
                assert_eq!(usage.peak_input_tokens, 40);
            }
            other => panic!("expected text, got {other:?}"),
        }
    }

    /// An empty script is a mistake worth refusing at load, not a provider that
    /// fails on the first prompt.
    #[test]
    fn an_empty_script_is_refused() {
        let err = Script::parse(r#"{"steps":[]}"#).unwrap_err().to_string();
        assert!(err.contains("at least one step"), "{err}");
    }

    #[test]
    fn a_malformed_script_says_what_it_was_reading() {
        let err = Script::parse("not json").unwrap_err().to_string();
        assert!(err.contains("script"), "{err}");
    }

    /// A step carrying both is ambiguous; tool calls win, because that is the
    /// step that keeps the ReAct loop going.
    #[test]
    fn tool_calls_win_over_text_in_the_same_step() {
        let provider = ScriptedProvider::new(
            Script::parse(
                r#"{"steps":[{"text":"ignored","toolCalls":[{"id":"c1","name":"LS"}]}]}"#,
            )
            .unwrap(),
        );

        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::ToolCalls { calls, .. } => assert_eq!(calls[0].name, "LS"),
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    /// `arguments` is optional, since plenty of tools take none.
    #[test]
    fn a_tool_call_may_omit_its_arguments() {
        let provider = ScriptedProvider::new(
            Script::parse(r#"{"steps":[{"toolCalls":[{"id":"c1","name":"LS"}]}]}"#).unwrap(),
        );

        match provider.chat_with_tools(&[], &[]).unwrap() {
            LlmResponse::ToolCalls { calls, .. } => assert!(calls[0].arguments.is_null()),
            other => panic!("expected tool calls, got {other:?}"),
        }
    }
}
