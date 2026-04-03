//! Agent: orchestrates LlmProvider, ConversationMemory, ToolRegistry, and ReAct loop.
//!
//! Simplified from voice-agent/crates/lib/src/lib.rs:
//!   - No voice/capture/backchannel/MCP/UniFFI
//!   - Mutable self (single-threaded, no Arc<Mutex<>>)
//!   - Plain step() → chat() fallback when provider doesn't support tools

use crate::llm::{ChatMessage, LlmProvider, TokenUsage};
use crate::memory::ConversationMemory;
use crate::tool::{ToolAccess, ToolRegistry};
use crate::{react, AgentError};

/// Agent configuration.
pub struct AgentConfig {
    /// Optional system prompt injected at the start of every turn.
    pub system_prompt: Option<String>,
    /// Max new tokens per generation (passed to provider).
    pub max_tokens: u32,
    /// Model context window size in tokens (used for compaction triggering).
    pub context_window: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_tokens: 512,
            context_window: 32_000,
        }
    }
}

/// Response returned from a single agent turn.
pub struct AgentResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    /// Context window usage as a percentage (0–100), 0 if unknown.
    pub context_percent: f32,
}

/// Main agent struct (single-threaded, mutable).
pub struct Agent {
    client: Box<dyn LlmProvider>,
    memory: ConversationMemory,
    tool_registry: ToolRegistry,
    config: AgentConfig,
    last_input_tokens: u64,
}

impl Agent {
    pub fn new(
        client: Box<dyn LlmProvider>,
        tool_registry: ToolRegistry,
        config: AgentConfig,
    ) -> Self {
        Self {
            client,
            memory: ConversationMemory::new(),
            tool_registry,
            config,
            last_input_tokens: 0,
        }
    }

    /// Process a single user turn and return the agent's response.
    pub fn step(&mut self, user_input: String) -> Result<AgentResponse, AgentError> {
        // Compact if last turn used >= 90% of context window.
        self.maybe_compact();

        self.memory.add_message(ChatMessage::user(user_input));

        let mut messages = self.memory.get_messages();

        // Prepend system prompt if set.
        if let Some(ref prompt) = self.config.system_prompt {
            messages.insert(0, ChatMessage::system(prompt.clone()));
        }

        let (response_text, reasoning, usage) =
            if self.client.supports_tools() && !self.tool_registry.is_empty() {
                // ReAct loop.
                let (text, reasoning, usage) = react::run(
                    self.client.as_ref(),
                    &mut messages,
                    &self.tool_registry,
                    None,
                )?;
                (text, reasoning, usage)
            } else {
                // Plain chat fallback (Gallium provider, or OpenAI with no tools).
                let text = self.client.chat(&messages)
                    .map_err(|e| AgentError::NetworkError(e.to_string()))?;
                (text, None, TokenUsage::default())
            };

        self.last_input_tokens = usage.input_tokens;
        self.memory.add_message(ChatMessage::assistant(response_text.clone()));

        let context_percent = if self.config.context_window > 0 {
            (usage.input_tokens as f64 / self.config.context_window as f64 * 100.0) as f32
        } else {
            0.0
        };

        Ok(AgentResponse {
            content: response_text,
            reasoning,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            context_percent,
        })
    }

    /// Reset the conversation memory.
    pub fn reset(&mut self) {
        self.memory.clear();
        tracing::info!("Conversation reset.");
    }

    /// Override the system prompt for subsequent turns.
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.config.system_prompt = Some(prompt);
    }

    fn maybe_compact(&mut self) {
        if self.last_input_tokens == 0 { return; }
        let threshold = (self.config.context_window as f64 * 0.9) as u64;
        if self.last_input_tokens >= threshold {
            let target = self.config.context_window as usize / 2;
            let dropped = self.memory.compact(target);
            if dropped > 0 {
                tracing::info!(
                    "Memory compacted: dropped {} messages (last input: {} tokens)",
                    dropped, self.last_input_tokens
                );
            }
        }
    }
}
