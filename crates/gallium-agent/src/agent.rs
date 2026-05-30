//! Agent: orchestrates LlmProvider, ConversationMemory, ToolRegistry, and ReAct loop.
//!
//! Simplified from voice-agent/crates/lib/src/lib.rs:
//!   - No voice/capture/backchannel/MCP/UniFFI
//!   - Mutable self (single-threaded, no Arc<Mutex<>>)
//!   - Plain step() → chat() fallback when provider doesn't support tools

use std::sync::Arc;

use crate::llm::{ChatMessage, LlmProvider, TokenUsage};
use crate::memory::ConversationMemory;
use crate::skill::SkillRegistry;
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
    skill_registry: Arc<SkillRegistry>,
    config: AgentConfig,
    last_input_tokens: u64,
}

impl Agent {
    pub fn new(
        client: Box<dyn LlmProvider>,
        tool_registry: ToolRegistry,
        config: AgentConfig,
    ) -> Self {
        Self::new_with_skills(client, tool_registry, config, Arc::new(SkillRegistry::new()))
    }

    pub fn new_with_skills(
        client: Box<dyn LlmProvider>,
        tool_registry: ToolRegistry,
        config: AgentConfig,
        skill_registry: Arc<SkillRegistry>,
    ) -> Self {
        Self {
            client,
            memory: ConversationMemory::new(),
            tool_registry,
            skill_registry,
            config,
            last_input_tokens: 0,
        }
    }

    /// Process a single user turn and return the agent's response.
    pub fn step(&mut self, user_input: String) -> Result<AgentResponse, AgentError> {
        self.step_with_images(user_input, vec![])
    }

    /// Process a user turn that includes inline images.
    pub fn step_with_images(
        &mut self,
        user_input: String,
        images: Vec<crate::llm::ImageContent>,
    ) -> Result<AgentResponse, AgentError> {
        self.maybe_compact();

        let mut msg = ChatMessage::user(user_input);
        msg.images = images;
        self.memory.add_message(msg);

        let mut messages = self.memory.get_messages();

        if let Some(ref prompt) = self.config.system_prompt {
            messages.insert(0, ChatMessage::system(prompt.clone()));
        }
        if let Some(catalog) = self.skill_registry.catalog() {
            messages.push(ChatMessage::system(catalog));
        }

        let (response_text, reasoning, usage) =
            if self.client.supports_tools() && !self.tool_registry.is_empty() {
                let (text, reasoning, usage) = react::run(
                    self.client.as_ref(),
                    &mut messages,
                    &self.tool_registry,
                    None,
                )?;
                (text, reasoning, usage)
            } else {
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

    /// Register a skill by name.
    pub fn add_skill(&self, name: String, description: String, prompt: String) {
        self.skill_registry.add(name, description, prompt);
    }

    /// Push a message directly into memory (used when restoring a saved session).
    pub fn memory_push(&mut self, msg: crate::llm::ChatMessage) {
        self.memory.add_message(msg);
    }

    /// Get the conversation history as JSON.
    pub fn get_conversation_history(&self) -> String {
        serde_json::to_string_pretty(&self.memory.get_messages()).unwrap_or_default()
    }

    /// Clone the skill registry (for sharing with the tool registry).
    pub fn skill_registry(&self) -> Arc<SkillRegistry> {
        Arc::clone(&self.skill_registry)
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
