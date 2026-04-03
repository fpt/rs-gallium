//! ReAct (Reason+Act) loop.
//!
//! Copied from voice-agent/crates/lib/src/react.rs with minor adaptation
//! (crate-local AgentError, no uniffi dependencies).

use crate::llm::{ChatMessage, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo};
use crate::tool::{ToolAccess, ToolResult};
use crate::AgentError;

const DEFAULT_MAX_ITERATIONS: u32 = 10;

/// Run a ReAct loop: call LLM with tools, execute tool calls, repeat until text response.
///
/// Returns `(final_text, optional_reasoning, accumulated_token_usage)`.
pub fn run(
    client: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    tools: &dyn ToolAccess,
    max_iterations: Option<u32>,
) -> Result<(String, Option<String>, TokenUsage), AgentError> {
    let max_iter = max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);
    let tool_defs = tools.get_definitions();
    let mut total_usage = TokenUsage::default();

    for iteration in 0..max_iter {
        tracing::info!("ReAct iteration {}/{}", iteration + 1, max_iter);

        let response = client
            .chat_with_tools(messages, &tool_defs)
            .map_err(|e| AgentError::NetworkError(e.to_string()))?;

        match response {
            LlmResponse::Text { content, reasoning, usage } => {
                if let Some(ref u) = usage { total_usage.add(u); }
                tracing::info!(
                    "ReAct complete after {} iteration(s) (in={} out={} total={})",
                    iteration + 1, total_usage.input_tokens, total_usage.output_tokens, total_usage.total_tokens
                );
                return Ok((content, reasoning, total_usage));
            }
            LlmResponse::ToolCalls(calls, usage) => {
                if let Some(ref u) = usage { total_usage.add(u); }
                tracing::info!("ReAct iteration {}: {} tool call(s)", iteration + 1, calls.len());

                messages.push(ChatMessage::assistant_tool_calls(calls.clone()));

                for call in &calls {
                    let result = execute_tool_call(tools, call);
                    tracing::info!("Tool '{}' ({}): {} chars", call.name, call.id, result.text.len());

                    if result.images.is_empty() {
                        messages.push(ChatMessage::tool_result(
                            call.id.clone(), call.name.clone(), result.text,
                        ));
                    } else {
                        messages.push(ChatMessage::tool_result_with_images(
                            call.id.clone(), call.name.clone(), result.text, result.images,
                        ));
                    }
                }
            }
        }
    }

    Err(AgentError::InternalError(format!(
        "ReAct loop exceeded maximum iterations ({})",
        max_iter
    )))
}

fn execute_tool_call(tools: &dyn ToolAccess, call: &ToolCallInfo) -> ToolResult {
    match tools.call(&call.name, call.arguments.clone()) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!("Tool '{}' error: {}", call.name, e);
            ToolResult::text(format!("Error executing tool '{}': {}", call.name, e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatRole, ToolDefinition};
    use crate::tool::ToolRegistry;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockProvider {
        responses: Vec<LlmResponse>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self { responses, call_count: AtomicUsize::new(0) }
        }
    }

    impl LlmProvider for MockProvider {
        fn chat(&self, _: &[ChatMessage]) -> anyhow::Result<String> { Ok("mock".to_string()) }

        fn supports_tools(&self) -> bool { true }

        fn chat_with_tools(&self, _: &[ChatMessage], _: &[ToolDefinition]) -> anyhow::Result<LlmResponse> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                match &self.responses[idx] {
                    LlmResponse::Text { content, reasoning, usage } => Ok(LlmResponse::Text {
                        content: content.clone(), reasoning: reasoning.clone(), usage: usage.clone(),
                    }),
                    LlmResponse::ToolCalls(calls, usage) => Ok(LlmResponse::ToolCalls(calls.clone(), usage.clone())),
                }
            } else {
                Ok(LlmResponse::Text { content: "fallback".to_string(), reasoning: None, usage: None })
            }
        }
    }

    #[test]
    fn test_direct_text_response() {
        let provider = MockProvider::new(vec![LlmResponse::Text {
            content: "Hello!".to_string(), reasoning: None, usage: None,
        }]);
        let mut messages = vec![ChatMessage::user("Hi".to_string())];
        let tools = ToolRegistry::new();
        let (text, reasoning, _) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "Hello!");
        assert!(reasoning.is_none());
    }

    #[test]
    fn test_tool_then_text() {
        use crate::llm::ToolCallInfo;
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "call_1".to_string(),
                name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::Text { content: "No tasks.".to_string(), reasoning: None, usage: None },
        ]);
        let mut messages = vec![ChatMessage::user("List tasks".to_string())];
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(crate::tool::TaskTool::new()));
        let (text, _, _) = run(&provider, &mut messages, &tools, Some(5)).unwrap();
        assert_eq!(text, "No tasks.");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].role, ChatRole::Tool);
    }

    #[test]
    fn test_max_iterations() {
        use crate::llm::ToolCallInfo;
        let provider = MockProvider::new(vec![
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "c1".to_string(), name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
            LlmResponse::ToolCalls(vec![ToolCallInfo {
                id: "c2".to_string(), name: "tasks".to_string(),
                arguments: serde_json::json!({"action": "list"}),
            }], None),
        ]);
        let mut messages = vec![ChatMessage::user("loop".to_string())];
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(crate::tool::TaskTool::new()));
        let err = run(&provider, &mut messages, &tools, Some(2)).unwrap_err();
        assert!(err.to_string().contains("maximum iterations"));
    }
}
