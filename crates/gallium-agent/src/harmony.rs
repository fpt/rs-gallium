use crate::llm::{ChatMessage, ChatRole};

/// Harmony template formatter for gpt-oss
///
/// Harmony is the chat template format used by gpt-oss models.
/// It follows a specific structure for system/user/assistant messages.
pub struct HarmonyTemplate;

impl HarmonyTemplate {
    /// Format messages according to Harmony template
    ///
    /// The Harmony template structure:
    /// - System messages are wrapped with specific tokens
    /// - User/assistant messages follow a turn-based structure
    pub fn format_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
        // For now, we'll let llama.cpp server handle the Harmony template
        // via the --jinja flag. This function can be expanded later if we
        // need client-side template application.

        // Basic filtering: ensure we have valid alternating turns
        let mut formatted = Vec::new();
        let mut last_role: Option<ChatRole> = None;

        for msg in messages {
            // Skip duplicate consecutive roles (except system)
            if msg.role == ChatRole::System || Some(msg.role.clone()) != last_role {
                formatted.push(msg.clone());
                last_role = Some(msg.role.clone());
            }
        }

        formatted
    }

    /// Create a system message for the agent
    pub fn create_system_message(content: String) -> ChatMessage {
        ChatMessage::system(content)
    }

    /// Default system prompt for the kessel
    pub fn default_system_prompt() -> String {
        "You are a helpful voice assistant running locally in a car. \
         Be concise and natural in your responses, as they will be spoken aloud. \
         Avoid long explanations unless specifically asked. \
         You have access to tools for file operations and can help with various tasks."
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_messages() {
        let messages = vec![
            ChatMessage::system("System".to_string()),
            ChatMessage::user("Hello".to_string()),
            ChatMessage::user("Hello again".to_string()),
            ChatMessage::assistant("Hi".to_string()),
        ];

        let formatted = HarmonyTemplate::format_messages(&messages);

        // Should remove duplicate User message
        assert_eq!(formatted.len(), 3);
        assert_eq!(formatted[0].role, ChatRole::System);
        assert_eq!(formatted[1].role, ChatRole::User);
        assert_eq!(formatted[2].role, ChatRole::Assistant);
    }

    #[test]
    fn test_create_system_message() {
        let msg = HarmonyTemplate::create_system_message("Test".to_string());
        assert_eq!(msg.role, ChatRole::System);
        assert_eq!(msg.content, "Test");
    }
}
