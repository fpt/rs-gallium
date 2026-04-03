//! LLM provider abstraction.
//!
//! Copied and adapted from voice-agent/crates/lib/src/llm.rs.
//! Removed: llama.cpp local backend, UniFFI, skill/situation/harmony references.
//! Kept: OpenAI Responses API provider, all core types.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ============================================================================
// Core types
// ============================================================================

/// Token usage information from an LLM API call.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Chat message role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Image content for multimodal messages.
#[derive(Debug, Clone)]
pub struct ImageContent {
    pub base64: String,
    pub media_type: String,
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip)]
    pub images: Vec<ImageContent>,
    #[serde(skip)]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[serde(skip)]
    pub tool_call_id: Option<String>,
    #[serde(skip)]
    pub tool_name: Option<String>,
}

impl ChatMessage {
    pub fn user(content: String) -> Self {
        Self { role: ChatRole::User, content, images: vec![], tool_calls: None, tool_call_id: None, tool_name: None }
    }

    pub fn assistant(content: String) -> Self {
        Self { role: ChatRole::Assistant, content, images: vec![], tool_calls: None, tool_call_id: None, tool_name: None }
    }

    pub fn system(content: String) -> Self {
        Self { role: ChatRole::System, content, images: vec![], tool_calls: None, tool_call_id: None, tool_name: None }
    }

    pub fn assistant_tool_calls(calls: Vec<ToolCallInfo>) -> Self {
        Self { role: ChatRole::Assistant, content: String::new(), images: vec![], tool_calls: Some(calls), tool_call_id: None, tool_name: None }
    }

    pub fn tool_result(call_id: String, name: String, content: String) -> Self {
        Self { role: ChatRole::Tool, content, images: vec![], tool_calls: None, tool_call_id: Some(call_id), tool_name: Some(name) }
    }

    pub fn tool_result_with_images(call_id: String, name: String, content: String, images: Vec<ImageContent>) -> Self {
        Self { role: ChatRole::Tool, content, images, tool_calls: None, tool_call_id: Some(call_id), tool_name: Some(name) }
    }
}

/// Tool definition for LLM.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call info returned by LLM.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// LLM response — either text or tool calls.
#[derive(Debug)]
pub enum LlmResponse {
    Text {
        content: String,
        reasoning: Option<String>,
        usage: Option<TokenUsage>,
    },
    ToolCalls(Vec<ToolCallInfo>, Option<TokenUsage>),
}

// ============================================================================
// LlmProvider trait
// ============================================================================

pub trait LlmProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String>;

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        Err(anyhow::anyhow!("Tool calling not supported by this provider"))
    }

    fn supports_tools(&self) -> bool {
        false
    }
}

// ============================================================================
// OpenAI Provider — Responses API
// ============================================================================

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message { role: String, content: serde_json::Value },
    #[serde(rename = "function_call")]
    FunctionCall { call_id: String, name: String, arguments: String },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    tool_type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Serialize)]
struct ReasoningParam {
    effort: String,
    summary: String,
}

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningParam>,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    status: String,
    output: Vec<ResponseOutput>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct IncompleteDetails {
    reason: String,
}

#[derive(Debug, Deserialize)]
struct ResponseOutput {
    #[serde(rename = "type")]
    output_type: String,
    #[serde(default)]
    content: Option<Vec<ResponseContent>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    summary: Option<Vec<ReasoningSummary>>,
}

#[derive(Debug, Deserialize)]
struct ReasoningSummary {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ResponseContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

pub struct OpenAiProvider {
    api_key: String,
    model: String,
    temperature: Option<f32>,
    max_tokens: u32,
    reasoning_effort: Option<String>,
    http_agent: ureq::Agent,
}

impl OpenAiProvider {
    fn build_tls_with_custom_ca(cert_file: &str) -> Result<native_tls::TlsConnector> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(cert_file)
            .map_err(|e| anyhow::anyhow!("Failed to open certificate file: {}", e))?;
        let mut cert_data = Vec::new();
        file.read_to_end(&mut cert_data)
            .map_err(|e| anyhow::anyhow!("Failed to read certificate file: {}", e))?;

        let mut builder = native_tls::TlsConnector::builder();
        let cert_str = String::from_utf8_lossy(&cert_data);
        let mut found_cert = false;

        for pem_block in cert_str.split("-----END CERTIFICATE-----") {
            if let Some(cert_start) = pem_block.find("-----BEGIN CERTIFICATE-----") {
                let pem_cert = format!("{}-----END CERTIFICATE-----", &pem_block[cert_start..]);
                match native_tls::Certificate::from_pem(pem_cert.as_bytes()) {
                    Ok(cert) => { builder.add_root_certificate(cert); found_cert = true; }
                    Err(e) => tracing::warn!("Failed to parse PEM certificate: {}", e),
                }
            }
        }

        if !found_cert {
            match native_tls::Certificate::from_der(&cert_data) {
                Ok(cert) => { builder.add_root_certificate(cert); }
                Err(e) => return Err(anyhow::anyhow!("No valid certificates found in file: {}", e)),
            }
        }

        builder.build().map_err(|e| anyhow::anyhow!("Failed to build TLS connector: {}", e))
    }

    pub fn new(
        api_key: String,
        model: String,
        temperature: Option<f32>,
        max_tokens: u32,
        reasoning_effort: Option<String>,
    ) -> Self {
        tracing::info!("Initializing OpenAI provider: model={}", model);

        let http_agent = if let Ok(cert_file) = std::env::var("SSL_CERT_FILE") {
            match Self::build_tls_with_custom_ca(&cert_file) {
                Ok(tls) => ureq::AgentBuilder::new().tls_connector(std::sync::Arc::new(tls)).build(),
                Err(e) => { tracing::error!("Custom CA load failed: {}", e); ureq::agent() }
            }
        } else {
            ureq::agent()
        };

        Self { api_key, model, temperature, max_tokens, reasoning_effort, http_agent }
    }

    fn reasoning_param(&self) -> Option<ReasoningParam> {
        self.reasoning_effort.as_ref().map(|effort| ReasoningParam {
            effort: effort.clone(),
            summary: "auto".to_string(),
        })
    }

    fn convert_to_input_items(messages: &[ChatMessage]) -> Vec<ResponsesInputItem> {
        messages.iter().flat_map(|msg| {
            if let Some(ref calls) = msg.tool_calls {
                return calls.iter().map(|c| ResponsesInputItem::FunctionCall {
                    call_id: c.id.clone(),
                    name: c.name.clone(),
                    arguments: serde_json::to_string(&c.arguments).unwrap_or_default(),
                }).collect::<Vec<_>>();
            }

            if let Some(ref call_id) = msg.tool_call_id {
                let mut items = vec![ResponsesInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: msg.content.clone(),
                }];
                if !msg.images.is_empty() {
                    let mut parts = vec![serde_json::json!({
                        "type": "input_text",
                        "text": format!("[Screenshot from tool '{}']", msg.tool_name.as_deref().unwrap_or("unknown")),
                    })];
                    for img in &msg.images {
                        parts.push(serde_json::json!({
                            "type": "input_image",
                            "image_url": format!("data:{};base64,{}", img.media_type, img.base64),
                        }));
                    }
                    items.push(ResponsesInputItem::Message {
                        role: "user".to_string(),
                        content: serde_json::Value::Array(parts),
                    });
                }
                return items;
            }

            let role = match msg.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::Tool => return vec![],
            };

            let content = if msg.images.is_empty() {
                serde_json::Value::String(msg.content.clone())
            } else {
                let mut parts = vec![serde_json::json!({ "type": "input_text", "text": msg.content })];
                for img in &msg.images {
                    parts.push(serde_json::json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", img.media_type, img.base64),
                    }));
                }
                serde_json::Value::Array(parts)
            };

            vec![ResponsesInputItem::Message { role: role.to_string(), content }]
        }).collect()
    }

    fn convert_tools(tools: &[ToolDefinition]) -> Vec<ResponsesTool> {
        tools.iter().map(|t| ResponsesTool {
            tool_type: "function".to_string(),
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
            strict: false,
        }).collect()
    }

    fn send_request(&self, request: &ResponsesRequest) -> Result<ResponsesResponse> {
        let url = "https://api.openai.com/v1/responses";
        let auth_header = format!("Bearer {}", self.api_key);

        let response_result = self.http_agent.post(url)
            .set("Content-Type", "application/json")
            .set("Authorization", &auth_header)
            .send_json(request);

        let response: ResponsesResponse = match response_result {
            Ok(resp) => {
                let body = resp.into_string()?;
                serde_json::from_str(&body).map_err(|e| {
                    anyhow::anyhow!("Failed to parse OpenAI response: {}\nBody: {}", e, body)
                })?
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_else(|_| "Unable to read error body".to_string());
                return Err(anyhow::anyhow!("OpenAI API error {}: {}", code, body));
            }
            Err(e) => return Err(e.into()),
        };

        if response.status == "incomplete" {
            let reason = response.incomplete_details.as_ref()
                .map(|d| d.reason.clone())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(anyhow::anyhow!("Response incomplete: {}. Increase --max-tokens.", reason));
        }

        Ok(response)
    }

    fn extract_text(output: &[ResponseOutput]) -> Option<String> {
        output.iter()
            .find(|o| o.output_type == "message" || o.output_type == "text")
            .and_then(|o| {
                if let Some(ref text) = o.text { return Some(text.clone()); }
                o.content.as_ref().and_then(|c| c.first()).map(|c| c.text.clone())
            })
    }

    fn extract_reasoning(output: &[ResponseOutput]) -> Option<String> {
        let items: Vec<_> = output.iter().filter(|o| o.output_type == "reasoning").collect();
        if items.is_empty() { return None; }

        let content_parts: Vec<&str> = items.iter()
            .flat_map(|o| o.content.iter().flat_map(|c| c.iter().map(|r| r.text.as_str())))
            .collect();
        if !content_parts.is_empty() { return Some(content_parts.join("\n")); }

        let summary_parts: Vec<&str> = items.iter()
            .flat_map(|o| o.summary.iter().flat_map(|s| s.iter().map(|r| r.text.as_str())))
            .collect();
        if !summary_parts.is_empty() { Some(summary_parts.join("\n")) } else { None }
    }

    fn extract_tool_calls(output: &[ResponseOutput]) -> Vec<ToolCallInfo> {
        output.iter()
            .filter(|o| o.output_type == "function_call")
            .filter_map(|o| {
                let call_id = o.call_id.as_ref()?;
                let name = o.name.as_ref()?;
                let arguments_str = o.arguments.as_ref()?;
                let arguments = serde_json::from_str(arguments_str).unwrap_or_default();
                Some(ToolCallInfo { id: call_id.clone(), name: name.clone(), arguments })
            })
            .collect()
    }

    fn convert_usage(usage: &Option<ResponseUsage>) -> Option<TokenUsage> {
        usage.as_ref().map(|u| TokenUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
        })
    }
}

impl LlmProvider for OpenAiProvider {
    fn supports_tools(&self) -> bool { true }

    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let request = ResponsesRequest {
            model: self.model.clone(),
            input: Self::convert_to_input_items(messages),
            temperature: self.temperature,
            max_output_tokens: Some(self.max_tokens),
            tools: None,
            reasoning: self.reasoning_param(),
        };
        let response = self.send_request(&request)?;
        Self::extract_text(&response.output)
            .ok_or_else(|| anyhow::anyhow!("No text content in response"))
    }

    fn chat_with_tools(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<LlmResponse> {
        let wire_tools = Self::convert_tools(tools);
        let request = ResponsesRequest {
            model: self.model.clone(),
            input: Self::convert_to_input_items(messages),
            temperature: self.temperature,
            max_output_tokens: Some(self.max_tokens),
            tools: if wire_tools.is_empty() { None } else { Some(wire_tools) },
            reasoning: self.reasoning_param(),
        };

        let response = self.send_request(&request)?;
        let usage = Self::convert_usage(&response.usage);

        if let Some(ref u) = usage {
            tracing::info!("Tokens: in={} out={} total={}", u.input_tokens, u.output_tokens, u.total_tokens);
        }

        let tool_calls = Self::extract_tool_calls(&response.output);
        if !tool_calls.is_empty() {
            tracing::info!("OpenAI returned {} tool call(s)", tool_calls.len());
            return Ok(LlmResponse::ToolCalls(tool_calls, usage));
        }

        let text = Self::extract_text(&response.output)
            .ok_or_else(|| anyhow::anyhow!("No text content or tool calls in response"))?;
        let reasoning = Self::extract_reasoning(&response.output);
        Ok(LlmResponse::Text { content: text, reasoning, usage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_user_message_plain() {
        let msgs = vec![ChatMessage::user("hello".to_string())];
        let items = OpenAiProvider::convert_to_input_items(&msgs);
        assert_eq!(items.len(), 1);
        let json = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_convert_tool_result_without_images() {
        let msg = ChatMessage::tool_result("call_1".to_string(), "my_tool".to_string(), "result".to_string());
        let items = OpenAiProvider::convert_to_input_items(&[msg]);
        assert_eq!(items.len(), 1);
        let json = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(json["type"], "function_call_output");
        assert_eq!(json["call_id"], "call_1");
        assert_eq!(json["output"], "result");
    }

    #[test]
    fn test_convert_assistant_tool_calls() {
        let msg = ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
            id: "call_1".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({"file_path": "src/main.rs"}),
        }]);
        let items = OpenAiProvider::convert_to_input_items(&[msg]);
        assert_eq!(items.len(), 1);
        let json = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(json["type"], "function_call");
        assert_eq!(json["name"], "read");
    }
}
