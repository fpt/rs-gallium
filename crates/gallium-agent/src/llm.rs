use std::borrow::Cow;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;

// ============================================================================
// Core types
// ============================================================================

/// Context size the in-process llama.cpp backend is created with, and the
/// compaction window assumed for a local model when nothing configures one.
/// Override per model with `llm.contextWindow` / `CONTEXT_WINDOW` — a local
/// session that assumes far more window than the backend has will never compact
/// in time.
pub const LOCAL_CONTEXT_WINDOW: u32 = 8192;

/// How long a model call spent in each of its two halves.
///
/// Split rather than one total because the halves scale differently and a
/// single average hides which one a change moved: prefill is one forward over
/// the whole prompt, decode is one forward per sampled token. A tuning knob
/// that doubles prefill throughput and costs 10% of decode looks like a wash in
/// a combined number.
///
/// Only a provider that can actually see the first token fills this in. A
/// blocking API returns a finished string and cannot say when generation
/// started, so [`TokenUsage::timing`] stays `None` there rather than reporting
/// the round trip as if it were prefill.
///
/// It carries its own token counts rather than reading them off the
/// [`TokenUsage`] it hangs on, so that a total covering *both* timed and
/// untimed calls still divides timed durations by timed tokens only. Taking the
/// counts from the usage would price another provider's output against this
/// one's clock — a rate that is wrong in the flattering direction, and
/// invisible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timing {
    /// Call start → first sampled token: tokenization, context setup, prompt
    /// eval, and the first sample. Summed as calls accumulate.
    pub prefill: Duration,
    /// First token → last token.
    pub decode: Duration,
    /// The *first* call's `prefill`, kept as calls accumulate — the latency a
    /// user actually waited through before seeing anything. A turn's summed
    /// prefill is a cost; this is the wait, and they are different numbers.
    pub ttft: Duration,
    /// Prompt tokens evaluated during `prefill`.
    pub prefill_tokens: u64,
    /// Tokens sampled during `decode`: the output of the calls covered here,
    /// less each call's *first* token, which its prefill produced.
    pub decode_tokens: u64,
    /// How many model calls this covers.
    pub calls: u32,
}

impl Timing {
    /// Timing for one call, whose TTFT is by definition its own prefill.
    ///
    /// `output_tokens` is the whole generation; the first of them is prefill's
    /// output, and subtracting it here is what keeps that rule in one place
    /// rather than in every backend.
    pub fn for_call(
        prefill: Duration,
        decode: Duration,
        prompt_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        Self {
            prefill,
            decode,
            ttft: prefill,
            prefill_tokens: prompt_tokens,
            decode_tokens: output_tokens.saturating_sub(1),
            calls: 1,
        }
    }

    /// Accumulate a later call. `ttft` is deliberately not touched: it belongs
    /// to the first call in the run and later ones cannot improve on it.
    pub fn add(&mut self, other: &Timing) {
        self.prefill += other.prefill;
        self.decode += other.decode;
        self.prefill_tokens += other.prefill_tokens;
        self.decode_tokens += other.decode_tokens;
        self.calls += other.calls;
    }

    /// Prompt tokens per second, `None` when there is nothing to divide (an
    /// empty prompt, or a clock that did not move).
    pub fn prefill_rate(&self) -> Option<f64> {
        rate(self.prefill_tokens, self.prefill)
    }

    /// Generated tokens per second, over the decode intervals only.
    pub fn decode_rate(&self) -> Option<f64> {
        rate(self.decode_tokens, self.decode)
    }
}

/// A throughput as text. An unmeasurable rate prints `n/a` rather than `0.0`,
/// which would read as "this backend is infinitely slow" instead of "nothing
/// was timed".
pub fn fmt_rate(rate: Option<f64>) -> String {
    match rate {
        Some(r) => format!("{r:.1} tok/s"),
        None => "n/a".to_string(),
    }
}

fn rate(tokens: u64, over: Duration) -> Option<f64> {
    let secs = over.as_secs_f64();
    (tokens > 0 && secs > 0.0).then(|| tokens as f64 / secs)
}

/// Token usage information from an LLM API call
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    /// Largest single-call prompt in whatever this usage covers. `input_tokens`
    /// sums every call in a turn, so it says nothing about how close the
    /// conversation came to the context window — a five-iteration ReAct turn
    /// reports roughly five prompts' worth. This is the high-water mark, and is
    /// what compaction triggers on.
    pub peak_input_tokens: u64,
    /// Wall clock for the call(s) these counts cover, when the provider could
    /// measure it. `None` means "not measured", never "instant".
    pub timing: Option<Timing>,
}

impl TokenUsage {
    /// Usage for a single call, whose peak is by definition its own prompt.
    pub fn single(input_tokens: u64, output_tokens: u64, total_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens,
            peak_input_tokens: input_tokens,
            timing: None,
        }
    }

    /// The same, from a provider that watched the tokens come out. The timing
    /// is built here from the call's own counts, so it cannot disagree with
    /// them.
    pub fn timed(
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        prefill: Duration,
        decode: Duration,
    ) -> Self {
        Self::timed_partial_prefill(
            input_tokens,
            output_tokens,
            total_tokens,
            input_tokens,
            prefill,
            decode,
        )
    }

    /// The same, for a call that only had to **evaluate part of its prompt** —
    /// the rest served from a warm KV cache.
    ///
    /// `input_tokens` is still the whole prompt: that is what the model was
    /// given, and what a context gauge measures. `prefill_tokens` is what was
    /// actually computed, and it is what the prefill *rate* is priced on. A
    /// cache hit that divided the whole prompt by the time to evaluate a
    /// hundred tokens would report a throughput the hardware never achieved,
    /// and it would climb the better the cache worked.
    pub fn timed_partial_prefill(
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        prefill_tokens: u64,
        prefill: Duration,
        decode: Duration,
    ) -> Self {
        Self {
            timing: Some(Timing::for_call(
                prefill,
                decode,
                prefill_tokens,
                output_tokens,
            )),
            ..Self::single(input_tokens, output_tokens, total_tokens)
        }
    }

    /// Accumulate usage from another call.
    ///
    /// An untimed call adds its tokens to the counts but nothing to the
    /// timing — which is why [`Timing`] keeps its own counts. The totals then
    /// describe the whole turn while the rates describe only its timed part,
    /// rather than one silently contaminating the other.
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.peak_input_tokens = self.peak_input_tokens.max(other.peak_input_tokens);
        match (&mut self.timing, &other.timing) {
            (Some(mine), Some(theirs)) => mine.add(theirs),
            (slot @ None, Some(theirs)) => *slot = Some(*theirs),
            (_, None) => {}
        }
    }

    /// Prompt throughput over the *timed* calls this usage covers.
    pub fn prefill_rate(&self) -> Option<f64> {
        self.timing?.prefill_rate()
    }

    /// Generation throughput over the *timed* calls this usage covers.
    pub fn decode_rate(&self) -> Option<f64> {
        self.timing?.decode_rate()
    }
}

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Image content for multimodal messages
#[derive(Debug, Clone)]
pub struct ImageContent {
    pub base64: String,
    pub media_type: String, // "image/png", "image/jpeg"
}

/// Audio content for multimodal messages.
///
/// Deliberately a separate type from [`ImageContent`] rather than one
/// `MediaContent` with a discriminant: the providers treat them differently —
/// OpenAI has `input_image` and `input_audio` as distinct item types, and a
/// projector can support one modality and not the other — so the distinction
/// has to survive to the point of use rather than being re-derived from a
/// media-type string.
#[derive(Debug, Clone)]
pub struct AudioContent {
    pub base64: String,
    pub media_type: String, // "audio/wav", "audio/mpeg", "audio/flac"
}

/// One attachment, in the position its author put it.
///
/// A single ordered list rather than an images vec beside an audio vec, because
/// **order across modalities is meaning**: "here is a photo, and here is my
/// voice note about it" is not the same prompt as the reverse. Two vecs cannot
/// express that, and reassembling one from them requires picking an order —
/// which is a silent rewrite of what the user wrote.
///
/// It also removes a whole class of bug from the llama.cpp path, where mtmd
/// pairs media with `<__media__>` markers *positionally*: with one list there
/// is one order, so markers and bytes cannot disagree.
#[derive(Debug, Clone)]
pub enum MediaContent {
    Image(ImageContent),
    Audio(AudioContent),
}

impl MediaContent {
    /// The declared media type, whichever kind this is.
    pub fn media_type(&self) -> &str {
        match self {
            MediaContent::Image(i) => &i.media_type,
            MediaContent::Audio(a) => &a.media_type,
        }
    }

    /// The base64 payload, whichever kind this is.
    pub fn base64(&self) -> &str {
        match self {
            MediaContent::Image(i) => &i.base64,
            MediaContent::Audio(a) => &a.base64,
        }
    }

    /// `"image"` or `"audio"` — the modality a provider is being asked for.
    pub fn kind(&self) -> &'static str {
        match self {
            MediaContent::Image(_) => "image",
            MediaContent::Audio(_) => "audio",
        }
    }
}

impl From<ImageContent> for MediaContent {
    fn from(image: ImageContent) -> Self {
        MediaContent::Image(image)
    }
}

impl From<AudioContent> for MediaContent {
    fn from(audio: AudioContent) -> Self {
        MediaContent::Audio(audio)
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Attachments, in the order they were given. See [`MediaContent`] for why
    /// this is one ordered list and not a vec per modality.
    #[serde(skip)]
    pub media: Vec<MediaContent>,
    /// Tool calls made by assistant (set by ReAct loop)
    #[serde(skip)]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    /// Tool call ID this message is responding to (for role=Tool)
    #[serde(skip)]
    pub tool_call_id: Option<String>,
    /// Tool name this message is responding to (for role=Tool)
    #[serde(skip)]
    pub tool_name: Option<String>,
    /// What the model reasoned before this turn's answer or tool calls, with
    /// its wrapper removed — `reasoning_content` to a chat template that
    /// carries prior-turn thinking.
    ///
    /// `None` means *nothing to report*, not "reasoned nothing", and the two
    /// have to stay distinguishable: a template that renders
    /// `<think>{{ reasoning_content }}</think>` for every assistant turn will
    /// otherwise tell the model its own earlier reasoning was empty. An empty
    /// string is folded into `None` at the source ([`crate::profile::wire::think::think_content`])
    /// so a template branching on `is string` sees one state and not two.
    #[serde(skip)]
    pub reasoning: Option<String>,
}

impl ChatMessage {
    pub fn user(content: String) -> Self {
        Self {
            role: ChatRole::User,
            content,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
        }
    }

    /// A user turn carrying attachments, in the order they were given. An empty
    /// list is exactly [`ChatMessage::user`], so a frontend does not branch.
    pub fn user_with_media(content: String, media: Vec<MediaContent>) -> Self {
        Self {
            role: ChatRole::User,
            content,
            media,
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
        }
    }

    /// The images among this message's attachments, in order. For providers
    /// that carry pictures but not sound.
    pub fn images(&self) -> impl Iterator<Item = &ImageContent> {
        self.media.iter().filter_map(|m| match m {
            MediaContent::Image(i) => Some(i),
            MediaContent::Audio(_) => None,
        })
    }

    /// How many attachments of each kind this message carries: `(images, audio)`.
    pub fn media_counts(&self) -> (usize, usize) {
        self.media.iter().fold((0, 0), |(i, a), m| match m {
            MediaContent::Image(_) => (i + 1, a),
            MediaContent::Audio(_) => (i, a + 1),
        })
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: ChatRole::Assistant,
            content,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
        }
    }

    pub fn system(content: String) -> Self {
        Self {
            role: ChatRole::System,
            content,
            media: vec![],
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
        }
    }

    pub fn assistant_tool_calls(calls: Vec<ToolCallInfo>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: String::new(),
            media: vec![],
            tool_calls: Some(calls),
            tool_call_id: None,
            tool_name: None,
            reasoning: None,
        }
    }

    /// This message, carrying the reasoning the model produced before it.
    ///
    /// A builder rather than a parameter on every constructor: only an
    /// assistant turn has reasoning, and only a provider that saw the raw
    /// output can supply it — everywhere else the answer is `None` and should
    /// not have to be typed.
    pub fn with_reasoning(mut self, reasoning: Option<String>) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn tool_result(call_id: String, name: String, content: String) -> Self {
        Self {
            role: ChatRole::Tool,
            content,
            media: vec![],
            tool_calls: None,
            tool_call_id: Some(call_id),
            tool_name: Some(name),
            reasoning: None,
        }
    }

    pub fn tool_result_with_images(
        call_id: String,
        name: String,
        content: String,
        images: Vec<ImageContent>,
    ) -> Self {
        Self {
            role: ChatRole::Tool,
            content,
            // Images only: `ToolContent` has no audio variant, so no tool can
            // produce a clip to interleave here.
            media: images.into_iter().map(MediaContent::Image).collect(),
            tool_calls: None,
            tool_call_id: Some(call_id),
            tool_name: Some(name),
            reasoning: None,
        }
    }
}

/// Tool definition for LLM
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call info returned by LLM
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// LLM response — either text or tool calls
#[derive(Debug)]
pub enum LlmResponse {
    Text {
        content: String,
        reasoning: Option<String>,
        usage: Option<TokenUsage>,
    },
    ToolCalls {
        calls: Vec<ToolCallInfo>,
        usage: Option<TokenUsage>,
        /// What the model reasoned on the way to these calls. Named rather
        /// than positional because a third `Option` in a tuple is unreadable
        /// at the ~40 sites that build one.
        reasoning: Option<String>,
    },
}

// ============================================================================
// LlmProvider trait
// ============================================================================

/// Refuse a request carrying images a backend has no way to look at *as built*.
///
/// Both local backends build a *text* prompt — llama.cpp through the GGUF's
/// jinja template, candle through a [`crate::protocol::PromptRenderer`] — so
/// neither has anywhere to put pixels. Dropping them would send the model a
/// caption with nothing attached, and it would answer confidently about an
/// image it never received: a wrong answer that looks exactly like a model
/// failing to see. Refusing says which of the two it was.
///
/// "As built" is the whole qualification, and the message says so rather than
/// implying the engines are incapable. **llama.cpp has multimodal support** —
/// `mtmd`, driven by a `--mmproj` projector file, covering image *and* audio —
/// and `llama-cpp-2` already wraps it (`llama_cpp_2::mtmd`, behind that crate's
/// `mtmd` feature, which gallium does not enable). Wiring it is a real piece of
/// work, not a flag: `MtmdContext` tokenizes and encodes chunks itself, which
/// is a different prompt path than the jinja-rendered string `llm_local` builds
/// today. Until someone does it, this is an honest refusal and not a verdict on
/// the engine.
///
/// The check is cheap and message-shaped rather than a capability flag on the
/// trait, because it is the *request* that is unservable, not the provider that
/// is misconfigured — the same backend answers every text turn fine.
pub(crate) fn reject_media(messages: &[ChatMessage], backend: &str) -> Result<()> {
    let (images, clips) = count_media(messages);
    if images == 0 && clips == 0 {
        return Ok(());
    }
    Err(crate::AgentError::InvalidInput(format!(
        "{backend} cannot see images as built ({images} image(s), {clips} audio clip(s) \
         attached): gallium gives it a text prompt and no projector. Use the llama.cpp \
         backend with `mmprojPath` set, or a provider that accepts media."
    ))
    .into())
}

/// Refuse audio on a provider that takes images but not sound.
///
/// The OpenAI path carries `input_image` and has for a while; audio would be a
/// second item type with its own encoding rules, and sending one gallium has
/// not verified would fail at the API with a worse message than this. Refusing
/// keeps the rule the rest of the crate follows — an attachment either reaches
/// the model or the turn says why not.
/// Attachments across a whole history: `(images, audio)`.
pub(crate) fn count_media(messages: &[ChatMessage]) -> (usize, usize) {
    messages.iter().fold((0, 0), |(i, a), m| {
        let (mi, ma) = m.media_counts();
        (i + mi, a + ma)
    })
}

pub(crate) fn reject_audio(messages: &[ChatMessage], backend: &str) -> Result<()> {
    let (_, clips) = count_media(messages);
    if clips == 0 {
        return Ok(());
    }
    Err(crate::AgentError::InvalidInput(format!(
        "{backend} does not carry audio ({clips} clip(s) attached). Audio reaches a model \
         only on the llama.cpp backend, with a projector that has an audio encoder."
    ))
    .into())
}

/// Trait for LLM providers
pub trait LlmProvider: Send + Sync {
    /// Send a chat completion request
    fn chat(&self, messages: &[ChatMessage]) -> Result<String>;

    /// Send a chat completion request with JSON Schema for structured output
    fn chat_with_schema(
        &self,
        messages: &[ChatMessage],
        _schema: serde_json::Value,
        _schema_name: &str,
    ) -> Result<String> {
        tracing::warn!(
            "chat_with_schema not supported by this provider, falling back to regular chat"
        );
        self.chat(messages)
    }

    /// Send a chat request with tool definitions, returning either text or tool calls
    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        Err(anyhow::anyhow!(
            "Tool calling not supported by this provider"
        ))
    }

    /// As [`LlmProvider::chat_with_tools`], abandoning generation if `cancel`
    /// fires partway.
    ///
    /// The default ignores the token, which is the truthful answer for a
    /// provider whose call is one blocking HTTP round trip: there is nothing to
    /// interrupt between sending and receiving, and the ReAct loop stops as
    /// soon as the response lands. The local backends sample a token at a time
    /// and override this.
    fn chat_with_tools_cancellable(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        _cancel: &CancellationToken,
    ) -> Result<LlmResponse> {
        self.chat_with_tools(messages, tools)
    }

    /// Check if this provider supports structured output
    fn supports_structured_output(&self) -> bool {
        false
    }

    /// Check if this provider supports tool calling
    fn supports_tools(&self) -> bool {
        false
    }

    /// The context window this provider actually runs in, when it knows one.
    ///
    /// `None` is the honest answer for a provider that cannot say — an OpenAI
    /// model's window is a property of the model name, not of anything the API
    /// reports back. A caller drawing a gauge shows nothing rather than drawing
    /// one against a number it made up: gallium's own fallbacks
    /// ([`LOCAL_CONTEXT_WINDOW`], [`memory::DEFAULT_CONTEXT_WINDOW`]) are sized
    /// to decide *when to compact*, which tolerates being wrong in a way a
    /// number shown to a user does not.
    ///
    /// Explicit configuration still wins over this: a user who sets
    /// `contextWindow` has said something about their setup that the model file
    /// does not know.
    fn context_window(&self) -> Option<u32> {
        None
    }

    /// A fixed instruction this provider's model family needs ahead of
    /// whatever system prompt the operator or client supplies — gallium's
    /// protocol ABI for that family, not its persona or task. See
    /// `crate::profile::ModelProfile::agent_preamble`, which is where a local
    /// provider's answer actually comes from.
    ///
    /// `None` is the honest answer for OpenAI (there is no `ModelProfile` on
    /// that path — the Responses API's own tool-calling format is not a
    /// gallium wire protocol to remind a model about) and for any local model
    /// whose profile has nothing to add.
    fn agent_preamble(&self) -> Option<Cow<'static, str>> {
        None
    }
}

// ============================================================================
// OpenAI Provider (cloud API) — Responses API
// ============================================================================

// -- Wire format types for Responses API --

/// Input item for Responses API
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: serde_json::Value,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

/// Tool definition for Responses API
#[derive(Debug, Serialize)]
struct ResponsesTool {
    #[serde(rename = "type")]
    tool_type: String,
    name: String,
    description: String,
    parameters: serde_json::Value,
    strict: bool,
}

/// Reasoning parameter for OpenAI reasoning models
#[derive(Debug, Serialize)]
struct ReasoningParam {
    effort: String,
    summary: String, // "auto", "concise", or "detailed" — must be set to get reasoning output
}

/// OpenAI Responses API request
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
    text: Option<ResponseTextFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningParam>,
}

/// Text format specification for structured output
#[derive(Debug, Serialize)]
struct ResponseTextFormat {
    format: ResponseFormatSpec,
}

/// Format specification with JSON Schema
#[derive(Debug, Serialize)]
struct ResponseFormatSpec {
    #[serde(rename = "type")]
    format_type: String, // "json_schema"
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

/// OpenAI Responses API response
#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    status: String,
    output: Vec<ResponseOutput>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

/// Token usage from the Responses API
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
    // For "message" type
    #[serde(default)]
    content: Option<Vec<ResponseContent>>,
    #[serde(default)]
    text: Option<String>,
    // For "function_call" type
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    // For "reasoning" type
    #[serde(default)]
    summary: Option<Vec<ReasoningSummary>>,
}

#[derive(Debug, Deserialize)]
struct ReasoningSummary {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ResponseContent {
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
    /// Build TLS connector with custom CA certificates
    fn build_tls_with_custom_ca(cert_file: &str) -> Result<native_tls::TlsConnector> {
        use std::fs::File;
        use std::io::Read;

        // Read certificate file
        let mut file = File::open(cert_file)
            .map_err(|e| anyhow::anyhow!("Failed to open certificate file: {}", e))?;
        let mut cert_data = Vec::new();
        file.read_to_end(&mut cert_data)
            .map_err(|e| anyhow::anyhow!("Failed to read certificate file: {}", e))?;

        // Parse certificate(s) - PEM format can contain multiple certificates
        let mut builder = native_tls::TlsConnector::builder();

        // Try to parse as PEM (most common format)
        let cert_str = String::from_utf8_lossy(&cert_data);
        let mut found_cert = false;

        // Split by PEM boundaries
        for pem_block in cert_str.split("-----END CERTIFICATE-----") {
            if let Some(cert_start) = pem_block.find("-----BEGIN CERTIFICATE-----") {
                let pem_cert = format!("{}-----END CERTIFICATE-----", &pem_block[cert_start..]);

                match native_tls::Certificate::from_pem(pem_cert.as_bytes()) {
                    Ok(cert) => {
                        builder.add_root_certificate(cert);
                        found_cert = true;
                        tracing::debug!("Added certificate from PEM");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse PEM certificate: {}", e);
                    }
                }
            }
        }

        if !found_cert {
            // Try DER format as fallback
            match native_tls::Certificate::from_der(&cert_data) {
                Ok(cert) => {
                    builder.add_root_certificate(cert);
                    tracing::debug!("Added certificate from DER");
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "No valid certificates found in file: {}",
                        e
                    ));
                }
            }
        }

        builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build TLS connector: {}", e))
    }

    pub fn new(
        api_key: String,
        model: String,
        temperature: Option<f32>,
        max_tokens: u32,
        reasoning_effort: Option<String>,
    ) -> Self {
        tracing::info!("Initializing OpenAI provider (Responses API)");
        tracing::info!("  Model: {}", model);
        tracing::info!("  Reasoning effort: {:?}", reasoning_effort);

        let http_agent = http_agent_with_ca(None);

        Self {
            api_key,
            model,
            temperature,
            max_tokens,
            reasoning_effort,
            http_agent,
        }
    }

    /// Build reasoning param if configured. Uses `summary: "detailed"` so the
    /// Responses API reliably returns a reasoning summary (with "auto" the model
    /// often omits it on simpler prompts, so nothing prints to the console).
    fn reasoning_param(&self) -> Option<ReasoningParam> {
        self.reasoning_effort.as_ref().map(|effort| ReasoningParam {
            effort: effort.clone(),
            summary: "detailed".to_string(),
        })
    }

    /// Convert ChatMessages to Responses API input items
    fn convert_to_input_items(messages: &[ChatMessage]) -> Vec<ResponsesInputItem> {
        messages
            .iter()
            .flat_map(|msg| {
                // Handle assistant messages with tool calls
                if let Some(ref calls) = msg.tool_calls {
                    return calls
                        .iter()
                        .map(|c| ResponsesInputItem::FunctionCall {
                            call_id: c.id.clone(),
                            name: c.name.clone(),
                            arguments: serde_json::to_string(&c.arguments).unwrap_or_default(),
                        })
                        .collect::<Vec<_>>();
                }

                // Handle tool result messages
                if let Some(ref call_id) = msg.tool_call_id {
                    let mut items = vec![ResponsesInputItem::FunctionCallOutput {
                        call_id: call_id.clone(),
                        output: msg.content.clone(),
                    }];
                    // function_call_output only accepts string; send images as
                    // a follow-up user message so the LLM can actually see them.
                    if msg
                        .media
                        .iter()
                        .any(|m| matches!(m, MediaContent::Image(_)))
                    {
                        let mut parts = vec![serde_json::json!({
                            "type": "input_text",
                            "text": format!("[Screenshot from tool '{}']",
                                msg.tool_name.as_deref().unwrap_or("unknown")),
                        })];
                        for img in msg.images() {
                            let data_url = format!("data:{};base64,{}", img.media_type, img.base64);
                            parts.push(serde_json::json!({
                                "type": "input_image",
                                "image_url": data_url,
                            }));
                        }
                        items.push(ResponsesInputItem::Message {
                            role: "user".to_string(),
                            content: serde_json::Value::Array(parts),
                        });
                    }
                    return items;
                }

                // Regular message
                let role = match msg.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => return vec![], // Handled above via tool_call_id
                };

                // Build content: array with images if present, plain string otherwise
                let content = if msg.media.is_empty() {
                    serde_json::Value::String(msg.content.clone())
                } else {
                    let mut parts = vec![serde_json::json!({
                        "type": "input_text",
                        "text": msg.content,
                    })];
                    // Audio never reaches here: `reject_audio` runs first.
                    for img in msg.images() {
                        let data_url = format!("data:{};base64,{}", img.media_type, img.base64);
                        parts.push(serde_json::json!({
                            "type": "input_image",
                            "image_url": data_url,
                        }));
                    }
                    serde_json::Value::Array(parts)
                };

                vec![ResponsesInputItem::Message {
                    role: role.to_string(),
                    content,
                }]
            })
            .collect()
    }

    /// Convert ToolDefinitions to Responses API tools
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<ResponsesTool> {
        tools
            .iter()
            .map(|t| ResponsesTool {
                tool_type: "function".to_string(),
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
                strict: false,
            })
            .collect()
    }

    /// Send request and parse response
    fn send_request(&self, request: &ResponsesRequest) -> Result<ResponsesResponse> {
        let url = "https://api.openai.com/v1/responses";
        let auth_header = format!("Bearer {}", self.api_key);

        tracing::debug!("Sending request to OpenAI Responses API");
        tracing::debug!("Model: {}", self.model);

        let response_result = self
            .http_agent
            .post(url)
            .set("Content-Type", "application/json")
            .set("Authorization", &auth_header)
            .send_json(request);

        let response: ResponsesResponse = match response_result {
            Ok(resp) => {
                let body = resp.into_string()?;
                tracing::debug!("Raw OpenAI response: {}", body);
                serde_json::from_str(&body).map_err(|e| {
                    tracing::error!("Failed to parse OpenAI response: {}", e);
                    tracing::error!("Response body: {}", body);
                    anyhow::anyhow!("Failed to read JSON: {}", e)
                })?
            }
            Err(ureq::Error::Status(code, resp)) => {
                let error_body = resp
                    .into_string()
                    .unwrap_or_else(|_| "Unable to read error body".to_string());
                tracing::error!("OpenAI API error (status {}): {}", code, error_body);
                return Err(anyhow::anyhow!("OpenAI API error {}: {}", code, error_body));
            }
            Err(e) => return Err(e.into()),
        };

        // Check if response is complete
        if response.status == "incomplete" {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|d| d.reason.clone())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(anyhow::anyhow!(
                "Response incomplete: {}. Consider increasing max_output_tokens.",
                reason
            ));
        }

        Ok(response)
    }

    /// Extract text content from response output
    fn extract_text(output: &[ResponseOutput]) -> Option<String> {
        output
            .iter()
            .find(|o| o.output_type == "message" || o.output_type == "text")
            .and_then(|o| {
                if let Some(ref text) = o.text {
                    return Some(text.clone());
                }
                o.content
                    .as_ref()
                    .and_then(|c| c.first())
                    .map(|c| c.text.clone())
            })
    }

    /// Extract reasoning from response output (checks content first, then summary)
    fn extract_reasoning(output: &[ResponseOutput]) -> Option<String> {
        let reasoning_items: Vec<&ResponseOutput> = output
            .iter()
            .filter(|o| o.output_type == "reasoning")
            .collect();

        if reasoning_items.is_empty() {
            return None;
        }

        // Try content first (primary reasoning text)
        let content_parts: Vec<&str> = reasoning_items
            .iter()
            .flat_map(|o| {
                o.content
                    .iter()
                    .flat_map(|c| c.iter().map(|r| r.text.as_str()))
            })
            .collect();

        if !content_parts.is_empty() {
            return Some(content_parts.join("\n"));
        }

        // Fall back to summary
        let summary_parts: Vec<&str> = reasoning_items
            .iter()
            .flat_map(|o| {
                o.summary
                    .iter()
                    .flat_map(|s| s.iter().map(|r| r.text.as_str()))
            })
            .collect();

        if !summary_parts.is_empty() {
            Some(summary_parts.join("\n"))
        } else {
            tracing::debug!("Reasoning items found but no content or summary text");
            None
        }
    }

    /// Convert API usage to TokenUsage
    fn convert_usage(usage: &Option<ResponseUsage>) -> Option<TokenUsage> {
        usage
            .as_ref()
            .map(|u| TokenUsage::single(u.input_tokens, u.output_tokens, u.total_tokens))
    }

    /// Extract tool calls from response output
    fn extract_tool_calls(output: &[ResponseOutput]) -> Vec<ToolCallInfo> {
        output
            .iter()
            .filter(|o| o.output_type == "function_call")
            .filter_map(|o| {
                let call_id = o.call_id.as_ref()?;
                let name = o.name.as_ref()?;
                let arguments_str = o.arguments.as_ref()?;
                let arguments: serde_json::Value =
                    serde_json::from_str(arguments_str).unwrap_or_default();

                Some(ToolCallInfo {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments,
                })
            })
            .collect()
    }
}

impl LlmProvider for OpenAiProvider {
    fn supports_structured_output(&self) -> bool {
        true
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let input = Self::convert_to_input_items(messages);

        let request = ResponsesRequest {
            model: self.model.clone(),
            input,
            temperature: self.temperature,
            max_output_tokens: Some(self.max_tokens),
            tools: None,
            text: None,
            reasoning: self.reasoning_param(),
        };

        let response = self.send_request(&request)?;

        Self::extract_text(&response.output)
            .ok_or_else(|| anyhow::anyhow!("No text content in response"))
    }

    fn chat_with_schema(
        &self,
        messages: &[ChatMessage],
        schema: serde_json::Value,
        schema_name: &str,
    ) -> Result<String> {
        let input = Self::convert_to_input_items(messages);

        let request = ResponsesRequest {
            model: self.model.clone(),
            input,
            temperature: self.temperature,
            max_output_tokens: Some(self.max_tokens),
            tools: None,
            text: Some(ResponseTextFormat {
                format: ResponseFormatSpec {
                    format_type: "json_schema".to_string(),
                    name: schema_name.to_string(),
                    schema,
                    strict: true,
                },
            }),
            reasoning: self.reasoning_param(),
        };

        tracing::debug!("Sending request to OpenAI Responses API with JSON Schema");

        let response = self.send_request(&request)?;

        Self::extract_text(&response.output)
            .ok_or_else(|| anyhow::anyhow!("No text content in response"))
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        // Images go over the wire as `input_image`; audio has no path here and
        // must not be dropped on the floor.
        reject_audio(messages, "the OpenAI backend")?;
        let input = Self::convert_to_input_items(messages);
        let wire_tools = Self::convert_tools(tools);

        let request = ResponsesRequest {
            model: self.model.clone(),
            input,
            temperature: self.temperature,
            max_output_tokens: Some(self.max_tokens),
            tools: if wire_tools.is_empty() {
                None
            } else {
                Some(wire_tools)
            },
            text: None,
            reasoning: self.reasoning_param(),
        };

        tracing::debug!("Sending chat_with_tools request to OpenAI Responses API");

        let response = self.send_request(&request)?;
        let usage = Self::convert_usage(&response.usage);

        if let Some(ref u) = usage {
            tracing::info!(
                "Token usage: input={}, output={}, total={}",
                u.input_tokens,
                u.output_tokens,
                u.total_tokens
            );
        }

        // Check for tool calls first
        let tool_calls = Self::extract_tool_calls(&response.output);
        if !tool_calls.is_empty() {
            tracing::info!("OpenAI returned {} tool calls", tool_calls.len());
            return Ok(LlmResponse::ToolCalls {
                calls: tool_calls,
                usage,
                reasoning: None,
            });
        }

        // Text response
        let text = Self::extract_text(&response.output)
            .ok_or_else(|| anyhow::anyhow!("No text content or tool calls in response"))?;
        let reasoning = Self::extract_reasoning(&response.output);
        tracing::debug!(
            "Response output types: {:?}",
            response
                .output
                .iter()
                .map(|o| &o.output_type)
                .collect::<Vec<_>>()
        );

        Ok(LlmResponse::Text {
            content: text,
            reasoning,
            usage,
        })
    }
}

// ============================================================================
// Factory function
// ============================================================================

/// Create LLM provider based on runtime configuration
///
/// Selection logic:
/// 1. If model_path is provided → local FFI (in-process llama.cpp)
/// 2. If api_key is provided → OpenAI (cloud)
/// 3. Otherwise → error
/// Build a ureq agent that honors `SSL_CERT_FILE` — a custom CA bundle, e.g. a
/// corporate TLS-intercepting proxy like Zscaler. Used by every outbound HTTPS
/// client (the LLM provider and the model downloader) so they all trust the
/// same roots. `redirects` overrides the redirect limit (None = ureq default).
/// Falls back to default TLS if the cert file is unset or unreadable.
pub(crate) fn http_agent_with_ca(redirects: Option<u32>) -> ureq::Agent {
    let mut builder = ureq::AgentBuilder::new();
    if let Some(r) = redirects {
        builder = builder.redirects(r);
    }
    if let Ok(cert_file) = std::env::var("SSL_CERT_FILE") {
        tracing::info!("Loading custom CA certificates from: {}", cert_file);
        match OpenAiProvider::build_tls_with_custom_ca(&cert_file) {
            Ok(tls) => {
                tracing::info!("Custom CA certificates loaded successfully");
                builder = builder.tls_connector(std::sync::Arc::new(tls));
            }
            Err(e) => {
                tracing::error!("Failed to load custom CA certificates: {}", e);
                tracing::warn!("Falling back to default TLS configuration");
            }
        }
    }
    builder.build()
}

/// Which local inference backend runs a `model_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceEngine {
    /// In-process llama.cpp (GGUF) via `llama-cpp-2` FFI — the `local` feature.
    LlamaCpp,
    /// Native pure-Rust candle engine — the `candle` feature.
    Candle,
    /// Replays a canned script from `model_path` instead of running a model.
    /// For testing everything that is not sampling — the app-server wire
    /// format, the ReAct loop, approvals — including from another process's CI.
    /// See [`crate::llm_scripted`].
    Scripted,
}

/// Resolve the local inference engine: explicit config (`llm.inference_engine`)
/// takes precedence, then the `INFERENCE_ENGINE` env var, else the default
/// (llama.cpp). The `model_path` is neutral to this choice — the same
/// `hf:`/local spec drives either backend. Both must be compiled in for a
/// switch without a rebuild.
pub fn resolve_inference_engine(explicit: Option<String>) -> InferenceEngine {
    let selector = explicit
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("INFERENCE_ENGINE").ok())
        .map(|s| s.trim().to_ascii_lowercase());

    match selector.as_deref() {
        Some("candle") => InferenceEngine::Candle,
        Some("scripted") => InferenceEngine::Scripted,
        Some("llamacpp") | Some("llama.cpp") | Some("llama-cpp") | Some("llama_cpp")
        | Some("llama") => InferenceEngine::LlamaCpp,
        Some(other) => {
            tracing::warn!(
                "Unknown inference_engine '{}' (expected 'llamacpp', 'candle', or \
                 'scripted'); using llamacpp",
                other
            );
            InferenceEngine::LlamaCpp
        }
        None => InferenceEngine::LlamaCpp,
    }
}

/// Parse `reasoningEffort` for a local backend (llama.cpp or candle — both
/// call this, since PR #139 covered llama.cpp and this extended the same
/// config key to candle). `None` for anything unrecognized (a typo, or an
/// OpenAI-only value like `"minimal"`) — logged once and treated as unset,
/// not a load failure: this is a soft quality knob, not a routing decision
/// like `[llm] profile`/`inference_engine`, and the same config value has a
/// legitimate OpenAI-only meaning this function has no opinion on.
///
/// Gated the same way `crate::profile` itself is (`lib.rs`): both of this
/// function's callers below are already behind `#[cfg(feature = "local")]` /
/// `#[cfg(feature = "candle")]`, but the function itself was not, so a build
/// with neither feature (klein-cli's "no model backends" CI build,
/// `cargo build --no-default-features`) still type-checked this body and
/// failed on `crate::profile` not existing at all in that configuration.
#[cfg(any(feature = "local", feature = "candle"))]
fn local_reasoning_effort(
    reasoning_effort: Option<&str>,
) -> Option<crate::profile::ReasoningEffort> {
    reasoning_effort.and_then(|s| match crate::profile::ReasoningEffort::parse(s) {
        Some(e) => Some(e),
        None => {
            tracing::warn!(
                "reasoningEffort '{s}' is not a recognized value for the local backend \
                 (low/medium/high/xhigh/max); ignoring — the model's own default applies"
            );
            None
        }
    })
}

/// Validate `topK`/`LLM_TOP_K` before it reaches `LlamaSampler::top_k`, which
/// takes a signed `i32`: an unchecked `as i32` cast would silently wrap a
/// `u32` above `i32::MAX` into a negative number. llama.cpp's own contract
/// for `llama_sampler_init_top_k` documents `k <= 0` as a no-op (verified
/// directly in `llama-sampler.cpp`, not assumed), so a wrapped value would
/// not crash — but relying on wraparound to reach that no-op is an accident,
/// not a design, and deserves its own explicit path. `0` is treated the same
/// as unset for the identical reason: passing it through would construct a
/// stage whose only effect is doing nothing, when omitting the stage (what
/// `None` already means for `top_p`) says the same thing plainly.
fn validated_top_k(top_k: Option<u32>) -> Option<u32> {
    match top_k {
        None | Some(0) => None,
        Some(k) if k > i32::MAX as u32 => {
            tracing::warn!(
                "topK {k} exceeds the sampler's range; clamping to {} \
                 (a value this large is effectively unrestricted anyway)",
                i32::MAX
            );
            Some(i32::MAX as u32)
        }
        Some(k) => Some(k),
    }
}

pub fn create_provider(
    model_path: Option<String>,
    // The llama.cpp backend's multimodal projector (`mmproj-*.gguf`). `None`
    // is text only. Ignored by every other engine: candle has no mtmd, and a
    // cloud model takes images over the wire.
    mmproj_path: Option<String>,
    _base_url: String,
    model: String,
    api_key: Option<String>,
    temperature: Option<f32>,
    // Nucleus-sampling threshold, llama.cpp backend only — see
    // `llm_local::LlamaLocalProvider::top_p` and the candle sampler's `top_p`.
    // Ignored by the cloud providers, which have their own.
    top_p: Option<f32>,
    // Top-k sampling cutoff, llama.cpp backend only — see
    // `llm_local::LlamaLocalProvider::top_k` and the candle sampler's `top_k`.
    // Ignored by the cloud providers, which have their own.
    top_k: Option<u32>,
    max_tokens: u32,
    reasoning_effort: Option<String>,
    inference_engine: Option<String>,
    // Where the native candle backend should find `tokenizer.json`. Ignored by
    // every other engine: llama.cpp reads the one inside the GGUF, and a cloud
    // model tokenizes server-side.
    tokenizer_path: Option<String>,
    // Layers to offload to the GPU, llama.cpp backend only. `None` leaves it
    // to llama.cpp's own default (999, offload everything); ignored by every
    // other engine, same as `mmproj_path`.
    gpu_layers: Option<u32>,
    // Move MoE expert tensors to CPU, llama.cpp backend only. Ignored by
    // every other engine.
    cpu_moe: bool,
    // Which model profile reads the model's output, `env > config` already
    // applied by the caller (`GALLIUM_PROFILE` / `[llm] profile`). `None` means
    // detect it from what the model file reports. llama.cpp backend only until
    // the candle path moves onto profiles too.
    profile: Option<String>,
) -> Result<Box<dyn LlmProvider>, anyhow::Error> {
    if let Some(ref path) = model_path {
        match resolve_inference_engine(inference_engine) {
            InferenceEngine::Scripted => {
                tracing::info!("Using scripted provider (no model) from '{}'", path);
                let provider =
                    crate::llm_scripted::ScriptedProvider::load(std::path::Path::new(path))?;
                return Ok(Box::new(provider));
            }
            InferenceEngine::Candle => {
                #[cfg(feature = "candle")]
                {
                    tracing::info!("Using native candle provider");
                    // Said out loud rather than dropped: this engine still
                    // dispatches through `protocol.rs`, so a configured profile
                    // changes nothing here. A setting that is silently ignored
                    // reads as a setting that did not work.
                    if let Some(name) = &profile {
                        tracing::warn!(
                            "[llm] profile '{name}' is ignored by the candle engine, which uses \
                             its own protocol adapters; it applies to the llama.cpp backend"
                        );
                    }
                    let provider = crate::llm_candle::load_candle_provider(
                        path,
                        temperature,
                        top_p,
                        validated_top_k(top_k),
                        max_tokens,
                        tokenizer_path.as_deref(),
                        local_reasoning_effort(reasoning_effort.as_deref()),
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to load candle model '{}': {}", path, e)
                    })?;
                    return Ok(Box::new(provider));
                }
                #[cfg(not(feature = "candle"))]
                anyhow::bail!(
                    "Candle inference engine not compiled in. Build with --features candle"
                );
            }
            InferenceEngine::LlamaCpp => {
                #[cfg(feature = "local")]
                {
                    tracing::info!("Using local llama.cpp provider (FFI)");
                    // Resolve `hf:` specs (download into the HF cache if needed);
                    // plain paths pass through unchanged.
                    let resolved = crate::model_downloader::ensure_model(path)
                        .with_context(|| format!("Failed to resolve model '{path}'"))?;
                    let resolved = resolved.to_string_lossy().to_string();
                    // The projector resolves the same way, and eagerly: a
                    // download failure should be reported while the provider is
                    // being built, not on the first turn that attaches an image.
                    let mmproj = mmproj_path
                        .as_deref()
                        .map(|spec| {
                            crate::model_downloader::ensure_model(spec).map_err(|e| {
                                anyhow::anyhow!("Failed to resolve mmproj '{}': {}", spec, e)
                            })
                        })
                        .transpose()?
                        .map(|p| p.to_string_lossy().to_string());
                    let temp = temperature.unwrap_or(0.7);
                    let provider = crate::llm_local::LlamaLocalProvider::new(
                        &resolved,
                        crate::llm_local::LocalModelOptions {
                            mmproj_path: mmproj.as_deref(),
                            temperature: temp,
                            top_p,
                            top_k: validated_top_k(top_k),
                            reasoning_effort: local_reasoning_effort(reasoning_effort.as_deref()),
                            max_tokens,
                            n_ctx: LOCAL_CONTEXT_WINDOW,
                            gpu_layers,
                            cpu_moe,
                            profile: profile.as_deref(),
                        },
                    )
                    .map_err(|e| {
                        tracing::error!("Failed to create local provider: {}", e);
                        anyhow::anyhow!("Failed to load model from {}: {}", resolved, e)
                    })?;
                    return Ok(Box::new(provider));
                }
                #[cfg(not(feature = "local"))]
                anyhow::bail!(
                    "Local (llama.cpp) inference engine not compiled in. Build with --features local"
                );
            }
        }
    }

    if let Some(key) = api_key {
        tracing::info!("Using OpenAI provider (API key provided)");
        Ok(Box::new(OpenAiProvider::new(
            key,
            model,
            temperature,
            max_tokens,
            reasoning_effort,
        )))
    } else {
        anyhow::bail!("No model_path or api_key provided. Set MODEL_PATH for local inference or OPENAI_API_KEY for cloud.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn top_k_zero_and_unset_both_skip_the_stage() {
        assert_eq!(validated_top_k(None), None);
        assert_eq!(validated_top_k(Some(0)), None);
    }

    #[test]
    fn top_k_above_i32_max_clamps_instead_of_wrapping() {
        // The bug this guards: `u32::MAX as i32` wraps to -1, which
        // llama.cpp's own contract treats as a no-op — an accident that
        // happens to be silently "safe," not a validated value.
        assert_eq!(validated_top_k(Some(u32::MAX)), Some(i32::MAX as u32));
        assert_eq!(
            validated_top_k(Some(i32::MAX as u32 + 1)),
            Some(i32::MAX as u32)
        );
    }

    #[test]
    fn top_k_within_range_passes_through_unchanged() {
        assert_eq!(validated_top_k(Some(64)), Some(64));
        assert_eq!(
            validated_top_k(Some(i32::MAX as u32)),
            Some(i32::MAX as u32)
        );
    }

    #[test]
    fn a_single_call_prices_both_halves() {
        // 100 prompt tokens in 0.5s, then 20 more tokens (the first came out of
        // prefill) in 2s.
        let usage = TokenUsage::timed(100, 21, 121, ms(500), ms(2000));
        assert_eq!(usage.prefill_rate(), Some(200.0));
        assert_eq!(usage.decode_rate(), Some(10.0));
    }

    #[test]
    fn accumulating_keeps_the_first_calls_ttft_and_sums_the_rest() {
        let mut total = TokenUsage::default();
        total.add(&TokenUsage::timed(100, 11, 111, ms(500), ms(1000)));
        total.add(&TokenUsage::timed(300, 21, 321, ms(1500), ms(1000)));

        let timing = total.timing.expect("both calls were timed");
        // The wait before the turn showed any sign of life is the first call's,
        // not the sum — 2s here would be a latency nobody experienced.
        assert_eq!(timing.ttft, ms(500));
        assert_eq!(timing.prefill, ms(2000));
        assert_eq!(timing.decode, ms(2000));
        assert_eq!(timing.calls, 2);
        assert_eq!(timing.prefill_tokens, 400);
        // 32 generated, less one first token per call.
        assert_eq!(timing.decode_tokens, 30);
        assert_eq!(total.decode_rate(), Some(15.0));
    }

    #[test]
    fn an_untimed_call_is_counted_but_not_priced() {
        let timed = TokenUsage::timed(100, 11, 111, ms(500), ms(1000));

        // Untimed first: the timed call's numbers survive intact.
        let mut a = TokenUsage::single(50, 5, 55);
        a.add(&timed);
        assert_eq!(a.timing.map(|t| t.ttft), Some(ms(500)));

        // Timed first: the untimed call's tokens raise the totals and change no
        // rate. Pricing them against the timed call's clock would report a
        // throughput this backend never achieved — and would look like a win.
        let mut b = timed.clone();
        b.add(&TokenUsage::single(50, 5, 55));
        assert_eq!(b.input_tokens, 150);
        assert_eq!(b.output_tokens, 16);
        assert_eq!(b.prefill_rate(), timed.prefill_rate());
        assert_eq!(b.decode_rate(), timed.decode_rate());
        let timing = b.timing.expect("still timed");
        assert_eq!(timing.calls, 1);
        assert_eq!(timing.prefill_tokens, 100);
        assert_eq!(timing.decode_tokens, 10);
    }

    #[test]
    fn nothing_to_divide_is_no_rate_rather_than_zero() {
        // A call that produced exactly one token: the first token is prefill's,
        // so decode measured nothing at all.
        let usage = TokenUsage::timed(10, 1, 11, ms(100), ms(0));
        assert_eq!(usage.decode_rate(), None);
        assert_eq!(fmt_rate(usage.decode_rate()), "n/a");
        assert_eq!(fmt_rate(usage.prefill_rate()), "100.0 tok/s");
    }

    #[test]
    fn a_provider_that_does_not_measure_reports_nothing() {
        let usage = TokenUsage::single(100, 20, 120);
        assert!(usage.timing.is_none());
        assert_eq!(usage.prefill_rate(), None);
        assert_eq!(usage.decode_rate(), None);
    }

    #[test]
    fn engine_explicit_config_wins() {
        // An explicit, non-empty selector short-circuits before the env var is
        // read, so these are deterministic regardless of the environment.
        for v in ["candle", "Candle", "CANDLE", "  candle "] {
            assert_eq!(
                resolve_inference_engine(Some(v.to_string())),
                InferenceEngine::Candle
            );
        }
        for v in [
            "llamacpp",
            "llama.cpp",
            "llama-cpp",
            "llama_cpp",
            "llama",
            "LlamaCpp",
        ] {
            assert_eq!(
                resolve_inference_engine(Some(v.to_string())),
                InferenceEngine::LlamaCpp
            );
        }
    }

    #[test]
    fn engine_unknown_selector_defaults_to_llamacpp() {
        // An unknown (but non-empty) selector is consumed (env not read) and
        // falls back to the default backend.
        assert_eq!(
            resolve_inference_engine(Some("bogus".to_string())),
            InferenceEngine::LlamaCpp
        );
    }

    /// `gallium` was this engine's name before it was renamed to `candle` (the
    /// old name said nothing — llama.cpp is in gallium too). The rename was
    /// deliberately hard, with no alias, so the old value is now merely unknown:
    /// it warns and runs llama.cpp. Pinned because it is a silent change of
    /// engine for anyone with an old config, and should stay a conscious choice
    /// rather than drift back in as an accident.
    #[test]
    fn the_old_gallium_selector_is_no_longer_the_candle_engine() {
        assert_eq!(
            resolve_inference_engine(Some("gallium".to_string())),
            InferenceEngine::LlamaCpp
        );
    }

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
    fn test_convert_user_message_with_images() {
        let mut msg = ChatMessage::user("describe this".to_string());
        msg.media = vec![MediaContent::Image(ImageContent {
            base64: "AAAA".to_string(),
            media_type: "image/png".to_string(),
        })];

        let items = OpenAiProvider::convert_to_input_items(&[msg]);

        assert_eq!(items.len(), 1);
        let json = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "user");

        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "describe this");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn test_convert_tool_result_without_images() {
        let msg = ChatMessage::tool_result(
            "call_1".to_string(),
            "my_tool".to_string(),
            "result text".to_string(),
        );

        let items = OpenAiProvider::convert_to_input_items(&[msg]);

        assert_eq!(items.len(), 1);
        let json = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(json["type"], "function_call_output");
        assert_eq!(json["call_id"], "call_1");
        assert_eq!(json["output"], "result text");
    }

    #[test]
    fn test_convert_tool_result_with_images_emits_two_items() {
        let msg = ChatMessage::tool_result_with_images(
            "call_42".to_string(),
            "capture_screen".to_string(),
            "Window: Chrome, Size: 1920x1080".to_string(),
            vec![ImageContent {
                base64: "iVBORw0KGgo=".to_string(),
                media_type: "image/png".to_string(),
            }],
        );

        let items = OpenAiProvider::convert_to_input_items(&[msg]);

        // Should produce 2 items: function_call_output + user message with image
        assert_eq!(
            items.len(),
            2,
            "Expected 2 items: function_call_output + image message"
        );

        // First: the function output (text only)
        let fco = serde_json::to_value(&items[0]).unwrap();
        assert_eq!(fco["type"], "function_call_output");
        assert_eq!(fco["call_id"], "call_42");
        assert_eq!(fco["output"], "Window: Chrome, Size: 1920x1080");

        // Second: user message with the image
        let img_msg = serde_json::to_value(&items[1]).unwrap();
        assert_eq!(img_msg["type"], "message");
        assert_eq!(img_msg["role"], "user");

        let content = img_msg["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "input_text");
        assert!(content[0]["text"]
            .as_str()
            .unwrap()
            .contains("capture_screen"));
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(
            content[1]["image_url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
    }

    #[test]
    fn test_convert_tool_result_with_multiple_images() {
        let msg = ChatMessage::tool_result_with_images(
            "call_99".to_string(),
            "multi_capture".to_string(),
            "Two screenshots".to_string(),
            vec![
                ImageContent {
                    base64: "IMG1".to_string(),
                    media_type: "image/png".to_string(),
                },
                ImageContent {
                    base64: "IMG2".to_string(),
                    media_type: "image/jpeg".to_string(),
                },
            ],
        );

        let items = OpenAiProvider::convert_to_input_items(&[msg]);
        assert_eq!(items.len(), 2);

        let img_msg = serde_json::to_value(&items[1]).unwrap();
        let content = img_msg["content"].as_array().unwrap();
        // 1 text + 2 images = 3 parts
        assert_eq!(content.len(), 3);
        assert_eq!(content[1]["image_url"], "data:image/png;base64,IMG1");
        assert_eq!(content[2]["image_url"], "data:image/jpeg;base64,IMG2");
    }

    #[test]
    fn test_convert_full_tool_call_roundtrip() {
        // Simulate: user asks -> assistant calls tool -> tool returns with image -> messages
        let msgs = vec![
            ChatMessage::user("capture Chrome".to_string()),
            ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
                id: "call_1".to_string(),
                name: "capture_screen".to_string(),
                arguments: serde_json::json!({"process_name": "Chrome"}),
            }]),
            ChatMessage::tool_result_with_images(
                "call_1".to_string(),
                "capture_screen".to_string(),
                "Window: Chrome".to_string(),
                vec![ImageContent {
                    base64: "SCREENSHOT".to_string(),
                    media_type: "image/png".to_string(),
                }],
            ),
        ];

        let items = OpenAiProvider::convert_to_input_items(&msgs);

        // user message + function_call + function_call_output + image message = 4
        assert_eq!(items.len(), 4);

        let json: Vec<_> = items
            .iter()
            .map(|i| serde_json::to_value(i).unwrap())
            .collect();

        assert_eq!(json[0]["type"], "message");
        assert_eq!(json[0]["role"], "user");

        assert_eq!(json[1]["type"], "function_call");
        assert_eq!(json[1]["name"], "capture_screen");

        assert_eq!(json[2]["type"], "function_call_output");
        assert_eq!(json[2]["output"], "Window: Chrome");

        assert_eq!(json[3]["type"], "message");
        assert_eq!(json[3]["role"], "user");
        let content = json[3]["content"].as_array().unwrap();
        assert_eq!(content[1]["type"], "input_image");
        assert!(content[1]["image_url"]
            .as_str()
            .unwrap()
            .contains("SCREENSHOT"));
    }
}
