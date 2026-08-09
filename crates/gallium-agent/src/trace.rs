//! What a turn did, written down.
//!
//! Everything a turn produced used to be observable only while it ran: the
//! events go to whoever is watching, `tracing` lines go to a log that keeps no
//! structure, and once the turn returns the only survivor is its final text.
//! That is enough to *use* the agent and not enough to work on it. Asking why a
//! model called `write` with the wrong path, or which approval refused an edit,
//! meant reproducing the whole run against a model that answers differently
//! every time.
//!
//! A trace is one turn, recorded whole: the prompt the model was given, the
//! tools it was offered, what it answered, every tool call with its arguments
//! and result, the approval decisions those calls provoked, token usage, and
//! how the turn ended — including [`TurnEnding::Cancelled`], which is a way for
//! a turn to end and not a reason to have no record of it.
//!
//! # A trace is also a script
//!
//! The recorded model outputs are exactly what [`crate::llm_scripted`] replays.
//! [`TurnTrace::to_script`] projects one into a [`Script`], so a recorded turn
//! can be run again through the real binary with real tools and no model, and
//! [`TurnTrace::diff`] says where the second run diverged from the first. That
//! is what makes a trace a regression fixture rather than a log file.
//!
//! `diff` compares what is meant to be reproducible — the sequence of tool
//! calls with their arguments, the approval outcomes, and the final text — and
//! ignores what is not: timings, token counts, and tool result bodies, which
//! embed absolute paths and would differ between two runs for reasons that say
//! nothing about the agent.
//!
//! # What is not in here
//!
//! **The model's pre-parse output.** Tool calls are extracted inside the
//! providers (see [`crate::gemma`]), so what the ReAct loop sees — and what this
//! records — is the parsed [`LlmResponse`]. Capturing the raw string means
//! widening `LlmResponse` and touching every provider; it is worth doing the day
//! a tool-call parse bug costs an afternoon, and not before.
//!
//! **Later iterations' prompts.** The first model call's messages are recorded
//! in full; the prompts after it are that list plus the tool transcript already
//! in [`TurnTrace::steps`], so recording each one whole would multiply the
//! largest thing in the file by the iteration count to say nothing new.
//!
//! # Traces are local, and they hold everything the model saw
//!
//! File contents a tool read, the system prompt, whatever the user typed. A
//! trace belongs next to the workspace it came from, not in a bug report.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::approval::{ApprovalBroker, ApprovalPolicy, ApprovalRecord};
use crate::llm::{
    resolve_inference_engine, ChatMessage, ChatRole, LlmResponse, TokenUsage, ToolCallInfo,
    ToolDefinition,
};
use crate::llm_scripted::{Script, ScriptStep, ScriptToolCall};
use crate::tool::{ToolContent, ToolResult};

/// Bumped when the format changes in a way a reader has to know about, so a
/// replayer meeting a future trace can say so rather than misread it.
pub const TRACE_FORMAT_VERSION: u32 = 1;

/// How much of any one text is kept. A `read` of a large file otherwise makes
/// the trace bigger than the thing it describes, and past this point nobody is
/// reading it anyway — what is kept says how much was dropped.
const MAX_CAPTURED_CHARS: usize = 16 * 1024;

/// Where traces go when nothing says otherwise: beside the workspace they
/// describe, since that is the thing they are about and the thing whose
/// `.gitignore` can exclude them.
pub const DEFAULT_TRACE_DIR: &str = ".gallium/traces";

// ============================================================================
// The record
// ============================================================================

/// One turn, recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnTrace {
    pub version: u32,
    pub turn_id: String,
    /// Milliseconds since the epoch. Not a formatted timestamp: the crate has
    /// no date dependency, and 13 digits sort correctly in a directory listing,
    /// which is the only thing the name has to do.
    pub started_at_epoch_ms: u64,
    pub duration_ms: u64,
    /// Which backend ran the turn, and which model it asked for.
    pub engine: String,
    pub model: String,
    /// The approval policy in force, in the wording the startup banner uses.
    pub approval_policy: String,
    pub workspace_root: String,
    /// What the user typed.
    pub user_input: String,
    /// The messages handed to the model on the first iteration — system prompt,
    /// skill catalog, history, and the user message — as it saw them.
    pub prompt: Vec<MessageRecord>,
    /// The catalog offered, recorded once: it is computed once per turn and
    /// does not change between iterations.
    pub tools: Vec<ToolDefinitionRecord>,
    /// Messages compaction dropped before the turn ran.
    pub compacted: usize,
    pub steps: Vec<TraceStep>,
    /// The turn's total. Zero for a turn that failed or was stopped, since the
    /// loop reports a total only on the way out — what each call cost is still
    /// on the steps.
    pub usage: UsageRecord,
    pub ending: TurnEnding,
}

/// One ReAct iteration: a model call and the tool calls it asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceStep {
    pub iteration: u32,
    pub response: ModelOutput,
    pub usage: Option<UsageRecord>,
    /// How long the model call took. The tool calls have their own.
    pub duration_ms: u64,
    pub calls: Vec<ToolCallRecord>,
}

/// What the model answered, after the provider parsed it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ModelOutput {
    Text {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    ToolCalls {
        calls: Vec<ToolCallRequest>,
    },
}

/// A call the model asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// A call that ran: what it was asked, who approved it, what came back.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    /// The decisions this call provoked, in the order the broker made them. A
    /// read-only call has none: that tier is never asked about, so recording a
    /// decision for it would be recording a question nobody was asked.
    pub approvals: Vec<ApprovalRecordJson>,
    /// Absent when the call never returned one: the turn was cancelled while it
    /// was running, so it was abandoned mid-flight. The call is still recorded —
    /// it ran, and it may already have been approved, which is exactly what
    /// someone reading back an interrupted turn needs to see.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ToolResultRecord>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultRecord {
    pub model_text: String,
    /// Only when the tool offered a different one; absent means the model text
    /// is what a UI showed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_text: Option<String>,
    /// Media type and size, not the bytes. A base64 screenshot would be most of
    /// the file and none of the information.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageRecord>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRecord {
    pub media_type: String,
    pub base64_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageRecord {
    pub role: ChatRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Attachment *counts*, not payloads: a base64 image is megabytes that
    /// would dwarf the rest of the file, and a trace replays as a script of
    /// tool calls, which no attachment participates in.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub images: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub audio: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinitionRecord {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageRecord {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub peak_input_tokens: u64,
    /// Absent from a provider that cannot time itself, and absent from every
    /// trace written before timing existed — which is why it is optional rather
    /// than zeroed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingRecord>,
}

/// Wall clock for the calls a [`UsageRecord`] covers.
///
/// Milliseconds rather than a `Duration` or a float: it keeps the record `Eq`
/// (a trace is compared, not just read), and it is the unit anything consuming
/// these files for tuning will plot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingRecord {
    /// The first call's time to first token.
    pub ttft_ms: u64,
    /// Summed over calls.
    pub prefill_ms: u64,
    /// Summed over calls.
    pub decode_ms: u64,
    /// Prompt tokens evaluated during `prefill_ms`. Not necessarily
    /// `input_tokens`: a turn whose calls were not all timed reports every
    /// call's prompt there and only the timed ones here, which is what lets a
    /// rate be computed from this record alone.
    pub prefill_tokens: u64,
    /// Tokens sampled during `decode_ms` — the timed calls' output, less each
    /// call's first.
    pub decode_tokens: u64,
    pub calls: u32,
}

/// One approval decision, as the trace stores it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRecordJson {
    pub action: String,
    pub target: String,
    pub risk: String,
    pub outcome: String,
}

/// How a turn ended. Cancellation is one of the three: a turn stopped on
/// request is exactly the case worth having a record of.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum TurnEnding {
    Completed {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    Cancelled,
    Failed {
        message: String,
    },
}

impl TurnEnding {
    /// The word `diff` compares, so an ending that differs is reported by kind
    /// rather than by its whole payload.
    fn status(&self) -> &'static str {
        match self {
            Self::Completed { .. } => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed { .. } => "failed",
        }
    }
}

// ============================================================================
// Capture helpers
// ============================================================================

/// Keep at most [`MAX_CAPTURED_CHARS`], saying how much was dropped. Truncating
/// silently would let a `diff` claim two texts match when it only ever saw
/// their first 16 KiB.
fn capture(text: &str) -> String {
    if text.len() <= MAX_CAPTURED_CHARS {
        return text.to_string();
    }
    let mut end = MAX_CAPTURED_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n… (trace truncated: {} of {} bytes)",
        &text[..end],
        end,
        text.len()
    )
}

impl From<&ToolCallInfo> for ToolCallRequest {
    fn from(call: &ToolCallInfo) -> Self {
        Self {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        }
    }
}

impl From<&ChatMessage> for MessageRecord {
    fn from(message: &ChatMessage) -> Self {
        Self {
            role: message.role.clone(),
            content: capture(&message.content),
            tool_calls: message
                .tool_calls
                .as_ref()
                .map(|calls| calls.iter().map(Into::into).collect()),
            tool_call_id: message.tool_call_id.clone(),
            tool_name: message.tool_name.clone(),
            images: message.media_counts().0,
            audio: message.media_counts().1,
        }
    }
}

impl From<&ToolDefinition> for ToolDefinitionRecord {
    fn from(def: &ToolDefinition) -> Self {
        Self {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
        }
    }
}

impl From<&TokenUsage> for UsageRecord {
    fn from(usage: &TokenUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            peak_input_tokens: usage.peak_input_tokens,
            timing: usage.timing.map(|t| TimingRecord {
                ttft_ms: t.ttft.as_millis() as u64,
                prefill_ms: t.prefill.as_millis() as u64,
                decode_ms: t.decode.as_millis() as u64,
                prefill_tokens: t.prefill_tokens,
                decode_tokens: t.decode_tokens,
                calls: t.calls,
            }),
        }
    }
}

impl From<&ToolResult> for ToolResultRecord {
    fn from(result: &ToolResult) -> Self {
        let model_text = result.model_text().into_owned();
        let display = result.display_text().into_owned();
        Self {
            display_text: (display != model_text).then(|| capture(&display)),
            model_text: capture(&model_text),
            images: result
                .model_content
                .iter()
                .filter_map(|part| match part {
                    ToolContent::Image(image) => Some(ImageRecord {
                        media_type: image.media_type.clone(),
                        base64_len: image.base64.len(),
                    }),
                    ToolContent::Text(_) => None,
                })
                .collect(),
            is_error: result.is_error,
        }
    }
}

impl From<&ApprovalRecord> for ApprovalRecordJson {
    fn from(record: &ApprovalRecord) -> Self {
        Self {
            action: record.action.clone(),
            target: capture(&record.target),
            risk: record.risk.label().to_string(),
            outcome: record.outcome.label().to_string(),
        }
    }
}

// ============================================================================
// Recording
// ============================================================================

/// What tracing needs to know that is true for a whole session rather than for
/// one turn.
#[derive(Debug, Clone)]
pub struct TraceMeta {
    pub engine: String,
    pub model: String,
    pub workspace_root: String,
    pub approval_policy: String,
}

impl TraceMeta {
    pub fn new(
        engine: String,
        model: String,
        workspace_root: String,
        policy: ApprovalPolicy,
    ) -> Self {
        Self {
            engine,
            model,
            workspace_root,
            approval_policy: policy.to_string(),
        }
    }

    /// The engine label, resolved the same way the provider itself is chosen,
    /// so a trace cannot claim a backend the turn did not run on.
    pub fn engine_label(model_path: Option<&str>, inference_engine: Option<String>) -> String {
        match model_path {
            // No local model: the OpenAI-compatible HTTP provider.
            None => "openai".to_string(),
            Some(_) => format!("{:?}", resolve_inference_engine(inference_engine)).to_lowercase(),
        }
    }
}

/// Where a session's traces go, and what every one of them says about the
/// session.
///
/// Built once per REPL run or per app-server thread — the same scope as the
/// [`ApprovalBroker`] whose decisions it attributes, which is what makes the
/// attribution unambiguous: one broker, one turn at a time.
pub struct TraceSession {
    dir: PathBuf,
    meta: TraceMeta,
    /// Consulted after each tool call for the decisions that call provoked.
    broker: Option<Arc<ApprovalBroker>>,
    /// Names a turn whose caller has no id of its own — the REPL.
    next_turn: AtomicU64,
}

impl TraceSession {
    pub fn new(dir: PathBuf, meta: TraceMeta, broker: Option<Arc<ApprovalBroker>>) -> Self {
        Self {
            dir,
            meta,
            broker,
            next_turn: AtomicU64::new(1),
        }
    }

    /// Resolve whether to trace, and where to.
    ///
    /// `GALLIUM_TRACE_DIR` names a directory and turns tracing on;
    /// `GALLIUM_TRACE` is the on/off switch for the default location or for a
    /// directory the config named. An explicit `GALLIUM_TRACE=0` wins over
    /// everything, because a run has to be able to say "not this one" without
    /// editing a config file.
    ///
    /// `configured_dir` is the config file's `[agent.trace] dir`, already
    /// resolved against the config's own directory by the caller.
    pub fn from_env(
        configured_dir: Option<PathBuf>,
        meta: TraceMeta,
        broker: Option<Arc<ApprovalBroker>>,
    ) -> Option<Self> {
        let switch = std::env::var("GALLIUM_TRACE").ok().map(|s| {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
        if switch == Some(false) {
            return None;
        }

        let env_dir = std::env::var("GALLIUM_TRACE_DIR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from);

        let dir = match (env_dir, configured_dir, switch) {
            (Some(dir), _, _) => dir,
            (None, Some(dir), _) => dir,
            (None, None, Some(true)) => Path::new(&meta.workspace_root).join(DEFAULT_TRACE_DIR),
            (None, None, _) => return None,
        };

        Some(Self::new(dir, meta, broker))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A recorder for one turn. `turn_id` is the caller's own name for it — the
    /// app-server has one and it is worth keeping; the REPL does not, and gets a
    /// sequence number.
    pub(crate) fn recorder(&self, turn_id: Option<&str>) -> Arc<TurnRecorder> {
        let turn_id = turn_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("turn-{}", self.next_turn.fetch_add(1, Ordering::SeqCst)));
        Arc::new(TurnRecorder::new(
            turn_id,
            self.meta.clone(),
            self.broker.clone(),
        ))
    }

    /// Write one trace, returning where it landed.
    ///
    /// A failure is reported and swallowed: tracing observes a turn, and
    /// something that can fail the turn it is observing is worse than no trace.
    pub(crate) fn write(&self, trace: &TurnTrace) -> Option<PathBuf> {
        let name = format!(
            "turn-{}-{}.json",
            trace.started_at_epoch_ms,
            sanitize(&trace.turn_id)
        );
        let path = self.dir.join(name);
        match write_trace(&path, trace) {
            Ok(()) => {
                tracing::debug!("wrote turn trace {}", path.display());
                Some(path)
            }
            Err(e) => {
                tracing::warn!("could not write turn trace {}: {}", path.display(), e);
                None
            }
        }
    }
}

fn write_trace(path: &Path, trace: &TurnTrace) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(trace)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Keep a turn id usable as a filename. Ids are ours today, but an id from a
/// driving client is not.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Collects one turn as it runs.
///
/// Reached through [`crate::TurnContext`], so the ReAct loop records without any
/// call site having to pass a recorder around. Every method takes `&self`: the
/// turn holds one `Arc` of this and the loop is single-threaded within it, so
/// the lock is never contended and is there to satisfy sharing, not to
/// coordinate.
pub struct TurnRecorder {
    turn_id: String,
    meta: TraceMeta,
    broker: Option<Arc<ApprovalBroker>>,
    started: Instant,
    started_at_epoch_ms: u64,
    state: Mutex<Recording>,
}

impl std::fmt::Debug for TurnRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnRecorder")
            .field("turn_id", &self.turn_id)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct Recording {
    user_input: String,
    compacted: usize,
    prompt: Vec<MessageRecord>,
    tools: Vec<ToolDefinitionRecord>,
    steps: Vec<TraceStep>,
}

impl TurnRecorder {
    fn new(turn_id: String, meta: TraceMeta, broker: Option<Arc<ApprovalBroker>>) -> Self {
        // Start from an empty journal. The broker is session-scoped, so anything
        // still queued was decided before this turn began — by a turn that ended
        // without draining it, or during a stretch with tracing off — and
        // attributing it to this turn's first tool call would be a lie in the
        // one direction a trace must not lie.
        if let Some(broker) = &broker {
            broker.take_journal();
        }
        Self {
            turn_id,
            meta,
            broker,
            started: Instant::now(),
            started_at_epoch_ms: epoch_ms(),
            state: Mutex::new(Recording::default()),
        }
    }

    /// What the user asked, and what compaction cost to make room for it.
    pub(crate) fn record_input(&self, user_input: &str, compacted: usize) {
        let mut state = self.lock();
        state.user_input = capture(user_input);
        state.compacted = compacted;
    }

    /// The first model call's messages and the catalog it was offered. Called
    /// once per turn: later prompts are this list plus the recorded transcript.
    pub(crate) fn record_prompt(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) {
        let mut state = self.lock();
        if !state.prompt.is_empty() {
            return;
        }
        state.prompt = messages.iter().map(Into::into).collect();
        state.tools = tools.iter().map(Into::into).collect();
    }

    /// One model call and what it answered.
    pub(crate) fn record_response(
        &self,
        iteration: u32,
        response: &LlmResponse,
        elapsed: Duration,
    ) {
        let (output, usage) = match response {
            LlmResponse::Text {
                content,
                reasoning,
                usage,
            } => (
                ModelOutput::Text {
                    content: capture(content),
                    reasoning: reasoning.as_deref().map(capture),
                },
                usage.as_ref().map(Into::into),
            ),
            LlmResponse::ToolCalls(calls, usage) => (
                ModelOutput::ToolCalls {
                    calls: calls.iter().map(Into::into).collect(),
                },
                usage.as_ref().map(Into::into),
            ),
        };

        self.lock().steps.push(TraceStep {
            iteration,
            response: output,
            usage,
            duration_ms: millis(elapsed),
            calls: Vec::new(),
        });
    }

    /// One executed tool call, with whatever the broker decided while it ran.
    ///
    /// The decisions are drained from the broker rather than passed in: they are
    /// made deep inside a `Tool::call` that takes no turn context, and threading
    /// one through every tool impl to carry them out would be a large change to
    /// record a small fact.
    pub(crate) fn record_tool_call(
        &self,
        call: &ToolCallInfo,
        result: &ToolResult,
        elapsed: Duration,
    ) {
        let approvals: Vec<ApprovalRecordJson> = self
            .broker
            .as_ref()
            .map(|b| b.take_journal().iter().map(Into::into).collect())
            .unwrap_or_default();

        let record = ToolCallRecord {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            approvals,
            result: Some(result.into()),
            duration_ms: millis(elapsed),
        };

        self.attach(call, record);
    }

    /// A call the turn was cancelled out from under. Recorded with no result,
    /// and — the part that matters beyond the trace — it drains the approval
    /// journal, so a decision this call provoked is attributed to it rather than
    /// to the first tool call of the next turn. The broker is session-scoped and
    /// its journal outlives any one turn.
    pub(crate) fn record_cancelled_tool_call(&self, call: &ToolCallInfo, elapsed: Duration) {
        let approvals: Vec<ApprovalRecordJson> = self
            .broker
            .as_ref()
            .map(|b| b.take_journal().iter().map(Into::into).collect())
            .unwrap_or_default();

        self.attach(
            call,
            ToolCallRecord {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                approvals,
                result: None,
                duration_ms: millis(elapsed),
            },
        );
    }

    fn attach(&self, call: &ToolCallInfo, record: ToolCallRecord) {
        let mut state = self.lock();
        match state.steps.last_mut() {
            Some(step) => step.calls.push(record),
            // A tool call with no model call before it cannot happen through the
            // ReAct loop; dropping it silently would be the wrong answer if it
            // ever did.
            None => tracing::warn!(
                "trace: tool call '{}' with no step to attach it to",
                call.name
            ),
        }
    }

    /// Close the turn. Every ending produces a trace, cancellation included.
    pub(crate) fn finish(&self, ending: TurnEnding, usage: &TokenUsage) -> TurnTrace {
        let state = self.lock();
        TurnTrace {
            version: TRACE_FORMAT_VERSION,
            turn_id: self.turn_id.clone(),
            started_at_epoch_ms: self.started_at_epoch_ms,
            duration_ms: millis(self.started.elapsed()),
            engine: self.meta.engine.clone(),
            model: self.meta.model.clone(),
            approval_policy: self.meta.approval_policy.clone(),
            workspace_root: self.meta.workspace_root.clone(),
            user_input: state.user_input.clone(),
            prompt: state.prompt.clone(),
            tools: state.tools.clone(),
            compacted: state.compacted,
            steps: state.steps.clone(),
            usage: usage.into(),
            ending,
        }
    }

    /// A poisoned lock is a panic in another thread's recording, not a reason to
    /// take the turn down with it — the trace is the least important thing here.
    fn lock(&self) -> std::sync::MutexGuard<'_, Recording> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn millis(d: Duration) -> u64 {
    d.as_millis() as u64
}

// ============================================================================
// Replay
// ============================================================================

/// One place two traces disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// Where, in words: `"step 2, call 1: name"`.
    pub at: String,
    pub expected: String,
    pub actual: String,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: expected {}, got {}",
            self.at, self.expected, self.actual
        )
    }
}

impl TurnTrace {
    /// Load a trace from disk.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading trace '{}': {}", path.display(), e))?;
        let trace: Self = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("parsing trace '{}': {}", path.display(), e))?;
        if trace.version > TRACE_FORMAT_VERSION {
            anyhow::bail!(
                "trace '{}' is format version {}, and this build reads up to {}",
                path.display(),
                trace.version,
                TRACE_FORMAT_VERSION
            );
        }
        Ok(trace)
    }

    /// The model outputs, as a script that replays this turn without a model.
    ///
    /// Only the model's side is scripted. The tools run for real against
    /// whatever workspace the replay is pointed at, which is the point: a replay
    /// that also replayed the tool results would only be testing the trace
    /// reader.
    pub fn to_script(&self) -> Script {
        Script {
            steps: self
                .steps
                .iter()
                .map(|step| match &step.response {
                    ModelOutput::Text { content, reasoning } => ScriptStep {
                        text: Some(content.clone()),
                        reasoning: reasoning.clone(),
                        tool_calls: Vec::new(),
                        input_tokens: step.usage.as_ref().map(|u| u.input_tokens),
                    },
                    ModelOutput::ToolCalls { calls } => ScriptStep {
                        text: None,
                        reasoning: None,
                        tool_calls: calls
                            .iter()
                            .map(|c| ScriptToolCall {
                                id: c.id.clone(),
                                name: c.name.clone(),
                                arguments: c.arguments.clone(),
                            })
                            .collect(),
                        input_tokens: step.usage.as_ref().map(|u| u.input_tokens),
                    },
                })
                .collect(),
        }
    }

    /// Where `other` diverged from this trace.
    ///
    /// Compares what a replay is supposed to reproduce: the tool calls with
    /// their arguments, in order; the approval outcome for each; and how the
    /// turn ended, including its text. Deliberately not compared are timings,
    /// token counts, and tool result bodies — a result names absolute paths, so
    /// two runs in two directories differ there for reasons that say nothing
    /// about the agent, and a diff that reported them would be one nobody reads.
    ///
    /// Tool call **ids** are not compared either: a real model invents fresh
    /// ones every run.
    pub fn diff(&self, other: &Self) -> Vec<Divergence> {
        let mut out = Vec::new();

        if self.steps.len() != other.steps.len() {
            out.push(Divergence {
                at: "steps".to_string(),
                expected: self.steps.len().to_string(),
                actual: other.steps.len().to_string(),
            });
        }

        for (i, (a, b)) in self.steps.iter().zip(other.steps.iter()).enumerate() {
            let step = i + 1;
            diff_response(step, &a.response, &b.response, &mut out);

            if a.calls.len() != b.calls.len() {
                out.push(Divergence {
                    at: format!("step {step}: tool calls"),
                    expected: a.calls.len().to_string(),
                    actual: b.calls.len().to_string(),
                });
            }
            for (j, (x, y)) in a.calls.iter().zip(b.calls.iter()).enumerate() {
                let at = format!("step {step}, call {}", j + 1);
                if x.name != y.name {
                    out.push(Divergence {
                        at: format!("{at}: name"),
                        expected: x.name.clone(),
                        actual: y.name.clone(),
                    });
                }
                if x.arguments != y.arguments {
                    out.push(Divergence {
                        at: format!("{at}: arguments"),
                        expected: x.arguments.to_string(),
                        actual: y.arguments.to_string(),
                    });
                }
                // The tier and the answer, not the wording: an error message
                // that gains a comma is not a behavior change, and a call that
                // stopped being asked about is.
                let outcomes = |c: &ToolCallRecord| {
                    c.approvals
                        .iter()
                        .map(|a| format!("{}={}", a.risk, a.outcome))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let (xa, ya) = (outcomes(x), outcomes(y));
                if xa != ya {
                    out.push(Divergence {
                        at: format!("{at}: approvals"),
                        expected: xa,
                        actual: ya,
                    });
                }
                // An absent result means the call was abandoned when the turn
                // was cancelled. A replay that completed a call the recording
                // never finished (or the reverse) is a divergence, not a detail.
                let outcome = |c: &ToolCallRecord| match &c.result {
                    Some(r) if r.is_error => "error",
                    Some(_) => "ok",
                    None => "cancelled",
                };
                if outcome(x) != outcome(y) {
                    out.push(Divergence {
                        at: format!("{at}: outcome"),
                        expected: outcome(x).to_string(),
                        actual: outcome(y).to_string(),
                    });
                }
            }
        }

        if self.ending.status() != other.ending.status() {
            out.push(Divergence {
                at: "ending".to_string(),
                expected: self.ending.status().to_string(),
                actual: other.ending.status().to_string(),
            });
        } else if let (
            TurnEnding::Completed { text: a, .. },
            TurnEnding::Completed { text: b, .. },
        ) = (&self.ending, &other.ending)
        {
            if a != b {
                out.push(Divergence {
                    at: "ending: text".to_string(),
                    expected: a.clone(),
                    actual: b.clone(),
                });
            }
        }

        out
    }
}

fn diff_response(step: usize, a: &ModelOutput, b: &ModelOutput, out: &mut Vec<Divergence>) {
    let kind = |o: &ModelOutput| match o {
        ModelOutput::Text { .. } => "text",
        ModelOutput::ToolCalls { .. } => "toolCalls",
    };
    if kind(a) != kind(b) {
        out.push(Divergence {
            at: format!("step {step}: response"),
            expected: kind(a).to_string(),
            actual: kind(b).to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::approval::{ApprovalPolicy, ApprovalRule};
    use crate::llm_scripted::ScriptedProvider;
    use crate::runtime::{run_turn, TurnSetup};
    use crate::skill::SkillRegistry;
    use crate::tool::{create_default_registry_with_session, ToolRegistry, ToolSession};
    use crate::{CancellationToken, TurnContext};

    /// A turn that reads a file and then writes one — a tool call that only
    /// observes, a tool call that has to be approved, and a final answer.
    const READ_THEN_WRITE: &str = r#"{
        "steps": [
            { "toolCalls": [{ "id": "c1", "name": "Read", "arguments": { "file_path": "input.txt" } }] },
            { "toolCalls": [{ "id": "c2", "name": "Write",
                              "arguments": { "file_path": "out.txt", "content": "written" } }] },
            { "text": "Wrote out.txt.", "inputTokens": 40 }
        ]
    }"#;

    /// A workspace with `input.txt` in it, the registry over it, and the broker
    /// the trace will collect decisions from.
    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        tools: ToolRegistry,
        broker: Arc<ApprovalBroker>,
    }

    fn fixture(policy: ApprovalPolicy) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::write(root.join("input.txt"), "hello").unwrap();

        let broker = Arc::new(ApprovalBroker::new(policy));
        let session = Arc::new(ToolSession::with_broker(root.clone(), Arc::clone(&broker)));
        let tools = create_default_registry_with_session(
            root.clone(),
            Arc::new(SkillRegistry::new()),
            session,
        );

        Fixture {
            _dir: dir,
            root,
            tools,
            broker,
        }
    }

    impl Fixture {
        fn session(&self, traces: &Path) -> TraceSession {
            TraceSession::new(
                traces.to_path_buf(),
                TraceMeta::new(
                    "scripted".to_string(),
                    "none".to_string(),
                    self.root.display().to_string(),
                    self.broker.policy(),
                ),
                Some(Arc::clone(&self.broker)),
            )
        }
    }

    /// Run one turn against `provider`, returning the trace it wrote.
    fn run(
        fixture: &Fixture,
        provider: &dyn crate::llm::LlmProvider,
        traces: &Path,
        ctx: Option<&TurnContext>,
    ) -> (Result<String, AgentError>, TurnTrace) {
        let session = fixture.session(traces);
        let setup = TurnSetup {
            provider,
            tools: &fixture.tools,
            skills: None,
            max_iterations: Some(10),
            context_window: crate::memory::DEFAULT_CONTEXT_WINDOW,
            observer: None,
            context: ctx,
            trace: Some(&session),
            turn_id: None,
        };

        let mut history = vec![ChatMessage::system("sys".to_string())];
        let outcome = run_turn(&setup, &mut history, 0, "make out.txt".to_string())
            .map(|outcome| outcome.text);

        // Exactly one turn ran, so exactly one trace is on disk.
        let mut written: Vec<PathBuf> = std::fs::read_dir(traces)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        written.sort();
        assert_eq!(written.len(), 1, "one turn should leave one trace");

        (outcome, TurnTrace::load(&written[0]).unwrap())
    }

    fn scripted(json: &str) -> ScriptedProvider {
        ScriptedProvider::new(Script::parse(json).unwrap())
    }

    use crate::llm::ChatMessage;
    use crate::AgentError;

    #[test]
    fn a_turn_is_recorded_whole() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy::default());

        let (text, trace) = run(&fixture, &scripted(READ_THEN_WRITE), traces.path(), None);

        assert_eq!(text.unwrap(), "Wrote out.txt.");
        assert_eq!(trace.version, TRACE_FORMAT_VERSION);
        assert_eq!(trace.user_input, "make out.txt");

        // The prompt as the model first saw it, and the catalog it was offered.
        assert!(
            trace.prompt.iter().any(|m| m.content == "make out.txt"),
            "the prompt must hold what the user asked"
        );
        assert!(
            trace.tools.iter().any(|t| t.name == "Write"),
            "the catalog offered is part of what made the turn what it was"
        );

        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.steps[0].calls[0].name, "Read");
        assert_eq!(trace.steps[1].calls[0].name, "Write");
        assert_eq!(trace.steps[1].calls[0].arguments["file_path"], "out.txt");
        assert!(!trace.steps[1].calls[0].result.as_ref().unwrap().is_error);
        assert!(matches!(trace.ending, TurnEnding::Completed { .. }));

        // …and the write actually happened, so the trace describes a real turn.
        assert_eq!(
            std::fs::read_to_string(fixture.root.join("out.txt")).unwrap(),
            "written"
        );
    }

    /// The tier decides who is asked; the trace says who answered. A read is
    /// never asked about, so recording a decision for it would be recording a
    /// question nobody put.
    #[test]
    fn approvals_are_attributed_to_the_call_that_provoked_them() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy::default());

        let (_, trace) = run(&fixture, &scripted(READ_THEN_WRITE), traces.path(), None);

        assert!(
            trace.steps[0].calls[0].approvals.is_empty(),
            "a read is not an approval question"
        );
        let write = &trace.steps[1].calls[0].approvals;
        assert_eq!(write.len(), 1);
        assert_eq!(write[0].risk, "workspace write");
        assert_eq!(write[0].outcome, "allowed");
        assert!(write[0].target.ends_with("out.txt"));
    }

    /// A refusal is a decision, and the trace has to be able to say which call
    /// was refused and under what rule — that is the whole reason to record
    /// them.
    #[test]
    fn a_refused_call_is_recorded_as_refused() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy {
            workspace_write: ApprovalRule::Deny,
            ..ApprovalPolicy::default()
        });

        let (_, trace) = run(&fixture, &scripted(READ_THEN_WRITE), traces.path(), None);

        let write = &trace.steps[1].calls[0];
        assert_eq!(write.approvals[0].outcome, "denied-by-policy");
        assert!(
            write.result.as_ref().unwrap().is_error,
            "the model is told the call failed, and so is the trace"
        );
        assert!(!fixture.root.join("out.txt").exists());
    }

    /// The headline: a recorded turn replays through the real loop with real
    /// tools and no model, and lands in the same place.
    #[test]
    fn a_recorded_turn_replays_into_the_same_turn() {
        let traces = tempfile::tempdir().unwrap();
        let recorded = fixture(ApprovalPolicy::default());
        let (_, first) = run(&recorded, &scripted(READ_THEN_WRITE), traces.path(), None);

        // A fresh workspace and a fresh broker: the replay re-runs the tools for
        // real, so it must not inherit the first run's files or its grants.
        let replay_traces = tempfile::tempdir().unwrap();
        let replayed_in = fixture(ApprovalPolicy::default());
        let (_, second) = run(
            &replayed_in,
            &ScriptedProvider::new(first.to_script()),
            replay_traces.path(),
            None,
        );

        let divergences = first.diff(&second);
        assert!(
            divergences.is_empty(),
            "a replay should reproduce the turn, but: {}",
            divergences
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        );
        assert_eq!(
            std::fs::read_to_string(replayed_in.root.join("out.txt")).unwrap(),
            "written",
            "the tools ran for real; only the model was replayed"
        );
    }

    /// A test that cannot fail is not a test. Replaying a *different* script has
    /// to be reported, and reported where it happened.
    #[test]
    fn a_replay_that_diverges_says_where() {
        let traces = tempfile::tempdir().unwrap();
        let recorded = fixture(ApprovalPolicy::default());
        let (_, first) = run(&recorded, &scripted(READ_THEN_WRITE), traces.path(), None);

        let other_traces = tempfile::tempdir().unwrap();
        let other = fixture(ApprovalPolicy::default());
        let (_, second) = run(
            &other,
            &scripted(
                r#"{
                    "steps": [
                        { "toolCalls": [{ "id": "c1", "name": "Read",
                                          "arguments": { "file_path": "input.txt" } }] },
                        { "toolCalls": [{ "id": "c2", "name": "Write",
                                          "arguments": { "file_path": "elsewhere.txt",
                                                         "content": "written" } }] },
                        { "text": "Wrote elsewhere.txt." }
                    ]
                }"#,
            ),
            other_traces.path(),
            None,
        );

        let divergences = first.diff(&second);
        assert!(
            divergences
                .iter()
                .any(|d| d.at == "step 2, call 1: arguments"),
            "the changed argument should be named: {divergences:?}"
        );
        assert!(
            divergences.iter().any(|d| d.at == "ending: text"),
            "so should the answer it led to: {divergences:?}"
        );
    }

    /// A stopped turn rolls history back to before its prompt, so without a
    /// trace it leaves no evidence at all. That is exactly the turn worth having
    /// a record of.
    #[test]
    fn a_cancelled_turn_still_leaves_a_trace() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy::default());

        let ctx = TurnContext::new(CancellationToken::new());
        ctx.cancellation.cancel();

        let (result, trace) = run(
            &fixture,
            &scripted(READ_THEN_WRITE),
            traces.path(),
            Some(&ctx),
        );

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert!(matches!(trace.ending, TurnEnding::Cancelled));
        assert_eq!(trace.user_input, "make out.txt");
    }

    /// A tool that gets itself approved and then reports the cancellation a real
    /// tool sees when the turn is stopped while it works. Both halves are the
    /// point: the approval has to be journaled, and the call has to end without
    /// a result.
    struct ApproveThenCancelTool {
        broker: Arc<ApprovalBroker>,
    }

    impl crate::tool::Tool for ApproveThenCancelTool {
        fn name(&self) -> &str {
            "ApproveThenCancel"
        }
        fn description(&self) -> &str {
            "test tool: takes an approval, then observes cancellation"
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        fn call(&self, _args: Value) -> Result<ToolResult, AgentError> {
            self.broker
                .authorize(&crate::approval::ApprovalRequest::new(
                    "write file",
                    "out.txt",
                    crate::approval::RiskLevel::WorkspaceWrite,
                ))?;
            Err(AgentError::Cancelled)
        }
    }

    const CANCEL_MID_CALL: &str = r#"{
        "steps": [
            { "toolCalls": [ { "id": "c1", "name": "ApproveThenCancel", "arguments": {} } ] },
            { "text": "never reached" }
        ]
    }"#;

    fn cancelling_fixture() -> Fixture {
        let mut fixture = fixture(ApprovalPolicy::default());
        fixture.tools.register(Box::new(ApproveThenCancelTool {
            broker: Arc::clone(&fixture.broker),
        }));
        fixture
    }

    /// A call the turn was cancelled out from under is still recorded, with the
    /// approval it provoked and no result. Without this the trace of an
    /// interrupted turn shows the model asking for a tool and nothing happening,
    /// which is the opposite of what a record is for.
    #[test]
    fn a_call_cancelled_mid_flight_is_recorded_with_its_approval() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = cancelling_fixture();

        let (result, trace) = run(&fixture, &scripted(CANCEL_MID_CALL), traces.path(), None);

        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert!(matches!(trace.ending, TurnEnding::Cancelled));

        let call = &trace.steps[0].calls[0];
        assert_eq!(call.name, "ApproveThenCancel");
        assert!(
            call.result.is_none(),
            "an abandoned call has no result to record"
        );
        assert_eq!(
            call.approvals.len(),
            1,
            "the approval it took belongs to it"
        );
        assert_eq!(call.approvals[0].risk, "workspace write");
    }

    /// The reason recording that call matters beyond the record: the broker is
    /// session-scoped and its journal outlives a turn, so an approval left
    /// undrained is collected by the *next* turn's first tool call and attributed
    /// to a call that never asked for it.
    #[test]
    fn an_approval_does_not_leak_into_the_next_turns_trace() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        // One fixture, so both turns share the broker a real session would.
        let fixture = cancelling_fixture();

        let (result, _) = run(&fixture, &scripted(CANCEL_MID_CALL), first.path(), None);
        assert!(matches!(result, Err(AgentError::Cancelled)));

        // A second turn whose only call is a read — a tier that is never asked
        // about, so anything in its `approvals` came from somewhere else.
        let (_, trace) = run(
            &fixture,
            &scripted(
                r#"{"steps":[
                    {"toolCalls":[{"id":"r1","name":"Read","arguments":{"file_path":"input.txt"}}]},
                    {"text":"read it"}
                ]}"#,
            ),
            second.path(),
            None,
        );

        let call = &trace.steps[0].calls[0];
        assert_eq!(call.name, "Read");
        assert!(
            call.approvals.is_empty(),
            "a read inherited the previous turn's approval: {:?}",
            call.approvals
        );
    }

    /// A failure is the other ending that leaves nothing behind in history.
    #[test]
    fn a_failed_turn_records_why() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy::default());

        // One step, and the loop needs a second: the provider runs dry.
        let (result, trace) = run(
            &fixture,
            &scripted(
                r#"{"steps":[{"toolCalls":[{"id":"c1","name":"Read","arguments":{"file_path":"input.txt"}}]}]}"#,
            ),
            traces.path(),
            None,
        );

        assert!(result.is_err());
        match trace.ending {
            TurnEnding::Failed { ref message } => {
                assert!(message.contains("exhausted"), "{message}")
            }
            ref other => panic!("expected a failure, got {other:?}"),
        }
        // What it managed before failing is still there.
        assert_eq!(trace.steps[0].calls[0].name, "Read");
    }

    #[test]
    fn a_trace_round_trips_through_json() {
        let traces = tempfile::tempdir().unwrap();
        let fixture = fixture(ApprovalPolicy::default());
        let (_, trace) = run(&fixture, &scripted(READ_THEN_WRITE), traces.path(), None);

        let json = serde_json::to_string(&trace).unwrap();
        let back: TurnTrace = serde_json::from_str(&json).unwrap();

        assert!(trace.diff(&back).is_empty());
    }

    /// A future format is refused rather than half-read: a reader that guesses
    /// at fields it does not know reports differences that are its own.
    #[test]
    fn a_newer_format_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("turn.json");
        std::fs::write(
            &path,
            serde_json::json!({
                "version": TRACE_FORMAT_VERSION + 1,
                "turnId": "t1",
                "startedAtEpochMs": 0,
                "durationMs": 0,
                "engine": "scripted",
                "model": "none",
                "approvalPolicy": "",
                "workspaceRoot": "/tmp",
                "userInput": "hi",
                "prompt": [],
                "tools": [],
                "compacted": 0,
                "steps": [],
                "usage": {"inputTokens":0,"outputTokens":0,"totalTokens":0,"peakInputTokens":0},
                "ending": {"status":"cancelled"}
            })
            .to_string(),
        )
        .unwrap();

        let err = TurnTrace::load(&path).unwrap_err().to_string();

        assert!(err.contains("format version"), "{err}");
    }

    /// Truncation says so. A silent cut would let `diff` call two texts equal
    /// when it only ever compared their first 16 KiB.
    #[test]
    fn a_long_text_is_truncated_out_loud() {
        let long = "x".repeat(MAX_CAPTURED_CHARS + 500);

        let captured = capture(&long);

        assert!(captured.len() < long.len());
        assert!(captured.contains("trace truncated"), "{}", &captured[..80]);
        assert!(captured.contains(&(MAX_CAPTURED_CHARS + 500).to_string()));
    }

    /// Truncating mid-character would panic; the cut walks back to a boundary.
    #[test]
    fn truncation_lands_on_a_character_boundary() {
        let text = "あ".repeat(MAX_CAPTURED_CHARS); // 3 bytes each

        let captured = capture(&text);

        assert!(captured.contains("trace truncated"));
    }

    /// A screenshot is most of a file and none of the information.
    #[test]
    fn an_image_is_recorded_by_size_not_by_content() {
        let result = ToolResult::with_images(
            "captured".to_string(),
            vec![crate::llm::ImageContent {
                base64: "AAAA".repeat(1000),
                media_type: "image/png".to_string(),
            }],
        );

        let record = ToolResultRecord::from(&result);

        assert_eq!(record.images.len(), 1);
        assert_eq!(record.images[0].media_type, "image/png");
        assert_eq!(record.images[0].base64_len, 4000);
        assert!(!record.model_text.contains("AAAA"));
    }

    /// The env vars are process-global, so the tests that read them take a lock
    /// rather than racing each other over it.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<_> = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var_os(k)))
            .collect();
        for (k, v) in vars {
            // Safety: single-threaded within the lock, and restored before release.
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        let out = f();
        for (k, v) in saved {
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        out
    }

    fn meta() -> TraceMeta {
        TraceMeta::new(
            "scripted".to_string(),
            "none".to_string(),
            "/work".to_string(),
            ApprovalPolicy::default(),
        )
    }

    /// Traces are opt-in. They hold every byte the model saw, so nothing should
    /// start writing them because a config file was silent.
    #[test]
    fn tracing_is_off_unless_something_asks_for_it() {
        let session = with_env(
            &[("GALLIUM_TRACE", None), ("GALLIUM_TRACE_DIR", None)],
            || TraceSession::from_env(None, meta(), None),
        );

        assert!(session.is_none());
    }

    #[test]
    fn the_switch_alone_puts_traces_beside_the_workspace() {
        let session = with_env(
            &[("GALLIUM_TRACE", Some("1")), ("GALLIUM_TRACE_DIR", None)],
            || TraceSession::from_env(None, meta(), None),
        )
        .expect("GALLIUM_TRACE=1 turns tracing on");

        assert_eq!(session.dir(), Path::new("/work").join(DEFAULT_TRACE_DIR));
    }

    /// A run has to be able to say "not this one" without editing the config
    /// that asked for traces.
    #[test]
    fn an_explicit_off_beats_a_configured_directory() {
        let session = with_env(
            &[("GALLIUM_TRACE", Some("0")), ("GALLIUM_TRACE_DIR", None)],
            || TraceSession::from_env(Some(PathBuf::from("/configured")), meta(), None),
        );

        assert!(session.is_none());
    }

    #[test]
    fn the_env_directory_wins_over_the_configured_one() {
        let session = with_env(
            &[
                ("GALLIUM_TRACE", None),
                ("GALLIUM_TRACE_DIR", Some("/from-env")),
            ],
            || TraceSession::from_env(Some(PathBuf::from("/configured")), meta(), None),
        )
        .expect("a directory turns tracing on");

        assert_eq!(session.dir(), Path::new("/from-env"));
    }

    /// A client's turn id reaches the filename, so a trace can be matched with
    /// the notifications the client saw — and cannot become a path of its own.
    #[test]
    fn a_turn_id_is_kept_but_made_safe_for_a_filename() {
        assert_eq!(sanitize("turn_12"), "turn_12");
        assert_eq!(sanitize("../../etc/passwd"), "------etc-passwd");
    }

    /// Tracing observes a turn; it must not be able to fail one. An unwritable
    /// destination is reported and swallowed.
    #[test]
    fn a_trace_that_cannot_be_written_does_not_fail_the_turn() {
        let dir = tempfile::tempdir().unwrap();
        // A file where the directory should be, so `create_dir_all` fails.
        let blocked = dir.path().join("not-a-dir");
        std::fs::write(&blocked, "").unwrap();

        let fixture = fixture(ApprovalPolicy::default());
        let session = TraceSession::new(blocked, meta(), None);
        let setup = TurnSetup {
            provider: &scripted(r#"{"steps":[{"text":"fine"}]}"#),
            tools: &fixture.tools,
            skills: None,
            max_iterations: Some(5),
            context_window: crate::memory::DEFAULT_CONTEXT_WINDOW,
            observer: None,
            context: None,
            trace: Some(&session),
            turn_id: None,
        };

        let mut history = vec![ChatMessage::system("sys".to_string())];
        let outcome = run_turn(&setup, &mut history, 0, "hi".to_string());

        assert_eq!(outcome.unwrap().text, "fine");
    }
}
