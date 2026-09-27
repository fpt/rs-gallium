//! A Responses-API-shaped HTTP endpoint over gallium's inference engines —
//! **not** a rendering of `runtime::run_turn`. This is the opposite kind of
//! surface from the REPL and the app-server: those two hand a turn to
//! gallium's own ReAct loop, which calls the model as many times as it needs
//! and executes gallium's own tools. This one exists so a harness that runs
//! its *own* agent loop and its *own* tools — Codex, or anything else that
//! speaks `POST /v1/responses` — can point at a local gallium-hosted model
//! (starting with Qwen3.8-Flash-Next) as a raw inference backend: one
//! `LlmProvider` call in, one response out, nothing executed here.
//!
//! Deliberately narrow, for now:
//!
//! - No ReAct loop, no gallium tool execution, no approvals. A `tools` array
//!   in the request becomes [`ToolDefinition`]s handed to the model; a tool
//!   call the model makes comes back as a `function_call` output item for the
//!   *caller* to run, exactly as OpenAI's own Responses API works for
//!   `function` tools (it never executes them either).
//! - No `previous_response_id`-keyed session store. Verified against Codex's
//!   own source (`codex-rs/codex-api/src/common.rs`): its plain-HTTP request
//!   (`ResponsesApiRequest`) has no such field at all, sends `store: false`,
//!   and resends the *entire* conversation in `input` every call —
//!   `previous_response_id` is a websocket-transport-only optimization Codex
//!   falls back away from (426 Upgrade Required) when a server doesn't offer
//!   it, which this one doesn't. So the KV-cache-reuse benefit this endpoint
//!   exists for needs no new bookkeeping: `LlamaLocalProvider`'s existing
//!   token-id slot matching (issue #86) already reuses the cache across calls
//!   whose rendered prompt keeps growing as a prefix, regardless of whether
//!   the *client* resent everything — as long as the *same* provider instance
//!   (one per `ResponsesApiServer`, held for the process's lifetime) serves
//!   every call in the conversation. This is therefore single-session by the
//!   same construction the app-server already documents (one KV slot by
//!   default): a second concurrent caller contends for it exactly as two
//!   interleaved app-server threads would.
//! - Only `message` / `function_call` / `function_call_output` input items
//!   and `type: "function"` tools are understood. Codex's actual wire dialect
//!   is much larger (`AgentMessage`, `LocalShellCall`, `CustomToolCall`,
//!   hosted tools like `web_search`, MCP `namespace` tool groups, encrypted
//!   reasoning …) — captured directly from a live Codex request during this
//!   feature's spike, not assumed from the public API docs. An item or tool
//!   this endpoint doesn't recognise is skipped and logged rather than
//!   guessed at; grow this list only against what a real session with a real
//!   model actually needs, the same way the wire-format matrix in
//!   `profile::tests` only supports formats a real model has been seen to
//!   emit.
//! - Always answers as one batched SSE body (`response.created` →
//!   `response.output_item.done` (whole item) → `response.completed`), not
//!   live per-token deltas. Confirmed sufficient: this is the exact shape
//!   Codex's own test fixtures (`core/tests/common/responses.rs`) build to
//!   drive its SSE parser, and a live Codex run against this endpoint
//!   displayed the reply correctly. Streaming `chat_with_tools_streaming`'s
//!   deltas as `response.output_text.delta` is a follow-up, not required for
//!   a first working version.
//!
//! ## Two ways to pick a model
//!
//! [`ResponsesApiServer::new`] (`--config <file>`) loads exactly one model at
//! startup and serves every request against it, ignoring the request's own
//! `model` field — the shape described above.
//!
//! [`ResponsesApiServer::new_dynamic`] (`--config-dir <dir>`) loads nothing
//! upfront and resolves `<dir>/<model>.toml` **per request**, from the
//! Responses API's own `model` field — useful because that field can name
//! anything, and a harness like Codex may point different sessions (or the
//! same session, retried with `-m`) at different local models without this
//! process being restarted for each one.
//!
//! There is still only one GPU, so this cannot mean "keep every named model
//! resident" — it means *swap*, the same "one client at a time, newest wins"
//! rule the app-server already lives by for its KV slot pool, extended from
//! one conversation to one *model*. A request naming a different model than
//! the one currently loaded evicts it (dropping the `Arc<dyn LlmProvider>`,
//! which releases its VRAM/RAM the same way the app-server's `Drop` does) and
//! blocks on the new one loading — full GGUF-load latency, synchronously,
//! inside that request. That block is held under the same lock every other
//! request (even one naming the *still-loaded* model) waits on, deliberately:
//! serving the old model on its stale `Arc` while the new one loads would mean
//! two resident models fighting over the same 12GB card, which is worse than
//! making everyone wait for the one swap that is actually happening.
//! Concurrent requests that all name the *same, already-loaded* model are
//! unaffected — the lock is only held for the lookup/swap-or-not check, never
//! for the `chat_with_tools` call itself, so they still run in parallel
//! exactly as the single-model mode's do.
//!
//! A `model` naming no `<dir>/<model>.toml` is refused, listing what *is*
//! available — the same UX `ModelProfile` naming gives an unknown profile.
//! `model` is also checked for path separators and `..` before it ever
//! touches the filesystem: this endpoint already has no authentication (see
//! `ResponsesApiServer::run`), and a client-controlled string is not a
//! filename component just because it usually looks like one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::appserver::server::default_provider_factory;
use crate::appserver::ServerConfig;
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo, ToolDefinition};

/// Builds the provider for one model name, in `--config-dir` mode. Owned and
/// supplied by `main.rs`: loading a `configs/<model>.toml` needs the binary's
/// own (private) config-file/env-resolution code, which this library crate
/// cannot see — see `main.rs`'s `responses_api_provider_loader`.
pub type ProviderLoader = Arc<dyn Fn(&str) -> anyhow::Result<Arc<dyn LlmProvider>> + Send + Sync>;

/// Unique-enough id for a response/message/call, in this repo's own house
/// style (`mcp::generate_session_id`) rather than pulling in the `uuid`
/// crate: pid + timestamp + a per-process counter, so two ids never collide
/// regardless of clock resolution.
fn gen_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{:x}{:x}{:x}", std::process::id(), ts, seq)
}

#[derive(Debug, Deserialize, Default)]
struct ResponsesRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    instructions: String,
    #[serde(default)]
    input: Vec<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    /// Accepted and logged, never required: see the module doc for why this
    /// endpoint's cache reuse doesn't depend on it.
    #[serde(default)]
    previous_response_id: Option<String>,
}

/// `content` on a `message` item: either a plain string, or (what Codex
/// actually sends) an array of `{"type": "input_text"|"output_text"|"text", "text": …}`.
fn extract_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|it| it.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// `function_call_output.output`: a plain string, or an array of structured
/// content items — same two shapes `content` above takes, per Codex's own
/// comment on `FunctionCallOutputPayload`.
fn extract_function_output(output: Option<&Value>) -> String {
    match output {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|it| it.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// `input` → gallium's own message shape. `instructions` becomes the first
/// system message, matching how every gallium frontend puts a system prompt
/// first.
fn translate_input(instructions: &str, input: &[Value]) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(input.len() + 1);
    if !instructions.is_empty() {
        messages.push(ChatMessage::system(instructions.to_string()));
    }

    // `function_call_output` carries only `call_id`, not the tool name —
    // recovered from the matching `function_call` earlier in the same array.
    // Codex resends the whole conversation every call (see module doc), so
    // the call this answers is always present earlier in `input`.
    let mut call_names: HashMap<String, String> = HashMap::new();

    for item in input {
        let Some(ty) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match ty {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let text = extract_text(item.get("content"));
                messages.push(match role {
                    "assistant" => ChatMessage::assistant(text),
                    "system" | "developer" => ChatMessage::system(text),
                    _ => ChatMessage::user(text),
                });
            }
            "function_call" => {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                // Responses API carries arguments as a JSON *string*, not an
                // already-parsed object (Codex's own comment on this field
                // says the same) — gallium's `ToolCallInfo::arguments` wants
                // the parsed `Value`.
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(Value::Null);
                call_names.insert(call_id.clone(), name.clone());
                messages.push(ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
                    id: call_id,
                    name,
                    arguments,
                }]));
            }
            "function_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = call_names.get(&call_id).cloned().unwrap_or_default();
                let content = extract_function_output(item.get("output"));
                messages.push(ChatMessage::tool_result(call_id, name, content));
            }
            other => {
                tracing::debug!("responses-api: skipping unsupported input item type '{other}'");
            }
        }
    }
    messages
}

/// `tools` → gallium's [`ToolDefinition`]s, keeping only `type: "function"`
/// entries. Everything else in a real Codex request — hosted tools like
/// `web_search`, MCP `namespace` groups — names a capability no local model
/// can invoke by emitting text, so there is nothing to translate; skipped and
/// logged rather than silently degrading tool-calling for the whole turn.
fn translate_tools(tools: &[Value]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .filter_map(|t| {
            if t.get("type").and_then(Value::as_str) != Some("function") {
                if let Some(ty) = t.get("type").and_then(Value::as_str) {
                    tracing::debug!("responses-api: skipping non-function tool type '{ty}'");
                }
                return None;
            }
            let name = t.get("name")?.as_str()?.to_string();
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let parameters = t
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            Some(ToolDefinition {
                name,
                description,
                parameters,
            })
        })
        .collect()
}

/// [`LlmResponse`] → Responses API output items. A tool call's `arguments`
/// goes back out as a JSON *string*, mirroring how it arrived.
///
/// A `function_call` item carries two distinct ids: `id` names the output
/// item itself, `call_id` names the tool round-trip (what a later
/// `function_call_output` echoes back). Codex's own `ResponseItem::FunctionCall`
/// has both as separate fields — `call_id` is not a substitute for `id`, so
/// this mints one the same way the `message` item already does.
fn llm_response_to_output_items(resp: &LlmResponse) -> Vec<Value> {
    match resp {
        LlmResponse::Text { content, .. } => vec![json!({
            "type": "message",
            "role": "assistant",
            "id": gen_id("msg"),
            "content": [{"type": "output_text", "text": content}],
        })],
        LlmResponse::ToolCalls { calls, .. } => calls
            .iter()
            .map(|c| {
                json!({
                    "type": "function_call",
                    "id": gen_id("fc"),
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": c.arguments.to_string(),
                })
            })
            .collect(),
    }
}

fn push_event(out: &mut String, value: Value) {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    out.push_str(&format!("event: {kind}\ndata: {value}\n\n"));
}

fn build_sse(resp_id: &str, items: &[Value]) -> String {
    let mut out = String::new();
    push_event(
        &mut out,
        json!({"type": "response.created", "response": {"id": resp_id}}),
    );
    for item in items {
        push_event(
            &mut out,
            json!({"type": "response.output_item.done", "item": item}),
        );
    }
    push_event(
        &mut out,
        json!({
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": null,
                    "output_tokens": 0,
                    "output_tokens_details": null,
                    "total_tokens": 0,
                },
            },
        }),
    );
    out
}

fn build_failed_sse(resp_id: &str, message: &str) -> String {
    let mut out = String::new();
    push_event(
        &mut out,
        json!({"type": "response.created", "response": {"id": resp_id}}),
    );
    push_event(
        &mut out,
        json!({
            "type": "response.failed",
            "response": {"id": resp_id, "error": {"code": "gallium_error", "message": message}},
        }),
    );
    out
}

fn sse_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap()
}

fn json_header() -> tiny_http::Header {
    tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap()
}

/// How this server decides which model serves a given request — see the
/// module doc's "Two ways to pick a model".
enum ModelSource {
    /// One provider, loaded once, held for the process's lifetime. Never
    /// rebuilt per request, or every call would reload the model and start
    /// its KV cache cold, defeating the entire point of this endpoint.
    Fixed {
        provider: Arc<dyn LlmProvider>,
        model_name: String,
    },
    /// Loaded on demand and swapped as different `model` names arrive.
    /// `current` is `None` only before the first request.
    Dynamic {
        config_dir: PathBuf,
        loader: ProviderLoader,
        current: Mutex<Option<(String, Arc<dyn LlmProvider>)>>,
    },
}

pub struct ResponsesApiServer {
    source: ModelSource,
}

impl ResponsesApiServer {
    /// `--config <file>`: one model, loaded now.
    pub fn new(config: &ServerConfig) -> anyhow::Result<Self> {
        let provider = default_provider_factory(config, &config.model)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self {
            source: ModelSource::Fixed {
                provider: Arc::from(provider),
                model_name: config.model.clone(),
            },
        })
    }

    /// `--config-dir <dir>`: nothing loaded yet — the first request's `model`
    /// field picks it. `loader` resolves a name to a provider; see
    /// `ProviderLoader`'s own doc for why this crate cannot do that itself.
    pub fn new_dynamic(config_dir: PathBuf, loader: ProviderLoader) -> Self {
        Self {
            source: ModelSource::Dynamic {
                config_dir,
                loader,
                current: Mutex::new(None),
            },
        }
    }

    /// The provider to serve `requested_model` with, loading or swapping in
    /// `--config-dir` mode as needed — see the module doc for what a swap
    /// costs and why it blocks every other request while it happens.
    fn resolve_provider(&self, requested_model: &str) -> anyhow::Result<Arc<dyn LlmProvider>> {
        match &self.source {
            ModelSource::Fixed { provider, .. } => Ok(Arc::clone(provider)),
            ModelSource::Dynamic {
                loader, current, ..
            } => {
                let mut guard = current.lock();
                if let Some((name, provider)) = guard.as_ref() {
                    if name == requested_model {
                        return Ok(Arc::clone(provider));
                    }
                }
                if requested_model.is_empty() {
                    anyhow::bail!(
                        "the \"model\" field is required in --config-dir mode, to pick which config to load"
                    );
                }
                if requested_model.contains('/')
                    || requested_model.contains('\\')
                    || requested_model.contains("..")
                {
                    anyhow::bail!(
                        "\"model\" ({requested_model:?}) must be a plain config name, not a path"
                    );
                }
                // "resolving", not "loading": the request is only confirmed
                // to be a real model change once `loader` actually succeeds,
                // below. A failed lookup (unknown name, bad TOML) must not
                // read in a log as if a swap happened — `current` is only
                // written to on the `Ok` path, by construction (the `?`
                // above returns before it on `Err`), but the log line should
                // say the same thing the state does.
                tracing::info!(
                    "responses-api: resolving model '{requested_model}' ({})",
                    match guard.as_ref() {
                        Some((name, _)) => format!("currently '{name}'"),
                        None => "nothing loaded yet".to_string(),
                    }
                );
                let provider = (loader)(requested_model)?;
                tracing::info!("responses-api: loaded model '{requested_model}'");
                *guard = Some((requested_model.to_string(), Arc::clone(&provider)));
                Ok(provider)
            }
        }
    }

    /// Binds `addr` and serves until the process is killed. There is
    /// deliberately no separate "stdio" mode here (unlike the app-server):
    /// an HTTP API has no meaningful non-socket transport, so `--listen` is
    /// required to reach this function at all — see `main.rs`.
    ///
    /// Takes `Arc<Self>` rather than `&self` so each request can be handed to
    /// its own thread (below): a single slow generation must not block the
    /// accept loop from taking the *next* connection, or `GET /models` and
    /// any future health probe would hang behind it too. Model calls
    /// themselves still serialize — `LlamaLocalProvider` guards its slot pool
    /// with its own `Mutex` — so concurrent callers contend for that lock
    /// exactly as the module doc says, rather than for the ability to be
    /// accepted at all.
    pub fn run(self: Arc<Self>, addr: &str) -> anyhow::Result<()> {
        let server = tiny_http::Server::http(addr)
            .map_err(|e| anyhow::anyhow!("cannot bind {addr}: {e}"))?;
        // Same warning `appserver::tcp` gives a listening app-server, for the
        // same reason: this socket carries no identity and no authentication,
        // so anything that can reach it gets a completion from this process's
        // model at this process's expense.
        tracing::warn!(
            "responses-api listening on {addr} — every interface, including public \
             ones, if this address is not loopback. This endpoint has no \
             authentication: anything that can reach it can run completions \
             against this model at this machine's expense. It executes no tools \
             and holds no approvals of its own, but bind a loopback or private \
             overlay (Tailscale/WireGuard) address, not a public one."
        );
        for request in server.incoming_requests() {
            let this = Arc::clone(&self);
            std::thread::spawn(move || this.handle(request));
        }
        Ok(())
    }

    fn handle(&self, request: tiny_http::Request) {
        let path = request.url().to_string();
        let path_only = path.split('?').next().unwrap_or("");
        match (request.method(), path_only) {
            (&tiny_http::Method::Get, p) if p.ends_with("/models") => {
                let ids: Vec<String> = match &self.source {
                    ModelSource::Fixed { model_name, .. } => vec![model_name.clone()],
                    ModelSource::Dynamic { config_dir, .. } => list_model_names(config_dir),
                };
                let body = json!({
                    "object": "list",
                    "data": ids.iter().map(|id| json!({"id": id, "object": "model"})).collect::<Vec<_>>(),
                });
                let _ = request.respond(
                    tiny_http::Response::from_string(body.to_string()).with_header(json_header()),
                );
            }
            (&tiny_http::Method::Post, p) if p.ends_with("/responses") => {
                self.handle_responses(request);
            }
            _ => {
                let _ = request
                    .respond(tiny_http::Response::from_string("Not Found").with_status_code(404));
            }
        }
    }

    fn handle_responses(&self, mut request: tiny_http::Request) {
        let mut body = String::new();
        if request.as_reader().read_to_string(&mut body).is_err() {
            let _ = request
                .respond(tiny_http::Response::from_string("Bad Request").with_status_code(400));
            return;
        }
        let parsed: ResponsesRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                let _ = request.respond(
                    tiny_http::Response::from_string(format!("invalid JSON: {e}"))
                        .with_status_code(400),
                );
                return;
            }
        };
        if let Some(prev) = &parsed.previous_response_id {
            tracing::debug!("responses-api: previous_response_id={prev} (recorded, not required)");
        }

        let resp_id = gen_id("resp");
        let provider = match self.resolve_provider(&parsed.model) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    "responses-api: cannot resolve model '{}': {e}",
                    parsed.model
                );
                let sse = build_failed_sse(&resp_id, &e.to_string());
                let _ = request
                    .respond(tiny_http::Response::from_string(sse).with_header(sse_header()));
                return;
            }
        };

        let messages = translate_input(&parsed.instructions, &parsed.input);
        let tools = translate_tools(&parsed.tools);
        tracing::info!(
            "responses-api: turn for model '{}' — {} message(s), {} tool(s)",
            parsed.model,
            messages.len(),
            tools.len()
        );

        let sse = match provider.chat_with_tools(&messages, &tools) {
            Ok(resp) => {
                let items = llm_response_to_output_items(&resp);
                build_sse(&resp_id, &items)
            }
            Err(e) => {
                tracing::error!("responses-api: model call failed: {e}");
                build_failed_sse(&resp_id, &e.to_string())
            }
        };
        let _ = request.respond(tiny_http::Response::from_string(sse).with_header(sse_header()));
    }
}

/// `--config-dir` mode's `GET /models`: every `<dir>/*.toml` as a servable id,
/// not just what happens to be loaded right now — matches what a real
/// inference server's model listing means (what *can* be served).
fn list_model_names(config_dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(config_dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatRole;

    /// The exact `message` shape a live Codex request sends (captured during
    /// this feature's spike): `content` is an array of `{"type": "input_text",
    /// "text": …}`, not a plain string.
    #[test]
    fn a_user_message_with_array_content_becomes_a_chat_message() {
        let input = vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "say hi"}],
        })];
        let messages = translate_input("", &input);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[0].content, "say hi");
    }

    /// `instructions` becomes the first system message, ahead of anything in
    /// `input` — the same position every gallium frontend puts a system
    /// prompt in.
    #[test]
    fn instructions_become_a_leading_system_message() {
        let messages = translate_input("be terse", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[0].content, "be terse");
    }

    /// `developer` folds into `System`: gallium has no separate role for it,
    /// and OpenAI's own convention treats the two as equivalent for a model
    /// that doesn't distinguish them either.
    #[test]
    fn a_developer_role_message_is_system() {
        let input = vec![json!({
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "skills etc."}],
        })];
        let messages = translate_input("", &input);
        assert_eq!(messages[0].role, ChatRole::System);
    }

    /// `function_call_output` carries only `call_id`, never the tool's name
    /// (confirmed against Codex's actual `ResponseItem::FunctionCallOutput`) —
    /// the name has to come from the matching `function_call` earlier in the
    /// same array.
    #[test]
    fn a_function_call_output_recovers_its_tool_name_from_the_earlier_call() {
        let input = vec![
            json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read it"}]}),
            json!({"type": "function_call", "call_id": "call_0", "name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}),
            json!({"type": "function_call_output", "call_id": "call_0", "output": "contents"}),
        ];
        let messages = translate_input("", &input);
        assert_eq!(messages.len(), 3);

        assert_eq!(messages[1].role, ChatRole::Assistant);
        let calls = messages[1].tool_calls.as_ref().expect("tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, json!({"path": "a.txt"}));

        assert_eq!(messages[2].role, ChatRole::Tool);
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_0"));
        assert_eq!(messages[2].tool_name.as_deref(), Some("read_file"));
        assert_eq!(messages[2].content, "contents");
    }

    /// An item type this endpoint doesn't understand yet (Codex's dialect has
    /// many: `local_shell_call`, `reasoning`, …) is skipped, not guessed at —
    /// see the module doc for why growing this list waits on a real need.
    #[test]
    fn an_unrecognised_item_type_is_skipped_not_guessed_at() {
        let input = vec![
            json!({"type": "reasoning", "id": "r1", "summary": []}),
            json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}),
        ];
        let messages = translate_input("", &input);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hi");
    }

    /// Only `type: "function"` tools translate. Hosted tools (`web_search`,
    /// `image_generation`) and MCP `namespace` groups name a capability no
    /// local model can invoke by emitting text — captured from a live Codex
    /// request, not assumed.
    #[test]
    fn only_function_tools_translate() {
        let tools = vec![
            json!({
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }),
            json!({"type": "web_search", "external_web_access": false}),
            json!({"type": "namespace", "name": "multi_agent_v1", "tools": []}),
        ];
        let defs = translate_tools(&tools);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "read_file");
    }

    /// A `function` tool with no `parameters` still translates, with an empty
    /// object schema rather than being dropped — some callers omit it for a
    /// no-argument tool.
    #[test]
    fn a_function_tool_without_parameters_gets_an_empty_schema() {
        let tools = vec![json!({"type": "function", "name": "ping"})];
        let defs = translate_tools(&tools);
        assert_eq!(defs.len(), 1);
        assert_eq!(
            defs[0].parameters,
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn a_text_response_becomes_one_message_output_item() {
        let resp = LlmResponse::Text {
            content: "Paris".to_string(),
            reasoning: None,
            usage: None,
            raw: None,
        };
        let items = llm_response_to_output_items(&resp);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[0]["role"], "assistant");
        assert_eq!(items[0]["content"][0]["type"], "output_text");
        assert_eq!(items[0]["content"][0]["text"], "Paris");
    }

    /// Arguments go back out as a JSON *string*, not an object — the same
    /// shape they arrived in on `function_call`, and what Codex's own
    /// `ResponseItem::FunctionCall::arguments` doc comment says the Responses
    /// API always uses.
    #[test]
    fn a_tool_call_response_becomes_a_function_call_item_with_stringified_arguments() {
        let resp = LlmResponse::ToolCalls {
            calls: vec![ToolCallInfo {
                id: "call_0".to_string(),
                name: "read_file".to_string(),
                arguments: json!({"path": "a.txt"}),
            }],
            usage: None,
            reasoning: None,
            raw: None,
        };
        let items = llm_response_to_output_items(&resp);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call");
        // `id` (the output item's own id) and `call_id` (the tool round-trip
        // id) are distinct fields — asserted separately so a regression that
        // collapses them back into one is caught.
        assert!(items[0]["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert_ne!(items[0]["id"], items[0]["call_id"]);
        assert_eq!(items[0]["call_id"], "call_0");
        assert_eq!(items[0]["name"], "read_file");
        assert_eq!(items[0]["arguments"], json!("{\"path\":\"a.txt\"}"));
    }

    /// The wire-level contract this whole endpoint's value proposition rests
    /// on (see the module doc): a growing SSE payload always has `created`
    /// first and `completed` last, so a client can rely on both no matter
    /// what landed in between.
    #[test]
    fn sse_always_opens_with_created_and_closes_with_completed() {
        let items =
            vec![json!({"type": "message", "role": "assistant", "id": "m1", "content": []})];
        let sse = build_sse("resp_1", &items);
        let events: Vec<&str> = sse.lines().filter(|l| l.starts_with("event: ")).collect();
        assert_eq!(events.first(), Some(&"event: response.created"));
        assert_eq!(events.last(), Some(&"event: response.completed"));
    }

    /// A provider that answers nothing real and, critically, counts how many
    /// times it was *constructed* — the thing `--config-dir` mode's reuse-vs-
    /// swap logic is actually about, not what it says once loaded.
    struct FakeProvider;
    impl LlmProvider for FakeProvider {
        fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    /// A `--config-dir` server whose loader panics if ever called, for
    /// asserting a request is rejected *before* touching the filesystem —
    /// not merely that it's eventually rejected.
    fn dynamic_server_with_unreachable_loader() -> ResponsesApiServer {
        ResponsesApiServer::new_dynamic(
            PathBuf::from("/nonexistent"),
            Arc::new(|model: &str| panic!("loader should not run for {model:?}")),
        )
    }

    #[test]
    fn an_empty_model_name_is_rejected_before_the_loader_runs() {
        let server = dynamic_server_with_unreachable_loader();
        // `.unwrap_err()` needs `T: Debug`, which `Arc<dyn LlmProvider>` isn't.
        let err = server
            .resolve_provider("")
            .err()
            .expect("expected an error");
        assert!(err.to_string().contains("\"model\" field is required"));
    }

    #[test]
    fn a_model_name_with_a_path_separator_is_rejected_before_the_loader_runs() {
        let server = dynamic_server_with_unreachable_loader();
        for bad in ["../secrets", "a/b", "a\\b", "..", "sub/../../etc/passwd"] {
            let err = server
                .resolve_provider(bad)
                .err()
                .expect("expected an error");
            assert!(
                err.to_string().contains("must be a plain config name"),
                "{bad:?} should have been rejected as a path, got: {err}"
            );
        }
    }

    #[test]
    fn dynamic_mode_reuses_the_provider_for_repeated_requests_naming_the_same_model() {
        let calls = Arc::new(AtomicU64::new(0));
        let calls_in_loader = Arc::clone(&calls);
        let server = ResponsesApiServer::new_dynamic(
            PathBuf::from("/nonexistent"),
            Arc::new(move |_model: &str| {
                calls_in_loader.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(FakeProvider) as Arc<dyn LlmProvider>)
            }),
        );
        server.resolve_provider("gpt-oss-20b").unwrap();
        server.resolve_provider("gpt-oss-20b").unwrap();
        server.resolve_provider("gpt-oss-20b").unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "should load once, not per request"
        );
    }

    #[test]
    fn dynamic_mode_reloads_when_the_requested_model_changes() {
        let calls = Arc::new(AtomicU64::new(0));
        let calls_in_loader = Arc::clone(&calls);
        let server = ResponsesApiServer::new_dynamic(
            PathBuf::from("/nonexistent"),
            Arc::new(move |_model: &str| {
                calls_in_loader.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(FakeProvider) as Arc<dyn LlmProvider>)
            }),
        );
        server.resolve_provider("gpt-oss-20b").unwrap();
        server.resolve_provider("lfm2").unwrap();
        server.resolve_provider("gpt-oss-20b").unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "every switch is a real reload, including switching back"
        );
    }

    #[test]
    fn a_failed_load_does_not_evict_the_previously_loaded_model() {
        let server = ResponsesApiServer::new_dynamic(
            PathBuf::from("/nonexistent"),
            Arc::new(|model: &str| {
                if model == "gpt-oss-20b" {
                    Ok(Arc::new(FakeProvider) as Arc<dyn LlmProvider>)
                } else {
                    anyhow::bail!("no such model")
                }
            }),
        );
        server.resolve_provider("gpt-oss-20b").unwrap();
        assert!(server.resolve_provider("does-not-exist").is_err());
        // Still resolves without the loader running again — the failed
        // attempt above must not have overwritten `current`.
        server.resolve_provider("gpt-oss-20b").unwrap();
    }
}
