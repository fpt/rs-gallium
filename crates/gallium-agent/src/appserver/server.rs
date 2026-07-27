//! The gallium app-server: exposes the agent as a whole-turn backend over
//! JSON-RPC, speaking a subset of the codex app-server protocol.
//!
//! Gallium does not own this protocol — it implements enough of it that a client
//! already driving `codex app-server` (klein's `internal/codex` runner) can
//! drive gallium by swapping the binary. Methods served:
//!
//! | method          | direction | purpose                                   |
//! |-----------------|-----------|-------------------------------------------|
//! | `initialize`    | in        | capability negotiation                    |
//! | `account/read`  | in        | readiness probe (gallium needs no login)   |
//! | `thread/start`  | in        | create a thread (an `Agent` + registry)   |
//! | `turn/start`    | in        | begin a turn; answers at once, runs on    |
//! |                 |           | its own thread, reports by notification   |
//! | `item/tool/call`| out       | invoke a client-provided dynamic tool     |
//! | `item/*/requestApproval` | out | ask the client to permit a mutation  |
//! | `item/started`  | out       | a tool call was announced                 |
//! | `item/completed`, `turn/completed`, `turn/failed` | out | progress |

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::appserver::rpc::{Connection, HandlerResult, RequestHandler, RpcFault};
use crate::appserver::tools::{AutoApproveSink, DynamicToolSpec, RemoteApprovalSink, RemoteTool};
use crate::event::{AgentEvent, AgentObserver};
use crate::llm::{create_provider, ChatMessage, LlmProvider};
use crate::memory;
use crate::runtime::{self, TurnSetup};
use crate::skill::SkillRegistry;
use crate::tool::{
    create_default_registry_with_session, ApprovalSink, ToolAccess, ToolRegistry, ToolSession,
    ToolSource,
};
use crate::{AgentError, McpServerConfig};

/// Settings the process is launched with; a thread inherits these unless
/// `thread/start` overrides them.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_path: Option<String>,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    /// Local inference backend: "llamacpp" (default) or "gallium". `None`
    /// auto-detects (and still honors the `INFERENCE_ENGINE` env var).
    pub inference_engine: Option<String>,
    pub max_iterations: Option<u32>,
    /// Model context window, in tokens. Drives per-thread compaction; `0`
    /// disables it, which is only ever right for a test.
    pub context_window: u32,
    /// Extra SKILL.md directories from the launch config's `skillPaths`.
    pub skill_paths: Vec<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            base_url: String::new(),
            model: String::new(),
            api_key: None,
            temperature: None,
            max_tokens: 0,
            reasoning_effort: None,
            inference_engine: None,
            max_iterations: None,
            context_window: memory::DEFAULT_CONTEXT_WINDOW,
            skill_paths: Vec::new(),
        }
    }
}

/// One conversation. Owns its tools and message history, and shares a provider
/// with every other thread on the same model — the client's `threadId` is the
/// handle.
struct Thread {
    provider: Arc<dyn LlmProvider>,
    registry: ToolRegistry,
    /// Catalogued into every turn's prompt. Was built empty and never loaded,
    /// which left `lookup_skill` advertised but unable to find anything.
    skills: Arc<SkillRegistry>,
    messages: Mutex<Vec<ChatMessage>>,
    max_iterations: Option<u32>,
    /// The turn currently running, read by this thread's `RemoteTool`s so their
    /// callbacks carry the right `turnId`.
    current_turn: Arc<Mutex<String>>,
    /// The id of the turn in flight, or `None` between turns.
    ///
    /// One turn at a time per thread. That used to be enforced by accident —
    /// `turn/start` ran the turn on the request's own thread while holding
    /// `messages`, so a second call simply blocked. Now that a turn is answered
    /// immediately and runs in the background, a second one has to be refused
    /// out loud, which is also codex's model: it rejects an interrupt whose
    /// `turnId` is not the active one, because there is only ever one.
    active_turn: Mutex<Option<String>>,
    context_window: u32,
    /// Peak prompt size of the previous turn, which is what tells us whether
    /// this turn needs history compacted first. `0` until a turn reports usage.
    last_input_tokens: AtomicU64,
}

/// Relays ReAct progress to the client, so a long turn shows its work rather
/// than going silent for minutes.
///
/// A tool call is **two** notifications sharing one item id: `item/started` when
/// it is announced, `item/completed` when it returns. That pairing is the
/// protocol's, not ours — a client dispatches on the method to decide whether an
/// item is still running, and on the item's `type` to decide how to render it.
///
/// Both were wrong here, in ways that cancelled out into silence. The start
/// announcement went out as `item/completed`, so a client saw every tool as
/// already finished the moment it began; and the real result carried
/// `type: "toolResult"`, which is not a variant in the protocol's item
/// taxonomy, so a client dispatching on `type` dropped it and never showed the
/// output at all. `../klein-cli` (`internal/agentserver/runner.go`) is the
/// worked example: `render` switches over `commandExecution` / `fileChange` /
/// `mcpToolCall` / `dynamicToolCall` / `webSearch`, and its `item/started`
/// branch had nothing to receive.
struct NotifyingObserver<'a> {
    conn: &'a Arc<Connection>,
    thread_id: &'a str,
    turn_id: &'a str,
    /// Tool name → where that tool came from, which is what decides the item
    /// variant. The event carries only a name, and `ToolSource` lives on the
    /// descriptor, so the catalog is read once per turn rather than per call.
    sources: HashMap<String, ToolSource>,
}

impl<'a> NotifyingObserver<'a> {
    fn new(
        conn: &'a Arc<Connection>,
        thread_id: &'a str,
        turn_id: &'a str,
        tools: &dyn ToolAccess,
    ) -> Self {
        let sources = tools
            .descriptors()
            .into_iter()
            .map(|d| (d.name, d.source))
            .collect();
        Self {
            conn,
            thread_id,
            turn_id,
            sources,
        }
    }

    fn identify(&self, name: &str) -> Value {
        identify_tool(&self.sources, name)
    }
}

/// The item variant for a tool, and the fields that identify it.
///
/// `mcpToolCall` and `dynamicToolCall` are the two variants that name an
/// arbitrary tool and carry its arguments and result, which is the shape every
/// gallium tool has. `commandExecution` is deliberately not used even for
/// `Bash`: it is the protocol's *sandboxed shell* item, identified by an
/// `exitCode` and an `aggregatedOutput` that gallium does not track, and a
/// client rendering it labels the call `exec` and treats the tool name as the
/// command line — which is how `Read` came to display as a shell run.
///
/// A free function rather than a method so the mapping is testable without a
/// live `Connection`; the MCP arm is otherwise only reachable with a real
/// server attached.
fn identify_tool(sources: &HashMap<String, ToolSource>, name: &str) -> Value {
    match sources.get(name) {
        Some(ToolSource::Mcp { server }) => json!({
            "type": "mcpToolCall",
            "server": server,
            "tool": name,
        }),
        // A tool absent from the catalog is one the model named and the registry
        // refused. It still gets an item, so the client can show the attempt and
        // the error it produced.
        _ => json!({ "type": "dynamicToolCall", "tool": name }),
    }
}

impl AgentObserver for NotifyingObserver<'_> {
    fn on_event(&self, event: AgentEvent<'_>) {
        let (method, item) = match event {
            AgentEvent::ToolStarted {
                call_id,
                name,
                arguments,
            } => {
                let mut item = self.identify(name);
                merge(
                    &mut item,
                    json!({
                        "id": call_id,
                        // camelCase: the protocol's own spelling, which
                        // `../klein-cli` records as `stInProgress = "inProgress"`.
                        "status": "inProgress",
                        "arguments": arguments,
                    }),
                );
                ("item/started", item)
            }
            AgentEvent::ToolCompleted {
                call_id,
                name,
                result,
            } => {
                let mut item = self.identify(name);
                merge(
                    &mut item,
                    json!({
                        "id": call_id,
                        "status": if result.is_error { "failed" } else { "completed" },
                        "result": truncate_for_notification(&result.display_text()),
                    }),
                );
                ("item/completed", item)
            }
            // The turn's own text and usage reach the client through the
            // `turn/start` reply and `item/completed`, so relaying them here
            // would duplicate them on the wire. Errors surface as `turn/failed`.
            AgentEvent::Usage { .. }
            | AgentEvent::TurnCompleted { .. }
            | AgentEvent::Error { .. } => return,
        };
        self.conn.notify(
            method,
            json!({ "threadId": self.thread_id, "turnId": self.turn_id, "item": item }),
        );
    }
}

/// Fold `extra`'s fields into `target`, both JSON objects. Lets the variant and
/// the per-event fields be built separately and then be one item on the wire.
fn merge(target: &mut Value, extra: Value) {
    if let (Some(target), Some(extra)) = (target.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            target.insert(k.clone(), v.clone());
        }
    }
}

/// Tool output can be enormous (a whole file). The client only renders progress
/// from these, so cap what crosses the wire; the model still sees the full text.
///
/// This is the fallback. A tool that knows a better short form supplies one via
/// `ToolResult::displaying`, and the event already carries that instead — the
/// cap only catches tools that have not been given one.
const NOTIFICATION_TEXT_LIMIT: usize = 2000;

fn truncate_for_notification(text: &str) -> String {
    if text.len() <= NOTIFICATION_TEXT_LIMIT {
        return text.to_string();
    }
    // Cut on a char boundary at or below the limit.
    let mut end = NOTIFICATION_TEXT_LIMIT;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… ({} bytes total)", &text[..end], text.len())
}

/// Builds the LLM provider for a thread, given the server settings and the
/// model the thread asked for. Injectable so tests can drive a turn without a
/// real model behind it.
pub type ProviderFactory =
    Box<dyn Fn(&ServerConfig, &str) -> Result<Box<dyn LlmProvider>, AgentError> + Send + Sync>;

fn default_provider_factory(
    config: &ServerConfig,
    model: &str,
) -> Result<Box<dyn LlmProvider>, AgentError> {
    create_provider(
        config.model_path.clone(),
        config.base_url.clone(),
        model.to_string(),
        config.api_key.clone(),
        config.temperature,
        config.max_tokens,
        config.reasoning_effort.clone(),
        config.inference_engine.clone(),
    )
    .map_err(|e| AgentError::ConfigError(e.to_string()))
}

pub struct AppServer {
    config: ServerConfig,
    make_provider: ProviderFactory,
    /// Providers, keyed by the model they load. One process serves many threads,
    /// and a local provider owns multi-GB weights, so threads share these.
    providers: Mutex<HashMap<String, Arc<dyn LlmProvider>>>,
    threads: Mutex<HashMap<String, Arc<Thread>>>,
    next_thread: AtomicU64,
    next_turn: AtomicU64,
}

impl AppServer {
    pub fn new(config: ServerConfig) -> Self {
        Self::with_provider_factory(config, Box::new(default_provider_factory))
    }

    pub fn with_provider_factory(config: ServerConfig, make_provider: ProviderFactory) -> Self {
        Self {
            config,
            make_provider,
            providers: Mutex::new(HashMap::new()),
            threads: Mutex::new(HashMap::new()),
            next_thread: AtomicU64::new(1),
            next_turn: AtomicU64::new(1),
        }
    }

    /// The provider for `model`, built once and shared by every thread that asks
    /// for it.
    ///
    /// The key is the local model path when there is one: `create_provider`
    /// ignores the thread's `model` for a local config, so two threads naming
    /// different models still resolve to the same GGUF and must not each load it.
    fn provider_for(&self, model: &str) -> Result<Arc<dyn LlmProvider>, AgentError> {
        let key = self
            .config
            .model_path
            .clone()
            .unwrap_or_else(|| model.to_string());

        // Held across the build so two concurrent thread/starts cannot both load
        // the same model. Loading a GGUF takes seconds; a thread/start that waits
        // is better than one that duplicates gigabytes.
        let mut providers = self.providers.lock();
        if let Some(provider) = providers.get(&key) {
            tracing::debug!("reusing provider for '{}'", key);
            return Ok(Arc::clone(provider));
        }
        let provider: Arc<dyn LlmProvider> = Arc::from((self.make_provider)(&self.config, model)?);
        providers.insert(key, Arc::clone(&provider));
        Ok(provider)
    }

    fn handle_initialize(&self, params: &Value) -> HandlerResult {
        let client = params
            .get("clientInfo")
            .and_then(|c| c.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let experimental = params
            .get("capabilities")
            .and_then(|c| c.get("experimentalApi"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        // `dynamicTools` on thread/start is gated behind this capability in the
        // protocol. Gallium accepts threads either way, but a client that has not
        // negotiated it will never get its own tools registered.
        if !experimental {
            tracing::warn!(
                "client '{}' did not negotiate experimentalApi; its dynamicTools will be ignored",
                client
            );
        }
        tracing::info!("initialize from client '{}'", client);

        Ok(json!({
            "userAgent": format!("gallium/{}", env!("CARGO_PKG_VERSION")),
        }))
    }

    /// klein probes this before its first turn to catch an unauthenticated
    /// backend at startup. Gallium authenticates via its own config (an API key
    /// or a local GGUF), which `thread/start` validates by building the provider.
    fn handle_account_read(&self) -> HandlerResult {
        Ok(json!({ "requiresOpenaiAuth": false, "account": null }))
    }

    fn handle_thread_start(&self, conn: &Arc<Connection>, params: Value) -> HandlerResult {
        let params: ThreadStartParams = serde_json::from_value(params)
            .map_err(|e| RpcFault::invalid_params(format!("thread/start: {e}")))?;

        let thread_id = format!("thread_{}", self.next_thread.fetch_add(1, Ordering::SeqCst));

        let working_dir = params
            .cwd
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let model = params
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let provider = self.provider_for(&model)?;

        // Mutations are approved by the client, not by a terminal prompt — except
        // under `approvalPolicy: "never"`, where the client has said it does not
        // want to be asked. An absent policy is treated as "ask": failing toward
        // a question is safer than silently granting write access.
        let approver: Arc<dyn ApprovalSink> = match params.approval_policy.as_deref() {
            Some("never") => Arc::new(AutoApproveSink),
            _ => Arc::new(RemoteApprovalSink::new(Arc::clone(conn), thread_id.clone())),
        };
        let session = Arc::new(ToolSession::with_approver(approver));

        // Load the same skills the REPL does: the working dir's own, the
        // user-global ones, and anything the launch config listed.
        let skills = Arc::new(SkillRegistry::new());
        crate::skill::load_skills(&skills, &working_dir);
        for dir in &self.config.skill_paths {
            skills.load_from_dir(dir);
        }
        let mut registry =
            create_default_registry_with_session(working_dir, Arc::clone(&skills), session);

        // External MCP servers the client asked us to reach.
        crate::register_mcp_servers(&mut registry, &params.mcp_servers());

        // The client's own tools, dispatched back over this connection. They read
        // the live turn id out of the shared cell that `run_turn` sets.
        let current_turn = Arc::new(Mutex::new(String::new()));
        let dynamic_tools = params.dynamic_tools.clone();
        for spec in &dynamic_tools {
            registry.register(Box::new(RemoteTool::new(
                Arc::clone(conn),
                spec.clone(),
                thread_id.clone(),
                Arc::clone(&current_turn),
            )));
        }

        let mut messages = Vec::new();
        if let Some(instructions) = params.developer_instructions.filter(|s| !s.is_empty()) {
            messages.push(ChatMessage::system(instructions));
        }

        let thread = Arc::new(Thread {
            provider,
            registry,
            skills,
            messages: Mutex::new(messages),
            max_iterations: self.config.max_iterations,
            current_turn,
            active_turn: Mutex::new(None),
            context_window: self.config.context_window,
            last_input_tokens: AtomicU64::new(0),
        });
        self.threads.lock().insert(thread_id.clone(), thread);

        tracing::info!(
            "thread {} started ({} dynamic tools)",
            thread_id,
            dynamic_tools.len()
        );
        Ok(json!({ "threadId": thread_id }))
    }

    fn handle_turn_start(&self, conn: &Arc<Connection>, params: Value) -> HandlerResult {
        let params: TurnStartParams = serde_json::from_value(params)
            .map_err(|e| RpcFault::invalid_params(format!("turn/start: {e}")))?;

        let thread = self
            .threads
            .lock()
            .get(&params.thread_id)
            .cloned()
            .ok_or_else(|| {
                RpcFault::invalid_params(format!("unknown thread '{}'", params.thread_id))
            })?;

        let turn_id = format!("turn_{}", self.next_turn.fetch_add(1, Ordering::SeqCst));
        let prompt = params.prompt();

        // Claim the thread's one turn slot before answering. Refusing here — with
        // the running turn named — is the only honest answer: the second turn
        // would otherwise sit on `messages` until the first finished, and the
        // client would have no idea its request had not started.
        {
            let mut active = thread.active_turn.lock();
            if let Some(running) = active.as_ref() {
                return Err(RpcFault::invalid_params(format!(
                    "thread '{}' is already running turn '{running}'; \
                     one turn at a time",
                    params.thread_id
                )));
            }
            *active = Some(turn_id.clone());
        }

        // The turn runs in the background and reports through notifications.
        // Answering `turn/start` immediately is what codex does, and what makes
        // a turn interruptible at all: a reply that only arrives once the turn
        // is over cannot be the thing a client waits on while stopping it.
        let worker = TurnWorker {
            conn: Arc::clone(conn),
            thread: Arc::clone(&thread),
            thread_id: params.thread_id.clone(),
            turn_id: turn_id.clone(),
        };
        // `Builder::spawn` rather than `thread::spawn`, which panics when the OS
        // refuses a thread. Panicking here would take down the request handler
        // *after* the slot was claimed and *before* the client was answered — a
        // thread wedged forever against a turn that never ran. Rare, and cheap
        // to hand back honestly: release the slot and say the turn did not
        // start, which is the one thing the client can act on.
        if let Err(e) = std::thread::Builder::new()
            .name(format!("gallium-{turn_id}"))
            .spawn(move || worker.run(prompt))
        {
            *thread.active_turn.lock() = None;
            return Err(RpcFault::from(AgentError::InternalError(format!(
                "could not start turn '{turn_id}': {e}"
            ))));
        }

        Ok(json!({ "turn": { "id": turn_id, "status": "inProgress" } }))
    }
}

/// One background turn: everything it needs to run and to report, owned rather
/// than borrowed, because it outlives the `turn/start` that started it.
struct TurnWorker {
    conn: Arc<Connection>,
    thread: Arc<Thread>,
    thread_id: String,
    turn_id: String,
}

impl TurnWorker {
    /// Run the turn and report how it ended, then release the thread's turn slot.
    ///
    /// Every exit reports something. `turn/start` has already been answered, so
    /// a turn that ended without a notification would leave the client waiting
    /// for one forever — this is the only place that can tell it.
    fn run(self, prompt: String) {
        let result = run_turn(
            &self.conn,
            &self.thread,
            &self.thread_id,
            &self.turn_id,
            prompt,
        );

        // Clear the slot and report the ending as one step, holding the slot's
        // lock across both.
        //
        // Either order alone races. Release first and a turn accepted in the gap
        // interleaves its notifications ahead of this turn's ending, so the
        // client sees two turns overlap. Notify first and a client that starts
        // its next turn the instant it reads `turn/completed` is refused for a
        // turn that has already finished.
        //
        // Holding the lock across both closes both: a concurrent `turn/start`
        // waits until the ending is on the wire, and by the time it can be
        // accepted the slot is already free. The clear happens first inside the
        // section only so that no ordering of the two writes can strand it.
        let mut active = self.thread.active_turn.lock();
        *active = None;

        match result {
            Ok(text) => {
                self.conn.notify(
                    "item/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "item": { "type": "agentMessage", "text": text },
                    }),
                );
                self.conn.notify(
                    "turn/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turn": { "id": self.turn_id, "status": "completed" },
                    }),
                );
            }
            // Kept as `turn/failed` rather than folded into `turn/completed`
            // with a failed status. Codex has no `turn/failed` and spells every
            // ending as `turn/completed` — but clients key off the method, not
            // the status (klein's `classifyNote` does exactly that), so making
            // that change here would turn every failure into a silent success on
            // the client. It is a real divergence and belongs in its own change,
            // alongside the client work that makes it safe.
            Err(e) => {
                tracing::warn!(
                    "thread {} turn {} failed: {}",
                    self.thread_id,
                    self.turn_id,
                    e
                );
                self.conn.notify(
                    "turn/failed",
                    json!({
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "error": { "message": e.to_string() },
                    }),
                );
            }
        }

        // Explicit, because the guard's lifetime is the whole point: until here,
        // no other turn on this thread can be accepted.
        drop(active);
    }
}

/// Run the ReAct loop for one turn against the thread's accumulated history.
///
/// A free function rather than a method: it runs on a thread that outlives the
/// request, so it cannot borrow the server.
fn run_turn(
    conn: &Arc<Connection>,
    thread: &Thread,
    thread_id: &str,
    turn_id: &str,
    prompt: String,
) -> Result<String, AgentError> {
    // Publish the turn id before any tool can fire a callback for it.
    *thread.current_turn.lock() = turn_id.to_string();

    let mut messages = thread.messages.lock();

    let observer = NotifyingObserver::new(conn, thread_id, turn_id, &thread.registry);
    let setup = TurnSetup {
        provider: thread.provider.as_ref(),
        tools: &thread.registry,
        skills: Some(&thread.skills),
        max_iterations: thread.max_iterations,
        context_window: thread.context_window,
        observer: Some(&observer),
        // No context yet: the protocol has no method that would cancel a
        // turn, so a token here would be one nothing can ever set. #28 adds
        // that method — `turn/interrupt` — and this is the field it fills
        // in. The turn now running in the background, with its id in the
        // thread's `active_turn`, is what that method will target.
        context: None,
    };

    let last_input_tokens = thread.last_input_tokens.load(Ordering::Relaxed);
    let outcome = runtime::run_turn(&setup, &mut messages, last_input_tokens, prompt)?;

    if outcome.compacted > 0 {
        tracing::info!(
            "thread {}: compacted history, dropped {} messages \
                 (last turn peaked at {} tokens, window {})",
            thread_id,
            outcome.compacted,
            last_input_tokens,
            thread.context_window,
        );
    }

    // Drives the next turn's compaction decision.
    thread
        .last_input_tokens
        .store(outcome.usage.peak_input_tokens, Ordering::Relaxed);

    Ok(outcome.text)
}

impl RequestHandler for AppServer {
    fn handle_request(&self, conn: &Arc<Connection>, method: &str, params: Value) -> HandlerResult {
        match method {
            "initialize" => self.handle_initialize(&params),
            "account/read" => self.handle_account_read(),
            "thread/start" => self.handle_thread_start(conn, params),
            "turn/start" => self.handle_turn_start(conn, params),
            _ => Err(RpcFault::method_not_found(method)),
        }
    }

    /// A notification we do not know is the client speaking a protocol we have
    /// not caught up with, which is worth saying out loud.
    ///
    /// It logs at `warn`, not `debug`: the default filter is `info`, so `debug`
    /// is another way of spelling "silently ignored". Both sides of this
    /// protocol are hand-written against the same spec and drift apart quietly —
    /// #49 was exactly that, invisible for as long as it took someone to compare
    /// the two implementations by hand. An unknown method is cheap to report and
    /// there is no legitimate flood of them.
    fn handle_notification(&self, _conn: &Arc<Connection>, method: &str, _params: Value) {
        match method {
            "initialized" => tracing::debug!("client finished initialization"),
            other => tracing::warn!(
                "ignoring unknown notification '{}' — the client may speak a \
                 newer app-server protocol than this build implements",
                other
            ),
        }
    }
}

// ============================================================================
// Wire params
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ThreadStartParams {
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    developer_instructions: Option<String>,
    /// `never` auto-approves mutations; anything else asks the client. Gallium has
    /// no sandbox of its own, so codex's `sandbox` field is ignored.
    #[serde(default)]
    approval_policy: Option<String>,
    #[serde(default)]
    dynamic_tools: Vec<DynamicToolSpec>,
    /// `config.mcp_servers` — codex nests MCP config under a free-form table.
    #[serde(default)]
    config: Option<Value>,
}

impl ThreadStartParams {
    /// Pull `config.mcp_servers` out of codex's free-form config table. An entry
    /// carries either `url` (Streamable HTTP) or `command`/`args` (stdio); an
    /// entry with neither is skipped.
    fn mcp_servers(&self) -> Vec<McpServerConfig> {
        let Some(servers) = self
            .config
            .as_ref()
            .and_then(|c| c.get("mcp_servers"))
            .and_then(Value::as_object)
        else {
            return Vec::new();
        };

        servers
            .values()
            .filter_map(|entry| {
                let url = entry
                    .get("url")
                    .and_then(Value::as_str)
                    .filter(|u| !u.is_empty())
                    .map(str::to_string);
                let command = entry.get("command").and_then(Value::as_str).unwrap_or("");
                if url.is_none() && command.is_empty() {
                    return None;
                }
                let args = entry
                    .get("args")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                Some(McpServerConfig {
                    command: command.to_string(),
                    args,
                    url,
                })
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnStartParams {
    thread_id: String,
    #[serde(default)]
    input: Vec<Value>,
}

impl TurnStartParams {
    /// Concatenate the text items of the turn input. Non-text items (images) are
    /// not yet carried through.
    fn prompt(&self) -> String {
        self.input
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(pairs: &[(&str, ToolSource)]) -> HashMap<String, ToolSource> {
        pairs
            .iter()
            .map(|(n, s)| (n.to_string(), s.clone()))
            .collect()
    }

    /// An MCP tool names both its server and itself: a client renders it as
    /// `server/tool`, and without the server it loses which one answered.
    #[test]
    fn an_mcp_tool_is_identified_by_its_server_and_name() {
        let sources = sources(&[(
            "read_godoc",
            ToolSource::Mcp {
                server: "godevmcp".to_string(),
            },
        )]);

        let item = identify_tool(&sources, "read_godoc");

        assert_eq!(item["type"], "mcpToolCall");
        assert_eq!(item["server"], "godevmcp");
        assert_eq!(item["tool"], "read_godoc");
    }

    /// Built-ins and client-declared tools share a variant. Both are "a named
    /// tool with arguments and a result", which is all the protocol offers for
    /// something that is not a shell, a file change, or a web search.
    #[test]
    fn builtin_and_client_tools_are_identified_as_named_tool_calls() {
        let sources = sources(&[
            ("Read", ToolSource::Builtin),
            ("memory", ToolSource::Dynamic),
        ]);

        for name in ["Read", "memory"] {
            let item = identify_tool(&sources, name);
            assert_eq!(item["type"], "dynamicToolCall", "{name}");
            assert_eq!(item["tool"], name);
        }
    }

    /// `Bash` is not the protocol's `commandExecution`: that item is identified
    /// by an `exitCode` and an `aggregatedOutput` gallium does not track, and a
    /// client renders it as a shell line rather than as the tool it was.
    #[test]
    fn no_tool_is_identified_as_a_sandboxed_shell() {
        let sources = sources(&[("Bash", ToolSource::Builtin)]);

        assert_eq!(identify_tool(&sources, "Bash")["type"], "dynamicToolCall");
    }

    /// A name the model invented, which the registry then refused. It still
    /// gets an item, so the client can show the attempt and the error.
    #[test]
    fn a_tool_missing_from_the_catalog_still_gets_an_item() {
        let item = identify_tool(&HashMap::new(), "no_such_tool");

        assert_eq!(item["type"], "dynamicToolCall");
        assert_eq!(item["tool"], "no_such_tool");
    }

    #[test]
    fn turn_prompt_joins_text_items() {
        let params: TurnStartParams = serde_json::from_value(json!({
            "threadId": "t1",
            "input": [
                { "type": "text", "text": "hello" },
                { "type": "text", "text": "world" },
            ],
        }))
        .unwrap();
        assert_eq!(params.thread_id, "t1");
        assert_eq!(params.prompt(), "hello\nworld");
    }

    #[test]
    fn turn_prompt_skips_non_text_items() {
        let params: TurnStartParams = serde_json::from_value(json!({
            "threadId": "t1",
            "input": [{ "type": "image", "imageUrl": "data:..." }, { "type": "text", "text": "hi" }],
        }))
        .unwrap();
        assert_eq!(params.prompt(), "hi");
    }

    #[test]
    fn thread_start_parses_dynamic_tools_and_instructions() {
        let params: ThreadStartParams = serde_json::from_value(json!({
            "cwd": "/tmp",
            "developerInstructions": "be brief",
            "dynamicTools": [
                { "type": "function", "name": "memory", "description": "d", "inputSchema": {"type": "object"} },
            ],
        }))
        .unwrap();
        assert_eq!(params.cwd.as_deref(), Some("/tmp"));
        assert_eq!(params.developer_instructions.as_deref(), Some("be brief"));
        assert_eq!(params.dynamic_tools.len(), 1);
        assert_eq!(params.dynamic_tools[0].name, "memory");
    }

    #[test]
    fn thread_start_tolerates_a_bare_params_object() {
        let params: ThreadStartParams = serde_json::from_value(json!({})).unwrap();
        assert!(params.dynamic_tools.is_empty());
        assert!(params.mcp_servers().is_empty());
    }

    #[test]
    fn extracts_stdio_mcp_servers() {
        let params: ThreadStartParams = serde_json::from_value(json!({
            "config": { "mcp_servers": { "local": { "command": "srv", "args": ["--a"] } } },
        }))
        .unwrap();

        let servers = params.mcp_servers();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].command, "srv");
        assert_eq!(servers[0].args, vec!["--a"]);
        assert!(
            servers[0].url.is_none(),
            "stdio server must not carry a url"
        );
    }

    #[test]
    fn extracts_http_mcp_servers() {
        let params: ThreadStartParams = serde_json::from_value(json!({
            "config": { "mcp_servers": { "remote": { "url": "https://example.com/mcp" } } },
        }))
        .unwrap();

        let servers = params.mcp_servers();
        assert_eq!(
            servers.len(),
            1,
            "url servers reach the Streamable HTTP transport"
        );
        assert_eq!(servers[0].url.as_deref(), Some("https://example.com/mcp"));
    }

    #[test]
    fn skips_mcp_entries_with_neither_command_nor_url() {
        let params: ThreadStartParams = serde_json::from_value(json!({
            "config": {
                "mcp_servers": {
                    "broken": { "env": { "A": "1" } },
                    "empty_url": { "url": "" },
                    "good": { "command": "srv" },
                },
            },
        }))
        .unwrap();

        let servers = params.mcp_servers();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].command, "srv");
    }

    #[test]
    fn notification_text_is_truncated_on_a_char_boundary() {
        let text = "é".repeat(NOTIFICATION_TEXT_LIMIT); // 2 bytes each
        let out = truncate_for_notification(&text);
        assert!(out.contains("bytes total"));
        // Must not have panicked or produced invalid UTF-8.
        assert!(out.starts_with('é'));
    }

    #[test]
    fn short_notification_text_passes_through_unchanged() {
        assert_eq!(truncate_for_notification("hi"), "hi");
    }
}
