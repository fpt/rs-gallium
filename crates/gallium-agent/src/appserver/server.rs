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
//! | `turn/steer`    | in        | add user text to the turn already running |
//! | `turn/interrupt`| in        | stop the running turn; answers once it    |
//! |                 |           | has actually stopped                      |
//! | `item/tool/call`| out       | invoke a client-provided dynamic tool     |
//! | `item/*/requestApproval` | out | ask the client to permit a mutation  |
//! | `item/started`  | out       | a tool call was announced                 |
//! | `item/completed`, `turn/completed` | out | progress; the turn's    |
//! |                 |           | `status` says how it ended                |

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};

use crossbeam::channel::{Receiver, RecvTimeoutError, Sender};

use crate::approval::{ApprovalBroker, ApprovalPolicy, ApprovalSink};
use crate::appserver::rpc::{Connection, HandlerResult, RequestHandler, RpcFault};
use crate::appserver::tools::{AutoApproveSink, DynamicToolSpec, RemoteApprovalSink, RemoteTool};
use crate::cancel::{CancellationToken, SteerInbox, TurnContext};
use crate::event::{AgentEvent, AgentObserver};
use crate::input::{self, UserInput};
use crate::llm::{create_provider, ChatMessage, LlmProvider, MediaContent, TokenUsage};
use crate::memory;
use crate::runtime::{self, TurnSetup};
use crate::skill::SkillRegistry;
use crate::tool::{
    create_default_registry_with_session, create_registry_without_workspace_tools, ToolAccess,
    ToolRegistry, ToolResult, ToolSession, ToolSource,
};
use crate::tool_search::ToolSearchTool;
use crate::trace::{TraceMeta, TraceSession};
use crate::{AgentError, McpServerConfig};

/// Settings the process is launched with; a thread inherits these unless
/// `thread/start` overrides them.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_path: Option<String>,
    /// Multimodal projector (`mmproj-*.gguf`) for the llama.cpp backend. `None`
    /// is text only, and a turn carrying an image is refused rather than
    /// answered blind.
    pub mmproj_path: Option<String>,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    /// Nucleus-sampling threshold, llama.cpp backend only — see
    /// `llm_local::LlamaLocalProvider::top_p`. `LLM_TOP_P` still overrides
    /// it, matching `temperature`.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff, llama.cpp backend only — see
    /// `llm_local::LlamaLocalProvider::top_k`. `LLM_TOP_K` still overrides
    /// it, matching `temperature`.
    pub top_k: Option<u32>,
    pub max_tokens: u32,
    pub reasoning_effort: Option<String>,
    /// Local inference backend: "llamacpp" (default) or "candle". `None`
    /// auto-detects (and still honors the `INFERENCE_ENGINE` env var).
    pub inference_engine: Option<String>,
    /// Where the native candle backend finds `tokenizer.json`. Only that engine
    /// reads it; llama.cpp uses the one inside the GGUF.
    pub tokenizer_path: Option<String>,
    /// GPU layers to offload for the llama.cpp backend. `None` leaves it to
    /// llama.cpp's own default (999, offload everything); `GALLIUM_GPU_LAYERS`
    /// still overrides it.
    pub gpu_layers: Option<u32>,
    /// Move MoE expert tensors to CPU for the llama.cpp backend. `false`
    /// leaves them offloaded same as everything else; `GALLIUM_CPU_MOE`
    /// still overrides it.
    pub cpu_moe: bool,
    /// Which model profile reads the model's output, or `None` to detect it from
    /// what the model file reports. `GALLIUM_PROFILE` still overrides it.
    pub profile: Option<String>,
    pub max_iterations: Option<u32>,
    /// Model context window, in tokens, as *explicitly configured*. `None` lets
    /// the provider report its own (see `LlmProvider::context_window`), and
    /// falls back to a guess only if it cannot. `Some(0)` disables compaction,
    /// which is only ever right for a test.
    pub context_window: Option<u32>,
    /// Extra SKILL.md directories from the launch config's `skillPaths`.
    pub skill_paths: Vec<PathBuf>,
    /// Whether this process offers tools that act on its own machine (`Read`,
    /// `Write`, `Bash`, …). `false` leaves a thread with only the tools that
    /// touch nothing here — task bookkeeping and skill lookup — and expects the
    /// client's `dynamicTools` to be the hands. That is the split the TCP
    /// transport exists for: the model runs where the GPU is, the files are
    /// where the user is.
    pub workspace_tools: bool,
    /// Where per-turn traces go, from the launch config's `[agent.trace] dir`.
    /// `None` leaves it to the `GALLIUM_TRACE` env vars, and to nothing after
    /// that.
    pub trace_dir: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            mmproj_path: None,
            base_url: String::new(),
            model: String::new(),
            api_key: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: 0,
            reasoning_effort: None,
            inference_engine: None,
            tokenizer_path: None,
            gpu_layers: None,
            cpu_moe: false,
            profile: None,
            max_iterations: None,
            context_window: None,
            skill_paths: Vec::new(),
            // A server that offers no tools is the deliberate arrangement, never
            // the accident of an unset field.
            workspace_tools: true,
            trace_dir: None,
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
    /// That turn's stop switch, or `None` between turns. Shared with the
    /// `RemoteApprovalSink`, which fires it when the client answers an approval
    /// with `cancel` — refuse *and* stop, which is one decision the protocol
    /// makes and `ApprovalDecision` cannot carry on its own.
    ///
    /// Separate from `ActiveTurn::cancel` (the same token) because the sink is
    /// built at `thread/start`, before any turn exists, and holds no reference
    /// to the thread it approves for.
    current_cancel: Arc<Mutex<Option<CancellationToken>>>,
    /// The tool call being executed, so an approval can name the item it belongs
    /// to — codex's `itemId` on both `requestApproval` methods.
    ///
    /// Written by `NotifyingObserver` on `ToolStarted`, which `react.rs` emits
    /// immediately before running that call, and cleared on `ToolCompleted`.
    /// That is exact rather than approximate because the ReAct loop runs tool
    /// calls one at a time on one thread: the approval a tool raises can only
    /// belong to the call the observer last announced.
    ///
    /// The alternative was threading a call id through `ApprovalRequest` and
    /// every `Tool` impl that builds one, to carry a fact only the app-server
    /// has any use for.
    current_item: Arc<Mutex<Option<String>>>,
    /// The item id of the `agentMessage` currently being streamed as
    /// `item/agentMessage/delta`, or `None` when no deltas are in flight.
    ///
    /// The first delta of a run mints it; the deltas, and then the
    /// `item/completed` that finalises that message — whether it is the ending's
    /// final answer, a steered mid-turn message, or a message cut short by a
    /// tool call — all carry the same id, so a client keys them together. Shared
    /// with `TurnWorker` because the ending's `item/completed` is emitted there,
    /// not by the observer.
    streaming_item: Arc<Mutex<Option<String>>>,
    /// The turn in flight, or `None` between turns.
    ///
    /// One turn at a time per thread. That used to be enforced by accident —
    /// `turn/start` ran the turn on the request's own thread while holding
    /// `messages`, so a second call simply blocked. Now that a turn is answered
    /// immediately and runs in the background, a second one has to be refused
    /// out loud, which is also codex's model: it rejects an interrupt whose
    /// `turnId` is not the active one, because there is only ever one.
    active_turn: Mutex<Option<ActiveTurn>>,
    /// What compaction measures against. Always a number, because compaction
    /// needs a policy even when nobody can say what the real window is.
    context_window: u32,
    /// The same window, but only when it is *known* — configured explicitly, or
    /// reported by the provider from the model's own metadata. `None` is a
    /// built-in fallback that nobody vouched for, and a client is told nothing
    /// rather than shown a gauge drawn against a guess.
    known_context_window: Option<u32>,
    /// Peak prompt size of the previous turn, which is what tells us whether
    /// this turn needs history compacted first. `0` until a turn reports usage.
    last_input_tokens: AtomicU64,
    /// Everything this thread has spent, across all its turns — codex's `total`
    /// beside its `last`. Cumulative because a context gauge is about the
    /// conversation, not about the turn that happens to be running.
    total_usage: Mutex<TokenUsage>,
    /// Where this thread's turns are recorded, when the operator asked for it.
    /// Per thread rather than per process: it carries the thread's own workspace
    /// and approval policy, and the broker whose decisions it attributes.
    trace: Option<TraceSession>,
}

/// The turn running on a thread right now: what to call it, how to stop it, and
/// how to tell when it has.
struct ActiveTurn {
    id: String,
    /// Set by `turn/interrupt`; checked at every loop boundary in the ReAct
    /// loop, between sampled tokens, and between polls of `bash`'s child.
    cancel: CancellationToken,
    /// Written by `turn/steer`; drained by the ReAct loop before each model
    /// call, and again when the model returns text.
    steer: SteerInbox,
    /// Closes when the worker finishes, whatever the outcome — the worker holds
    /// the sending half and never sends on it, so `recv` returning `Err` is the
    /// turn being over.
    ///
    /// This is how `turn/interrupt` waits. Codex answers an interrupt only once
    /// the turn has actually aborted, which is the difference between a stop
    /// button and a doorbell, and gallium can say the same thing by blocking:
    /// every request already runs on its own thread.
    finished: Receiver<Never>,
}

/// A channel that carries nothing; only its closing is the signal.
enum Never {}

/// The turns `AppServer::cancel_turns` asked to stop, not yet stopped.
///
/// Waiting is deliberately a separate step — see `cancel_turns` — and the
/// `#[must_use]` there is what keeps a caller from cancelling and walking away,
/// which reads as "the turns are stopped" and is not.
pub struct StoppingTurns(Vec<Receiver<Never>>);

impl StoppingTurns {
    /// Block until every cancelled turn has ended, whatever its outcome.
    pub fn wait(self) {
        for finished in self.0 {
            // Err is the worker dropping the sending half, i.e. the turn ending.
            // Nothing is ever sent: only the close is the signal.
            let _ = finished.recv();
        }
    }

    /// Wait until `deadline` for the turns to end, keeping the ones that have
    /// not. `true` means they are all stopped.
    ///
    /// Retaining what is still running is what makes this callable again: the
    /// turns that ended are not waited on twice, so a caller that gives up can
    /// come back and ask the same question about what is left.
    fn wait_until(&mut self, deadline: Instant) -> bool {
        self.0
            .retain(|finished| match finished.recv_deadline(deadline) {
                // The worker dropped its sender: this turn is over.
                Err(RecvTimeoutError::Disconnected) => false,
                // Still running. Keep it, and note that the next receiver in this
                // pass gets the same *absolute* deadline rather than a fresh
                // interval, so the whole call is bounded by it however many turns
                // are being waited on.
                Err(RecvTimeoutError::Timeout) => true,
                // Nothing is ever sent on this channel; `Never` has no values.
                Ok(never) => match never {},
            });
        self.0.is_empty()
    }
}

/// How long a turn will wait for the connection it displaced to finish.
///
/// Long enough to cover the ending this is really about — a cloud round trip
/// with no interruption point, which completes in seconds — and short enough
/// that a client is told something rather than left holding a turn that never
/// starts. A local model is cancelled between sampled tokens and never comes
/// near this.
const PREDECESSOR_GRACE: Duration = Duration::from_secs(60);

/// The turns a displaced connection was running, inherited by the connection
/// that displaced it.
///
/// This is where the one-at-a-time rule is actually enforced. It used to be
/// enforced on the accept loop, which held the *whole* replacement connection —
/// socket accepted, nobody reading it — until the old turn stopped. That put
/// the lockout displacement exists to prevent back one step: over an OpenAI
/// provider, whose round trip has no interruption point, the reconnect the user
/// made to get their machine back is the thing that hangs.
///
/// The invariant was never about the socket. It is that two turns must not talk
/// to the model — and share its KV slots — at once. So the connection is served
/// immediately: it initializes, starts threads, and is visibly alive. Only its
/// first *turn* waits, and only for as long as the old turn takes to stop.
#[derive(Default)]
pub struct Predecessor(Mutex<Option<StoppingTurns>>);

impl Predecessor {
    fn adopt(&self, stopping: StoppingTurns) {
        *self.0.lock() = Some(stopping);
    }

    /// Wait for the displaced turns to stop. `false` means they had not within
    /// `within`, and the caller must not proceed to the model.
    ///
    /// The lock is held across the wait deliberately: every turn on this
    /// connection needs the same answer, so concurrent starts queue behind the
    /// first rather than each opening their own deadline. It is the outermost
    /// lock in this file — taken before `accepting_turns`, and never while
    /// holding it or `active_turn`, since the turns being waited for take those
    /// on their way out.
    /// Hand back whatever has not stopped yet.
    ///
    /// A connection can be displaced while still waiting on the one *it*
    /// displaced — three reconnects in quick succession, which is a laptop
    /// roaming between networks. Passing the unfinished inheritance along keeps
    /// the guarantee transitive; dropping it would let the third connection's
    /// turn reach the model beside the first's, which is the overlap this whole
    /// mechanism exists to prevent.
    fn take_unsettled(&self) -> Vec<Receiver<Never>> {
        self.0.lock().take().map(|s| s.0).unwrap_or_default()
    }

    fn settle(&self, within: Duration) -> bool {
        let mut held = self.0.lock();
        let Some(stopping) = held.as_mut() else {
            return true;
        };
        if stopping.wait_until(Instant::now() + within) {
            // Drop the handle so later turns take the cheap path above rather
            // than re-examining an empty list.
            *held = None;
            return true;
        }
        false
    }
}

/// Where gallium keeps its user-level state — codex's `$CODEX_HOME`, answered
/// with the directory gallium actually uses (`~/.config/gallium`, the same one
/// `config::default_config_path` and the global skill loader read).
///
/// Absolute, because codex's field is an `AbsolutePathBuf`. Falls back to the
/// working directory when there is no home to speak of, which is somewhere real
/// rather than a path that would fail to parse.
fn gallium_home() -> String {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| {
            PathBuf::from(home)
                .join(".config")
                .join("gallium")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .to_string_lossy()
                .into_owned()
        })
}

/// Codex's `Turn`, which both `turn/start` and `turn/completed` carry whole.
///
/// `items` is required and gallium never fills it: items are streamed as
/// `item/*` notifications and are not reassembled into the turn. `itemsView`
/// says which of the two silences this is — `full` when the turn genuinely has
/// no items yet (it is only starting), `notLoaded` when it had them and this
/// payload simply is not where they live. The distinction is codex's own, and
/// it is the difference between "nothing happened" and "look elsewhere".
/// The item id the turn's final `agentMessage` carries — shared by the
/// `item/completed` the ending emits and the `item/agentMessage/delta`
/// notifications streamed on the way there, so a client keys them together. Its
/// namespace (`_item_final`) is one the observer's per-turn counter never mints,
/// so a steered mid-turn message can never collide with it.
fn final_message_item_id(turn_id: &str) -> String {
    format!("{turn_id}_item_final")
}

fn turn_object(id: &str, status: &str, items_loaded: bool) -> Value {
    json!({
        "id": id,
        "items": [],
        "itemsView": if items_loaded { "full" } else { "notLoaded" },
        "status": status,
    })
}

/// The same `Turn`, ended by a failure.
///
/// `error` is populated only when the status is `failed` — codex says so on the
/// field — so this is a separate constructor rather than an `Option` parameter
/// on every call: the two always travel together, and neither is meaningful
/// without the other.
///
/// Only `message` is filled. `codexErrorInfo` is codex's own taxonomy of its own
/// failures and gallium has nothing honest to put in it; `additionalDetails`
/// would be a second copy of the same string.
fn failed_turn(id: &str, message: &str) -> Value {
    let mut turn = turn_object(id, "failed", false);
    merge(&mut turn, json!({ "error": { "message": message } }));
    turn
}

/// Seconds since the epoch, codex's timestamp unit for threads and turns.
fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Milliseconds since the epoch, codex's unit for `startedAtMs` on approvals.
pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
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
    /// The thread's running total, which this observer adds to as usage arrives.
    /// Borrowed rather than owned: the total outlives the turn.
    total_usage: &'a Mutex<TokenUsage>,
    /// The window to report alongside, or `None` to report none.
    known_context_window: Option<u32>,
    /// Published so an approval raised inside a tool call can name the item that
    /// call belongs to. See `Thread::current_item`.
    current_item: &'a Mutex<Option<String>>,
    /// The streamed `agentMessage`'s id — see `Thread::streaming_item`. Shared
    /// so the ending's `item/completed` (emitted by `TurnWorker`, not here) uses
    /// the same id the deltas did.
    streaming_item: &'a Mutex<Option<String>>,
    /// The delta text accumulated for the message in `streaming_item`, so a
    /// message cut short by a tool call can still be finalised with what the
    /// client already saw. Owned: only the observer touches it.
    streaming_text: Mutex<String>,
    /// Mints ids for the items that are not tool calls — the agent messages,
    /// which have no call id to borrow. Per turn, which is enough: an item id
    /// only has to be unique among the items a client is holding.
    next_item: AtomicU64,
}

impl<'a> NotifyingObserver<'a> {
    // Eight thread-owned cells, each a distinct concern; a params struct would
    // just move the list.
    #[allow(clippy::too_many_arguments)]
    fn new(
        conn: &'a Arc<Connection>,
        thread_id: &'a str,
        turn_id: &'a str,
        tools: &dyn ToolAccess,
        total_usage: &'a Mutex<TokenUsage>,
        known_context_window: Option<u32>,
        current_item: &'a Mutex<Option<String>>,
        streaming_item: &'a Mutex<Option<String>>,
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
            total_usage,
            known_context_window,
            current_item,
            streaming_item,
            streaming_text: Mutex::new(String::new()),
            next_item: AtomicU64::new(0),
        }
    }

    /// Finalise the streamed `agentMessage`, if one is open, with the text the
    /// client has already seen — for a message a tool call cuts short. `id` is
    /// returned so the caller can announce the tool under a fresh one.
    fn close_streaming_message(&self) {
        let Some(id) = self.streaming_item.lock().take() else {
            return;
        };
        let text = std::mem::take(&mut *self.streaming_text.lock());
        self.conn.notify(
            "item/completed",
            json!({
                "threadId": self.thread_id,
                "turnId": self.turn_id,
                "item": { "type": "agentMessage", "id": id, "text": text },
            }),
        );
    }

    fn identify(&self, name: &str) -> Value {
        identify_tool(&self.sources, name)
    }

    /// An id for an item that has none of its own, scoped to this turn.
    fn mint_item_id(&self) -> String {
        format!(
            "{}_item_{}",
            self.turn_id,
            self.next_item.fetch_add(1, Ordering::SeqCst)
        )
    }

    /// Codex's `thread/tokenUsage/updated`, sent as each model call reports what
    /// it cost.
    ///
    /// Per call rather than per turn, which is both codex's cadence and the more
    /// useful one: a tool-using turn can spend minutes growing its prompt, and a
    /// gauge that only moves when the turn ends is a gauge that does not move
    /// while the user is watching. A client wanting one number per turn takes
    /// the last of these before `turn/completed`.
    fn report_usage(&self, usage: &TokenUsage) {
        let total = {
            let mut total = self.total_usage.lock();
            total.add(usage);
            breakdown(&total)
        };
        self.conn.notify(
            "thread/tokenUsage/updated",
            json!({
                "threadId": self.thread_id,
                "turnId": self.turn_id,
                "tokenUsage": {
                    "total": total,
                    "last": breakdown(usage),
                    // Explicitly null rather than omitted when unknown: codex's
                    // field is nullable, and a client that sees the key and no
                    // value has been told "no gauge" rather than left to wonder
                    // whether this build sends the field at all.
                    "modelContextWindow": self.known_context_window,
                },
            }),
        );
    }
}

/// One usage record in codex's `TokenUsageBreakdown` shape.
///
/// The three fields gallium does not track are sent as zero rather than
/// omitted: the shape is fixed, and a consumer summing `inputTokens +
/// cachedInputTokens` should get the truth on either backend. Cache accounting
/// and a reasoning-token split are things the providers here do not report,
/// which is different from their being zero — but zero is the arithmetic-safe
/// spelling of "not counted separately".
fn breakdown(usage: &TokenUsage) -> Value {
    json!({
        "totalTokens": usage.total_tokens,
        "inputTokens": usage.input_tokens,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "outputTokens": usage.output_tokens,
        "reasoningOutputTokens": 0,
    })
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

/// A finished call's output, in whichever shape the item variant declares.
///
/// The two variants disagree, and gallium used to send one field — `result`, a
/// bare string — to both. Codex defines no such field on either: on
/// `mcpToolCall` it is an `McpToolCallResult` object, and on `dynamicToolCall`
/// the output lives in `contentItems` beside a `success` flag. A client
/// deserializing into those types rejected the string outright, and the output
/// it was carrying was the whole point of the notification.
///
/// `contentItems` mirrors what `dynamicTools` calls *send back* to gallium
/// (`tools::parse_tool_response` reads the same `inputText` shape), so the two
/// directions of a dynamic tool now speak the same language.
fn tool_output(item: &Value, result: &ToolResult) -> Value {
    let text = truncate_for_notification(&result.display_text());
    match item.get("type").and_then(Value::as_str) {
        Some("mcpToolCall") => json!({
            "result": {
                // MCP content blocks. Gallium's `ToolResult` is text by the time
                // it reaches here, so this is the one block it can honestly claim.
                "content": [{ "type": "text", "text": text }],
                "structuredContent": null,
                "_meta": null,
            },
        }),
        _ => json!({
            "contentItems": [{ "type": "inputText", "text": text }],
            "success": !result.is_error,
        }),
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
                // The model may have streamed a few words of answer before
                // calling this tool; finalise that message under its own id so
                // the deltas already on the wire belong to a completed item.
                self.close_streaming_message();
                // Published before the notification: the tool runs on this same
                // thread the moment this returns, and whatever approval it
                // raises has to find the item id already there.
                *self.current_item.lock() = Some(call_id.to_string());
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
                arguments,
            } => {
                *self.current_item.lock() = None;
                let mut item = self.identify(name);
                // `arguments` again, not only on the announcement: codex's
                // `mcpToolCall` and `dynamicToolCall` both require it, and an
                // item is meant to be a complete description of itself rather
                // than a patch against the `item/started` that preceded it.
                merge(
                    &mut item,
                    json!({
                        "id": call_id,
                        "status": if result.is_error { "failed" } else { "completed" },
                        "arguments": arguments,
                    }),
                );
                let output = tool_output(&item, result);
                merge(&mut item, output);
                ("item/completed", item)
            }
            // The model answered and steering carried the turn on. The same
            // `agentMessage` item the ending emits, sent here because this text
            // is *not* the ending — without it a steered turn would show only
            // its last answer and swallow every one before it. If this message
            // was streamed, it keeps the id its deltas carried; `text` is the
            // wire layer's cleaned version, which is authoritative over the
            // accumulated fragments.
            AgentEvent::AgentMessage { text } => {
                let id = self
                    .streaming_item
                    .lock()
                    .take()
                    .unwrap_or_else(|| self.mint_item_id());
                self.streaming_text.lock().clear();
                (
                    "item/completed",
                    json!({ "type": "agentMessage", "id": id, "text": text }),
                )
            }
            // A fragment of the answer, streamed as it decodes. codex's flat
            // `{threadId, turnId, itemId, delta}` shape, no `item` wrapper. The
            // first fragment mints the `itemId`; every later fragment, and the
            // `item/completed` that finalises this message — the ending's final
            // answer, a steered mid-turn message, or one a tool call cut short —
            // carries the same id, so a client keys them together. Best-effort:
            // only the candle backend produces these, and the finished message
            // is authoritative over what the fragments accumulate to.
            AgentEvent::MessageDelta { text } => {
                let id = self
                    .streaming_item
                    .lock()
                    .get_or_insert_with(|| self.mint_item_id())
                    .clone();
                self.streaming_text.lock().push_str(text);
                self.conn.notify(
                    "item/agentMessage/delta",
                    json!({
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "itemId": id,
                        "delta": text,
                    }),
                );
                return;
            }
            // Usage is not an item — it is a running property of the thread —
            // so it goes out as its own notification rather than through the
            // item stream.
            AgentEvent::Usage { usage } => {
                self.report_usage(usage);
                return;
            }
            // The turn's own text reaches the client through `item/completed`,
            // so relaying it here would duplicate it on the wire. Errors surface
            // as `turn/completed` with a `failed` status.
            AgentEvent::TurnCompleted { .. } | AgentEvent::Error { .. } => return,
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

pub(crate) fn default_provider_factory(
    config: &ServerConfig,
    model: &str,
) -> Result<Box<dyn LlmProvider>, AgentError> {
    create_provider(
        config.model_path.clone(),
        config.mmproj_path.clone(),
        config.base_url.clone(),
        model.to_string(),
        config.api_key.clone(),
        config.temperature,
        config.top_p,
        config.top_k,
        config.max_tokens,
        config.reasoning_effort.clone(),
        config.inference_engine.clone(),
        config.tokenizer_path.clone(),
        config.gpu_layers,
        config.cpu_moe,
        config.profile.clone(),
    )
    .map_err(|e| AgentError::ConfigError(e.to_string()))
}

/// The loaded models, keyed by the model they came from. One process serves many
/// threads, and a local provider owns multi-GB weights, so threads share these.
///
/// It is a separate object from `AppServer` because an `AppServer` serves one
/// *connection*, and the TCP listener has several: two clients on one GPU box
/// must share the weights, while sharing their thread tables would let one
/// client's `threadId` name the other client's conversation. So the expensive
/// half is shared between connections and the stateful half is not.
pub struct ProviderPool {
    make_provider: ProviderFactory,
    loaded: Mutex<HashMap<String, Arc<dyn LlmProvider>>>,
}

impl ProviderPool {
    pub fn new(make_provider: ProviderFactory) -> Arc<Self> {
        Arc::new(Self {
            make_provider,
            loaded: Mutex::new(HashMap::new()),
        })
    }

    /// The provider for `model`, built once and shared by every thread that asks
    /// for it.
    ///
    /// The key is the local model path when there is one: `create_provider`
    /// ignores the thread's `model` for a local config, so two threads naming
    /// different models still resolve to the same GGUF and must not each load it.
    fn provider_for(
        &self,
        config: &ServerConfig,
        model: &str,
    ) -> Result<Arc<dyn LlmProvider>, AgentError> {
        let key = config
            .model_path
            .clone()
            .unwrap_or_else(|| model.to_string());

        // Held across the build so two concurrent thread/starts cannot both load
        // the same model. Loading a GGUF takes seconds; a thread/start that waits
        // is better than one that duplicates gigabytes.
        let mut loaded = self.loaded.lock();
        if let Some(provider) = loaded.get(&key) {
            tracing::debug!("reusing provider for '{}'", key);
            return Ok(Arc::clone(provider));
        }
        let provider: Arc<dyn LlmProvider> = Arc::from((self.make_provider)(config, model)?);
        loaded.insert(key, Arc::clone(&provider));
        Ok(provider)
    }
}

pub struct AppServer {
    config: ServerConfig,
    providers: Arc<ProviderPool>,
    threads: Mutex<HashMap<String, Arc<Thread>>>,
    /// Whether this connection may still start turns. Set false by
    /// `cancel_turns`, and read by `turn/start` while it claims a thread's turn
    /// slot — the two under the same lock, which is what makes the cancellation
    /// snapshot complete. It is the outermost of this type's locks: `accepting`,
    /// then `threads`, then a thread's `active_turn`.
    accepting_turns: Mutex<bool>,
    /// The turns of the connection this one displaced, if it displaced one.
    /// Empty for a fresh listener and for stdio, where there is no predecessor
    /// to be had.
    predecessor: Arc<Predecessor>,
    next_thread: AtomicU64,
    next_turn: AtomicU64,
    /// Ids for items the server mints itself rather than taking from a tool
    /// call — today only the user messages `turn/steer` echoes back.
    next_item: AtomicU64,
}

impl AppServer {
    pub fn new(config: ServerConfig) -> Self {
        Self::with_provider_factory(config, Box::new(default_provider_factory))
    }

    pub fn with_provider_factory(config: ServerConfig, make_provider: ProviderFactory) -> Self {
        Self::with_pool(config, ProviderPool::new(make_provider))
    }

    /// A server sharing an already-built pool: one connection of several on the
    /// TCP listener, with its own threads and one set of weights between them.
    pub fn with_pool(config: ServerConfig, providers: Arc<ProviderPool>) -> Self {
        Self {
            config,
            providers,
            threads: Mutex::new(HashMap::new()),
            accepting_turns: Mutex::new(true),
            predecessor: Arc::new(Predecessor::default()),
            next_thread: AtomicU64::new(1),
            next_turn: AtomicU64::new(1),
            next_item: AtomicU64::new(1),
        }
    }

    /// Take on the turns of the connection this one is replacing, so this
    /// connection's first turn waits for them instead of its socket doing so.
    ///
    /// Called between cancelling those turns and serving this connection — see
    /// `Predecessor` for why the wait belongs here and not on the accept loop.
    pub fn adopt_predecessor(&self, stopping: StoppingTurns) {
        self.predecessor.adopt(stopping);
    }

    /// Cancel every turn running on this connection's threads, returning a
    /// handle that waits for them to actually stop.
    ///
    /// Cancelling and waiting are **two steps on purpose**, because the caller
    /// has to break the connection between them. A turn blocked in
    /// `Connection::request` — awaiting a dynamic tool result or an approval
    /// from the client being displaced — is not released by its cancellation
    /// token; only the reader loop exiting drops the pending table and unblocks
    /// it. Waiting before the socket is down would therefore hang on exactly the
    /// turn most in need of stopping.
    ///
    /// Stopping is prompt, not instant, on the same terms as `turn/interrupt`:
    /// the token is read at loop boundaries and between sampled tokens, so the
    /// wait is bounded by the slowest thing a turn is currently inside.
    #[must_use = "cancellation is not finished until the returned handle is waited on"]
    pub fn cancel_turns(&self) -> StoppingTurns {
        // Close the door *before* looking at what is running, and hold it
        // closed across the snapshot. A `turn/start` already dispatched on its
        // own handler thread but not yet registered would otherwise register
        // after the snapshot and run on beside the replacement client —
        // cancelled by nothing, waited for by nobody. Sharing this lock with the
        // slot claim leaves that request two outcomes and no third: it registers
        // before this call and is cancelled below, or it finds the door shut and
        // is refused.
        let mut accepting = self.accepting_turns.lock();
        *accepting = false;

        let threads: Vec<Arc<Thread>> = self.threads.lock().values().cloned().collect();
        let mut stopping = Vec::new();
        for thread in threads {
            // Cancel under the lock and wait outside it: the worker takes the
            // same lock to clear the slot when it finishes.
            let active = thread.active_turn.lock();
            if let Some(running) = active.as_ref() {
                running.cancel.cancel();
                stopping.push(running.finished.clone());
            }
        }
        // Whatever this connection was itself still waiting for goes with it:
        // it displaced someone too, and that turn may not have stopped yet.
        stopping.extend(self.predecessor.take_unsettled());
        StoppingTurns(stopping)
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

        // Codex's `InitializeResponse` requires all four of these — none is an
        // `Option` — so a client deserializing into that type failed at the
        // handshake when gallium sent only `userAgent`.
        //
        // `codexHome` is the odd one: gallium has no `$CODEX_HOME`. The honest
        // analogue is the directory gallium's own user-level config and global
        // skills live in, which is what the field is *for* — where this server
        // keeps its state. It is reported whether or not a config file is
        // actually there, because the question is where the server would look.
        Ok(json!({
            "userAgent": format!("gallium/{}", env!("CARGO_PKG_VERSION")),
            "codexHome": gallium_home(),
            // `unix` / `windows`, and `macos` / `linux` / `windows` — the same
            // values codex derives from its build target.
            "platformFamily": std::env::consts::FAMILY,
            "platformOs": std::env::consts::OS,
        }))
    }

    /// klein probes this before its first turn to catch an unauthenticated
    /// backend at startup. Gallium authenticates via its own config (an API key
    /// or a local GGUF), which `thread/start` validates by building the provider.
    fn handle_account_read(&self) -> HandlerResult {
        Ok(json!({ "requiresOpenaiAuth": false, "account": null }))
    }

    fn handle_thread_start(&self, conn: &Arc<Connection>, params: Value) -> HandlerResult {
        // The same door `turn/start` reads. A displaced connection's reader is
        // ending, but a `thread/start` already dispatched on its own handler
        // thread is not: without this it goes on building a thread — loading a
        // model, on the shared pool — for a client whose socket is already shut
        // and whose answer nobody will read. (rs-gallium#167)
        //
        // This **narrows** that window rather than closing it, and deliberately
        // so. The check is not atomic with the insertion at the end of this
        // function, so a displacement landing in between still leaves a thread
        // in a displaced connection's map. Nothing can run on it: `turn/start`
        // reads this same gate while it claims the turn slot, and the socket
        // that would carry such a request is already shut. The only cost is the
        // work this check exists to skip, sometimes not skipped.
        //
        // Making it atomic would mean holding `accepting_turns` across a
        // multi-GB GGUF load, which is the one thing that must not happen here:
        // `cancel_turns` takes that lock, so displacement — the thing this whole
        // file exists to keep prompt — would block behind a model load.
        if !*self.accepting_turns.lock() {
            return Err(RpcFault::invalid_params(
                "this connection has been displaced by a newer client".to_string(),
            ));
        }

        let params: ThreadStartParams = serde_json::from_value(params)
            .map_err(|e| RpcFault::invalid_params(format!("thread/start: {e}")))?;

        let thread_id = format!("thread_{}", self.next_thread.fetch_add(1, Ordering::SeqCst));

        // An empty `cwd` is a client that has one and did not fill it in, not a
        // request to root the workspace at "". Left as given it becomes a
        // working directory no process can enter, and every tool in the thread
        // fails with ENOENT and no hint of why.
        let claimed = params
            .cwd
            .as_deref()
            .map(str::trim)
            .filter(|dir| !dir.is_empty());
        let working_dir = match claimed {
            Some(dir) => PathBuf::from(dir),
            None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        };

        // Refused here rather than discovered one tool call at a time — but only
        // when this process's own tools are the ones that will use it. With the
        // workspace tools off, the client's `cwd` is a path in the *client's*
        // filesystem and is not expected to exist here at all: that is the whole
        // arrangement, a Mac's `/Users/...` named to a Linux GPU box. Validating
        // it then would refuse precisely the configuration it was meant to help.
        if self.config.workspace_tools && !working_dir.is_dir() {
            return Err(RpcFault::invalid_params(format!(
                "thread/start: cwd '{}' is not a directory on the machine \
                 running gallium",
                working_dir.display()
            )));
        }

        // Which directory a thread's tools see, and — the part worth saying out
        // loud — whether the client chose it. A client that sends no `cwd` gets
        // the directory gallium itself was started in, which is right for a
        // client that spawned gallium as a child and wrong for every other
        // arrangement; silently, in both cases, until someone runs `pwd`.
        match claimed {
            Some(_) if !self.config.workspace_tools => tracing::info!(
                "thread {}: workspace {} — the client's own path, where its \
                 tools run; nothing here reads it",
                thread_id,
                working_dir.display()
            ),
            Some(_) => tracing::info!(
                "thread {}: workspace {} (from the client's cwd)",
                thread_id,
                working_dir.display()
            ),
            None => tracing::info!(
                "thread {}: workspace {} — the client sent no cwd, so this is \
                 gallium's own working directory",
                thread_id,
                working_dir.display()
            ),
        }

        let model = params
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let provider = self.providers.provider_for(&self.config, &model)?;
        // `openai` / `local` / `candle` — where this thread's model actually
        // runs, which is the only sense in which gallium has a "provider". The
        // trace records the same label for the same reason.
        let model_provider = TraceMeta::engine_label(
            self.config.model_path.as_deref(),
            self.config.inference_engine.clone(),
        );

        // Three cells the thread shares with everything built here that outlives
        // no single turn — the approval sink and the client's own tools — because
        // all three name something that does not exist yet at `thread/start`.
        // `run_turn` fills the first two in; the observer fills the third.
        //
        // The turn currently running, read for every `turnId` a callback carries.
        let current_turn = Arc::new(Mutex::new(String::new()));
        // Its stop switch, so a `cancel` decision can interrupt the turn and not
        // merely refuse the one action.
        let current_cancel = Arc::new(Mutex::new(None));

        // The item the running tool call belongs to, so an approval can name it.
        // Same shape and same reason as `current_cancel`: the sink is built here
        // and the item does not exist yet.
        let current_item = Arc::new(Mutex::new(None));
        // The streamed `agentMessage`'s id, shared so both the observer's deltas
        // and the ending's `item/completed` use it. See `Thread::streaming_item`.
        let streaming_item = Arc::new(Mutex::new(None));

        // Mutations are approved by the client, not by a terminal prompt — except
        // under `approvalPolicy: "never"`, where the client has said it does not
        // want to be asked. An absent policy is treated as "ask": failing toward
        // a question is safer than silently granting write access.
        //
        // `approval_policy` is the *resolved* answer in codex's spelling, which
        // is what `thread/start` reports back. `never` is this branch; every
        // other input — including none — lands on the broker that asks, which is
        // codex's `untrusted`: nothing mutating proceeds unasked.
        let (approver, approval_policy): (Arc<dyn ApprovalSink>, &str) =
            match params.approval_policy.as_deref() {
                Some("never") => (Arc::new(AutoApproveSink), "never"),
                _ => (
                    Arc::new(RemoteApprovalSink::new(
                        Arc::clone(conn),
                        thread_id.clone(),
                        Arc::clone(&current_cancel),
                        Arc::clone(&current_turn),
                        Arc::clone(&current_item),
                    )),
                    "untrusted",
                ),
            };
        // `CAUTIOUS`, not the default policy: under a driving client every
        // mutation is the client's question to answer, including the workspace
        // writes the REPL's own policy grants. A tier the policy allowed would
        // never reach the client at all, which would silently stop the
        // `item/fileChange/requestApproval` round trip its UI is built around.
        let broker = Arc::new(ApprovalBroker::with_sink(
            ApprovalPolicy::CAUTIOUS,
            approver,
        ));
        // Recorded against the thread's own workspace and policy, so a trace
        // says what this thread was actually running under rather than what the
        // process was launched with.
        let trace = TraceSession::from_env(
            self.config.trace_dir.clone(),
            TraceMeta::new(
                TraceMeta::engine_label(
                    self.config.model_path.as_deref(),
                    self.config.inference_engine.clone(),
                ),
                model.clone(),
                working_dir.display().to_string(),
                ApprovalPolicy::CAUTIOUS,
            ),
            Some(Arc::clone(&broker)),
        );
        let session = Arc::new(ToolSession::with_broker(working_dir.clone(), broker));

        // Load the same skills the REPL does: the working dir's own, the
        // user-global ones, and anything the launch config listed. Then the
        // client's own `skillPaths`, last so they win a name collision — the
        // client knows what this thread is for, and the process was launched
        // by someone else.
        let skills = Arc::new(SkillRegistry::new());
        // A path is only ours to read when the workspace is. `workspace_tools`
        // false means a client on a socket, whose `cwd` and `skillPaths` name
        // its *own* filesystem — dereferencing them here would be this process
        // reading files the client chose, with this user's privileges, and
        // returning their contents through the prompt and `LookupSkill`. That is
        // the local-tool primitive the transport just took away, arriving by
        // another door. Only what the operator configured is loaded.
        if self.config.workspace_tools {
            crate::skill::load_skills(&skills, &working_dir);
        } else {
            crate::skill::load_global_skills(&skills);
        }
        for dir in &self.config.skill_paths {
            skills.load_from_dir(dir);
        }
        let mut from_client = 0;
        if !self.config.workspace_tools && !params.skill_paths.is_empty() {
            // Never silent, for the same reason an empty load is not: a client
            // whose skills never arrive sees only a model that behaves as though
            // it has none.
            tracing::warn!(
                "thread {}: ignoring {} skillPaths from the client — a client on \
                 a socket names paths in its own filesystem, and reading them \
                 here would be this host reading files it did not choose",
                thread_id,
                params.skill_paths.len()
            );
        } else {
            for path in &params.skill_paths {
                let path = working_dir.join(path); // absolute paths pass through
                let loaded = skills.load_from_path(&path);
                if loaded == 0 {
                    // Never silent: a client that names a path it thinks holds
                    // skills and gets nothing has no other way to find that out,
                    // and the symptom downstream is a model concluding it has no
                    // skills at all.
                    tracing::warn!("thread/start skillPaths: no skills found in {:?}", path);
                }
                from_client += loaded;
            }
        }
        let skill_count = skills.count();
        // Whether this process offers tools that act on *its own* machine. Off is
        // the arrangement the TCP transport exists for: gallium runs where the
        // GPU is, and everything that reads, writes, or executes belongs to the
        // machine the user is sitting at, arriving as the client's `dynamicTools`.
        let mut registry = if self.config.workspace_tools {
            create_default_registry_with_session(working_dir.clone(), Arc::clone(&skills), session)
        } else {
            create_registry_without_workspace_tools(Arc::clone(&skills))
        };

        // External MCP servers the client asked us to reach — but only when its
        // machine is ours. A stdio MCP server *is a command line*:
        // `register_mcp_servers` spawns it here, as this user, so honoring one
        // named over a socket is arbitrary code execution handed to whoever
        // reached the port. The same argument as the skill paths above, one
        // rung worse — that door reads files, this one runs programs.
        //
        // An MCP server belongs to the machine whose files and processes it is
        // for. A client that wants one runs it beside itself and exposes its
        // tools as `dynamicTools`, which come back over this connection and
        // execute under whoever is running the client.
        let mcp_servers = params.mcp_servers();
        if self.config.workspace_tools {
            crate::register_mcp_servers(&mut registry, &mcp_servers);
        } else if !mcp_servers.is_empty() {
            tracing::warn!(
                "thread {}: ignoring {} MCP server(s) named by the client — one \
                 would run on this host, as this user. Run them beside the \
                 client and send their tools as dynamicTools.",
                thread_id,
                mcp_servers.len()
            );
        }

        // The client's own tools, dispatched back over this connection. They read
        // the live turn id out of `current_turn`, the same cell the approval sink
        // names its `turnId` from.
        let dynamic_tools = params.dynamic_tools.clone();

        // Whether this thread will install a discovery tool, decided before any
        // client tool is registered because it decides whether `ToolSearch` is a
        // name the client may still use. Nothing is deferred in the common case,
        // gallium claims no name, and the client keeps every name it sent.
        let installs_tool_search = dynamic_tools.iter().any(|spec| !spec.advertised);

        for spec in &dynamic_tools {
            // The one name a deferring thread keeps for itself. `resolve`
            // returns the first exact match and `register_replacing` drops what
            // it displaces, so registering the client's tool here would either
            // lose it to the discovery tool below or — if the client deferred
            // its own `ToolSearch` — leave the *mask* hiding the name gallium is
            // about to register under, making discovery itself invisible and
            // every deferred tool unreachable.
            //
            // Refused rather than renamed: a tool the model can call under a
            // name the client never chose is worse than one it cannot call,
            // since the client's result handler routes on the name it sent.
            // Logged because a client cannot be told — `thread/start`'s response
            // is codex's shape and has nowhere truthful to put this — so the log
            // is the only place the collision is visible.
            if installs_tool_search && ToolSearchTool::claims_name(&spec.name) {
                tracing::warn!(
                    "thread {}: ignoring the client's '{}' — this thread defers \
                     tools, so that name belongs to gallium's own discovery \
                     tool. Rename it to offer it.",
                    thread_id,
                    spec.name
                );
                continue;
            }

            // Replacing, not adding: a client that names `Bash` means *its*
            // Bash, and behind the built-in of that name it would never be
            // called. See `ToolRegistry::register_replacing`.
            registry.register_replacing(Box::new(RemoteTool::new(
                Arc::clone(conn),
                spec.clone(),
                thread_id.clone(),
                Arc::clone(&current_turn),
            )));
            // Registered either way — deferral decides what the model is *told*
            // about, never what it may reach. Set explicitly in both directions
            // so a client re-registering a name it deferred earlier gets the
            // advertisement it asked for this time.
            if spec.advertised {
                registry.visibility().reveal(&spec.name);
            } else {
                registry.visibility().hide(&spec.name, &spec.description);
            }
        }

        // The way back to whatever was deferred. Registered only when something
        // is actually hidden: a thread with nothing to find would be paying a
        // schema to advertise a search over an empty set, which is the cost this
        // whole mechanism exists to avoid.
        let deferred = registry.visibility().hidden_count();
        if deferred > 0 {
            tracing::info!(
                "thread {}: {} of {} client tool(s) registered but not \
                 advertised; offering ToolSearch to reach them",
                thread_id,
                deferred,
                dynamic_tools.len()
            );
            let visibility = Arc::clone(registry.visibility());
            registry.register_replacing(Box::new(ToolSearchTool::new(visibility)));
        }

        // Said out loud: with the workspace tools off and no client tools, the
        // model can read nothing, write nothing, and run nothing — a
        // configuration that looks like a broken model rather than a missing
        // half of the arrangement.
        if !self.config.workspace_tools && dynamic_tools.is_empty() {
            tracing::warn!(
                "thread {}: this server lends no tools of its own and the client \
                 registered no dynamicTools, so the thread can read nothing, \
                 write nothing and run nothing. A client on a socket must send \
                 its own tools on thread/start.",
                thread_id
            );
        }

        let mut messages = Vec::new();
        // The profile's own preamble first — gallium's protocol ABI for this
        // model family (see `ModelProfile::agent_preamble`) — same reasoning
        // and same ordering as the REPL's: it is what the model needs to use
        // gallium at all, ahead of whatever this client's own instructions ask
        // the persona/task layer to do.
        if let Some(preamble) = provider.agent_preamble() {
            messages.push(ChatMessage::system(preamble.into_owned()));
        }
        if let Some(instructions) = params.developer_instructions.filter(|s| !s.is_empty()) {
            messages.push(ChatMessage::system(instructions));
        }

        // Settled per thread rather than per process: threads can run different
        // models, and the window is the model's property.
        let window = memory::resolve_context_window(
            self.config.context_window,
            provider.context_window(),
            self.fallback_context_window(),
        );

        let thread = Arc::new(Thread {
            provider,
            registry,
            skills,
            messages: Mutex::new(messages),
            max_iterations: self.config.max_iterations,
            current_turn,
            current_cancel,
            current_item,
            streaming_item,
            active_turn: Mutex::new(None),
            context_window: window.effective,
            known_context_window: window.known,
            last_input_tokens: AtomicU64::new(0),
            total_usage: Mutex::new(TokenUsage::default()),
            trace,
        });
        self.threads.lock().insert(thread_id.clone(), thread);

        tracing::info!(
            "thread {} started ({} dynamic tools, {} skills, {} of them from skillPaths)",
            thread_id,
            dynamic_tools.len(),
            skill_count,
            from_client,
        );
        // Codex's `ThreadStartResponse` shape, and nothing beside it.
        //
        // A flat `threadId` and a `skillCount` used to ride along here. Both
        // were additive and harmless to a client that ignored them, which is
        // the argument that kept them — but the id is at `thread.id` in codex's
        // response and nowhere else, and klein stopped reading the flat one
        // ("Read thread and turn ids only where codex puts them"), so all the
        // second spelling did was let a client work against gallium in a way
        // that would not work against codex. That is the failure this whole
        // surface exists to avoid, and it is worse than the divergence it
        // papered over because it only shows up on the switch.
        //
        // `skillCount` answered a real question — did the client's `skillPaths`
        // land — but no protocol has a field for it, so a client that wants the
        // answer is reading gallium, not the protocol. The zero case, the one
        // worth knowing about, is already logged as a warning above.
        let now = now_secs();
        Ok(json!({
            "thread": {
                "id": thread_id,
                // Gallium has no session tree and no forking, so a thread is its
                // own session. Saying so is truer than inventing a parent.
                "sessionId": thread_id,
                // The first user message, which does not exist yet — a thread
                // starts before its first turn.
                "preview": "",
                // Nothing is written to disk: gallium's threads live in this
                // process's memory and end with it. `ephemeral` is exactly that
                // claim, which is also why `path` is left absent.
                "ephemeral": true,
                "modelProvider": model_provider,
                "createdAt": now,
                "updatedAt": now,
                // Tagged enum: `{"type": "idle"}`. Idle is accurate — the thread
                // exists and no turn is running.
                "status": { "type": "idle" },
                "cwd": working_dir.to_string_lossy(),
                "cliVersion": env!("CARGO_PKG_VERSION"),
                // This *is* an app-server, which is one of codex's own variants.
                //
                // `appServer`, not `mcp`. There are two `SessionSource` enums in
                // codex and only one of them is on the wire: the *core* enum
                // (`protocol/src/protocol.rs`) is lowercase and spells this
                // `Mcp`, but `Thread.source` is typed with the app-server's own
                // (`app-server-protocol/.../v2/thread_data.rs`), which is
                // camelCase and spells it `AppServer` — its
                // `From<CoreSessionSource>` maps `Mcp => AppServer` precisely
                // here. Sending `mcp` would hit that enum's `#[serde(other)]`
                // and land on `Unknown`, which is the bug this looks like it
                // would fix.
                "source": "appServer",
                // Only ever populated by the history methods gallium does not
                // implement; codex sends an empty list from `thread/start` too.
                "turns": [],
            },
            "model": model,
            "modelProvider": model_provider,
            "cwd": working_dir.to_string_lossy(),
            // What the thread will actually do, not what was asked for: `never`
            // is the branch that installed `AutoApproveSink`, and every other
            // input lands on the broker that asks. Reporting the request rather
            // than the resolution would tell a client its absent `approvalPolicy`
            // meant nothing.
            "approvalPolicy": approval_policy,
            // Approvals go to whoever is driving this connection. Gallium has no
            // reviewing subagent, so this is always the human.
            "approvalsReviewer": "user",
            // Tagged, kebab-case. Gallium runs no sandbox — the approval tiers
            // are the containment, and claiming a sandbox it does not have is
            // the one answer here that could get someone hurt.
            "sandbox": { "type": "danger-full-access" },
        }))
    }

    /// The window to compact against when neither the config nor the model says.
    ///
    /// Split by where the model runs, because the two are orders of magnitude
    /// apart: assuming a cloud window for a local model means compaction never
    /// fires before llama.cpp is out of room.
    fn fallback_context_window(&self) -> u32 {
        if self.config.model_path.is_some() {
            crate::llm::LOCAL_CONTEXT_WINDOW
        } else {
            memory::DEFAULT_CONTEXT_WINDOW
        }
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

        let prompt = params.prompt();

        // Codex rejects empty input on both `turn/start` and `turn/steer`;
        // `turn/steer` already refuses one with no text to steer with (below).
        // `UserInput::is_empty()`, not `text.is_empty()`: an image with no
        // caption is still a turn worth starting.
        if prompt.is_empty() {
            return Err(RpcFault::invalid_params(
                "turn/start: input has no text or attachments".to_string(),
            ));
        }

        let turn_id = format!("turn_{}", self.next_turn.fetch_add(1, Ordering::SeqCst));

        // Said out loud rather than swallowed: a client whose image never
        // reached the model would otherwise read the reply as the model failing
        // to see, and go looking in the wrong place.
        match params.unreadable_images() {
            0 => {}
            n => tracing::warn!(
                "thread {} turn {}: dropped {} image item(s) — only base64 \
                 `data:image/…` URLs are carried",
                params.thread_id,
                turn_id,
                n
            ),
        }

        // Claim the thread's one turn slot before answering. Refusing here — with
        // the running turn named — is the only honest answer: the second turn
        // would otherwise sit on `messages` until the first finished, and the
        // client would have no idea its request had not started.
        // The worker holds the sending half for the turn's lifetime and never
        // sends: dropping it is what tells `turn/interrupt` the turn is over.
        let (finished_tx, finished) = crossbeam::channel::bounded::<Never>(0);
        let cancel = CancellationToken::new();
        let steer = SteerInbox::new();
        {
            // Held across the claim, not merely checked before it: this is the
            // same lock `cancel_turns` closes and snapshots under, and only
            // holding it here makes "cancel everything running" mean everything.
            let accepting = self.accepting_turns.lock();
            if !*accepting {
                return Err(RpcFault::invalid_params(
                    "this connection has been displaced by a newer client".to_string(),
                ));
            }
            let mut active = thread.active_turn.lock();
            if let Some(running) = active.as_ref() {
                return Err(RpcFault::invalid_params(format!(
                    "thread '{}' is already running turn '{}'; \
                     one turn at a time",
                    params.thread_id, running.id
                )));
            }
            *active = Some(ActiveTurn {
                id: turn_id.clone(),
                cancel: cancel.clone(),
                steer: steer.clone(),
                finished,
            });
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
            predecessor: Arc::clone(&self.predecessor),
            cancel,
            steer,
            _finished: finished_tx,
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

        // `items` is empty and `full` here, and that is accurate: the turn has
        // not produced anything yet.
        Ok(json!({ "turn": turn_object(&turn_id, "inProgress", true) }))
    }

    /// Hand more user text to the turn that is already running.
    ///
    /// Codex's shape (`turn/steer`, `{threadId, expectedTurnId, input, …}` →
    /// `{turnId}`): the turn id does not change, the text is injected as a user
    /// message, and the turn goes on to end as an ordinary `turn/completed`.
    /// This is not interrupt-and-restart — nothing is rolled back and no second
    /// turn is created.
    ///
    /// `expectedTurnId` is a precondition, not a hint. A client that steers
    /// after the turn it meant has already ended would otherwise put its text
    /// into a *later* turn, which is worse than being told no: the message
    /// arrives, out of context, attached to work the user has moved on from.
    ///
    /// Delivery is at the next ReAct boundary — after the current generation
    /// and the tool calls it asked for. On a local model that can be tens of
    /// seconds, which is the honest cost of not being able to interrupt a model
    /// mid-sentence.
    fn handle_turn_steer(&self, conn: &Arc<Connection>, params: Value) -> HandlerResult {
        let params: TurnSteerParams = serde_json::from_value(params)
            .map_err(|e| RpcFault::invalid_params(format!("turn/steer: {e}")))?;

        // Text-only, and never empty. An input carrying nothing we can render
        // into a message would be accepted and then do nothing at all, which
        // reads to the client as a steer that was silently ignored.
        let steering = prompt_input(&params.input);
        if steering.text.trim().is_empty() {
            return Err(RpcFault::invalid_params(
                "turn/steer: input has no text to steer with".to_string(),
            ));
        }
        // A steer rides `SteerInbox`, which carries a `String`: the ReAct loop
        // drains it into a user message mid-turn, and there is no image on that
        // path. Refused rather than dropped, for the same reason `turn/start`
        // logs — an attachment nobody looked at must not read as a model that
        // could not see it. `turn/start` is where an image belongs today.
        if !steering.media.is_empty() {
            return Err(RpcFault::invalid_params(
                "turn/steer: images are not carried by a steer; \
                 attach them to turn/start"
                    .to_string(),
            ));
        }
        let text = steering.text;

        let thread = self
            .threads
            .lock()
            .get(&params.thread_id)
            .cloned()
            .ok_or_else(|| {
                RpcFault::invalid_params(format!("unknown thread '{}'", params.thread_id))
            })?;

        // Checked and pushed under the one lock, so the turn cannot be replaced
        // between deciding it is the right one and speaking to it.
        //
        // The slot alone is not enough: a turn that has left the ReAct loop but
        // has not yet cleared the slot is still named here, and holding the
        // slot's lock across the whole loop is what `turn/interrupt` already
        // cannot do. So the inbox itself is the authority on whether anyone is
        // still reading — `push` refuses once the loop has stopped, and that
        // refusal is what the client is told, rather than an acknowledgement for
        // text that would go nowhere.
        {
            let active = thread.active_turn.lock();
            match active.as_ref() {
                None => {
                    return Err(RpcFault::invalid_params(
                        "no active turn to steer".to_string(),
                    ))
                }
                Some(running) if running.id != params.expected_turn_id => {
                    return Err(RpcFault::invalid_params(format!(
                        "expected active turn id {} but found {}",
                        params.expected_turn_id, running.id
                    )))
                }
                Some(running) => {
                    if !running.steer.push(text) {
                        return Err(RpcFault::invalid_params(format!(
                            "turn {} has finished and can no longer be steered",
                            params.expected_turn_id
                        )));
                    }
                }
            }
        }

        tracing::info!(
            "thread {} turn {}: steered",
            params.thread_id,
            params.expected_turn_id
        );

        // Echoed back as an item so the turn's stream holds everything the model
        // was given, in the order it was given. Note the asymmetry with
        // `turn/start`, whose prompt gallium does not echo: the client already
        // has that one in the reply it is holding, whereas a steer accepted into
        // a running turn is otherwise invisible to any other view of the thread.
        //
        // Started *and* completed, both, which is what codex emits for a user
        // message (`Session::record_user_prompt_and_emit_turn_item`) and what
        // `../klein-cli` records as verified against it. A message has no work
        // to do and is complete the moment it exists, so a client tracking item
        // lifecycle would otherwise hold this one open for the rest of the turn.
        // `clientId` omitted rather than sent as `null` when absent: codex's
        // `UserMessageItem` has `#[serde(skip_serializing_if = "Option::is_none")]`
        // on this field, and a TypeScript client generated from that schema with
        // `exactOptionalPropertyTypes` rejects `{"clientId": null}`.
        let mut item = json!({
            "type": "userMessage",
            "id": format!("msg_{}", self.next_item.fetch_add(1, Ordering::SeqCst)),
            "content": params.input,
        });
        if let Some(client_id) = &params.client_user_message_id {
            item["clientId"] = json!(client_id);
        }
        let notification = json!({
            "threadId": params.thread_id,
            "turnId": params.expected_turn_id,
            "item": item,
        });
        conn.notify("item/started", notification.clone());
        conn.notify("item/completed", notification);

        Ok(json!({ "turnId": params.expected_turn_id }))
    }

    /// Stop the turn named in `params`, and answer once it has actually stopped.
    ///
    /// Codex's shape exactly (`turn/interrupt`, `{threadId, turnId}` → `{}`),
    /// including the part that is easy to miss: it parks the request and replies
    /// when the turn aborts, so a successful response means *the turn has
    /// stopped*, not *we heard you*. Gallium can say the same by blocking —
    /// every request already runs on its own thread — so this waits on the
    /// worker's `finished` channel rather than answering optimistically.
    ///
    /// Stopping is prompt, not instant: a turn inside an OpenAI round trip
    /// finishes that call first, and an MCP request is abandoned rather than
    /// interrupted (see `cancel.rs`). The wait is therefore bounded by the
    /// slowest thing the turn is currently inside, which is the honest answer.
    fn handle_turn_interrupt(&self, params: Value) -> HandlerResult {
        let params: TurnInterruptParams = serde_json::from_value(params)
            .map_err(|e| RpcFault::invalid_params(format!("turn/interrupt: {e}")))?;

        // Codex reads an empty `turnId` as "cancel startup". Gallium has no
        // startup phase to cancel — a thread is ready the moment `thread/start`
        // returns — so say that rather than report a turn-id mismatch against
        // the empty string, which is what the check below would otherwise do.
        if params.turn_id.is_empty() {
            return Err(RpcFault::invalid_params(
                "turn/interrupt requires a turnId; gallium has no startup phase \
                 to cancel"
                    .to_string(),
            ));
        }

        let thread = self
            .threads
            .lock()
            .get(&params.thread_id)
            .cloned()
            .ok_or_else(|| {
                RpcFault::invalid_params(format!("unknown thread '{}'", params.thread_id))
            })?;

        // Cancel under the lock, then wait outside it: the worker takes the same
        // lock to clear the slot, so holding it here would deadlock against the
        // very turn we are waiting for.
        let finished = {
            let active = thread.active_turn.lock();
            match active.as_ref() {
                None => {
                    return Err(RpcFault::invalid_params(
                        "no active turn to interrupt".to_string(),
                    ))
                }
                Some(running) if running.id != params.turn_id => {
                    return Err(RpcFault::invalid_params(format!(
                        "expected active turn id {} but found {}",
                        params.turn_id, running.id
                    )))
                }
                Some(running) => {
                    running.cancel.cancel();
                    running.finished.clone()
                }
            }
        };

        tracing::info!(
            "thread {} turn {}: interrupt requested",
            params.thread_id,
            params.turn_id
        );
        // Err is the sending half being dropped, which is the worker returning.
        // There is nothing to receive: the channel carries no values.
        let _ = finished.recv();

        Ok(json!({}))
    }
}

/// One background turn: everything it needs to run and to report, owned rather
/// than borrowed, because it outlives the `turn/start` that started it.
struct TurnWorker {
    conn: Arc<Connection>,
    thread: Arc<Thread>,
    thread_id: String,
    turn_id: String,
    /// The displaced connection's turns, which this one must not overlap.
    predecessor: Arc<Predecessor>,
    /// The other end of what `turn/interrupt` sets.
    cancel: CancellationToken,
    /// The other end of what `turn/steer` writes to.
    steer: SteerInbox,
    /// Held, never sent on. Dropping it when `run` returns is what releases a
    /// `turn/interrupt` waiting for this turn to stop.
    _finished: Sender<Never>,
}

impl TurnWorker {
    /// Run the turn and report how it ended, then release the thread's turn slot.
    ///
    /// Every exit reports something. `turn/start` has already been answered, so
    /// a turn that ended without a notification would leave the client waiting
    /// for one forever — this is the only place that can tell it.
    fn run(self, prompt: UserInput) {
        // Before anything reaches the model: the connection this one replaced
        // may still have a turn inside a call that cannot be interrupted, and
        // two turns sharing the provider's KV slots is the overlap the
        // one-client rule exists to prevent. `turn/start` has already been
        // answered, so waiting here costs the client nothing it can see.
        let result = if self.predecessor.settle(PREDECESSOR_GRACE) {
            run_turn(
                &self.conn,
                &self.thread,
                &self.thread_id,
                &self.turn_id,
                &self.cancel,
                &self.steer,
                prompt,
            )
        } else {
            // Refused rather than run anyway. Proceeding would put this turn on
            // the same slots as one that is still going — quietly halving the
            // cache the transport exists to keep warm — and the client can act
            // on being told to try again, which it cannot do about a turn that
            // silently ran slowly.
            tracing::warn!(
                "thread {} turn {}: the displaced connection's turn has not \
                 stopped after {}s; refusing rather than overlapping it",
                self.thread_id,
                self.turn_id,
                PREDECESSOR_GRACE.as_secs()
            );
            Err(AgentError::InternalError(format!(
                "the connection this one replaced is still finishing a turn \
                 after {}s; try again",
                PREDECESSOR_GRACE.as_secs()
            )))
        };

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
        // The stop switch goes with the slot: a `cancel` decision arriving after
        // this has no turn to stop, and firing this turn's token would be worse
        // than doing nothing — the next turn clones a fresh one, but a stale
        // `Some` here is a token nobody is watching.
        //
        // Inside the slot's critical section, and that is load-bearing: it is
        // what stops this clear from landing on the *next* turn's token. A turn
        // cannot claim the slot until `active` drops below, so the order is
        // total — this clear, then the claim, then that turn's spawn, then its
        // own publish in `run_turn`. Hoisting this out of the section reads like
        // a tidy-up and opens exactly the window it looks like it closes.
        *self.thread.current_cancel.lock() = None;
        // The item cell goes with it, for the same reason and in the same
        // place. `NotifyingObserver` clears this on `ToolCompleted`, which the
        // ReAct loop does not emit when a tool call is cancelled — it records
        // the call and returns — so a turn stopped mid-tool leaves a `Some`
        // naming an item that is over.
        //
        // Nothing reads it in that state today: an approval is only raised from
        // inside a tool call, and every such call publishes its own id first.
        // It is cleared because the two cells are one mechanism, and an
        // invariant that holds for one of them by accident is one nobody can
        // rely on for the next thing that reads it.
        *self.thread.current_item.lock() = None;

        // The final answer's item id: whatever the observer streamed this
        // answer's deltas under, or a fresh well-known one if it did not stream.
        let final_id = self
            .thread
            .streaming_item
            .lock()
            .take()
            .unwrap_or_else(|| final_message_item_id(&self.turn_id));

        match result {
            Ok(text) => {
                self.conn.notify(
                    "item/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turnId": self.turn_id,
                        "item": {
                            "type": "agentMessage",
                            "id": final_id,
                            "text": text,
                        },
                    }),
                );
                self.conn.notify(
                    "turn/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turn": turn_object(&self.turn_id, "completed", false),
                    }),
                );
            }
            // A turn that was asked to stop ended the way it was told to, so it
            // ends as `turn/completed` with codex's `interrupted` status rather
            // than as a failure. That is both the codex shape and the one a
            // client reads correctly: klein maps `turn/completed` to "done" and
            // returns what the turn had produced, which is right for an
            // interrupt and would be wrong for an error.
            //
            // `run_turn` has already rolled history back to before the prompt,
            // so the thread is left exactly as it was.
            Err(AgentError::Cancelled) => {
                tracing::info!(
                    "thread {} turn {}: stopped on request",
                    self.thread_id,
                    self.turn_id
                );
                self.conn.notify(
                    "turn/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turn": turn_object(&self.turn_id, "interrupted", false),
                    }),
                );
            }
            // Every ending is a `turn/completed`; the `status` says which one.
            // That is codex's whole vocabulary here — it has no `turn/failed` —
            // and gallium's extra method meant a codex-native client watched a
            // failed turn simply never end.
            //
            // The reason this waited for its own change is that the same fact
            // cuts the other way for a client keying off the *method*: klein's
            // `classifyNote` treated any `turn/completed` as success, so making
            // this switch alone would have converted every failure into a silent
            // one. fpt/klein-cli#95 reads the status and landed first.
            Err(e) => {
                tracing::warn!(
                    "thread {} turn {} failed: {}",
                    self.thread_id,
                    self.turn_id,
                    e
                );
                self.conn.notify(
                    "turn/completed",
                    json!({
                        "threadId": self.thread_id,
                        "turn": failed_turn(&self.turn_id, &e.to_string()),
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
    cancel: &CancellationToken,
    steer: &SteerInbox,
    prompt: UserInput,
) -> Result<String, AgentError> {
    // Publish the turn id before any tool can fire a callback for it, and the
    // stop switch before any of them can be approved — an approval is the first
    // thing that can arrive with a `cancel` on it.
    *thread.current_turn.lock() = turn_id.to_string();
    *thread.current_cancel.lock() = Some(cancel.clone());

    let mut messages = thread.messages.lock();

    let ctx = TurnContext::new(cancel.clone()).with_steering(steer.clone());
    let observer = NotifyingObserver::new(
        conn,
        thread_id,
        turn_id,
        &thread.registry,
        &thread.total_usage,
        thread.known_context_window,
        &thread.current_item,
        &thread.streaming_item,
    );
    let setup = TurnSetup {
        provider: thread.provider.as_ref(),
        tools: &thread.registry,
        skills: Some(&thread.skills),
        max_iterations: thread.max_iterations,
        context_window: thread.context_window,
        observer: Some(&observer),
        // The token `turn/interrupt` sets. It reaches token generation in
        // both local backends, the `bash` child's process group, and every
        // ReAct loop boundary.
        context: Some(&ctx),
        trace: thread.trace.as_ref(),
        // The id the client knows this turn by, so a trace can be matched up
        // with the notifications the client saw.
        turn_id: Some(turn_id),
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
            "turn/steer" => self.handle_turn_steer(conn, params),
            "turn/interrupt" => self.handle_turn_interrupt(params),
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
    /// Extra skill locations for this thread — the client's own, which are not
    /// in any of the standard directories `skill::load_skills` searches.
    ///
    /// A driving client keeps its skills wherever its own repo puts them, and
    /// before this field the only way to get them in front of the model was to
    /// paste them into `developerInstructions`; `LookupSkill` was advertised
    /// and always answered empty, which reads to a model like "no skills
    /// exist". Codex spells this `skills/extraRoots/set`; gallium takes it at
    /// thread start because a thread's skills do not change under it.
    ///
    /// Each entry is a directory of skills or a single `SKILL.md`, relative to
    /// the thread's `cwd` or absolute.
    #[serde(default)]
    skill_paths: Vec<String>,
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

/// Codex's `TurnSteerParams`. `expectedTurnId` is required there and here: it
/// is the precondition that keeps a late steer from landing in the wrong turn.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnSteerParams {
    thread_id: String,
    expected_turn_id: String,
    #[serde(default)]
    input: Vec<Value>,
    /// The client's own id for this message, echoed back on the item so it can
    /// match what it sent against what the thread accepted.
    #[serde(default)]
    client_user_message_id: Option<String>,
}

/// Codex's `TurnInterruptParams` (`app-server-protocol/src/protocol/v2/turn.rs`).
/// Both fields are required there, and naming the turn is what lets the server
/// refuse an interrupt aimed at one that is no longer running.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnInterruptParams {
    thread_id: String,
    turn_id: String,
}

impl TurnStartParams {
    fn prompt(&self) -> UserInput {
        prompt_input(&self.input)
    }

    /// `image` items the client sent that could not be read.
    fn unreadable_images(&self) -> usize {
        declared_images(&self.input).saturating_sub(self.prompt().media.len())
    }
}

/// Read a turn input into the text and attachments it carries.
///
/// Text items are concatenated; `image` items carrying a base64 `imageUrl` data
/// URL become attachments. An `image` item we cannot read — a remote URL, a
/// media type that is not an image — is **dropped**, deliberately: the reader is
/// shared with `turn/steer`, which cannot carry images at all, so refusing here
/// would put the rejection in the wrong place. `turn/start` is where an
/// unreadable image is worth saying something about, and it logs it.
///
/// Shared by `turn/start` and `turn/steer`: the two carry the same `input`
/// shape, and a steer that read it differently would be a second, quieter set
/// of rules for what a client may say. What differs is what each can *do* with
/// the result, not how it is parsed.
fn prompt_input(input: &[Value]) -> UserInput {
    let text = input
        .iter()
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");

    let media = input
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .filter_map(|item| item.get("imageUrl").and_then(Value::as_str))
        .filter_map(input::image_from_data_url)
        .map(MediaContent::Image)
        .collect();

    // No audio here yet: a client sends media as `imageUrl`, and there is no
    // agreed `audioUrl` item in the protocol codex defines. The REPL's
    // `@audio:` is the only way in today.
    UserInput { text, media }
}

/// How many `image` items an input declared, readable or not — so `turn/start`
/// can tell "the client sent none" from "the client sent one we dropped".
fn declared_images(input: &[Value]) -> usize {
    input
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .count()
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

    /// The two variants carry output in different fields, and codex defines no
    /// `result` string on either — which is what gallium used to send to both.
    ///
    /// Unit tests because the `mcpToolCall` arm is otherwise unreachable from a
    /// test: it needs a live MCP server attached, so no end-to-end run exercises
    /// the one shape here with a nested object in it.
    #[test]
    fn an_mcp_call_reports_its_output_as_mcp_content_blocks() {
        let item = json!({ "type": "mcpToolCall" });
        let out = tool_output(&item, &ToolResult::text("the answer".to_string()));

        assert_eq!(out["result"]["content"][0]["type"], "text");
        assert_eq!(out["result"]["content"][0]["text"], "the answer");
        // Not `result: "the answer"`, which is what a client deserializing into
        // `McpToolCallResult` rejected outright.
        assert!(out["result"].is_object());
        assert!(out.get("contentItems").is_none());
    }

    #[test]
    fn a_named_tool_call_reports_its_output_as_content_items() {
        let item = json!({ "type": "dynamicToolCall" });

        let ok = tool_output(&item, &ToolResult::text("done".to_string()));
        assert_eq!(ok["contentItems"][0]["type"], "inputText");
        assert_eq!(ok["contentItems"][0]["text"], "done");
        assert_eq!(ok["success"], true);
        // `result` is not a field of this variant at all.
        assert!(ok.get("result").is_none());

        // `success` is the tool's own outcome, which is separate from the
        // item's `status` and is what a client reads to colour the row.
        let bad = tool_output(&item, &ToolResult::error("nope".to_string()));
        assert_eq!(bad["success"], false);
        assert_eq!(bad["contentItems"][0]["text"], "nope");
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
        let prompt = params.prompt();
        assert_eq!(prompt.text, "hello\nworld");
        assert!(prompt.media.is_empty());
    }

    #[test]
    fn turn_prompt_carries_a_base64_image_item() {
        let params: TurnStartParams = serde_json::from_value(json!({
            "threadId": "t1",
            "input": [
                { "type": "image", "imageUrl": "data:image/png;base64,AAAA" },
                { "type": "text", "text": "what is this?" },
            ],
        }))
        .unwrap();
        let prompt = params.prompt();
        assert_eq!(prompt.text, "what is this?");
        assert_eq!(prompt.media.len(), 1);
        assert_eq!(prompt.images().next().unwrap().media_type, "image/png");
        assert_eq!(prompt.images().next().unwrap().base64, "AAAA");
        assert_eq!(params.unreadable_images(), 0);
    }

    /// An image item we cannot read is dropped — and counted, so `turn/start`
    /// can say so rather than letting it look like a model that did not see.
    #[test]
    fn turn_prompt_counts_an_unreadable_image_item() {
        let params: TurnStartParams = serde_json::from_value(json!({
            "threadId": "t1",
            "input": [
                { "type": "image", "imageUrl": "https://example.com/cat.png" },
                { "type": "text", "text": "hi" },
            ],
        }))
        .unwrap();
        let prompt = params.prompt();
        assert_eq!(prompt.text, "hi");
        assert!(prompt.media.is_empty());
        assert_eq!(params.unreadable_images(), 1);
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

    /// A turn on the replacing connection waits for the displaced one, and is
    /// told no rather than run anyway when it will not stop.
    ///
    /// The grace period is a minute in production, which is why this exercises
    /// `Predecessor` directly: the behavior worth pinning is what happens on
    /// each side of the deadline, not how long the deadline is.
    #[test]
    fn a_turn_waits_for_the_displaced_connections_turns() {
        let (still_running, finished) = crossbeam::channel::bounded::<Never>(0);
        let predecessor = Predecessor::default();
        predecessor.adopt(StoppingTurns(vec![finished]));

        // A turn that has not stopped: the replacement must not proceed to the
        // model, however long it has been asked to wait.
        assert!(
            !predecessor.settle(Duration::from_millis(50)),
            "a turn was let through while the displaced one was still running"
        );

        // The worker dropping its sender is the turn ending.
        drop(still_running);
        assert!(
            predecessor.settle(Duration::from_millis(50)),
            "the displaced turn has stopped; the replacement should proceed"
        );
    }

    /// Asking again is cheap and correct: once the displaced turns are gone the
    /// handle is dropped, so every later turn on the connection takes the fast
    /// path instead of re-examining an empty list.
    #[test]
    fn a_settled_predecessor_stays_settled() {
        let predecessor = Predecessor::default();
        assert!(
            predecessor.settle(Duration::from_millis(1)),
            "nothing to wait for"
        );

        let (running, finished) = crossbeam::channel::bounded::<Never>(0);
        predecessor.adopt(StoppingTurns(vec![finished]));
        assert!(!predecessor.settle(Duration::from_millis(20)));
        drop(running);
        assert!(predecessor.settle(Duration::from_millis(20)));
        assert!(
            predecessor.settle(Duration::from_millis(0)),
            "a settled predecessor should not be waited on a second time"
        );
    }

    /// Displacement is transitive: a connection displaced while still waiting on
    /// the one *it* displaced passes that unfinished wait along.
    ///
    /// Three reconnects in quick succession is a laptop roaming between
    /// networks, not a contrived case. Dropping the inheritance would let the
    /// third connection's turn reach the model beside the first connection's,
    /// which is the overlap the whole mechanism exists to prevent.
    #[test]
    fn an_unfinished_wait_is_handed_on_when_its_owner_is_displaced() {
        let server = AppServer::with_provider_factory(
            ServerConfig::default(),
            Box::new(|_cfg, _model| Err(AgentError::InternalError("unused".to_string()))),
        );

        // The first connection's turn, still running.
        let (first_turn, finished) = crossbeam::channel::bounded::<Never>(0);
        server.adopt_predecessor(StoppingTurns(vec![finished]));

        // The second connection is displaced in its turn, before that ever
        // settled. What it hands the third must still contain the first's turn.
        let inherited = Predecessor::default();
        inherited.adopt(server.cancel_turns());
        assert!(
            !inherited.settle(Duration::from_millis(20)),
            "the third connection was let through while the first's turn was still running"
        );

        drop(first_turn);
        assert!(inherited.settle(Duration::from_millis(20)));
    }

    /// One slow turn must not extend the wait for the ones beside it: the
    /// deadline is absolute, not per-turn, or displacing a connection running
    /// three turns would take three times as long to answer.
    #[test]
    fn the_wait_is_bounded_across_all_the_turns_it_covers() {
        let held: Vec<_> = (0..3)
            .map(|_| crossbeam::channel::bounded::<Never>(0))
            .collect();
        let mut stopping = StoppingTurns(held.iter().map(|(_, rx)| rx.clone()).collect());

        let started = Instant::now();
        assert!(!stopping.wait_until(started + Duration::from_millis(100)));
        assert!(
            started.elapsed() < Duration::from_millis(400),
            "three turns took {:?}, so the deadline is being applied per turn",
            started.elapsed()
        );
    }
}
