//! End-to-end exercise of a full turn over the wire.
//!
//! The interesting path is reentrant: partway through a turn gallium sends the
//! client an `item/tool/call` request and blocks awaiting *that*, so both sides
//! must keep reading. These tests drive `serve()` through in-memory pipes and
//! play the client by hand.
//!
//! `turn/start` is answered as soon as the turn is accepted, so a test that
//! wants the *outcome* waits for `turn/completed` — see `drive_turn`. Reading
//! the turn's effects off the `turn/start` response would race the turn.

use std::io::{BufReader, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::channel::{unbounded, Receiver, Sender};
use serde_json::{json, Value};

use crate::appserver::rpc::{serve, Connection};
use crate::appserver::server::{AppServer, ServerConfig};
use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo, ToolDefinition};

// ---------------------------------------------------------------------------
// In-memory duplex plumbing
// ---------------------------------------------------------------------------

/// A `Read` fed by a channel, so a test can supply input lazily — in response to
/// what it sees the server write.
struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(bytes) => {
                    self.buf = bytes;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // all senders dropped == EOF
            }
        }
        let n = std::cmp::min(out.len(), self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// A `Write` that forwards each complete line to a channel.
struct ChannelWriter {
    tx: Sender<String>,
    buf: Vec<u8>,
}

impl Write for ChannelWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        while let Some(i) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=i).collect();
            let line = String::from_utf8_lossy(&line[..line.len() - 1]).to_string();
            if !line.trim().is_empty() {
                let _ = self.tx.send(line);
            }
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The test's view of the connection: send lines to the server, read what it writes.
struct ClientSide {
    to_server: Sender<Vec<u8>>,
    from_server: Receiver<String>,
}

impl ClientSide {
    fn send(&self, msg: Value) {
        let line = format!("{msg}\n");
        self.to_server
            .send(line.into_bytes())
            .expect("server alive");
    }

    /// Next message from the server, or panic on timeout — a hang here means a
    /// deadlock, which is exactly what these tests are guarding against.
    fn recv(&self) -> Value {
        let line = self
            .from_server
            .recv_timeout(Duration::from_secs(5))
            .expect("server produced a message within 5s");
        serde_json::from_str(&line).expect("server writes valid JSON")
    }
}

/// Boot `serve()` on a background thread wired to in-memory pipes.
fn start_server(server: AppServer) -> (ClientSide, std::thread::JoinHandle<()>) {
    let (to_server, server_rx) = unbounded::<Vec<u8>>();
    let (server_tx, from_server) = unbounded::<String>();

    let reader = BufReader::new(ChannelReader {
        rx: server_rx,
        buf: Vec::new(),
        pos: 0,
    });
    let conn = Connection::new(Box::new(ChannelWriter {
        tx: server_tx,
        buf: Vec::new(),
    }));

    let handle = std::thread::spawn(move || serve(reader, conn, Arc::new(server)));
    (
        ClientSide {
            to_server,
            from_server,
        },
        handle,
    )
}

// ---------------------------------------------------------------------------
// A provider that plays a fixed script
// ---------------------------------------------------------------------------

struct ScriptedProvider {
    steps: Vec<LlmResponse>,
    calls: AtomicUsize,
}

impl LlmProvider for ScriptedProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(match self.steps.get(i) {
            Some(LlmResponse::ToolCalls(calls, usage)) => {
                LlmResponse::ToolCalls(calls.clone(), usage.clone())
            }
            Some(LlmResponse::Text {
                content,
                reasoning,
                usage,
            }) => LlmResponse::Text {
                content: content.clone(),
                reasoning: reasoning.clone(),
                usage: usage.clone(),
            },
            None => panic!("provider called more times than the script has steps"),
        })
    }
}

fn scripted_server(steps: Vec<LlmResponse>) -> AppServer {
    let provider = Arc::new(ScriptedProvider {
        steps,
        calls: AtomicUsize::new(0),
    });
    AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            // One scripted script per server; cloning the Arc shares the cursor,
            // which is fine because these tests start a single thread.
            Ok(Box::new(SharedProvider(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    )
}

/// Lets several threads share one `ScriptedProvider` behind `Box<dyn LlmProvider>`.
struct SharedProvider(Arc<ScriptedProvider>);

impl LlmProvider for SharedProvider {
    fn chat(&self, m: &[ChatMessage]) -> anyhow::Result<String> {
        self.0.chat(m)
    }
    fn supports_tools(&self) -> bool {
        true
    }
    fn chat_with_tools(
        &self,
        m: &[ChatMessage],
        t: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.0.chat_with_tools(m, t)
    }
}

/// Replies with the same text every turn, recording the history it was handed
/// and reporting a fixed usage — so a test can drive the compaction trigger and
/// then assert on what the model actually saw. `input_tokens: 0` reports no
/// usage at all, the way the native candle backend does.
struct RecordingProvider {
    seen: std::sync::Mutex<Vec<Vec<ChatMessage>>>,
    input_tokens: u64,
}

impl LlmProvider for RecordingProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.seen.lock().unwrap().push(messages.to_vec());
        Ok(LlmResponse::Text {
            content: "ok".to_string(),
            reasoning: None,
            usage: (self.input_tokens > 0).then(|| {
                crate::llm::TokenUsage::single(self.input_tokens, 1, self.input_tokens + 1)
            }),
        })
    }
}

/// Shares one `RecordingProvider` with the thread the server builds.
struct SharedRecorder(Arc<RecordingProvider>);

impl LlmProvider for SharedRecorder {
    fn chat(&self, m: &[ChatMessage]) -> anyhow::Result<String> {
        self.0.chat(m)
    }
    fn supports_tools(&self) -> bool {
        true
    }
    fn chat_with_tools(
        &self,
        m: &[ChatMessage],
        t: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.0.chat_with_tools(m, t)
    }
}

/// Sits inside the model call until released, so a test can look at a turn
/// while it is still running. Answering `turn/start` before the turn is over is
/// the behavior under test, and it cannot be observed with a provider that
/// returns instantly.
struct BlockingProvider {
    entered: Sender<()>,
    release: Receiver<()>,
}

impl LlmProvider for BlockingProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        let _ = self.entered.send(());
        let _ = self.release.recv();
        Ok(LlmResponse::Text {
            content: "eventually".to_string(),
            reasoning: None,
            usage: None,
        })
    }
}

/// Waits in the model call for either a release or a cancellation.
///
/// `reports_cancellation` picks which kind of backend it imitates. The local
/// ones notice the token between sampled tokens and return
/// `AgentError::Cancelled`; a cloud round trip has no interruption point and
/// returns its answer regardless, which `react.rs` then discards at the next
/// boundary check. A turn must end as interrupted either way.
struct InterruptibleProvider {
    entered: Sender<()>,
    release: Receiver<()>,
    reports_cancellation: bool,
}

impl LlmProvider for InterruptibleProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        Ok(LlmResponse::Text {
            content: "eventually".to_string(),
            reasoning: None,
            usage: None,
        })
    }

    fn chat_with_tools_cancellable(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        cancel: &crate::cancel::CancellationToken,
    ) -> anyhow::Result<LlmResponse> {
        let _ = self.entered.send(());
        loop {
            if cancel.is_cancelled() {
                if self.reports_cancellation {
                    return Err(crate::AgentError::Cancelled.into());
                }
                // No interruption point: answer anyway, too late to matter.
                return Ok(LlmResponse::Text {
                    content: "answered after the stop".to_string(),
                    reasoning: None,
                    usage: None,
                });
            }
            if self.release.recv_timeout(Duration::from_millis(5)).is_ok() {
                return Ok(LlmResponse::Text {
                    content: "eventually".to_string(),
                    reasoning: None,
                    usage: None,
                });
            }
        }
    }
}

/// A server whose turns wait in the model call until released or cancelled.
fn interruptible_server(reports_cancellation: bool) -> (AppServer, Receiver<()>, Sender<()>) {
    let (entered_tx, entered_rx) = unbounded::<()>();
    let (release_tx, release_rx) = unbounded::<()>();
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(InterruptibleProvider {
                entered: entered_tx.clone(),
                release: release_rx.clone(),
                reports_cancellation,
            }) as Box<dyn LlmProvider>)
        }),
    );
    (server, entered_rx, release_tx)
}

/// A server whose turns hang in the model call. Returns the "a turn has reached
/// the model" signal and the release handle.
fn blocking_server() -> (AppServer, Receiver<()>, Sender<()>) {
    let (entered_tx, entered_rx) = unbounded::<()>();
    let (release_tx, release_rx) = unbounded::<()>();
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(BlockingProvider {
                entered: entered_tx.clone(),
                release: release_rx.clone(),
            }) as Box<dyn LlmProvider>)
        }),
    );
    (server, entered_rx, release_tx)
}

/// Parks in every model call *and* records the history it was handed.
///
/// Steering can only be observed on a turn that is still running, and its whole
/// point is what the *next* model call sees — so a test needs both halves at
/// once: a turn it can catch mid-flight, and a record of the prompt that
/// followed.
struct ParkedRecordingProvider {
    entered: Sender<()>,
    release: Receiver<()>,
    seen: std::sync::Mutex<Vec<Vec<ChatMessage>>>,
    calls: AtomicUsize,
}

impl LlmProvider for ParkedRecordingProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.seen.lock().unwrap().push(messages.to_vec());
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        let _ = self.entered.send(());
        let _ = self.release.recv();
        Ok(LlmResponse::Text {
            content: format!("answer {}", i + 1),
            reasoning: None,
            usage: None,
        })
    }
}

/// Shares one `ParkedRecordingProvider` with the thread the server builds.
struct SharedParked(Arc<ParkedRecordingProvider>);

impl LlmProvider for SharedParked {
    fn chat(&self, m: &[ChatMessage]) -> anyhow::Result<String> {
        self.0.chat(m)
    }
    fn supports_tools(&self) -> bool {
        true
    }
    fn chat_with_tools(
        &self,
        m: &[ChatMessage],
        t: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.0.chat_with_tools(m, t)
    }
}

/// A server whose turns park in the model call, recording each prompt.
fn steerable_server() -> (
    AppServer,
    Receiver<()>,
    Sender<()>,
    Arc<ParkedRecordingProvider>,
) {
    let (entered_tx, entered_rx) = unbounded::<()>();
    let (release_tx, release_rx) = unbounded::<()>();
    let provider = Arc::new(ParkedRecordingProvider {
        entered: entered_tx,
        release: release_rx,
        seen: std::sync::Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
    });
    let handle = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedParked(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    (server, entered_rx, release_tx, handle)
}

fn recording_server(context_window: u32, input_tokens: u64) -> (AppServer, Arc<RecordingProvider>) {
    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens,
    });
    let handle = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            context_window: Some(context_window),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    (server, handle)
}

/// Drive one turn to completion, draining the notifications it emits.
/// Run one turn and wait for it to actually finish.
///
/// `turn/start` answers as soon as the turn is accepted, so its response says
/// nothing about the outcome — a test that stopped there would inspect the
/// thread's history while the turn was still writing to it. The ending is
/// `turn/completed`, whatever the outcome — the `status` says which one, which
/// is where a client reads it too.
fn drive_turn(client: &ClientSide, id: u64, thread_id: &str, text: &str) {
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": text}] },
    }));
    loop {
        let msg = client.recv();
        if msg["id"] == id && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "turn/start refused: {msg}");
            assert_eq!(
                msg["result"]["turn"]["status"], "inProgress",
                "turn/start should answer with a turn in progress: {msg}"
            );
            continue;
        }
        if msg["method"] == "turn/completed" {
            // The status, not the method: every ending arrives this way now, so
            // returning without looking would report a failed turn as a fine
            // one — the same mistake this change fixed in the client.
            assert_ne!(
                msg["params"]["turn"]["status"], "failed",
                "turn failed: {msg}"
            );
            return;
        }
    }
}

fn handshake(client: &ClientSide, dynamic_tools: Value) -> String {
    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    let init = client.recv();
    assert_eq!(init["id"], 1);

    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": "/tmp", "dynamicTools": dynamic_tools },
    }));
    let started = client.recv();
    started["result"]["thread"]["id"]
        .as_str()
        .expect("thread.id")
        .to_string()
}

/// `initialize`, then one `thread/start` with exactly the params given — for the
/// tests that care what a client did or did not send.
fn thread_start_with(client: &ClientSide, params: Value) -> Value {
    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"},
                    "capabilities": {"experimentalApi": true} },
    }));
    assert_eq!(client.recv()["id"], 1);

    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start", "params": params,
    }));
    client.recv()
}

/// Start an extra thread on an already-initialized connection, optionally naming
/// a model. Returns its thread id.
fn start_thread(client: &ClientSide, id: u64, model: Option<&str>) -> String {
    let mut params = json!({ "cwd": "/tmp" });
    if let Some(model) = model {
        params["model"] = json!(model);
    }
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "thread/start", "params": params,
    }));
    let started = client.recv();
    started["result"]["thread"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("thread.id in {started}"))
        .to_string()
}

/// A server whose factory counts how many providers it was asked to build.
fn counting_server(config: ServerConfig) -> (AppServer, Arc<AtomicUsize>) {
    let builds = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&builds);
    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens: 0,
    });
    let server = AppServer::with_provider_factory(
        config,
        Box::new(move |_cfg, _model| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SharedRecorder(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    (server, builds)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn a_threads_skills_are_loaded_and_catalogued_into_the_prompt() {
    // Regression: `thread/start` built an empty `SkillRegistry` and nothing ever
    // injected a catalog, so `lookup_skill` was advertised to the model in every
    // app-server thread and could never find anything.
    let dir = std::env::temp_dir().join(format!("gallium_skills_{}", std::process::id()));
    let skills_dir = dir.join(".agents").join("skills");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&skills_dir).unwrap();
    std::fs::write(
        skills_dir.join("deploy.md"),
        "---\nname: deploy\ndescription: How to deploy the service\n---\nRun the deploy script.\n",
    )
    .unwrap();

    let (server, provider) = recording_server(128_000, 0);
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    client.recv();
    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": dir.to_string_lossy() },
    }));
    let thread_id = client.recv()["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    drive_turn(&client, 3, &thread_id, "deploy it");

    let seen = provider.seen.lock().unwrap();
    assert!(
        seen[0]
            .iter()
            .any(|m| m.content.contains("deploy") && m.content.contains("How to deploy")),
        "the thread's skills must reach the model: {:?}",
        seen[0].iter().map(|m| &m.content).collect::<Vec<_>>()
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A client keeps its skills wherever its own repo puts them — `skills/`, not
/// `.claude/skills` — so before `skillPaths` existed `LookupSkill` was
/// advertised to the model and always answered empty, whatever the client did.
#[test]
fn a_client_can_name_its_own_skill_directory_on_thread_start() {
    let dir = std::env::temp_dir().join(format!("gallium_skillpaths_{}", std::process::id()));
    let elsewhere = dir.join("their-repo").join("skills");
    let cwd = dir.join("workspace");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(elsewhere.join("triage")).unwrap();
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::write(
        elsewhere.join("triage").join("SKILL.md"),
        "---\nname: triage\ndescription: How to triage an incident\n---\nPage the on-call.\n",
    )
    .unwrap();

    let (server, provider) = recording_server(128_000, 0);
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    client.recv();
    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": {
            "cwd": cwd.to_string_lossy(),
            "skillPaths": [elsewhere.to_string_lossy()],
        },
    }));
    let started = client.recv();
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    drive_turn(&client, 3, &thread_id, "something broke");

    let seen = provider.seen.lock().unwrap();
    assert!(
        seen[0]
            .iter()
            .any(|m| m.content.contains("triage") && m.content.contains("How to triage")),
        "a skill named by skillPaths must reach the model: {:?}",
        seen[0].iter().map(|m| &m.content).collect::<Vec<_>>()
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A relative `skillPaths` entry is the client's natural spelling — it already
/// told us its `cwd` — and must resolve against that, not against whatever
/// directory the app-server process happens to have been launched from.
#[test]
fn a_relative_skill_path_resolves_against_the_threads_cwd() {
    let cwd = std::env::temp_dir().join(format!("gallium_skillrel_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cwd);
    std::fs::create_dir_all(cwd.join("skills")).unwrap();
    std::fs::write(
        cwd.join("skills").join("release.md"),
        "---\nname: release\ndescription: How to cut a release\n---\nTag, then publish.\n",
    )
    .unwrap();

    let (server, provider) = recording_server(128_000, 0);
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    client.recv();
    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": cwd.to_string_lossy(), "skillPaths": ["skills"] },
    }));
    let thread_id = client.recv()["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    drive_turn(&client, 3, &thread_id, "ship it");

    let seen = provider.seen.lock().unwrap();
    assert!(
        seen[0]
            .iter()
            .any(|m| m.content.contains("How to cut a release")),
        "a relative skillPaths entry must resolve against cwd: {:?}",
        seen[0].iter().map(|m| &m.content).collect::<Vec<_>>()
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
    let _ = std::fs::remove_dir_all(&cwd);
}

#[test]
fn threads_share_one_provider_instead_of_building_it_per_thread() {
    let (server, builds) = counting_server(ServerConfig {
        max_iterations: Some(5),
        ..Default::default()
    });
    let (client, handle) = start_server(server);

    let t1 = handshake(&client, json!([]));
    let t2 = start_thread(&client, 10, None);
    let t3 = start_thread(&client, 11, None);
    assert_ne!(t1, t2, "each thread/start gets its own thread");
    assert_ne!(t2, t3);

    assert_eq!(
        builds.load(Ordering::SeqCst),
        1,
        "three threads must share one provider — a local one owns GB of weights",
    );

    drop(client);
    handle.join().unwrap();
}

#[test]
fn a_local_config_keys_the_provider_on_the_model_path() {
    // `create_provider` ignores the thread's `model` when a model_path is set, so
    // two threads naming different models still resolve to the same GGUF and must
    // not each load it.
    let (server, builds) = counting_server(ServerConfig {
        model_path: Some("/models/only-one.gguf".to_string()),
        max_iterations: Some(5),
        ..Default::default()
    });
    let (client, handle) = start_server(server);

    handshake(&client, json!([]));
    start_thread(&client, 10, Some("some-other-model"));

    assert_eq!(
        builds.load(Ordering::SeqCst),
        1,
        "the model_path decides what gets loaded, not the requested model name",
    );

    drop(client);
    handle.join().unwrap();
}

#[test]
fn a_thread_compacts_its_history_once_a_turn_nears_the_context_window() {
    // Window 1000 → compaction triggers at 900 reported tokens, targeting 500.
    let (server, provider) = recording_server(1000, 950);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    // ~1010 estimated tokens on its own, so it cannot survive a 500-token target.
    let bulky = "x".repeat(4000);
    drive_turn(&client, 3, &thread_id, &bulky);
    drive_turn(&client, 4, &thread_id, "second");

    let seen = provider.seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one provider call per turn");
    assert!(
        seen[0].iter().any(|m| m.content == bulky),
        "the first turn must see its own prompt"
    );
    assert!(
        !seen[1].iter().any(|m| m.content == bulky),
        "the bulky first turn should have been compacted away, saw: {:?}",
        seen[1].iter().map(|m| m.content.len()).collect::<Vec<_>>()
    );
    assert!(
        seen[1].iter().any(|m| m.content == "second"),
        "the current prompt must survive compaction"
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
}

#[test]
fn a_thread_compacts_even_when_the_backend_reports_no_token_usage() {
    // The native candle backend reports no usage, so the trigger has to fall
    // back to gallium's own estimate of the history.
    let (server, provider) = recording_server(1000, 0);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let bulky = "x".repeat(4000);
    drive_turn(&client, 3, &thread_id, &bulky);
    drive_turn(&client, 4, &thread_id, "second");

    let seen = provider.seen.lock().unwrap();
    assert!(
        !seen[1].iter().any(|m| m.content == bulky),
        "history must compact on the estimate alone when usage is unreported"
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
}

#[test]
fn a_thread_keeps_its_history_while_it_fits_the_context_window() {
    // Same history, but 950 tokens is nowhere near a 128k window.
    let (server, provider) = recording_server(128_000, 950);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let bulky = "x".repeat(4000);
    drive_turn(&client, 3, &thread_id, &bulky);
    drive_turn(&client, 4, &thread_id, "second");

    let seen = provider.seen.lock().unwrap();
    assert!(
        seen[1].iter().any(|m| m.content == bulky),
        "nothing should be dropped while the history fits"
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
}

#[test]
fn turn_with_no_tools_returns_final_text() {
    let server = scripted_server(vec![LlmResponse::Text {
        content: "hello there".to_string(),
        reasoning: None,
        usage: None,
    }]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "hi"}] },
    }));

    // The turn/start response comes first and only says the turn was accepted;
    // then item/completed(agentMessage), then turn/completed ends it.
    let mut saw_agent_message = false;
    let mut saw_accepted = false;
    loop {
        let msg = client.recv();
        match msg["method"].as_str() {
            Some("item/completed") => {
                if msg["params"]["item"]["type"] == "agentMessage" {
                    assert_eq!(msg["params"]["item"]["text"], "hello there");
                    saw_agent_message = true;
                }
            }
            Some("turn/completed") => {
                assert_eq!(msg["params"]["turn"]["status"], "completed");
                break;
            }
            None => {
                assert_eq!(msg["id"], 3, "expected the turn/start response");
                assert_eq!(msg["result"]["turn"]["status"], "inProgress");
                assert!(msg["result"]["turn"]["id"].is_string());
                saw_accepted = true;
            }
            other => panic!("unexpected method {other:?}"),
        }
    }
    assert!(
        saw_accepted,
        "turn/start must be answered before the turn ends"
    );
    assert!(saw_agent_message);

    drop(client);
    handle.join().unwrap();
}

#[test]
fn turn_calls_back_into_the_client_for_a_dynamic_tool() {
    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "memory".to_string(),
                arguments: json!({"query": "birthday"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "It is in June.".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(
        &client,
        json!([{ "type": "function", "name": "memory", "description": "recall", "inputSchema": {"type": "object"} }]),
    );

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "when?"}] },
    }));

    let mut tool_call_seen = false;
    let mut final_text = None;

    loop {
        let msg = client.recv();

        // A server→client request: the dynamic tool. Answer it, mid-turn.
        if msg["method"] == "item/tool/call" && msg["id"].is_number() {
            let params = &msg["params"];
            assert_eq!(params["tool"], "memory");
            assert_eq!(params["arguments"]["query"], "birthday");
            assert_eq!(params["threadId"], thread_id);
            // The turn id must be the live one, not a placeholder.
            assert!(
                params["turnId"]
                    .as_str()
                    .is_some_and(|t| t.starts_with("turn_")),
                "turnId was {:?}",
                params["turnId"]
            );
            tool_call_seen = true;

            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"],
                "result": { "success": true, "contentItems": [{"type": "inputText", "text": "June 3"}] },
            }));
            continue;
        }

        if msg["method"] == "item/completed" && msg["params"]["item"]["type"] == "agentMessage" {
            final_text = Some(msg["params"]["item"]["text"].as_str().unwrap().to_string());
        }

        if msg["id"] == 3 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "turn/start refused: {msg}");
        }
        if msg["method"] == "turn/completed" {
            break;
        }
    }

    assert!(tool_call_seen, "gallium never called the client's tool");
    assert_eq!(final_text.as_deref(), Some("It is in June."));

    drop(client);
    handle.join().unwrap();
}

#[test]
fn tool_failure_reported_by_the_client_is_fed_back_to_the_model() {
    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "memory".to_string(),
                arguments: json!({}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "I could not recall.".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(
        &client,
        json!([{ "type": "function", "name": "memory", "description": "recall", "inputSchema": {"type": "object"} }]),
    );

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "when?"}] },
    }));

    let mut started = None;
    let mut completed = None;
    loop {
        let msg = client.recv();

        if msg["method"] == "item/tool/call" && msg["id"].is_number() {
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"],
                "result": { "success": false, "contentItems": [{"type": "inputText", "text": "disk on fire"}] },
            }));
            continue;
        }

        if msg["method"] == "item/started" {
            started = Some(msg["params"]["item"].clone());
        }
        if msg["method"] == "item/completed" && msg["params"]["item"]["type"] != "agentMessage" {
            completed = Some(msg["params"]["item"].clone());
        }

        if msg["id"] == 3 && msg["method"].is_null() {
            // A failing client tool is a normal ReAct outcome, not a turn failure.
            assert!(
                msg["error"].is_null(),
                "turn should survive a failing tool: {msg}"
            );
        }
        if msg["method"] == "turn/completed" {
            break;
        }
    }

    // A tool call is announced and then completed, as two notifications sharing
    // one item id. A client keys its dedupe on that id, and decides from the
    // method whether the call is still running.
    let started = started.expect("an item/started announcing the tool call");
    let completed = completed.expect("an item/completed carrying the result");
    assert_eq!(started["id"], "c1", "started: {started}");
    assert_eq!(completed["id"], "c1", "completed: {completed}");
    assert_eq!(started["status"], "inProgress", "started: {started}");

    // A client-declared tool is a `dynamicToolCall`, named by `tool` — not a
    // `commandExecution`, which is the protocol's sandboxed shell item.
    for item in [&started, &completed] {
        assert_eq!(item["type"], "dynamicToolCall", "{item}");
        assert_eq!(item["tool"], "memory", "{item}");
    }

    // `failed` is what tells a client to render this as an error; there is no
    // `isError` field in the item taxonomy.
    assert_eq!(completed["status"], "failed", "completed: {completed}");

    // The output lives in `contentItems`, codex's shape for a dynamic tool call
    // — the same `inputText` shape the client sends its results back in. There
    // is no `result` field on this variant; gallium used to invent one.
    assert_eq!(completed["success"], false, "completed: {completed}");
    let text = completed["contentItems"][0]["text"]
        .as_str()
        .expect("a contentItems entry carrying the output");
    assert_eq!(completed["contentItems"][0]["type"], "inputText");
    assert!(text.contains("disk on fire"), "got: {text}");
    assert!(
        text.contains("Error executing tool 'memory'"),
        "got: {text}"
    );
    // Both notifications describe the call, so the completed item stands alone
    // rather than being a patch against the one before it.
    assert_eq!(completed["arguments"], started["arguments"]);

    drop(client);
    handle.join().unwrap();
}

#[test]
fn turn_against_an_unknown_thread_is_an_error_not_a_panic() {
    let server = scripted_server(vec![]);
    let (client, handle) = start_server(server);
    handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": "nope", "input": [{"type": "text", "text": "hi"}] },
    }));

    let msg = client.recv();
    assert_eq!(msg["id"], 3);
    assert!(msg["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown thread"));

    drop(client);
    handle.join().unwrap();
}

/// Codex rejects empty input on both `turn/start` and `turn/steer`; `turn/steer`
/// already refused one with no text (`a_steer_with_no_text_is_refused_rather_than_silently_dropped`)
/// but `turn/start` accepted it and sent an empty prompt. `UserInput::is_empty()`,
/// not `text.is_empty()`, is the right check: an image with no caption is still a
/// turn worth starting, which is why this test's input is *text-shaped-but-blank*
/// rather than "no input items at all."
#[test]
fn a_turn_start_with_no_text_or_media_is_refused() {
    let server = scripted_server(vec![]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "   "}] },
    }));

    let msg = client.recv();
    assert_eq!(msg["id"], 3);
    assert!(
        msg["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no text or attachments"),
        "{msg}"
    );

    drop(client);
    handle.join().unwrap();
}

/// Under the default policy a `write` must round-trip an approval to the client,
/// and a decline must stop the write.
#[test]
fn write_asks_the_client_for_approval_and_a_decline_blocks_it() {
    let target = std::env::temp_dir().join("gallium_appserver_declined.txt");
    let _ = std::fs::remove_file(&target);

    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": target.to_str().unwrap(), "content": "nope"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "blocked".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write it"}] },
    }));

    let mut asked = false;
    loop {
        let msg = client.recv();
        if msg["method"] == "item/fileChange/requestApproval" && msg["id"].is_number() {
            asked = true;
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"], "result": { "decision": "decline" },
            }));
            continue;
        }
        if msg["method"] == "turn/completed" {
            break;
        }
    }

    assert!(asked, "gallium wrote without asking the client");
    assert!(
        !target.exists(),
        "declined write must not touch the filesystem"
    );

    drop(client);
    handle.join().unwrap();
}

/// `acceptForSession` grants the tier, so the *second* write of the same turn is
/// never asked about.
///
/// The regression this guards: gallium matched the decision against
/// `accept_for_session`, but codex spells both approval-decision enums
/// `#[serde(rename_all = "camelCase")]`. A client answering in the protocol's
/// own spelling fell through the wildcard arm to `Deny` — "yes to all" read as a
/// refusal, and every subsequent write was blocked without a word.
///
/// The writes go *inside* the thread's own cwd on purpose: that makes them
/// `WorkspaceWrite`, the tier a session grant actually covers. `Destructive` —
/// which is what a write outside the workspace root is — is never granted for
/// the session, so aiming this test outside would pass for the wrong reason.
#[test]
fn accept_for_session_grants_the_tier_for_the_rest_of_the_turn() {
    let workspace = std::env::temp_dir().join("gallium_appserver_session");
    std::fs::create_dir_all(&workspace).unwrap();
    let first = workspace.join("one.txt");
    let second = workspace.join("two.txt");
    let _ = std::fs::remove_file(&first);
    let _ = std::fs::remove_file(&second);

    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": first.to_str().unwrap(), "content": "one"}),
            }],
            None,
        ),
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c2".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": second.to_str().unwrap(), "content": "two"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "wrote both".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    client.recv();
    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": workspace.to_str().unwrap() },
    }));
    let thread_id = client.recv()["result"]["thread"]["id"]
        .as_str()
        .expect("thread.id")
        .to_string();

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write both"}] },
    }));

    let mut asked = 0;
    loop {
        let msg = client.recv();
        if msg["method"] == "item/fileChange/requestApproval" && msg["id"].is_number() {
            asked += 1;
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"], "result": { "decision": "acceptForSession" },
            }));
            continue;
        }
        if msg["method"] == "turn/completed" {
            break;
        }
    }

    assert_eq!(
        asked, 1,
        "the session grant should have covered the second write"
    );
    assert!(first.exists(), "the approved write did not happen");
    assert!(
        second.exists(),
        "the second write was blocked despite a session grant"
    );

    let _ = std::fs::remove_file(&first);
    let _ = std::fs::remove_file(&second);
    drop(client);
    handle.join().unwrap();
}

/// `cancel` is codex's fourth decision: refuse *and* stop the turn.
///
/// Distinguishable from `decline` only by what happens next — a declined write
/// leaves the model free to try something else, and this turn must not reach its
/// second tool call at all. So the script offers one, and the test asserts the
/// turn ends `interrupted` without it having run.
#[test]
fn cancel_at_an_approval_stops_the_turn() {
    let blocked = std::env::temp_dir().join("gallium_appserver_cancelled.txt");
    let after = std::env::temp_dir().join("gallium_appserver_after_cancel.txt");
    let _ = std::fs::remove_file(&blocked);
    let _ = std::fs::remove_file(&after);

    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": blocked.to_str().unwrap(), "content": "nope"}),
            }],
            None,
        ),
        // Never reached: cancelling stops the loop before it asks the model again.
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c2".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": after.to_str().unwrap(), "content": "also nope"}),
            }],
            None,
        ),
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write it"}] },
    }));

    let status = loop {
        let msg = client.recv();
        if msg["method"] == "item/fileChange/requestApproval" && msg["id"].is_number() {
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"], "result": { "decision": "cancel" },
            }));
            continue;
        }
        if msg["method"] == "turn/completed" {
            break msg["params"]["turn"]["status"].clone();
        }
    };

    assert_eq!(
        status, "interrupted",
        "a cancelled approval must end the turn as interrupted"
    );
    assert!(!blocked.exists(), "cancel must also refuse the write");
    assert!(
        !after.exists(),
        "the turn continued past a cancel and ran the next tool call"
    );

    drop(client);
    handle.join().unwrap();
}

/// A `cancel` in the turn that starts the instant the previous one ends still
/// stops *that* turn.
///
/// `Thread::current_cancel` is one cell shared across turns, so the obvious
/// worry is a finishing worker clearing the token a newly started turn has just
/// published. It cannot: the clear happens inside the `active_turn` critical
/// section, and a turn cannot claim the slot until that guard drops — so the
/// clear strictly precedes the next turn's claim, its spawn, and its publish.
///
/// The lock is the guarantee; this test is the guard on it. Moving the clear
/// out of that section — which reads like a harmless tidy-up — is what would
/// open the window, and this is what fails when someone does.
#[test]
fn a_cancel_stops_the_turn_that_started_right_after_the_last_one_ended() {
    let blocked = std::env::temp_dir().join("gallium_appserver_cancel_handoff.txt");
    let _ = std::fs::remove_file(&blocked);

    let server = scripted_server(vec![
        // Turn one: ends immediately, freeing the slot.
        LlmResponse::Text {
            content: "first done".to_string(),
            reasoning: None,
            usage: None,
        },
        // Turn two: asks to write, and is cancelled at the approval.
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": blocked.to_str().unwrap(), "content": "nope"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "should never be reached".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "first"}] },
    }));
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }

    // No pause: the second turn is accepted the moment the first is observably
    // over, which is the narrowest gap a client can produce.
    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "second"}] },
    }));

    let status = loop {
        let msg = client.recv();
        if msg["method"] == "item/fileChange/requestApproval" && msg["id"].is_number() {
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"], "result": { "decision": "cancel" },
            }));
            continue;
        }
        if msg["method"] == "turn/completed" {
            break msg["params"]["turn"]["status"].clone();
        }
    };

    assert_eq!(
        status, "interrupted",
        "the second turn published a token the first turn's ending had already cleared"
    );
    assert!(!blocked.exists(), "cancel must also refuse the write");

    drop(client);
    handle.join().unwrap();
}

/// `approvalPolicy: "never"` means the client does not want to be consulted.
#[test]
fn approval_policy_never_writes_without_asking() {
    let target = std::env::temp_dir().join("gallium_appserver_auto.txt");
    let _ = std::fs::remove_file(&target);

    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": target.to_str().unwrap(), "content": "hello"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "wrote".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "capabilities": {"experimentalApi": true} },
    }));
    client.recv();
    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": "/tmp", "approvalPolicy": "never" },
    }));
    let thread_id = client.recv()["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write it"}] },
    }));

    loop {
        let msg = client.recv();
        assert_ne!(
            msg["method"], "item/fileChange/requestApproval",
            "approvalPolicy=never must not ask"
        );
        if msg["id"] == 3 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "turn/start refused: {msg}");
        }
        if msg["method"] == "turn/completed" {
            break;
        }
    }

    assert_eq!(std::fs::read_to_string(&target).unwrap(), "hello");
    let _ = std::fs::remove_file(&target);

    drop(client);
    handle.join().unwrap();
}

#[test]
fn developer_instructions_become_the_system_prompt() {
    // The provider asserts on what it is handed, so a single Text step suffices.
    struct AssertingProvider;
    impl LlmProvider for AssertingProvider {
        fn chat(&self, _m: &[ChatMessage]) -> anyhow::Result<String> {
            Ok(String::new())
        }
        fn supports_tools(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            messages: &[ChatMessage],
            _t: &[ToolDefinition],
        ) -> anyhow::Result<LlmResponse> {
            assert_eq!(messages[0].role, crate::llm::ChatRole::System);
            assert_eq!(messages[0].content, "be terse");
            Ok(LlmResponse::Text {
                content: "ok".to_string(),
                reasoning: None,
                usage: None,
            })
        }
    }

    let server = AppServer::with_provider_factory(
        ServerConfig::default(),
        Box::new(|_c, _m| Ok(Box::new(AssertingProvider))),
    );
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "capabilities": {"experimentalApi": true} },
    }));
    client.recv();

    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": "/tmp", "developerInstructions": "be terse" },
    }));
    let thread_id = client.recv()["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "hi"}] },
    }));

    loop {
        let msg = client.recv();
        if msg["id"] == 3 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "turn failed: {msg}");
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// `turn/start` is answered while the turn is still running, which is what
/// codex does and what makes a turn interruptible at all: a reply that only
/// arrives once the turn is over cannot be the thing a client waits on while
/// trying to stop it (#28).
#[test]
fn turn_start_is_answered_before_the_turn_finishes() {
    let (server, entered, release) = blocking_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "hi"}] },
    }));

    // The turn is now sitting in the model call, and has answered anyway.
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the turn should reach the model");
    let reply = client.recv();
    assert_eq!(reply["id"], 3, "expected the turn/start response: {reply}");
    assert_eq!(reply["result"]["turn"]["status"], "inProgress");
    assert!(reply["result"]["turn"]["id"].is_string());

    // And it still ends the normal way once the model returns.
    let _ = release.send(());
    loop {
        let msg = client.recv();
        if msg["method"] == "turn/completed" {
            assert_eq!(msg["params"]["turn"]["status"], "completed", "{msg}");
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// One turn at a time per thread, refused out loud.
///
/// While `turn/start` blocked, a second call simply waited on the history lock
/// and the client could not tell. Now that turns run in the background, a
/// silent wait would look like a turn that had started and produced nothing.
#[test]
fn a_second_turn_on_a_busy_thread_is_refused() {
    let (server, entered, release) = blocking_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "first"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the first turn should reach the model");
    assert_eq!(client.recv()["id"], 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "second"}] },
    }));
    let refusal = client.recv();
    assert_eq!(refusal["id"], 4);
    let message = refusal["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("already running"),
        "a busy thread must say so: {refusal}"
    );
    // Naming the turn in flight is what lets a client interrupt the right one.
    assert!(message.contains("turn_1"), "{refusal}");

    // The slot frees up again, so the thread is usable after the turn ends.
    // Two releases: the provider blocks on every call, and the third turn below
    // needs one waiting for it. The channel is unbounded, so they queue.
    let _ = release.send(());
    let _ = release.send(());
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }
    drive_turn(&client, 5, &thread_id, "third");

    drop(client);
    handle.join().unwrap();
}

/// A turn started the moment the previous one is seen to end is accepted.
///
/// This is the half of the slot-handoff race a test can actually pin. The other
/// half — a turn accepted in the gap interleaving its notifications ahead of the
/// previous turn's ending — is a few instructions wide and cannot be landed on
/// reliably from out here; it is closed by holding the slot's lock across both
/// steps rather than by this test. Reverting that fix does not fail anything,
/// which is exactly why the lock is the guarantee and this is only a guard on
/// the ordering it must not be "fixed" into.
#[test]
fn the_next_turn_is_accepted_as_soon_as_the_last_one_is_seen_to_end() {
    let (server, entered, release) = blocking_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "first"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the first turn should reach the model");
    assert_eq!(client.recv()["id"], 3);
    // Enough releases for both turns; the channel buffers them.
    let _ = release.send(());
    let _ = release.send(());

    // Wait for the first turn to be *observably* over, then start another with
    // no pause at all. A slot released after the notification would refuse this.
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }
    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "second"}] },
    }));

    loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            assert!(
                msg["error"].is_null(),
                "a turn started right after the last one ended was refused: {msg}"
            );
            assert_eq!(msg["result"]["turn"]["status"], "inProgress");
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// Starts a turn and waits until it is inside the model call, so there is
/// something real to interrupt. Returns the thread id.
fn start_a_stuck_turn(client: &ClientSide, entered: &Receiver<()>, id: u64) -> String {
    let thread_id = handshake(client, json!([]));
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the turn should reach the model");
    assert_eq!(client.recv()["id"], id, "the turn/start response");
    thread_id
}

/// The whole point of #28: a running turn stops, and the client is told the
/// turn ended — not that it failed.
///
/// The response ordering is the contract worth pinning. Codex answers an
/// interrupt only once the turn has actually aborted, so `{}` arriving before
/// `turn/completed` would make it a doorbell rather than a stop button.
#[test]
fn an_interrupt_stops_the_turn_and_answers_once_it_has() {
    let (server, entered, _release) = interruptible_server(true);
    let (client, handle) = start_server(server);
    let thread_id = start_a_stuck_turn(&client, &entered, 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_1" },
    }));

    let mut order = Vec::new();
    loop {
        let msg = client.recv();
        if msg["method"] == "turn/completed" {
            assert_eq!(
                msg["params"]["turn"]["status"], "interrupted",
                "a stopped turn ends as interrupted, not completed: {msg}"
            );
            order.push("ended");
        }

        if msg["id"] == 4 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "interrupt refused: {msg}");
            assert_eq!(msg["result"], json!({}), "codex answers with {{}}");
            order.push("acknowledged");
            break;
        }
    }
    assert_eq!(
        order,
        vec!["ended", "acknowledged"],
        "the interrupt must be answered only once the turn has stopped"
    );

    drop(client);
    handle.join().unwrap();
}

/// A backend with no interruption point — a cloud round trip — runs its call to
/// completion. The turn still ends as interrupted, because `react.rs` discards
/// that answer at the next boundary rather than feeding it to a turn that is
/// over. This is the "prompt, not instantaneous" half of the contract.
#[test]
fn a_backend_that_cannot_be_interrupted_still_ends_the_turn_as_interrupted() {
    let (server, entered, _release) = interruptible_server(false);
    let (client, handle) = start_server(server);
    let thread_id = start_a_stuck_turn(&client, &entered, 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_1" },
    }));

    let mut ended_as = None;
    loop {
        let msg = client.recv();
        if msg["method"] == "turn/completed" {
            ended_as = msg["params"]["turn"]["status"].as_str().map(str::to_string);
        }
        // The answer that arrived too late must not reach the client as the
        // turn's reply.
        if msg["method"] == "item/completed" {
            assert_ne!(
                msg["params"]["item"]["text"], "answered after the stop",
                "a late answer must not be delivered: {msg}"
            );
        }
        if msg["id"] == 4 && msg["method"].is_null() {
            break;
        }
    }
    assert_eq!(ended_as.as_deref(), Some("interrupted"));

    drop(client);
    handle.join().unwrap();
}

/// Codex refuses an interrupt aimed at a turn that is not the running one, and
/// names both — without that, a client racing an ending would silently kill the
/// turn after it.
#[test]
fn an_interrupt_naming_another_turn_is_refused() {
    let (server, entered, _release) = interruptible_server(true);
    let (client, handle) = start_server(server);
    let thread_id = start_a_stuck_turn(&client, &entered, 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_9" },
    }));

    let refusal = loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            break msg;
        }
    };
    let message = refusal["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("turn_9"), "{refusal}");
    assert!(message.contains("turn_1"), "{refusal}");

    // And the real turn is untouched: still running, still interruptible.
    client.send(json!({
        "jsonrpc": "2.0", "id": 5, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_1" },
    }));
    loop {
        let msg = client.recv();
        if msg["id"] == 5 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "{msg}");
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

#[test]
fn an_interrupt_with_no_turn_running_is_refused() {
    let (server, _entered, _release) = interruptible_server(true);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_1" },
    }));
    let refusal = client.recv();
    assert!(refusal["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .contains("no active turn"));

    drop(client);
    handle.join().unwrap();
}

/// Codex reads an empty `turnId` as "cancel startup". Gallium has none, and
/// says so rather than reporting a mismatch against the empty string.
#[test]
fn an_interrupt_without_a_turn_id_says_there_is_no_startup_to_cancel() {
    let (server, entered, _release) = interruptible_server(true);
    let (client, handle) = start_server(server);
    let thread_id = start_a_stuck_turn(&client, &entered, 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "" },
    }));
    let refusal = loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            break msg;
        }
    };
    let message = refusal["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("startup"), "{refusal}");

    drop(client);
    handle.join().unwrap();
}

/// An interrupted thread is usable again: the slot is released, and history was
/// rolled back to before the interrupted prompt.
#[test]
fn a_thread_is_usable_after_an_interrupt() {
    let (server, entered, release) = interruptible_server(true);
    let (client, handle) = start_server(server);
    let thread_id = start_a_stuck_turn(&client, &entered, 3);

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/interrupt",
        "params": { "threadId": thread_id, "turnId": "turn_1" },
    }));
    loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "{msg}");
            break;
        }
    }

    // The next turn runs to completion normally.
    let _ = release.send(());
    drive_turn(&client, 5, &thread_id, "after");

    drop(client);
    handle.join().unwrap();
}

// ---------------------------------------------------------------------------
// turn/steer
// ---------------------------------------------------------------------------

/// The whole point of steering, end to end: text handed to a turn already in
/// flight reaches the *next* model call, under the same turn id.
#[test]
fn a_steered_turn_carries_the_new_text_into_the_next_model_call() {
    let (server, entered, release, provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write it"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the turn should reach the model");

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_1",
            "clientUserMessageId": "client-msg-1",
            "input": [{"type": "text", "text": "wait — in Python"}],
        },
    }));

    // The steer is acknowledged with the turn it joined — not a new one.
    let mut echoed = None;
    let mut echo_methods = Vec::new();
    loop {
        let msg = client.recv();
        if msg["method"] == "item/started" || msg["method"] == "item/completed" {
            let item = &msg["params"]["item"];
            if item["type"] == "userMessage" {
                echo_methods.push(msg["method"].as_str().unwrap_or_default().to_string());
                echoed = Some(msg.clone());
            }
            continue;
        }
        if msg["id"] == 4 && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "steer refused: {msg}");
            assert_eq!(msg["result"]["turnId"], "turn_1", "{msg}");
            break;
        }
    }

    let echoed = echoed.expect("the steered message is echoed back as an item");
    assert_eq!(echoed["params"]["turnId"], "turn_1");
    assert_eq!(echoed["params"]["item"]["clientId"], "client-msg-1");
    assert_eq!(
        echoed["params"]["item"]["content"][0]["text"],
        "wait — in Python"
    );
    // Both halves of the lifecycle, as codex emits for a user message. A client
    // that renders on `item/completed` would otherwise never show the steer, and
    // one tracking open items would hold it open for the rest of the turn.
    assert_eq!(
        echo_methods,
        vec!["item/started".to_string(), "item/completed".to_string()],
        "the echoed message needs both halves of the item lifecycle"
    );
    assert!(
        echoed["params"]["item"]["id"].is_string(),
        "an item a client has to match across two notifications needs an id: {echoed}"
    );

    // Two releases: the first answer is superseded by the steer, so the loop
    // asks the model again.
    let _ = release.send(());
    let _ = release.send(());

    let mut superseded = None;
    loop {
        let msg = client.recv();
        if msg["method"] == "item/completed" && msg["params"]["item"]["type"] == "agentMessage" {
            let text = msg["params"]["item"]["text"].as_str().unwrap_or_default();
            if text == "answer 1" {
                superseded = Some(text.to_string());
            }
            continue;
        }
        if msg["method"] == "turn/completed" {
            assert_eq!(
                msg["params"]["turn"]["id"], "turn_1",
                "steering must not mint a second turn: {msg}"
            );
            assert_eq!(msg["params"]["turn"]["status"], "completed", "{msg}");
            break;
        }
    }
    assert!(
        superseded.is_some(),
        "the answer the steer superseded must still reach the client"
    );

    let seen = provider.seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "the model is asked again after a steer");
    assert!(
        seen[1]
            .iter()
            .any(|m| m.role == crate::llm::ChatRole::User && m.content == "wait — in Python"),
        "the steered text must be in the second prompt: {:?}",
        seen[1]
    );

    drop(client);
    handle.join().unwrap();
}

/// Codex's `UserMessageItem.clientId` is `#[serde(skip_serializing_if =
/// "Option::is_none")]`, so a TypeScript client generated from that schema with
/// `exactOptionalPropertyTypes` rejects `{"clientId": null}`. A steer that omits
/// `clientUserMessageId` must therefore omit the key entirely, not send it null.
#[test]
fn a_steer_with_no_client_id_omits_the_key_rather_than_sending_null() {
    let (server, entered, release, _provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "write it"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("the turn should reach the model");

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_1",
            "input": [{"type": "text", "text": "wait — in Python"}],
        },
    }));

    let echoed = loop {
        let msg = client.recv();
        if msg["method"] == "item/started" && msg["params"]["item"]["type"] == "userMessage" {
            break msg;
        }
    };
    assert!(
        echoed["params"]["item"].get("clientId").is_none(),
        "clientId must be absent, not null: {echoed}"
    );

    let _ = release.send(());
    let _ = release.send(());
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// A steer that arrives after the turn has stopped reading is refused, not
/// acknowledged. "Accepted" has to mean the model will see it — an ack for text
/// that lands in a turn nobody is reading is the one failure a client cannot
/// detect for itself.
#[test]
fn a_steer_arriving_after_the_turn_ended_is_refused() {
    let (server, entered, release, provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("parked");
    let _ = release.send(());
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_1",
            "input": [{"type": "text", "text": "one more thing"}],
        },
    }));
    let refusal = loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            break msg;
        }
    };
    assert!(
        !refusal["error"].is_null(),
        "a steer nobody will read must be refused: {refusal}"
    );
    assert_eq!(
        provider.seen.lock().unwrap().len(),
        1,
        "a refused steer must not reach the model"
    );

    drop(client);
    handle.join().unwrap();
}

/// `expectedTurnId` is a precondition. A client steering a turn that has been
/// replaced would otherwise drop its message into unrelated work.
#[test]
fn a_steer_naming_another_turn_is_refused() {
    let (server, entered, release, provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("parked");

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_9",
            "input": [{"type": "text", "text": "too late"}],
        },
    }));
    let refusal = loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            break msg;
        }
    };
    let message = refusal["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("turn_9"), "{refusal}");
    assert!(message.contains("turn_1"), "{refusal}");

    // The running turn is untouched — it ends with one model call, not two.
    let _ = release.send(());
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }
    assert_eq!(
        provider.seen.lock().unwrap().len(),
        1,
        "a refused steer must not reach the model"
    );

    drop(client);
    handle.join().unwrap();
}

#[test]
fn a_steer_with_no_turn_running_is_refused() {
    let (server, _entered, _release, _provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_1",
            "input": [{"type": "text", "text": "nobody is listening"}],
        },
    }));
    let refusal = client.recv();
    assert!(
        refusal["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no active turn"),
        "{refusal}"
    );

    drop(client);
    handle.join().unwrap();
}

/// A steer carrying nothing we can render into a message would be accepted and
/// then do nothing, which reads to the client as silent loss.
#[test]
fn a_steer_with_no_text_is_refused_rather_than_silently_dropped() {
    let (server, entered, release, _provider) = steerable_server();
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));
    entered
        .recv_timeout(Duration::from_secs(5))
        .expect("parked");

    client.send(json!({
        "jsonrpc": "2.0", "id": 4, "method": "turn/steer",
        "params": {
            "threadId": thread_id,
            "expectedTurnId": "turn_1",
            "input": [{"type": "image", "imageUrl": "data:..."}],
        },
    }));
    let refusal = loop {
        let msg = client.recv();
        if msg["id"] == 4 && msg["method"].is_null() {
            break msg;
        }
    };
    assert!(
        refusal["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("no text"),
        "{refusal}"
    );

    let _ = release.send(());
    loop {
        if client.recv()["method"] == "turn/completed" {
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

// ---------------------------------------------------------------------------
// thread/tokenUsage/updated
// ---------------------------------------------------------------------------

/// Collect every `thread/tokenUsage/updated` a turn emits, then its ending.
fn drive_turn_collecting_usage(
    client: &ClientSide,
    id: u64,
    thread_id: &str,
    text: &str,
) -> Vec<Value> {
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": text}] },
    }));
    let mut usage = Vec::new();
    loop {
        let msg = client.recv();
        if msg["method"] == "thread/tokenUsage/updated" {
            usage.push(msg["params"].clone());
            continue;
        }
        if msg["method"] == "turn/completed" {
            assert_ne!(
                msg["params"]["turn"]["status"], "failed",
                "turn failed: {msg}"
            );
            return usage;
        }
    }
}

/// What a context gauge is drawn from. Without this the client has the turn's
/// text and nothing else — which is why `fpt/voice-agent#18` had to delete its
/// gauge rather than fix it.
#[test]
fn a_turn_reports_what_it_spent_and_the_window_it_spent_it_in() {
    let (server, _provider) = recording_server(12_800, 3382);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let usage = drive_turn_collecting_usage(&client, 3, &thread_id, "hi");

    assert_eq!(usage.len(), 1, "one model call, one report: {usage:?}");
    let u = &usage[0];
    assert_eq!(u["threadId"], thread_id);
    assert_eq!(u["turnId"], "turn_1");
    assert_eq!(u["tokenUsage"]["last"]["inputTokens"], 3382);
    assert_eq!(u["tokenUsage"]["last"]["outputTokens"], 1);
    assert_eq!(u["tokenUsage"]["last"]["totalTokens"], 3383);
    assert_eq!(
        u["tokenUsage"]["modelContextWindow"], 12_800,
        "the configured window is a known one: {u}"
    );

    drop(client);
    handle.join().unwrap();
}

/// `total` is the thread's, not the turn's — a gauge is about the conversation,
/// so a second turn must add to the first rather than replace it.
#[test]
fn the_total_accumulates_across_a_threads_turns() {
    let (server, _provider) = recording_server(12_800, 100);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let first = drive_turn_collecting_usage(&client, 3, &thread_id, "one");
    let second = drive_turn_collecting_usage(&client, 4, &thread_id, "two");

    assert_eq!(first[0]["tokenUsage"]["total"]["totalTokens"], 101);
    assert_eq!(
        second[0]["tokenUsage"]["total"]["totalTokens"], 202,
        "the thread's total, not this turn's: {:?}",
        second[0]
    );
    assert_eq!(
        second[0]["tokenUsage"]["last"]["totalTokens"], 101,
        "`last` stays the most recent call"
    );

    drop(client);
    handle.join().unwrap();
}

/// A provider that reports nothing produces no gauge rather than a zeroed one.
#[test]
fn a_turn_that_measured_nothing_reports_nothing() {
    let (server, _provider) = recording_server(12_800, 0);
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let usage = drive_turn_collecting_usage(&client, 3, &thread_id, "hi");

    assert!(
        usage.is_empty(),
        "no usage reported means no notification, not a zero one: {usage:?}"
    );

    drop(client);
    handle.join().unwrap();
}

/// The honesty rule on the wire. The test provider knows no window, and this
/// server configures none, so the fallback is in play — a number gallium
/// compacts against but nobody vouched for, and the client is told so.
#[test]
fn a_window_nobody_can_vouch_for_is_reported_as_null() {
    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens: 3382,
    });
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            context_window: None,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let usage = drive_turn_collecting_usage(&client, 3, &thread_id, "hi");

    assert_eq!(usage.len(), 1);
    assert!(
        usage[0]["tokenUsage"]["modelContextWindow"].is_null(),
        "a guessed window must not be presented as the model's: {}",
        usage[0]
    );
    // The counts are still real and still reported — only the denominator is
    // missing, which is exactly the distinction being drawn.
    assert_eq!(usage[0]["tokenUsage"]["last"]["inputTokens"], 3382);

    drop(client);
    handle.join().unwrap();
}

/// `contextWindow = 0` switches compaction off. It is not a window, and a client
/// handed `modelContextWindow: 0` would divide by it — so the wire carries null,
/// the same as any other case where nothing can be vouched for.
#[test]
fn compaction_switched_off_does_not_put_a_zero_denominator_on_the_wire() {
    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens: 3382,
    });
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            context_window: Some(0),
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    let usage = drive_turn_collecting_usage(&client, 3, &thread_id, "hi");

    assert_eq!(usage.len(), 1);
    assert!(
        usage[0]["tokenUsage"]["modelContextWindow"].is_null(),
        "zero is a compaction sentinel, not a denominator: {}",
        usage[0]
    );

    drop(client);
    handle.join().unwrap();
}

// ---------------------------------------------------------------------------
// Codex type fidelity
// ---------------------------------------------------------------------------

/// Codex's own types, transcribed.
///
/// The point of the exercise in #77 is that a client deserializing gallium's
/// answers into codex's Rust/TS types succeeds. Asserting on individual JSON
/// fields cannot show that: it checks the fields someone thought to list, and a
/// *missing required* field is exactly the failure that gets overlooked — it was
/// missing from `initialize` for the whole life of the app-server.
///
/// So these mirror the required shape and let serde be the judge. Transcribed
/// from `../codex` at `74b8f8db93`, `codex-rs/app-server-protocol/src/protocol`,
/// keeping only fields codex requires: anything `Option` (serde defaults those
/// to `None` even without `#[serde(default)]`), anything carrying
/// `#[serde(default)]`, and the experimental extras are all genuinely optional
/// and are left out. Unknown fields are allowed, as codex allows them, so these
/// prove the shape is *sufficient* rather than exact — which is why the
/// non-codex extras gallium used to answer `thread/start` with (a flat
/// `threadId`, a `skillCount`) passed here for as long as they existed.
///
/// This will drift. When it does, the fix is to re-transcribe from codex rather
/// than to relax the struct until it passes — a mirror that has been loosened to
/// fit gallium proves nothing about codex.
// Most fields below are read by nothing but serde: deserializing them
// successfully *is* the test (see the module doc above), so `dead_code`'s
// "never read" is exactly the point, not a bug.
#[allow(dead_code)]
mod codex_shapes {
    use serde::Deserialize;
    use serde_json::Value;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct InitializeResponse {
        pub user_agent: String,
        pub codex_home: String,
        pub platform_family: String,
        pub platform_os: String,
    }

    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum ThreadStatus {
        NotLoaded,
        Idle,
        SystemError,
        Active { active_flags: Vec<Value> },
    }

    /// The app-server's own `SessionSource` (`v2/thread_data.rs`), which is the
    /// one `Thread.source` is typed with — **not** the core enum of the same
    /// name in `protocol/src/protocol.rs`, which is lowercase and spells this
    /// variant `Mcp`. Codex converts between them (`Mcp => AppServer`), so the
    /// wire value for an app-server thread is `appServer`.
    ///
    /// Transcribed as camelCase-with-`AppServer` because that is what codex
    /// declares here, not because it is what gallium happens to send.
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub enum SessionSource {
        Cli,
        #[serde(rename = "vscode")]
        VsCode,
        Exec,
        AppServer,
        Custom(String),
        #[serde(other)]
        Unknown,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub enum TurnStatus {
        Completed,
        Interrupted,
        Failed,
        InProgress,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Turn {
        pub id: String,
        pub items: Vec<Value>,
        pub status: TurnStatus,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Thread {
        pub id: String,
        pub session_id: String,
        pub preview: String,
        pub ephemeral: bool,
        pub model_provider: String,
        pub created_at: i64,
        pub updated_at: i64,
        pub status: ThreadStatus,
        pub cwd: String,
        pub cli_version: String,
        pub source: SessionSource,
        pub turns: Vec<Turn>,
    }

    /// Kebab-case, and `granular` carries a payload — but gallium only ever
    /// answers with one of the unit variants.
    #[derive(Deserialize, PartialEq, Debug)]
    #[serde(rename_all = "kebab-case")]
    pub enum AskForApproval {
        Untrusted,
        OnRequest,
        Never,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ApprovalsReviewer {
        User,
        AutoReview,
    }

    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "kebab-case")]
    pub enum SandboxPolicy {
        DangerFullAccess,
        ReadOnly,
        ExternalSandbox,
        WorkspaceWrite,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThreadStartResponse {
        pub thread: Thread,
        pub model: String,
        pub model_provider: String,
        pub cwd: String,
        pub approval_policy: AskForApproval,
        pub approvals_reviewer: ApprovalsReviewer,
        pub sandbox: SandboxPolicy,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct TurnStartResponse {
        pub turn: Turn,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct McpToolCallResult {
        pub content: Vec<Value>,
    }

    /// `dynamicToolCall`'s output block — the same `inputText` shape a
    /// `dynamicTools` client sends its results back in.
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum ContentItem {
        InputText { text: String },
        InputImage { image_url: String },
        InputAudio { audio_url: String },
    }

    /// The subset of `ThreadItem` gallium emits. `#[serde(tag = "type")]` with
    /// camelCase variants, as codex declares it.
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "camelCase")]
    pub enum ThreadItem {
        #[serde(rename_all = "camelCase")]
        AgentMessage { id: String, text: String },
        #[serde(rename_all = "camelCase")]
        DynamicToolCall {
            id: String,
            tool: String,
            arguments: Value,
            status: DynamicToolCallStatus,
            /// Optional in codex, and left optional here — a mirror tightened
            /// past codex proves as little as one loosened past it. What gallium
            /// *promises* to send is asserted in the test body instead, so the
            /// two claims stay separable: this type says what a codex client
            /// requires, the assertions say what gallium delivers.
            ///
            /// Typed rather than `Value`, so a regression to a bare `result`
            /// string fails here as a parse error and not only as an assertion.
            content_items: Option<Vec<ContentItem>>,
            success: Option<bool>,
        },
        #[serde(rename_all = "camelCase")]
        McpToolCall {
            id: String,
            server: String,
            tool: String,
            arguments: Value,
            status: DynamicToolCallStatus,
            result: Option<McpToolCallResult>,
        },
    }

    #[derive(Deserialize, PartialEq, Debug)]
    #[serde(rename_all = "camelCase")]
    pub enum DynamicToolCallStatus {
        InProgress,
        Completed,
        Failed,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FileChangeRequestApprovalParams {
        pub thread_id: String,
        pub turn_id: String,
        pub item_id: String,
        pub started_at_ms: i64,
    }
}

/// Deserialize or fail the test with the field serde objected to.
fn parse_as<T: serde::de::DeserializeOwned>(what: &str, value: &Value) -> T {
    serde_json::from_value(value.clone())
        .unwrap_or_else(|e| panic!("{what} does not satisfy codex's type: {e}\npayload: {value}"))
}

/// Every response and item a one-tool turn produces parses as codex's type.
///
/// One test rather than several, because it is one claim: a codex-native client
/// can drive gallium through a turn without a deserialization failure. Splitting
/// it would mean re-driving the same turn to look at the next message.
#[test]
fn a_whole_turn_parses_as_codex_types() {
    // A built-in `write` inside the thread's cwd: it raises an approval *and*
    // renders as a `dynamicToolCall` item, so one tool call exercises both the
    // approval params and the item shape. (`mcpToolCall` needs a live MCP
    // server; `identify_tool`'s mapping onto it is covered by its own unit test.)
    let workspace = std::env::temp_dir().join("gallium_appserver_codex_shapes");
    std::fs::create_dir_all(&workspace).unwrap();
    let target = workspace.join("out.txt");
    let _ = std::fs::remove_file(&target);

    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "write".to_string(),
                arguments: json!({"file_path": target.to_str().unwrap(), "content": "hi"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "all done".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);

    client.send(json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "clientInfo": {"name": "test"}, "capabilities": {"experimentalApi": true} },
    }));
    let init: codex_shapes::InitializeResponse = parse_as("initialize", &client.recv()["result"]);
    assert!(init.user_agent.starts_with("gallium/"));
    assert!(!init.codex_home.is_empty());
    assert!(!init.platform_os.is_empty());
    assert!(!init.platform_family.is_empty());

    client.send(json!({
        "jsonrpc": "2.0", "id": 2, "method": "thread/start",
        "params": { "cwd": workspace.to_str().unwrap() },
    }));
    let started = client.recv();
    let thread: codex_shapes::ThreadStartResponse = parse_as("thread/start", &started["result"]);
    // `thread.id` is the only place the id is answered, and it is what every
    // later request addresses this thread by.
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(thread.thread.id, thread_id);
    assert_eq!(thread.cwd, workspace.to_string_lossy());
    // Nothing is persisted, and the response says so rather than implying a
    // stored thread a client could later ask to resume.
    assert!(thread.thread.ephemeral);
    // Approvals are asked for, so this must not read as `never`.
    assert_eq!(
        thread.approval_policy,
        codex_shapes::AskForApproval::Untrusted
    );

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));

    let mut items = 0;
    let mut saw_approval = false;
    let mut saw_agent_message = false;
    let mut saw_tool_output = false;
    loop {
        let msg = client.recv();
        match msg["method"].as_str() {
            Some("item/fileChange/requestApproval") => {
                saw_approval = true;
                let params: codex_shapes::FileChangeRequestApprovalParams =
                    parse_as("fileChange/requestApproval", &msg["params"]);
                assert_eq!(params.thread_id, thread_id);
                assert!(!params.item_id.is_empty());
                client.send(json!({
                    "jsonrpc": "2.0", "id": msg["id"], "result": { "decision": "accept" },
                }));
            }
            Some("item/started") | Some("item/completed") => {
                items += 1;
                let is_completed = msg["method"] == "item/completed";
                let item: codex_shapes::ThreadItem =
                    parse_as("thread item", &msg["params"]["item"]);
                match item {
                    codex_shapes::ThreadItem::AgentMessage { id, text } => {
                        saw_agent_message = true;
                        assert!(!id.is_empty(), "an agentMessage must carry an id");
                        assert_eq!(text, "all done");
                    }
                    // Codex leaves these optional, so the mirror cannot require
                    // them — but gallium promises them, and without this the
                    // test would still pass on an item that had regressed to
                    // carrying no output at all. The mirror says what a codex
                    // client *requires*; this says what gallium *delivers*.
                    codex_shapes::ThreadItem::DynamicToolCall {
                        content_items,
                        success,
                        arguments,
                        ..
                    } if is_completed => {
                        saw_tool_output = true;
                        let blocks =
                            content_items.expect("a finished tool call must carry contentItems");
                        let codex_shapes::ContentItem::InputText { text } = &blocks[0] else {
                            panic!("tool output should be an inputText block");
                        };
                        assert!(text.contains("out.txt"), "got: {text}");
                        assert_eq!(success, Some(true));
                        // Repeated from the announcement, so the finished item
                        // describes itself rather than patching the one before.
                        assert_eq!(arguments["file_path"], target.to_str().unwrap());
                    }
                    _ => {}
                }
            }
            Some("turn/completed") => {
                let _: codex_shapes::Turn = parse_as("turn/completed", &msg["params"]["turn"]);
                break;
            }
            // The `turn/start` response, which is not a notification.
            None if msg["id"] == 3 => {
                let started: codex_shapes::TurnStartResponse =
                    parse_as("turn/start", &msg["result"]);
                assert!(matches!(
                    started.turn.status,
                    codex_shapes::TurnStatus::InProgress
                ));
            }
            _ => {}
        }
    }

    assert!(items >= 3, "expected the tool's two items and the message");
    assert!(
        saw_agent_message,
        "the turn's text never arrived as an item"
    );
    assert!(
        saw_tool_output,
        "no finished tool call carried its output in contentItems"
    );
    // `CAUTIOUS` asks about a workspace write, so the approval round trip — and
    // with it the params a client needs to attach the prompt to an item — is on
    // the path of an ordinary turn, not an exotic one.
    assert!(saw_approval, "the write ran without an approval");

    let _ = std::fs::remove_file(&target);
    drop(client);
    handle.join().unwrap();
}

/// A provider that refuses, so a turn ends the one way the scripted one cannot
/// produce: as a failure.
struct FailingProvider;

impl LlmProvider for FailingProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        anyhow::bail!("the model is on fire")
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        anyhow::bail!("the model is on fire")
    }
}

/// A failed turn ends as `turn/completed` with `status: "failed"`, carrying the
/// reason — codex's only vocabulary for an ending.
///
/// Gallium used to send `turn/failed`, a method codex does not define, so a
/// codex-native client watched a failed turn simply never end. The reason the
/// switch waited for its own change is that it cuts the other way for a client
/// reading the *method*: `../klein-cli` treated any `turn/completed` as success,
/// and would have reported every failure as a silent one. It reads the status as
/// of fpt/klein-cli#95.
#[test]
fn a_failed_turn_ends_as_completed_with_a_failed_status() {
    let server = AppServer::with_provider_factory(
        ServerConfig::default(),
        Box::new(|_cfg, _model| Ok(Box::new(FailingProvider) as Box<dyn LlmProvider>)),
    );
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
    }));

    loop {
        let msg = client.recv();
        assert_ne!(
            msg["method"], "turn/failed",
            "turn/failed is not a codex method: {msg}"
        );
        if msg["method"] == "turn/completed" {
            let turn = &msg["params"]["turn"];
            assert_eq!(turn["status"], "failed", "{msg}");
            // The reason travels with the ending. Without it a client can say a
            // turn failed and nothing about why, which is barely better than
            // the turn never ending at all.
            assert!(
                turn["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("on fire"),
                "the failure must carry its reason: {msg}"
            );
            // It is still a `Turn`, so a codex client parses this ending with
            // the same type as any other.
            let _: codex_shapes::Turn = parse_as("failed turn", turn);
            break;
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// A thread survives a failed turn and can run another.
///
/// The slot is released and the history rolled back on every ending, but the
/// failure path is the one where that is easiest to get wrong — and a client
/// that cannot retry after one bad turn has effectively lost the thread.
#[test]
fn a_thread_still_works_after_a_turn_fails() {
    let failing = AtomicUsize::new(0);
    let server = AppServer::with_provider_factory(
        ServerConfig::default(),
        Box::new(move |_cfg, _model| {
            // The first thread's provider fails; this factory is called once per
            // thread, so the same thread keeps the same one.
            let _ = failing.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FailingProvider) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);
    let thread_id = handshake(&client, json!([]));

    for id in [3, 4] {
        client.send(json!({
            "jsonrpc": "2.0", "id": id, "method": "turn/start",
            "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
        }));
        loop {
            let msg = client.recv();
            // The second turn must be accepted, not refused as "already
            // running" — which is what a slot leaked by the failure path looks
            // like from out here.
            if msg["id"] == id && msg["method"].is_null() {
                assert!(msg["error"].is_null(), "turn {id} was refused: {msg}");
            }
            if msg["method"] == "turn/completed" {
                assert_eq!(msg["params"]["turn"]["status"], "failed", "{msg}");
                break;
            }
        }
    }

    drop(client);
    handle.join().unwrap();
}

/// A client that sends no `cwd` gets the directory gallium itself was started
/// in. Right for a client that spawned gallium as a child; wrong for one that
/// dialed it over TCP, where "the server's directory" is a different machine's
/// idea of where to work — and either way the log line now says which it was.
#[test]
fn a_thread_without_a_client_cwd_falls_back_to_gallium_s_own() {
    let (client, handle) = start_server(scripted_server(vec![]));
    let started = thread_start_with(&client, json!({}));

    let expected = std::env::current_dir().unwrap();
    assert_eq!(
        started["result"]["cwd"].as_str(),
        Some(expected.to_string_lossy().as_ref())
    );

    drop(client);
    handle.join().unwrap();
}

/// An empty `cwd` is a client that has one and did not fill it in. Taken
/// literally it is a working directory no process can enter, so every tool in
/// the thread would fail with ENOENT and nothing would say why.
#[test]
fn an_empty_client_cwd_is_treated_as_none_rather_than_as_a_root() {
    let (client, handle) = start_server(scripted_server(vec![]));
    let started = thread_start_with(&client, json!({ "cwd": "" }));

    let expected = std::env::current_dir().unwrap();
    assert_eq!(
        started["result"]["cwd"].as_str(),
        Some(expected.to_string_lossy().as_ref()),
        "an empty cwd must not become the workspace root"
    );

    drop(client);
    handle.join().unwrap();
}

/// Over TCP the client's `cwd` is a path in *its* filesystem, which may name
/// nothing on the machine running the model. Refused at `thread/start`, where it
/// is one answer, rather than discovered one failing tool call at a time.
#[test]
fn a_cwd_that_does_not_exist_here_is_refused_at_thread_start() {
    let (client, handle) = start_server(scripted_server(vec![]));
    let started = thread_start_with(
        &client,
        json!({ "cwd": "/nowhere/this/machine/has/heard/of" }),
    );

    let message = started["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("expected a refusal, got {started}"));
    assert!(
        message.contains("/nowhere/this/machine/has/heard/of")
            && message.contains("not a directory"),
        "refused for the wrong reason: {message}"
    );

    drop(client);
    handle.join().unwrap();
}

/// Records the tool catalog the model is offered, so a test can assert on what
/// the model could even *reach* rather than on what it happened to call.
struct ToolCatalogProvider {
    seen: std::sync::Mutex<Vec<String>>,
}

impl LlmProvider for ToolCatalogProvider {
    fn chat(&self, _messages: &[ChatMessage]) -> anyhow::Result<String> {
        Ok("unused".to_string())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        *self.seen.lock().unwrap() = tools.iter().map(|t| t.name.clone()).collect();
        Ok(LlmResponse::Text {
            content: "ok".to_string(),
            reasoning: None,
            usage: None,
        })
    }
}

struct SharedCatalog(Arc<ToolCatalogProvider>);

impl LlmProvider for SharedCatalog {
    fn chat(&self, m: &[ChatMessage]) -> anyhow::Result<String> {
        self.0.chat(m)
    }
    fn supports_tools(&self) -> bool {
        true
    }
    fn chat_with_tools(
        &self,
        m: &[ChatMessage],
        t: &[ToolDefinition],
    ) -> anyhow::Result<LlmResponse> {
        self.0.chat_with_tools(m, t)
    }
}

/// A client that registers a tool named like a built-in means *its* tool.
///
/// `resolve` returns the first exact match, so registered behind the built-in
/// the client's `Bash` would never be reached — every call would run the command
/// on the machine hosting the model, which is the one machine the client did not
/// mean. The model would also be offered the name twice.
#[test]
fn a_client_tool_replaces_the_builtin_of_the_same_name() {
    let server = scripted_server(vec![
        LlmResponse::ToolCalls(
            vec![ToolCallInfo {
                id: "c1".to_string(),
                name: "Bash".to_string(),
                arguments: json!({"command": "pwd"}),
            }],
            None,
        ),
        LlmResponse::Text {
            content: "done".to_string(),
            reasoning: None,
            usage: None,
        },
    ]);
    let (client, handle) = start_server(server);
    let thread_id = handshake(
        &client,
        json!([{ "type": "function", "name": "Bash", "description": "the client's shell",
                 "inputSchema": {"type": "object"} }]),
    );

    client.send(json!({
        "jsonrpc": "2.0", "id": 3, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": "where am i?"}] },
    }));

    let mut called_the_client = false;
    loop {
        let msg = client.recv();
        if msg["method"] == "item/tool/call" && msg["id"].is_number() {
            assert_eq!(msg["params"]["tool"], "Bash");
            called_the_client = true;
            client.send(json!({
                "jsonrpc": "2.0", "id": msg["id"],
                "result": { "success": true,
                            "contentItems": [{"type": "inputText", "text": "/on/the/client"}] },
            }));
            continue;
        }
        if msg["method"] == "turn/completed" {
            assert_eq!(msg["params"]["turn"]["status"], "completed", "{msg}");
            break;
        }
    }
    assert!(
        called_the_client,
        "the built-in Bash answered a call the client had claimed"
    );

    drop(client);
    handle.join().unwrap();
}

/// With workspace tools off, nothing that touches this machine is offered at
/// all — the arrangement where gallium is the head and the client is the hands.
/// What remains is the two that touch no filesystem: in-memory task bookkeeping
/// and skill lookup, which reads prompt text.
#[test]
fn workspace_tools_off_leaves_the_model_only_what_touches_no_machine() {
    let provider = Arc::new(ToolCatalogProvider {
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let recorder = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            workspace_tools: false,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedCatalog(Arc::clone(&recorder))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);
    let thread_id = handshake(
        &client,
        json!([{ "type": "function", "name": "Bash", "description": "the client's shell",
                 "inputSchema": {"type": "object"} }]),
    );
    drive_turn(&client, 3, &thread_id, "hello");

    let offered = provider.seen.lock().unwrap().clone();
    for local in ["Read", "Write", "Edit", "MultiEdit", "Glob", "LS", "Grep"] {
        assert!(
            !offered.iter().any(|t| t == local),
            "{local} was offered though workspace tools are off: {offered:?}"
        );
    }
    assert!(
        offered.iter().any(|t| t == "Bash"),
        "the client's own Bash should still be offered: {offered:?}"
    );
    assert!(
        offered.iter().any(|t| t == "LookupSkill") && offered.iter().any(|t| t == "Tasks"),
        "skills and tasks touch no machine and should remain: {offered:?}"
    );

    drop(client);
    handle.join().unwrap();
}

/// The same `cwd` that is refused when gallium's own tools would use it is
/// *accepted* when they are switched off.
///
/// With the client holding the hands, its `cwd` is a path in its own
/// filesystem — a Mac's `/Users/...` named to a Linux GPU box is the intended
/// arrangement, not a mistake. Validating it here would refuse exactly the
/// configuration the split exists for.
#[test]
fn a_client_cwd_need_not_exist_here_when_the_client_holds_the_tools() {
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            workspace_tools: false,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedCatalog(Arc::new(ToolCatalogProvider {
                seen: std::sync::Mutex::new(Vec::new()),
            }))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);
    let started = thread_start_with(
        &client,
        json!({ "cwd": "/Users/someone/on-the-other-machine" }),
    );

    assert_eq!(
        started["result"]["cwd"].as_str(),
        Some("/Users/someone/on-the-other-machine"),
        "the client's path should be carried through, not refused: {started}"
    );

    drop(client);
    handle.join().unwrap();
}

/// The mirror of `a_threads_skills_are_loaded_and_catalogued_into_the_prompt`:
/// on a networked thread the same directories are **not** read.
///
/// A client on a socket names `cwd` and `skillPaths` in its own filesystem.
/// Dereferencing them here would be this host reading files the client chose,
/// with this user's privileges, and handing their contents back through the
/// prompt and `LookupSkill` — the local-file primitive the transport takes away,
/// arriving by another door.
#[test]
fn a_networked_thread_reads_no_skill_path_the_client_named() {
    let dir = std::env::temp_dir().join(format!("gallium_remote_skills_{}", std::process::id()));
    let workspace_skills = dir.join(".agents").join("skills");
    let named = dir.join("named");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&workspace_skills).unwrap();
    std::fs::create_dir_all(&named).unwrap();
    std::fs::write(
        workspace_skills.join("deploy.md"),
        "---\nname: deploy\ndescription: SECRET_FROM_THE_WORKSPACE\n---\nbody\n",
    )
    .unwrap();
    std::fs::write(
        named.join("other.md"),
        "---\nname: other\ndescription: SECRET_FROM_A_NAMED_PATH\n---\nbody\n",
    )
    .unwrap();

    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens: 0,
    });
    let recorder = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            // What `appserver::tcp::serve_listener` forces on every connection.
            workspace_tools: false,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&recorder))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);

    let started = thread_start_with(
        &client,
        json!({ "cwd": dir.to_string_lossy(),
                "skillPaths": [named.to_string_lossy()] }),
    );
    let thread_id = started["result"]["thread"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("thread.id in {started}"))
        .to_string();
    drive_turn(&client, 3, &thread_id, "go");

    let seen = provider.seen.lock().unwrap();
    let prompt: String = seen[0].iter().map(|m| m.content.clone()).collect();
    assert!(
        !prompt.contains("SECRET_FROM_THE_WORKSPACE"),
        "the client's cwd was read on the server: {prompt}"
    );
    assert!(
        !prompt.contains("SECRET_FROM_A_NAMED_PATH"),
        "a client-named skillPath was read on the server: {prompt}"
    );

    drop(seen);
    drop(client);
    handle.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stdio MCP server is a command line, and `register_mcp_servers` runs it
/// *here*. Honoring one named over a socket would hand whoever reached the port
/// arbitrary code execution as the user gallium runs as — the skill-path
/// problem one rung worse, since that door reads files and this one runs
/// programs.
#[test]
fn a_networked_thread_runs_no_mcp_server_the_client_named() {
    let marker = std::env::temp_dir().join(format!("gallium_mcp_spawned_{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);

    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens: 0,
    });
    let recorder = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            // What `appserver::tcp::serve_listener` forces on every connection.
            workspace_tools: false,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&recorder))) as Box<dyn LlmProvider>)
        }),
    );
    let (client, handle) = start_server(server);

    let started = thread_start_with(
        &client,
        json!({
            "cwd": "/tmp",
            "config": { "mcp_servers": { "evil": {
                "command": "sh",
                "args": ["-c", format!("touch {}", marker.display())],
            }}},
        }),
    );
    assert!(
        started["result"]["thread"]["id"].is_string(),
        "the thread should start, just without the server: {started}"
    );

    // Spawning is synchronous inside `thread/start`, so by the time it has
    // answered the file would exist if the server had been run.
    assert!(
        !marker.exists(),
        "a client-named MCP server was spawned on this host"
    );

    drop(client);
    handle.join().unwrap();
    let _ = std::fs::remove_file(&marker);
}
