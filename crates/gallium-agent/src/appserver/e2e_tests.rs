//! End-to-end exercise of a full turn over the wire.
//!
//! The interesting path is reentrant: while the client is blocked awaiting its
//! `turn/start` response, gallium sends it an `item/tool/call` request and blocks
//! awaiting *that*. Both sides must keep reading. These tests drive `serve()`
//! through in-memory pipes and play the client by hand.

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

fn recording_server(context_window: u32, input_tokens: u64) -> (AppServer, Arc<RecordingProvider>) {
    let provider = Arc::new(RecordingProvider {
        seen: std::sync::Mutex::new(Vec::new()),
        input_tokens,
    });
    let handle = Arc::clone(&provider);
    let server = AppServer::with_provider_factory(
        ServerConfig {
            max_iterations: Some(5),
            context_window,
            ..Default::default()
        },
        Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedRecorder(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }),
    );
    (server, handle)
}

/// Drive one turn to completion, draining the notifications it emits.
fn drive_turn(client: &ClientSide, id: u64, thread_id: &str, text: &str) {
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "turn/start",
        "params": { "threadId": thread_id, "input": [{"type": "text", "text": text}] },
    }));
    loop {
        let msg = client.recv();
        if msg["id"] == id && msg["method"].is_null() {
            assert!(msg["error"].is_null(), "turn failed: {msg}");
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
    started["result"]["threadId"]
        .as_str()
        .expect("threadId")
        .to_string()
}

/// Start an extra thread on an already-initialized connection, optionally naming
/// a model. Returns its threadId.
fn start_thread(client: &ClientSide, id: u64, model: Option<&str>) -> String {
    let mut params = json!({ "cwd": "/tmp" });
    if let Some(model) = model {
        params["model"] = json!(model);
    }
    client.send(json!({
        "jsonrpc": "2.0", "id": id, "method": "thread/start", "params": params,
    }));
    let started = client.recv();
    started["result"]["threadId"]
        .as_str()
        .unwrap_or_else(|| panic!("threadId in {started}"))
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
    let skills_dir = dir.join(".gallium").join("skills");
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
    let thread_id = client.recv()["result"]["threadId"]
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

    // item/completed(agentMessage), turn/completed, then the turn/start response.
    let mut saw_agent_message = false;
    let mut saw_turn_completed = false;
    loop {
        let msg = client.recv();
        match msg["method"].as_str() {
            Some("item/completed") => {
                if msg["params"]["item"]["type"] == "agentMessage" {
                    assert_eq!(msg["params"]["item"]["text"], "hello there");
                    saw_agent_message = true;
                }
            }
            Some("turn/completed") => saw_turn_completed = true,
            None => {
                assert_eq!(msg["id"], 3, "expected the turn/start response");
                assert!(msg["result"]["turnId"].is_string());
                break;
            }
            other => panic!("unexpected method {other:?}"),
        }
    }
    assert!(saw_agent_message && saw_turn_completed);

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
            assert!(msg["error"].is_null(), "turn failed: {msg}");
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

    let text = completed["result"].as_str().expect("a result string");
    assert!(text.contains("disk on fire"), "got: {text}");
    assert!(
        text.contains("Error executing tool 'memory'"),
        "got: {text}"
    );

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
        if msg["id"] == 3 && msg["method"].is_null() {
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
    let thread_id = client.recv()["result"]["threadId"]
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
            assert!(msg["error"].is_null(), "turn failed: {msg}");
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
    let thread_id = client.recv()["result"]["threadId"]
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
