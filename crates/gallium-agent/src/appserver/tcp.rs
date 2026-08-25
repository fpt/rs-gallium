//! The same app-server, reached over TCP instead of stdio.
//!
//! The protocol does not change: line-delimited JSON-RPC on one persistent,
//! bidirectional connection, which is what `rpc::serve` already speaks against
//! any `BufRead`/`Write` pair. Only the pair changes — a `TcpStream` and its
//! clone instead of stdin and stdout.
//!
//! The reason it is a *stream* transport and not HTTP is the traffic: a turn
//! pushes `item/*` notifications, and mid-turn gallium *originates* requests
//! (`item/tool/call`, approvals) that the client answers. That is a peer
//! relationship, and a request/response protocol would have to reinvent the
//! reverse direction it already has here for free.
//!
//! What the reverse direction buys is the point of this transport: the model
//! runs on the machine with the GPU, and the client's own tools — its
//! filesystem, its running applications — keep running on the machine the user
//! is sitting at. `dynamicTools` stops being only a codex-compatibility
//! feature and becomes the split between the agent's head and its hands.
//!
//! **There is no authentication and no transport encryption.** Anything that
//! reaches the port can run tools with this process's privileges, so the
//! address to bind is a loopback or a private-overlay one (Tailscale,
//! WireGuard) and the overlay is what does the authenticating. Binding
//! anywhere else is logged as the warning it is.

use std::io::BufReader;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::appserver::rpc::{self, Connection};
use crate::appserver::server::{default_provider_factory, AppServer, ProviderPool, ServerConfig};

/// Serve the agent on `addr` until the process is stopped.
///
/// One thread per connection, and connections are independent: each gets its
/// own `AppServer`, so a `threadId` names a conversation on the connection that
/// started it and nowhere else. What they share is the `ProviderPool` — the
/// loaded weights, which is the whole reason to put the agent on the GPU box.
///
/// Returns only on a listener error; a *connection* error closes that
/// connection and leaves the listener up, since the client on the other end of
/// a dropped Tailscale link will reconnect.
pub fn run_tcp(addr: &str, config: ServerConfig) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?;
    warn_if_exposed(&local);
    tracing::info!("gallium app-server listening on tcp://{}", local);

    serve_listener(
        listener,
        config,
        ProviderPool::new(Box::new(default_provider_factory)),
    );
    Ok(())
}

/// The accept loop, over a listener and a pool the caller built — which is how a
/// test drives it on an ephemeral port with a scripted model behind it.
///
/// **One client at a time, and the newest one wins.** The limit is the llama.cpp
/// KV cache: the slot pool holds one context by default (`GALLIUM_KV_CACHE_SLOTS`),
/// and its whole value is that iteration *N*'s prompt is a prefix of *N+1*'s —
/// 11.62s of re-prefill turned into 0.16s. Two conversations interleaving on one
/// slot are not prefixes of each other, so each turn evicts the other's tokens
/// and both pay full price. Serving one client keeps the property that makes the
/// cache worth having.
///
/// The newest connection displacing the older one, rather than being refused, is
/// about how this transport is actually reached: over an overlay network from a
/// laptop that sleeps and roams. A TCP connection that died with the link is not
/// distinguishable from a live one until the OS gives up on it, so refusing would
/// lock the user out of their own GPU box for as long as that takes — on the
/// reconnect that was meant to fix it. The displaced client gets a clean EOF.
///
/// Displacement is **three steps, in this order**, and the order is the whole of
/// the correctness argument:
///
/// 1. **Cancel** the old connection's turns. Closing its socket does not: a turn
///    runs on its own thread (`turn/start` answers immediately), so it is not
///    among the handlers `serve()` joins on the way out, and it would otherwise
///    keep calling the model — for the rest of the turn, tool calls and all —
///    beside the replacement client's turn, on the shared provider and its KV
///    slots. That unbounded overlap is what the one-client rule exists to
///    prevent.
/// 2. **Shut the socket down**, which ends the old reader loop and, through the
///    dropped pending table, releases any turn blocked awaiting a tool result or
///    an approval from the client that just went away — the one case a
///    cancellation token cannot reach on its own.
/// 3. **Hand those turns to the replacement**, whose first turn waits for them
///    before it calls the model — so nothing else is talking to the model when
///    it does. This step used to be a `wait()` on *this* thread, which was
///    correct about the invariant and wrong about where to enforce it: the
///    replacement's socket had been accepted but nobody was reading it, so a
///    displaced turn inside a call with no interruption point (an OpenAI round
///    trip completes; it cannot be cut short) left the reconnecting client
///    holding a connection that answered nothing. That is the lockout this
///    whole design exists to prevent, one step further back. The invariant was
///    never about the socket — it is that two turns must not share the provider
///    and its KV slots — so it is enforced where the model is, in
///    `Predecessor`.
///
/// What is *not* waited for is an in-flight request handler on the old
/// connection — a `thread/start` still loading a GGUF, say. It touches no KV
/// cache, and the provider pool's own lock already serializes it against the new
/// client's first `thread/start`.
fn serve_listener(listener: TcpListener, mut config: ServerConfig, providers: Arc<ProviderPool>) {
    // **Not a setting.** Gallium's own tools run as the user gallium was started
    // as, and this socket has no authentication: whoever reaches the port gets
    // `Bash` with those privileges. A listening server therefore has no local
    // hands at all — everything that reads, writes, or executes belongs to the
    // client and runs under whoever is running that.
    //
    // Same machine is not the same user, so loopback earns no exception. Making
    // this configurable would mean an approval policy that has to reason about
    // which user a call is really acting for, across a boundary that carries no
    // identity — and the failure mode of getting it wrong is one account
    // executing commands as another.
    config.workspace_tools = false;

    let current: Arc<Mutex<Option<Serving>>> = Arc::new(Mutex::new(None));
    let next_id = AtomicU64::new(1);

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            // A failed accept is that connection's problem, not the listener's.
            Err(e) => {
                tracing::warn!("accept failed: {}", e);
                continue;
            }
        };
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let socket = match stream.try_clone() {
            Ok(handle) => handle,
            Err(e) => {
                tracing::warn!("{}: could not split socket: {}", peer, e);
                continue;
            }
        };
        let server = Arc::new(AppServer::with_pool(config.clone(), Arc::clone(&providers)));
        let id = next_id.fetch_add(1, Ordering::SeqCst);

        // Registering the replacement and inheriting what it displaced are one
        // critical section, because they are one fact: this connection is now
        // the one being served, *and* it owes a wait to the turns it stopped.
        //
        // Both happen on this thread, which today is the only one accepting, so
        // no third connection can observe the half-built state in between. The
        // lock makes that structural rather than incidental: seen apart, a
        // third connection could take this server's still-empty `Predecessor`,
        // and the wait it should have inherited would be handed to a connection
        // that is already dead — leaving the newest client's turn free to reach
        // the model beside a turn from two connections ago.
        //
        // Holding it across the teardown is safe *because* the wait moved: the
        // three steps below no longer block on anything (`cancel_turns` returns
        // a handle, `shutdown` is a syscall, `adopt_predecessor` is a store), so
        // a displaced connection's thread taking this same lock on its way out
        // waits microseconds. It would have been a deadlock hazard when step
        // three was `stopping.wait()`.
        {
            let mut held = current.lock();
            let displaced = held.replace(Serving {
                id,
                socket,
                server: Arc::clone(&server),
            });

            if let Some(old) = displaced {
                tracing::info!(
                    "new client from {} displaces the one being served: gallium \
                     app-server serves one at a time",
                    peer
                );
                let stopping = old.server.cancel_turns();
                let _ = old.socket.shutdown(Shutdown::Both);
                // Handed to the replacement instead of waited on here. Waiting
                // on this thread would hold the accepted socket unread until
                // the old turn stopped — see `Predecessor`.
                server.adopt_predecessor(stopping);
            }
        }

        let current = Arc::clone(&current);
        std::thread::spawn(move || {
            serve_connection(stream, peer, server);
            // Deregister, unless a newer client has already taken the slot —
            // shutting *that* one down on this one's way out is the bug this
            // id comparison exists to prevent.
            let mut held = current.lock();
            if held.as_ref().is_some_and(|held| held.id == id) {
                *held = None;
            }
        });
    }
}

/// The connection being served: enough of it to stop what it is doing.
struct Serving {
    id: u64,
    socket: TcpStream,
    server: Arc<AppServer>,
}

/// Run one client to completion on the calling thread.
fn serve_connection(stream: TcpStream, peer: String, server: Arc<AppServer>) {
    // Nagle would sit on a small notification waiting for more to send, which is
    // exactly wrong for a protocol whose messages are one line and whose sender
    // then blocks on the reply.
    if let Err(e) = stream.set_nodelay(true) {
        tracing::warn!("{}: could not disable Nagle: {}", peer, e);
    }

    // The two halves of one socket: `rpc::serve` reads on this thread while turn
    // threads write through the `Connection`, and a `TcpStream` clone shares the
    // underlying socket rather than duplicating a buffer.
    let reader = match stream.try_clone() {
        Ok(r) => BufReader::new(r),
        Err(e) => {
            tracing::warn!("{}: could not split socket: {}", peer, e);
            return;
        }
    };

    tracing::info!("gallium app-server: client connected from {}", peer);
    let conn = Connection::new(Box::new(stream));
    rpc::serve(reader, conn, server);
    tracing::info!("gallium app-server: {} disconnected", peer);
}

/// Say plainly what binding to a reachable address means, once, at startup.
///
/// Not a refusal: binding to a Tailscale address is the intended deployment and
/// only the operator knows which interface that is. But a server that answers
/// anyone should never end up on a public interface by a typo nobody was told
/// about.
///
/// The wording is deliberately narrower than it used to be. A networked thread
/// has no tools of its own, so reaching this port is not a shell on this
/// machine; it is the model, the machine's time, and whatever the operator's
/// skills say. Overstating that trains an operator to discount the warning, and
/// the accurate version is alarming enough.
fn warn_if_exposed(addr: &SocketAddr) {
    if addr.ip().is_loopback() {
        return;
    }
    if addr.ip().is_unspecified() {
        tracing::warn!(
            "listening on {} — every interface, including public ones. \
             gallium app-server has no authentication: anything that can reach \
             this port can start turns, spend this machine's time, and read \
             whatever the model and the operator's skills will tell it. It \
             cannot run tools here — a networked thread has none of its own — \
             but that is the only limit. Bind a loopback or private overlay \
             (Tailscale/WireGuard) address instead.",
            addr
        );
        return;
    }
    tracing::warn!(
        "listening on {} — reachable from the network. gallium app-server has \
         no authentication or transport encryption; the network it is on is the \
         only thing keeping other machines out.",
        addr
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crossbeam::channel::{unbounded, Receiver, Sender};
    use serde_json::{json, Value};

    use crate::appserver::rpc::RequestHandler;
    use crate::llm::{ChatMessage, LlmProvider, LlmResponse, ToolCallInfo, ToolDefinition};

    /// Plays a fixed list of responses, one per model call, shared by every
    /// connection so a test can also count how many providers were built.
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
            let i = self.calls.fetch_add(1, Ordering::SeqCst) % self.steps.len();
            Ok(match &self.steps[i] {
                LlmResponse::ToolCalls(calls, usage) => {
                    LlmResponse::ToolCalls(calls.clone(), usage.clone())
                }
                LlmResponse::Text {
                    content,
                    reasoning,
                    usage,
                } => LlmResponse::Text {
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    usage: usage.clone(),
                },
            })
        }
    }

    /// Lets one `ScriptedProvider` sit behind several `Box<dyn LlmProvider>`.
    struct Shared(Arc<ScriptedProvider>);

    impl LlmProvider for Shared {
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

    /// A listener on an ephemeral loopback port, with a scripted model behind
    /// it. Returns the address to connect to and how many providers have been
    /// built — the number that must stay 1 however many clients connect.
    fn scripted_listener(steps: Vec<LlmResponse>) -> (SocketAddr, Arc<AtomicUsize>) {
        let provider = Arc::new(ScriptedProvider {
            steps,
            calls: AtomicUsize::new(0),
        });
        let builds = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&builds);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let pool = ProviderPool::new(Box::new(move |_cfg, _model| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Shared(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }));
        let config = ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        };
        // Detached: the loop ends only with the process, which is what
        // `run_tcp` does too.
        std::thread::spawn(move || serve_listener(listener, config, pool));
        (addr, builds)
    }

    /// Announces every model call and blocks inside it until released, so a test
    /// can displace a client whose turn is provably mid-inference — and can then
    /// tell whether that turn ever called the model *again*.
    struct GatedProvider {
        steps: Vec<LlmResponse>,
        calls: Arc<AtomicUsize>,
        entered: Sender<usize>,
        release: Receiver<()>,
    }

    impl LlmProvider for GatedProvider {
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
            let _ = self.entered.send(i);
            let _ = self.release.recv();
            Ok(match &self.steps[i.min(self.steps.len() - 1)] {
                LlmResponse::ToolCalls(calls, usage) => {
                    LlmResponse::ToolCalls(calls.clone(), usage.clone())
                }
                LlmResponse::Text {
                    content,
                    reasoning,
                    usage,
                } => LlmResponse::Text {
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    usage: usage.clone(),
                },
            })
        }
    }

    struct SharedGate(Arc<GatedProvider>);

    impl LlmProvider for SharedGate {
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

    /// A listener whose model blocks in its first call. Returns the address, the
    /// signal that the call has been entered, the switch that releases it, and
    /// the running model-call count.
    fn gated_listener(
        steps: Vec<LlmResponse>,
    ) -> (SocketAddr, Receiver<usize>, Sender<()>, Arc<AtomicUsize>) {
        let (entered_tx, entered_rx) = unbounded();
        let (release_tx, release_rx) = unbounded();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(GatedProvider {
            steps,
            calls: Arc::clone(&calls),
            entered: entered_tx,
            release: release_rx,
        });

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let pool = ProviderPool::new(Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedGate(Arc::clone(&provider))) as Box<dyn LlmProvider>)
        }));
        let config = ServerConfig {
            max_iterations: Some(5),
            ..Default::default()
        };
        std::thread::spawn(move || serve_listener(listener, config, pool));
        (addr, entered_rx, release_tx, calls)
    }

    /// The test's end of one connection: a socket, read line by line.
    struct Client {
        out: TcpStream,
        lines: std::io::Lines<BufReader<TcpStream>>,
    }

    impl Client {
        fn connect(addr: SocketAddr) -> Self {
            let out = TcpStream::connect(addr).expect("connect");
            // A hang here is a deadlock, which is what these tests exist to
            // catch; without a timeout it would be a hung test run instead.
            out.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let lines = BufReader::new(out.try_clone().unwrap()).lines();
            Self { out, lines }
        }

        fn send(&mut self, msg: Value) {
            writeln!(self.out, "{msg}").expect("write to server");
            self.out.flush().unwrap();
        }

        fn recv(&mut self) -> Value {
            let line = self
                .lines
                .next()
                .expect("server did not close the connection")
                .expect("server produced a line within 5s");
            serde_json::from_str(&line).expect("server writes valid JSON")
        }

        /// Whether the server has closed this connection, draining anything it
        /// wrote first — a displaced client should see EOF, not a hang.
        fn closed(&mut self) -> bool {
            self.lines.next().is_none()
        }

        /// Read until the server hangs up, discarding whatever it wrote on the
        /// way. Panics rather than returning on a read error, since a timeout
        /// here is the hang these tests exist to catch.
        fn drain_until_eof(&mut self) {
            while let Some(line) = self.lines.next() {
                line.expect("server closed the connection within 5s");
            }
        }

        /// `initialize` + `thread/start`, returning the thread id.
        fn handshake(&mut self, dynamic_tools: Value) -> String {
            self.send(json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "clientInfo": {"name": "tcp-test"},
                            "capabilities": {"experimentalApi": true} },
            }));
            assert_eq!(self.recv()["id"], 1);

            self.send(json!({
                "jsonrpc": "2.0", "id": 2, "method": "thread/start",
                "params": { "cwd": "/tmp", "dynamicTools": dynamic_tools },
            }));
            let started = self.recv();
            started["result"]["thread"]["id"]
                .as_str()
                .unwrap_or_else(|| panic!("thread.id in {started}"))
                .to_string()
        }
    }

    fn memory_tool() -> Value {
        json!([{ "type": "function", "name": "memory", "description": "recall",
                 "inputSchema": {"type": "object"} }])
    }

    fn recall_then_answer() -> Vec<LlmResponse> {
        vec![
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
        ]
    }

    /// The whole point of the transport: gallium's *own* request reaches the
    /// client across the socket mid-turn, and the answer comes back on the same
    /// connection. This is the direction a request/response transport would
    /// have had to reinvent.
    #[test]
    fn a_turn_over_tcp_calls_back_into_the_client_for_a_dynamic_tool() {
        let (addr, _builds) = scripted_listener(recall_then_answer());
        let mut client = Client::connect(addr);
        let thread_id = client.handshake(memory_tool());

        client.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": thread_id, "input": [{"type": "text", "text": "when?"}] },
        }));

        let mut tool_call_seen = false;
        let mut final_text = None;
        loop {
            let msg = client.recv();

            if msg["method"] == "item/tool/call" && msg["id"].is_number() {
                assert_eq!(msg["params"]["tool"], "memory");
                assert_eq!(msg["params"]["arguments"]["query"], "birthday");
                assert_eq!(msg["params"]["threadId"], thread_id);
                tool_call_seen = true;
                client.send(json!({
                    "jsonrpc": "2.0", "id": msg["id"],
                    "result": { "success": true,
                                "contentItems": [{"type": "inputText", "text": "June 3"}] },
                }));
                continue;
            }

            if msg["method"] == "item/completed" && msg["params"]["item"]["type"] == "agentMessage"
            {
                final_text = msg["params"]["item"]["text"].as_str().map(str::to_string);
            }

            if msg["method"] == "turn/completed" {
                assert_eq!(
                    msg["params"]["turn"]["status"], "completed",
                    "turn did not complete: {msg}"
                );
                break;
            }
        }

        assert!(tool_call_seen, "gallium never called the client's tool");
        assert_eq!(final_text.as_deref(), Some("It is in June."));
    }

    /// A reconnect — a laptop that woke up — finds the model still loaded, and
    /// gets a thread namespace of its own: ids are per connection, so the new
    /// connection's first thread is `thread_1` again, naming its *own*
    /// conversation and not the previous client's.
    ///
    /// The weights staying loaded is the point of the shared `ProviderPool`, and
    /// with llama.cpp it is also what keeps the KV cache slots warm across the
    /// drop: the reconnecting client's next prompt is still a prefix of what the
    /// slot holds.
    #[test]
    fn a_reconnect_reuses_the_loaded_model_and_starts_its_own_threads() {
        let (addr, builds) = scripted_listener(vec![LlmResponse::Text {
            content: "ok".to_string(),
            reasoning: None,
            usage: None,
        }]);

        let mut first = Client::connect(addr);
        let first_thread = first.handshake(json!([]));
        drop(first);

        let mut second = Client::connect(addr);
        let second_thread = second.handshake(json!([]));

        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "the reconnecting client loaded the model again"
        );
        assert_eq!(
            first_thread, second_thread,
            "thread ids are per connection, so both start at the same one"
        );

        second.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": second_thread, "input": [{"type": "text", "text": "hi"}] },
        }));
        loop {
            let msg = second.recv();
            if msg["method"] == "turn/completed" {
                assert_eq!(msg["params"]["turn"]["status"], "completed");
                break;
            }
        }
    }

    /// One client at a time, and the newest wins: the older connection is shut
    /// down rather than the newer one refused, because a link that died with a
    /// sleeping laptop looks alive to this process until the OS gives up on it.
    #[test]
    fn a_new_client_displaces_the_one_being_served() {
        let (addr, builds) = scripted_listener(vec![LlmResponse::Text {
            content: "ok".to_string(),
            reasoning: None,
            usage: None,
        }]);

        let mut first = Client::connect(addr);
        first.handshake(json!([]));

        let mut second = Client::connect(addr);
        let thread_id = second.handshake(json!([]));

        assert!(
            first.closed(),
            "the displaced client should see EOF, not a hang"
        );
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "displacing a client must not reload the model"
        );

        // And the survivor is fully served, not merely connected.
        second.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": thread_id, "input": [{"type": "text", "text": "hi"}] },
        }));
        loop {
            let msg = second.recv();
            if msg["method"] == "turn/completed" {
                assert_eq!(msg["params"]["turn"]["status"], "completed");
                break;
            }
        }
    }

    /// Displacement stops the old client's turn — it does not merely close the
    /// socket under it.
    ///
    /// A turn runs on its own thread (`turn/start` answers immediately), so
    /// nothing about the socket closing reaches it: it would go on calling the
    /// model, for the rest of the turn, beside the replacement client's turn and
    /// on the same provider and KV slots. The script here has a second model call
    /// waiting after the tool call, and the proof is that it never happens.
    ///
    /// The ordering that makes this deterministic is the same one that makes it
    /// correct: turns are cancelled *before* the socket is shut down, so the EOF
    /// the displaced client sees is already proof the cancel landed.
    ///
    /// It also pins where the waiting happens. The replacement connects, hand-
    /// shakes and has its `turn/start` answered while the old turn is still
    /// inside its model call — none of that touches the model. Only the turn
    /// itself waits.
    #[test]
    fn displacement_stops_the_turn_it_displaces() {
        let (addr, entered, release, calls) = gated_listener(vec![
            LlmResponse::ToolCalls(
                vec![ToolCallInfo {
                    id: "c1".to_string(),
                    name: "LS".to_string(),
                    arguments: json!({"path": "."}),
                }],
                None,
            ),
            LlmResponse::Text {
                content: "should never be reached".to_string(),
                reasoning: None,
                usage: None,
            },
        ]);

        let mut first = Client::connect(addr);
        let thread_id = first.handshake(json!([]));
        first.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": thread_id, "input": [{"type": "text", "text": "go"}] },
        }));
        assert_eq!(
            entered.recv_timeout(Duration::from_secs(5)),
            Ok(0),
            "the turn should be inside its first model call"
        );

        // Displace it, from this thread: the replacement is served while the
        // displaced turn is still stuck in the model, which is the property the
        // next assertion is about.
        let mut second = Client::connect(addr);
        let second_thread = second.handshake(json!([]));
        assert_eq!(
            second_thread, "thread_1",
            "the replacement gets its own thread namespace"
        );

        // EOF means displacement has begun, and therefore that the cancel is
        // already set — before the model call is allowed to return.
        first.drain_until_eof();

        // The replacement is *connected* while the old turn is still inside a
        // model call it cannot be interrupted out of. It has to be: this is a
        // laptop that slept, and holding its socket unread until a cloud round
        // trip finishes is the lockout displacement exists to prevent.
        second.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": second_thread, "input": [{"type": "text", "text": "hi"}] },
        }));
        assert_eq!(
            second.recv()["id"],
            3,
            "turn/start must answer at once, as it does on any other turn"
        );

        // What it is *not* is talking to the model. Two turns on one provider
        // share its KV slots, and each evicts the other's tokens — so the
        // replacement's turn waits, and the model call count stays where the
        // displaced turn left it.
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the replacement reached the model while the displaced turn was still in it"
        );

        let _ = release.send(());

        // Now it runs — and the call that arrives is the replacement's, not the
        // displaced turn's second one. The displaced turn returns from the call
        // it was stuck in, meets its cancellation, and stops; the count at the
        // end is what tells the two apart.
        assert_eq!(
            entered.recv_timeout(Duration::from_secs(5)),
            Ok(1),
            "the replacement's turn never reached the model"
        );
        let _ = release.send(());

        loop {
            let msg = second.recv();
            if msg["method"] == "turn/completed" {
                assert_eq!(
                    msg["params"]["turn"]["status"], "completed",
                    "the replacement's turn did not complete: {msg}"
                );
                break;
            }
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "exactly one further model call, the replacement's"
        );

        // And the displaced turn is gone rather than merely quiet: its own
        // second call never happened, or the count above would be 3.
        assert!(
            entered.recv_timeout(Duration::from_millis(300)).is_err(),
            "the displaced turn called the model again"
        );
    }

    /// A listening server has no hands of its own, whatever it was configured
    /// with.
    ///
    /// Gallium's built-ins run as the user gallium was started as, and this
    /// socket carries no identity: whoever reaches the port would get `Bash`
    /// with those privileges. The client's `dynamicTools` are the only tools a
    /// networked thread gets, and they run under whoever is running the client.
    #[test]
    fn a_listening_server_offers_no_tools_of_its_own() {
        let provider = Arc::new(ToolCatalogProvider {
            seen: Mutex::new(Vec::new()),
        });
        let recorder = Arc::clone(&provider);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let pool = ProviderPool::new(Box::new(move |_cfg, _model| {
            Ok(Box::new(SharedCatalog(Arc::clone(&recorder))) as Box<dyn LlmProvider>)
        }));
        // Asked for local tools explicitly. The transport still refuses.
        let config = ServerConfig {
            max_iterations: Some(5),
            workspace_tools: true,
            ..Default::default()
        };
        std::thread::spawn(move || serve_listener(listener, config, pool));

        let mut client = Client::connect(addr);
        let thread_id = client.handshake(json!([{
            "type": "function", "name": "Bash", "description": "the client's shell",
            "inputSchema": {"type": "object"}
        }]));
        client.send(json!({
            "jsonrpc": "2.0", "id": 3, "method": "turn/start",
            "params": { "threadId": thread_id, "input": [{"type": "text", "text": "hi"}] },
        }));
        loop {
            let msg = client.recv();
            if msg["method"] == "turn/completed" {
                assert_eq!(msg["params"]["turn"]["status"], "completed", "{msg}");
                break;
            }
        }

        let offered = provider.seen.lock().clone();
        for local in ["Read", "Write", "Edit", "MultiEdit", "Glob", "LS", "Grep"] {
            assert!(
                !offered.iter().any(|t| t == local),
                "{local} was offered over a socket: {offered:?}"
            );
        }
        assert!(
            offered.iter().any(|t| t == "Bash"),
            "the client's own Bash should be there: {offered:?}"
        );
    }

    /// Records the catalog the model is offered, so a test can assert on what
    /// the model could reach rather than on what it happened to call.
    struct ToolCatalogProvider {
        seen: Mutex<Vec<String>>,
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
            *self.seen.lock() = tools.iter().map(|t| t.name.clone()).collect();
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

    /// The other half of stopping a displaced connection: it must not start
    /// anything *new* either.
    ///
    /// Cancelling walks the turns that are registered, and a `turn/start`
    /// dispatched on its own handler thread is admitted before it registers. If
    /// the two could interleave, that turn would register after the snapshot and
    /// run on beside the replacement — cancelled by nothing, waited for by
    /// nobody. Driving the handler directly is the only way to ask this
    /// question: over a socket the request would have to arrive after a shutdown
    /// that already ended the reader.
    #[test]
    fn a_displaced_server_admits_no_new_turns() {
        let provider = Arc::new(ScriptedProvider {
            steps: vec![LlmResponse::Text {
                content: "unreachable".to_string(),
                reasoning: None,
                usage: None,
            }],
            calls: AtomicUsize::new(0),
        });
        let server = Arc::new(AppServer::with_provider_factory(
            ServerConfig {
                max_iterations: Some(5),
                ..Default::default()
            },
            Box::new(move |_cfg, _model| {
                Ok(Box::new(Shared(Arc::clone(&provider))) as Box<dyn LlmProvider>)
            }),
        ));
        // Nothing reads what this connection writes; the answers come back from
        // `handle_request` directly.
        let conn = Connection::new(Box::new(std::io::sink()));

        let started = server
            .handle_request(&conn, "thread/start", json!({ "cwd": "/tmp" }))
            .expect("thread/start");
        let thread_id = started["thread"]["id"].as_str().expect("thread.id");

        server.cancel_turns().wait();

        let refused = server
            .handle_request(
                &conn,
                "turn/start",
                json!({ "threadId": thread_id,
                        "input": [{"type": "text", "text": "hi"}] }),
            )
            .expect_err("a displaced connection must not start a turn");
        assert!(
            refused.message.contains("displaced"),
            "refused for the wrong reason: {}",
            refused.message
        );
    }

    /// A displaced connection starts no *threads* either, not just no turns.
    ///
    /// Its reader loop is ending, but a `thread/start` already dispatched on its
    /// own handler thread is not — and that handler goes on to build a thread
    /// and load a model through the shared pool, for a client whose socket is
    /// already shut and whose answer nobody will read. Cheap to refuse at the
    /// same door `turn/start` reads. (rs-gallium#167)
    #[test]
    fn a_displaced_server_starts_no_new_threads() {
        let provider = Arc::new(ScriptedProvider {
            steps: vec![LlmResponse::Text {
                content: "unreachable".to_string(),
                reasoning: None,
                usage: None,
            }],
            calls: AtomicUsize::new(0),
        });
        let server = Arc::new(AppServer::with_provider_factory(
            ServerConfig {
                max_iterations: Some(5),
                ..Default::default()
            },
            Box::new(move |_cfg, _model| {
                Ok(Box::new(Shared(Arc::clone(&provider))) as Box<dyn LlmProvider>)
            }),
        ));
        let conn = Connection::new(Box::new(std::io::sink()));

        // Before displacement it is served, so the refusal below is about the
        // door and not about the request.
        server
            .handle_request(&conn, "thread/start", json!({ "cwd": "/tmp" }))
            .expect("thread/start");

        server.cancel_turns().wait();

        let refused = server
            .handle_request(&conn, "thread/start", json!({ "cwd": "/tmp" }))
            .expect_err("a displaced connection must not start a thread");
        assert!(
            refused.message.contains("displaced"),
            "refused for the wrong reason: {}",
            refused.message
        );
    }

    /// A client that hangs up takes its own connection down and nothing else:
    /// the next one — a laptop that woke up, a Tailscale link that dropped —
    /// still gets served.
    #[test]
    fn the_listener_outlives_a_disconnected_client() {
        let (addr, _builds) = scripted_listener(vec![LlmResponse::Text {
            content: "ok".to_string(),
            reasoning: None,
            usage: None,
        }]);

        let mut first = Client::connect(addr);
        first.handshake(json!([]));
        drop(first);

        let mut second = Client::connect(addr);
        assert!(!second.handshake(json!([])).is_empty());
    }
}
