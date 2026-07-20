//! Bidirectional line-delimited JSON-RPC 2.0 over a byte stream (stdio).
//!
//! Unlike `mcp_server.rs` (strict request → response), an agent app-server must
//! interleave three kinds of traffic on one connection: it answers client
//! requests, pushes notifications while a turn is running, and *originates*
//! requests of its own (dynamic tool calls, approvals) that the client answers.
//!
//! So the reader loop demultiplexes each line three ways:
//!   - `id` + `method` → a client request; dispatched on its own thread
//!   - `method`, no `id` → a client notification
//!   - `id`, no `method` → a response to one of *our* requests

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam::channel::{bounded, Sender};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::mcp::{JsonRpcError, INTERNAL_ERROR, INVALID_PARAMS, JSONRPC_VERSION, METHOD_NOT_FOUND};
use crate::AgentError;

/// A JSON-RPC fault: an error code paired with a message.
#[derive(Debug)]
pub struct RpcFault {
    pub code: i32,
    pub message: String,
}

impl RpcFault {
    pub fn method_not_found(method: &str) -> Self {
        Self { code: METHOD_NOT_FOUND, message: format!("unknown method '{method}'") }
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self { code: INVALID_PARAMS, message: msg.into() }
    }
}

/// Any agent failure surfaces to the client as an internal error.
impl From<AgentError> for RpcFault {
    fn from(e: AgentError) -> Self {
        Self { code: INTERNAL_ERROR, message: e.to_string() }
    }
}

/// How a handler answers an inbound request.
pub type HandlerResult = Result<Value, RpcFault>;

/// Services inbound traffic from the client. Implementations must be `Sync`:
/// requests are dispatched concurrently so a long-running `turn/start` cannot
/// block the reader — the turn needs the reader alive to receive the responses
/// to the tool-call requests it originates.
pub trait RequestHandler: Send + Sync + 'static {
    fn handle_request(&self, conn: &Arc<Connection>, method: &str, params: Value) -> HandlerResult;

    fn handle_notification(&self, _conn: &Arc<Connection>, _method: &str, _params: Value) {}
}

/// The writable half plus the table of requests we are awaiting answers to.
pub struct Connection {
    out: Mutex<Box<dyn Write + Send>>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, Sender<Result<Value, JsonRpcError>>>>,
}

impl Connection {
    pub fn new(out: Box<dyn Write + Send>) -> Arc<Self> {
        Arc::new(Self {
            out: Mutex::new(out),
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
        })
    }

    fn write_msg(&self, msg: &Value) -> Result<(), AgentError> {
        let line = serde_json::to_string(msg)
            .map_err(|e| AgentError::InternalError(format!("JSON serialize: {e}")))?;
        let mut out = self.out.lock();
        writeln!(out, "{line}")
            .and_then(|_| out.flush())
            .map_err(|e| AgentError::InternalError(format!("write to client: {e}")))
    }

    /// Push a notification (no response expected).
    pub fn notify(&self, method: &str, params: Value) {
        let msg = json!({ "jsonrpc": JSONRPC_VERSION, "method": method, "params": params });
        if let Err(e) = self.write_msg(&msg) {
            tracing::warn!("failed to send notification '{}': {}", method, e);
        }
    }

    /// Send a server→client request and block until the client answers.
    ///
    /// Called from a turn thread while the reader thread keeps running; the
    /// reader hands the response back through the pending table.
    pub fn request(&self, method: &str, params: Value) -> Result<Value, AgentError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = bounded(1);
        self.pending.lock().insert(id, tx);

        let msg = json!({ "jsonrpc": JSONRPC_VERSION, "id": id, "method": method, "params": params });
        if let Err(e) = self.write_msg(&msg) {
            self.pending.lock().remove(&id);
            return Err(e);
        }

        match rx.recv() {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(err)) => Err(AgentError::InternalError(format!(
                "client returned error for '{method}' ({}): {}",
                err.code, err.message
            ))),
            // The sender is dropped only when the reader loop exits, i.e. the
            // client closed the connection while we were waiting.
            Err(_) => Err(AgentError::InternalError(format!(
                "connection closed while awaiting response to '{method}'"
            ))),
        }
    }

    fn respond(&self, id: Value, result: Value) {
        let msg = json!({ "jsonrpc": JSONRPC_VERSION, "id": id, "result": result });
        if let Err(e) = self.write_msg(&msg) {
            tracing::warn!("failed to send response: {}", e);
        }
    }

    fn respond_error(&self, id: Value, code: i32, message: String) {
        let msg = json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": id,
            "error": { "code": code, "message": message },
        });
        if let Err(e) = self.write_msg(&msg) {
            tracing::warn!("failed to send error response: {}", e);
        }
    }

    /// Deliver a response to whichever `request()` call is awaiting this id.
    fn deliver_response(&self, id: &Value, result: Result<Value, JsonRpcError>) {
        let Some(key) = id.as_u64() else {
            tracing::warn!("response with non-numeric id {:?}", id);
            return;
        };
        match self.pending.lock().remove(&key) {
            Some(tx) => {
                let _ = tx.send(result);
            }
            None => tracing::warn!("response for unknown request id {}", key),
        }
    }

    /// Fail every in-flight request. Called when the reader loop ends so turn
    /// threads blocked in `request()` unblock instead of hanging forever.
    fn cancel_pending(&self) {
        self.pending.lock().clear();
    }
}

/// Read messages until the input closes, dispatching each to `handler`.
///
/// Blocks. Returns once the client has hung up *and* every in-flight request has
/// been answered — otherwise a caller that exits on return would drop responses
/// for requests still being handled.
pub fn serve<R: BufRead>(reader: R, conn: Arc<Connection>, handler: Arc<dyn RequestHandler>) {
    let mut inflight: Vec<std::thread::JoinHandle<()>> = Vec::new();

    for line in reader.lines() {
        // Reap handlers that have already answered, so a long session does not
        // accumulate handles for every request it ever served.
        inflight.retain(|h| !h.is_finished());

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("read error, closing connection: {}", e);
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("ignoring unparseable line: {}", e);
                continue;
            }
        };

        let method = msg.get("method").and_then(Value::as_str).map(str::to_string);
        let id = msg.get("id").cloned();

        match (method, id) {
            // A request from the client. Dispatch on its own thread: the handler
            // may take minutes (a full agent turn) and may itself call
            // `conn.request()`, whose response only arrives if we keep reading.
            (Some(method), Some(id)) => {
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                let conn = Arc::clone(&conn);
                let handler = Arc::clone(&handler);
                inflight.push(std::thread::spawn(move || {
                    match handler.handle_request(&conn, &method, params) {
                        Ok(result) => conn.respond(id, result),
                        Err(fault) => {
                            tracing::warn!("request '{}' failed: {}", method, fault.message);
                            conn.respond_error(id, fault.code, fault.message);
                        }
                    }
                }));
            }
            (Some(method), None) => {
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                handler.handle_notification(&conn, &method, params);
            }
            (None, Some(id)) => {
                let result = match msg.get("error") {
                    Some(err) => Err(serde_json::from_value(err.clone()).unwrap_or(JsonRpcError {
                        code: INTERNAL_ERROR,
                        message: "malformed error object".to_string(),
                        data: None,
                    })),
                    None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
                };
                conn.deliver_response(&id, result);
            }
            (None, None) => tracing::warn!("ignoring message with neither method nor id"),
        }
    }

    // Fail outstanding server→client requests first: a handler blocked in
    // `request()` is waiting on a client that has now hung up, and joining it
    // before unblocking it would deadlock.
    conn.cancel_pending();

    for handle in inflight {
        let _ = handle.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Collects everything written, so a test can assert on the wire bytes.
    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Sink {
        fn lines(&self) -> Vec<Value> {
            let bytes = self.0.lock().clone();
            String::from_utf8_lossy(&bytes)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).expect("valid JSON line"))
                .collect()
        }
    }

    struct EchoHandler;

    impl RequestHandler for EchoHandler {
        fn handle_request(&self, _conn: &Arc<Connection>, method: &str, params: Value) -> HandlerResult {
            match method {
                "echo" => Ok(params),
                _ => Err(RpcFault::method_not_found(method)),
            }
        }
    }

    #[test]
    fn dispatches_request_and_writes_response() {
        let sink = Sink::default();
        let conn = Connection::new(Box::new(sink.clone()));
        let input = r#"{"jsonrpc":"2.0","id":7,"method":"echo","params":{"hi":1}}"#;

        serve(Cursor::new(input), Arc::clone(&conn), Arc::new(EchoHandler));

        // The request is handled on a spawned thread; serve() returns as soon as
        // input is exhausted, so give the dispatch thread a moment to finish.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let msgs = sink.lines();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["id"], 7);
        assert_eq!(msgs[0]["result"]["hi"], 1);
    }

    #[test]
    fn unknown_method_yields_error_response() {
        let sink = Sink::default();
        let conn = Connection::new(Box::new(sink.clone()));
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"nope"}"#;

        serve(Cursor::new(input), Arc::clone(&conn), Arc::new(EchoHandler));
        std::thread::sleep(std::time::Duration::from_millis(100));

        let msgs = sink.lines();
        assert_eq!(msgs[0]["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn response_unblocks_a_pending_outbound_request() {
        let sink = Sink::default();
        let conn = Connection::new(Box::new(sink.clone()));

        // Answer id=1 — the id `request()` will allocate first. Delay the reader
        // so `request()` has registered the pending entry before the response
        // lands, which is the ordering a real client always produces.
        let input = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
        let reader_conn = Arc::clone(&conn);
        let reader = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            serve(Cursor::new(input), reader_conn, Arc::new(EchoHandler));
        });

        let got = conn.request("item/tool/call", json!({"tool": "t"})).expect("response");
        assert_eq!(got["ok"], true);
        reader.join().unwrap();
    }

    #[test]
    fn closed_connection_unblocks_pending_request() {
        let sink = Sink::default();
        let conn = Connection::new(Box::new(sink.clone()));

        // Empty input: the reader loop exits immediately and must cancel pending.
        let reader_conn = Arc::clone(&conn);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            serve(Cursor::new(""), reader_conn, Arc::new(EchoHandler));
        });

        let err = conn.request("item/tool/call", Value::Null).unwrap_err();
        assert!(err.to_string().contains("connection closed"), "got: {err}");
    }
}
