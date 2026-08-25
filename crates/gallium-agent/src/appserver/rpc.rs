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
        Self {
            code: METHOD_NOT_FOUND,
            message: format!("unknown method '{method}'"),
        }
    }

    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self {
            code: INVALID_PARAMS,
            message: msg.into(),
        }
    }
}

/// Any agent failure surfaces to the client as an internal error.
impl From<AgentError> for RpcFault {
    fn from(e: AgentError) -> Self {
        Self {
            code: INTERNAL_ERROR,
            message: e.to_string(),
        }
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

        let msg =
            json!({ "jsonrpc": JSONRPC_VERSION, "id": id, "method": method, "params": params });
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

/// The longest single JSON-RPC line this server will read.
///
/// Over stdio the bound is a formality: the peer is the process that spawned
/// this one, and its bytes already arrive with its privileges. It is `--listen`
/// that makes the limit worth having — bytes off a socket are bytes from another
/// machine, and an unbounded read hands that machine the choice of how much
/// memory gallium uses, by sending no newline at all.
///
/// 8 MiB, the number klein already bounds its own side of the same connection at
/// (`rpc.DefaultMaxMessageBytes`), so neither end accepts a message the other
/// would have refused to send. Deliberately not conditional on the transport: a
/// JSON-RPC line anywhere near this size is pathological over stdio too, and a
/// limit that only applies to the dangerous path is one more thing to get wrong
/// when a third transport appears.
pub const MAX_MESSAGE_BYTES: usize = 8 << 20;

/// Read one line, refusing one that runs past `limit` bytes.
///
/// `Ok(None)` is end of input. A line that reaches the limit without a newline
/// ends the connection rather than being truncated into a message: the bytes
/// cannot parse as JSON anyway, and resynchronizing by discarding to the next
/// newline would let a peer that sends none keep this loop reading forever.
fn read_line_limited<R: BufRead>(reader: &mut R, limit: usize) -> std::io::Result<Option<String>> {
    // `limit + 1` is what separates a line that exactly fills the budget from
    // one that would have run past it: the extra byte is only ever read when
    // there was more to come.
    let mut buf = Vec::new();
    // UFCS so `take` consumes a reborrow of the reference rather than the reader
    // it points at: `reader.take(..)` resolves to `R::take` and moves it.
    let read = std::io::Read::take(&mut *reader, limit as u64 + 1).read_until(b'\n', &mut buf)?;

    if read == 0 {
        return Ok(None);
    }
    if !buf.ends_with(b"\n") && read > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("message exceeds the {limit}-byte line limit"),
        ));
    }

    // A final line the peer never terminated is still a message. `lines()`
    // yielded it, and dropping it here would change how stdio behaves for the
    // sake of a socket.
    if buf.ends_with(b"\n") {
        buf.pop();
        if buf.ends_with(b"\r") {
            buf.pop();
        }
    }

    // Invalid UTF-8 takes the same path a read error does, which is what
    // `lines()` did with it.
    String::from_utf8(buf)
        .map(Some)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Read messages until the input closes, dispatching each to `handler`.
///
/// Blocks. Returns once the client has hung up *and* every in-flight request has
/// been answered — otherwise a caller that exits on return would drop responses
/// for requests still being handled.
pub fn serve<R: BufRead>(mut reader: R, conn: Arc<Connection>, handler: Arc<dyn RequestHandler>) {
    let mut inflight: Vec<std::thread::JoinHandle<()>> = Vec::new();

    loop {
        // Reap handlers that have already answered, so a long session does not
        // accumulate handles for every request it ever served.
        inflight.retain(|h| !h.is_finished());

        let line = match read_line_limited(&mut reader, MAX_MESSAGE_BYTES) {
            Ok(Some(l)) => l,
            Ok(None) => break,
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

        let method = msg
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_string);
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
        fn handle_request(
            &self,
            _conn: &Arc<Connection>,
            method: &str,
            params: Value,
        ) -> HandlerResult {
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

        let got = conn
            .request("item/tool/call", json!({"tool": "t"}))
            .expect("response");
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

    /// A message that exactly fills the budget is a message, not an attack.
    #[test]
    fn a_line_at_the_limit_is_still_served() {
        let padding = "x".repeat(64);
        let line =
            format!(r#"{{"jsonrpc":"2.0","id":7,"method":"echo","params":{{"pad":"{padding}"}}}}"#);
        let limit = line.len();

        let mut reader = Cursor::new(format!("{line}\n"));
        let got = read_line_limited(&mut reader, limit)
            .expect("a line of exactly the limit is accepted")
            .expect("not end of input");
        assert_eq!(got, line);
    }

    /// One byte past it is not, and the reason is memory rather than parsing:
    /// the bytes are refused *while* being read, not after being buffered.
    #[test]
    fn a_line_over_the_limit_is_refused() {
        let mut reader = Cursor::new("y".repeat(64) + "\n");
        let err = read_line_limited(&mut reader, 32).expect_err("over the limit");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("line limit"), "got: {err}");
    }

    /// The case the limit exists for: a peer that sends bytes and no newline at
    /// all. Without a bound this read never returns and the buffer never stops
    /// growing.
    #[test]
    fn a_peer_that_sends_no_newline_is_cut_off() {
        let mut reader = Cursor::new("z".repeat(1024));
        let err = read_line_limited(&mut reader, 64).expect_err("no newline, over the limit");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// Behavior `lines()` had that the replacement has to keep: a final message
    /// the peer never terminated is still delivered, and `\r\n` is not part of it.
    #[test]
    fn unterminated_and_crlf_lines_read_as_lines() {
        let mut reader = Cursor::new("{\"a\":1}\r\n{\"b\":2}");
        assert_eq!(
            read_line_limited(&mut reader, MAX_MESSAGE_BYTES).unwrap(),
            Some("{\"a\":1}".to_string())
        );
        assert_eq!(
            read_line_limited(&mut reader, MAX_MESSAGE_BYTES).unwrap(),
            Some("{\"b\":2}".to_string())
        );
        assert_eq!(
            read_line_limited(&mut reader, MAX_MESSAGE_BYTES).unwrap(),
            None,
            "end of input"
        );
    }

    /// An over-long line ends the connection rather than being resynchronized
    /// past: the request behind it is never dispatched.
    #[test]
    fn an_over_long_line_closes_the_connection() {
        let sink = Sink::default();
        let conn = Connection::new(Box::new(sink.clone()));
        let flood = "w".repeat(MAX_MESSAGE_BYTES + 1);
        let input = format!(
            "{flood}\n{}\n",
            r#"{"jsonrpc":"2.0","id":7,"method":"echo","params":{"hi":1}}"#
        );

        serve(Cursor::new(input), Arc::clone(&conn), Arc::new(EchoHandler));
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(
            sink.lines().is_empty(),
            "nothing after the refused line should be served: {:?}",
            sink.lines()
        );
    }
}
