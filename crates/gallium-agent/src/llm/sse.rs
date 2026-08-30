//! SSE decoding for the streaming OpenAI Responses API (`stream: true`).
//!
//! Split out of `llm.rs` to keep that file navigable. This is a **child module**
//! of `llm`, so it reads the private wire types (`ResponsesResponse`, …) and the
//! `OpenAiProvider` helper methods directly, without any of them having to widen
//! to `pub(crate)`.

use anyhow::Result;
use serde::Deserialize;

use super::ResponsesResponse;

/// One decoded SSE event from the Responses API stream. Every event's `data:`
/// line is a self-contained JSON object with a `type`; the fields below are the
/// union of what the events we act on carry, all optional.
#[derive(Debug, Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    event_type: String,
    /// `response.output_text.delta` — the answer fragment.
    #[serde(default)]
    delta: Option<String>,
    /// `response.completed` / `response.incomplete` / `response.failed` — the
    /// full response object, in `ResponsesResponse` shape.
    #[serde(default)]
    response: Option<serde_json::Value>,
    /// Top-level `error` event.
    #[serde(default)]
    message: Option<String>,
}

/// Parse an OpenAI Responses API SSE stream: forward visible answer deltas to
/// `on_delta`, and return the terminal `ResponsesResponse` the
/// `response.completed` (or `response.incomplete`) event carries.
///
/// Only `response.output_text.delta` is forwarded. Reasoning-summary deltas
/// (`response.reasoning_summary_text.delta`) and tool-call-argument deltas
/// (`response.function_call_arguments.delta`) are deliberately dropped —
/// `on_delta` feeds a user-facing progressive render, and the full reasoning
/// and tool calls are reconstructed from the terminal response by the caller.
pub(super) fn parse_sse_stream<R: std::io::BufRead>(
    reader: R,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ResponsesResponse> {
    let mut terminal: Option<ResponsesResponse> = None;

    for line in reader.lines() {
        let line = line?;
        // SSE frames are `field: value` lines terminated by a blank line; only
        // `data:` carries the payload. `event:`, `id:`, `:`-comments and the
        // blank separators are skipped.
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let event: SseEvent = match serde_json::from_str(data) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Skipping unparseable SSE event ({e}): {data}");
                continue;
            }
        };

        match event.event_type.as_str() {
            "response.output_text.delta" => {
                if let Some(ref d) = event.delta {
                    on_delta(d);
                }
            }
            "response.completed" | "response.incomplete" => {
                let value = event.response.ok_or_else(|| {
                    anyhow::anyhow!("{} event carried no response object", event.event_type)
                })?;
                terminal = Some(serde_json::from_value(value).map_err(|e| {
                    anyhow::anyhow!("Failed to parse {} response: {e}", event.event_type)
                })?);
            }
            "response.failed" | "error" => {
                let msg = event
                    .message
                    .or_else(|| {
                        event.response.as_ref().and_then(|r| {
                            r.get("error")
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .map(String::from)
                        })
                    })
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(anyhow::anyhow!("OpenAI streaming error: {msg}"));
            }
            _ => {}
        }
    }

    terminal
        .ok_or_else(|| anyhow::anyhow!("OpenAI SSE stream ended without a terminal response event"))
}

#[cfg(test)]
mod tests {
    use super::parse_sse_stream;
    use crate::llm::OpenAiProvider;

    /// A canned Responses API SSE stream: two text deltas, then
    /// `response.completed` carrying a `message` output. The deltas reach
    /// `on_delta`; the terminal response feeds the normal extractors.
    #[test]
    fn sse_stream_forwards_text_deltas_and_returns_completed_response() {
        let stream = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\",\"output\":[]}}\n",
            "\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"The capital \"}\n",
            "\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"is Paris.\"}\n",
            "\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"text\":\"The capital is Paris.\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n",
            "\n",
            "data: [DONE]\n",
        );

        let mut deltas = String::new();
        let response =
            parse_sse_stream(std::io::Cursor::new(stream), &mut |d| deltas.push_str(d)).unwrap();

        assert_eq!(deltas, "The capital is Paris.");
        assert_eq!(response.status, "completed");
        assert_eq!(
            OpenAiProvider::extract_text(&response.output).as_deref(),
            Some("The capital is Paris.")
        );
        let usage = OpenAiProvider::convert_usage(&response.usage).unwrap();
        assert_eq!(usage.total_tokens, 15);
    }

    /// Reasoning-summary and tool-call-argument deltas are not answer text and
    /// must not reach `on_delta`; the tool call is reconstructed from the
    /// terminal response instead.
    #[test]
    fn sse_stream_ignores_reasoning_and_tool_arg_deltas() {
        let stream = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"thinking about it\"}\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"path\\\":\"}\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\"}}\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"Read\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}]}}\n",
        );

        let mut deltas = String::new();
        let response =
            parse_sse_stream(std::io::Cursor::new(stream), &mut |d| deltas.push_str(d)).unwrap();

        assert!(deltas.is_empty(), "no visible answer text in this stream");
        let calls = OpenAiProvider::extract_tool_calls(&response.output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    /// A `response.failed` / top-level `error` event ends the stream with an
    /// error rather than a silent empty response.
    #[test]
    fn sse_stream_surfaces_a_failed_event() {
        let stream = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n",
            "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"model overloaded\"}}}\n",
        );
        let err = parse_sse_stream(std::io::Cursor::new(stream), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("model overloaded"), "{err}");
    }

    /// A stream that stops before any terminal event is an error, not an empty
    /// success.
    #[test]
    fn sse_stream_without_a_terminal_event_errors() {
        let stream = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n";
        let err = parse_sse_stream(std::io::Cursor::new(stream), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("without a terminal"), "{err}");
    }

    /// An `incomplete` terminal status is rejected by `require_complete` the
    /// same way the blocking path rejects it.
    #[test]
    fn sse_incomplete_terminal_status_is_rejected() {
        let stream = "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"output\":[],\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n";
        let response = parse_sse_stream(std::io::Cursor::new(stream), &mut |_| {}).unwrap();
        let err = OpenAiProvider::require_complete(response).unwrap_err();
        assert!(err.to_string().contains("max_output_tokens"), "{err}");
    }
}
