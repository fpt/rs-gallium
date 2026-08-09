# 0002 — No Chat Completions API in gallium itself

**Status:** Accepted, 2026-08-09
**Related:** #15 (agentclientprotocol declined), [ADR 0001](0001-prompt-purity-and-explicit-context.md)

## Context

Gallium is asked, periodically, to expose an OpenAI-compatible
`/v1/chat/completions` endpoint. It is a natural request: the model is loaded,
inference works, and every tool in the ecosystem speaks that protocol.

It is also a different kind of program. Gallium's two frontends sit on one
abstraction — *hand it a turn, and it runs as many model calls and tool calls as
that turn needs, then reports the result*. The REPL and the app-server are two
renderings of `runtime::run_turn`. A chat-completions endpoint is the opposite
shape: one stateless request, one model response, no tools executed, no
approvals, no turn.

```
        REPL ─┐
              ├─ ReAct runtime ─┬─ llama.cpp
 app-server ─┘                  └─ candle
```

Adding a third frontend that bypasses the runtime would blur the boundary
between what gallium *is* (an agent runtime) and what it *contains* (an
inference engine).

**KV cache reuse sharpens this.** Reuse belongs to a conversation with a
lifetime — a thread that owns its context, its cache, and its tool state.
`thread/start` and `turn/start` give gallium exactly that. Chat Completions is
defined as stateless: the client resends the full `messages[]` every time and
the server is not supposed to remember anything. Gallium's slot pool would
recover *some* of it by prefix matching, but the protocol has no place to say
"this is the continuation of that conversation", no way to scope a cache to a
session, and no signal when a conversation ends and its slot can be released.
The optimization direction and the protocol point opposite ways.

Compatibility is also not a small surface. Claiming the name creates
expectations about streaming semantics, tool-call shape, `finish_reason`, usage
accounting, `logprobs`, `response_format`, `seed`, `stop`, parallel tool calls,
multimodal message shape, model listing, and cancellation — plus the
undocumented behaviours clients depend on. A partial implementation is worse
than none: it fails in ways users read as gallium bugs.

And the events that matter for an agent — a turn starting, a tool executing, an
approval being requested, a file changing, an interruption, a turn completing —
have natural homes in the app-server's `item/*` notifications. In Chat
Completions they would have to be forced into `assistant` messages or stream
chunks.

## Decision

**Gallium does not implement a Chat Completions API.** Its scope is:

> An agent runtime with stateful turns, tools, approvals, and pluggable
> inference engines.

It is not an OpenAI-compatible LLM server.

If OpenAI-compatible access is needed, it belongs in a **separate process**:

```
OpenAI-compatible client → chat-completions adapter → gallium app-server
```

An adapter can be replaced or discarded without touching the thread/turn model,
and its compatibility debt stays outside the runtime.

This also confirms the existing engine boundary: `llama.cpp` and `candle` are
*inference engines behind* the ReAct runtime. Adding, swapping, or removing one
— including a remote backend — must not change the runtime above it.

Note gallium remains an OpenAI **client** (`OpenAiProvider`, the Responses API).
Consuming that protocol and serving it are unrelated decisions.

## Consequences

**Good.** One turn abstraction, and both frontends stay renderings of it. Thread
lifetime stays gallium's to own, which is what KV cache reuse, prefix lifetime,
context compaction and engine tuning all need. Effort goes to the agent runtime
rather than to protocol-compatibility corners. Agent-shaped events keep an
honest place to be reported.

**Bad.** Tools that speak only Chat Completions cannot point at gallium without
an adapter that does not exist yet, and writing one is on whoever needs it. This
will be re-proposed; that is what this record is for.

**Boundary.** This ADR rules out gallium *serving* Chat Completions. It says
nothing about which protocol a *client* may speak to gallium, nor about how
gallium talks to remote inference backends.

## Alternatives considered

**Implement a minimal Chat Completions endpoint** — text in, text out, no tools,
no streaming. Cheap to write. Rejected because the name is the problem: clients
assume the full contract, and the failures land on gallium. A minimal endpoint
under a gallium-specific name would be honest, but nothing wants that, which
says the demand is for compatibility rather than for the endpoint.

**Implement it fully.** Rejected on cost and on direction. It would take
substantial sustained effort from the agent runtime, and it would pull gallium
toward statelessness exactly as the roadmap moves toward conversation-scoped
caches.

**Adopt agentclientprotocol.com instead** (`session/new` / `session/prompt`).
Already declined in #15 (Option A: keep the surface small; no editor-integration
requirement justified a second wire format, and a second transport translating
the same `AgentEvent` stream stays open if one appears). Recorded here because it
is the same question wearing a different hat. Gallium serves the **codex
app-server protocol**; call it that.
