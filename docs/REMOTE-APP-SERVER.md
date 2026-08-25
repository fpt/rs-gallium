# The remote app-server

How `gallium app-server` works when the client is somewhere else: the transport,
the connection model, and — the part that shapes everything else — which machine
each piece of a turn runs on.

Current as of PR #160 (the TCP transport) and #161 (the head/hands split).

## The shape

```
Mac / Windows                                     GPU Linux
┌──────────────────────────────┐                  ┌──────────────────────────┐
│ klein-cli                    │                  │ gallium app-server       │
│                              │  item/tool/call  │                          │
│  dynamicTools:               │◄─────────────────│  model + ReAct loop      │
│   Read Write Edit Bash …     │                  │  Tasks, LookupSkill      │
│   blender.* kicad.*          │─────────────────►│                          │
│                              │   tool result    │  no filesystem tools     │
│  the project, the shell,     │                  │  no client MCP servers   │
│  the running applications    │  ◄── turn/* ──►  │  no client skill paths   │
└──────────────────────────────┘                  └──────────────────────────┘
        the hands                                        the head
```

Gallium is the thing that is expensive to move: multi-gigabyte weights, a GPU, a
warm KV cache. Everything a turn actually *touches* — files, processes,
applications with state — stays where the user is. `dynamicTools` is what carries
the second half, and over this transport it stops being a codex-compatibility
feature and becomes the split itself.

## Transport

### Why a stream

The app-server protocol is not request/response. Within one turn:

```
klein → gallium    turn/start
gallium → klein    item/started, item/completed, thread/tokenUsage/updated …
gallium → klein    item/tool/call            ← gallium originates, and blocks
klein → gallium    the tool result
gallium → klein    item/fileChange/requestApproval
klein → gallium    the decision
gallium → klein    turn/completed
```

Two of those go server→client and *wait*. `rpc.rs` already had the reader loop
and the pending-request table that requires, so a persistent bidirectional stream
is the shape the protocol already had. HTTP POST/SSE would have to reinvent the
reverse direction; it is the direction the whole feature is for.

### What changed to get there

Nothing in `rpc.rs`. `rpc::serve` reads any `BufRead` and writes any `Write`, so
`appserver/tcp.rs` hands it a `TcpStream` and a clone of the same socket where
`run_stdio` hands it stdin and stdout. Messages stay line-delimited JSON-RPC 2.0.

Per connection: Nagle is disabled (`set_nodelay`), because every message is one
line and the sender then blocks on a reply — coalescing is exactly wrong. The
socket is split with `try_clone` so the reader loop and the turn threads writing
notifications share one file description rather than a duplicated buffer.

### Configuring it

`--listen <host:port>`, and nothing else. This is the one setting gallium takes
from the command line and refuses to take from anywhere else, which is worth the
paragraph because the reasoning generalizes.

Every other setting configures the server a client *spawns* — the model, the
sampling, the skills — and a client that spawns gallium wants stdio, since it has
just wired up the pipes it intends to talk on. An address arriving from the
environment or from `~/.config/gallium/config.toml` could therefore only ever do
one thing: turn that spawned server into one that opens a socket and never reads
the stdin it was handed. The client is told nothing, because from its side
nothing failed — it waits for a reply that is not coming, and hangs until its own
timeout.

That is not hypothetical; it cost the klein side an afternoon while an `[agent]
listen` key still existed. The first fix was an escape hatch (`GALLIUM_LISTEN=`
meaning "no, stdio"), which works and leaves the hazard in place for whoever does
not know to reach for it. Removing every other source of the address makes the
failure unrepresentable instead. A config that still names `listen` is warned
about at startup rather than ignored, since serde skips unknown fields and the
symptom would otherwise be a machine that was listening yesterday and speaks
stdio to nobody today.

A listener that cannot bind **exits with the reason**. Falling back to stdio
would leave a process that looks alive with nothing able to reach it.

## Connection model

### One client at a time

The limit is the llama.cpp KV cache, not the protocol. The slot pool holds one
context by default (`GALLIUM_KV_CACHE_SLOTS`), and its whole value is that
iteration *N*'s prompt is a prefix of *N+1*'s — 11.62s of re-prefill turned into
0.16s on a 2.3k-token turn. Two conversations interleaving on one slot are not
prefixes of each other, so each turn evicts the other's tokens and both pay full
price. Serving one client keeps the property that makes the cache worth having.

Raising the slot count is the precondition for lifting this. A slot is a whole KV
cache, so the second costs as much memory as the first.

### The newest client wins

A new connection **displaces** the one being served rather than being refused.
This transport is reached from a laptop that sleeps and roams: a TCP connection
that died with the link looks alive to this process until the OS gives up on it,
so refusing would lock the user out of their own GPU box for as long as that
takes — on the reconnect meant to fix it. The displaced client gets a clean EOF.

Displacement is four steps, and the order is the correctness argument:

| # | Step | Why here |
|---|---|---|
| 1 | Close the old server's turn-admission gate | A `turn/start` already dispatched on its own handler thread has not registered yet; closing the gate first leaves it two outcomes and no third |
| 2 | Cancel its running turns | A turn runs on its own thread, so it is not among the handlers `serve()` joins; closing the socket does not reach it |
| 3 | Shut the socket down | Ends the reader loop and, through the dropped pending table, releases any turn blocked awaiting a tool result or approval from the client that just left |
| 4 | Hand those stopping turns to the replacement | Cancelling is not stopping; the replacement's first turn waits for them before it calls the model, so the two never share the provider |

Steps 1 and 2 are one call — `AppServer::cancel_turns` closes the gate, walks the
threads, and returns a `#[must_use] StoppingTurns`, which step 4 gives to the new
connection via `adopt_predecessor`. The split exists because step 3 has to happen
in between: a turn blocked in `Connection::request` is not released by its
cancellation token, only by the reader loop exiting, so waiting before the
shutdown would hang on precisely the turn most in need of stopping.

**Step 4 used to be `stopping.wait()`, on the accept loop.** That was right about
the invariant and wrong about where to enforce it. The replacement's socket had
been accepted but nothing was reading it, so a displaced turn inside a call that
cannot be interrupted — an OpenAI round trip completes; there is no interruption
point in it — left the reconnecting client connected to a server that answered
nothing. That is the lockout this whole design exists to prevent, one step
further back, and it fires in exactly the situation displacement was built for.

The invariant is not about the socket. It is that two turns must not talk to the
model, and share its KV slots, at once. So the replacement is served at once —
`initialize`, `thread/start`, and a `turn/start` that answers immediately as
always — and the waiting moves to where the model is: `Predecessor::settle`,
called by the turn worker before its first model call. A turn whose predecessor
has not stopped within `PREDECESSOR_GRACE` (60 seconds) ends as a failed
`turn/completed` naming that reason, rather than running beside it — the overlap
would quietly halve the cache reuse this transport exists for, and a client can
act on being told to try again.

The gate (`accepting_turns`) is read by `turn/start` *while it claims the
thread's turn slot*, under the same lock. Checking before the claim would leave
the same race one instruction narrower. `thread/start` reads it too: its handler
is not the reader loop either, so a displaced connection would otherwise go on
building a thread, and loading a model on the shared pool, for a client whose
socket is already shut.

Lock order: `Predecessor` (taken by a turn worker on its way to the model, never
while holding the others), then `accepting_turns`, then `threads`, then a
thread's `active_turn`.

Not waited for: an in-flight request handler on the old connection — a
`thread/start` still loading a GGUF. It touches no KV cache, and the provider
pool's lock already serializes it against the new client's first `thread/start`.

Registration is by id, so a connection ending normally deregisters only *itself*
and never the newer client that already took the slot.

### What connections share

One `AppServer` per connection; one `ProviderPool` between them.

Weights are the thing that must be shared: a reconnect must not reload multi-GB
of model, and with llama.cpp the pool also keeps the KV slots warm across the
drop, so the returning client's next prompt is still a prefix of what the slot
holds. Thread tables are the thing that must not be shared: ids are sequential,
so `thread_1` exists on every connection, and one table would let a client's
`threadId` name another client's conversation.

## Which machine runs what

This is the part that determines everything else, and the argument is privilege,
not convenience.

### The threat model

Three things are true of the socket whatever is on the other end: it has **no
authentication, no authorization, and no transport encryption**; anything that
can reach the port is a client; and gallium's own tools — `Bash`, `Write`, the
rest — run as the user *gallium* was started as.

Together they describe the arrangement this design exists to avoid, not the one
it implements. *Were* a listening server to offer those built-in tools, every
call would execute with that user's rights, on that user's files, at the request
of someone the process cannot identify. Gallium started by user A and dialed by
user B's klein is the case that settles it: same machine is not the same user, so
loopback earns no exception, and "127.0.0.1 means trust me" stops holding the day
someone puts an SSH tunnel in front of it.

**It does not offer them.** The next section is the rule that follows, and
[What this does not protect](#what-this-does-not-protect) is what reaching the
port *does* get an unauthenticated peer — which is real, and is not a shell.

### The rule

**A listening server executes nothing on behalf of a client, and reads no path a
client named.**

`appserver::tcp::serve_listener` sets `ServerConfig::workspace_tools = false` on
every connection — overwriting the field rather than reading it, so the invariant
lives in the function that creates the exposure and no configuration can undo it.
There is no env var and no config key for this. An earlier version made it a
default with an override for the same-machine daemon case; that traded the
privilege boundary for a convenience, which is the wrong way round.

### What a networked thread has, and what it refuses

| `thread/start` gives | Over stdio | Over a socket | Why |
|---|---|---|---|
| built-in workspace tools (`Read`, `Write`, `Edit`, `MultiEdit`, `Glob`, `LS`, `Grep`, `Bash`) | registered | **none** | They run here, as this user |
| `Tasks` | registered | registered | In-memory bookkeeping; touches no machine |
| `LookupSkill` | registered | registered | Returns prompt text from directories the *operator* configured |
| `dynamicTools` | registered | registered | Dispatched back to the client; run under whoever runs the client |
| `config.mcp_servers` | registered | **ignored, logged** | A stdio MCP server *is a command line*, spawned here |
| `skillPaths` | loaded | **ignored, logged** | Paths in the client's filesystem; reading them here is this host opening files it did not choose |
| `<cwd>/.claude/skills`, `<cwd>/.agents/skills` | loaded | **not read** | Same — `cwd` is client-supplied |
| `~/.config/gallium/skills`, `agent.skillPaths` | loaded | loaded | The operator chose these |

The three refusals are one rule found three times. Removing the built-in tools
shut the front door; the skill paths were a file-read primitive by another door,
returning contents through the prompt catalog and `LookupSkill`; the MCP servers
were the same door one rung worse, since that one reads files and this one runs
programs. Every remaining `thread/start` parameter has been checked against the
rule. `ThreadStartParams` is `cwd`, `model`, `developerInstructions`,
`approvalPolicy`, `dynamicTools`, `skillPaths`, and a free-form `config`: `model`
selects among the operator's own providers, `approvalPolicy` is policy,
`developerInstructions` is prompt text, `dynamicTools` are dispatched outward,
and the only key read out of `config` is `mcp_servers`. Codex's `sandbox` is not
deserialized at all. None of them names a local path or process — but `config` is
free-form, so a future key read out of it is a new door and needs this paragraph
re-run.

Both refusals are **logged, never silent**. The symptom otherwise is a model
quietly missing capabilities the client believes it has.

### `cwd` means something different on each side

| Sent | Over stdio | Over a socket |
|---|---|---|
| a path | The workspace root for gallium's tools. Refused at `thread/start` if it is not a directory here | The client's own path. Carried through and reported, never dereferenced |
| empty string | Treated as absent — taken literally it is a directory no process can enter, and every tool fails with ENOENT saying nothing | Same |
| absent | gallium's own working directory, which is right only for a client that spawned it | gallium's own working directory, which is right for nobody — and the log says so |

The last row is worth the emphasis, because it is how the cwd bug was found: a
client that sends no `cwd` gets the directory gallium was started in, and a `pwd`
through a client-side tool is the first thing that shows it. `thread/start` now
logs the workspace *and whether the client chose it*:

```
thread thread_1: workspace /…/rs-gallium — the client sent no cwd, so this is gallium's own working directory
thread thread_2: workspace /…/klein-cli (from the client's cwd)
thread thread_3: workspace /Users/… — the client's own path, where its tools run; nothing here reads it
```

### What this does *not* protect

Being explicit, because a boundary half-described is worse than none:

- **No authentication.** Anything that reaches the port can start threads and
  turns, spend GPU time, and read whatever the model will say.
- **The operator's skills are readable.** `LookupSkill` still serves
  `~/.config/gallium/skills` and the launch config's `agent.skillPaths`. Those
  are the operator's own files, inside the boundary as drawn — but a host serving
  a client whose user should not see them is the next question of this shape.
- **Prompts and model output are in the clear**, on the wire and in any trace the
  operator has enabled.
- **Resource use is unbounded.** One client at a time is the only limit; there is
  no quota on turns, tokens, or time.

The overlay network is what supplies the missing half. See *Deployment*.

## Tool dispatch

A client tool becomes a `RemoteTool` in the thread's registry. When the model
calls it:

```
LLM → react.rs → RemoteTool::call → conn.request("item/tool/call") → TCP
                                                                      ↓
                                        klein runs it on its own machine
                                                                      ↓
       ToolResult ← parse_tool_response ← the JSON-RPC response ← ────┘
```

`conn.request` blocks the turn thread while the reader thread keeps running; the
response arrives through the pending table. This is why the reader must never be
the thread that runs a turn.

**A client tool replaces a built-in of the same name.** `ToolRegistry::resolve`
returns the first exact name match, so a client `Bash` registered *behind* the
built-in one would never be called — the model would see the name twice and every
call would land on the local shell. `register_replacing` backs the `dynamicTools`
loop for that reason. It is deliberately not what `register` does: two MCP servers
offering one name is an ambiguity nobody resolved, and silently keeping the last
is how a call runs the wrong operation and reports success.

Names therefore want to match gallium's built-ins — `Read`, `Write`, `Edit`,
`MultiEdit`, `Glob`, `LS`, `Grep`, `Bash`, PascalCase — because models emit those
from habit and gallium's own prompts talk about them. Argument shapes matter as
much (`Bash` takes `command`, `Read` and `Write` take `file_path`); a mismatch
presents as a model that cannot use tools, not as a schema error.

Input schemas are passed through **verbatim** — `DynamicToolSpec::input_schema`
is a `Value` handed to the provider unchanged — so nested object and array
schemas survive. A client whose own renderer flattens them will hand the model
`{"type": "array"}` with no element shape, which fails the same way.

### Who asks for approval

This changes when the tools move, and it is worth knowing precisely.

Gallium's approvals are enforced **inside its own tool implementations**, through
`ToolSession::request_write` / `request_exec` / `request_github`, which consult
the `ApprovalBroker`. A tool gallium does not implement — an MCP tool, or a
`RemoteTool` — never reaches the broker. `RemoteTool`'s `ToolAnnotations::EXTERNAL`
is catalog metadata (it round-trips MCP hints and describes the tool to a client),
not an enforcement point.

So on a networked thread **gallium asks nothing**, because it does nothing. The
client holds the tools, and the client is the side whose files and processes are
at stake, so the client is where the question belongs. A client taking tools over
from gallium must therefore take the prompt over too, or an interactive user
silently loses the confirmation they used to get before a write or a command.

The consequence for `[agent.approvals]` on a listening gallium: it governs
nothing, because nothing it governs runs there.

## What the client must provide

A client on this transport is not optional infrastructure — without it the model
has no hands at all. The contract:

1. **`dynamicTools` on `thread/start`** covering everything the model should be
   able to do: file reads and writes, search, shell. With real JSON schemas.
2. **`cwd`**, its own working directory, which its tools act against.
3. **Its own MCP servers**, run beside itself, exposed as more `dynamicTools`.
   The persistent connection is an advantage here over a remote MCP server: the
   client can hold state between calls — an open document, a current selection, a
   CAD session — because it is a process that stays running, not a stateless
   endpoint.

   This is the item a client is most likely not to meet yet. Forwarding servers
   as `config.mcp_servers` is what a client written against a *spawned* gallium
   does, and it is the right thing there; over a socket those are ignored (see
   the table above), so a client that forwards them and assumes they ran has
   tools the model was never offered. klein does this today and knows it.
4. **Its own approval prompt**, per *Who asks for approval*.
5. **Nothing about the address.** A client that also spawns gallium for stdio
   elsewhere needs no precaution: `--listen` is the only way a socket is opened,
   and a spawned server never gets one by accident.

A client that connects and registers nothing gets a warning in gallium's log and
a model that can read nothing, write nothing and run nothing:

```
thread thread_1: this server lends no tools of its own and the client registered
no dynamicTools, so the thread can read nothing, write nothing and run nothing.
```

## Deployment

```
Mac                          Tailscale / WireGuard              GPU Linux
klein-cli  ─────────────────────────────────────────────────►  gallium :47821
```

Bind a loopback address (with an SSH tunnel) or a private overlay address, and
let the overlay do what gallium does not: encrypted transport, machine
authentication, NAT traversal, device ACLs, endpoint discovery. Re-implementing
any of that inside the app-server protocol would be the wrong place for it.

Binding elsewhere is **logged, not refused** — only the operator knows which
interface is the private one:

```
listening on 0.0.0.0:47821 — every interface, including public ones. gallium
app-server has no authentication: anything that can reach this port can start
turns, spend this machine's time, and read whatever the model and the operator's
skills will tell it. It cannot run tools here — a networked thread has none of
its own — but that is the only limit. Bind a loopback or private overlay
(Tailscale/WireGuard) address instead.
```

The wording is deliberately narrower than the first version, which said a peer
"can run tools as this user". That was true when it was written and false one
commit later. Overstating a risk trains an operator to discount the warning, and
the accurate version is alarming enough.

## Diagnostics

The questions this deployment actually raises, and the line that answers each:

| Question | Where |
|---|---|
| Which directory does this thread work in, and who chose it? | `thread N: workspace …` at `thread/start` |
| Did the client's tools land? | `Registered tool: X (source: Dynamic)`, and `Tool 'X' (Builtin) replaced by the client's 'X'` |
| Why does the model have no tools? | `this server lends no tools of its own and the client registered no dynamicTools` |
| Why did my MCP server / skills not appear? | `ignoring N MCP server(s) named by the client`, `ignoring N skillPaths from the client` |
| Who is connected, and what happened to the last one? | `client connected from …`, `new client from … displaces the one being served` |
| Is it even listening where I think? | `gallium app-server listening on tcp://…` |

Per-turn traces (`[agent.trace] dir`, `GALLIUM_TRACE=1`) record tool calls with
their arguments and results whichever side ran them, so a trace from a networked
turn is a record of what the *client* did on the model's behalf.

## Testing

The transport and the boundary are pinned by tests over real loopback sockets
(`appserver/tcp.rs`) and in-memory pipes (`appserver/e2e_tests.rs`):

| Test | Pins |
|---|---|
| `a_turn_over_tcp_calls_back_into_the_client_for_a_dynamic_tool` | The reverse direction: gallium's own request crosses the wire mid-turn and the answer returns on the same connection |
| `a_new_client_displaces_the_one_being_served` | The displaced client sees EOF; the model is not reloaded |
| `displacement_stops_the_turn_it_displaces` | All of it: the replacement is served while the old turn is stuck in the model, its turn does not reach the model until that one stops, and the displaced turn never calls again |
| `a_turn_waits_for_the_displaced_connections_turns` | `Predecessor` on both sides of the deadline: refused while the old turn runs, allowed once it stops |
| `a_displaced_server_admits_no_new_turns` | The admission gate |
| `a_displaced_server_starts_no_new_threads` | The same gate on `thread/start` |
| `a_reconnect_reuses_the_loaded_model_and_starts_its_own_threads` | One `ProviderPool`, separate thread tables |
| `a_listening_server_offers_no_tools_of_its_own` | The transport overrides a config that asks for local tools |
| `a_networked_thread_reads_no_skill_path_the_client_named` | Planted skills in `<cwd>` and in a named path are absent from the prompt |
| `a_networked_thread_runs_no_mcp_server_the_client_named` | Names `sh -c 'touch <marker>'` as its MCP server and asserts the marker never appears |
| `a_client_tool_replaces_the_builtin_of_the_same_name` | The call goes over the wire, not to the local shell |
| `a_client_cwd_need_not_exist_here_when_the_client_holds_the_tools` | A Mac path against a Linux box is not an error |

The last two boundary tests are shaped as the exploit rather than as an
assertion about structure: the MCP one would create a file if the door were
open, and the skills one plants recognizable strings and looks for them in the
prompt the model is handed. Both were confirmed to fail against the code before
the fix — a boundary test that has never failed is a claim, not a test.

## Known gaps

- **No authentication of any kind.** The overlay network is load-bearing. If
  gallium ever needs to stand on its own here, it needs an identity for the peer
  before it needs anything else.
- **One client at a time.** Lifting it means raising `GALLIUM_KV_CACHE_SLOTS`
  first, and then deciding what a `threadId` means across clients.
- **No `ToolLocation`.** Tools are `Builtin` / `Mcp` / `Dynamic`, which happens
  to coincide with "here" and "there" for one client. Several clients — a Mac
  with `blender.*`, a Windows box with `fusion.*` — would need a tool to know
  *which* peer it belongs to, and a turn to be routable to the right one.
- **Operator-configured MCP for the app-server.** `[[mcpServers]]` in the launch
  config is REPL-only, so a networked thread has no MCP at all — including
  servers the operator chose, which the boundary would permit.
- **Approvals are per-implementation, not per-call.** Because the broker is
  reached from inside gallium's own tools, MCP tools bypass it too. That is not
  hypothetical over stdio: a client that forwards its MCP servers as
  `config.mcp_servers` has them launched here and their tools called with no
  approval asked, on the same machine the client is on.
- **Traces do not record steering**, and a steered turn therefore replays without
  the input it actually had. Unrelated to this transport but visible through it.
- **No streaming.** There is no `item/agentMessage/delta`, so a user watching a
  remote turn sees nothing until a tool call or the final message — the same on
  stdio, more noticeable over a network.

## Related

- [ADR 0002](adr/0002-no-chat-completions-api.md) — why the thread/turn surface
  rather than a stateless `messages[]` endpoint, which has nowhere to say "this
  continues that conversation" and so nowhere for the KV cache to stand.
- [Optimization](OPTIMIZATION.md) — the KV slot pool this connection model
  protects.
- `CLAUDE.md`, *app-server protocol* — the method-by-method reference.
