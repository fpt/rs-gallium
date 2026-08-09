# 0001 — Prompt purity: no implicit volatile context injection

**Status:** Accepted, 2026-08-09
**Related:** #86 (KV cache reuse), [docs/OPTIMIZATION.md](../OPTIMIZATION.md)

## Context

`llm_local` now retains KV caches and evaluates only the suffix of a prompt that
diverges from what a slot already holds (#86). Iteration 2 of a measured turn
went from evaluating 2336 tokens to evaluating 29.

That win is **entirely a property of the prompt being a prefix of the next
one**. Reuse is computed as the longest common *token* prefix, so a single
changed token anywhere invalidates everything after it. A system prompt carrying
`Current time: 11:28:31` and then `Current time: 11:28:32` is, to the cache, a
different conversation — and because the injection sits near the *front* of the
prompt, one changed second discards the entire transcript behind it.

The cost is asymmetric in the worst way: the more valuable the cache (long
conversation, big fixed preamble), the more a volatile prefix throws away.

Two things make this a design question rather than a tuning detail.

**Freshness.** An injected `git status` is a snapshot from turn start. A model on
ReAct iteration 10 has no way to know it is stale and will reason about it as
current. A tool call returns the state *at the moment the model asked*, which is
the semantics an agent actually wants.

**Ownership.** Under the app-server, the client knows the workspace, the
selection, the git state and the IDE context precisely. Gallium re-deriving them
creates a second source of truth that can disagree with the first, and gallium's
is the one with less information.

### What gallium injects today

Better than feared, and worth recording before it drifts:

| Source | Volatile? | Notes |
|---|---|---|
| `HarmonyProtocol` (`protocol.rs`) | **Yes** — `Current date: {date}` from `current_date_ymd()` | Candle backend, GPT-OSS only. Changes once a day; invalidates every cache at midnight |
| jinja `strftime_now` stub (`llm_local.rs`) | No — returns `""` | Templates that print a date print nothing. Accidental purity, now deliberate |
| Skill catalog (`runtime.rs`) | No, in practice | Re-inserted each turn at the first non-system index and lifted out after. Same content at the same position, so the prefix survives — but it is inserted *mid-prompt*, so any change to it invalidates the whole conversation |
| `AGENTS.md`/`CLAUDE.md` (`project.rs`) | No | Read once at REPL start |
| Tool schemas | No | Computed once per tool at registration |
| Compaction (`memory.rs`) | **By design** | Drops messages from the middle, after the system block. Every prefix behind the drop point is invalidated |

So there is one live offender, one accidental save, and one structural
invalidator (compaction) that is doing its job.

## Decision

**Gallium does not silently inject volatile local state into the prompt.** The
prompt is a stable prefix followed by an append-only conversation:

```
Stable prefix                      Dynamic suffix
├ model/protocol system prompt     ├ user message
├ agent policy                     ├ assistant tool calls
├ tool definitions                 ├ tool results
└ skill catalog                    └ explicitly requested observations
```

Concretely:

1. **Volatile state reaches the model as a tool result or as client-supplied
   input, never as an injection.** Time, cwd, git state, terminal and
   environment details are *observations* the model asks for, or that the client
   chooses to send as part of a turn. They land at the end of the conversation,
   where they cost nothing.
2. **The same logical conversation produces the same token prefix.** This is the
   property to protect when touching prompt construction. A change that is
   correct but makes the prefix non-deterministic is a regression.
3. **Anything mid-prompt is treated as part of the stable prefix.** The skill
   catalog qualifies today. New material must go at the end unless it genuinely
   belongs to the prefix.
4. **Runtime metadata the model does not need stays inside gallium.** Not every
   fact gallium knows is prompt material.
5. **Under the app-server, the client owns environment context.** Gallium's
   responsibility shifts from *sensing* the environment to *reasoning
   efficiently over the environment it is given*.

The existing `Bash`, `LS`, `Grep` and `Read` tools already cover most of what
would otherwise be injected. A dedicated read-only `Environment` tool is a
reasonable convenience for the REPL and would satisfy this ADR; injecting the
same content into the system prompt would not.

The Harmony `Current date` line is the one known violation. It is inherited from
the model's canonical prompt rather than invented by gallium, which is an
argument for keeping it — but it is still a daily cache invalidation, and it
should be revisited rather than treated as settled.

### The cache identity that follows

A conversation's reusable prefix is a function of:

```
model revision · chat template · system prompt · tool schemas · skill set
```

and **explicitly not** of cwd, wall-clock time, or git state. Today reuse is
discovered by comparing tokens, which needs no such key; the list matters
because it says what a future block-level or persisted prefix cache would key
*on*, and what must never enter that key.

### Measurement

Reuse ratio should be recorded per call so a tuning trial can tell a fast
configuration from a lucky one — otherwise a trial that happened to hit a warm
cache looks like a good `ubatch` setting. `Timing::prefill_tokens` already
carries what was evaluated, against `TokenUsage::input_tokens` for the whole
prompt, so the ratio is derivable from a trace today. Naming it outright —
`prefix_tokens_reused`, `cache_reuse_ratio`, and an invalidation reason when the
prefix is dropped — is a small addition and is where this should go next.

## Consequences

**Good.** Reuse survives long conversations, which is where it pays most.
Volatile facts arrive fresh instead of stale-by-construction. The app-server
stops competing with its client over who knows the workspace. Traces replay to
the same token sequence, which is what makes `TurnTrace::diff` and any
benchmark comparison meaningful — implicit injection is directly opposed to the
replay design gallium already has.

**Bad.** A model must now *ask* for what it used to be handed, which costs an
iteration when it needs it, and a weaker model may not think to ask. That is a
real cost, paid where the information is actually used rather than on every turn
whether or not anyone reads it. Prompts also become the client's problem under
the app-server, so a client that supplies nothing gets an agent that knows
nothing about its environment until it looks.

**Watch for.** Compaction remains a deliberate prefix invalidator: it drops from
the middle, so the cache behind the drop point goes with it. That is the correct
trade — the alternative is running out of context — but it means a compacting
turn is expected to be slow, and that should not be mistaken for a cache bug.

## Alternatives considered

**Keep injecting, and accept the cache miss.** Simple, and it was the status quo.
Rejected because the miss is not small: a volatile token near the front of the
prompt discards the entire transcript behind it, which is precisely the case
#86 exists to fix.

**Inject volatile state at the *end* of the prompt instead of the front.** Keeps
the prefix intact and needs no model cooperation. Rejected as a general answer
because it does not fix staleness — the value is still a turn-start snapshot the
model will read as current on iteration 10 — and it re-injects on every
iteration, so the suffix never becomes cacheable either. It remains a reasonable
mechanism for something a client explicitly attaches to *one* turn, which is the
app-server path this ADR already allows.

**Hash volatile values into a cache key and re-inject only on change.** Preserves
both injection and most reuse. Rejected as complexity that buys nothing the tool
route does not: it still delivers stale values, and it makes prompt content
depend on cache state, which is exactly the coupling that makes traces
irreproducible.
