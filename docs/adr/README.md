# Architecture Decision Records

Decisions about gallium's *shape* — what it is responsible for and what it
refuses to be — as opposed to how a particular piece works. The how belongs in
`CLAUDE.md` and the topic docs; this directory is for the choices that would
otherwise be re-argued from scratch every few months, together with the reasons
that would be lost.

An ADR is written when a decision closes off an option someone will reasonably
propose again. It records the alternative *and why it was declined*, because a
record that only states the winner reads as arbitrary the next time the question
comes up.

| # | Title | Status |
|---|---|---|
| [0001](0001-prompt-purity-and-explicit-context.md) | Prompt purity: no implicit volatile context injection | Accepted |
| [0002](0002-no-chat-completions-api.md) | No Chat Completions API in gallium itself | Accepted |
| [0003](0003-model-profiles.md) | Model profiles: one compiled-in profile per model family | Accepted |
| [0004](0004-execution-traces-as-training-data.md) | Execution traces are the source of truth; training data is an export | Accepted |

Format: Status, Context, Decision, Consequences, Alternatives considered. Keep
them short enough to read in full, and amend rather than rewrite — a superseded
ADR stays, marked superseded, because the reasoning that was overturned is part
of why the new decision is right.
