# 0003 — Model profiles: one compiled-in profile per model family

**Status:** Accepted, 2026-08-13
**Related:** #105 (MiniMax-M2.7 native tool calls), #116 (DeepSeek-V4 DSML tool calls), [ADR 0001](0001-prompt-purity-and-explicit-context.md)

## Context

Gallium's documented target models are four families. It now runs six: GPT-OSS,
Gemma 4, Qwen 3.6, LFM2.5, MiniMax-M2.7, DeepSeek-V4-Flash. Each arrived with its
own native tool-call wire format and its own way of marking reasoning, and each
was added by **widening shared code** rather than by adding a variant beside it.

Model-specific knowledge is now spread across five sites, none of which knows
which model is loaded:

| Site | What it holds |
|---|---|
| `llm_local::parse_tool_calls` | a cascade: JSON-prose → MiniMax → DSML → Harmony → Python-style → Gemma, first non-empty wins |
| `llm_local::is_native_tool_template` | substring sniffing of the GGUF's jinja template (`<minimax:tool_call>`, `｜DSML｜tool_calls`, `<\|channel\|>`, …) |
| `llm_local::clean_reply` | Harmony `final`, Gemma `<channel\|>`, `<think>`, and MiniMax's opener-less `</think>`, in a carefully ordered sequence |
| `llm_local::sample_until_done` | hardcoded Gemma stop literals (`<tool_call\|>`, `<\|tool_response>`) |
| `protocol.rs` + `llm_candle::Arch` | a **second, parallel dispatch** doing prompt rendering *and* parsing |

The cascade is the load-bearing problem. Every format is tried against every
model's output, leniently, in a fixed order — so each family's parser is in the
path of every other family's text. The order encodes real constraints (Harmony's
`final` channel must be read before Gemma's "everything after the last
`<channel|>`" heuristic, or the wrong slice is silently returned) but nothing
declares them, because there is no per-model scope in which to declare them.

The recent bug history is all one class:

| Fix | What went wrong |
|---|---|
| `26d0f80` | a stray `to=` inside Harmony *argument content* read as a tool-call boundary |
| `eb34344` | DSML's `string=` lookup unbounded, borrowing a later tag's attribute |
| `6f80ba8` / `16334fd` | MiniMax's opener-less `</think>` passed through; then a Unicode offset hazard in the fix |
| `8b96a70` | a dropped `<\|constrain\|>` tag left a bare value the GPT-OSS parser rejected |

Each is a parser being too permissive, or being permissive in the wrong other
model's output. The cost is not linear in the number of models: adding family N+1
puts a new lenient parser in front of all N existing families' text. And the
permissiveness is not a mistake — it is what makes an *unrecognized* GGUF work at
all. The defect is that there is no way to say which situation you are in.

Meanwhile the candle backend duplicates the same knowledge on a different axis.
`Arch::from_hint` selects a `ModelProtocol`, which renders prompts **and** parses
replies and tool calls. Two wire parsers (`crate::harmony`, `crate::gemma`) were
already extracted so both backends could share them — that extraction is the
shape this ADR generalizes — but JSON-prose, Python-style, MiniMax and DSML exist
only on the llama.cpp side. The divergence is not merely duplication: candle's
`parse_tool_call` returns `Option<(name, args)>`, **one** call, while MiniMax and
DSML both put several `<invoke>` blocks in one wrapper and llama.cpp returns them
all. The same model on two engines has different capabilities, decided by which
file its parser happened to land in.

## Decision

**One `ModelProfile` per model family, compiled into the binary, consumed by both
inference engines.**

A profile is a trait whose **default method bodies are the generic-model
behavior**; concrete profiles are unit structs overriding only what their family
does differently. That is the base-type/derived-type relationship, without
inheriting state nobody needs.

A profile owns everything that is a property of **the model** rather than of the
engine running it:

| Profile answers | Replaces |
|---|---|
| tool-call parsing | the cascade, and `ModelProtocol::parse_tool_call` |
| reply cleaning (reasoning/channel stripping) | `clean_reply`, `ModelProtocol::parse_response` |
| generation stop markers | the Gemma literals, `ModelProtocol::tool_stop_tokens` |
| protocol system preamble | `HarmonyProtocol`'s injected channel instructions |
| whether the GGUF template renders tools natively | `is_native_tool_template` |
| (candle only) full prompt rendering | `ModelProtocol` itself |

**`ModelProtocol` stops parsing.** It shrinks to prompt formatting and is renamed
`PromptRenderer`, hung off the profile that needs one. This is the consolidation:
prompt *rendering* legitimately differs per engine — llama.cpp has the GGUF's own
jinja template, candle has nothing and must render the format itself — while
everything on the **wire** is a property of the model and must not differ. After
this, one parser per family serves both engines, and a profile with no
`PromptRenderer` is refused **by name** on candle rather than silently handed a
prompt shape the model never saw in training.

**Profiles are code, not configuration.** A config selects one by name
(`[llm] profile = "deepseek-v4"`); it cannot define one. Selection is
`GALLIUM_PROFILE` > `[llm] profile` > auto-detection from what the loaded model
reports (`general.architecture` / `model_type`, the embedded chat template) >
`Generic`. Naming a profile that does not exist is an **error listing the valid
names** — never a silent fallback, the same rule `resolve_device` follows for an
absent device.

**`Generic` is the fallback and keeps today's permissiveness verbatim** — the full
cascade, in today's order, with today's reply-cleaning sequence. It matches
nothing during detection; it is only what detection falls back *to*. So an
unrecognized GGUF behaves exactly as it does now, and permissiveness stops being
the path every *known* model takes.

**A profile may carry a system prompt, and the two kinds are distinct.** A
*preamble* is protocol text the wire format requires (Harmony's channel
instructions) and is always prepended. A family *default* system prompt — the
behavioral nudge `configs/gemma4-system-prompt.md` is today — applies only when
the config names no `systemPromptPath`, because which persona the agent adopts is
the user's decision and burning one in would silently fight an explicit one. Both
must be static text: a preamble is part of the KV-cache prefix, so ADR 0001
applies to it in full. Harmony's `Current date:` line remains that ADR's one known
violation, now localized to the profile that emits it instead of the shared
prompt path.

Adding a model family becomes: one `profile/<family>.rs`, one line in the
registry, and a `profile =` key in its config. It touches no other family's
parsing.

### Landing order

Each step is green on its own.

1. **Extract the wire layer** behind `Generic`, no behavior change. *(landed)*
2. **Detection, the config key, and the six family profiles.** *(landed)*
3. **Move candle onto the shared parsers.** `ModelProtocol` loses
   `parse_response` / `parse_tool_call` / `tool_stop_tokens` and becomes
   `PromptRenderer`. This is where a candle-hosted model stops being limited to
   one tool call per reply.

   > **Amended during implementation.** Reading the code before writing any,
   > this step is a **port, not a move**: the profiles do not yet hold what
   > candle would lose, and three pieces of format knowledge have to migrate.
   >
   > `QwenProtocol::parse_tool_call` reads a native XML form —
   > `<tool_call><function=write><parameter=file_path>…</parameter></function>`
   > — that the `Qwen3` profile has no parser for. Step 2's `qwen3.rs` asserted
   > Qwen "claims no native format, deliberately" because it wraps JSON in
   > `<tool_call>` tags the balanced-span scan reads out of the middle. That is
   > true of the **llama.cpp** path only; on candle the model emits the
   > XML-parameter form, which the JSON fallback cannot read. Moving candle over
   > as-is would delete Qwen's tool calling on that engine.
   >
   > `GemmaProtocol::parse_response` strips trailing `<turn|>` / `<eos>` /
   > `<end_of_turn>`, which `Gemma4::clean_reply` does not — a no-op on
   > llama.cpp, where those are EOG tokens that never reach the text, but on
   > candle (`decode(&ids, false)` keeps specials) it would leave them in the
   > user-visible reply.
   >
   > And `GemmaProtocol::parse_tool_call` applies `crate::gemma`'s
   > `normalise_tool_name` / `normalise_path_args`, which that module documents
   > as **opt-in** — candle opts in, llama.cpp deliberately does not, so mixed-case
   > MCP tool names are not folded there. Unifying means choosing. The resolution
   > is to normalise **only when the verbatim name matches none of the offered
   > tools**: `parse_native_tool_calls` already receives `tools`, so aliasing
   > becomes a fallback rather than a rewrite, which removes the hazard of an MCP
   > tool named `write_file` being hijacked into `Write` and lets both engines
   > share one parser without either losing behavior.
   >
   > Two further corrections. `Arch` keeps selecting the renderer and now also
   > *names the profile* (`Arch::profile()`), rather than the profile carrying an
   > `Option<PromptRenderer>` as this ADR first described: `Arch::from_hint` is
   > already the established, total mapping from a model's hint to its family on
   > both the GGUF and safetensors paths, and re-deriving it from `DetectHints`
   > would risk a detection regression on candle for no gain (safetensors
   > `model_type` spellings are looser than the exact architecture names step 2
   > matches). An unsupported model is still refused at `Arch::from_hint`, which
   > is the same outcome by the existing mechanism. And there is **no
   > `system_preamble`**: Harmony's channel instructions are prompt *rendering*,
   > the one thing that legitimately differs per engine, so they stay in the
   > renderer. Adding a profile-level preamble would double up with what the GGUF
   > template already emits on llama.cpp — a trap, not an abstraction.
   >
   > **Verifiability is the real constraint, and it is poor.** Of candle's four
   > families only LFM2 runs on the development machine (`capital`, `file_read`
   > pass): Gemma 4 E4B is OOM-killed (`exit 137`, 24 GB), and neither Qwen 3.6
   > nor GPT-OSS is cached. Gemma has by far the most candle-specific leniency
   > and is the one that cannot be run. Since the migration targets are *shared*
   > parsers, a false positive in a ported parser also reaches the llama.cpp path
   > for that family. Hence the split below: 3-a is additive and unit-testable,
   > 3-b is the switch that needs a machine with the models.

3-a. **Give the profiles the format knowledge candle holds**, with tests
   transcribed from `protocol.rs`'s own — those are the current spec for the
   behavior being moved.

   > **Amended: behavior-preserving only.** The first cut of 3-a also *wired* the
   > ported knowledge in, which changed what the llama.cpp path does — Qwen's XML
   > calls began parsing where they had fallen through to a text reply, and
   > Gemma's tool-name aliasing started applying where names had been verbatim.
   > Both are arguably fixes, and that is the problem: a refactor that quietly
   > fixes things cannot be verified as a refactor, and the two changes land
   > safely only with a live run of the affected model behind them. So 3-a now
   > carries the parsers and the engine-difference handling that is a **no-op on
   > the existing path** — `wire::qwen_xml` present but unwired, the trailing
   > marker strip (those tokens are EOG on llama.cpp and never reach the string),
   > and LFM2's marker tolerance (inert when the markers are absent, which is
   > llama.cpp's case). Wiring, and the Gemma aliasing decision, become their own
   > changes after 3-b.

   Evidence for one of them arrived late and is worth recording: `unsloth/
   Qwen3.5-9B-GGUF` (`arch = "qwen35"`) has `<function=` / `<parameter=` in its
   embedded template and no `"name"` — so the XML form is what Qwen renders on
   **both** engines, and this ADR's step-2 claim that the family needs no native
   parser was wrong about llama.cpp as well, not just candle.

3-b. **Switch candle over** and reduce `ModelProtocol` to `PromptRenderer`.
   Verifiable on LFM2 only; candle + Gemma 4 / Qwen 3.6 / GPT-OSS need more RAM
   or a host with those models cached, and must be labelled unverified-by-run
   until then.

   > **Amended: the RAM ceiling was a property of one machine, not of candle.**
   > Landed and re-verified on a 121GB-RAM host, where Gemma 4 E4B — recorded
   > above as OOM-killed at 24GB — loads and runs to completion on CPU. Qwen
   > 3.6 and GPT-OSS remain unverified-by-run here (neither is cached), but
   > `gemma4-candle` now exists as a real backend (`configs/gemma4-candle.toml`,
   > `testsuite/backends.txt`) and is the first end-to-end run of Gemma 4
   > through candle this project has had. `ModelProtocol` shrank to
   > `PromptRenderer` (`format_prompt` / `format_prompt_with_tools` only);
   > `CandleProvider` now holds a `profile: &'static dyn ModelProfile` — the
   > same shared instance `llm_local.rs` selects for the identical
   > architecture — alongside the renderer, and `Arch::profile()` names it by a
   > direct match rather than a second `profile::detect` pass (candle's
   > four-family mapping has no ambiguity for that pass to resolve; an
   > unsupported model is still refused at `Arch::from_hint`, unchanged).
   > `tool_stop_tokens` is gone with the rest of `ModelProtocol`'s parsing
   > half — its early-stop job is now `profile.stops_generation()`, checked
   > per sampled token exactly as `llm_local.rs` already checks it, so the two
   > engines share the one mechanism instead of candle keeping a separate
   > EOS-token-id list.
   >
   > Running Gemma 4 through candle for the first time found two things a
   > refactor with no prior baseline to compare against can't tell apart on
   > its own — a parser bug and a model-quality question — recorded here
   > rather than blocking the switch on either:
   >
   > - `refactoring` sends `<|tool_call>call:LS{path:".}<tool_call|>` — an
   >   **ordinary** `"` where Gemma's format requires `<|"|>`. This is a real
   >   gap in `crate::gemma::scan_call_body` / `parse_kv_args`, **shared by
   >   both engines** since step 2: an ordinary quote that never closes
   >   "consumes the remainder" (deliberate, for a value that is genuinely
   >   unterminated), which here swallows the call's own closing `}` and the
   >   `<tool_call|>` marker into the argument value. It was reachable on
   >   llama.cpp in principle since the parser was shared, but never observed
   >   there — llama.cpp's real Gemma 4 output reliably uses `<|"|>`, so it
   >   took candle actually running to hit a model quoting it wrong.
   > - `needle_in_haystack` answers `FALCON-RIDGE-782}` for the needle
   >   `FALCON-RIDGE-7823` — one wrong final character, sampled directly by
   >   the model. Nothing in `clean_reply`'s marker-trimming can produce that
   >   from a correct answer, so this is upstream of the wire layer entirely —
   >   a `gemma4_q.rs` candle-implementation question (precision, a kernel
   >   difference from llama.cpp's quantized path, or plain model noise),
   >   outside this ADR's boundary. Recorded, not investigated further here.
   >
   > `lfm2-candle` re-verified unchanged (5/7, the same two failures as
   > #118) — the switch itself is not what those trace to.
3a. **Accept `{"ToolName": {args}}` in `wire::json`**, gated on the key naming a
   tool in the call's own `tools` list. Found by running LFM2.5: asked for a
   file write it answers with `{"Write": {"file_path": …, "content": …}}`, which
   `extract_calls` does not recognize — it looks for `name`/`arguments`, or
   `function`, or `tool_calls` — so the call is returned as a **text reply and
   printed to the user**. `json::parse_calls` currently ignores the `tools`
   argument it is handed, which is exactly what makes the shape safe to accept:
   without that gate, any single-key JSON object in a reply becomes a tool call.
   Sequenced after 3 because it changes a format shared with `Generic`, so it
   wants the engines already reading one parser. Necessary but likely not
   sufficient for that model: its `content` also carries `\\n` where `\n` was
   meant, so the file it writes would still not compile.

   > **Amended: this shape is narrower than what a live model actually sends.**
   > End-to-end testsuite verification of steps 1-2 on `lfm2`/`lfm2-candle`
   > (LFM2.5-8B-A1B-Q4_K_M, `coding`/`refactoring` testcases, deterministic
   > across repeats at the fixed sampler seed) never once reproduced a clean
   > `{"Write": {args}}` reply. Two different shapes came back instead, neither
   > of which this fix reaches:
   >
   > - `coding` sends the tool's **argument object with no name anywhere**:
   >   `{"file_path": "hello.go", "content": "…"}`. There is no key to gate on —
   >   recovering the tool would mean matching the key set against every offered
   >   tool's schema, a heuristic this bullet doesn't propose and which risks
   >   guessing wrong.
   > - `refactoring` sends the `{"ToolName": {args}}` shape this bullet targets
   >   (two of them, `Read` and `Edit` as sibling keys) but the `Edit` value is
   >   **not valid JSON** — a missing `}` before `file_path` leaves the object
   >   unclosed. `serde_json::from_str` fails before the shape-acceptance gate
   >   this bullet adds would ever run.
   >
   > So step 3a as scoped would fix a reply this model was not observed to send,
   > on this quant, across either testcase it currently fails. Confirmed this is
   > not a regression from steps 1-2: `Generic`'s cascade reads the same
   > `wire::json` logic either way, so both replies fail identically on `main`.
   > Left as a known gap rather than reworked here — the fix now needs either a
   > schema-matching fallback for the name-less case or tolerance for this
   > specific truncated-object pattern, both bigger than the gate this bullet
   > described.
4. **Retire the now-unreferenced globals** — `is_native_tool_template`, the
   hardcoded Gemma stop literals, `protocol.rs`'s parsing docs.
5. **Split tool calls out at the sampler**, as a field rather than a substring.
   The decoded string is lossy in two ways and this recovers both. Markers that
   are CONTROL tokens are **destroyed** by `special=false` (LFM2's
   `<|tool_call_start|>`, id 124905), which is why gallium sees only a bare
   `[Read(…)]` from that model. Markers that are USER_DEFINED **survive but
   become ambiguous** — Gemma 4's `<tool_call|>` (id 49) is indistinguishable
   from those characters appearing inside an argument value, so a `Write` whose
   content quotes Gemma's own markup is unparseable in principle at the string
   level and trivial at the token level. That is the same class as MiniMax's
   `</parameter>`-in-content, which means `wire::tags::value_boundaries` is an
   elaborate mitigation for a problem that does not exist upstream of the
   decoder.

   It also settles an engine divergence: candle decodes `.decode(&ids, false)`
   and **keeps** special tokens, while llama.cpp's `special=false` **drops**
   CONTROL ones, so the same model's output reaches the same shared parser as
   different bytes.

   The profile names the markers, each engine resolves them to token ids at load
   and falls back to string mode (loudly) when the vocab has no single token for
   one — so the ADR's boundary holds: model knowledge in the profile, mechanism
   in the engine. Two smaller wins ride along: Gemma 4's stop check currently
   runs `contains` over the whole accumulated string per sampled token (O(n²)
   across a long reply) and becomes an id comparison, and knowing prose from
   call-in-progress is a prerequisite for the streaming gallium still lacks
   against codex (`item/agentMessage/delta`).

   **Additive, not a replacement.** A model can emit a call with no markers at
   all — LFM2's `{"Write": …}` above is exactly that — so the string parsers
   stay. And a bounded region is not a bounded *argument*: it removes truncation
   past the region, not LFM2's paren problem inside it. Unverified for MiniMax
   and DeepSeek, whose markers are XML-ish text that may well not be single
   tokens; neither model is cached locally to check.

## Consequences

**Good.** A known model's output is parsed only by its own family's formats, so
the false-positive class above stops being reachable from a new model's arrival.
One parser per wire format serves both engines, so multi-call formats work on
candle for free and a fix lands once. Where a model's knowledge lives becomes a
single answer. The registry is static, so `PROFILES` can list itself and the
whole set is checkable by unit test against llama.cpp's own architecture table.

> **Amended during implementation (step 2).** This section first claimed that
> pinning `profile =` in the testsuite configs would make a detection regression
> "fail as wrong profile rather than as flaky tool calls". That is wrong: an
> explicit name *overrides* detection rather than checking it, so pinning every
> backend would have disabled the only end-to-end test of detection there is —
> the testsuite is the only place real GGUFs get loaded. The configs are
> deliberately left unpinned, and the diagnosis comes from `detect` logging the
> architecture it did **not** recognize, which names the failure directly. Note
> the exposure is small either way: a GGUF llama.cpp can load must report one of
> the architectures in its own dispatch table, which is where the detection
> strings come from.

**Bad.** A new model needs a profile before it works well, where today it gets the
cascade and often stumbles through; a *misdetected* model is worse off than an
undetected one, which is why an explicit name overrides detection and why an
unknown name is an error. Detection reads model-supplied strings, so it is a trust
boundary — a hostile GGUF could name an architecture it is not, and the blast
radius is a wrong parser, not code execution. Six unit structs is more surface
than one function, and the trait's default bodies must stay genuinely generic or
they become a sixth place model-specific knowledge hides.

**Boundary.** This ADR governs the **wire layer**: prompts out, text in, per
model. It says nothing about which weights loader runs (`Arch` stays; it selects a
model implementation, a different axis), nor about the provider/engine split ADR
0002 fixed. A profile is not a place to put engine tuning — `gpuLayers`, `cpuMoe`,
quant choice and context size stay in config, because they are properties of the
machine, not the family.

## Alternatives considered

**Keep the cascade and add tests.** The cheapest option, and it is what the four
fixes above already did. Rejected because tests cannot cover the interaction that
breaks: the failure is one family's parser reading another family's output, so the
case that matters is the *pair*, and the pairs grow quadratically while each new
model only ever ships tests for itself. The `to=` and `string=` bugs were both
found by a live model, not by a test that could have been written in advance.

**Profiles defined in TOML** — regexes, tag literals and template snippets as
config, so a new model needs no rebuild. Genuinely attractive, and rejected on the
evidence in this ADR: the parsers are not patterns but algorithms with hard-won
boundary rules (`value_boundaries`' bounded-window `rfind`, the byte-offset
handling behind `16334fd`). Expressing those in config would recreate the
permissiveness problem in a language with no tests and no types, and the failure
mode — a silently truncated `MultiEdit` payload — is exactly the one that is
hardest to notice. Config selects; code decides.

**An enum instead of trait objects.** `enum ProfileId` with a `match` per
behavior. Rejected because it inverts the growth: every match site grows with each
family, which is the shape being escaped. Trait objects put a family's answers in
one file, and the crate already dispatches this way (`Tool`, `ModelProtocol`,
`LlmProvider`).

**One profile per model rather than per family.** Rejected: Gemma 4 E4B and 26B-A4B
differ in projector and audio support but share the wire format exactly, so
per-model profiles would duplicate parsers to express a difference that belongs in
config. Where a *single* model genuinely diverges, that is a profile of its own —
the registry does not care that its neighbours are families.

**Let llama.cpp parse tool calls.** Its OAI-compat chat layer did this until
llama-cpp-2 0.1.150 removed it (`parse_response_oaicompat`), which is why gallium
parses at all. Rejected as unavailable rather than as wrong, and it would not help
candle, which has no such layer.

**Detect from the chat template only**, not from `general.architecture`. Rejected
because the template is absent on some GGUFs and candle has none at all, so a
template-only rule cannot serve both engines. Detection therefore takes several
hints and each profile decides which of them identify it.

**Decode with `special=true` instead of step 5's sampler split** — one flag
rather than a state machine, and every marker then reaches the parsers as text.
Rejected as not a drop-in, and as not solving the harder half. LFM2's reply would
become `<|tool_call_start|>[Read(…)]<|tool_call_end|>`, which fails
`wire::python`'s gate that the whole reply be a bracketed call list, so the flag
alone breaks the model it was meant to help; every parser would need updating in
the same change. And it leaves marker-versus-content ambiguity exactly where it
is, since a marker rendered as text is still just text. It remains the right
fallback for a family whose markers are not single tokens.

**Weigh both hints equally**, in one pass down the registry, first match wins.
This is what step 2 implemented first and it is wrong. Architecture names come
from llama.cpp's own dispatch table, so they are exact; template literals are
whatever a family's format happens to spell, and some are loose — Gemma 4 is
identifiable by `declaration:`, an ordinary word with a colon. In one pass, that
loose *template* hit from a profile early in the registry outranks an exact
*architecture* hit from a profile later in it, so a DeepSeek-V4 model whose
template merely contains the word "declaration:" is parsed as a Gemma. Detection
is therefore two passes: **every** profile is asked about the architecture before
**any** is asked about the template, and the template pass is the rescue for a
model whose architecture nobody here recognizes.
