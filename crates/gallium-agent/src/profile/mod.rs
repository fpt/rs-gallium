//! Model profiles: what gallium knows about one model family's wire behavior.
//!
//! A profile answers the questions that are properties of **the model** rather
//! than of the engine running it — how it writes a tool call, how it marks its
//! reasoning, where generation must stop — so that one answer serves both local
//! backends. See [ADR 0003](../../../docs/adr/0003-model-profiles.md) for why
//! this exists and what it replaces.
//!
//! [`ModelProfile`]'s default method bodies **are** the generic-model behavior:
//! a concrete profile is a unit struct overriding only what its family does
//! differently. That is the base/derived relationship without inherited state.
//!
//! Profiles are compiled in. A config selects one by name; it cannot define one,
//! because the parsers are algorithms with boundary rules (see
//! [`wire::tags::value_boundaries`]) rather than patterns a config could carry.
//!
//! ```text
//! GALLIUM_PROFILE  >  [llm] profile  >  detect(DetectHints)  >  Generic
//! ```
//!
//! Six families are compiled in — [`GptOss`], [`Gemma4`], [`Qwen3`], [`Lfm2`],
//! [`MiniMaxM2`], [`DeepSeekV4`] — plus [`Generic`], which every unrecognized
//! model falls back to and which keeps the permissive
//! try-everything behavior gallium has always had. [`GptOss20b`] is not a
//! seventh family: same wire format as [`GptOss`], explicit-name-only, opted
//! out of the one thing that turned out to be per-checkpoint rather than
//! per-family (see its own doc comment).
//!
//! The candle backend still has its own `protocol.rs` dispatch and does not
//! consult profiles yet; that is the next step in the ADR.

pub mod wire;

mod deepseek;
mod gemma4;
mod generic;
mod gpt_oss;
mod lfm2;
mod minimax;
mod qwen3;

pub use deepseek::DeepSeekV4;
pub use gemma4::Gemma4;
pub use generic::Generic;
pub use gpt_oss::{GptOss, GptOss20b};
pub use lfm2::Lfm2;
pub use minimax::MiniMaxM2;
pub use qwen3::Qwen3;

use std::borrow::Cow;

use anyhow::Result;

use crate::llm::{ToolCallInfo, ToolDefinition};

/// What a backend can tell a profile about the model it just loaded.
///
/// Every field is optional because the two engines know different things: the
/// llama.cpp path has the GGUF's metadata and its embedded chat template, while
/// candle has a `config.json` `model_type` and no template at all. A profile
/// decides for itself which of these identify it.
///
/// These strings come **from the model file**, so they are model-supplied input:
/// a mislabeled GGUF gets the wrong parser, which is why an explicitly named
/// profile always wins over detection.
#[derive(Debug, Default, Clone, Copy)]
pub struct DetectHints<'a> {
    /// GGUF `general.architecture`, or safetensors `model_type`.
    pub arch: Option<&'a str>,
    /// The chat template embedded in the GGUF, if it has one.
    pub chat_template: Option<&'a str>,
    /// The model path or `hf:` spec it was loaded from — a last-resort hint.
    pub model_id: Option<&'a str>,
}

/// A portable reasoning-effort level, mapped by each profile onto whatever
/// its family's own chat template actually understands
/// (`ModelProfile::reasoning_params`). `Max` means "no gallium-imposed
/// ceiling" — a profile maps it to its family's own highest level, not to
/// an unbounded literal value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    /// Case-insensitive, matching the leniency [`by_name`] already applies
    /// to profile names. `None` for anything unrecognized — this type has
    /// no opinion on what to do with that (e.g. OpenAI-only values like
    /// `"minimal"`); the caller decides.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "x-high" => Some(Self::XHigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// What a profile wants merged into the chat-template render context for a
/// given [`ReasoningEffort`]. Both fields are independently optional because
/// the families disagree about which axis they expose: GPT-OSS only has
/// `effort_text`, Qwen3.6/Gemma4 only have `thinking`, DeepSeek-V4 has both,
/// and LFM2.5/MiniMax have neither. `None` on either field means **omit
/// that template variable entirely** — not "pass it as null" — because at
/// least one template (DeepSeek-V4's) branches on `... is defined`, which a
/// null-valued key would satisfy differently than a truly absent one.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReasoningParams {
    /// Merged into the render context as **both** `thinking` and
    /// `enable_thinking` when `Some` — the two literal variable names found
    /// across the families surveyed for this (see `docs/adr/0003-model-profiles.md`
    /// and issue #138). Harmless for a template that only reads one of
    /// them; minijinja ignores unused context keys.
    pub thinking: Option<bool>,
    /// Merged into the render context as `reasoning_effort` when `Some`.
    pub effort_text: Option<&'static str>,
    /// Merged into the render context as `preserve_thinking` when `Some`.
    ///
    /// Unlike the other two this is **not** a function of the requested
    /// effort — it is the family's standing policy, from
    /// [`ModelProfile::preserve_prior_reasoning`], and a provider fills it in
    /// whether or not a `reasoningEffort` was configured. Leaving it out when
    /// nobody asked for an effort would mean the policy applied only to the
    /// turns that happened to set one.
    pub preserve_thinking: Option<bool>,
}

/// One model family's wire knowledge.
///
/// Implementors override what their family does differently and inherit the rest.
/// The defaults are deliberately the *narrow* generic behavior — gallium's own
/// JSON protocol and `<think>` stripping — rather than the permissive
/// everything-at-once cascade, which lives in [`Generic`] where it belongs: a
/// profile that forgets to override gets the conservative answer, not every other
/// family's parser.
pub trait ModelProfile: Send + Sync {
    /// The name a config selects this profile by (`[llm] profile = "…"`).
    /// Stable, kebab-case, and matched leniently by [`by_name`].
    fn name(&self) -> &'static str;

    /// Whether `arch` is one of this family's architecture names — the GGUF's
    /// `general.architecture`, or safetensors `model_type`.
    ///
    /// This is the **authoritative** signal and [`detect`] consults it before it
    /// looks at any template: a GGUF llama.cpp can load must report one of the
    /// names in its own `llama-arch.cpp` dispatch table, so the string is both
    /// present and precise. Match exactly rather than by prefix wherever a
    /// sibling generation exists — `gemma3` is not `gemma4`, `deepseek2` is not
    /// `deepseek4` — because a wrong profile reads a known model's output by
    /// another family's rules, which is worse than no profile at all.
    fn matches_arch(&self, _arch: &str) -> bool {
        false
    }

    /// Whether this family can be recognized from its chat template alone.
    ///
    /// Only consulted for a model whose architecture nobody here recognizes —
    /// a fork, or a name llama.cpp renamed under us — so it is a rescue, not the
    /// main path. Defaults to the tool-format check, since the literal that says
    /// "this template declares tools in my format" is the same literal that
    /// identifies the family.
    ///
    /// Deliberately second: some of these literals are loose (Gemma 4's
    /// `declaration:` is an ordinary English word with a colon), and a loose
    /// template match must never outrank another family's exact architecture.
    fn matches_template(&self, template: &str) -> bool {
        self.template_formats_tools_natively(template)
    }

    /// Tool calls in the model's reply, **with ids assigned**. This is what
    /// callers use; implementors override [`ModelProfile::parse_tool_calls`].
    fn tool_calls(&self, text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        let mut calls = self.parse_tool_calls(text, tools);
        wire::number_ids(&mut calls);
        calls
    }

    /// The family's **own** wire format, the one its fine-tuning taught it.
    /// This is the method a family profile overrides; everything around it —
    /// stripping reasoning first, falling back to the prose protocols after —
    /// comes from [`ModelProfile::parse_tool_calls`] below.
    ///
    /// `text` arrives with reasoning already removed. Ids are ignored.
    ///
    /// `tools` is here for the formats that cannot describe themselves:
    /// MiniMax's renders `"42"` and `42` identically and needs the schema to
    /// tell them apart. Formats that name their own types (DSML) and JSON
    /// ignore it.
    ///
    /// Default: none. A family with no native format (Qwen 3.6, LFM2.5) reads
    /// only what gallium asked it for, which is what the fallback provides.
    fn parse_native_tool_calls(&self, _text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        Vec::new()
    }

    /// Tool calls in a reply: reasoning off, the family's own format, then the
    /// prose protocols gallium asked for.
    ///
    /// Override [`ModelProfile::parse_native_tool_calls`] instead of this
    /// wherever the shape fits, because the two rules baked in here are the ones
    /// worth being unable to forget:
    ///
    /// Reasoning is stripped **once, before any format is tried**. A model
    /// reasoning *about* a call ("I could write `<invoke name=\"rm\">`…") has not
    /// made one, and every native format is as findable inside a `<think>` block
    /// as JSON is — parsing thinking as action is the one misreading here with
    /// consequences outside the turn. Once and not per-format because
    /// [`wire::think::strip_think_blocks`] is not idempotent.
    ///
    /// And the native format is tried **before** the JSON scan, not after. A
    /// native call's *argument* may itself be JSON carrying a `name` key
    /// (`<parameter name="body">{"name":"x"}</parameter>`), which the
    /// balanced-span scan would happily return as a call to `x`. Asking the
    /// format the model was trained on first makes that unreachable.
    fn parse_tool_calls(&self, text: &str, tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        let cleaned = wire::think::strip_think_blocks(text);
        let text = cleaned.as_str();
        let native = self.parse_native_tool_calls(text, tools);
        if !native.is_empty() {
            return native;
        }
        wire::fallback_calls(text, tools)
    }

    /// The reply with the model's reasoning taken out of it.
    ///
    /// Default: `<think>…</think>`, the one shape that is common enough to be
    /// worth trying on a model nothing is known about.
    fn clean_reply(&self, text: &str) -> String {
        wire::think::strip_think_blocks(text).trim().to_string()
    }

    /// The answer-so-far that is safe to show *progressively*, while the model
    /// is still decoding — [`ModelProfile::clean_reply`]'s incremental
    /// counterpart, consumed by `crate::streaming::StreamingReply`.
    ///
    /// `clean_reply` is written for a complete reply and owes nothing to the
    /// prefixes of one: Qwen3's visible text collapses to `""` the instant
    /// `</think>` lands (#233), and Harmony's `analysis` channel streams as
    /// prose until a freeze marker latches the stream shut for good (#231).
    /// Deriving a stream by diffing it therefore leaks reasoning or streams
    /// nothing. This method is the family's own statement of what may stream,
    /// and it carries the contract the batch method never had — **prefix
    /// monotonicity**:
    ///
    /// - for `raw₁` a prefix of `raw₂`: `stream_reply(raw₁)` is `None`, or a
    ///   string that `stream_reply(raw₂)`'s value extends (a still-forming
    ///   marker at the tail excepted — `StreamingReply`'s lookback holds that
    ///   region back, so it is never emitted);
    /// - `None` means the protocol has not decided yet — hold everything;
    /// - on the complete raw text the value must agree with what
    ///   [`ModelProfile::clean_reply`] returns, since that final message
    ///   supersedes the accumulated fragments on every client.
    ///
    /// `StreamingReply` enforces the contract at runtime by freezing when an
    /// emitted prefix stops matching — safe, but the rest of that call streams
    /// nothing, so an override that violates monotonicity has quietly disabled
    /// itself. `profile::tests::every_family_streams_its_answer_and_nothing_else`
    /// replays each family's shape character by character to catch that in CI
    /// instead.
    ///
    /// Reasoning whose opener lives in the *prompt* (a template that pre-fills
    /// `<think>\n` — Qwen3.8 with thinking on, MiniMax-M2.7 always) cannot be
    /// recognised from the raw text alone; the engine prepends the dangling
    /// opener before calling this — see `streaming::prompt_prefills_thinking`.
    ///
    /// Default: [`wire::think::stream_visible`] — closed `<think>` blocks
    /// removed, an unclosed one held back rather than shown.
    fn stream_reply(&self, raw: &str) -> Option<String> {
        Some(wire::think::stream_visible(raw))
    }

    /// Whether generation should stop here, given everything sampled so far.
    ///
    /// This is a predicate rather than a list of stop strings because the
    /// families disagree about the test: Gemma's tool-call close must match at
    /// the *end* of the output, while its tool-response marker may appear
    /// anywhere. A profile expresses its own rule.
    ///
    /// The fallback an engine keeps when [`ModelProfile::stop_markers`] can't
    /// answer for this model's vocabulary (see there) — checked on decoded
    /// text, after the token is already in, so it cannot stop generation
    /// *before* a marker is emitted the way an id comparison can.
    fn stops_generation(&self, _text: &str) -> bool {
        false
    }

    /// Marker strings whose presence — the instant one is sampled, as a
    /// single token — ends generation, named as data rather than folded into
    /// [`ModelProfile::stops_generation`]'s predicate.
    ///
    /// An engine resolves each marker to a token id once, at load, against
    /// the model's own vocabulary (see `llm_local.rs::resolve_stop_markers` /
    /// `llm_candle.rs`'s equivalent), and compares the id directly on every
    /// sampled token when resolution succeeds for *all* markers — replacing
    /// `stops_generation`'s decoded-string scan, not just speeding it up: a
    /// marker that is one token cannot appear embedded inside another
    /// token's text the way its string form can appear inside an argument
    /// value a model happens to quote, so the ambiguity `stops_generation`'s
    /// per-family test predicate exists to resolve (suffix vs. "anywhere")
    /// doesn't arise at this level — every marker is simply "was this token
    /// just sampled." See ADR 0003 step 5.
    ///
    /// If even one marker fails to resolve to exactly one token for a given
    /// vocabulary (0, because the model doesn't have it as an added token;
    /// or 2+, because it splits into pieces), the engine logs once and keeps
    /// calling `stops_generation` on decoded text for that model, unchanged.
    /// Default: none, which leaves every profile without an override exactly
    /// where it is today — `stops_generation`'s default body is `false` and
    /// never touches text, so there is nothing to replace.
    fn stop_markers(&self) -> &[&'static str] {
        &[]
    }

    /// Control-token markers whose **text is wire syntax** — an engine that
    /// decodes special tokens away must put these back into the decoded string
    /// when their token is sampled, or the family's parser is handed a reply
    /// with its boundaries erased.
    ///
    /// This is LFM2.5's situation, and the bug it fixes was silent in the worst
    /// way: `<|tool_call_start|>` / `<|tool_call_end|>` are CONTROL tokens, so
    /// llama.cpp's `special=false` decode drops them and a native call arrives
    /// as bare `[Glob(...)]`. When the whole reply is the call,
    /// [`wire::python`]'s whole-reply gate still reads it — which is why this
    /// worked for as long as the model led with the call. The moment it wrote a
    /// sentence of prose first, the gate (correctly — a bare `name(...)` inside
    /// prose is how documentation becomes a phantom call) refused, and a turn
    /// that stopped *at the end-marker's token id* was reported as a text
    /// response with the call sitting in it, unread.
    ///
    /// The engine resolves each marker against the model's vocabulary at load,
    /// all-or-nothing like [`ModelProfile::stop_markers`]: restoring the opener
    /// without the closer would synthesize a shape neither engine ever
    /// produces. On the candle backend, which decodes with specials kept, these
    /// markers reach the text anyway and this list changes nothing — that is
    /// the invariant: both engines hand the profile the same reply.
    ///
    /// Default: none. A family whose wire format lives in ordinary text has
    /// nothing to restore.
    fn restore_markers(&self) -> &[&'static str] {
        &[]
    }

    /// Whether this model's chat template renders tool definitions in its own
    /// native protocol, so the llama.cpp backend should feed it structured tools
    /// instead of gallium's JSON-prose instructions.
    ///
    /// Takes the template source because that is the only evidence: a family's
    /// GGUFs differ in whether their template was built with tool support at all.
    fn template_formats_tools_natively(&self, _template: &str) -> bool {
        false
    }

    /// The reasoning in this family's own wrapper, for the `reasoning_content`
    /// a chat template renders prior-turn thinking from.
    ///
    /// The inverse of what [`ModelProfile::clean_reply`] removes, and it has to
    /// stay that way: text this misses is text the user is shown as part of the
    /// answer, and text it over-claims is answer the model never gets credited
    /// with. So a family that overrides `clean_reply` to strip its own wrapper
    /// must override this too — the default reads `<think>…</think>` and returns
    /// `None` for anything else, which for a family with a different wrapper is
    /// silence rather than an error.
    ///
    /// **Gallium does not decide whether prior turns keep their reasoning; the
    /// template does.** Gemma's own gates it on
    /// `loop.index0 > ns_turn.last_user_idx`, which implements Google's
    /// guidance that no thoughts from previous turns remain in the context
    /// window, and Qwen3.8's `preserve_thinking` defaults the other way. Both
    /// are the vendor's own statement about their own model, made in the file
    /// that model shipped with, and a policy applied here as well would either
    /// duplicate that or quietly contradict it.
    ///
    /// Not overridden by `gpt-oss` (Harmony's analysis channel) or
    /// `deepseek-v4`, which is "nobody has looked" rather than "no reasoning
    /// here" — the distinction #116 was about. Both currently return `None`,
    /// which is what they returned before this method existed.
    fn reasoning_content(&self, text: &str) -> Option<String> {
        wire::think::think_content(text)
    }

    /// Whether this family's **earlier turns** keep their reasoning in the
    /// prompt, as the `preserve_thinking` a chat template branches on.
    ///
    /// A real per-family difference, and the reason it lives here rather than
    /// being left to each template's own default: the three templates surveyed
    /// read the same variable and disagree about it — Gemma 4 and LFM2.5
    /// default it `false`, Qwen3.8 defaults to preserving everything — so
    /// gallium's behaviour differed by model with nothing in gallium's source
    /// saying so. The other two reasoning knobs
    /// ([`ModelProfile::reasoning_params`]) are already set here rather than
    /// inherited; deferring on the third alone was inconsistent rather than
    /// principled, and it left the policy unpinned against a quantizer patching
    /// a template.
    ///
    /// An override states its family's answer **and cites where that answer
    /// comes from** — the same bar [`ModelProfile::agent_preamble_suffix`]
    /// sets. Vendor guidance and a vendor's own template default both count;
    /// a guess does not.
    ///
    /// `None` leaves the variable unset, so the template's own default applies.
    /// It means *nobody has looked*, which is a different state from "this
    /// family has no policy" — the distinction #116 was about — and is why the
    /// families without an override say so in their own files.
    ///
    /// This decides only what happens to **earlier** turns. Reasoning within
    /// the current turn is gated separately by every one of these templates
    /// (`loop.index0 > last_user_idx`) and is what keeps a multi-step tool
    /// sequence coherent; nothing here turns that off.
    ///
    /// **The candle backend does not read this.** Its renderers build prompts
    /// by hand and drop prior-turn reasoning unconditionally — `QwenProtocol`
    /// runs `strip_qwen_thinking` over prior assistant content, and no
    /// `PromptRenderer` renders `ChatMessage::reasoning` at all. That predates
    /// this method rather than being introduced by it, and honouring it there
    /// would mean teaching those renderers to emit prior reasoning, which is a
    /// change to a path no config or testsuite backend currently exercises.
    fn preserve_prior_reasoning(&self) -> Option<bool> {
        None
    }

    /// Map a portable effort level onto this family's own template
    /// variables (see [`ReasoningParams`]). Default: no override — the
    /// model's own template default applies unchanged, which is exactly
    /// today's behavior for every family (nothing is currently wired up).
    fn reasoning_params(&self, _effort: ReasoningEffort) -> ReasoningParams {
        ReasoningParams::default()
    }

    /// This family's own addition to [`BASE_AGENT_PREAMBLE`] — the only
    /// lever a profile has over [`ModelProfile::agent_preamble`], which is
    /// *provided* rather than meant to be overridden directly (see there for
    /// why the split). Default: none, which leaves [`ModelProfile::agent_preamble`]
    /// answering `None` too — a family is opted in by giving it a suffix, one
    /// at a time, not by the trait's default.
    ///
    /// This is the narrow slot: a correction for a *specific, observed*
    /// failure mode (a reasoning model that over-explores before acting, a
    /// family whose own wire format degrades under quantization pressure,
    /// one that narrates a call in prose instead of emitting it), not a
    /// restatement of what the base text already says, and not what a tool's
    /// own schema already carries. Keep it short, and prefer a claim a
    /// scripted-engine test can pin (`appserver/e2e_tests.rs`'s
    /// `ScriptedProvider`, or a fixed sample through
    /// [`ModelProfile::tool_calls`]) over one that can't be checked without a
    /// multi-GB model and a testsuite run.
    ///
    /// A family with nothing to add is not automatically opted into the base
    /// contract alone — `verify-preamble` against `gemma4` (E4B) tried
    /// exactly that (an empty suffix) and it regressed `multimodal_audio`,
    /// reproducibly: the base text's "use only available tools" framing read,
    /// on that model, as a claim that tool use is the *only* input modality,
    /// displacing the native (non-tool) mtmd audio path. So `None` here means
    /// what it says — opted out — not merely "nothing tried yet".
    fn agent_preamble_suffix(&self) -> Option<&'static str> {
        None
    }

    /// The preamble sent as its own system message ahead of the
    /// operator's/client's own system prompt — gallium's protocol **ABI**
    /// for this model family, not its persona or task. The boundary that
    /// decides whether something belongs here at all: *does the model need
    /// this to use gallium's agent loop correctly?* If yes, it goes here,
    /// where every caller gets it regardless of what system prompt they
    /// supply; if it's about who the agent is or what it's for, it belongs
    /// in the caller's own system prompt instead, not here.
    ///
    /// Two layers, not one, and both live in this crate rather than a
    /// per-family copy: [`BASE_AGENT_PREAMBLE`] is gallium's own agent-runtime
    /// contract — observe before acting, correct a failed call instead of
    /// repeating it, verify before claiming success — which is a property of
    /// *this agent loop*, not of any one model family, so every family that
    /// gets a preamble at all gets the same base text rather than five
    /// copies that drift the next time it's edited.
    /// [`ModelProfile::agent_preamble_suffix`] is the narrow per-family
    /// remainder. Default: `None` — a profile opts in by overriding the
    /// suffix hook, not this method, so the composition (and the shared
    /// base) can't be accidentally dropped by a profile that means to add
    /// its own text.
    fn agent_preamble(&self) -> Option<Cow<'static, str>> {
        self.agent_preamble_suffix()
            .map(|suffix| Cow::Owned(format!("{BASE_AGENT_PREAMBLE}\n\n{suffix}")))
    }
}

/// Gallium's own agent-runtime contract: observe before acting, correct a
/// failed tool call instead of repeating it unchanged, verify before
/// claiming success. Not a property of any model family — every profile
/// that sends a preamble at all (see [`ModelProfile::agent_preamble_suffix`])
/// sends this same text, family-specific steering appended after it.
///
/// Effectiveness against gallium's own testsuite (`testsuite/`) is what
/// decides whether a family is opted in, not a guess at what should help —
/// see the eval-improve skill and `docs/adr/0003-model-profiles.md`'s own
/// insistence on evidence over assumption for this same reason.
pub const BASE_AGENT_PREAMBLE: &str = "You are operating as an agent with access to tools.\n\
\n\
Use tools to observe facts you do not know rather than guessing.\n\
Inspect relevant state before changing it.\n\
Prefer existing patterns when they satisfy the task.\n\
\n\
When a tool fails, inspect the error and correct the cause.\n\
Do not repeat an unchanged failing call.\n\
If an approach does not make progress, try another approach or report the blocker.\n\
\n\
After making important changes, verify the result when practical.\n\
Do not claim success unless it has been observed.\n\
\n\
Never fabricate tool results.\n\
Use only available tools and follow their schemas.";

/// Every profile compiled into this binary, in detection order: most specific
/// first, [`Generic`] last.
///
/// The families match on disjoint architecture names, so their relative
/// order is not load-bearing today — but the list is ordered rather than a map
/// because [`detect`] takes the first match, and a future profile for a *variant*
/// of a family here would have to sit above it, unless (like [`GptOss20b`]) it
/// answers `false` to both `matches_arch` and `matches_template` and is meant to
/// be reachable only by explicit name.
///
/// `Generic` is in the list for [`by_name`]'s sake — it can be selected
/// explicitly — but never wins detection, since [`Generic::matches`] is always
/// false. It is what [`detect`] falls back *to*. [`GptOss20b`] is the same
/// explicit-only shape for a different reason: a GGUF's metadata cannot tell a
/// 20b GPT-OSS checkpoint from a 120b one, so nothing here could route to it by
/// detection even if it wanted to.
pub static PROFILES: &[&dyn ModelProfile] = &[
    &GptOss,
    &GptOss20b,
    &Gemma4,
    &Qwen3,
    &Lfm2,
    &MiniMaxM2,
    &DeepSeekV4,
    &Generic,
];

/// The profile every unrecognized model gets.
pub static FALLBACK: &'static dyn ModelProfile = &Generic;

/// Look a profile up by the name a config or env var gave. Case-insensitive, and
/// `_` reads as `-`, so `deepseek_v4` and `DeepSeek-V4` both find `deepseek-v4`
/// — the same leniency `ToolRegistry` applies to tool names, for the same reason:
/// a name that obviously means one profile should not be an error.
pub fn by_name(name: &str) -> Option<&'static dyn ModelProfile> {
    let wanted = normalize(name);
    PROFILES
        .iter()
        .copied()
        .find(|p| normalize(p.name()) == wanted)
}

fn normalize(name: &str) -> String {
    name.trim().to_lowercase().replace('_', "-")
}

/// The names a config may use, for error messages and `--help`.
pub fn names() -> Vec<&'static str> {
    PROFILES.iter().map(|p| p.name()).collect()
}

/// The first profile that recognizes `hints`, or [`FALLBACK`].
///
/// Two passes, and the order between them is the point: **architecture first**,
/// across every profile, and only then the chat template. Within one pass the
/// registry order decides, but a template match can never outrank an
/// architecture match belonging to a profile further down the list — which it
/// would under a single pass, since `PROFILES` has to be in *some* order and
/// several template literals are loose enough to appear in another family's
/// template by accident.
pub fn detect(hints: &DetectHints) -> &'static dyn ModelProfile {
    if let Some(arch) = hints.arch {
        if let Some(profile) = PROFILES.iter().copied().find(|p| p.matches_arch(arch)) {
            tracing::info!("  Model profile: {} (from arch '{arch}')", profile.name());
            return profile;
        }
    }

    // No architecture, or one nobody here knows. The template is the rescue: a
    // model still declaring a family's tool format is still that family.
    if let Some(template) = hints.chat_template {
        if let Some(profile) = PROFILES
            .iter()
            .copied()
            .find(|p| p.matches_template(template))
        {
            tracing::info!(
                "  Model profile: {} (from chat template; arch {:?} not recognized)",
                profile.name(),
                hints.arch
            );
            return profile;
        }
    }

    // Said out loud, with the arch named: the fallback is the permissive path,
    // and knowing a model landed on it is the difference between "this model is
    // unsupported" and "this model was misdetected".
    tracing::info!(
        "  Model profile: {} (nothing matched arch={:?})",
        FALLBACK.name(),
        hints.arch
    );
    FALLBACK
}

/// Settle which profile a loaded model gets: an explicitly named one if there is
/// one, otherwise detection.
///
/// An explicit name that matches nothing is an **error naming the valid
/// profiles**, never a silent fallback — the same rule `resolve_device` follows
/// for a device that isn't there. Asking for a profile and quietly getting the
/// generic one would show up as a model that merely answers badly.
pub fn resolve(explicit: Option<&str>, hints: &DetectHints) -> Result<&'static dyn ModelProfile> {
    match explicit.map(str::trim).filter(|s| !s.is_empty()) {
        Some(name) => {
            let profile = by_name(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown model profile '{name}'; valid profiles: {}",
                    names().join(", ")
                )
            })?;
            tracing::info!("  Model profile: {} (configured)", profile.name());
            Ok(profile)
        }
        None => Ok(detect(hints)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample of each native wire format, owned by exactly one profile.
    const MINIMAX: &str = "<minimax:tool_call>\n<invoke name=\"read\">\n\
                           <parameter name=\"file_path\">a.txt</parameter>\n\
                           </invoke>\n</minimax:tool_call>";
    const DSML: &str = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n\
                        <｜DSML｜parameter name=\"file_path\" string=\"true\">a.txt\
                        </｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const HARMONY: &str = "<|start|>assistant to=functions.Glob<|channel|>commentary \
                           <|constrain|>json<|message|>{\"pattern\":\"crates/*\"}<|call|>";
    const GEMMA: &str = "<|tool_call>call:read{file_path:<|\"|>a.txt<|\"|>}<tool_call|>";

    /// The property the whole profile layer exists for. Every one of these
    /// parsers used to run over every model's output, in one cascade — which is
    /// where `26d0f80` (a stray `to=` in Harmony argument content), `eb34344`
    /// (DSML's unbounded `string=` lookup) and `6f80ba8` (MiniMax's opener-less
    /// `</think>`) all came from. A family must read its own format and be blind
    /// to the rest, so that arrival of family N+1 cannot reach family N.
    #[test]
    fn a_family_reads_its_own_wire_format_and_no_others() {
        let samples = [
            ("minimax-m2", MINIMAX),
            ("deepseek-v4", DSML),
            ("gpt-oss", HARMONY),
            ("gemma4", GEMMA),
        ];
        for (owner, text) in samples {
            for (family, _) in samples {
                let profile = by_name(family).expect("sample names a real profile");
                let calls = profile.tool_calls(text, &[]);
                if family == owner {
                    assert_eq!(calls.len(), 1, "{family} must read its own format");
                    assert_eq!(
                        calls[0].name,
                        if owner == "gpt-oss" { "Glob" } else { "read" }
                    );
                } else {
                    assert!(
                        calls.is_empty(),
                        "{family} must not read {owner}'s format, got {calls:?}"
                    );
                }
            }
        }
    }

    /// The `stream_reply` contract, checked through the real pipeline for
    /// every family: each raw sample replayed character by character — as the
    /// decode loop feeds it — through `stream_reply` into
    /// `crate::streaming::StreamingReply`, asserting three things. The
    /// reasoning never streams (the #233 leak). The answer *does* stream —
    /// a filter that freezes into silence is safe but is also the #231 bug,
    /// and this is the assertion that catches its return. And the fragments
    /// accumulate to exactly `clean_reply` of the whole text, which is the
    /// final message every client will replace them with.
    ///
    /// Samples whose reasoning opener lives in the *prompt* (Qwen3.8 with
    /// thinking on, MiniMax-M2.7) are written here with the `<think>` prefix
    /// the engine prepends — `streaming::prompt_prefills_thinking` is the
    /// other half of that contract.
    #[test]
    fn every_family_streams_its_answer_and_nothing_else() {
        // (family, raw generation, a word that appears only in the reasoning)
        let samples = [
            (
                "qwen3",
                // The candle shape: the model emits its own opener.
                "<think>\nConsidering the question about France.\n</think>\n\nThe capital is Paris.<|im_end|>",
                "Considering",
            ),
            (
                "qwen3",
                // The llama.cpp shape: the template pre-filled `<think>\n`, so
                // the engine prepends the opener (#233's crash shape).
                "<think>Considering the question about France.\n</think>\n\nThe capital is Paris.",
                "Considering",
            ),
            (
                "gpt-oss",
                "<|channel|>analysis<|message|>Considering the question about France.<|end|><|start|>assistant<|channel|>final<|message|>The capital is Paris.<|return|>",
                "Considering",
            ),
            (
                "gemma4",
                "<|channel>thought\nConsidering the question about France.\n<channel|>The capital is Paris.<turn|>",
                "Considering",
            ),
            (
                "lfm2",
                "<think>Considering the question about France.</think>The capital is Paris.<|im_end|>",
                "Considering",
            ),
            (
                "minimax-m2",
                // Pre-filled opener, engine-prepended, same as Qwen3.8's case.
                "<think>Considering the question about France.\n</think>\n\nThe capital is Paris.",
                "Considering",
            ),
            (
                "deepseek-v4",
                "<think>Considering the question about France.</think>The capital is Paris.",
                "Considering",
            ),
            (
                "generic",
                "<think>Considering the question about France.</think>The capital is Paris.",
                "Considering",
            ),
        ];
        for (family, raw, reasoning_word) in samples {
            let profile = by_name(family).expect("sample names a real profile");
            let mut stream = crate::streaming::StreamingReply::default();
            let mut out = String::new();
            for (i, _) in raw.char_indices().skip(1) {
                if let Some(visible) = profile.stream_reply(&raw[..i]) {
                    if let Some(chunk) = stream.advance(&visible, false) {
                        out.push_str(chunk);
                    }
                }
            }
            if let Some(visible) = profile.stream_reply(raw) {
                if let Some(chunk) = stream.advance(&visible, true) {
                    out.push_str(chunk);
                }
            }
            assert!(
                !out.contains(reasoning_word),
                "{family} leaked reasoning into the stream: {out:?}"
            );
            assert!(
                !stream.frozen,
                "{family} tripped the monotonicity freeze on its own well-formed reply"
            );
            assert_eq!(
                out,
                profile.clean_reply(raw),
                "{family}: fragments must accumulate to the final message"
            );
            assert!(!out.is_empty(), "{family} streamed nothing (#231's shape)");
        }
    }

    /// A family with no native format of its own reads only what gallium asked
    /// it for, so a native block belonging to someone else is not a call.
    #[test]
    fn a_prose_protocol_family_reads_no_native_format() {
        for family in ["qwen3", "lfm2"] {
            let profile = by_name(family).unwrap();
            for text in [MINIMAX, DSML, HARMONY, GEMMA] {
                let calls = profile.tool_calls(text, &[]);
                assert!(calls.is_empty(), "{family}: {calls:?}");
            }
        }
    }

    /// Detection, on the architecture names llama.cpp itself registers
    /// (`llama-arch.cpp`), and on the near misses that must **not** match: a
    /// wrong profile is worse than no profile, since it reads a known model's
    /// output by another family's rules.
    #[test]
    fn architectures_map_to_the_families_that_own_them() {
        let expected = [
            ("gpt-oss", "gpt-oss"),
            ("gemma4", "gemma4"),
            ("gemma4-assistant", "gemma4"),
            ("qwen3", "qwen3"),
            ("qwen3moe", "qwen3"),
            ("qwen35moe", "qwen3"),
            ("qwen3next", "qwen3"),
            ("lfm2", "lfm2"),
            ("lfm2moe", "lfm2"),
            ("minimax-m2", "minimax-m2"),
            ("deepseek4", "deepseek-v4"),
            // Near misses: real llama.cpp architectures whose wire format nobody
            // here has verified. Each must fall through to `generic`.
            ("gemma", "generic"),
            ("gemma2", "generic"),
            ("gemma3", "generic"),
            ("gemma3n", "generic"),
            ("gemma-embedding", "generic"),
            ("qwen2", "generic"),
            ("qwen2moe", "generic"),
            ("minimax-m3", "generic"),
            ("deepseek", "generic"),
            ("deepseek2", "generic"),
            ("deepseek32", "generic"),
            ("seed_oss", "generic"),
        ];
        for (arch, profile) in expected {
            let hints = DetectHints {
                arch: Some(arch),
                ..DetectHints::default()
            };
            assert_eq!(detect(&hints).name(), profile, "arch {arch:?}");
        }
    }

    /// A GGUF that reports no architecture — repackaged, or converted by a tool
    /// that dropped the key — can still be identified by the tool format its
    /// template declares.
    #[test]
    fn a_template_identifies_a_family_when_the_architecture_does_not() {
        let by_template = [
            (
                "{{- \"<|start|>assistant<|channel|>final<|message|>\" }}",
                "gpt-oss",
            ),
            ("<|tool>declaration:{{ tool.name }}<tool|>", "gemma4"),
            (
                "{% for tc in tool_calls %}<minimax:tool_call>{% endfor %}",
                "minimax-m2",
            ),
            ("write <｜DSML｜tool_calls> to call a tool", "deepseek-v4"),
            (
                "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>",
                "generic",
            ),
        ];
        for (template, profile) in by_template {
            let hints = DetectHints {
                chat_template: Some(template),
                ..DetectHints::default()
            };
            assert_eq!(detect(&hints).name(), profile, "template {template:?}");
        }
    }

    /// The architecture is authoritative, and this is why the two passes are
    /// separate. Gemma 4 is recognized by, among others, the template literal
    /// `declaration:` — an ordinary word with a colon that could appear in
    /// anyone's template — and it sits above DeepSeek in the registry. Matched in
    /// one pass, that loose *template* hit would beat DeepSeek's exact
    /// *architecture* hit, and a V4 model would be parsed as a Gemma.
    #[test]
    fn a_loose_template_match_never_outranks_an_architecture_match() {
        let hints = DetectHints {
            arch: Some("deepseek4"),
            chat_template: Some("… see the declaration: section above …"),
            ..DetectHints::default()
        };
        assert_eq!(detect(&hints).name(), "deepseek-v4");

        // With no architecture to go on, the same template is all there is, and
        // the family that owns the literal gets it.
        let template_only = DetectHints {
            chat_template: hints.chat_template,
            ..DetectHints::default()
        };
        assert_eq!(detect(&template_only).name(), "gemma4");
    }

    /// An architecture nobody knows plus a template that plainly declares a
    /// family's tool format: the case the template pass exists for, e.g. a fork,
    /// or a name llama.cpp renamed under us.
    #[test]
    fn an_unknown_architecture_still_gets_a_family_from_its_template() {
        let hints = DetectHints {
            arch: Some("deepseek5-experimental"),
            chat_template: Some("call tools with <｜DSML｜tool_calls>"),
            ..DetectHints::default()
        };
        assert_eq!(detect(&hints).name(), "deepseek-v4");
    }

    /// The family names are the config surface (`[llm] profile`), so a
    /// rename is a breaking change to every config and testsuite pin. Spelled
    /// out here so it cannot happen quietly.
    #[test]
    fn the_configurable_names_are_stable() {
        let mut got = names();
        got.sort();
        assert_eq!(
            got,
            [
                "deepseek-v4",
                "gemma4",
                "generic",
                "gpt-oss",
                "gpt-oss-20b",
                "lfm2",
                "minimax-m2",
                "qwen3"
            ]
        );
    }

    #[test]
    fn every_profile_has_a_unique_normalized_name() {
        let mut seen: Vec<String> = PROFILES.iter().map(|p| normalize(p.name())).collect();
        let before = seen.len();
        seen.sort();
        seen.dedup();
        assert_eq!(before, seen.len(), "two profiles answer to the same name");
    }

    #[test]
    fn a_name_is_matched_leniently() {
        assert_eq!(by_name("generic").map(|p| p.name()), Some("generic"));
        assert_eq!(by_name("  GENERIC ").map(|p| p.name()), Some("generic"));
    }

    #[test]
    fn an_unknown_name_is_an_error_that_lists_the_real_ones() {
        let err = resolve(Some("gpt-42"), &DetectHints::default())
            // A trait object is not Debug; the name is all the failure needs.
            .map(|p| p.name())
            .expect_err("an unknown profile must not silently fall back")
            .to_string();
        assert!(err.contains("gpt-42"), "{err}");
        assert!(err.contains("generic"), "{err}");
    }

    /// An absent or blank key is "nothing was configured", not a profile named
    /// "" — a config with `profile = ""` should detect, not fail.
    #[test]
    fn no_configured_name_falls_through_to_detection() {
        for explicit in [None, Some(""), Some("   ")] {
            let profile = resolve(explicit, &DetectHints::default())
                .expect("detection has a fallback and cannot fail");
            assert_eq!(profile.name(), "generic");
        }
    }

    #[test]
    fn an_unrecognized_model_gets_the_fallback() {
        let hints = DetectHints {
            arch: Some("some-arch-nobody-wrote-a-profile-for"),
            ..DetectHints::default()
        };
        assert_eq!(detect(&hints).name(), "generic");
    }

    /// Which families take ADR 0003 step 5's id-comparison path, and which are
    /// deliberately still on the decoded-text scan. Spelled out rather than
    /// derived, so extending the step to another family is a decision recorded
    /// here rather than a quiet edit to one profile.
    ///
    /// A family qualifies when its tool-call boundary is a *single token* in the
    /// vocabularies it ships with — checked against the real GGUFs:
    /// Gemma 4 `<tool_call|>` (id 49) / `<|tool_response>` (50), Qwen 3.5
    /// `</tool_call>` (248059), LFM2.5 `<|tool_call_end|>` (124906).
    ///
    /// The three without markers are not a gap. GPT-OSS's Harmony terminators
    /// (`<|call|>` / `<|return|>`) are already end-of-turn tokens both engines
    /// stop on; MiniMax's `</minimax:tool_call>` and DeepSeek's
    /// `</｜DSML｜tool_calls>` are multi-character tags unlikely to be one token,
    /// and neither model is cached anywhere this could be checked — claiming them
    /// unverified is what the step is meant to avoid.
    #[test]
    fn stop_markers_are_named_by_exactly_the_families_that_have_them() {
        let expected: &[(&str, bool)] = &[
            ("gemma4", true),
            ("qwen3", true),
            ("lfm2", true),
            ("gpt-oss", false),
            ("gpt-oss-20b", false),
            ("minimax-m2", false),
            ("deepseek-v4", false),
            ("generic", false),
        ];
        for profile in PROFILES {
            let want = expected
                .iter()
                .find(|(n, _)| *n == profile.name())
                .map(|(_, w)| *w)
                .unwrap_or_else(|| panic!("profile {} missing from this list", profile.name()));
            assert_eq!(
                !profile.stop_markers().is_empty(),
                want,
                "{} stop_markers: {:?}",
                profile.name(),
                profile.stop_markers()
            );
        }
    }

    /// Every family's prior-reasoning policy, spelled out, for the same reason
    /// the two lists below are: this is a claim about a model's vendor, and a
    /// list is what stops one being added quietly to a family nobody checked.
    ///
    /// The three answers and where each comes from:
    ///
    /// - `gemma4` → `false`. Google, explicitly: "no generated thoughts from
    ///   previous turns remain in the context window"
    ///   (<https://ai.google.dev/gemma/docs/capabilities/thinking>).
    /// - `lfm2` → `false`. Its own template's default.
    /// - `qwen3` → `true`. Its own template's default, which is the opposite.
    ///
    /// `None` everywhere else means the template's default applies *and* that
    /// nobody has looked — the two are not the same and the profiles say which.
    ///
    /// Pinned here rather than only through the render tests in
    /// `llm_local_templates`: those check that whatever a profile says is what
    /// the prompt does, which is the wiring. Changing a profile's answer moves
    /// both sides of that check together, so it cannot notice. This can.
    #[test]
    fn prior_reasoning_policy_is_named_by_exactly_the_families_that_have_one() {
        let expected: &[(&str, Option<bool>)] = &[
            ("gpt-oss", None),
            ("gpt-oss-20b", None),
            ("gemma4", Some(false)),
            ("qwen3", Some(true)),
            ("lfm2", Some(false)),
            ("minimax-m2", None),
            ("deepseek-v4", None),
            ("generic", None),
        ];
        for profile in PROFILES {
            let want = expected
                .iter()
                .find(|(n, _)| *n == profile.name())
                .map(|(_, w)| *w)
                .unwrap_or_else(|| panic!("profile {} missing from this list", profile.name()));
            assert_eq!(
                profile.preserve_prior_reasoning(),
                want,
                "{} preserve_prior_reasoning",
                profile.name()
            );
        }
    }

    /// Which families carry an `agent_preamble`, spelled out for the same
    /// reason as `stop_markers_are_named_by_exactly_the_families_that_have_them`
    /// above: a profile earns one from an observed failure, and this list is
    /// what stops that from drifting into a guess added quietly to a profile
    /// nobody has actually seen misbehave.
    #[test]
    fn agent_preamble_is_named_by_exactly_the_families_that_have_one() {
        let expected: &[(&str, bool)] = &[
            ("gpt-oss", true),
            ("gpt-oss-20b", false),
            ("gemma4", false),
            ("qwen3", true),
            ("lfm2", false),
            ("minimax-m2", false),
            ("deepseek-v4", true),
            ("generic", false),
        ];
        for profile in PROFILES {
            let want = expected
                .iter()
                .find(|(n, _)| *n == profile.name())
                .map(|(_, w)| *w)
                .unwrap_or_else(|| panic!("profile {} missing from this list", profile.name()));
            assert_eq!(
                profile.agent_preamble().is_some(),
                want,
                "{} agent_preamble: {:?}",
                profile.name(),
                profile.agent_preamble()
            );
        }
    }
}
