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
//! try-everything behavior gallium has always had.
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
pub use gpt_oss::GptOss;
pub use lfm2::Lfm2;
pub use minimax::MiniMaxM2;
pub use qwen3::Qwen3;

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
        wire::fallback_calls(text)
    }

    /// The reply with the model's reasoning taken out of it.
    ///
    /// Default: `<think>…</think>`, the one shape that is common enough to be
    /// worth trying on a model nothing is known about.
    fn clean_reply(&self, text: &str) -> String {
        wire::think::strip_think_blocks(text).trim().to_string()
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

    /// Whether this model's chat template renders tool definitions in its own
    /// native protocol, so the llama.cpp backend should feed it structured tools
    /// instead of gallium's JSON-prose instructions.
    ///
    /// Takes the template source because that is the only evidence: a family's
    /// GGUFs differ in whether their template was built with tool support at all.
    fn template_formats_tools_natively(&self, _template: &str) -> bool {
        false
    }
}

/// Every profile compiled into this binary, in detection order: most specific
/// first, [`Generic`] last.
///
/// The six families match on disjoint architecture names, so their relative
/// order is not load-bearing today — but the list is ordered rather than a map
/// because [`detect`] takes the first match, and a future profile for a *variant*
/// of a family here would have to sit above it.
///
/// `Generic` is in the list for [`by_name`]'s sake — it can be selected
/// explicitly — but never wins detection, since [`Generic::matches`] is always
/// false. It is what [`detect`] falls back *to*.
pub static PROFILES: &[&dyn ModelProfile] = &[
    &GptOss,
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

    /// The six family names are the config surface (`[llm] profile`), so a
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

    /// ADR 0003 step 5 is scoped to Gemma 4 only — its `stop_markers` names
    /// the id-comparison fast path an engine may take instead of
    /// `stops_generation`'s decoded-text scan. Every other profile keeps the
    /// trait default (empty), which is not a gap: none of them override
    /// `stops_generation` either, so there is nothing for an id comparison to
    /// replace yet.
    #[test]
    fn only_gemma4_names_stop_markers() {
        for profile in PROFILES {
            let markers = profile.stop_markers();
            if profile.name() == "gemma4" {
                assert!(!markers.is_empty(), "gemma4 should name stop markers");
            } else {
                assert!(
                    markers.is_empty(),
                    "{} unexpectedly names stop markers: {markers:?}",
                    profile.name()
                );
            }
        }
    }
}
