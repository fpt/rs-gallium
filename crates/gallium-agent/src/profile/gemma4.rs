//! Gemma 4: `<|tool_call>call:NAME{…}<tool_call|>` and the thought channel.

use crate::llm::{ToolCallInfo, ToolDefinition};

use super::wire;
use super::{ModelProfile, ReasoningEffort, ReasoningParams};

/// Note `crate::gemma`'s `normalise_tool_name` / `normalise_path_args` are
/// **not** applied here. They are opt-in, and only the candle path
/// (`protocol.rs`) opts in — llama.cpp keeps names verbatim so a mixed-case MCP
/// tool is never folded. Unifying the two is a behavior change on one engine
/// whichever way it goes, so it waits for its own change rather than riding along
/// with a refactor.
///
/// Gemma 4 (E2B/E4B, 12B, 26B-A4B), whose template declares tools as
/// `<|tool>declaration:…<tool|>` and which answers with
/// `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>` — `<|"|>` being its
/// quote token. See docs/GEMMA4.md.
pub struct Gemma4;

impl ModelProfile for Gemma4 {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    /// The exact Gemma **4** architecture names, not `gemma` by prefix: Gemma 2
    /// and 3 are different formats with no `<|tool_call>` at all, and claiming
    /// them would hand their output a parser built for another generation.
    /// `gemma4-assistant` is llama.cpp's separate arch for the draft/assistant
    /// variant, same wire format.
    fn matches_arch(&self, arch: &str) -> bool {
        matches!(arch, "gemma4" | "gemma4-assistant")
    }

    fn parse_native_tool_calls(&self, text: &str, _tools: &[ToolDefinition]) -> Vec<ToolCallInfo> {
        wire::gemma_calls(text)
    }

    /// Also trims the trailing turn markers, which matters on one engine only:
    /// the candle backend decodes with special tokens kept as text, so a reply
    /// arrives ending in `<turn|>` / `<eos>` and would show them to the user.
    /// On llama.cpp those are EOG tokens that stop generation and never reach
    /// the string, so the trim is a no-op there rather than a second behavior.
    fn clean_reply(&self, text: &str) -> String {
        let s = crate::gemma::strip_thinking_blocks(text);
        let s = wire::think::strip_think_blocks(&s);
        // `<end_of_turn>` is Gemma 2's spelling, kept for a GGUF converted
        // from that generation's template. `<|tool_call>` / `<tool_call|>` are
        // trimmed too: a parsed call has already been taken out of the reply by
        // this point, so a *trailing* tool-call delimiter here is a degenerate
        // tail (a struggling quant emitting the marker and no body) rather than
        // quoted syntax, which only ever appears mid-text.
        wire::strip_trailing_markers(
            s.trim(),
            &[
                "<turn|>",
                "<eos>",
                "<end_of_turn>",
                "<tool_call|>",
                "<|tool_call>",
            ],
        )
        .to_string()
    }

    /// Only `<|think|>`, and the asymmetry is the finding (marker token-type
    /// audit of `unsloth/gemma-4-E4B-it-GGUF`, 2026-08-30 —
    /// `scripts/marker_audit.py`): every tool-call and channel marker in this
    /// family's vocabulary is USER_DEFINED and survives the `special=false`
    /// decode — which is why Gemma never had LFM2's parse bug — but the paired
    /// thinking form's opener is a CONTROL token, while its closer `<|/think|>`
    /// is not in the vocabulary at all and arrives as ordinary multi-token
    /// text. Undropped closer plus dropped opener means a paired-form reply
    /// reaches `crate::gemma::strip_thinking_blocks` as an *orphan*
    /// `reasoning…<|/think|>answer`, a shape it does not pair — the reasoning
    /// would be shown as answer. Restoring the opener re-forms the pair; when
    /// the model never closes it, the unclosed-`<|think|>` rule already drops
    /// the tail as thinking. (`<turn|>` is CONTROL too, but it is an EOG that
    /// ends generation before it could reach the text.)
    fn restore_markers(&self) -> &[&'static str] {
        &["<|think|>"]
    }

    /// [`Gemma4::clean_reply`] is already incremental-safe, so it streams as it
    /// is: `crate::gemma::strip_thinking_blocks` drops an *unclosed*
    /// `<|channel>thought` — thinking, per the #199 rule — so the visible text
    /// stays empty through the thought and only grows once the channel closes,
    /// and the trailing turn markers all begin with `<`, which the stream
    /// filter's lookback holds while they form. (A reply with a *second*
    /// `<channel|>` close rewrites the visible text — the runtime monotonicity
    /// freeze catches that; a thought channel closes once.)
    fn stream_reply(&self, raw: &str) -> Option<String> {
        Some(self.clean_reply(raw))
    }

    /// Stop as soon as the model closes a tool call, or claims a tool
    /// *response*: the first means the call is complete and gallium should run
    /// it, the second means the model has started writing the result itself,
    /// which it must not be allowed to finish.
    ///
    /// The two take different tests. A closing `<tool_call|>` is only a boundary
    /// at the very end of what has been sampled — it appears mid-text in a reply
    /// that merely quotes the syntax — while `<|tool_response>` anywhere is
    /// already the failure.
    fn stops_generation(&self, text: &str) -> bool {
        text.ends_with("<tool_call|>") || text.contains("<|tool_response>")
    }

    /// Both are single tokens in every Gemma 4 vocabulary this has been
    /// checked against (`<tool_call|>` id 49, `<|tool_response>` id 50 — see
    /// `protocol.rs`'s `GemmaProtocol` doc comment) so the id-comparison path
    /// (ADR 0003 step 5) is expected to apply here, with `stops_generation`
    /// above as the fallback should a converted GGUF ever split one.
    fn stop_markers(&self) -> &[&'static str] {
        &["<tool_call|>", "<|tool_response>"]
    }

    /// Three literals because a Gemma 4 GGUF's template may spell its tool
    /// section any of these ways depending on how it was converted.
    fn template_formats_tools_natively(&self, template: &str) -> bool {
        template.contains("<|tool_call>")
            || template.contains("<|tool>")
            || template.contains("declaration:")
    }

    /// Both of this family's wrappers, matching what [`Gemma4::clean_reply`]
    /// strips — the channel form and the paired `<|think|>` form.
    ///
    /// The default would find neither: it reads `<think>…</think>`, and Gemma
    /// writes `<|channel>thought … <channel|>`. Without this the family's
    /// reasoning is extracted as `None` on every turn, and its template's own
    /// thought channel — which it renders for the *current* turn's tool calls,
    /// and which is what keeps a multi-step sequence coherent — is handed
    /// nothing.
    fn reasoning_content(&self, text: &str) -> Option<String> {
        crate::gemma::thinking_content(text)
    }

    /// No. Google is explicit about it:
    ///
    /// > The historical model output must only include the final response.
    /// > Ensure that no generated thoughts from previous turns remain in the
    /// > context window before the next user turn begins.
    ///
    /// — <https://ai.google.dev/gemma/docs/capabilities/thinking>
    ///
    /// This family's own template already defaults `preserve_thinking` to
    /// `false`, so saying it here changes no behaviour today. That is the
    /// point: the policy is now gallium's, stated where it can be read
    /// alongside the other families' and pinned against a template that gets
    /// patched on its way through a quantizer.
    ///
    /// Setting it `true` would be reachable — Gemma's gate is
    /// `(loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and
    /// message.get('tool_calls'))`, so the flag would carry prior turns'
    /// tool-call reasoning forward — which is exactly what the guidance above
    /// forbids. The current turn's reasoning is unaffected either way; that is
    /// the first half of the gate.
    fn preserve_prior_reasoning(&self) -> Option<bool> {
        Some(false)
    }

    /// Gemma 4's own GGUF template reads only a boolean `enable_thinking` —
    /// same variable name as Qwen 3.6's, but the **opposite default**: this
    /// template treats it as off unless explicitly set `true` (Qwen's
    /// defaults on unless explicitly set `false`). The mapping below still
    /// comes out identical to Qwen's, because gallium always sets the
    /// variable explicitly whenever an effort was configured at all — the
    /// differing template default only matters when nothing was configured,
    /// which `ReasoningParams::default()` (omit the key) already leaves
    /// alone.
    fn reasoning_params(&self, effort: ReasoningEffort) -> ReasoningParams {
        ReasoningParams {
            thinking: Some(effort != ReasoningEffort::Low),
            effort_text: None,
            preserve_thinking: None,
        }
    }

    // Deliberately no `agent_preamble_suffix` override — not "nothing observed
    // yet" but a tried-and-reverted result. `Some("")` (opting into
    // `BASE_AGENT_PREAMBLE` alone, no family text) was tried via the
    // `verify-preamble` skill against `gemma4` (E4B): it made the model refuse
    // `multimodal_audio`, a case it passes with no preamble at all — 4/4 clean
    // runs before, 4/4 identical refusals ("I can only process text and use
    // the provided tools to interact with a file system") after, so this is a
    // reproduced regression, not a single noisy sample. The base text's own
    // "use only available tools" framing appears to read, on this model, as a
    // claim that tool use is the *only* input modality — displacing the native
    // mtmd audio path, which isn't a tool call at all. Left unset rather than
    // silently retried, so this isn't rediscovered the same way next time.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 2 and 3 are real llama.cpp architectures with a different format.
    /// Matching the family by substring would have claimed them.
    #[test]
    fn only_gemma_4_is_claimed() {
        for arch in ["gemma4", "gemma4-assistant"] {
            assert!(Gemma4.matches_arch(arch), "{arch}");
        }
        for arch in ["gemma", "gemma2", "gemma3", "gemma3n", "gemma-embedding"] {
            assert!(!Gemma4.matches_arch(arch), "{arch}");
        }
    }

    #[test]
    fn only_low_turns_thinking_off() {
        assert_eq!(
            Gemma4.reasoning_params(ReasoningEffort::Low).thinking,
            Some(false)
        );
        for effort in [
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            let params = Gemma4.reasoning_params(effort);
            assert_eq!(params.thinking, Some(true));
            assert_eq!(params.effort_text, None);
        }
    }

    #[test]
    fn parses_its_native_call_including_hyphenated_mcp_names() {
        let calls = Gemma4.tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
            &[],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc");
        assert_eq!(calls[0].arguments["query"], "mcp-go");
    }

    /// The inverse property, on both wrappers: what `clean_reply` strips is
    /// what `reasoning_content` returns. Asserted together, because the bug
    /// this guards is the two drifting apart — reasoning shown as an answer, or
    /// an answer claimed as reasoning.
    #[test]
    fn reasoning_content_is_what_clean_reply_strips() {
        let channel = "<|channel>thought\nchecking git log<channel|>Here it is.";
        assert_eq!(
            Gemma4.reasoning_content(channel).as_deref(),
            Some("checking git log")
        );
        assert_eq!(Gemma4.clean_reply(channel), "Here it is.");

        let think = "<|think|>reasoning<|/think|>The answer.";
        assert_eq!(
            Gemma4.reasoning_content(think).as_deref(),
            Some("reasoning")
        );
        assert_eq!(Gemma4.clean_reply(think), "The answer.");
    }

    /// The default reads `<think>…</think>` and would find nothing in either of
    /// this family's wrappers — which is silence, not an error, and is why the
    /// override exists at all.
    #[test]
    fn the_generic_extractor_would_find_nothing_here() {
        use super::super::wire::think::think_content;
        assert_eq!(
            think_content("<|channel>thought\nweighing<channel|>a"),
            None
        );
        assert_eq!(think_content("<|think|>weighing<|/think|>a"), None);
    }

    /// A model that ran out of tokens mid-thought produced thinking, not an
    /// answer — `clean_reply` discards that tail, so this has to claim it.
    #[test]
    fn an_unclosed_think_block_is_reasoning() {
        assert_eq!(
            Gemma4
                .reasoning_content("<|think|>still weighing")
                .as_deref(),
            Some("still weighing")
        );
    }

    /// The regression: Gemma 4 26B-A4B under heavy CPU offload emitted
    /// `<|channel>thought<tool_call|>` and nothing else on `data_analysis` and
    /// `refactoring`, and that raw markup reached the user as the reply because
    /// `strip_thinking_blocks` only knew the *closed* channel form. An unclosed
    /// `<|channel>thought` is thinking the same way an unclosed `<|think|>` is.
    #[test]
    fn an_unclosed_channel_thought_does_not_leak_as_a_reply() {
        assert_eq!(Gemma4.clean_reply("<|channel>thought<tool_call|>"), "");
        assert_eq!(
            Gemma4.clean_reply("<|channel>thought\nweighing the columns"),
            ""
        );
        // The bare closing delimiter is structure, not thought.
        assert_eq!(
            Gemma4
                .reasoning_content("<|channel>thought weighing the columns<tool_call|>")
                .as_deref(),
            Some("weighing the columns")
        );
    }

    /// Even a *closed* thought followed by a lone `<tool_call|>` — no `call:`
    /// body, so nothing was parsed out — must not show that delimiter.
    #[test]
    fn a_trailing_bare_tool_call_delimiter_is_trimmed() {
        assert_eq!(
            Gemma4.clean_reply("<|channel>thought done<channel|>Here.<tool_call|>"),
            "Here."
        );
        assert_eq!(
            Gemma4.clean_reply("Here is the answer.<|tool_call>"),
            "Here is the answer."
        );
    }

    #[test]
    fn a_plain_answer_has_no_reasoning() {
        assert_eq!(Gemma4.reasoning_content("Just the answer."), None);
    }

    /// The thought channel reached a user in a real gemma4-12b session. Both of
    /// Gemma's wrapper shapes have to come off.
    #[test]
    fn both_thinking_wrappers_are_stripped() {
        assert_eq!(
            Gemma4.clean_reply("<|channel>thought\nchecking git log<channel|>Here it is."),
            "Here it is."
        );
        assert_eq!(
            Gemma4.clean_reply("<|think|>reasoning<|/think|>The answer."),
            "The answer."
        );
    }

    #[test]
    fn generation_stops_at_a_closed_call_or_any_claimed_response() {
        assert!(Gemma4.stops_generation("<|tool_call>call:read{}<tool_call|>"));
        assert!(Gemma4.stops_generation("<|tool_response>response:read{...}"));
        // Mid-call: the arguments are still being written.
        assert!(!Gemma4.stops_generation("<|tool_call>call:read{path:<|\"|>a"));
        assert!(!Gemma4.stops_generation("The answer is 42."));
    }

    /// The two names `stop_markers` returns must be exactly the two literals
    /// `stops_generation` above tests against — an engine that resolves them
    /// to token ids is replacing that predicate, not answering a different
    /// question.
    #[test]
    fn stop_markers_match_stops_generation() {
        assert_eq!(Gemma4.stop_markers(), &["<tool_call|>", "<|tool_response>"]);
    }

    /// No preamble at all — see the comment on the (absent) override: this
    /// was tried and reverted, not left unconsidered.
    #[test]
    fn no_agent_preamble() {
        assert!(Gemma4.agent_preamble().is_none());
    }
}

#[cfg(test)]
mod port_tests {
    use super::*;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    /// The candle path decodes with special tokens kept as text, so a reply
    /// arrives with its turn marker attached. Transcribed from `protocol.rs`'s
    /// `strip_gemma_specials`, which is the spec for this.
    #[test]
    fn trailing_turn_markers_do_not_reach_the_user() {
        assert_eq!(
            Gemma4.clean_reply("The answer is 42.<turn|>"),
            "The answer is 42."
        );
        assert_eq!(Gemma4.clean_reply("Done.<turn|><eos>"), "Done.");
        assert_eq!(Gemma4.clean_reply("Done.<end_of_turn>"), "Done.");
        // A no-op on llama.cpp, where these never reach the string.
        assert_eq!(Gemma4.clean_reply("The answer is 42."), "The answer is 42.");
    }

    /// Mixed-case MCP names still survive when offered, which is the property
    /// `crate::gemma`'s "verbatim" contract exists to protect.
    #[test]
    fn a_hyphenated_mcp_name_survives() {
        let tools = [tool("search-godoc")];
        let calls = Gemma4.tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
            &tools,
        );
        assert_eq!(calls[0].name, "search-godoc");
    }

    /// The one Gemma marker the llama.cpp decode drops is the paired thinking
    /// form's opener; see `restore_markers`'s comment for the audit. Restored,
    /// the pair strips as reasoning; without it the closer arrives orphaned —
    /// `<|/think|>` is not even a vocabulary token, so it always survives as
    /// text — and the reasoning before it would be shown as answer.
    #[test]
    fn the_paired_think_opener_is_the_marker_to_restore() {
        assert_eq!(Gemma4.restore_markers(), &["<|think|>"]);
        // The pair, re-formed by restoration, is reasoning…
        assert_eq!(
            Gemma4.clean_reply("<|think|>weighing it<|/think|>The answer."),
            "The answer."
        );
        // …and the orphan shape restoration prevents would have leaked it.
        assert_eq!(
            Gemma4.clean_reply("weighing it<|/think|>The answer."),
            "weighing it<|/think|>The answer."
        );
    }
}
