//! What a real chat template has to survive, asserted against the templates
//! themselves.
//!
//! Every template-level bug gallium has hit was found by loading a multi-GB
//! GGUF and reading the output — one cost Qwen3.8 a testcase (`refactoring`),
//! and issue #182 records one that had been silently degrading a model since
//! the day its config landed. None of them needed the weights: a chat template
//! is text, and the failures are in how gallium's message shapes meet it.
//!
//! So the fixtures in `tests/fixtures/chat_templates/` are the real embedded
//! templates (see the README there for provenance), and these tests render
//! through the real [`chat_env`] and the real [`render_native_prompt`] — not a
//! lookalike environment built to match. That distinction is the point: an
//! environment assembled by the test would keep passing after the one gallium
//! uses changed.
//!
//! **A known gap is declared, not skipped.** [`Fixture`] carries what gallium
//! cannot do with that template yet, each field naming its issue. The tests
//! assert the declared state rather than the desired one, so the gap is visible
//! in the source, and closing it is a one-line edit here plus a test that goes
//! green for the right reason.

use serde_json::json;

use super::{chat_env, render_chat_once, render_native_prompt, LlamaLocalProvider};
use crate::llm::{ChatMessage, ToolCallInfo, ToolDefinition};
use crate::profile::{Gemma4, Lfm2, ModelProfile, Qwen3, ReasoningParams};

/// One model's embedded chat template, plus what gallium can and cannot
/// currently do with it.
struct Fixture {
    /// File name under `tests/fixtures/chat_templates/`, for failure messages.
    name: &'static str,
    src: &'static str,
    /// Whether minijinja can parse this template at all. `false` means gallium
    /// never renders it and falls back to the manual ChatML layout — see #182.
    registers: bool,
    /// Whether the template tolerates the several system messages gallium
    /// actually sends (profile preamble, operator prompt, project context,
    /// skill catalog). `false` means `render_native` raises and the tool
    /// protocol silently changes under the model — see #175.
    admits_extra_system_messages: bool,
    /// The profile that speaks for this template's family — the one a real
    /// load would detect. Held so a test can ask the profile and the template
    /// the same question and require the same answer.
    profile: &'static dyn ModelProfile,
    /// Whether this template reads `preserve_thinking` at all. `false` means
    /// the family's policy is inert here and the template's own gate decides —
    /// which is how one profile answer covers two generations of Qwen.
    honors_preserve_thinking: bool,
    /// The `reasoning_effort` values this template accepts without raising.
    /// Empty means it never reads the variable, so any value is inert. See
    /// #176: gallium's `ReasoningEffort` has five variants and at least one
    /// family accepts three.
    reasoning_efforts: Option<&'static [&'static str]>,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "gemma4-e4b.jinja",
        honors_preserve_thinking: true,
        profile: &Gemma4,
        src: include_str!("../tests/fixtures/chat_templates/gemma4-e4b.jinja"),
        registers: true,
        admits_extra_system_messages: true,
        reasoning_efforts: None,
    },
    Fixture {
        name: "lfm2-8b-a1b.jinja",
        honors_preserve_thinking: true,
        profile: &Lfm2,
        src: include_str!("../tests/fixtures/chat_templates/lfm2-8b-a1b.jinja"),
        registers: true,
        admits_extra_system_messages: true,
        reasoning_efforts: None,
    },
    Fixture {
        // The *older* generation of the same profile, and the reason it is
        // here: it and `qwen3.8.jinja` disagree about prior-turn reasoning, so
        // one profile answer covers two templates and only a fixture can show
        // that it does. Its gate is `loop.index0 > ns.last_query_index` with no
        // `preserve_thinking` escape at all — the variable does not appear in
        // the file — so `Qwen3::preserve_prior_reasoning`'s `Some(true)` is
        // inert here and prior turns are dropped whatever gallium asks for.
        name: "qwen3.5-9b.jinja",
        honors_preserve_thinking: false,
        profile: &Qwen3,
        src: include_str!("../tests/fixtures/chat_templates/qwen3.5-9b.jinja"),
        registers: true,
        admits_extra_system_messages: false,
        // No `reasoning_effort` variable in this generation's template.
        reasoning_efforts: None,
    },
    Fixture {
        name: "qwen3.8.jinja",
        honors_preserve_thinking: true,
        profile: &Qwen3,
        src: include_str!("../tests/fixtures/chat_templates/qwen3.8.jinja"),
        registers: true,
        // #175: closed for this template. The bytes the 27B GGUF actually
        // carries are unsloth's patched template, which merges the leading
        // run of system/developer messages itself ("Unsloth fixes - developer
        // role, merged system messages, tool calling") rather than
        // `raise_exception('System message must be at the beginning.')` — the
        // Hub `Qwen/Qwen3.8-27B` template does raise, and this fixture used to
        // be that file. See the fixtures README.
        admits_extra_system_messages: true,
        // #176: unsloth's template silently upgrades `high` to `xhigh` before
        // the check, so `high` renders; `max` is still not one of
        // (`xhigh`, `medium`, `low`) and raises. The Hub template raises on
        // `high` too — another reason the GGUF's own bytes are the fixture.
        reasoning_efforts: Some(&["low", "medium", "high", "xhigh"]),
    },
];

/// The one tool every conversation below offers. Small on purpose: these tests
/// are about the template's structure, not about schema rendering.
fn tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "Read".to_string(),
        description: "Read a file from the filesystem".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        }),
    }]
}

/// The system messages gallium really sends, in the order `main.rs` pushes them
/// and `runtime.rs` inserts the catalog among them.
fn gallium_system_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("PROFILE PREAMBLE".to_string()),
        ChatMessage::system("OPERATOR SYSTEM PROMPT".to_string()),
        ChatMessage::system("PROJECT AGENTS.md".to_string()),
        ChatMessage::system("SKILL CATALOG".to_string()),
    ]
}

/// One completed ReAct round, in the shape `react.rs` leaves in `messages`:
/// a user turn, an assistant turn that is *only* tool calls, and the result.
///
/// The assistant turn carries reasoning, because a real one does — a reasoning
/// model does not emit a tool call out of nowhere, and a round built without it
/// would miss the case #177 was about.
fn react_round() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("OPERATOR SYSTEM PROMPT".to_string()),
        ChatMessage::user("read a.txt".to_string()),
        ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
            id: "call_1".to_string(),
            name: "Read".to_string(),
            arguments: json!({"file_path": "a.txt"}),
        }])
        .with_reasoning(Some(
            "The file is small, so reading it whole is fine.".to_string(),
        )),
        ChatMessage::tool_result(
            "call_1".to_string(),
            "Read".to_string(),
            "hello".to_string(),
        ),
    ]
}

/// Render `messages` the way a real turn does — including the system-message
/// merge `render_native_prompt` retries with — or the minijinja error that
/// stopped it.
fn render(
    fixture: &Fixture,
    messages: &[ChatMessage],
    reasoning: &ReasoningParams,
    add_generation_prompt: bool,
) -> Result<String, String> {
    let env = chat_env(fixture.src).map_err(|e| e.to_string())?;
    render_native_prompt(
        &env,
        messages,
        &tools(),
        "<bos>",
        "<eos>",
        reasoning,
        add_generation_prompt,
    )
    .map_err(|e| e.to_string())
}

/// What the **template** does with exactly these messages, with no retry. The
/// distinction from [`render`] is the subject of two tests below: a template
/// that refuses gallium's several system messages is a fact about the template,
/// and gallium rendering anyway is a fact about `render_native_prompt`.
fn render_once(
    fixture: &Fixture,
    messages: &[ChatMessage],
    reasoning: &ReasoningParams,
) -> Result<String, String> {
    let env = chat_env(fixture.src).map_err(|e| e.to_string())?;
    render_chat_once(&env, messages, &tools(), "<bos>", "<eos>", reasoning, true)
        .map_err(|e| e.to_string())
}

/// A template that will not parse is a template gallium never uses: `chat_env`
/// fails, so `render_native`, `render_template` and its system-folded retry all
/// fail, and `build_prompt` reaches `chatml_fallback` with one `warn!` line.
/// The model then never sees its own format.
#[test]
fn fixtures_register() {
    for f in FIXTURES {
        let got = chat_env(f.src).is_ok();
        assert_eq!(
            got, f.registers,
            "{}: registers = {got}, fixture declares {} \
             (a template that does not register is #182's failure mode)",
            f.name, f.registers
        );
    }
}

/// The conversation gallium actually builds has to render against every
/// template, whatever that template thinks of several system messages. Not a
/// reduced conversation: the four system messages are the whole point, since
/// they are what `main.rs` and `runtime.rs` produce.
///
/// A failure here is not cosmetic. `build_prompt` catches it and asks the model
/// for JSON prose instead — a different wire protocol, arriving with no error
/// anyone sees.
#[test]
fn the_system_messages_gallium_sends_always_render() {
    for f in FIXTURES.iter().filter(|f| f.registers) {
        let mut messages = gallium_system_messages();
        messages.push(ChatMessage::user("read a.txt".to_string()));

        let prompt = render(f, &messages, &ReasoningParams::default(), true)
            .unwrap_or_else(|e| panic!("{}: gallium's own message shape must render: {e}", f.name));

        // Merged or not, no system message may be dropped: each is a different
        // author, and losing one silently is worse than the raise.
        for expected in [
            "PROFILE PREAMBLE",
            "OPERATOR SYSTEM PROMPT",
            "PROJECT AGENTS.md",
            "SKILL CATALOG",
        ] {
            assert!(
                prompt.contains(expected),
                "{}: {expected} is missing from the rendered prompt:\n{prompt}",
                f.name
            );
        }
    }
}

/// The template's own opinion, which is what `admits_extra_system_messages`
/// records: some raise on any system message that is not the first, and
/// `render_native_prompt` merges and retries for exactly those. Asserted
/// separately so the retry cannot hide a template changing under us.
#[test]
fn whether_a_template_admits_extra_system_messages() {
    for f in FIXTURES.iter().filter(|f| f.registers) {
        let mut messages = gallium_system_messages();
        messages.push(ChatMessage::user("read a.txt".to_string()));

        let got = render_once(f, &messages, &ReasoningParams::default()).is_ok();
        assert_eq!(
            got, f.admits_extra_system_messages,
            "{}: the template rendered = {got}, fixture declares {} — see #175",
            f.name, f.admits_extra_system_messages
        );
    }
}

/// A tool call and its result have to survive the round trip into the template,
/// which is the part `render_message_native` exists for. One system message, so
/// this is about the assistant/tool turns and not about #175.
#[test]
fn a_react_round_renders() {
    for f in FIXTURES.iter().filter(|f| f.registers) {
        let prompt = render(f, &react_round(), &ReasoningParams::default(), true)
            .unwrap_or_else(|e| panic!("{}: a completed ReAct round must render: {e}", f.name));

        assert!(
            prompt.contains("Read"),
            "{}: the tool call's name is missing from the rendered prompt",
            f.name
        );
        assert!(
            prompt.contains("a.txt"),
            "{}: the tool call's argument is missing from the rendered prompt",
            f.name
        );
        assert!(
            prompt.contains("hello"),
            "{}: the tool result is missing from the rendered prompt",
            f.name
        );
    }
}

/// KV cache reuse (#86, #172) is worth only as much as iteration *N*'s prompt
/// is a prefix of *N+1*'s, and that is a property of the **template**, not of
/// the conversation: Qwen3.8's own has a backwards scan for the last user query
/// (`ns.last_query_index`) that, under `preserve_thinking = false`, decides
/// whether an *earlier* assistant turn keeps its think block. A template like
/// that re-renders history differently as the conversation grows, and no amount
/// of care on gallium's side makes the prefix hold.
///
/// The generation prompt is excluded because it is exactly the part that is not
/// shared: it is the trailing assistant header of render *N*, and the assistant
/// turn itself in render *N+1*.
#[test]
fn history_renders_as_a_prefix_of_itself() {
    let mut checked = 0;
    for f in FIXTURES.iter().filter(|f| f.registers) {
        let full = react_round();
        let Ok(later) = render(f, &full, &ReasoningParams::default(), false) else {
            continue; // covered by the tests above; not this one's subject
        };
        let Ok(earlier) = render(f, &full[..2], &ReasoningParams::default(), false) else {
            continue;
        };
        checked += 1;

        assert!(
            later.starts_with(&earlier),
            "{}: adding an assistant turn re-rendered the history before it, so \
             iteration N's prompt is not a prefix of N+1's and every cached token \
             is discarded.\n--- earlier ---\n{earlier}\n--- later ---\n{later}",
            f.name
        );
    }

    // Every `continue` above is a template excused for a reason another test
    // owns. If they all take one, this test proves nothing and should say so
    // rather than pass.
    assert!(
        checked > 0,
        "no fixture rendered, so prefix stability went unchecked"
    );
}

/// `ReasoningEffort` has five variants; a template may accept fewer, and at
/// least one raises on the rest rather than clamping (#176). A raise here is
/// not a visible error — `build_prompt` catches it and silently switches the
/// model to a different tool protocol — so the accepted set has to be known
/// per family rather than discovered in production.
#[test]
fn reasoning_effort_values_the_template_accepts() {
    // Every spelling `profile::ReasoningEffort` can produce, lowercased as
    // `ReasoningParams::effort_text` would carry it.
    const GALLIUM_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

    for f in FIXTURES.iter().filter(|f| f.registers) {
        let Some(accepted) = f.reasoning_efforts else {
            continue; // template never reads the variable
        };

        for effort in GALLIUM_EFFORTS {
            let params = ReasoningParams {
                thinking: Some(true),
                effort_text: Some(effort),
                preserve_thinking: None,
            };
            let messages = vec![
                ChatMessage::system("OPERATOR SYSTEM PROMPT".to_string()),
                ChatMessage::user("hi".to_string()),
            ];

            let ok = render(f, &messages, &params, true).is_ok();
            assert_eq!(
                ok,
                accepted.contains(effort),
                "{}: reasoning_effort = {effort:?} rendered = {ok}, but the \
                 fixture declares the accepted set as {accepted:?}. A value \
                 outside it raises, and a raise is a silent fallback to the \
                 JSON-prose tool protocol — see #176.",
                f.name
            );
        }
    }
}

/// `render_message_native` is what turns gallium's `ChatMessage` into the
/// object a template indexes into, and every fixture above depends on it
/// producing the keys those templates read. Pinned directly so a change there
/// fails here rather than in three render tests at once.
#[test]
fn a_tool_call_message_carries_the_keys_templates_read() {
    let msg = ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
        id: "call_1".to_string(),
        name: "Read".to_string(),
        arguments: json!({"file_path": "a.txt"}),
    }]);

    let v = LlamaLocalProvider::render_message_native(&msg);
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["tool_calls"][0]["function"]["name"], "Read");
    assert_eq!(
        v["tool_calls"][0]["function"]["arguments"]["file_path"],
        "a.txt"
    );

    let result =
        ChatMessage::tool_result("call_1".to_string(), "Read".to_string(), "hi".to_string());
    let v = LlamaLocalProvider::render_message_native(&result);
    assert_eq!(v["role"], "tool");
    assert_eq!(v["content"], "hi");
    assert_eq!(v["tool_call_id"], "call_1");
}

/// The rendered LFM2 prompt, printed rather than asserted field by field: what
/// this model's own template produces is the thing #182 was hiding, and a
/// reviewer should be able to see it. Asserts only the structure that identifies
/// it as LFM2's own layout rather than the manual ChatML fallback.
#[test]
fn lfm2_renders_its_own_layout() {
    let f = FIXTURES
        .iter()
        .find(|f| f.name == "lfm2-8b-a1b.jinja")
        .expect("the LFM2 fixture");

    let prompt = render(f, &react_round(), &ReasoningParams::default(), true)
        .expect("LFM2's template must render");
    println!("{prompt}");

    // Its own tool wire format, which the ChatML fallback cannot produce: the
    // fallback has no concept of a tool call and folds one into prose.
    assert!(
        prompt.contains("<|tool_call_start|>"),
        "expected LFM2's native tool-call markers, got:\n{prompt}"
    );
    // The Python-ish call list `wire::python` exists to read — and the shape
    // the model has been trained on while gallium showed it JSON prose
    // instead. This is the evidence #118 was missing.
    assert!(
        prompt.contains("[Read(file_path='a.txt')]"),
        "expected LFM2's Python-ish call list, got:\n{prompt}"
    );
    // A tool result is a plain `tool` turn here — this template has no
    // `<|tool_response_start|>` of its own, which is worth pinning so nobody
    // adds one on the strength of the tool-*call* markers above.
    assert!(
        prompt.contains("<|im_start|>tool\nhello<|im_end|>"),
        "expected a plain tool turn, got:\n{prompt}"
    );
    // The tools it was given are declared in the system turn by the template
    // itself, not by gallium's JSON-prose instruction block.
    assert!(
        prompt.contains("List of tools:"),
        "expected the template's own tool declaration, got:\n{prompt}"
    );
}

/// The merge, seen. Qwen3.8's template used to be the one that refused gallium's
/// four system messages, so this was `render_native_prompt`'s retry. The GGUF's
/// own bytes (unsloth's patched template) merge the leading system run
/// themselves, so the seam is now the *template's* choice, not gallium's — and
/// the question is the same one: are the four authors still distinguishable
/// afterwards. unsloth joins with a single newline, not the blank line
/// `merge_system_messages` uses.
#[test]
fn qwen38_renders_the_merged_system_block() {
    let f = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.8.jinja")
        .expect("the Qwen3.8 fixture");
    assert!(
        f.admits_extra_system_messages,
        "this test is about the template's own merge; if it has started \
         raising again, gallium's retry owns the merge and this test should \
         assert blank-line separation instead"
    );

    let mut messages = gallium_system_messages();
    messages.push(ChatMessage::user("read a.txt".to_string()));
    let prompt = render(f, &messages, &ReasoningParams::default(), true).expect("must render");
    println!("{prompt}");

    // One system turn, carrying all four in order, newline separated.
    assert_eq!(
        prompt.matches("<|im_start|>system").count(),
        1,
        "expected exactly one system turn:\n{prompt}"
    );
    assert!(
        prompt
            .contains("PROFILE PREAMBLE\nOPERATOR SYSTEM PROMPT\nPROJECT AGENTS.md\nSKILL CATALOG"),
        "expected the four system messages in order, newline separated:\n{prompt}"
    );
}

/// The reasoning the model actually produced has to reach a template that
/// renders prior-turn thinking. Qwen3.8's does, unconditionally
/// (`preserve_thinking` is undefined unless someone sets it, and the branch
/// short-circuits on that), so before #177 every prior assistant turn arrived
/// as `<think>\n\n</think>` — a claim about the model's own reasoning, made to
/// the model, and false.
#[test]
fn prior_reasoning_reaches_a_template_that_renders_it() {
    let f = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.8.jinja")
        .expect("the Qwen3.8 fixture");

    let prompt = render(f, &react_round(), &ReasoningParams::default(), true).expect("must render");
    assert!(
        prompt.contains("<think>\nThe file is small, so reading it whole is fine.\n</think>"),
        "the assistant turn's reasoning is missing from the rendered prompt:\n{prompt}"
    );
    assert!(
        !prompt.contains("<think>\n\n</think>\n\n<tool_call>"),
        "an empty think block was rendered where real reasoning existed:\n{prompt}"
    );
}

/// Nothing to report must not become "reasoned nothing": a turn with no
/// reasoning omits the key entirely rather than passing an empty string, so a
/// template's own `is string` / `is defined` branch decides what to do.
#[test]
fn a_turn_without_reasoning_omits_the_key() {
    let msg = ChatMessage::assistant_tool_calls(vec![ToolCallInfo {
        id: "call_1".to_string(),
        name: "Read".to_string(),
        arguments: json!({"file_path": "a.txt"}),
    }]);
    let v = LlamaLocalProvider::render_message_native(&msg);
    assert!(
        v.get("reasoning_content").is_none(),
        "expected no reasoning_content key at all, got {v}"
    );

    let v = LlamaLocalProvider::render_message_native(&msg.with_reasoning(Some("why".to_string())));
    assert_eq!(v["reasoning_content"], "why");
}
/// Google's guidance for Gemma is that prior turns must carry only the final
/// response: "Ensure that no generated thoughts from previous turns remain in
/// the context window before the next user turn begins."
/// (<https://ai.google.dev/gemma/docs/capabilities/thinking>)
///
/// Gemma's own template enforces that, and gallium relies on it rather than
/// duplicating the policy:
///
/// ```jinja
/// {%- set preserve_thinking = preserve_thinking | default(false) -%}
/// ...
/// {%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx)
///                         or (preserve_thinking and message.get('tool_calls')) -%}
/// ```
///
/// So supplying `reasoning_content` is both safe and wanted here: the current
/// turn's reasoning reaches the model, which is what keeps a multi-step tool
/// sequence coherent, and every earlier turn's is dropped by the gate. This
/// test is the check on that claim, since it is the reason gallium does not
/// gate reasoning itself.
#[test]
fn gemma_drops_prior_turn_reasoning_and_keeps_the_current_turns() {
    let f = FIXTURES
        .iter()
        .find(|f| f.name == "gemma4-e4b.jinja")
        .expect("the Gemma 4 fixture");

    let call = |id: &str, path: &str| {
        vec![ToolCallInfo {
            id: id.to_string(),
            name: "Read".to_string(),
            arguments: json!({ "file_path": path }),
        }]
    };
    let messages = vec![
        ChatMessage::system("SYS".to_string()),
        ChatMessage::user("first question".to_string()),
        ChatMessage::assistant_tool_calls(call("c1", "a.txt"))
            .with_reasoning(Some("OLD-TURN-THOUGHT".to_string())),
        ChatMessage::tool_result("c1".to_string(), "Read".to_string(), "x".to_string()),
        ChatMessage::user("second question".to_string()),
        ChatMessage::assistant_tool_calls(call("c2", "b.txt"))
            .with_reasoning(Some("CURRENT-TURN-THOUGHT".to_string())),
    ];

    let prompt = render(f, &messages, &ReasoningParams::default(), true).expect("must render");
    assert!(
        !prompt.contains("OLD-TURN-THOUGHT"),
        "a previous turn's thoughts reached the context window, which Gemma's \
         own guidance forbids:\n{prompt}"
    );
    assert!(
        prompt.contains("CURRENT-TURN-THOUGHT"),
        "the current turn's reasoning was dropped, so a multi-step tool \
         sequence loses its own context:\n{prompt}"
    );
}

/// The profile and the template, checked against each other rather than each
/// against its own idea of the other.
///
/// `Qwen3::reasoning_params` clamps gallium's five-point scale onto the three
/// values this family's template accepts (#176). That clamp is only correct if
/// the template agrees, and the template is right here — so ask it. Every
/// `ReasoningEffort` must produce a prompt, from the real profile through the
/// real template, with no raise.
///
/// A raise would not surface as an error: `build_prompt` catches it and asks
/// the model for JSON prose instead. `configs/default.toml` ships
/// `reasoningEffort = "high"`, so this is not a hypothetical value.
#[test]
fn every_reasoning_effort_renders_through_its_own_profile() {
    use crate::profile::{ModelProfile, Qwen3, ReasoningEffort};

    let f = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.8.jinja")
        .expect("the Qwen3.8 fixture");

    for effort in [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ] {
        let params = Qwen3.reasoning_params(effort);
        let messages = vec![
            ChatMessage::system("OPERATOR SYSTEM PROMPT".to_string()),
            ChatMessage::user("hi".to_string()),
        ];
        render(f, &messages, &params, true).unwrap_or_else(|e| {
            panic!(
                "{effort:?} → {params:?} does not render against {}: {e}",
                f.name
            )
        });
    }
}

/// The two backends must mean the same thing by the same setting.
///
/// `Qwen3::reasoning_params` sets both axes now, and the llama.cpp path reads
/// them through the model's own template while the candle path hand-renders
/// that template in `QwenProtocol`. Nothing but a test keeps the second
/// faithful to the first — and the divergence this guards was real: the candle
/// renderer read only `thinking`, so `Medium` through `Max` were four distinct
/// prompts on llama.cpp and one prompt on candle, from one config value.
///
/// Compared on the *reasoning instruction* rather than the whole prompt: the
/// two renderers legitimately differ in layout, and what has to agree is which
/// instruction the model is given.
#[test]
fn both_backends_render_the_same_reasoning_instruction() {
    use crate::llm::ToolDefinition;
    use crate::profile::{ModelProfile, Qwen3, ReasoningEffort};
    use crate::protocol::{PromptRenderer, QwenProtocol};

    let f = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.8.jinja")
        .expect("the Qwen3.8 fixture");

    // The sentence the template emits, if any. Located by its own opening
    // words so this reads the rendered prompt rather than trusting a constant.
    fn instruction(prompt: &str) -> Option<String> {
        let start = prompt.find("Reasoning effort is set to")?;
        let rest = &prompt[start..];
        let end = rest.find("\n").unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }

    let messages = vec![
        ChatMessage::system("OPERATOR SYSTEM PROMPT".to_string()),
        ChatMessage::user("hi".to_string()),
    ];
    let tools: Vec<ToolDefinition> = tools();
    let mut instructed: Vec<ReasoningEffort> = Vec::new();

    for effort in [
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ] {
        let params = Qwen3.reasoning_params(effort);

        let via_template =
            instruction(&render(f, &messages, &params, true).expect("the template must render"));

        let candle =
            QwenProtocol::with_reasoning(params.thinking.unwrap_or(true), params.effort_text);
        let via_candle = instruction(&candle.format_prompt_with_tools(&messages, &tools));

        assert_eq!(
            via_template, via_candle,
            "{effort:?} → {params:?} instructs the model differently on the two backends"
        );
        if via_template.is_some() {
            instructed.push(effort);
        }
    }

    // Which levels are instructed, not how many — a count is unreadable here,
    // because two different scales meet in this projection and it is easy to
    // read one for the other. Gallium's five levels map onto the template's
    // four states, and the template writes a sentence for only two of *its*
    // values:
    //
    //   Low    → thinking off      → no sentence (the guard skips it entirely)
    //   Medium → `low`             → "Reasoning effort is set to low. …"
    //   High   → `medium`          → no sentence (medium is the absence of one)
    //   XHigh  → `xhigh`           → "Reasoning effort is set to xhigh. …"
    //   Max    → `xhigh`           → same
    //
    // So three of gallium's levels carry an instruction while two of the
    // template's values produce one. Naming them makes a mapping change fail
    // with the answer rather than with a number.
    assert_eq!(
        instructed,
        vec![
            ReasoningEffort::Medium,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max
        ],
        "the set of levels carrying a reasoning instruction changed; if that was \
         intended, `Qwen3::reasoning_params` moved and this list should follow it"
    );
}

/// Two conversation turns: an older assistant turn that reasoned, a new user
/// question, and the current turn's reasoning. The shape that separates "keeps
/// prior thinking" from "keeps this turn's thinking", which every template
/// surveyed distinguishes and no single-turn conversation can show.
fn two_turns_with_reasoning() -> Vec<ChatMessage> {
    let call = |id: &str, path: &str| {
        vec![ToolCallInfo {
            id: id.to_string(),
            name: "Read".to_string(),
            arguments: json!({ "file_path": path }),
        }]
    };
    vec![
        ChatMessage::system("SYS".to_string()),
        ChatMessage::user("first question".to_string()),
        ChatMessage::assistant_tool_calls(call("c1", "a.txt"))
            .with_reasoning(Some("OLD-TURN-THOUGHT".to_string())),
        ChatMessage::tool_result("c1".to_string(), "Read".to_string(), "x".to_string()),
        ChatMessage::user("second question".to_string()),
        ChatMessage::assistant_tool_calls(call("c2", "b.txt"))
            .with_reasoning(Some("CURRENT-TURN-THOUGHT".to_string())),
    ]
}

/// Each family's `preserve_prior_reasoning` is what actually happens to that
/// family's prompt — the profile and the template asked the same question and
/// required to give the same answer.
///
/// **This checks the wiring, not the policy**, and the difference matters:
/// changing a profile's answer moves both sides of the comparison together, so
/// this test cannot notice it. What it does catch is the policy failing to
/// reach the prompt at all — which is the whole mechanism, and which is exactly
/// how `reasoning_content` was silently doing nothing for Gemma before #185.
/// The values themselves are pinned in
/// `profile::tests::prior_reasoning_policy_is_named_by_exactly_the_families_that_have_one`.
///
/// The current turn's reasoning must survive everywhere regardless: that is a
/// separate gate in every one of these templates (`loop.index0 >
/// last_user_idx`), and it is what keeps a multi-step tool sequence coherent.
#[test]
fn each_family_gets_the_prior_reasoning_policy_its_profile_states() {
    for f in FIXTURES.iter().filter(|f| f.registers) {
        let params = ReasoningParams {
            preserve_thinking: f.profile.preserve_prior_reasoning(),
            ..ReasoningParams::default()
        };
        let prompt = render(f, &two_turns_with_reasoning(), &params, true)
            .unwrap_or_else(|e| panic!("{}: must render: {e}", f.name));

        assert!(
            prompt.contains("CURRENT-TURN-THOUGHT"),
            "{}: the current turn's own reasoning was dropped, which no policy \
             here asks for:\n{prompt}",
            f.name
        );

        // Two things have to agree for a prior turn to keep its reasoning: the
        // family's policy says so, *and* the template reads the variable that
        // carries it. Neither alone decides, which is why both are asked rather
        // than one being hardcoded.
        let preserves =
            f.profile.preserve_prior_reasoning() == Some(true) && f.honors_preserve_thinking;
        assert_eq!(
            prompt.contains("OLD-TURN-THOUGHT"),
            preserves,
            "{}: prior-turn reasoning present = {}, expected {preserves}\n{prompt}",
            f.name,
            prompt.contains("OLD-TURN-THOUGHT")
        );
    }
}

/// `honors_preserve_thinking` is a claim about the file, so let the file check
/// it — a declaration that drifts from its fixture is worse than no
/// declaration, because the tests above read as though they verified it.
#[test]
fn the_declared_preserve_thinking_support_matches_the_templates() {
    for f in FIXTURES {
        assert_eq!(
            f.src.contains("preserve_thinking"),
            f.honors_preserve_thinking,
            "{}: fixture declares honors_preserve_thinking = {}",
            f.name,
            f.honors_preserve_thinking
        );
    }
}

/// The two generations behind one profile disagree, and the profile's single
/// answer is right for both — because the older template does not read the
/// variable at all.
///
/// Worth pinning rather than trusting: `Qwen3` answers `Some(true)` on the
/// strength of Qwen3.8's template, and the same profile also serves Qwen 3.6
/// (`Qwen/Qwen3.5-9B`), whose gate is `loop.index0 > ns.last_query_index` with
/// no `preserve_thinking` escape. If a future generation grows one with the
/// opposite default, one profile stops being able to speak for both and this
/// test is what says so.
#[test]
fn one_qwen_answer_covers_both_generations() {
    let old = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.5-9b.jinja")
        .expect("the Qwen 3.6 fixture");
    assert!(
        !old.src.contains("preserve_thinking"),
        "the older template now reads preserve_thinking, so one profile answer \
         may no longer be right for both generations — see #188"
    );

    // Asking it to preserve changes nothing there.
    for preserve in [None, Some(false), Some(true)] {
        let params = ReasoningParams {
            preserve_thinking: preserve,
            ..ReasoningParams::default()
        };
        let prompt = render(old, &two_turns_with_reasoning(), &params, true).expect("must render");
        assert!(
            !prompt.contains("OLD-TURN-THOUGHT"),
            "preserve_thinking = {preserve:?} carried a prior turn's reasoning \
             into a template that has no such variable:\n{prompt}"
        );
    }
}

/// The same guard as [`both_backends_render_the_same_reasoning_instruction`],
/// for the tool protocol — and for LFM2 it was guarding nothing until now,
/// because the two backends were not speaking the same protocol at all.
///
/// The candle renderer (`Lfm2Protocol`) has always declared tools natively:
/// `List of tools: […]` in the system message, prior calls as
/// `<|tool_call_start|>[name(arg='v')]<|tool_call_end|>`. The llama.cpp backend
/// renders through this fixture, but only when the profile says the template
/// formats tools itself — and `Lfm2::template_formats_tools_natively` returned
/// `false`, so gallium injected its JSON-prose instructions instead. One model,
/// one template, two wire formats, decided by which backend loaded it: `coding`
/// passed on neither and `refactoring` needed a JSON shape nobody asks for.
///
/// So this asserts the three protocol facts both renderers must agree on. It is
/// deliberately not a whole-prompt comparison — the layouts legitimately differ
/// (candle has no jinja and builds its own), and what has to agree is what the
/// model is asked for.
#[test]
fn both_backends_ask_lfm2_for_the_same_tool_protocol() {
    use crate::profile::Lfm2;
    use crate::protocol::{Lfm2Protocol, PromptRenderer};

    let f = FIXTURES
        .iter()
        .find(|f| f.name == "lfm2-8b-a1b.jinja")
        .expect("the LFM2 fixture");

    assert!(
        Lfm2.template_formats_tools_natively(f.src),
        "the template renders its own tool format; a profile that says otherwise \
         sends this model gallium's JSON prose on llama.cpp and its native format \
         on candle"
    );

    let messages = react_round();
    let via_template = render(f, &messages, &ReasoningParams::default(), true)
        .expect("the LFM2 template must render");
    let via_candle = Lfm2Protocol.format_prompt_with_tools(&messages, &tools());

    /// The `<|tool_call_start|>…<|tool_call_end|>` span, which is the wire
    /// format itself rather than a fact about where it sits in the prompt.
    fn call_span(prompt: &str) -> Option<&str> {
        let start = prompt.find("<|tool_call_start|>")?;
        let end = prompt[start..].find("<|tool_call_end|>")? + start;
        Some(&prompt[start..end + "<|tool_call_end|>".len()])
    }

    for (backend, prompt) in [("llama.cpp", &via_template), ("candle", &via_candle)] {
        assert!(
            prompt.contains("List of tools: ["),
            "{backend} does not declare the tools in this template's own form:\n{prompt}"
        );
        assert!(
            prompt.contains("<|im_start|>tool\n"),
            "{backend} does not return the result in a tool turn:\n{prompt}"
        );
    }

    assert_eq!(
        call_span(&via_template),
        call_span(&via_candle),
        "the two backends replay the model's own call differently, so its next \
         prompt teaches it a format only one of them uses"
    );
    assert_eq!(
        call_span(&via_template),
        Some("<|tool_call_start|>[Read(file_path='a.txt')]<|tool_call_end|>"),
        "the wire format moved; `wire::python` is what reads this back"
    );
}

/// The third of these backend-parity guards, and the one that was failing
/// silently in production rather than in a testcase.
///
/// A real turn carries **several** system messages — a profile's agent preamble,
/// the operator's prompt, the project's `AGENTS.md` / `CLAUDE.md`, the skill
/// catalog. `render_native_prompt` merges them for a template that admits one
/// (#184). Every renderer in `protocol.rs` took the *first* with `find_map` and
/// dropped the rest, so a candle turn went to the model without its project
/// context or its skills while the llama.cpp turn had both — and nothing said
/// so, because the testsuite's shim strips `[agent]` and sends exactly one.
#[test]
fn both_backends_carry_every_system_message() {
    use crate::protocol::{GemmaProtocol, Lfm2Protocol, PromptRenderer};

    let messages = vec![
        ChatMessage::system("PREAMBLE".to_string()),
        ChatMessage::system("OPERATOR PROMPT".to_string()),
        ChatMessage::system("PROJECT CONTEXT".to_string()),
        ChatMessage::user("hi".to_string()),
    ];
    let tools = tools();
    let carried = |prompt: &str, backend: &str| {
        for expected in ["PREAMBLE", "OPERATOR PROMPT", "PROJECT CONTEXT"] {
            assert!(
                prompt.contains(expected),
                "{backend} dropped the system message {expected:?}:\n{prompt}"
            );
        }
    };

    // llama.cpp, through the real templates.
    for name in ["lfm2-8b-a1b.jinja", "gemma4-e4b.jinja"] {
        let f = FIXTURES.iter().find(|f| f.name == name).expect("fixture");
        assert!(
            f.admits_extra_system_messages || f.registers,
            "{name} is declared unable to render this shape at all"
        );
        let rendered = render(f, &messages, &ReasoningParams::default(), true)
            .unwrap_or_else(|e| panic!("{name} must render: {e}"));
        carried(&rendered, name);
    }

    // candle, through the hand-written renderers for the same two families.
    carried(
        &Lfm2Protocol.format_prompt_with_tools(&messages, &tools),
        "Lfm2Protocol",
    );
    carried(
        &GemmaProtocol::new().format_prompt_with_tools(&messages, &tools),
        "GemmaProtocol",
    );
    // And on the no-tools path, which is a separate function in each renderer.
    carried(
        &Lfm2Protocol.format_prompt(&messages),
        "Lfm2Protocol (no tools)",
    );
    carried(
        &GemmaProtocol::new().format_prompt(&messages),
        "GemmaProtocol (no tools)",
    );
}

/// What the candle renderers do with a *prior* turn's reasoning, measured
/// against each family's stated policy rather than assumed.
///
/// The renderers in `protocol.rs` drop it unconditionally — they hand-build the
/// prompt and never read `ChatMessage::reasoning`. For the two families that run
/// on candle today that is not a gap but the correct answer:
/// `Lfm2::preserve_prior_reasoning` and `Gemma4`'s are both `Some(false)`, and
/// their own templates gate prior thinking off on the llama.cpp side too.
///
/// `Qwen3` says `Some(true)` and gets dropped anyway. That is the real gap, and
/// it is asserted here in the state it is actually in — declared, not skipped,
/// so closing it turns this test red for the right reason. Nothing exercises it:
/// no config runs a Qwen model through candle.
#[test]
fn candle_drops_prior_reasoning_and_only_qwen_minds() {
    use crate::profile::{Gemma4, Lfm2, ModelProfile, Qwen3};
    use crate::protocol::{GemmaProtocol, Lfm2Protocol, PromptRenderer, QwenProtocol};

    let messages = two_turns_with_reasoning();
    let tools = tools();
    let dropped = |prompt: &str| !prompt.contains("OLD-TURN-THOUGHT");

    assert_eq!(Lfm2.preserve_prior_reasoning(), Some(false));
    assert_eq!(Gemma4.preserve_prior_reasoning(), Some(false));
    assert!(
        dropped(&Lfm2Protocol.format_prompt_with_tools(&messages, &tools)),
        "LFM2 says drop prior reasoning and the renderer kept it"
    );
    assert!(
        dropped(&GemmaProtocol::new().format_prompt_with_tools(&messages, &tools)),
        "Gemma 4 says drop prior reasoning and the renderer kept it"
    );

    // The declared gap.
    assert_eq!(Qwen3.preserve_prior_reasoning(), Some(true));
    assert!(
        dropped(&QwenProtocol::new().format_prompt_with_tools(&messages, &tools)),
        "the candle Qwen renderer now carries prior reasoning — its profile has \
         always asked for that, so delete this assertion rather than fixing the \
         renderer back"
    );
}
