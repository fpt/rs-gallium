//! What a real chat template has to survive, asserted against the templates
//! themselves.
//!
//! Every template-level bug gallium has hit was found by loading a multi-GB
//! GGUF and reading the output — `configs/qwen3.8.toml` records one that cost a
//! testcase (`refactoring`), and issue #182 records one that had been silently
//! degrading a model since the day its config landed. None of them needed the
//! weights: a chat template is text, and the failures are in how gallium's
//! message shapes meet it.
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
use crate::profile::ReasoningParams;

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
    /// The `reasoning_effort` values this template accepts without raising.
    /// Empty means it never reads the variable, so any value is inert. See
    /// #176: gallium's `ReasoningEffort` has five variants and at least one
    /// family accepts three.
    reasoning_efforts: Option<&'static [&'static str]>,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "gemma4-e4b.jinja",
        src: include_str!("../tests/fixtures/chat_templates/gemma4-e4b.jinja"),
        registers: true,
        admits_extra_system_messages: true,
        reasoning_efforts: None,
    },
    Fixture {
        name: "lfm2-8b-a1b.jinja",
        src: include_str!("../tests/fixtures/chat_templates/lfm2-8b-a1b.jinja"),
        registers: true,
        admits_extra_system_messages: true,
        reasoning_efforts: None,
    },
    Fixture {
        name: "qwen3.8.jinja",
        src: include_str!("../tests/fixtures/chat_templates/qwen3.8.jinja"),
        registers: true,
        // #175: `raise_exception('System message must be at the beginning.')`
        admits_extra_system_messages: false,
        // #176: anything else raises, including gallium's `high` and `max`.
        reasoning_efforts: Some(&["low", "medium", "xhigh"]),
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

/// The merge, seen. Qwen3.8's template is the one that refuses gallium's four
/// system messages, so this is what `render_native_prompt`'s retry produces —
/// worth looking at rather than only counting, since the whole question is
/// whether the four authors are still distinguishable afterwards.
#[test]
fn qwen38_renders_the_merged_system_block() {
    let f = FIXTURES
        .iter()
        .find(|f| f.name == "qwen3.8.jinja")
        .expect("the Qwen3.8 fixture");
    assert!(
        !f.admits_extra_system_messages,
        "this test is about the retry; the fixture no longer needs it"
    );

    let mut messages = gallium_system_messages();
    messages.push(ChatMessage::user("read a.txt".to_string()));
    let prompt = render(f, &messages, &ReasoningParams::default(), true).expect("must render");
    println!("{prompt}");

    // One system turn, carrying all four in order, blank-line separated.
    assert_eq!(
        prompt.matches("<|im_start|>system").count(),
        1,
        "expected exactly one system turn:\n{prompt}"
    );
    assert!(
        prompt.contains(
            "PROFILE PREAMBLE\n\nOPERATOR SYSTEM PROMPT\n\nPROJECT AGENTS.md\n\nSKILL CATALOG"
        ),
        "expected the four system messages in order, blank-line separated:\n{prompt}"
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
