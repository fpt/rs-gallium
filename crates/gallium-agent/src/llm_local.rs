//! In-process llama.cpp provider via FFI (llama-cpp-2 crate).
//!
//! Loads a GGUF model directly — no server needed.
//!
//! Tool calling: llama-cpp-2 0.1.150 removed the OAI-compat chat layer
//! (`apply_chat_template_oaicompat` / `ChatTemplateResult` / `parse_response_oaicompat`),
//! so we implement tool calling ourselves: the available tools are injected into
//! the system prompt with a JSON output protocol, and the model's reply is parsed
//! leniently for a tool-call object. The model's own jinja chat template (from the
//! GGUF) is still used to format the conversation via `apply_chat_template`, which
//! only accepts role+content messages.

use anyhow::Result;
use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde_json::Value;

use crate::llm::{ChatMessage, ChatRole, LlmProvider, LlmResponse, TokenUsage, ToolCallInfo, ToolDefinition};

pub struct LlamaLocalProvider {
    backend: LlamaBackend,
    model: LlamaModel,
    /// The model's embedded jinja chat template (rendered via minijinja). None if
    /// the GGUF has no template — then we fall back to a manual ChatML format.
    template_src: Option<String>,
    /// Literal BOS/EOS token text (e.g. "<bos>", "<eos>"), fed to the template.
    bos: String,
    eos: String,
    temperature: f32,
    max_tokens: u32,
    n_ctx: u32,
}

// LlamaModel is Send+Sync. LlamaBackend and LlamaChatTemplate are safe to share.
unsafe impl Send for LlamaLocalProvider {}
unsafe impl Sync for LlamaLocalProvider {}

impl LlamaLocalProvider {
    pub fn new(
        model_path: &str,
        temperature: f32,
        max_tokens: u32,
        n_ctx: u32,
    ) -> Result<Self> {
        tracing::info!("Initializing local llama.cpp provider (FFI)");
        tracing::info!("  Model path: {}", model_path);
        tracing::info!("  Context size: {}", n_ctx);

        let mut backend = LlamaBackend::init()
            .map_err(|e| anyhow::anyhow!("Failed to init llama backend: {}", e))?;
        backend.void_logs();

        // On iOS simulator, Metal doesn't support residency sets — use CPU only.
        // Elsewhere, offload layers to the GPU backend (Metal/CUDA/Vulkan,
        // depending on the build features). On a CPU-only build these layers are
        // simply ignored by llama.cpp.
        let use_gpu = if cfg!(target_os = "ios") && cfg!(target_abi = "sim") {
            tracing::info!("  iOS simulator detected — using CPU only (no GPU)");
            false
        } else {
            true
        };

        if !use_gpu {
            // Prevent Metal residency set assertions on simulator
            unsafe { std::env::set_var("GGML_METAL_NO_RESIDENCY", "1"); }
        }

        // Layers to offload to the GPU. Override with KESSEL_GPU_LAYERS to
        // fit a smaller VRAM budget (e.g. a 6 GB card can't hold a 5 GB model
        // plus KV cache, so partial offload like 20 avoids an OOM). Default 999
        // = offload everything.
        let gpu_layers: u32 = if use_gpu {
            std::env::var("KESSEL_GPU_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(999)
        } else {
            0
        };
        tracing::info!("  GPU layers to offload: {}", gpu_layers);
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(gpu_layers);

        let model = LlamaModel::load_from_file(&backend, Path::new(model_path), &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

        tracing::info!("  Model loaded: {} params", model.n_params());
        tracing::info!("  Context train: {}", model.n_ctx_train());

        let template_src = match model.chat_template(None).and_then(|t| Ok(t.to_string()?)) {
            Ok(src) => Some(strip_unsupported_jinja(&src)),
            Err(_) => {
                tracing::warn!("No chat template in model, using ChatML fallback");
                None
            }
        };

        // Literal BOS/EOS strings (e.g. "<bos>"/"<eos>") so the jinja template's
        // `{{ bos_token }}`/`{{ eos_token }}` render to real special tokens.
        let mut dec = encoding_rs::UTF_8.new_decoder();
        let bos = model
            .token_to_piece(model.token_bos(), &mut dec, true, None)
            .unwrap_or_default();
        let mut dec = encoding_rs::UTF_8.new_decoder();
        let eos = model
            .token_to_piece(model.token_eos(), &mut dec, true, None)
            .unwrap_or_default();

        Ok(Self {
            backend,
            model,
            template_src,
            bos,
            eos,
            temperature,
            max_tokens,
            n_ctx,
        })
    }

    /// Render the conversation through the model's chat template, injecting tool
    /// definitions into the system message when tools are available.
    fn build_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<String> {
        // Gemma-4-style templates format tools natively (`<|tool>declaration:…`,
        // `<|tool_call>`, `<|tool_response>`). Feed those templates structured
        // tool inputs so the model sees tools in the exact form it was trained
        // on. Fall back to the generic JSON-prose protocol if it doesn't render.
        if let Some(tools) = tools {
            if self.template_supports_native_tools() {
                match self.render_native(messages, tools) {
                    Ok(prompt) => return Ok(prompt),
                    Err(e) => tracing::warn!(
                        "native tool template render failed ({e}); using JSON-prose fallback"
                    ),
                }
            }
        }

        // (role, content) pairs using only system/user/assistant roles, which
        // every chat template supports (a "tool" role is not universal).
        let mut pairs: Vec<(&'static str, String)> =
            messages.iter().map(Self::render_message).collect();

        if let Some(tools) = tools {
            let instr = Self::tool_instructions(tools);
            if let Some(sys) = pairs.iter_mut().find(|(role, _)| *role == "system") {
                sys.1 = format!("{}\n\n{}", sys.1, instr);
            } else {
                pairs.insert(0, ("system", instr));
            }
        }

        // No embedded template: go straight to the manual ChatML fallback.
        if self.template_src.is_none() {
            return Ok(self.chatml_fallback(&Self::fold_system(pairs)));
        }

        // Render the model's own jinja template. If it rejects the system role
        // (e.g. gemma calls raise_exception), fold system into the first user
        // turn and retry; if it still fails, fall back to manual ChatML.
        match self.render_template(&pairs) {
            Ok(prompt) => return Ok(prompt),
            Err(e) => tracing::debug!("template render failed ({e}); folding system and retrying"),
        }
        let folded = Self::fold_system(pairs);
        match self.render_template(&folded) {
            Ok(prompt) => Ok(prompt),
            Err(e) => {
                tracing::warn!("template render failed after system-fold ({e}); using ChatML fallback");
                Ok(self.chatml_fallback(&folded))
            }
        }
    }

    /// Build a minijinja environment with the model's chat template registered.
    fn jinja_env(&self) -> std::result::Result<minijinja::Environment<'static>, minijinja::Error> {
        let src = self.template_src.as_deref().ok_or_else(|| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "no chat template")
        })?;

        let mut env = minijinja::Environment::new();
        // Support Python-ish str methods (.strip/.startswith/.split/.get/...).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Templates call raise_exception(...) to reject unsupported inputs.
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
                Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
            },
        );
        // Some newer templates call strftime_now(fmt); a stub is sufficient here.
        env.add_function(
            "strftime_now",
            |_fmt: String| -> std::result::Result<String, minijinja::Error> { Ok(String::new()) },
        );
        env.add_template_owned("chat", src.to_string())?;
        Ok(env)
    }

    /// Render the model's embedded jinja chat template with minijinja.
    fn render_template(
        &self,
        pairs: &[(&'static str, String)],
    ) -> std::result::Result<String, minijinja::Error> {
        let env = self.jinja_env()?;
        let messages: Vec<Value> = pairs
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();

        let tmpl = env.get_template("chat")?;
        tmpl.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => self.bos,
            eos_token => self.eos,
        })
    }

    /// True if the embedded chat template formats tools in the Gemma-4 native
    /// protocol, so we can feed it structured tools rather than JSON prose.
    fn template_supports_native_tools(&self) -> bool {
        self.template_src.as_deref().map_or(false, |s| {
            s.contains("<|tool_call>") || s.contains("<|tool>") || s.contains("declaration:")
        })
    }

    /// Render via the model's native tool protocol: pass the OpenAI-style tools
    /// array and full message objects (with `tool_calls` / tool results) so the
    /// template emits `<|tool>declaration:…`, `<|tool_call>`, `<|tool_response>`.
    fn render_native(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<String> {
        let env = self.jinja_env().map_err(|e| anyhow::anyhow!("jinja env: {e}"))?;

        let msgs: Vec<Value> = messages.iter().map(Self::render_message_native).collect();
        let tool_defs: Vec<Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();

        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("get template: {e}"))?;
        let rendered = tmpl
            .render(minijinja::context! {
                messages => msgs,
                tools => tool_defs,
                add_generation_prompt => true,
                bos_token => self.bos,
                eos_token => self.eos,
            })
            .map_err(|e| anyhow::anyhow!("render: {e}"))?;
        tracing::debug!("rendered {} tools via native Gemma tool protocol", tools.len());
        Ok(rendered)
    }

    /// Convert a ChatMessage to the message object the native template expects,
    /// preserving assistant `tool_calls` and `tool` results.
    fn render_message_native(msg: &ChatMessage) -> Value {
        match msg.role {
            ChatRole::System => serde_json::json!({"role": "system", "content": msg.content}),
            ChatRole::User => serde_json::json!({"role": "user", "content": msg.content}),
            ChatRole::Assistant => {
                let mut m = serde_json::json!({"role": "assistant", "content": msg.content});
                if let Some(calls) = &msg.tool_calls {
                    let tc: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            serde_json::json!({
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            })
                        })
                        .collect();
                    m["tool_calls"] = Value::Array(tc);
                }
                m
            }
            ChatRole::Tool => serde_json::json!({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
                "name": msg.tool_name,
            }),
        }
    }

    /// Last-resort manual ChatML layout when the embedded template is missing or
    /// won't render.
    fn chatml_fallback(&self, pairs: &[(&'static str, String)]) -> String {
        let mut out = String::new();
        out.push_str(&self.bos);
        for (role, content) in pairs {
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    /// Fold all system messages into the first user turn, for templates that
    /// don't support a system role (e.g. gemma).
    fn fold_system(pairs: Vec<(&'static str, String)>) -> Vec<(&'static str, String)> {
        let mut system = String::new();
        let mut rest: Vec<(&'static str, String)> = Vec::new();
        for (role, content) in pairs {
            if role == "system" {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&content);
            } else {
                rest.push((role, content));
            }
        }
        if system.is_empty() {
            return rest;
        }
        if let Some(first_user) = rest.iter_mut().find(|(role, _)| *role == "user") {
            first_user.1 = format!("{}\n\n{}", system, first_user.1);
        } else {
            rest.insert(0, ("user", system));
        }
        rest
    }

    /// Map a ChatMessage to a (role, content) pair. Assistant tool calls and tool
    /// results are folded into text so the model sees prior turns in the same
    /// protocol format we ask it to emit.
    fn render_message(msg: &ChatMessage) -> (&'static str, String) {
        match msg.role {
            ChatRole::System => ("system", msg.content.clone()),
            ChatRole::User => ("user", msg.content.clone()),
            ChatRole::Assistant => {
                if let Some(ref calls) = msg.tool_calls {
                    let json = Self::serialize_tool_calls(calls);
                    let content = if msg.content.is_empty() {
                        json
                    } else {
                        format!("{}\n{}", msg.content, json)
                    };
                    ("assistant", content)
                } else {
                    ("assistant", msg.content.clone())
                }
            }
            ChatRole::Tool => ("user", format!("Tool result: {}", msg.content)),
        }
    }

    /// The tool-use instruction block appended to the system prompt.
    fn tool_instructions(tools: &[ToolDefinition]) -> String {
        let list = Self::tools_to_json(tools);
        format!(
            "You have access to the following tools (described as JSON Schema):\n\
             {list}\n\n\
             To call a tool, respond with ONLY a single JSON object and nothing else:\n\
             {{\"name\": \"<tool_name>\", \"arguments\": {{ ...json args... }}}}\n\
             To call several tools at once, respond with a JSON array of such objects.\n\
             If no tool is needed, reply normally in plain text (do not output JSON)."
        )
    }

    /// Serialize ToolDefinitions to an OpenAI-style tools JSON array string.
    fn tools_to_json(tools: &[ToolDefinition]) -> String {
        let json_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();

        serde_json::to_string(&json_tools).unwrap_or_else(|_| "[]".to_string())
    }

    /// Serialize prior assistant tool calls into the protocol JSON.
    fn serialize_tool_calls(calls: &[ToolCallInfo]) -> String {
        let arr: Vec<Value> = calls
            .iter()
            .map(|c| serde_json::json!({"name": c.name, "arguments": c.arguments}))
            .collect();
        if arr.len() == 1 {
            arr[0].to_string()
        } else {
            Value::Array(arr).to_string()
        }
    }

    /// Core generation loop. Tokenize, decode, sample until EOG or max tokens.
    /// Returns (generated_text, token_usage).
    fn generate(&self, prompt: &str) -> Result<(String, TokenUsage)> {
        // The template usually emits {{ bos_token }} already; only add a BOS at
        // tokenization time if the prompt doesn't already start with it.
        let add_bos = if !self.bos.is_empty() && prompt.starts_with(self.bos.as_str()) {
            AddBos::Never
        } else {
            AddBos::Always
        };
        let tokens = self
            .model
            .str_to_token(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let n_prompt = tokens.len() as u32;
        let n_ctx = self.n_ctx.max(n_prompt + self.max_tokens);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {}", e))?;

        // Feed prompt tokens
        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        let last_index = tokens.len().saturating_sub(1) as i32;
        for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
            batch
                .add(token, i, &[0], i == last_index)
                .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {}", e))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Initial decode failed: {}", e))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature),
            LlamaSampler::dist(1234),
        ]);

        // Generate tokens
        let mut n_cur = batch.n_tokens();
        let batch_start = n_cur;
        let max_tokens = n_cur + self.max_tokens as i32;
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut generated_text = String::new();

        while n_cur <= max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.model.is_eog_token(token) {
                break;
            }

            // A single token that can't be decoded (e.g. an unused/control id)
            // shouldn't abort the whole generation — skip it and keep going.
            match self
                .model
                .token_to_piece_bytes(token, 8, false, None)
                .or_else(|_| self.model.token_to_piece_bytes(token, 256, false, None))
            {
                Ok(output_bytes) => {
                    let mut output_string = String::with_capacity(32);
                    let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                    generated_text.push_str(&output_string);
                }
                Err(e) => tracing::debug!("skipping undecodable token {token:?}: {e}"),
            }

            // Stop at Gemma-4 tool boundaries: once the model closes a tool call
            // (`<tool_call|>`) or emits a tool-response marker, stop so we can run
            // the tool instead of letting it hallucinate a result. These literals
            // are gemma-specific, so this is a no-op for other local models.
            if generated_text.ends_with("<tool_call|>")
                || generated_text.contains("<|tool_response>")
            {
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| anyhow::anyhow!("Failed to add generated token: {}", e))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
        }

        let n_output = (n_cur - batch_start) as u64;
        let usage = TokenUsage {
            input_tokens: n_prompt as u64,
            output_tokens: n_output,
            total_tokens: n_prompt as u64 + n_output,
        };
        tracing::info!(
            "Local LLM usage: input={}, output={}, total={}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens
        );

        Ok((generated_text, usage))
    }

    /// Leniently extract tool calls from the model's reply. Accepts the whole
    /// reply as JSON, or the first balanced `{...}`/`[...]` block (handles models
    /// that wrap JSON in prose or ``` fences). Returns empty if none found.
    fn parse_tool_calls(text: &str) -> Vec<ToolCallInfo> {
        // Reasoning models emit <think>…</think> first; drop it so the JSON scan
        // doesn't latch onto braces inside the chain-of-thought.
        let cleaned = strip_think_blocks(text);
        let text = cleaned.as_str();

        let mut candidates: Vec<String> = Vec::new();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            candidates.push(trimmed.to_string());
        }
        if let Some(block) = first_balanced_json(text) {
            if candidates.first().map(|c| c != &block).unwrap_or(true) {
                candidates.push(block);
            }
        }

        for candidate in candidates {
            if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
                let mut calls = Self::extract_calls(&val);
                if !calls.is_empty() {
                    Self::number_ids(&mut calls);
                    return calls;
                }
            }
        }

        // Python/Llama-style calls some models prefer: `[name(arg=val, ...)]` or a
        // bare `name(arg=val)`. Gate on the whole reply looking like a call list
        // to avoid matching function names mentioned in prose.
        let t = text.trim();
        let looks_like_calls = (t.starts_with('[') && t.ends_with(']')) || is_single_call(t);
        if looks_like_calls {
            let mut calls = parse_python_calls(t);
            if !calls.is_empty() {
                Self::number_ids(&mut calls);
                return calls;
            }
        }

        // Gemma-style native format: some models ignore the JSON protocol and emit
        // `<|tool_call>call:NAME{k:<|"|>v<|"|>, ...}<tool_call|>` (with `<|"|>` as a
        // quote token). Parse it leniently as a last resort.
        let mut gemma = parse_gemma_calls(text);
        if !gemma.is_empty() {
            Self::number_ids(&mut gemma);
            return gemma;
        }

        Vec::new()
    }

    fn number_ids(calls: &mut [ToolCallInfo]) {
        for (i, call) in calls.iter_mut().enumerate() {
            call.id = format!("call_{i}");
        }
    }

    /// Pull ToolCallInfo out of a parsed JSON value in any of the shapes a model
    /// might emit: a bare object, an array of objects, `{"tool_calls": [...]}`,
    /// and either `{name, arguments}` or `{function: {name, arguments}}`.
    fn extract_calls(val: &Value) -> Vec<ToolCallInfo> {
        fn one(v: &Value) -> Option<ToolCallInfo> {
            let obj = v.as_object()?;
            let (name, raw_args) = if let Some(f) = obj.get("function").and_then(|f| f.as_object()) {
                (f.get("name")?.as_str()?.to_string(), f.get("arguments").cloned())
            } else {
                (obj.get("name")?.as_str()?.to_string(), obj.get("arguments").cloned())
            };
            let arguments = match raw_args {
                // OpenAI serializes arguments as a JSON string; accept that too.
                Some(Value::String(s)) => serde_json::from_str(&s).unwrap_or(Value::Object(Default::default())),
                Some(v) => v,
                None => Value::Object(Default::default()),
            };
            Some(ToolCallInfo { id: "call_0".to_string(), name, arguments })
        }

        match val {
            Value::Array(arr) => arr.iter().filter_map(one).collect(),
            Value::Object(o) if o.contains_key("tool_calls") => o
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(one).collect())
                .unwrap_or_default(),
            Value::Object(_) => one(val).into_iter().collect(),
            _ => Vec::new(),
        }
    }
}

impl LlmProvider for LlamaLocalProvider {
    fn chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let prompt = self.build_prompt(messages, None)?;
        tracing::debug!("Prompt length: {} chars", prompt.len());

        let (text, _usage) = self.generate(&prompt)?;

        tracing::debug!("Generated: {}", text);
        Ok(text)
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> Result<LlmResponse> {
        let prompt = self.build_prompt(messages, Some(tools))?;
        tracing::debug!("Prompt: {} chars, {} tools", prompt.len(), tools.len());

        let (generated, usage) = self.generate(&prompt)?;
        tracing::debug!("Raw generated: {}", generated);

        let calls = Self::parse_tool_calls(&generated);
        if !calls.is_empty() {
            tracing::info!("Local LLM returned {} tool call(s)", calls.len());
            return Ok(LlmResponse::ToolCalls(calls, Some(usage)));
        }

        Ok(LlmResponse::Text {
            content: generated,
            reasoning: None,
            usage: Some(usage),
        })
    }
}

/// True if the whole string is a single `name(args)` call.
fn is_single_call(s: &str) -> bool {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^[A-Za-z_]\w*\s*\(.*\)$").unwrap());
    re.is_match(s.trim())
}

/// Parse Python/Llama-style tool calls: `[name(k=v, ...), ...]` or `name(k=v)`.
/// Values are parsed as quoted strings, numbers, booleans, or JSON.
fn parse_python_calls(text: &str) -> Vec<ToolCallInfo> {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"([A-Za-z_]\w*)\s*\(([^)]*)\)").unwrap());

    let mut calls = Vec::new();
    for cap in re.captures_iter(text) {
        let name = cap[1].to_string();
        let mut args = serde_json::Map::new();
        for part in split_top_commas(&cap[2]) {
            if let Some((k, v)) = part.split_once('=') {
                args.insert(k.trim().to_string(), parse_py_value(v.trim()));
            }
        }
        calls.push(ToolCallInfo {
            id: "call_0".to_string(),
            name,
            arguments: Value::Object(args),
        });
    }
    calls
}

/// Split argument text on top-level commas, ignoring commas inside quotes.
fn split_top_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in s.chars() {
        match c {
            '"' | '\'' if quote.is_none() => {
                quote = Some(c);
                cur.push(c);
            }
            c if Some(c) == quote => {
                quote = None;
                cur.push(c);
            }
            ',' if quote.is_none() => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_string());
                }
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Parse a Python-literal-ish value into JSON.
fn parse_py_value(v: &str) -> Value {
    let v = v.trim();
    if v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')))
    {
        return Value::String(v[1..v.len() - 1].to_string());
    }
    match v {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    if let Ok(n) = v.parse::<i64>() {
        return Value::from(n);
    }
    if let Ok(f) = v.parse::<f64>() {
        return Value::from(f);
    }
    serde_json::from_str::<Value>(v).unwrap_or_else(|_| Value::String(v.to_string()))
}

/// Parse the gemma-style native tool-call format that some models emit instead
/// of the JSON protocol we ask for:
/// `<|tool_call>call:NAME{key:<|"|>value<|"|>, key2:123}<tool_call|>`.
/// `<|"|>` is the model's quote token; tool names may contain hyphens (e.g. the
/// MCP tool `search-godoc`). Delimiter-agnostic — we key on `call:NAME{...}`.
fn parse_gemma_calls(text: &str) -> Vec<ToolCallInfo> {
    // Shared wire-format parser (see `crate::gemma`). Names are kept verbatim
    // here — the llama.cpp path is the general-purpose local backend and must
    // not fold mixed-case MCP tool names.
    crate::gemma::parse_native_tool_calls(text)
        .into_iter()
        .map(|c| ToolCallInfo {
            id: "call_0".to_string(),
            name: c.name,
            arguments: c.arguments,
        })
        .collect()
}

/// Strip HF chat-template extensions minijinja can't parse. The `{% generation %}`
/// / `{% endgeneration %}` markers only tag assistant tokens for training masks
/// and are no-ops at inference time.
fn strip_unsupported_jinja(src: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"\{%-?\s*(end)?generation\s*-?%\}").unwrap());
    re.replace_all(src, "").into_owned()
}

/// Remove well-formed `<think>...</think>` blocks (case-insensitive). An unclosed
/// `<think>` (model still reasoning, no answer yet) is left as-is.
fn strip_think_blocks(text: &str) -> String {
    let mut s = text.to_string();
    loop {
        let lower = s.to_lowercase();
        let Some(start) = lower.find("<think>") else { break };
        let Some(end_rel) = lower[start..].find("</think>") else { break };
        let end = start + end_rel + "</think>".len();
        s.replace_range(start..end, "");
    }
    s
}

/// Find the first balanced `{...}` or `[...]` span in `text`, respecting JSON
/// string literals (so braces inside strings don't unbalance it). Returns the
/// substring including the brackets, or None.
fn first_balanced_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(text[start..=i].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_object() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"{"name": "read", "arguments": {"path": "a.txt"}}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn parses_object_wrapped_in_prose_and_fences() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "Sure, I'll do that.\n```json\n{\"name\": \"glob\", \"arguments\": {\"pattern\": \"*.rs\"}}\n```",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
    }

    #[test]
    fn parses_array_of_calls_with_unique_ids() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn parses_openai_shape_with_stringified_args() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"{"tool_calls": [{"function": {"name": "read", "arguments": "{\"path\": \"x\"}"}}]}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "x");
    }

    #[test]
    fn parses_call_after_think_block() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "<think>The user wants me to read a file. I should use {read}.</think>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.txt\"}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn parses_gemma_native_tool_call() {
        // Gemma's native envelope, calling an MCP tool (godevmcp's search-godoc).
        // The hyphens exercise the name charset (`[A-Za-z0-9_.-]`) on both sides.
        let calls = LlamaLocalProvider::parse_tool_calls(
            "<|tool_call>call:search-godoc{query:<|\"|>mcp-go<|\"|>}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search-godoc");
        assert_eq!(calls[0].arguments["query"], "mcp-go");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn parses_gemma_call_with_mixed_args() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "<|tool_call>call:grep{pattern:<|\"|>foo<|\"|>, limit:50}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments["pattern"], "foo");
        assert_eq!(calls[0].arguments["limit"], 50);
    }

    #[test]
    fn plain_prose_is_not_a_gemma_call() {
        let calls = LlamaLocalProvider::parse_tool_calls("Sure, I'll call the search tool for you.");
        assert!(calls.is_empty());
    }

    #[test]
    fn gemma_call_with_braced_source_still_parses() {
        // Regression for gemma-4-26B leaking raw `<|tool_call>` markup: when a
        // string arg holds content with `{`/`}`, the whole native call must still
        // parse through the full chain (JSON → python → gemma fallback),
        // otherwise the turn is misread as a final text answer. Payload mirrors
        // the real leaked reply (channel wrapper + braced arg value).
        let raw = "<|channel>thought<channel|><|tool_call>call:write\
            {file_path:<|\"|>a.json<|\"|>,content:<|\"|>{ \"loop\": true, \"body\": { \"n\": 3 } }\
            <|\"|>}<tool_call|>";
        let calls = LlamaLocalProvider::parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write");
        assert_eq!(calls[0].arguments["file_path"], "a.json");
        assert!(
            calls[0].arguments["content"]
                .as_str()
                .unwrap()
                .contains("\"body\": { \"n\": 3 }"),
            "braced content must survive intact"
        );
    }

    #[test]
    fn parses_python_style_bracket_call() {
        let calls = LlamaLocalProvider::parse_tool_calls(r#"[read(file_path="codeword.txt")]"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["file_path"], "codeword.txt");
    }

    #[test]
    fn parses_multiple_python_calls() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            r#"[glob(pattern="*.rs"), grep(pattern="fn main", path="src")]"#,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].arguments["path"], "src");
    }

    #[test]
    fn prose_mentioning_a_function_is_not_a_call() {
        let calls = LlamaLocalProvider::parse_tool_calls(
            "You can use the read() function to open files.",
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn plain_text_yields_no_calls() {
        let calls = LlamaLocalProvider::parse_tool_calls("The capital of France is Paris.");
        assert!(calls.is_empty());
    }
}
