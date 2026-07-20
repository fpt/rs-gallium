# Repository Review — Findings & TODO

Full-repo review (gallium-core, gallium-models, gallium-agent) on 2026-06-10.
Items are ordered by priority within each section. File references are `path:line`.

---

## 1. Correctness bugs (high priority)

### 1.1 Gemma 4: sliding-window mask skipped at decode — both variants
`gemma4.rs:414` and `gemma4_q.rs:460` build no mask when `seq_len <= 1`. Once the
context exceeds `sliding_window` (512), decode-time queries on sliding layers attend
to the **entire** KV cache instead of the last 512 positions. This is the exact bug
class already found and fixed in GPT-OSS (`gpt_oss.rs:341-357` has the correct
`needs_mask = seq_len > 1 || (is_sliding && pos + seq_len > window)` logic, with an
explanatory comment). Long Gemma conversations/coding sessions will degrade after
~512 tokens. Fix both files to mirror the GPT-OSS logic, and add a regression test
(decode at `pos > window`, assert masked scores).

### 1.2 KV cache overflow is broken (truncation vs. mask/RoPE mismatch)
`kv_cache.rs:30-40` silently truncates the cache to `max_seq_len`, but:
- `attention.rs` masks are built with `total_len = pos + seq_len`, which no longer
  matches the truncated K/V length → `broadcast_add` shape error (or silent
  misalignment) at the moment the cache first overflows.
- `pos` keeps growing past the RoPE table (`pos_enc.rs:189` `self.cos.i(pos..)`)
  → index out of range at `pos >= max_seq_len` anyway.
Since all models construct `KvCache::new(max_position_embeddings)`, overflow means
"crash with a confusing error" today. Either implement real ring-buffer semantics
(with position-aware masks) or fail fast with a clear "context window exceeded" error
in `generate()`.

### 1.3 EOS detection by substring match can stop generation mid-sentence
`provider.rs:65-82`: `k.contains("eos")` matches ordinary BPE vocab entries such as
`videos`, `rodeos`, `Theos` — any of these tokens being generated silently terminates
the turn. Similarly `k.contains("</s>")` is substring-based. Use exact matches against
the tokenizer's declared special tokens (and/or `eos_token_id` from config/GGUF
metadata), not substring scans over the whole vocab.

### 1.4 TurboQuant uses uniform random numbers where Gaussians are required
`turbo_quant.rs:305-365`: both `random_orthogonal` and `random_gaussian` sample
`rand::distributions::Standard` for `f32`, which yields **Uniform[0,1)**, not N(0,1)
as the names and comments claim.
- The rotation is still orthogonal after Gram-Schmidt but is far from Haar-distributed
  (all-positive first row), degrading the "coordinates ≈ N(0,1/d)" assumption that the
  Lloyd-Max codebook relies on.
- The QJL projection in InnerProduct mode is plainly wrong: `sign(S·r)` with an
  all-positive S is heavily biased, so the "unbiased inner product" guarantee from the
  paper does not hold. The unit test passes only because its tolerance is 0.5 relative.
Fix: sample N(0,1) (Box–Muller or `rand_distr::StandardNormal`), and tighten
`test_inner_product_unbiased`.

### 1.5 BashTool timeout doesn't actually time out
`tool.rs:545-560`: the worker is spawned inside `std::thread::scope`, which **joins
all scoped threads before returning**. After `recv_timeout` expires, the scope still
blocks until `Command::output()` completes, and the child process is never killed.
So a hung command hangs the agent forever despite the "Timeout: 30s" description.
Fix: spawn the `Child` directly, poll/wait with a deadline, and `kill()` on timeout.

### 1.6 `--session` is load-only — conversations are never persisted
`session::save` / `session::append` are never called anywhere (verified by grep).
`main.rs` loads a session at startup (`main.rs:480-491`) and deletes the file on
`/reset` (`run_repl`), but no turn is ever written back. Additionally,
`ChatMessage` marks `tool_calls` / `tool_call_id` / `tool_name` / `images` as
`#[serde(skip)]` (`llm.rs:48-60`), so even once saving is wired up, tool turns
round-trip as empty messages. Wire `append()` into the REPL loop and decide on a
serializable representation for tool calls.

### 1.7 `step_with_allowed_tools` silently ignores the allow-list
`lib.rs:165-171` takes `_allowed_tools` and just calls `step()`. The
`FilteredToolRegistry` infrastructure exists (`tool.rs:114-145`) but is never used.
Callers believe they are restricting the tool surface (bash! write!) when they
are not. Either implement it or remove the API.

### 1.8 YaRN interpolation mixes units (rotations vs. dim indices)
`pos_enc.rs:112-139`: `low`/`high` are computed as **rotation counts**
(`orig_max / (beta · 2π)`), but are then used (a) as thresholds against
`dim_ratio` scaled by `rotary_dim`, and (b) directly as **dimension indices** in
`t = (i - low) / (high - low)`. Reference YaRN converts rotations to dim indices via
`d·ln(orig_max/(rot·2π)) / (2·ln θ)` (`find_correction_dim`). The current ramp is
almost certainly wrong outside the two extremes; GPT-OSS short-context works because
most dims fall in the "keep" / "scale" branches. Compare against
`references/transformers` YaRN and fix (affects GPT-OSS long-context quality).

### 1.9 `softplus` is not numerically stable despite its doc comment
`linear_attn.rs:247-250`: `log(1 + exp(x))` overflows to `inf` for large `x`
(then `g = -A·inf` → state collapses to zero). Use
`max(x,0) + ln(1 + exp(-|x|))`. Note `a + dt_bias` magnitudes are usually small, but
nothing guards this. Also: the two doc comment lines above `rms_norm_gated`
(`linear_attn.rs:195-196`) contradict each other — delete the stale one.

### 1.10 Inconsistent `.contiguous()` after `expand` in attention
`attention.rs:204-217` (GQA repeat in `forward`) reshapes an expanded tensor without
`.contiguous()`, while `forward_shared` (`attention.rs:290-298`) adds it. CLAUDE.md
itself lists "after `expand()` call `.contiguous()`" as a pitfall. If the current code
works it's because candle's `reshape` copies in this case — make the two paths
consistent and intentional.

---

## 2. Feature claims that don't hold (TurboQuant / TurboKvCache)

### 2.1 TurboKvCache provides no memory savings
`turbo_kv_cache.rs:75-90` caches the **full dequantized** K/V (`cached_k_deq`/
`cached_v_deq`) alongside the compressed forms, so memory is *strictly worse* than a
plain `KvCache`. The compressed `cached_k`/`cached_v` vectors are pushed but never
read back. `max_seq_len` is an unimplemented `TODO`. The "5-8x memory reduction"
claim in the module docs and CLAUDE.md is not realized.

### 2.2 The quantized representation itself is not compact
`turbo_quant.rs:99-108`: indices are stored as a `u8` tensor (8 bits per coordinate
regardless of `bit_width` 1–4), and `qjl_signs` is an **f32** tensor (32 bits per
coordinate of ±1). For f16 K/V, "3-bit" MSE mode is at best 2×, and InnerProduct mode
is *larger than uncompressed*. To deliver the paper's ratios you need bit-packing
(and 1-bit sign packing) into raw byte buffers.

### 2.3 Nothing uses it
No model constructs `LayerCache::TurboKv` (grep: zero references outside core).
`quantize_scalar`/`dequantize_scalar` are also per-element CPU loops via `to_vec1`.
Decide: finish it (pack bits, drop the deq cache, implement window truncation, wire
into a model behind a flag) or move it to an `experimental/` module and soften the
docs.

---

## 3. Performance

### 3.1 GPT-OSS safetensors MoE: dequantizes full expert matrices per token
`gpt_oss.rs:161-222`: inside the per-token loop, `deq()` dequantizes the entire
`[2*inter, hidden]` gate_up and `[hidden, inter]` down matrices **per selected expert,
per token, per layer, per forward**. The GGUF path already learned this lesson
(docs/performance.md, 7.1× via expert batching). Apply the same here: group tokens by
expert per layer, dequantize each needed expert once per forward, or cache dequantized
experts with an LRU.

### 3.2 MoEFFN (gallium-core) routes tokens one at a time on the CPU
`ffn.rs:125-167`: `to_vec2` forces a device sync; then a per-token loop runs each
expert on a single token (`narrow(0, tok_idx, 1)`), allocating a zeros tensor per
token. Group by expert and batch (this is the path Qwen 3.5 MoE configs would hit).

### 3.3 The hand-written SIMD kernels module is dead code
`kernels/` (~780 lines: AVX-512/AVX2/NEON sgemm, rmsnorm, rope, Q8_0 dot) is never
referenced by any model or by gallium-agent — `KernelSet::detect()` only runs in its
own tests. Either wire it into the hot paths it was written for or delete it; right
now it is maintenance surface with zero benefit.

### 3.4 Misc
- `mask.rs:7`: `Tensor::zeros` result is discarded and rebuilt when `seq_len > 1` —
  wasted allocation; also masks are rebuilt per layer per step
  (`gpt_oss.rs:348-357`); build the (at most two) masks once per forward.
- `norm.rs:43`: `RmsOnePlus` computes `weight + 1` on **every forward**; the comment
  says "we add 1 at load time" — do that instead.
- `attention.rs:173`: `let q = q;` is a leftover no-op.
- Sliding-window layers allocate `KvCache::new(max_position_embeddings)` and keep
  full history (`gpt_oss.rs:305`); they only ever need `window` entries.
- `main.rs:204` hardcodes `Device::Cpu`; no `--device` flag despite candle Metal/CUDA
  support. Related: `RoPE::new` builds its tables in **f64** (`pos_enc.rs:168-171`),
  which will fail on Metal (no F64) — `from_inv_freq` already uses f32; unify.

---

## 4. Agent robustness & security

- **`--working-dir` is not a sandbox**: `read`/`write`/`edit`/`glob` accept absolute
  paths and `../` traversal (`tool.rs:173-177` resolve), `glob` with an absolute
  pattern escapes too (`PathBuf::join` semantics), and `bash` is unrestricted.
  CLAUDE.md describes `--working-dir` as the tools' "root directory" — either enforce
  containment (canonicalize + prefix check) or document that it is only a default cwd.
- **WebFetchTool has no timeout and no size cap** (`tool.rs:618-640`): default `ureq`
  agent never times out; a slow endpoint hangs the ReAct loop indefinitely. Set
  connect/read timeouts and cap the body.
- **Compaction never triggers for local models**: `Agent::maybe_compact`
  (`agent.rs:173-186`) keys off `usage.input_tokens`, which `GalliumProvider` never
  reports (always 0) → `--context-window` is a no-op for the gallium provider; local
  sessions rely solely on the 100-message cap. Use `ConversationMemory::estimate_tokens`
  as the fallback signal.
- **Tool transcripts are dropped from memory**: `Agent::step` only persists the final
  assistant text (`agent.rs:108-124`); the next turn's model has no record of what
  tools ran or returned. If intentional (context economy), document it; otherwise
  persist tool turns.
- **MCP client fragility** (`mcp_client.rs:67-95`): assumes exactly one response line
  per request — any server-initiated notification or log line on stdout breaks the
  protocol; no read timeout; response `id` is never matched to the request. Also the
  `unsafe impl Send/Sync for McpRemoteTool` (`mcp_client.rs:192-193`) looks
  unnecessary — all fields are already Send+Sync; try removing it.
- `sampling.rs`: `partial_cmp().unwrap()` panics on NaN logits (`sampling.rs:94,107`);
  `top_k: Some(0)` panics at `indexed[0]` (`sampling.rs:116`). Clamp/guard both.
- `llm.rs:427-434` `extract_text` takes only the first output item's first content
  part — multi-part responses are truncated.
- `protocol.rs`: `GemmaProtocol.tool_call_prefill` (`protocol.rs:444`) is written
  nowhere — dead field. The Gemma tool-call parsing stack carries four legacy formats
  (prefill-JSON, bare-JSON, `Action:`, native) — consider pruning to the native
  `<|tool_call>` format now that it works, since each heuristic is a false-positive
  risk on ordinary text.

---

## 5. Dead / unreachable code

| Item | Location | Note |
|---|---|---|
| `kernels/` module | gallium-core | never called outside its tests (§3.3) |
| `TurboKvCache` / `LayerCache::TurboKv` | gallium-core | no model uses it (§2.3) |
| `gemma4_vision.rs` (635 lines) | gallium-models | compiles, exported, but no caller — not reachable from the CLI (`--arch gemma4` is text-only) |
| `GemmaProtocol.tool_call_prefill` | protocol.rs:444 | never written |
| `ModelSource` enum | loader.rs:6-8 | unused |
| `parse_gemma_prefill_continuation` / `parse_gemma_tool_format` | protocol.rs | only referenced by tests; not in the parse chain |
| `session::append` | session.rs:68 | never called (see §1.6 — should be) |

---

## 6. Documentation drift

- **CLAUDE.md "Provider routing"** says the Gallium provider has
  `supports_tools() = false` → plain chat. Stale: all three protocols now return
  `true` and local models run the full ReAct loop. Same section limits OpenAI tools
  to `read`/`glob`/`tasks`; the default registry now has 8 tools.
- CLAUDE.md says memory has "token-based compaction" — true only for OpenAI (§4).
- `turbo_kv_cache.rs` / CLAUDE.md claim "5-8x memory reduction" (§2.1).
- `tool.rs:523` BashTool description promises a 30s timeout it doesn't deliver (§1.5).
- `--session` flag help implies persistence that doesn't happen (§1.6).

---

## 7. Testing gaps

- No regression test for sliding-window masking at decode time (would have caught
  §1.1 — and the GPT-OSS variant of the same bug earlier). A small synthetic model or
  a direct `Attention::forward` test with `pos > window` suffices.
- No test for KV-cache overflow behavior (§1.2).
- Integration tests are skip-if-model-missing, so CI (if any) exercises nothing
  end-to-end; consider one tiny-model (or random-weight) numerical test per
  architecture comparing a couple of layer outputs against precomputed references.
- `test_inner_product_unbiased` tolerance (0.5 relative) is too loose to detect §1.4.
- No tests for `parse_harmony_tool_call` against args containing `}` in strings
  *plus* trailing text (the first-`{`/last-`}` heuristic at protocol.rs:281-287 grabs
  trailing garbage if the model emits anything after the JSON).

---

## 8. Smaller cleanups

- `rand 0.8` is old (0.9 renamed the APIs in use); fine for now, but the
  `Standard`-vs-`StandardNormal` confusion (§1.4) is the kind of bug the 0.9 API
  makes harder to write.
- `generate()` (`model.rs:44-46`) invokes `on_token` for the EOS token itself;
  streaming frontends print it. Consider suppressing.
- `OpenAiProvider` `extract_reasoning` joins `content` and falls back to `summary`,
  but requests `summary: "auto"` only — content is never present; simplify.
- `epoch_days_to_ymd` hand-rolls calendar math (`protocol.rs:1676`); fine, but a
  one-line `time`/`chrono` call would be clearer if a date dep is ever added.
- `e8m0_to_f32(0)`/`(1)` returns a denormal instead of llama.cpp's exact semantics —
  verify against `ggml_e8m0_to_fp32` for bytes 0 and 1 (gpt_oss.rs:148 uses
  `e==0 → 0.0`, quantized.rs:278 uses a denormal — the two MXFP4 decoders in the
  repo disagree; also consider deduplicating them, `gpt_oss.rs` MXFP4_TABLE vs
  `quantized.rs` E2M1_LUT encode the same table at two different scales).

---

## Suggested priority order

1. §1.1 Gemma sliding-window decode mask (correctness, both files, +test)
2. §1.3 EOS substring matching (silent generation truncation)
3. §1.5 BashTool timeout + §4 WebFetch timeout (agent hangs)
4. §1.6 session persistence (advertised feature missing)
5. §1.2 KV overflow fail-fast
6. §1.4 TurboQuant gaussians + §2 memory claims (or demote to experimental)
7. §3.1/3.2 MoE batching (biggest perf win for safetensors GPT-OSS)
8. §1.8 YaRN verification against reference
9. Dead-code sweep (§5) and doc updates (§6)
