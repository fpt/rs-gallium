# Repository Review — Findings & TODO

Full-repo review (gallium-core, gallium-models, gallium-agent) on 2026-06-10.
Items are ordered by priority within each section. File references are `path:line`.

> **Status: partially stale — read with the notes below (checked 2026-07-23).**
>
> The gallium-core and gallium-models findings still map onto today's code. The
> **gallium-agent half predates an earlier repository merge**: `agent.rs`, `provider.rs`,
> and `session.rs` no longer exist, and the CLI is now env-var + TOML `--config`
> driven with no model flags at all. Any `path:line` reference into those files is
> dead — the finding may or may not still apply somewhere else in the crate, and
> needs re-deriving before acting on it.
>
> Spot-checked on 2026-07-23:
>
> | Item | Status |
> |---|---|
> | §1.1 Gemma sliding-window mask skipped at decode | **still present** — `gemma4.rs:414` still builds no mask when `seq_len <= 1` |
> | §1.5 BashTool timeout does not time out | **fixed** — now polls `try_wait()` and `kill()`s on deadline (`tool.rs:1731`) |
> | §1.7 `step_with_allowed_tools` ignores the allow-list | **fixed** — now calls `tool_registry.filtered(&allowed_tools)` (`lib.rs:459`) |
> | §1.3 EOS substring matching (`provider.rs`) | file gone; the EOS logic moved to `llm_candle.rs` and was revised — re-verify before acting |
> | §1.6 `--session` load-only (`session.rs`) | file gone; the flag no longer exists |
> | §6 Documentation drift | **addressed 2026-07-23** — README, CLAUDE.md, architecture.md, and the Makefile were rewritten against the current code |
> | §4 Compaction never triggers for local models | **fixed 2026-07-25** (with #8) — the trigger now falls back to `estimate_messages_tokens` when a provider reports no usage, so the candle backend compacts too |
>
> Live agent-side work now tracked as issues: #13 (epic: runtime/frontend
> separation), #14 (event model, cancellation, approval tiers, typed tool results,
> trace), #16, #17. #11 (GPU device selection) and #21 (the integration Dockerfile)
> are the remaining standalone ones. Fixed on 2026-07-25: #3 (Dockerfile), #8
> (app-server compaction), #9 (per-thread provider reload).
>
> **Re-checked 2026-09-01.** Two changes beyond per-item strikethroughs:
>
> - The §4 harness-hardening findings (WebFetch timeout, `--working-dir`
>   containment) are **retired rather than fixed** — that surface is out of
>   gallium's scope now. See the scope note at the top of §4: the advanced
>   harness is klein's.
> - New **§9: inference-vs-harness forensics** — the trace roadmap for telling
>   apart the five layers a malformed tool call can come from. §9.1 (raw
>   pre-parse output) is the one item that cannot be backfilled later, which
>   puts it at the top of the priority order. **§9.1 landed in two steps** —
>   raw *text* 2026-09-01 (`TRACE_FORMAT_VERSION` 2), the *token ids* behind it
>   2026-09-02 (`TRACE_FORMAT_VERSION` 4) — and is now closed.
>   **§9.2 per-call prompt sha256 + KV provenance landed 2026-09-01**
>   (`TRACE_FORMAT_VERSION` 3); slot index, the prefix-invariant warning, and
>   the full-render hash chain are still open.
>
> **Swept 2026-09-02: §5, §6, §7, §8.** §6 was already fully resolved by the
> 2026-07-23 rewrite. §5 removed `ModelSource`; `kernels/`, `TurboKvCache`, and
> `gemma4_vision.rs` are kept deliberately (no consumer yet — see §5). §7 closed
> the sliding-window-mask decode test (`gallium_core::mask::attention_mask_needed`,
> used by all three model forward passes) and the harmony parser gap (obsolete
> after the rewrite; regression tests added); the per-arch numerical harness is
> deferred and `test_inner_product_unbiased` stays blocked on §1.4. §8 did the
> EOS-token suppression in `model::generate_reusing` and the
> `extract_reasoning` simplification, and verified the `e8m0_to_f32` decoders
> are correct (a bit-for-bit ggml port, not a bug); the `rand 0.8` and
> `epoch_days_to_ymd` notes are left for a future dep change.

---

## 1. Correctness bugs (high priority)

### 1.1 ~~Gemma 4: sliding-window mask skipped at decode — both variants~~ — **fixed 2026-08-14** (`4bc04cb`)
Both variants built no mask when `seq_len <= 1`, so once the context passed
`sliding_window` (512) a decode-time query on a sliding layer attended to the
**entire** KV cache instead of the last 512 positions — the same bug already found
and fixed in GPT-OSS. Nothing errored; a long session just drifted outside what the
layer was trained to see.

Resolved the other way round from the GPT-OSS spelling
(`needs_mask = seq_len > 1 || (is_sliding && pos + seq_len > window)`): a sliding
layer now always gets a mask, and `build_sliding_window_mask` short-circuits to a
zeros tensor while `seq_len <= 1 && total_len <= window_size`. Same work done, but
the condition that decides whether the window matters lives in one place — the mask
builder — instead of being restated at each call site, which is what let two
independent forward passes get it wrong together. Only a *global* layer still skips
the mask at decode, where attending to all of the past is what causal means.

Covered by `mask.rs`'s `sliding_window_mask_at_decode_bounds_a_single_query` and
`sliding_window_mask_at_decode_is_all_visible_inside_the_window`. **Not** covered:
that `gemma4.rs` / `gemma4_q.rs` actually pass the mask at decode — the guarded
thing is the builder, while what regressed was each model's decision to call it, and
neither file has any test. Testing that means extracting the decision into a pure
function; the same decision exists in a third spelling in `gpt_oss.rs`, so it is a
three-site change rather than a two-line one.

### 1.2 ~~KV cache overflow is broken (truncation vs. mask/RoPE mismatch)~~ — **fail-fast landed 2026-09-01**

`RoPE::apply` (`pos_enc.rs`) now checks `pos + seq_len` against its cos/sin
table height (`= max_position_embeddings`) and returns `"context window
exceeded: position N..M is past this model's trained context length of L
tokens…"` instead of candle's `narrow invalid args start + len > dim_len`.
It is the first op to see the overflow — attention applies RoPE before it
touches the KV cache or a mask — and every LLM here routes Q/K through it, so
one check covers them all; the error propagates straight out of
`generate()`/`generate_reusing`. `KvCache`'s eviction branch is left in place
(now documented as unreachable from a model forward pass, kept for direct
`KvCache` users and a future window-aware attention). Real ring-buffer
semantics — position-aware masks, a sliding RoPE table — are still the
unbuilt alternative. Original finding below.

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

**Re-verified 2026-09-01, still present** — reproduced on CPU with synthetic
tensors, no model needed. `KvCache` was rewritten since the review
(preallocated buffers; overflow is now an explicit eviction keeping the last
`max_seq_len` positions, `kv_cache.rs:76-95`), but nothing downstream can use
an evicted cache: a `KvCache::new(8)` decoding its 9th position yields K of
length 8 against a `(1, 9)` mask → `shape mismatch in broadcast_add, lhs:
[1, 2, 1, 8], rhs: [1, 9]`; and RoPE at `pos == max_seq_len` fails first
anyway with `narrow invalid args start + len > dim_len` (`pos_enc.rs:203`).
Since models size both the cache and the RoPE table from
`max_position_embeddings`, the RoPE error is the one a user actually sees.
The fail-fast in `generate()` remains the right fix — the eviction path is
unreachable-in-practice dead weight until masks and positions are made
window-aware.

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

### 1.10 ~~Inconsistent `.contiguous()` after `expand` in attention~~ — **fixed 2026-07-30**
Two GQA expansions disagreed about `.contiguous()` (`attention.rs::forward` omitted
it, `forward_shared` added it), and CLAUDE.md lists the missing one as a pitfall.

Resolved by removing the expansion rather than by picking a spelling: `gqa.rs`
(`gqa_scores` / `gqa_weighted_sum`) groups Q's rows under their KV head instead of
growing K/V to `h` heads, so there is no expanded tensor to make contiguous. All six
attention sites — `attention.rs` ×2, `gemma4_q.rs` ×2, `gpt_oss_q.rs`,
`qwen35_q.rs`, `lfm2moe_q.rs` — now share it.

Note the *other* GQA expansion, `qwen35_q.rs`'s DeltaNet one, is a different
(tiled) layout feeding a recurrence rather than a matmul, and is deliberately
untouched.

Measured: [docs/CANDLE_BACKEND.md](CANDLE_BACKEND.md) — the two attention products at
context 1577 went 999 → 105 ms per decode step on Metal and 833 → 319 ms on the CPU,
and 2.89 GB of per-step temporaries are gone.

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
(docs/CANDLE_BACKEND.md, 7.1× via expert batching). Apply the same here: group tokens by
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
- ~~The candle backend hardcodes `Device::Cpu`; no way to select a device. Related:
  `RoPE::new` builds its tables in **f64**, which will fail on Metal (no F64) —
  `from_inv_freq` already uses f32; unify.~~ — **both fixed 2026-07-30** (#11).
  `GALLIUM_DEVICE` selects the device through `gallium_core::resolve_device`, and the
  RoPE tables are built in F32 (which is also what the references do). Env var, not a
  flag: the CLI takes only `--config`. See [docs/CANDLE_BACKEND.md](CANDLE_BACKEND.md).

---

## 4. Agent robustness & security

> **Scope note (2026-09-01): the advanced harness is klein's, not gallium's.**
> Gallium provides the model-side runtime — inference, wire protocols, traces,
> approvals, and built-in tools sufficient for a local REPL. Sandboxing, web
> fetch, and richer workspace tooling belong to the client on the other side of
> the app-server protocol (`../klein-cli`), whose `dynamicTools` run on the
> client's machine with the client's policy. Concretely: `WebFetchTool` has been
> **removed** (a client that wants web access brings its own tool), and
> `--working-dir` containment is superseded by the approval tiers
> (`approval.rs` — a write outside the workspace root is `Destructive` and asks;
> the app-server honestly reports `sandbox: danger-full-access` because there is
> no sandbox and claiming one would be the dangerous answer). Findings below
> that asked for hardening those surfaces are retired accordingly, not fixed.

- ~~**`--working-dir` is not a sandbox**~~ — **retired 2026-09-01** (scope note
  above). Containment-by-refusal was replaced by containment-by-approval: path
  resolution still allows absolute paths and `../`, but any write resolving
  outside the workspace root lands in the `Destructive` tier and asks. A real
  sandbox is a klein-side concern.
- ~~**WebFetchTool has no timeout and no size cap**~~ — **retired 2026-09-01**:
  the tool no longer exists (scope note above).
- ~~**Compaction never triggers for local models**~~ — **fixed 2026-07-25.**
  `CandleProvider` still reports no usage, but `memory::compaction_target` now takes
  the estimated history size as a floor, so the trigger fires on the candle backend
  too. The policy is shared by the REPL, `Agent`, and the app-server (#8).
- ~~**Tool transcripts are dropped from memory**~~ — **superseded** by the
  runtime rewrite (#13/#14): `react.rs` keeps tool calls and results in the
  transcript, and `memory.rs` compaction is the deliberate context-economy
  policy on top.
- **MCP client fragility** (`mcp_client.rs`) — **partially fixed**: responses
  are now matched to the request `id`, skipping server-initiated notifications
  (`mcp_client.rs:112`). Still open: the
  `unsafe impl Send/Sync for McpRemoteTool` (`mcp_client.rs:357`) looks
  unnecessary — all fields are already Send+Sync; try removing it.
- `sampling.rs`: `partial_cmp().unwrap()` panics on NaN logits — **still present
  2026-09-01** (`sampling.rs:105,118`); `top_k: Some(0)` guard also still worth
  checking. Clamp/guard both.
- `llm.rs` `extract_text` takes only the first output item's first content
  part — multi-part responses are truncated. (Line moved to ~`llm.rs:1185`;
  re-verify the finding before acting.)
- ~~`protocol.rs`: `GemmaProtocol.tool_call_prefill` is a dead field; the Gemma
  parsing stack carries four legacy formats~~ — **superseded** by the model
  profiles rework ([ADR 0003](adr/0003-model-profiles.md)): the field is gone,
  and the lenient cross-family cascade now lives in `Generic` alone while
  recognized families parse only their own formats.

---

## 5. Dead / unreachable code

Swept 2026-09-02. `ModelSource` (loader.rs) is **removed**. The three that were
already struck stay gone: `GemmaProtocol.tool_call_prefill`,
`parse_gemma_prefill_continuation` / `parse_gemma_tool_format` (all removed with
the profiles rework), and `session::append` (file no longer exists).

What remains is **kept deliberately**, not overlooked — each is finished or
near-finished infrastructure with no consumer *yet*, and the decision is to
revisit when one appears rather than delete and rewrite:

| Item | Location | Why it stays |
|---|---|---|
| `kernels/` module (~780 lines) | gallium-core | Hand-written AVX-512/AVX2/NEON sgemm/rmsnorm/rope/Q8_0. Not wired into any hot path (§3.3). Keep until the candle backend has a CPU path that would use it. |
| `TurboKvCache` / `LayerCache::TurboKv` | gallium-core | Experimental, no model constructs it (§2). Labelled experimental in CLAUDE.md; delete-or-finish tracked in §2, not here. |

`gemma4_vision.rs` is **no longer dead** (2026-09-02) — the candle backend's
Gemma 4 image path (`gemma4_image` preprocessor + `CandleProvider` staging)
now drives it, and the vision tower is verified bit-exact to a `transformers`
reference. **Still open:** captions come out garbled because the Gemma 4 E4B
*text* model on the safetensors path (which nothing else exercises) has a
~15-20% back-half activation drift. The proportional-RoPE fix for the global
layers landed with this; the rest of that drift is an unclosed `gemma4.rs`
bug. See docs/MULTIMODAL.md.

---

## 6. Documentation drift

Resolved. The 2026-07-23 doc rewrite fixed the "Provider routing" /
`supports_tools()` and "token-based compaction" claims (re-checked 2026-09-02 —
CLAUDE.md's Provider-routing and Context-window/Compaction sections match the
code), and the TurboQuant "5-8x", BashTool-timeout, and `--session` items were
resolved with §1.5 / §1.6 / §2.

---

## 7. Testing gaps

- ~~No regression test for sliding-window masking at decode time (would have
  caught §1.1).~~ — **done 2026-09-02.** The decision was extracted to
  `gallium_core::mask::attention_mask_needed(seq_len, pos, window)` — one tested
  function the three model forward passes (`gpt_oss`, `gemma4`, `gemma4_q`) now
  call instead of each spelling it inline. `mask.rs` covers the window boundary
  (`pos + seq_len > window`, off-by-one included) and that "no mask needed" for a
  sliding layer agrees with an all-zeros `build_sliding_window_mask`.
- ~~No test for KV-cache overflow behavior (§1.2).~~ — `pos_enc.rs`'s
  `apply_past_the_context_window_fails_with_a_clear_message` covers the
  fail-fast; the eviction/ring-buffer path is still untested (still unreachable).
- Integration tests are skip-if-model-missing, so CI exercises nothing
  end-to-end; consider one tiny-model (or random-weight) numerical test per
  architecture comparing a couple of layer outputs against precomputed
  references. **Deferred** — needs a per-arch reference-fixture harness, a
  larger piece of test infrastructure than a cleanup pass.
- `test_inner_product_unbiased` tolerance (0.5 relative) is too loose to detect
  §1.4. **Blocked on §1.4** — tightening it without first fixing the Uniform-vs-
  Gaussian sampling in `random_orthogonal`/`random_gaussian` just makes the test
  fail. Do both together (priority-order item 3).
- ~~No tests for harmony tool-call args containing `}` in strings *plus*
  trailing text.~~ — obsolete: the first-`{`/last-`}` heuristic is gone. The
  rewritten `harmony.rs` strict-parses the span between `<|message|>` and the
  final `<|call|>`, and `harmony::tests` now covers braces + a literal `<|call|>`
  inside a string value, and a trailing `final` channel.

---

## 8. Smaller cleanups

- `rand 0.8` is old (0.9 renamed the APIs in use); fine for now, but the
  `Standard`-vs-`StandardNormal` confusion (§1.4) is the kind of bug the 0.9 API
  makes harder to write. **Left as-is** — bundle with the §1.4 fix.
- ~~`generate()` invokes `on_token` for the EOS token itself.~~ — **done
  2026-09-02.** `model::generate_reusing` no longer passes an EOS token to
  `on_token` or includes it in the returned vec, so streaming frontends never
  print it and the §9.1 token-id record matches the visible text on both local
  backends (llama.cpp already stopped before the terminator; candle now does
  too).
- ~~`OpenAiProvider::extract_reasoning` joins `content` and falls back to
  `summary`.~~ — **done 2026-09-02.** `reasoning_param` requests
  `summary: "detailed"` and the Responses API never returns plaintext `content`
  for a reasoning item, so the content branch was dead; `extract_reasoning` now
  reads `summary` only.
- `epoch_days_to_ymd` hand-rolls calendar math (`protocol.rs`); fine, but a
  one-line `time`/`chrono` call would be clearer if a date dep is ever added.
  **Left as-is** — no date dep, and the hand-rolled version is correct.
- ~~`e8m0_to_f32(0)`/`(1)` returns a denormal instead of llama.cpp's exact
  semantics.~~ — **verified correct 2026-09-02, not a bug.** `quantized.rs`'s
  `e8m0_to_f32` is a bit-for-bit port of ggml's `ggml_e8m0_to_fp32_half`
  (`0x0020_0000` / `0x0040_0000` for bytes 0/1), which is the right variant to
  pair with the doubled `E2M1_LUT`. The safetensors GPT-OSS path uses the *true*
  LUT and the full `2^(e-127)` scale — the transformers convention for that
  format. The two decoders differ because the on-disk formats do; both land on
  the same number. Comments in both files now say so; dedup is not applicable.

---

## 9. Inference-vs-harness forensics (trace roadmap, added 2026-09-01)

A malformed tool call — klein's 17K-token turn, the `coding`/`refactoring`
failures behind #185/#192, the dsv4 drift in #209 — can originate in five
layers, and today's traces cannot say which:

1. **Prompt construction** (harness): template rendering, tool schemas,
   re-serialization of prior turns — the territory #185/#192 actually hit.
2. **KV cache reuse** (harness/engine boundary): continuation from
   almost-equal state, like the Δlogit 1.69 observed on `deepseek4` (#209).
3. **Sampling** (engine): LFM2 at `temperature 0.3` failing differently on
   every run, fixed by going greedy.
4. **The model itself**: genuine format degradation deep into a context.
5. **Parsing** (harness): which wire parser claimed the output, or didn't.

The pain point is the one `trace.rs`'s own module docs name: traces record the
*parsed* `LlmResponse`, not the model's pre-parse output — so layers 4 and 5
are indistinguishable in principle (was the mangled shape what the model wrote,
or what the parser left behind?). [ADR 0004](adr/0004-execution-traces-as-training-data.md)
already lists raw capture among what a full-fidelity mode must close; the items
below are that list turned into a triage instrument, in priority order.

### 9.1 Record raw pre-parse output (and token ids) per model call

**First, because it is the only item that cannot be backfilled**: every other
analysis below can be added after the fact, but raw text not captured at trace
time is gone forever. With the raw string, layer 5 becomes independently
testable offline — feed recorded raw output back through the parsers as
regression tests, no model needed. With the token id sequence as well,
detokenization-caused corruption is separable from generation-caused.

**Raw text: done 2026-09-01.** `LlmResponse::{Text,ToolCalls}` carry an
`Option<RawGeneration>` (`crate::llm`), filled by both local backends from the
exact decode `profile.tool_calls` / `clean_reply` saw; `trace.rs` records it as
`TraceStep::raw` (`TRACE_FORMAT_VERSION` → 2), `diff` ignores it, and OpenAI /
the scripted engine leave it `None` (structured items / no model).

**Token ids: done 2026-09-02** (`TRACE_FORMAT_VERSION` → 4).
`RawGeneration::token_ids` / `RawGenerationRecord::token_ids` now carry the
generated token id sequence behind the raw text — the candle backend from
`run_generate_ids`, the llama.cpp backend from `sample_until_done`'s decoded
list (`ids_of`). Not length-capped (it does not blow up on a big `read`, and
truncating would break the text ↔ ids correspondence), `diff` ignores it, and
it stops one short of the terminating EOG / stop-marker token. With this a
detokenization bug is separable from a generation bug offline. §9.1 is closed.

### 9.2 Promote prompt identity and KV provenance to first-class trace fields

Per iteration: a hash (or the full text) of the prompt actually rendered, the
evaluated-token count against the full prompt length, and the KV provenance of
the call — fresh prefill, slot reuse, or checkpoint restore, and which slot.
The `1827/1827 evaluated` number was the deciding evidence in #192, and
`Timing::prefill_tokens` already measures it; it just isn't a structured trace
field yet. On top of that, check the prefix invariant at runtime — iteration
N+1's render must be iteration N's render plus a suffix (the property ADR 0001
protects) — and record it as a hash chain, so re-serialization drift is named
at the moment it happens instead of reconstructed weeks later.

**Numbers done 2026-09-01.** `TokenUsage` carries `prompt_sha256` (sha256 of the
rendered prompt) and `kv: Option<KvProvenance>`
(`freshContext`/`slotReuse`/`checkpointRestore`/`cacheReset` + `reused_tokens`),
filled by both local backends; `trace.rs` records them per step as
`UsageRecord::prompt_sha256` / `UsageRecord::kv` (the latter with
`evaluated_tokens` spelled out — the #192 number), `TRACE_FORMAT_VERSION` → 3,
`diff` ignores both. Drift is visible in the data: `evaluatedTokens ≈
inputTokens` on a continuing turn, or two steps sharing a `promptSha256`.
**Still open:** the slot index; a runtime warning when the prefix invariant
breaks (needs to be compaction-aware, since a mid-turn compaction breaks it
legitimately); and the full rendered prompt as a hash *chain* rather than a
per-call digest — that needs the full render, a `GALLIUM_TRACE_FIDELITY=full`
concern (ADR 0004 §1).

### 9.3 Automatic forensics on parse failure

When a parse failure is detected, re-run the same prompt bytes (a) greedy and
(b) as a fresh prefill (`GALLIUM_KV_CACHE_SLOTS=0` equivalent), and store all
three outputs side by side in the trace. Greedy fixes it → sampling (layer 3);
fresh prefill fixes it → cache (layer 2); both still broken → the model
(layer 4). This automates the manual triage the dsv4 investigation did by hand
("slots=0, 3/3 identical"), and the cost is paid only on failure, so it can be
always-on. A cheap extension with high yield: record top-k logprobs at the
format branch points only (the opening/closing of `<tool_call>`-style tags) —
a correct close tag sitting at rank 2 within noise says sampling; out of the
top-k entirely says model state, quantitatively.

### 9.4 scripted-tools mode — the inverse of the scripted engine

`llm_scripted.rs` replays recorded *model output* against real tools; an
inference investigation needs the opposite — a **real model** fed recorded
*tool results*. Traces already hold every tool call's result, so a conversation
that broke on klein's eighth iteration becomes a deterministic reproduction
against the real model with no tool execution and no environment setup. That
makes "failure deep in a long loop" a unit the testsuite can carry, which also
solves the long-horizon-testcase problem (environment construction too heavy to
fixture) discussed alongside it.

---

## Suggested priority order

1. **§9.2 prompt hash + KV provenance fields** — the per-call numbers done
   2026-09-01; remaining: slot index, compaction-aware prefix-invariant warning,
   full-render hash chain (full-fidelity mode)
2. §9.3 parse-failure auto-forensics; §9.4 scripted-tools mode
3. §1.4 TurboQuant gaussians + §2 memory claims (or demote to experimental) —
   also unblocks the §7 `test_inner_product_unbiased` tightening and the §8
   `rand 0.9` note
4. §3.1/3.2 MoE batching (biggest perf win for safetensors GPT-OSS)
5. §1.8 YaRN verification against reference

(Resolved since the original ordering: §1.1, §1.2 fail-fast, §1.3*, §1.5, §1.6*,
§5 (swept — `ModelSource` gone, the rest kept deliberately), §6 (all resolved),
§7 (sliding-window-mask test + harmony parser), §8 (EOS suppression,
`extract_reasoning`, `e8m0` verified), §9.1 (raw text + token ids), §9.2 per-call
numbers, WebFetch/working-dir — *retired or moved rather than fixed; see the
per-item notes.)
