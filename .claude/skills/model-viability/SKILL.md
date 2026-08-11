---
name: model-viability
description: Check whether a HuggingFace model can actually run on gallium before spending time on a config or a download — GGUF availability, architecture support in the vendored llama.cpp, chat-template fit, and whether it fits the machine's disk/VRAM/RAM. Triggered when the user asks to "try", "run", "add", or "use" a model/HuggingFace repo they haven't used with gallium before, asks "can gallium run X", or asks to check/evaluate a model before downloading it.
argument-hint: "[huggingface repo or model name]  — e.g. meta-models/Muse-Glimmer-30B, unsloth/gemma-4-31B-it-qat-GGUF"
allowed-tools: Bash, Read, Grep, Glob, WebFetch, WebSearch
---

# Model viability check

Answer "can gallium actually run this?" *before* a multi-GB download and a
config file, not after. Every step below is cheap (a page fetch, a grep, a
`df -h`); the download is not. Do the steps in order — each one can end the
check early and save the expensive part.

This skill was written after two real investigations: getting Gemma 4
26B-A4B/31B stable on a 12GB card (see `configs/gemma4-26b-cuda-12gb.toml`,
`configs/gemma4-31b-cuda-12gb.toml`, issue #92) and finding that Muse Glimmer
30B cannot load at all on this repo's pinned llama.cpp (issue #95). Read
those two if you want worked examples of what "viable" and "not viable" look
like end to end.

## Step 1 — Does a GGUF exist?

The `local` (llama.cpp) backend is the one that matters first: it is
architecture-agnostic *if* llama.cpp itself supports the architecture (Step
3), and needs no gallium-side model code. The native `candle` backend, by
contrast, only runs GPT-OSS, Qwen 3.5, Gemma 4, and LFM2.5 (`gallium-models`)
— anything else needs a from-scratch implementation (CLAUDE.md's "Adding a
New Model"), which is a multi-session project, not a config change.

Fetch the model's HF page and its `-GGUF` sibling repos:

```
WebFetch https://huggingface.co/<org>/<model>
WebSearch "<model name> GGUF huggingface"
```

Unsloth quantizes almost everything shortly after release
(`unsloth/<model>-GGUF`); an official quant repo from the model's own org is
common too. Note the quant variants and sizes (`tree/main` via WebFetch) —
you'll need these for Step 5. If nothing turns up, this only works through
`candle`, and Step 1 has already answered the viability question: **not
viable without writing a new model implementation.** Say so and stop unless
the user explicitly wants that undertaken.

## Step 2 — Read the model card

Fetch the HF page (or `raw/main/config.json`) and extract:

- **`model_type`** / **`architectures`** — the string that decides Step 3.
  Novel-sounding names (anything not matching gpt-oss/gpt_oss, qwen, gemma,
  lfm2 by substring — see `Arch::from_hint` in `llm_candle.rs`) are the ones
  worth double-checking.
- **Modalities** — text only, or +image/+audio/+video. A vision/audio claim
  needs its own projector file (`mmproj-*.gguf`) to actually work through
  llama.cpp; check for one in the GGUF repo's file list. Don't take the
  model card's modality claims at face value once quantized — verify against
  the projector's own startup log line (`Projector supports: vision=..,
  audio=..`, see Step 6) rather than the marketing copy. See CLAUDE.md's
  Multimodal input section and the Gemma 4 26B-A4B case: the model card said
  "text + image", and that's exactly what the projector's own log confirmed
  — but don't assume a different model's card is equally precise.
- **Extra components** — a speculative-decoding drafter, a separate audio
  tokenizer, LoRA adapters, anything beyond "one model file + maybe one
  projector". Gallium has **no draft-model / speculative-decoding support at
  all** (`llm_local.rs` has no such plumbing on either backend) — if the repo
  ships one, it is dead weight for gallium today, not a bonus. Say so
  explicitly rather than silently ignoring it.
- **`transformers` version** the config.json declares. A `.dev0`/pre-release
  version is a signal the architecture is brand new — worth treating Step 3
  as unverified-until-checked rather than likely-to-work.

## Step 3 — Does *this repo's* llama.cpp know the architecture?

This is the step that actually decides it, and it is a grep, not a guess.
llama.cpp fails to load an unrecognized architecture near-instantly (under
100ms, no real allocation attempted) with a useless message — gallium wraps
it as `Failed to load model: null result from llama cpp`. That failure looks
identical to a dozen other causes unless you check the architecture table
directly:

```bash
LLCPP=$(find ~/.cargo/registry/src -iname "llama-cpp-sys-2-*" -type d | head -1)
grep -i "<architecture_string>" "$LLCPP/llama.cpp/src/llama-arch.cpp"
```

(No `-maxdepth` — the crate is vendored under a hashed index directory, e.g.
`~/.cargo/registry/src/index.crates.io-<hash>/llama-cpp-sys-2-<version>/`, one
level deeper than it looks. `-maxdepth 1` silently finds nothing and looks
like "not supported" when it's actually "looked in the wrong place" — confirm
`$LLCPP` is non-empty before trusting a negative grep.)

**Use the GGUF's own `general.architecture` metadata field as `<architecture_string>`,
not `config.json`'s `model_type`.** They can differ — DeepSeek-V4-Flash's
`config.json` says `model_type: "deepseek_v4"`, but the GGUF itself (checked
directly, see below) says `general.architecture = "deepseek4"`; grepping the
config.json string would have found nothing *and* missed that `deepseek`,
`deepseek2`, and `deepseek32` (older, unrelated versions of the same family)
are already supported — a false "not viable" for the wrong reason right next
to a true one for the right reason. Reading the metadata doesn't need the
whole file: GGUF puts it at the front, so the first split/shard alone (often
only a few MB) has it:

```bash
curl -sL -o /tmp/shard0.gguf "<url of the first shard or the whole file if unsharded>"
uv run --with gguf python3 -c '
from gguf import GGUFReader
r = GGUFReader("/tmp/shard0.gguf")
for f in r.fields.values():
    if "architecture" in f.name or f.name == "general.name":
        print(f.name, "=", f.parts[f.data[0]].tobytes().decode())
'
rm /tmp/shard0.gguf   # scratch download, not the HF cache — safe to delete unasked
```

No output from the `llama-arch.cpp` grep = not supported by the vendored
llama.cpp, full stop. That table (`LLM_ARCH_*` / `LLM_ARCH_NAMES`) is where
every architecture llama.cpp can build a graph for is named; nothing in it
means `llama_model_load_from_file` never gets past parsing the GGUF header.
This is not fixable by tuning `GALLIUM_GPU_LAYERS`, picking a different
quant, or anything else client-side — it needs `llama-cpp-2`'s pinned version
bumped past whatever upstream `ggml-org/llama.cpp` release added support,
which this repo doesn't control. If the source isn't vendored locally yet,
`cargo check --features cuda` (or whichever GPU feature this machine builds)
pulls and unpacks it — cheap compared to a 15GB model download.

If the grep finds nothing: **not viable today.** Report that plainly (see
Muse Glimmer #95, DeepSeek-V4-Flash #96) and stop — don't try workarounds (a
different quant, a different mmproj) that can't route around a missing
architecture entry.

## Step 4 — Chat template fit

Only matters once Step 3 passes. Two separate things can go wrong here:

- **llama.cpp backend**: renders the GGUF's own embedded jinja template via
  minijinja. `strip_unsupported_jinja` in `llm_local.rs` already patches one
  known incompatibility (`{% generation %}` / `{% endgeneration %}` tags);
  anything else minijinja's jinja2 subset doesn't support surfaces as a
  template render error at the first turn, not at load time — so this can't
  be fully ruled out without actually running a turn (Step 6).
- **candle backend** (only relevant if you're also adding a `ModelProtocol`):
  needs a hand-written `format_prompt`/`parse_response` pair, not a jinja
  interpreter — see `protocol.rs`. A new architecture here is exactly the
  "not viable without writing code" case from Step 1.

## Step 5 — Disk, VRAM, RAM

Cheap arithmetic before any download:

```bash
df -h ~/.cache/huggingface          # disk: need the full GGUF (+ mmproj if used) free
nvidia-smi --query-gpu=memory.total,memory.free --format=csv   # or system_profiler SPDisplaysDataType on macOS
free -h                             # RAM, for CPU-offloaded layers and mmap
```

Pick the quant from Step 1 that's closest to this repo's usual choice
(`UD-Q4_K_XL` when offered — matches every `configs/*.toml` in this repo) and
compare its file size against free VRAM. If the file alone exceeds VRAM, full
offload (`GALLIUM_GPU_LAYERS` unset, default 999) **will** fail — that's
expected, not a bug, and the fix is a lower `gpuLayers`/CPU fallback, covered
in Step 6. If the file exceeds free disk *or* free RAM (mmap still needs
address space, and CPU-offloaded layers need real RAM), stop here — a bigger
problem than tuning can fix.

## Step 6 — If it loads: tune GPU layers correctly, not by watching it load once

**The single most important lesson from tuning Gemma 4 on a 12GB card:**
`LlamaModel::load_from_file` succeeding only proves the *weights* fit.
Creating the actual context (KV cache + compute buffers, sized by
`n_ctx.max(n_prompt + max_tokens)`) is separate, later, and fails with the
same unhelpful `Failed to create context: null reference from llama.cpp` —
silently retried by the KV slot pool, easy to miss. Worse: the edge is **not
stable** — identical repeat runs of the same config, GPU idle in between,
started failing at a layer count that had just passed repeatedly (see issue
#92). A number is not "verified" until it has generated successfully, more
than once, at the actual `maxTokens`/system-prompt/projector combination the
real config will use — not the bare minimum that made it load once.

Procedure, in order:

1. `GALLIUM_GPU_LAYERS=0 gallium --config <cfg>` (or `MODEL_PATH=hf:...`
   directly) — CPU-only, confirms Steps 1-4 were right before spending time
   on GPU tuning at all.
2. Bisect upward with **actual single-turn generations** (a real prompt, not
   just watching "Model loaded" in the log), against the *heaviest* config
   this model will really run under — system prompt, skills, real
   `maxTokens`, and the projector loaded if vision/audio is in play. A bare
   "capital of France" prompt on a stripped-down config finds a different
   (higher, wrong) ceiling than the real one.
3. Once a candidate value passes, **re-run it 3-5 times**, not once. If it
   fails on any repeat with `nvidia-smi` back at idle between attempts and no
   dmesg/XID errors, back off — the true safe value is a margin below the
   edge, not the edge itself.
4. Record the number with its own reasoning in the config file's comments
   (model, card, VRAM, what combination it was tested against, how many
   repeats) — the next person (or the next card) needs to know it's a
   measurement, not a constant.

## Output format

End with a verdict, not a narrative:

```
## Verdict: <viable today / viable with caveats / not viable>

1. GGUF available: <yes, repo+quant / no — candle-only>
2. Model card: <architecture, modalities, extra components (drafter/LoRA/etc), transformers version>
3. llama.cpp architecture support: <found in llama-arch.cpp / NOT FOUND — full stop>
4. Chat template: <renders via minijinja / needs verification on first turn>
5. Fits this machine: <file size> vs <free VRAM/disk/RAM>
6. GPU layers (if applicable): <value>, verified via N repeats against <config used>

## If not viable, what would change that
<one of: llama-cpp-2 version bump needed once upstream adds support / new
gallium-models implementation needed / doesn't fit this hardware>
```

If the verdict is "not viable," stop there — don't attempt downloads,
workarounds, or partial implementations without the user asking for that
specific next step. File a GitHub issue if the user wants the finding kept
(see issue #95 for the format: what was tried, the confirmed root cause with
its evidence, and what would need to change).
