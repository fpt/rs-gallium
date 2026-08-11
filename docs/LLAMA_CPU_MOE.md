# MoE experts on CPU (`cpuMoe` / `GALLIUM_CPU_MOE`)

A build-time-cheap, config-only knob for the llama.cpp backend that mirrors
`llama.cpp`'s own `--n-cpu-moe` in spirit: move the mixture-of-experts FFN
tensors to CPU RAM, keep attention and the KV cache on the GPU. For a sparse
MoE this trades a slower per-token CPU hop — paid only for the handful of
experts actually routed to, same count as the GPU would have read — against a
much smaller VRAM footprint, since the expert tensors are most of the GGUF's
size but only a few of them are touched per token.

Set `[llm] cpuMoe = true` in a config, or `GALLIUM_CPU_MOE=1` (env wins, same
precedence as every other setting — see `config.rs`). Ignored by dense
models (nothing to move) and by every engine except llama.cpp.

## Why this exists

Every CUDA-tuned config in this repo before this knob (`gemma4-26b-cuda-12gb.toml`,
`gemma4-31b-cuda-12gb.toml`, `qwen3.6-cuda-12gb.toml`) was a `gpuLayers`
number bisected against a 12GB card — see issue #92 for how fragile that
process is even done carefully. `cpuMoe` doesn't remove the need to tune
`gpuLayers` (see below — it moves the ceiling, it doesn't eliminate it), but
for a MoE model it changes *what's competing for VRAM* in the first place:
without it, `gpuLayers` layers' worth of expert tensors — the majority of the
file — are what's filling the card. With it, only attention/embedding/output
weights and the KV cache are.

## Implementation

`llama-cpp-2` 0.1.151 exposes `LlamaModelParams::add_cpu_moe_override`, a
`Pin<&mut Self>` method — the params struct becomes self-referential once it
stores a pattern pointer into its own regex buffer, so it must be built with
every by-value builder call (`with_n_gpu_layers`, etc.) first, *then* pinned,
*then* have `add_cpu_moe_override` called on it, matching the crate's own doc
example for this family of methods (`append_kv_override`). `llm_local.rs`
does exactly that:

```rust
let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
let mut model_params = std::pin::pin!(model_params);
if cpu_moe {
    model_params.as_mut().add_cpu_moe_override();
}
let model_params: &LlamaModelParams = &model_params;
```

**This binding's `add_cpu_moe_override` is all-or-nothing** — every layer's
`ffn_(up|down|gate)_(ch|)exps` tensors move to CPU via one fixed regex.
llama.cpp's own `--n-cpu-moe N` CLI flag is graduated (only the first *N*
layers' experts move, the rest stay on GPU), built on the more general
`add_cpu_buft_override(pattern)` this crate also exposes — a layer-graduated
`nCpuMoe` integer knob is possible on top of the same API, just not what's
wired up here. Worth revisiting if the all-or-nothing version turns out too
coarse for some model/card pairing.

## Measured effect — architecture-dependent, not a fixed multiplier

Tested on a 12GB RTX 4070, against each model's *real* config (system
prompt + skills + `maxTokens=4096`, matching the discipline from issue #92 —
bisected with actual repeated generations, not a single load-then-generate
check):

| Model | File size | `gpuLayers` without `cpuMoe` | With `cpuMoe` |
|---|---|---|---|
| Qwen 3.6-35B-A3B (`qwen35moe`, 256 experts/top-8+1 shared, UD-Q3_K_XL) | 16.8GB | 26, 5/5 repeats (this session) | **999 (full offload)**, 5/5 repeats |
| Gemma 4 26B-A4B (128 experts/top-8+1 shared, UD-Q4_K_XL) | 14.3GB | 12 (`gemma4-26b-cuda-12gb.toml`, PR #94) | **20**, 5/5 repeats — still short of full offload |

The gap between "jumps to full offload" and "meaningfully better but still
capped" comes down to how much of each file *isn't* expert tensors. Qwen
3.6's non-expert weights (attention, embeddings, the 40-layer backbone) are
apparently small enough relative to the card that once experts are off the
GPU entirely, everything else fits with room to spare. Gemma 4 26B-A4B's
non-expert weights — plus its dual-RoPE/sliding-window buffers, plus the
multimodal projector this config also loads onto the GPU (`mmprojPath`, see
CLAUDE.md's Multimodal input section) — are still enough to need `gpuLayers`
bisection even with experts moved off. **Don't assume `cpuMoe` alone reaches
full offload for a new model** — verify per model, the same way `gpuLayers`
itself has to be.

## A separate bug this surfaced, not caused by `cpuMoe`

Testing Qwen 3.6 with a real multi-iteration ReAct turn (`file_read`: one
tool call, then an answer) failed with `Decode Error -1: n_tokens == 0` on
the second iteration — reproduced identically with `cpuMoe` on *and* off, at
multiple `gpuLayers` values, so it was unrelated to this feature. Filed as
issue #98 and since fixed: llama.cpp's recurrent/hybrid memory (this
model's Gated DeltaNet layers) can refuse a partial KV-cache trim, and
`generate_in_slot` in `llm_local.rs` was trusting that refusal as success
rather than checking `clear_kv_cache_seq`'s return value — which desynced
gallium's own bookkeeping from what the model's memory actually held. Fixed
by falling back to a full cache reset when a partial trim is refused.

## Using it

```bash
GALLIUM_CPU_MOE=1 gallium --config configs/qwen3.6.toml    # try full offload first
GALLIUM_CPU_MOE=1 GALLIUM_GPU_LAYERS=20 gallium --config configs/gemma4-26b.toml
```

Or bake into a config's `[llm]` block: `cpuMoe = true` alongside `gpuLayers`.
As with `gpuLayers`, verify any number with *repeated* real generations
against the actual config that will be used (system prompt, skills, a
projector if one is configured) — a single load-then-generate check proved
unreliable near the VRAM edge for the non-`cpuMoe` case (issue #92), and
there's no reason to assume `cpuMoe` changes that near its own edge.
