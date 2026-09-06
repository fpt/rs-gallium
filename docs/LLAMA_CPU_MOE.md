# MoE experts on CPU (`cpuMoe` / `GALLIUM_CPU_MOE`)

A build-time-cheap, config-only knob for the llama.cpp backend that mirrors
`llama.cpp`'s own `--n-cpu-moe` in spirit: move the mixture-of-experts FFN
tensors to CPU RAM, keep attention and the KV cache on the GPU. For a sparse
MoE this trades a slower per-token CPU hop — paid only for the handful of
experts actually routed to, same count as the GPU would have read — against a
much smaller VRAM footprint, since the expert tensors are most of the GGUF's
size but only a few of them are touched per token.

Set `[llm] cpuMoe = true` in a config, or `GALLIUM_CPU_MOE=1` (env wins, same
precedence as every other setting — see `config.rs`). Ignored by dense models
(nothing to move). The **candle** backend acts on it too now, for its GGUF MoE
models — see "On the candle backend" below; the effect there is
model-dependent, and for one family a large *win*.

## Why this exists

Before this knob, fitting a large model on a 12GB card meant a `gpuLayers`
number bisected against it by hand (`gemma4-26b.toml`, `gemma4-31b.toml`) —
see issue #92 for how fragile that process is even done carefully. `cpuMoe`
doesn't remove the need to tune
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
| Gemma 4 26B-A4B (128 experts/top-8+1 shared, UD-Q4_K_XL) | 14.3GB | 12 (`gemma4-26b.toml`, PR #94) | **20**, 5/5 repeats — still short of full offload |
| MiniMax-M2.7 (`minimax-m2`, 256 experts/top-8, UD-Q2_K_XL) | 75.3GB | not tested — cpuMoe treated as mandatory (file dwarfs the card even quantized) | **999 (full offload)**, 6/6 repeats + a multi-turn tool-calling run, ~6.5GB of 12GB still free |

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

## On the candle backend

`cpuMoe` reaches the native candle engine's GGUF MoE models —
`gpt_oss_q` (MXFP4 experts), `gemma4_q` and `lfm2moe_q` (generic block-quant
experts). It sets a `moe_device` on each MoE module: the expert matvec runs
there while the rest of the model stays on `GALLIUM_DEVICE`, and only the
`(n_e, hidden)` routed activations and expert outputs cross the bus.
`load_candle_provider` resolves `moe_device` to `Device::Cpu` when `cpuMoe`
is set **and** the device is an accelerator; otherwise it equals the model
device and every `to_device` in the path is a no-op — so `cpuMoe` on a
CPU-only run changes nothing.

**The effect splits by how good candle's accelerator quantized-matmul is for
that expert format.** Measured on a 12 GB RTX 4070, `--features cuda`, the
`coding` testcase, peak VRAM sampled during the run:

| model (candle, CUDA) | no cpuMoe | cpuMoe | |
|---|---|---|---|
| `gpt-oss-20b-candle` | 76 s, 4667 MiB | **36 s, 3995 MiB** | 2.1× faster, −0.7 GB |
| `gpt-oss-120b-candle` | 556 s, 5467 MiB | **108 s, 4571 MiB** | **5.1× faster**, −0.9 GB |
| `gemma4-26b-candle` (`needle_in_haystack`) | 6 s | 17 s | **0.35× — slower** |

**GPT-OSS: a large win.** candle-core has no MXFP4 at all, so on an
accelerator every active expert is re-uploaded and dequantized to the GPU per
token — brutal on the PCIe bus. The CPU fused MXFP4 matvec (`Tq2Tensor::matvec_expert`,
issues from #265/#267) plus keeping the bytes in host RAM beats that decisively
and frees ~1 GB of VRAM. `gpt-oss-20b-candle` and `gpt-oss-120b-candle` set
`cpuMoe = true` in their configs for this reason (a no-op on CPU/Metal-CPU).

**Gemma 4: a loss.** Its Q4_K / Q4_0 experts go through candle's native CUDA
`QMatMul`, which is fast once the bytes are resident; moving the compute to the
CPU just adds a serialization stall while the GPU idles. Leave `cpuMoe` off for
`gemma4-26b-candle` (and it already fits the card without it — see
docs/VERIFICATION_STATUS.md "Gemma 4 26B-A4B on candle").

LFM2's `lfm2moe_q` wiring follows the same pattern (`qmatmuls` built on
`moe_device`) but was not separately measured — LFM2.5-8B-A1B fits the card
comfortably either way.

## Using it

```bash
GALLIUM_CPU_MOE=1 gallium --config configs/gpt-oss-120b.toml    # try full offload first
GALLIUM_CPU_MOE=1 GALLIUM_GPU_LAYERS=20 gallium --config configs/gemma4-26b.toml
```

Or bake into a config's `[llm]` block: `cpuMoe = true` alongside `gpuLayers`.
As with `gpuLayers`, verify any number with *repeated* real generations
against the actual config that will be used (system prompt, skills, a
projector if one is configured) — a single load-then-generate check proved
unreliable near the VRAM edge for the non-`cpuMoe` case (issue #92), and
there's no reason to assume `cpuMoe` changes that near its own edge.
