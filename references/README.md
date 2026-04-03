# Reference Implementations

Working copies of upstream repos for cross-checking rs-gallium's model implementations.
The cloned directories are gitignored; only this README and `setup.sh` are tracked.

## Setup

```bash
bash references/setup.sh
```

## Contents

| Directory | Repo | Primary use |
|-----------|------|-------------|
| `transformers/` | [huggingface/transformers](https://github.com/huggingface/transformers) | Model forward-pass reference (GPT-OSS, Qwen, Gemma, …). Sparse-cloned to `src/transformers/models/` only. |
| `llama.cpp/` | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF file format, tensor naming conventions, quantization block layouts (Q4_K, Q8_0, MXFP4 type 39). |
| `vllm/` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | MoE expert dispatch, paged attention, continuous batching. Sparse-cloned to `vllm/` package only. |
| `mistral.rs/` | [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Rust LLM inference with candle — weight loading patterns, quantization wrappers, sampling. |

## Key Paths

### transformers — model implementations
```
src/transformers/models/gpt_oss/modeling_gpt_oss.py  ← GPT-OSS forward pass, sinks, FFN
src/transformers/models/qwen3/modeling_qwen3.py       ← Qwen 3 (check for Qwen 3.5)
src/transformers/models/gemma3/modeling_gemma3.py     ← Gemma 4 (check exact name)
```

### llama.cpp — GGUF format
```
gguf-py/gguf/constants.py          ← dtype codes, tensor name constants
gguf-py/gguf/tensor_mapping.py     ← HF-to-GGUF weight name mapping
src/llama-model.cpp                ← weight loading with shapes
convert_hf_to_gguf.py              ← per-model converters (GptOssModel, etc.)
ggml-quants.h / ggml-quants.c      ← block quantization layouts (Q4_K, Q8_0, TQ2_0)
```

### vllm — MoE patterns
```
vllm/model_executor/layers/fused_moe/   ← fused MoE dispatch
vllm/model_executor/models/             ← per-model implementations
```

### mistral.rs — Rust reference
```
mistralrs-core/src/models/   ← model implementations in Rust + candle
mistralrs-quant/src/         ← quantization (GGUF loading, GPTQ, AWQ)
```

## Why This Exists

During GPT-OSS debugging, two silent bugs persisted for many sessions because we lacked
a local reference to compare against:

1. **Wrong FFN activation** — GPT-OSS uses `gate * sigmoid(gate * 1.702)` with an
   interleaved gate/up split and a `(up + 1)` shift. None of this matches standard
   SwiGLU. Found only by reading `_apply_gate` in `modeling_gpt_oss.py`.

2. **Missing attention sinks** — Each attention layer concatenates a learnable
   per-head scalar to the key dimension before softmax, then drops it. Found by reading
   `eager_attention_forward` in the same file.

Having the reference locally makes it trivial to `grep` or read the exact forward pass
when implementing a new model or chasing a numerical bug.
