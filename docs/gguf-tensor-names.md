# GGUF Tensor Name Mapping

GGUF uses a different naming convention than HuggingFace safetensors. This doc maps between them for our target models.

## Common Pattern

| GGUF | HuggingFace |
|------|-------------|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `output.weight` | `lm_head.weight` |
| `blk.{i}.attn_norm.weight` | `model.layers.{i}.input_layernorm.weight` |
| `blk.{i}.attn_q.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `blk.{i}.attn_k.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `blk.{i}.attn_v.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `blk.{i}.attn_output.weight` | `model.layers.{i}.self_attn.o_proj.weight` |
| `blk.{i}.ffn_norm.weight` | `model.layers.{i}.post_attention_layernorm.weight` |
| `blk.{i}.ffn_gate.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `blk.{i}.ffn_up.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `blk.{i}.ffn_down.weight` | `model.layers.{i}.mlp.down_proj.weight` |

## GPT-OSS Specific

Architecture string: `"gpt_oss"` (GGUF metadata `general.architecture`)

### MoE Expert Weights (merged 3D tensors)

GGUF stores all experts in a single merged tensor with an expert dimension:

| GGUF | Shape |
|------|-------|
| `blk.{i}.ffn_gate_exps.weight` | `[n_expert, n_ff, n_embd]` |
| `blk.{i}.ffn_up_exps.weight` | `[n_expert, n_ff, n_embd]` |
| `blk.{i}.ffn_down_exps.weight` | `[n_expert, n_embd, n_ff]` |
| `blk.{i}.ffn_gate_inp.weight` | `[n_embd, n_expert]` (router) |

### Other GPT-OSS tensors

| GGUF | Purpose |
|------|---------|
| `blk.{i}.attn_q.bias` | Attention Q bias (GPT-OSS has attention_bias=true) |
| `blk.{i}.attn_k.bias` | Attention K bias |
| `blk.{i}.attn_v.bias` | Attention V bias |
| `blk.{i}.attn_output.bias` | Attention output bias |
| `blk.{i}.attn_sinks.weight` | Learnable attention sink bias (shape: [n_head]) |
| `blk.{i}.post_attention_norm.weight` | Post-attention RMSNorm (NOT `ffn_norm`) |

### GGUF Metadata Keys

| Key | Example Value |
|-----|---------------|
| `general.architecture` | `"gpt_oss"` |
| `gpt_oss.block_count` | `24` |
| `gpt_oss.embedding_length` | `2880` |
| `gpt_oss.attention.head_count` | `64` |
| `gpt_oss.attention.head_count_kv` | `8` |
| `gpt_oss.expert_count` | `32` |
| `gpt_oss.expert_used_count` | `4` |
| `gpt_oss.attention.sliding_window` | `128` |
| `gpt_oss.rope.freq_base` | `150000.0` |
| `gpt_oss.context_length` | `131072` |

## References

- `external/llama.cpp/gguf-py/gguf/constants.py` — tensor name constants
- `external/llama.cpp/gguf-py/gguf/tensor_mapping.py` — HF-to-GGUF mapping
- `external/llama.cpp/src/llama-model.cpp` — weight loading with shapes
- `external/llama.cpp/convert_hf_to_gguf.py` — `GptOssModel` converter
