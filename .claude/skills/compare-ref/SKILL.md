---
name: compare-ref
description: Compare an rs-gallium Rust model implementation against a reference Python implementation side by side. Triggered when the user asks to "compare with reference", "check against reference", "compare reference", "diff against transformers", or when debugging a numerical discrepancy and reference files are available.
argument-hint: "[model] [component]  — e.g. gpt-oss ffn, qwen35 attention, gemma4"
allowed-tools: Read, Grep, Glob, Bash
---

# Reference Comparison Skill

When the user asks to compare an rs-gallium model against a reference implementation, do the following:

## 1. Resolve files

Use this mapping to locate both files:

| Model arg     | Reference Python                                                                           | Rust (safetensors)              | Rust (GGUF)                      |
|---------------|--------------------------------------------------------------------------------------------|---------------------------------|----------------------------------|
| `gpt-oss`     | `references/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py`              | `crates/gallium-models/src/gpt_oss.rs`   | `crates/gallium-models/src/gpt_oss_q.rs`   |
| `qwen35`      | `references/transformers/src/transformers/models/qwen3/modeling_qwen3.py`                  | `crates/gallium-models/src/qwen35.rs`    | `crates/gallium-models/src/qwen35_q.rs`    |
| `gemma4`      | `references/transformers/src/transformers/models/gemma3/modeling_gemma3.py`                | `crates/gallium-models/src/gemma4.rs`    | `crates/gallium-models/src/gemma4_q.rs`    |

If references have not been cloned, instruct the user to run `bash references/setup.sh` first.

For GGUF/quantization questions, also consult:
- `references/llama.cpp/gguf-py/gguf/constants.py` — dtype codes
- `references/llama.cpp/gguf-py/gguf/tensor_mapping.py` — weight name mapping
- `references/llama.cpp/src/llama-model.cpp` — weight loading

## 2. Determine component scope

If the user specified a component, focus there. Otherwise compare all major components:

| Component   | Python functions to read                              | Rust location                          |
|-------------|-------------------------------------------------------|----------------------------------------|
| `attention` | `eager_attention_forward`, `forward` in attention class | `attention.rs` in gallium-core; model's attention block |
| `ffn`       | FFN/MoE `forward`, any `_apply_gate`-style helper     | `ffn.rs` in gallium-core; model's FFN/MoE struct |
| `norm`      | RMSNorm/LayerNorm forward                             | `norm.rs` in gallium-core                |
| `rope`      | RoPE apply, scaling config                            | `pos_enc.rs` in gallium-core             |
| `block`     | `TransformerBlock.forward`, residual order            | `block.rs`, model's block struct       |

## 3. Read and extract

Read both files. Extract the exact code for the component in question from each:
- Python: copy the relevant function(s) verbatim
- Rust: copy the relevant `forward` / `load` / `new` sections verbatim

## 4. Present side-by-side

Format the output as:

```
## [Model] — [Component] comparison

### Reference (Python — [function name])
```python
<exact Python code>
```

### rs-gallium (Rust — [file:line])
```rust
<exact Rust code>
```

### Differences
- List each meaningful difference as a bullet
- Flag anything that looks wrong in the Rust vs Python
- Call out any numerical constants (activation coefficients, clamp limits, scaling factors)
- Call out tensor shape assumptions (interleaved vs block splits, head counts)
- Call out any missing operations (e.g. sinks, biases, shift terms)
```

## 5. Verdict

End with one of:
- **Match** — implementations are equivalent
- **Divergence found** — list what differs and whether it would cause silent numerical errors
- **Needs investigation** — reference file not yet cloned or component not found; say what to check

## Key things to catch (learned from GPT-OSS debugging)

These are common silent bugs. Always check:

1. **Gate/up tensor split order** — is it block (first half / second half) or interleaved (even / odd indices)?
2. **Activation function** — SiLU vs GeLU vs custom (`gate * sigmoid(gate * 1.702)`); never assume SwiGLU
3. **Clamp asymmetry** — gate often clamped max-only, up clamped both sides
4. **Shift terms** — e.g. `(up + 1) * glu` in GPT-OSS
5. **Attention sinks / extra logits** — extra learnable column appended before softmax, dropped after
6. **Bias terms** — attention projection biases, QKV biases; check `attention_bias` in config
7. **Norm placement** — pre-norm vs post-norm; which norm applies to which branch
8. **RoPE parameters** — theta, scaling factor, original context length; partial rotary head_dim
9. **Weight name mapping** — Python `state_dict` key vs Rust `vb.pp()` path must match exactly
10. **GGUF vs safetensors weight layout** — GGUF may pre-split tensors that are interleaved in safetensors
