# Chat template fixtures

The embedded jinja chat template of each model this repo has a config for, so
that `llm_local`'s template harness can assert on real templates without a
multi-GB download and without a GPU.

## Where each one came from

| Fixture | Source | Extracted from |
|---|---|---|
| `gemma4-e4b.jinja` | `unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | the GGUF |
| `lfm2-8b-a1b.jinja` | `LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf` | the GGUF |
| `qwen3.8.jinja` | `Qwen/Qwen3.8-27B/chat_template.jinja` | the **Hub repo** |

**The GGUF is the authority, and the third row is not one.** The template a
model actually runs on is whatever is in `tokenizer.chat_template` inside its
GGUF, and a quantizer may patch it on the way through. Prefer:

```
uv run python scripts/extract_chat_template.py MODEL.gguf > fixture.jinja
```

`qwen3.8.jinja` comes from the Hub because no Qwen3.8 GGUF is cached on the
machine this was written on (it lives on the CUDA box — see
`configs/qwen3.8-cuda-12gb.toml`). Re-extract it from
`unsloth/Qwen3.8-27B-GGUF` there and replace this file if they differ; that
difference would itself be worth knowing about, because
`configs/qwen3.8.toml` records a behaviour ("silently upgrades `high` to
`xhigh`") that the Hub template does **not** have — it raises.

## Qwen3.8-27B and Qwen3.8-Flash-Next publish the same file

`Qwen/Qwen3.8-27B/chat_template.jinja` and
`Qwen/Qwen3.8-Flash-Next/chat_template.jinja` are byte-identical, which is why
there is one fixture and not two. That is also why this fixture is worth having
before llama.cpp can load Flash-Next at all ([PR #27742], unmerged): every
template-level finding about that model is a finding about the 27B this repo
already runs.

[PR #27742]: https://github.com/ggml-org/llama.cpp/pull/27742

## Adding one

Extract it, drop it here, add a row above and an entry to `FIXTURES` in
`llm_local_templates.rs`. The harness asserts the same things about every
fixture, so a new template is one line plus a file.
