# Model notes

Per-model documentation, kept separate from the framework docs in `docs/`.

| File | What it covers |
|---|---|
| [architectures.md](architectures.md) | Attention / FFN / RoPE / normalization per target family — the reference for what a config's numbers mean |
| [adding-models.md](adding-models.md) | Step-by-step: config struct → `load()` → `CausalLM` → registration → weight-name check |
| [gguf-tensor-names.md](gguf-tensor-names.md) | GGUF tensor-name and metadata-key mapping (GPT-OSS MoE in particular) |
| [gemma4.md](gemma4.md) | Gemma 4 implementation notes — dual RoPE, shared K=V, PLE, softcapping, the bug log |
| [gpt-oss.md](gpt-oss.md) | GPT-OSS implementation notes — MXFP4, interleaved SwiGLU, attention sinks |
| [qwen35.md](qwen35.md) | Qwen 3.5 family implementation notes — Gated DeltaNet, hybrid layers |

Per-model *verification* (what passed on which hardware) lives in
[`../VERIFICATION_STATUS.md`](../VERIFICATION_STATUS.md); per-model *tuning*
(the `gpuLayers` / `cpuMoe` values) is there too and in the config files.
