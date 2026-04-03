#!/usr/bin/env bash
# Clone reference implementations for rs-gallium.
# Run once from the repo root or from this directory.
# The cloned directories are gitignored — only this script and README.md are tracked.

set -euo pipefail
cd "$(dirname "$0")"

# ── transformers ─────────────────────────────────────────────────────────────
# HuggingFace Transformers: model implementations (GPT-OSS, Qwen, Gemma, …).
# We do a sparse checkout — only src/transformers/models/ to keep the clone small.
if [ ! -d transformers ]; then
  echo "Cloning transformers (sparse: src/transformers/models/)…"
  git clone --filter=blob:none --sparse \
    https://github.com/huggingface/transformers.git transformers
  cd transformers
  git sparse-checkout set src/transformers/models
  cd ..
else
  echo "transformers already cloned — pulling…"
  git -C transformers pull --ff-only
fi

# ── llama.cpp ─────────────────────────────────────────────────────────────────
# ggerganov/llama.cpp: GGUF format spec, tensor naming, quantization code.
if [ ! -d llama.cpp ]; then
  echo "Cloning llama.cpp…"
  git clone --depth=1 https://github.com/ggerganov/llama.cpp.git llama.cpp
else
  echo "llama.cpp already cloned — pulling…"
  git -C llama.cpp pull --ff-only
fi

# ── vllm ─────────────────────────────────────────────────────────────────────
# vllm-project/vllm: MoE dispatch, paged KV cache, speculative decoding patterns.
# Sparse checkout: only vllm/ source package (skip CUDA kernels).
if [ ! -d vllm ]; then
  echo "Cloning vllm (sparse: vllm/)…"
  git clone --filter=blob:none --sparse \
    https://github.com/vllm-project/vllm.git vllm
  cd vllm
  git sparse-checkout set vllm
  cd ..
else
  echo "vllm already cloned — pulling…"
  git -C vllm pull --ff-only
fi

# ── mistral.rs ────────────────────────────────────────────────────────────────
# EricLBuehler/mistral.rs: Rust LLM inference reference (candle-based).
if [ ! -d mistral.rs ]; then
  echo "Cloning mistral.rs…"
  git clone --depth=1 https://github.com/EricLBuehler/mistral.rs.git mistral.rs
else
  echo "mistral.rs already cloned — pulling…"
  git -C mistral.rs pull --ff-only
fi

echo ""
echo "Done. Quick-find commands:"
echo "  Model impl (transformers): ls references/transformers/src/transformers/models/"
echo "  GGUF format (llama.cpp):   ls references/llama.cpp/gguf-py/gguf/"
echo "  MoE layers (vllm):         ls references/vllm/vllm/model_executor/layers/"
echo "  Rust inference:            ls references/mistral.rs/mistralrs-core/src/models/"
