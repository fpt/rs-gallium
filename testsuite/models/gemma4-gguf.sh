# Gemma 4 E4B GGUF (Q4_K_M) — downloaded from HuggingFace on first run.
# Sampling defaults per Gemma 4 model card:
#   https://ai.google.dev/gemma/docs/core/model_card_4
MAX_TOKENS=2048
AGENT_FLAGS="--arch gemma4 --format gguf \
  --hf-repo unsloth/gemma-4-E4B-it-GGUF \
  --hf-file gemma-4-E4B-it-Q4_K_M.gguf \
  --hf-tokenizer-repo google/gemma-4-E4B \
  --thinking \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 64"
