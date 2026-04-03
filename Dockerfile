# Multi-stage build: compile Rust binary, then copy into a minimal runtime image.
#
# Usage:
#   docker build -t gallium .
#   docker run --rm \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     gallium --arch gemma4 --format gguf \
#             --model /root/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/<rev>/gemma-4-E4B-it-Q4_K_M.gguf \
#             --hf-tokenizer-repo google/gemma-4-E4B \
#             --prompt "The capital of France is" --max-tokens 20
#
# Or download from HuggingFace at runtime (requires HUGGING_FACE_HUB_TOKEN):
#   docker run --rm \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e HUGGING_FACE_HUB_TOKEN \
#     gallium --arch gemma4 --format gguf \
#             --hf-repo unsloth/gemma-4-E4B-it-GGUF \
#             --hf-file gemma-4-E4B-it-Q4_K_M.gguf \
#             --hf-tokenizer-repo google/gemma-4-E4B \
#             --prompt "The capital of France is" --max-tokens 20

# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM rust:1.87-slim AS builder

# git is required for candle (git dependency in Cargo.toml)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer-cached dependency fetch
COPY Cargo.toml Cargo.lock ./
COPY crates/gallium-core/Cargo.toml  crates/gallium-core/Cargo.toml
COPY crates/gallium-models/Cargo.toml crates/gallium-models/Cargo.toml
COPY crates/gallium-cli/Cargo.toml    crates/gallium-cli/Cargo.toml

# Stub source files so `cargo fetch` succeeds without full source
RUN mkdir -p crates/gallium-core/src  && echo "pub fn _stub() {}" > crates/gallium-core/src/lib.rs
RUN mkdir -p crates/gallium-models/src && echo "pub fn _stub() {}" > crates/gallium-models/src/lib.rs
RUN mkdir -p crates/gallium-cli/src   && echo "fn main() {}"       > crates/gallium-cli/src/main.rs

RUN cargo fetch

# Now copy real source and build release binary
COPY crates/ crates/
RUN cargo build --release -p gallium-cli

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gallium-cli /usr/local/bin/gallium-cli

# Mount your local HuggingFace cache here:
#   docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...
VOLUME /root/.cache/huggingface

ENTRYPOINT ["gallium-cli"]
