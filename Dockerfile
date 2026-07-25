# Multi-stage build: compile the `gallium` agent binary, then copy it into a
# minimal runtime image.
#
# The binary is **env-var + TOML driven**. It parses exactly one flag,
# `--config <path>`, plus an optional leading `app-server` positional — there are
# no --arch / --model / --prompt flags. In REPL mode prompts arrive on stdin,
# one line per turn, so `docker run` needs `-i`.
#
# ── Build ─────────────────────────────────────────────────────────────────────
#   docker build -t gallium .
#
# ── REPL against a local GGUF ─────────────────────────────────────────────────
# `hf:ORG/REPO[@REV]/file.gguf` downloads into the mounted HF cache on first use.
#   docker run --rm -it \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v "$PWD:/workspace" \
#     -e MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf \
#     gallium
#
# ── One-shot: pipe a prompt, stdin closes after one turn ──────────────────────
#   echo "Read Cargo.toml and summarize it" | docker run --rm -i \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v "$PWD:/workspace" \
#     -e MODEL_PATH=hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf \
#     gallium
#
# ── Cloud (OpenAI Responses API), no weights to download ──────────────────────
# The `configs/` dir ships in the image at /app/configs.
#   docker run --rm -it -e OPENAI_API_KEY -v "$PWD:/workspace" \
#     gallium --config /app/configs/openai.toml
#
# ── app-server mode: line-delimited JSON-RPC on stdio ─────────────────────────
# No `-t` here — stdout is the protocol stream.
#   docker run --rm -i -e OPENAI_API_KEY -v "$PWD:/workspace" \
#     gallium app-server --config /app/configs/openai.toml
#
# Mutating tools (write/edit/bash) prompt for approval on a TTY; without one,
# pass -e GALLIUM_AUTO_APPROVE=1 or they will be refused.

# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM rust:1.94-slim-bookworm AS builder

# git: candle (git dep) + llama.cpp vendored source
# pkg-config + libssl-dev: openssl-sys (via hf-hub / native-tls)
# build-essential (g++): esaxx-rs (C++ suffix-array lib, dep of tokenizers)
# cmake + libclang-dev: llama-cpp-sys-2 builds llama.cpp via cmake and its
#   bindings via bindgen, which needs libclang at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    git pkg-config libssl-dev build-essential cmake libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for layer-cached dependency fetch
COPY Cargo.toml Cargo.lock ./
COPY crates/gallium-core/Cargo.toml   crates/gallium-core/Cargo.toml
COPY crates/gallium-models/Cargo.toml crates/gallium-models/Cargo.toml
COPY crates/gallium-agent/Cargo.toml  crates/gallium-agent/Cargo.toml

# Stub source files so `cargo fetch` succeeds without full source. gallium-agent
# declares both a [lib] and a [[bin]] at explicit paths, so both must exist.
RUN mkdir -p crates/gallium-core/src   && echo "pub fn _stub() {}" > crates/gallium-core/src/lib.rs  \
 && mkdir -p crates/gallium-models/src && echo "pub fn _stub() {}" > crates/gallium-models/src/lib.rs \
 && mkdir -p crates/gallium-agent/src  && echo "pub fn _stub() {}" > crates/gallium-agent/src/lib.rs \
 && echo "fn main() {}" > crates/gallium-agent/src/main.rs

RUN cargo fetch

# Now copy real source and build the release binary. Both local backends are on
# by default (`local` = in-process llama.cpp, `gallium` = native candle); GPU
# backends are opt-in, e.g. --build-arg CARGO_FEATURES=cuda.
#
# RUSTFLAGS: candle's k-quant vec_dot gates on #[cfg(target_feature="avx2")] at
# compile time, so a portable build falls through to vec_dot_unopt even on AVX2
# hardware. Pass --build-arg RUSTFLAGS="-C target-feature=+avx2,+fma" when the
# image will only ever run on AVX2 hosts.
ARG CARGO_FEATURES=""
ARG RUSTFLAGS=""
COPY crates/ crates/
RUN if [ -n "$CARGO_FEATURES" ]; then \
      cargo build --release -p gallium-agent --features "$CARGO_FEATURES"; \
    else \
      cargo build --release -p gallium-agent; \
    fi

# ── Stage 2: runtime ────────────────────────────────────────────────────────
# Same Debian release as the builder so the glibc/libstdc++ the binary was
# linked against is the one it finds here.
FROM debian:bookworm-slim AS runtime

# libssl3: native-tls; libgomp1 + libstdc++6: llama.cpp (OpenMP, C++ runtime)
# git + curl: commonly reached for by the agent's `bash` tool
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libgomp1 libstdc++6 git curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/gallium /usr/local/bin/gallium
COPY configs/ /app/configs/

# Mount your local HuggingFace cache here so weights are shared with host runs:
#   docker run -v ~/.cache/huggingface:/root/.cache/huggingface ...
VOLUME /root/.cache/huggingface

# The agent's file tools are rooted at WORKING_DIR; mount the project you want
# it to work on at /workspace.
WORKDIR /workspace
ENV WORKING_DIR=/workspace

ENTRYPOINT ["gallium"]
