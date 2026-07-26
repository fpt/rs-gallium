.PHONY: build check test test-models fmt fmt-check clippy clean zip install \
	run run-app-server \
	docker-build docker-build-integration docker-build-intgration \
	docker-run-integration \
	testsuite testsuite-local

# Install location (override with: make install PREFIX=/usr/local)
PREFIX ?= $(HOME)
BINDIR := $(PREFIX)/bin

# Cargo binary. On non-Windows this is just `cargo`.
CARGO ?= cargo

# Extra cargo features for the built binary. Windows defaults to `cuda` (GPU
# offload via the llama.cpp backend) — set in the Windows block below; see
# docs/DEVELOPMENT.md. Undefined (= none) elsewhere. Override on either, e.g.
# `make build CARGO_FEATURES=` for a CPU-only Windows build.
FEATURES_FLAG = $(if $(strip $(CARGO_FEATURES)),--features $(CARGO_FEATURES))

# ── Windows (MSVC) build settings ──────────────────────────────────────────
# The in-process llama.cpp backend (`local` feature, on by default) builds
# llama.cpp through cmake, which on Windows only works against the MSVC
# toolchain. Four things are required; all are no-ops on macOS/Linux:
#
#   * PATH -> rustup's ~/.cargo/bin goes first so both `cargo` AND `rustc`
#     resolve to the MSVC toolchain. A stray GNU Rust earlier on PATH (e.g.
#     Chocolatey's) otherwise gets picked up — cargo invokes `rustc` by name,
#     so even the rustup cargo proxy compiles with the GNU rustc, producing
#     windows-gnu objects (can't link llama.cpp's MSVC .lib files) or E0514
#     "incompatible version of rustc". cygpath makes HOME a POSIX path so it
#     slots into the ':'-separated PATH (drive-letter colons break otherwise).
#   * CMAKE_GENERATOR=Ninja -> the default "MSYS Makefiles" generator mangles
#     MSVC-style linker paths (e.g. /pdb:) under Git Bash. Requires ninja on PATH.
#   * CFLAGS/CXXFLAGS=-MD -> esaxx-rs (a transitive C++ dep of tokenizers)
#     hardcodes the *static* CRT (/MT); force the *dynamic* CRT so it matches
#     Rust std and llama.cpp, both /MD. Without this the final link fails with
#     LNK2038 "RuntimeLibrary mismatch".
#
# The Windows binary defaults to the `cuda` feature (GPU offload). This needs a
# CUDA toolkit whose nvcc supports your GPU's arch — CUDA 13.x dropped Pascal
# (GTX 10-series, sm_61), so those need CUDA 12.x. Point CUDA_PATH / CUDACXX at
# a compatible toolkit and set CUDAARCHS (e.g. 61) in your environment; see
# docs/DEVELOPMENT.md. Build CPU-only with `make build CARGO_FEATURES=`.
ifeq ($(OS),Windows_NT)
  export PATH := $(shell cygpath -u "$(HOME)")/.cargo/bin:$(PATH)
  CARGO_FEATURES ?= cuda
  export CMAKE_GENERATOR := Ninja
  export CFLAGS := -MD
  export CXXFLAGS := -MD
endif

# Testsuite driver. Defaults to the `gallium` binary via testsuite/gallium_cli.sh,
# which locates the binary and forwards `--config <backend.toml>` (prompts arrive
# on stdin). Override CLI= to drive a different backend binary:
#   make testsuite CLI=/path/to/other-app-server
GALLIUM_TESTSUITE_CLI := $(CURDIR)/testsuite/gallium_cli.sh
CLI ?= $(GALLIUM_TESTSUITE_CLI)

build:
	$(CARGO) build --release $(FEATURES_FLAG)

check:
	$(CARGO) check --workspace

test:
	$(CARGO) test --workspace

# The model integration tests are #[ignore]d because each loads a multi-GB model
# from the HuggingFace cache; `make test` skips them. This runs them, skipping
# whichever models are not cached locally.
# Usage: make test-models
test-models:
	$(CARGO) test -p gallium-models --test integration -- --ignored --nocapture

# Install the `gallium` binary to $(BINDIR).
#
# It is the whole product: the text REPL and the `app-server` mode (the JSON-RPC
# whole-turn backend that rs-kessel and klein-cli spawn). Self-contained, so it
# does not care where this repo lives. Re-run `make install` after pulling so
# $(BINDIR) tracks the latest.
install: build
	@mkdir -p "$(BINDIR)"
	@cp target/release/gallium "$(BINDIR)/gallium"
	@echo "✅ Installed:"
	@echo "   $(BINDIR)/gallium  — ReAct agent: REPL + app-server (spawned by rs-kessel / klein-cli). Self-contained."
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; *) echo "   ⚠️  $(BINDIR) is not on your PATH — add it to use 'gallium' directly." ;; esac

# Run the CLI capability matrix (all testcases × all available backends).
# Filter with TESTS=... / BACKENDS=...; override the binary with CLI=...
testsuite:
	@if [ "$(CLI)" = "$(GALLIUM_TESTSUITE_CLI)" ]; then $(CARGO) build --release -p gallium-agent $(FEATURES_FLAG); fi
	@CLI="$(CLI)" bash testsuite/matrix_runner.sh

# Same matrix, local backends only (no OPENAI_API_KEY required). Keep in sync with
# the testsuite/backends/*.toml that carry a `modelPath` — every other one is cloud.
LOCAL_BACKENDS ?= gemma4,gemma4-26b,gpt-oss,lfm2,qwen3.6
testsuite-local:
	@if [ "$(CLI)" = "$(GALLIUM_TESTSUITE_CLI)" ]; then $(CARGO) build --release -p gallium-agent $(FEATURES_FLAG); fi
	@CLI="$(CLI)" BACKENDS="$(LOCAL_BACKENDS)" bash testsuite/matrix_runner.sh

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace -- -D warnings

clean:
	$(CARGO) clean

# Create a portable zip archive (excludes target/, references/, model weights, IDE files)
zip:
	cd .. && zip -r rs-gallium.zip rs-gallium/ \
		-x "rs-gallium/target/*" \
		-x "rs-gallium/references/*" \
		-x "rs-gallium/.git/*" \
		-x "rs-gallium/.claude/*" \
		-x "*.safetensors" \
		-x "*.gguf" \
		-x "*.bin" \
		-x "*.pt" \
		-x "*.onnx" \
		-x "*.pdf" \
		-x ".DS_Store" \
		-x "*.swp" \
		-x "*.swo"
	@echo "Created ../rs-gallium.zip"

# ── Run targets ───────────────────────────────────────────────────────────────
# The binary takes no model flags: settings come from environment variables
# layered over an optional TOML --config (env > config > default), and prompts
# arrive on stdin. Pick a model by pointing CONFIG at one of configs/*.toml, or
# skip the config entirely and export MODEL_PATH.
#
# Optional environment overrides (see README.md for the full list):
#   MODEL_PATH         local GGUF path, or hf:ORG/REPO[@REV]/file.gguf
#   INFERENCE_ENGINE   llamacpp (default) | gallium
#   MAX_TOKENS         max new tokens per turn
#   LLM_TEMPERATURE    sampling temperature
#   OPENAI_API_KEY     required by the cloud configs
CONFIG ?= configs/default.toml

# Interactive REPL (or one-shot when stdin is a pipe).
#   make run CONFIG=configs/qwen3.6.toml
#   echo "hi" | make run CONFIG=configs/gemma4.toml
run: build
	./target/release/gallium --config $(CONFIG)

# Whole-turn JSON-RPC backend on stdio — the mode rs-kessel and klein-cli spawn.
#   make run-app-server CONFIG=configs/openai.toml
run-app-server: build
	./target/release/gallium app-server --config $(CONFIG)

# Docker: build the gallium image (the `gallium` agent binary, env-var driven).
# Usage: make docker-build
DOCKER_IMAGE ?= gallium
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Build the image that runs the agent testsuite on Linux.
docker-build-integration:
	docker build -f Dockerfile.integration -t gallium-integration .

# Deprecated misspelling, kept so existing scripts keep working.
docker-build-intgration: docker-build-integration

# Docker: run the agent testsuite inside the integration image, with the host's
# HuggingFace cache and a logs dir mounted.
# Usage: make docker-run-integration ARGS="capital gemma4"
ARGS ?=
docker-run-integration:
	docker run --rm \
		-v "$(HOME)/.cache/huggingface:/root/.cache/huggingface" \
		-v "$${TMPDIR:-/tmp}:/logs" \
		$(if $(HUGGING_FACE_HUB_TOKEN),-e HUGGING_FACE_HUB_TOKEN) \
		gallium-integration $(ARGS)
