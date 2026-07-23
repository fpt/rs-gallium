.PHONY: build check test fmt fmt-check clippy clean zip install \
	run run-app-server \
	docker-build docker-build-integration docker-build-intgration \
	docker-run-integration \
	testsuite testsuite-local

# Install location (override with: make install PREFIX=/usr/local)
PREFIX ?= $(HOME)
BINDIR := $(PREFIX)/bin

# Testsuite driver. Defaults to the `gallium` binary via testsuite/gallium_cli.sh,
# which locates the binary and forwards `--config <backend.toml>` (prompts arrive
# on stdin). Override CLI= to drive a different backend binary:
#   make testsuite CLI=/path/to/other-app-server
GALLIUM_TESTSUITE_CLI := $(CURDIR)/testsuite/gallium_cli.sh
CLI ?= $(GALLIUM_TESTSUITE_CLI)

build:
	cargo build --release

check:
	cargo check --workspace

test:
	cargo test --workspace

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
	@if [ "$(CLI)" = "$(GALLIUM_TESTSUITE_CLI)" ]; then cargo build --release -p gallium-agent; fi
	@CLI="$(CLI)" bash testsuite/matrix_runner.sh

# Same matrix, local backends only (no OPENAI_API_KEY required). Keep in sync with
# the testsuite/backends/*.toml that carry a `modelPath` — every other one is cloud.
LOCAL_BACKENDS ?= gemma4,gemma4-26b,gpt-oss,lfm2,qwen3.6
testsuite-local:
	@if [ "$(CLI)" = "$(GALLIUM_TESTSUITE_CLI)" ]; then cargo build --release -p gallium-agent; fi
	@CLI="$(CLI)" BACKENDS="$(LOCAL_BACKENDS)" bash testsuite/matrix_runner.sh

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace -- -D warnings

clean:
	cargo clean

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

# Docker: build the gallium image.
# NOTE: the top-level Dockerfile still builds the removed `gallium-cli` crate and
# does not work — see issue #3. `docker-build-integration` below is the one that does.
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
