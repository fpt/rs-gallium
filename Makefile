.PHONY: build build-rust build-swift gen-uniffi check test fmt clippy clean zip \
	run-text run-voice \
	run-agent run-agent-local run-agent-gguf \
	run-gpt-oss run-gpt-oss-gguf run-gemma4-gguf run-gemma4-e2b-gguf run-gemma4-12b-gguf run-qwen35-gguf \
	run-agent-openai docker-build docker-run \
	gen-uniffi-cs build-winui run-winui

SWIFT_VENDOR_DIR := swift/vendor/uniffi-swift
GALLIUM_AGENT_LIB := target/release/libgallium_agent.a
SWIFT_BUILD_DIR := swift/.build/release

build: build-rust gen-uniffi build-swift

build-rust:
	cargo build --release

# Generate Swift+C bindings from the UDL (requires Rust lib already built)
gen-uniffi: $(GALLIUM_AGENT_LIB)
	mkdir -p $(SWIFT_VENDOR_DIR)
	cargo run --release --bin uniffi-bindgen -- generate \
		--language swift \
		--lib-file $(GALLIUM_AGENT_LIB) \
		--out-dir $(SWIFT_VENDOR_DIR) \
		crates/gallium-agent/src/agent.udl
	ln -sf ../../../vendor/uniffi-swift/gallium_agent.swift \
		swift/Sources/AgentBridge/gallium_agent.swift

build-swift:
	cd swift && swift build -c release

check:
	cargo check --workspace

test:
	cargo test --workspace

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace -- -D warnings

clean:
	cargo clean
	rm -rf swift/.build swift/vendor/uniffi-swift swift/Sources/AgentBridge/gallium_agent.swift

# Create a portable zip archive (excludes target/, external/, model weights, IDE files)
zip:
	cd .. && zip -r rs-gallium.zip rs-gallium/ \
		-x "rs-gallium/target/*" \
		-x "rs-gallium/external/*" \
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

# Run Swift CLI in text mode
# Usage: make run-text [GALLIUM_CONFIG=configs/my.yaml]
GALLIUM_CONFIG ?= configs/default.yaml
run-text: build
	$(SWIFT_BUILD_DIR)/gallium --config $(GALLIUM_CONFIG)

# Run Swift CLI in voice mode (requires macOS 26+ and microphone permission)
# Usage: make run-voice [GALLIUM_CONFIG=configs/my.yaml]
run-voice: build
	$(SWIFT_BUILD_DIR)/gallium --config $(GALLIUM_CONFIG) --voice

# ── Local model targets ───────────────────────────────────────────────────────
# Shared optional overrides (apply to all run-agent-* and canned targets):
#   DTYPE            weight dtype: f16 (default), bf16, f32  [safetensors only]
#   MAX_TOKENS       max new tokens per turn (default: 512)
#   TEMPERATURE      sampling temperature (default: 0.7)
#   AGENT_SYSTEM_PROMPT  optional system prompt
DTYPE        ?=
MAX_TOKENS   ?=
TEMPERATURE  ?=

# Generic safetensors target
# Usage: make run-agent-local ARCH=gemma4 HF_REPO=google/gemma-4-E4B DTYPE=bf16
run-agent-local:
	cargo run --release -p gallium-agent --bin gallium-agent -- \
		--arch $(ARCH) \
		--format safetensors \
		$(if $(HF_REPO),--hf-repo $(HF_REPO)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(DTYPE),--dtype $(DTYPE)) \
		$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS)) \
		$(if $(TEMPERATURE),--temperature $(TEMPERATURE)) \
		$(if $(AGENT_SYSTEM_PROMPT),--system-prompt "$(AGENT_SYSTEM_PROMPT)")

# Generic GGUF target
# Usage: make run-agent-gguf ARCH=gpt-oss HF_REPO=unsloth/gpt-oss-20b-GGUF \
#              HF_FILE=gpt-oss-20b-Q4_K_M.gguf HF_TOKENIZER_REPO=openai/gpt-oss-20b
run-agent-gguf:
	cargo run --release -p gallium-agent --bin gallium-agent -- \
		--arch $(ARCH) \
		--format gguf \
		$(if $(HF_REPO),--hf-repo $(HF_REPO)) \
		$(if $(HF_FILE),--hf-file $(HF_FILE)) \
		$(if $(HF_TOKENIZER_REPO),--hf-tokenizer-repo $(HF_TOKENIZER_REPO)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS)) \
		$(if $(TEMPERATURE),--temperature $(TEMPERATURE)) \
		$(if $(AGENT_SYSTEM_PROMPT),--system-prompt "$(AGENT_SYSTEM_PROMPT)")

# Canned: GPT-OSS 20B safetensors
# Usage: make run-gpt-oss [DTYPE=f16] [MAX_TOKENS=512]
run-gpt-oss:
	$(MAKE) run-agent-local ARCH=gpt-oss HF_REPO=openai/gpt-oss-20b \
		DTYPE=$(or $(DTYPE),f16)

# Canned: GPT-OSS 20B Q4_K_M GGUF
# Usage: make run-gpt-oss-gguf [MAX_TOKENS=512] [TEMPERATURE=0.7]
run-gpt-oss-gguf:
	$(MAKE) run-agent-gguf ARCH=gpt-oss \
		HF_REPO=unsloth/gpt-oss-20b-GGUF \
		HF_FILE=gpt-oss-20b-Q4_K_M.gguf \
		HF_TOKENIZER_REPO=openai/gpt-oss-20b

# Canned: Gemma 4 E2B Q4_K_M GGUF
# Usage: make run-gemma4-e2b-gguf [MAX_TOKENS=512] [TEMPERATURE=0.7]
run-gemma4-e2b-gguf:
	$(MAKE) run-agent-gguf ARCH=gemma4 \
		HF_REPO=unsloth/gemma-4-E2B-it-GGUF \
		HF_FILE=gemma-4-E2B-it-Q4_K_M.gguf \
		HF_TOKENIZER_REPO=google/gemma-4-E2B

# Canned: Gemma 4 E4B Q4_K_M GGUF
# Usage: make run-gemma4-gguf [MAX_TOKENS=512] [TEMPERATURE=0.7]
run-gemma4-gguf:
	$(MAKE) run-agent-gguf ARCH=gemma4 \
		HF_REPO=unsloth/gemma-4-E4B-it-GGUF \
		HF_FILE=gemma-4-E4B-it-Q4_K_M.gguf \
		HF_TOKENIZER_REPO=google/gemma-4-E4B

# Canned: Gemma 4 12B Q4_K_M GGUF
# Usage: make run-gemma4-12b-gguf [MAX_TOKENS=512] [TEMPERATURE=0.7]
run-gemma4-12b-gguf:
	$(MAKE) run-agent-gguf ARCH=gemma4 \
		HF_REPO=unsloth/gemma-4-12B-it-GGUF \
		HF_FILE=gemma-4-12b-it-Q4_K_M.gguf \
		HF_TOKENIZER_REPO=google/gemma-4-12B-it

# Canned: Qwen 3.5 9B Q4_K_M GGUF
# Usage: make run-qwen35-gguf [MAX_TOKENS=512] [TEMPERATURE=0.7]
run-qwen35-gguf:
	$(MAKE) run-agent-gguf ARCH=qwen35 \
		HF_REPO=unsloth/Qwen3.5-9B-GGUF \
		HF_FILE=Qwen3.5-9B-Q4_K_M.gguf \
		HF_TOKENIZER_REPO=Qwen/Qwen3.5-9B

# gallium-agent with local GPT-OSS (interactive REPL, plain chat)
run-agent:
	$(MAKE) run-gpt-oss

# Run gallium-agent with OpenAI (full ReAct loop with tools)
# Requires OPENAI_API_KEY env var or --openai-api-key flag.
# Usage: make run-agent-openai
# Options: AGENT_OPENAI_MODEL (default gpt-4o-mini), AGENT_SYSTEM_PROMPT
AGENT_OPENAI_MODEL ?= gpt-5.4-mini
run-agent-openai:
	cargo run --release -p gallium-agent --bin gallium-agent -- \
		--provider openai \
		--openai-model $(AGENT_OPENAI_MODEL) \
		$(if $(AGENT_SYSTEM_PROMPT),--system-prompt "$(AGENT_SYSTEM_PROMPT)")

# Docker: build the gallium image
# Usage: make docker-build
DOCKER_IMAGE ?= gallium
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-build-intgration:
	docker build -f Dockerfile.integration -t gallium-integration .

# ── Windows / WinUI 3 frontend ────────────────────────────────────────────────

GALLIUM_DLL   := target/release/gallium_agent.dll
WINUI_PROJECT := winui/GalliumWinUI/GalliumWinUI.csproj
WINUI_VENDOR  := winui/vendor
WINUI_EXE     := winui/GalliumWinUI/bin/x64/Release/net8.0-windows10.0.22621.0/GalliumWinUI.exe

# Generate C# P/Invoke bindings from the UDL (requires `make build-rust` first).
# Install the generator once: cargo install uniffi-bindgen-cs \
#   --git https://github.com/NordSecurity/uniffi-bindgen-cs --tag v0.9.1+v0.28.3
gen-uniffi-cs: $(GALLIUM_DLL)
	mkdir -p $(WINUI_VENDOR)
	uniffi-bindgen-cs generate \
		--library $(GALLIUM_DLL) \
		--out-dir $(WINUI_VENDOR) \
		crates/gallium-agent/src/agent.udl

# Build the WinUI 3 project (Release|x64).
build-winui:
	dotnet build $(WINUI_PROJECT) \
		-c Release \
		-p:Platform=x64 \
		--nologo

# Run the WinUI 3 app.
run-winui: build-winui
	"$(WINUI_EXE)"

# Docker: run with local HuggingFace cache mounted
# Usage: make docker-run ARCH=gemma4 FORMAT=gguf MODEL=/root/.cache/... PROMPT="Hello"
#   or with HF download:
#     make docker-run ARCH=gemma4 FORMAT=gguf HF_REPO=unsloth/gemma-4-E4B-it-GGUF \
#          HF_FILE=gemma-4-E4B-it-Q4_K_M.gguf HF_TOKENIZER_REPO=google/gemma-4-E4B \
#          PROMPT="The capital of France is"
FORMAT       ?= gguf
HF_REPO      ?=
HF_FILE      ?=
HF_TOKENIZER_REPO ?=
docker-run:
	docker run --rm \
		-v "$(HOME)/.cache/huggingface:/root/.cache/huggingface" \
		$(if $(HUGGING_FACE_HUB_TOKEN),-e HUGGING_FACE_HUB_TOKEN) \
		$(DOCKER_IMAGE) \
		--arch $(ARCH) \
		--format $(FORMAT) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(HF_REPO),--hf-repo $(HF_REPO)) \
		$(if $(HF_FILE),--hf-file $(HF_FILE)) \
		$(if $(HF_TOKENIZER_REPO),--hf-tokenizer-repo $(HF_TOKENIZER_REPO)) \
		--prompt "$(PROMPT)" \
		$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS))
