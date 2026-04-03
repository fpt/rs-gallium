.PHONY: build check test fmt clippy clean zip run-gpt-oss run-gpt-oss-gguf run-gemma4 run-gemma4-gguf run-qwen35 run-qwen35-gguf run-agent run-agent-openai docker-build docker-run

build:
	cargo build --release

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

# Run GPT-OSS 20B from HuggingFace with chat template
# Usage: make run-gpt-oss PROMPT="What is the capital of France?"
PROMPT ?= Hello
run-gpt-oss:
	cargo run --release -p gallium-cli -- \
		--arch gpt-oss \
		--format safetensors \
		--hf-repo openai/gpt-oss-20b \
		--dtype f16 \
		--chat \
		--prompt "$(PROMPT)"

# Run GPT-OSS 20B GGUF (Q4_K_M)
# Usage: make run-gpt-oss-gguf PROMPT="What is the capital of France?"
# Options: PROMPT, GPT_OSS_GGUF_MAX_TOKENS (default 256), GPT_OSS_GGUF_TEMPERATURE (default 0.7)
GPT_OSS_GGUF_MAX_TOKENS ?= 256
GPT_OSS_GGUF_TEMPERATURE ?= 0.7
run-gpt-oss-gguf:
	cargo run --release -p gallium-cli -- \
		--arch gpt-oss \
		--format gguf \
		--hf-repo unsloth/gpt-oss-20b-GGUF \
		--hf-file gpt-oss-20b-Q4_K_M.gguf \
		--hf-tokenizer-repo openai/gpt-oss-20b \
		--chat \
		--prompt "$(PROMPT)" \
		--max-tokens $(GPT_OSS_GGUF_MAX_TOKENS) \
		--temperature $(GPT_OSS_GGUF_TEMPERATURE)

run-gemma4:
	cargo run --release -p gallium-cli -- \
        --arch gemma4 \
        --format safetensors \
        --hf-repo google/gemma-4-E4B \
        --dtype bf16 \
        --prompt "The capital of France is" \
        --max-tokens 20

# Run Gemma 4 E4B GGUF (Q4_K_M)
# Usage: make run-gemma4-gguf PROMPT="The capital of France is"
# Options: PROMPT, GEMMA4_GGUF_MAX_TOKENS (default 20), GEMMA4_GGUF_TEMPERATURE (default 0.7)
GEMMA4_GGUF_MAX_TOKENS ?= 20
GEMMA4_GGUF_TEMPERATURE ?= 0.7
run-gemma4-gguf:
	cargo run --release -p gallium-cli -- \
		--arch gemma4 \
		--format gguf \
		--hf-repo unsloth/gemma-4-E4B-it-GGUF \
		--hf-file gemma-4-E4B-it-Q4_K_M.gguf \
		--hf-tokenizer-repo google/gemma-4-E4B \
		--prompt "$(PROMPT)" \
		--max-tokens $(GEMMA4_GGUF_MAX_TOKENS) \
		--temperature $(GEMMA4_GGUF_TEMPERATURE)

run-qwen35:
	cargo run --release -p gallium-cli -- \
		--arch qwen35 \
		--format safetensors \
		--hf-repo Qwen/Qwen3.5-9B \
		--dtype f16 \
		--prompt "$(PROMPT)" \
		--max-tokens 256

# Run Qwen3.5-9B GGUF (Q4_K_M)
# Qwen3.5-9B is a base model. Few-shot context is required for reliable factual output.
# Good:  make run-qwen35-gguf PROMPT="The capital of Japan is Tokyo. The capital of France is"
# Bad:   make run-qwen35-gguf PROMPT="The capital of France is"  (no context → unpredictable)
# Options: PROMPT, MAX_TOKENS (default 32), TEMPERATURE (default 0.0)
QWEN35_GGUF_MAX_TOKENS ?= 32
QWEN35_GGUF_TEMPERATURE ?= 0.0
run-qwen35-gguf:
	cargo run --release -p gallium-cli -- \
		--arch qwen35 \
		--format gguf \
		--hf-repo unsloth/Qwen3.5-9B-GGUF \
		--hf-file Qwen3.5-9B-Q4_K_M.gguf \
		--hf-tokenizer-repo Qwen/Qwen3.5-9B \
		--prompt "$(PROMPT)" \
		--max-tokens $(QWEN35_GGUF_MAX_TOKENS) \
		--temperature $(QWEN35_GGUF_TEMPERATURE)

# Run gallium-agent with local GPT-OSS (interactive REPL, plain chat)
# Usage: make run-agent
# Options: AGENT_SYSTEM_PROMPT (optional system prompt)
run-agent:
	cargo run --release -p gallium-agent -- \
		--arch gpt-oss \
		--hf-repo openai/gpt-oss-20b \
		--dtype f16 \
		$(if $(AGENT_SYSTEM_PROMPT),--system-prompt "$(AGENT_SYSTEM_PROMPT)")

# Run gallium-agent with OpenAI (full ReAct loop with tools)
# Requires OPENAI_API_KEY env var or --openai-api-key flag.
# Usage: make run-agent-openai
# Options: AGENT_OPENAI_MODEL (default gpt-4o-mini), AGENT_SYSTEM_PROMPT
AGENT_OPENAI_MODEL ?= gpt-4o-mini
run-agent-openai:
	cargo run --release -p gallium-agent -- \
		--provider openai \
		--openai-model $(AGENT_OPENAI_MODEL) \
		$(if $(AGENT_SYSTEM_PROMPT),--system-prompt "$(AGENT_SYSTEM_PROMPT)")

# Run with GGUF model
# Usage: make run-gguf ARCH=gpt-oss MODEL=/path/to/model.gguf PROMPT="Hello"
run-gguf:
	cargo run --release -p gallium-cli -- \
		--arch $(ARCH) \
		--format gguf \
		--model $(MODEL) \
		--prompt "$(PROMPT)"

# Run with safetensors model
# Usage: make run ARCH=gpt-oss MODEL=/path/to/model-dir PROMPT="Hello"
run:
	cargo run --release -p gallium-cli -- \
		--arch $(ARCH) \
		--format safetensors \
		--model $(MODEL) \
		--prompt "$(PROMPT)"

# Docker: build the gallium image
# Usage: make docker-build
DOCKER_IMAGE ?= gallium
docker-build:
	docker build -t $(DOCKER_IMAGE) .

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
