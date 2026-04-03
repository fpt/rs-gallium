---
name: ollama-narrow
description: Narrow down the root cause of a gallium inference bug by comparing our output against ollama running the same GGUF at greedy sampling. Triggered when the user asks to "narrow down", "compare with ollama", "find root cause", "why does our output differ from ollama", "debug inference divergence", or when a gallium-agent output is clearly malformed and a matching ollama model exists.
argument-hint: "[arch] [prompt-file]  — e.g. gemma4 /tmp/prompt.txt"
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Ollama Narrowing Skill

Use this when you need to decide whether a bug is in:
(a) **prompt formatting / tokenization** — our rendered prompt differs from what the reference expects
(b) **inference** — identical prompts produce different outputs
(c) **sampling** — identical prompts + same greedy params diverge late in generation

Ollama loads the same GGUF we load, so differences between our output and ollama's on identical inputs localise the bug to our forward pass.

## Preconditions

1. Ollama is installed and running. Check: `curl -s http://localhost:11434/api/tags | head`. If not running, tell the user to start `ollama serve` (or `brew services start ollama`). Do NOT start it yourself — user has local state.
2. The same GGUF is pullable by ollama. Map:

| Gallium `--hf-repo / --hf-file`                                  | Ollama model tag  |
|------------------------------------------------------------------|-------------------|
| `unsloth/gemma-4-E4B-it-GGUF` / `gemma-4-E4B-it-Q4_K_M.gguf`     | `gemma4:e4b`      |
| `unsloth/Qwen3.5-*-GGUF` / `*-Q4_K_M.gguf`                       | `qwen3.5:*`       |
| `unsloth/gpt-oss-20b-GGUF` / `*-mxfp4.gguf`                      | `gpt-oss:20b`     |

If the user's model isn't in ollama's library, report "ollama narrowing not applicable — no equivalent ollama model" and stop.

3. HF tokenizer for the architecture is cacheable. Needed to tokenize both prompts and diff at the token-id level.

## Step 1 — Dump our prompt

Our provider supports `DUMP_PROMPT=1` (see `crates/gallium-agent/src/provider.rs`). Run:

```bash
DUMP_PROMPT=1 ./target/release/gallium-agent \
    --arch <arch> --format gguf \
    --hf-repo <repo> --hf-file <file> \
    --hf-tokenizer-repo <tokenizer-repo> \
    --max-tokens 1 --temperature 0.0 \
    -f <prompt-file> 2>/tmp/dump.log
```

`--max-tokens 1` avoids long generation when you only want the prompt. Extract the text between `===== PROMPT BEGIN =====` and `===== PROMPT END =====` in `/tmp/dump.log` → `/tmp/ours_prompt.txt`.

## Step 2 — Render ollama's expected prompt

Locate ollama's renderer for this architecture:

| Arch | Renderer file |
|------|--------------|
| gemma4 | `~/go/pkg/mod/github.com/ollama/ollama@<version>/model/renderers/gemma4.go` (function `Gemma4Renderer.Render`) |
| gpt-oss | `model/renderers/harmony.go` |
| qwen | `model/renderers/chatml.go` (varies by variant) |

Read the `Render` function. Build a small Go script (or, if simpler, replicate the logic in Python) that produces the same prompt string given the same `(messages, tools)` input that gallium-agent used. If the gallium prompt was built from `ChatMessage` + `ToolDefinition` values, construct the equivalent ollama inputs.

Dump that to `/tmp/ollama_prompt.txt`.

## Step 3 — Tokenize both and diff

Use `uv` (not `python3`; the repo convention is `uv run --with <pkg>`):

```bash
uv run --with tokenizers --with requests python3 <<'PY'
from tokenizers import Tokenizer
t = Tokenizer.from_pretrained("google/gemma-4-E4B")   # or arch-appropriate repo
our = open("/tmp/ours_prompt.txt").read()
oll = open("/tmp/ollama_prompt.txt").read()
ot = t.encode(our).ids
lt = t.encode(oll).ids
print(f"ours: {len(ot)} tokens, ollama: {len(lt)} tokens")
for i, (a, b) in enumerate(zip(ot, lt)):
    if a != b:
        print(f"first divergence at index {i}: ours={a} ollama={b}")
        print(f"  context ours:   ...{t.decode(ot[max(0,i-3):i+3])}")
        print(f"  context ollama: ...{t.decode(lt[max(0,i-3):i+3])}")
        break
else:
    print("token IDs match in the overlap" if len(ot) == len(lt) else "prefix matches but lengths differ")
PY
```

**Verdict so far:**
- Tokens differ → bug is in our prompt formatting (`format_prompt_with_tools` in `protocol.rs`, or tokenizer special-token handling). Stop here and fix that.
- Tokens match → proceed.

## Step 4 — Feed the *exact* prompt to ollama raw

`raw:true` bypasses ollama's own chat templating so we test inference only:

```bash
PROMPT=$(cat /tmp/ours_prompt.txt | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
curl -s http://localhost:11434/api/generate -d @- <<EOF > /tmp/ollama_out.json
{"model":"<ollama-tag>","prompt":$PROMPT,"raw":true,"stream":false,"options":{"temperature":0.0,"num_predict":256}}
EOF
python3 -c 'import json; d=json.load(open("/tmp/ollama_out.json")); print(d.get("response",""))'
```

Save the response. This is the ground truth for "correct inference on this exact prompt at greedy".

## Step 5 — Run ours at greedy on the same prompt

```bash
./target/release/gallium-agent --arch <arch> --format gguf \
    --hf-repo <repo> --hf-file <file> --hf-tokenizer-repo <tokenizer-repo> \
    --max-tokens 256 --temperature 0.0 -f /tmp/ours_prompt.txt > /tmp/ours_out.txt
```

Diff `/tmp/ours_out.txt` vs ollama's response.

- **First-token matches** → sampling, not inference; check top-k/top-p, EOS handling.
- **First-token differs** → forward pass bug; continue narrowing.

## Step 6 — Length / content narrowing matrix

If inference is suspect, find the seq_len boundary where divergence starts by running shortened / alternate prompts:

| Case | What it rules in/out |
|------|----------------------|
| 17-tok plain prompt (`"Hello, world"`) — do outputs' first 5 tokens match? | If YES: Q4 dequant, weight loading, per-token ops, embeddings, sampling are all fine at short context |
| ~500-tok plain filler | If YES: forward pass is correct up to the sliding window boundary |
| ~600-tok plain filler | If ours degrades here but ollama doesn't: bug activates when `seq_len > sliding_window` — suspect **sliding-window attention, long-range RoPE, or KV cache at boundary** |
| Prompt with N tools (drives length up without changing "content") | Confirms the bug is length-driven, not content-driven |

Run each case through **both** ours and ollama. Tabulate outputs. The finest-grained change in prompt that flips ours from correct to broken (while ollama stays correct) localises the trigger.

## Step 7 — Name the suspect, inspect the GGUF

Once the trigger is known (e.g. "breaks when seq_len > 512"), read the relevant GGUF metadata + tensor shapes and compare against what our loader reads. `gguf` Python lib:

```bash
uv run --with gguf python3 <<'PY'
from gguf import GGUFReader
r = GGUFReader("/Users/$USER/.cache/huggingface/hub/models--<repo>/snapshots/*/<file>.gguf")
for key in ["<arch>.attention.sliding_window_pattern",
            "<arch>.attention.sliding_window",
            "<arch>.rope.dimension_count",
            "<arch>.rope.freq_base"]:
    f = r.get_field(key)
    print(key, "=", f.parts[f.data[0]].tolist() if f else "<missing>")
# Dump a tensor to see its actual content
t = next(t for t in r.tensors if t.name == "rope_freqs.weight")
print(t.name, list(t.shape), t.data.tolist()[:8], "...", t.data.tolist()[-4:])
PY
```

Cross-reference with how ollama's Go model reads the same keys (`model/models/<arch>/model_text.go` in the ollama source tree under `~/go/pkg/mod/...`). Our loader and ollama's loader should treat the same GGUF keys the same way. Discrepancies here are high-yield bugs (e.g. `rope_freqs.weight` as divisors vs inv_freq — see Gemma 4 Bug 10 in `docs/gemma4.md`).

## Step 8 — Write the fix, re-run step 5 at the failing length

The fix is valid iff:
1. Our output at the failing length now matches ollama's (exact or semantically equivalent).
2. Previously-passing short prompts still pass.

Update `docs/<arch>.md` "Debugging History" and the relevant memory file with what you learned.

## Guardrails

- Never start `ollama serve` yourself. If it's not running, stop and ask the user.
- Never modify the GGUF cache. Never delete files under `~/.cache/huggingface/`.
- Always use `uv run --with <pkg>` — never `python3` directly (project convention, see `CLAUDE.md`).
- If ollama says `model not found`, tell the user to `ollama pull <tag>` — do not pull on their behalf without asking (downloads several GB).
- Keep temperature at 0.0 and sampling greedy for every comparison. Any divergence under greedy is a real bug; under temperature > 0 divergence is expected.
- If our prompt and ollama's differ but the difference is only whitespace / BOS token, fix tokenizer handling before assuming it's a format bug.

## Output format

Always end your investigation with:

```
## Root cause

<one-sentence location — file:line and what it does wrong>

## Evidence chain

1. Prompt tokenization: <match / differ>
2. Ollama raw response: <first 100 chars or "malformed">
3. Ours at greedy: <first 100 chars or "malformed">
4. Length boundary: <seq_len where divergence begins>
5. GGUF vs loader mismatch: <what key / tensor / shape the loader misreads>

## Fix sketch

<file:line — what to change, in 2-3 lines>
```
