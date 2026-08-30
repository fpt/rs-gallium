"""Which wire markers survive llama.cpp's `special=false` decode?

A family's tool-call/reasoning markers are parsed out of *decoded text*, but a
marker that is a CONTROL token in the GGUF's vocabulary decodes to nothing —
erasing the boundary before any parser runs. That is the LFM2.5 bug
`ModelProfile::restore_markers` exists for (its `<|tool_call_start|>` /
`<|tool_call_end|>` are CONTROL), and the Gemma 4 `<|think|>` finding (opener
CONTROL, closer plain text — an orphan pair). This script reads the vocabulary
straight from the GGUF header, no weights loaded, and says which case each
marker is:

    uv run --with gguf python scripts/marker_audit.py model.gguf '<marker>' ...

Verdicts:
  CONTROL       — dropped at special=false: a parser never sees it. Either the
                  marker is an EOG that ends generation before it could matter,
                  or the profile needs it in `restore_markers()`.
  USER_DEFINED / NORMAL — one token, text survives.
  NOT one token — ordinary text the model spells out; always survives.

Markers worth auditing per family (from crates/gallium-agent/src/profile/):
  lfm2:     <|tool_call_start|> <|tool_call_end|> <|im_end|>
  gemma4:   <|tool_call> <tool_call|> <|channel> <channel|> <|think|> <turn|>
  qwen3:    <tool_call> </tool_call> <|im_end|>
  gpt-oss:  <|channel|> <|message|> <|end|> <|return|> <|call|>
  deepseek / minimax: their DSML / XML tags (usually multi-token text)
"""

import sys

from gguf import GGUFReader

TYPES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, markers = sys.argv[1], sys.argv[2:]
    reader = GGUFReader(path)
    tokens = [
        bytes(t).decode("utf-8", "replace") if not isinstance(t, str) else t
        for t in reader.fields["tokenizer.ggml.tokens"].contents()
    ]
    token_types = reader.fields["tokenizer.ggml.token_type"].contents()
    index = {t: i for i, t in enumerate(tokens)}
    for marker in markers:
        i = index.get(marker)
        if i is None:
            print(f"{marker!r}: NOT one token (ordinary text -> always survives)")
            continue
        kind = TYPES.get(int(token_types[i]), str(token_types[i]))
        verdict = "DROPPED at special=false" if kind == "CONTROL" else "survives"
        print(f"{marker!r}: id={i} type={kind} -> {verdict}")


if __name__ == "__main__":
    main()
