"""Print a GGUF metadata value — by default the embedded jinja chat template.

The chat template a model *actually* runs on is the one inside its GGUF, which
is not always the `chat_template.jinja` its Hugging Face repo publishes: a
quantizer may patch it on the way through. Anything asserting on template
behaviour (`llm_local`'s template fixtures) therefore wants this, not the Hub
file, whenever the GGUF is on hand.

    uv run python scripts/extract_chat_template.py MODEL.gguf > fixture.jinja
    uv run python scripts/extract_chat_template.py MODEL.gguf general.architecture
    uv run python scripts/extract_chat_template.py MODEL.gguf --list

Deliberately dependency-free (no `gguf` package): only the metadata block at the
head of the file is read, never the tensors, so this is fast on a 100GB model
and needs nothing installed.
"""

import struct
import sys

# GGUF scalar type ids → struct format. 8 (string) and 9 (array) are handled
# separately in `read_value`.
SCALARS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def read_string(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def read_value(f, type_id):
    if type_id == 8:
        return read_string(f)
    if type_id == 9:
        element_type = read(f, "I")
        count = read(f, "Q")
        return [read_value(f, element_type) for _ in range(count)]
    if type_id not in SCALARS:
        raise ValueError(f"unknown GGUF value type {type_id}")
    return read(f, SCALARS[type_id])


def metadata(path):
    """Yield (key, value) for every metadata pair, in file order."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        read(f, "I")  # version
        read(f, "Q")  # tensor count
        for _ in range(read(f, "Q")):
            key = read_string(f)
            yield key, read_value(f, read(f, "I"))


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    path, wanted = argv[0], (argv[1] if len(argv) > 1 else "tokenizer.chat_template")

    if wanted == "--list":
        for key, value in metadata(path):
            # Arrays are vocabularies and merge tables — thousands of entries
            # nobody wants printed. Name them and move on.
            if isinstance(value, list):
                print(f"{key} = [{len(value)} items]")
            elif isinstance(value, str) and len(value) > 120:
                print(f"{key} = <{len(value)} chars>")
            else:
                print(f"{key} = {value}")
        return 0

    for key, value in metadata(path):
        if key == wanted:
            # No trailing newline of our own: a template fixture should be the
            # bytes the model carries, so a diff against the GGUF is empty.
            sys.stdout.write(value if isinstance(value, str) else repr(value))
            return 0

    print(f"{path}: no metadata key {wanted!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
