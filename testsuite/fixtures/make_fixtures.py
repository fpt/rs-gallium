#!/usr/bin/env python3
"""Generate the multimodal testcase fixtures.

Run from the repo root:

    uv run python testsuite/fixtures/make_fixtures.py

The outputs are committed, so a test run needs neither Python nor this script.
It is here so the fixtures can be regenerated or adjusted — a test whose input
nobody can rebuild is a test nobody can change.

Standard library only, deliberately: a fixture generator that needs Pillow and
NumPy is one more thing to install before the testsuite can be touched.
"""

import pathlib
import struct
import zlib

# Fixtures live beside the testcase that uses them: `runner.sh` copies a
# testcase directory wholesale into the temp workdir, and only what is in there
# reaches the agent's cwd.
TESTSUITE = pathlib.Path(__file__).resolve().parent.parent
REPO = TESTSUITE.parent
IMAGE_CASE = TESTSUITE / "testcases" / "multimodal_image"
AUDIO_CASE = TESTSUITE / "testcases" / "multimodal_audio"

# ---------------------------------------------------------------------------
# Image: a two-digit number, drawn large enough that reading it is not the hard
# part. The number matters — asking for a *colour* is a test a blind model
# passes one time in six by guessing, and asking for a single digit one time in
# ten. Two digits puts a lucky guess at 1%.
# ---------------------------------------------------------------------------

ANSWER = "42"

# 5x7 block glyphs. '#' is ink.
GLYPHS = {
    "4": [
        "...#.",
        "..##.",
        ".#.#.",
        "#..#.",
        "#####",
        "...#.",
        "...#.",
    ],
    "2": [
        ".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#####",
    ],
}

SCALE = 32  # pixels per glyph cell
MARGIN = 40
GAP = 32


def render_number(text: str) -> tuple[int, int, list[list[int]]]:
    """Rasterize `text` to a greyscale pixel grid: 255 = white, 0 = ink."""
    glyph_w = 5 * SCALE
    glyph_h = 7 * SCALE
    width = MARGIN * 2 + len(text) * glyph_w + GAP * (len(text) - 1)
    height = MARGIN * 2 + glyph_h

    rows = [[255] * width for _ in range(height)]
    for index, char in enumerate(text):
        left = MARGIN + index * (glyph_w + GAP)
        for gy, line in enumerate(GLYPHS[char]):
            for gx, cell in enumerate(line):
                if cell != "#":
                    continue
                for dy in range(SCALE):
                    row = rows[MARGIN + gy * SCALE + dy]
                    start = left + gx * SCALE
                    row[start : start + SCALE] = [0] * SCALE
    return width, height, rows


def write_png(path: pathlib.Path, width: int, height: int, rows: list[list[int]]) -> None:
    """Write an 8-bit greyscale PNG. No filtering — these images are tiny."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # Each scanline is prefixed with its filter type (0 = None).
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# Audio: a spoken sentence, because that is the thing these models can actually
# do. The first version of this fixture was a pure tone and the test asked for
# its frequency in hertz — which nothing passes: Gemma 4 12B heard the clip
# (mtmd encoded it, the token count proves it) and still answered "440" for a
# 613 Hz tone. Pitch estimation is not what an audio LLM is for, and llama.cpp
# labels its audio front end experimental besides.
#
# Transcription separates hearing from not-hearing cleanly: the phrase is
# nowhere in the prompt, so a model without the audio has nothing to transcribe.
#
# The codeword matches testcases/file_read on purpose — same word, two very
# different routes to it.
#
# Generated with macOS `say` + `afconvert`. That makes *regeneration*
# macOS-only, which is acceptable because the .wav is committed: a test run
# needs nothing. On another platform, any 16 kHz mono WAV of someone saying
# PHRASE will do.
# ---------------------------------------------------------------------------

PHRASE = "The secret codeword is zucchini."
SAMPLE_RATE = 16_000


def write_speech(path: pathlib.Path) -> None:
    import shutil
    import subprocess
    import tempfile

    for tool in ("say", "afconvert"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"{tool} not found — speech generation needs macOS.\n"
                f"The committed {path.name} is what tests use; regenerate it on a Mac, "
                f"or supply any 16 kHz mono WAV saying: {PHRASE!r}"
            )

    with tempfile.TemporaryDirectory() as tmp:
        aiff = pathlib.Path(tmp) / "speech.aiff"
        subprocess.run(["say", "-o", str(aiff), PHRASE], check=True)
        # 16 kHz mono 16-bit: what the Gemma projectors declare
        # (clip.audio.num_mel_bins = 128 over a 16 kHz sample rate).
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
             str(aiff), str(path)],
            check=True,
        )


def main() -> None:
    width, height, rows = render_number(ANSWER)
    png = IMAGE_CASE / "number.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    write_png(png, width, height, rows)
    print(f"{png.relative_to(REPO)}: {width}x{height} showing {ANSWER!r}, "
          f"{png.stat().st_size} bytes")

    wav = AUDIO_CASE / "speech.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    write_speech(wav)
    print(f"{wav.relative_to(REPO)}: {PHRASE!r}, {wav.stat().st_size} bytes")


if __name__ == "__main__":
    main()
