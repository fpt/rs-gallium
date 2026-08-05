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
import wave
import zlib
from math import pi, sin

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
# Audio: a pure tone. A spoken phrase would need a TTS dependency and a voice to
# ship; a tone is generatable from the standard library and still asks a real
# question of an audio model — a model that cannot hear has no way to name the
# pitch.
#
# 613 Hz, and deliberately *not* 440. A 440 Hz fixture makes the test worthless:
# A4 is the tone everyone reaches for, and a model that never received the audio
# answers "440" from the word "tone" alone. This was not hypothetical — the
# first version of this fixture was 440 Hz and gpt-5.6-luna passed it without
# ever being sent a single sample. 613 is a prime number of hertz, near no
# musical note, and unreachable by guessing.
# ---------------------------------------------------------------------------

TONE_HZ = 613
SAMPLE_RATE = 16_000
SECONDS = 2.0
AMPLITUDE = 0.6


def write_wav(path: pathlib.Path) -> None:
    frames = bytearray()
    for n in range(int(SAMPLE_RATE * SECONDS)):
        value = int(AMPLITUDE * 32767 * sin(2 * pi * TONE_HZ * n / SAMPLE_RATE))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(bytes(frames))


def main() -> None:
    width, height, rows = render_number(ANSWER)
    png = IMAGE_CASE / "number.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    write_png(png, width, height, rows)
    print(f"{png.relative_to(REPO)}: {width}x{height} showing {ANSWER!r}, "
          f"{png.stat().st_size} bytes")

    wav = AUDIO_CASE / "tone.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    write_wav(wav)
    print(f"{wav.relative_to(REPO)}: {TONE_HZ} Hz, {wav.stat().st_size} bytes")


if __name__ == "__main__":
    main()
