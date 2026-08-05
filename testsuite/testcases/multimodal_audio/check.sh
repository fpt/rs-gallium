#!/usr/bin/env bash
# Audio: the transcription must contain the codeword spoken in speech.wav.
#
# Passes on the llama.cpp backend when `mmprojPath` names a projector with an
# audio encoder (every Gemma 4 ships one). Everywhere else the turn is refused,
# which is a pass for the refusal and a fail for this testcase — the point is
# that no backend answers about a clip it never received.
#
# $1 = output file, $2 = error file. cwd = temp test dir.
set -uo pipefail

resp="$(./extract_response.sh "$1")"

# speech.wav is in the agent's cwd, so a pass could also come from Bash running
# something over the samples rather than from the model hearing them. Not the
# capability under test.
if grep -q 'ReAct iteration .*tool call' "$2"; then
    echo "✗ the agent used tools — speech.wav was processed, not heard."
    echo "  reply: $resp"
    exit 1
fi

# The codeword alone. Exact transcription is too strict a bar for an audio front
# end llama.cpp itself labels experimental: "codeword"/"code word" both appear
# depending on the model, and articles drift. Hearing the distinctive noun is
# what separates a model that got the audio from one that did not.
if grep -iq "zucchini" <<<"$resp"; then
    echo "✓ transcribed 'zucchini' out of the audio"
    exit 0
fi

# Heard-but-garbled is a different result from did-not-hear, and saying so is
# the point of this suite. Gemma 4 12B lands here: it transcribes "the secret
# code word is zuki" — unmistakably the right sentence, one word short. Its
# projector is encoder-free (linear layers, no audio transformer stack), which
# buys a 167 MB download at the cost of exactly this.
if grep -iqE "secret|code ?word" <<<"$resp"; then
    echo "✗ heard the clip but did not transcribe 'zucchini'."
    echo "  reply: $resp"
    echo "  note: the audio reached the model — the rest of the sentence is"
    echo "        there. This is transcription quality, not a broken path."
    exit 1
fi

echo "✗ 'zucchini' not found — the model did not hear the clip."
echo "  reply: $resp"
# Distinguishable causes, each wanting a different fix.
if grep -q "no multimodal projector" "$2"; then
    echo "  cause: no mmprojPath configured for this backend."
elif grep -q "no audio encoder" "$2"; then
    echo "  cause: the configured projector is vision-only."
elif grep -q "does not carry audio\|cannot see images" "$2"; then
    echo "  cause: this backend has no audio path at all (OpenAI/candle)."
fi
exit 1
