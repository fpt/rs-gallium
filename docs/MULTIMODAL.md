# Multimodal support

What gallium can perceive, where, and what it costs. Current as of the mtmd
work (PR #76); the testsuite tables below are measured, not aspirational.

## Status at a glance

| | Image in | Audio in |
|---|---|---|
| **llama.cpp backend** (`local`, default) | ✅ with a projector | ✅ with a projector that has an audio encoder |
| **OpenAI backend** (Responses API) | ✅ | ❌ refuses |
| **native candle backend** (`candle`) | ❌ refuses | ❌ refuses |

Nothing is ever dropped silently. A backend that cannot carry an attachment
**fails the turn and says which piece is missing** — see [Refusals](#refusals).

Neither modality goes *out*: gallium does not generate images or audio.

## Using it

### From the REPL

A prompt line may carry `@image:<path>` and `@audio:<path>` markers. They are
lifted out of the text; what remains is the prompt.

```
@image:screenshot.png What is wrong with this layout?
@audio:memo.wav Transcribe this audio exactly.
@image:"my shot.png" @image:other.jpg Compare these two.
```

- Paths resolve against the agent's **working directory**, not the process cwd,
  so one config works from anywhere. Absolute paths are taken as given.
- Quote a path containing spaces: `@image:"my shot.png"`.
- Recognized only at a **whitespace boundary** — `mail user@image:host` stays
  text.
- Formats: `png`, `jpeg`, `gif`, `webp` for images; `wav`, `mp3`, `flac` for
  audio. These are what llama.cpp's bundled stb_image and miniaudio decode —
  gallium decodes neither format itself, it hands over the bytes.
- A marker whose file will not load **ends the turn with an error**. An
  attachment that silently vanished is indistinguishable from a model that
  cannot see, which is the distinction all of this exists to preserve.

Order matters and is preserved. Attachments live in one ordered
`Vec<MediaContent>` from parsing through to the projector, so
`@audio:note.wav @image:shot.png` reaches the model in that order and its
reverse in the other. This is not cosmetic: mtmd pairs media with markers
*positionally*, and a representation that kept images and audio in separate
lists would have to pick an order when recombining them — silently rewriting a
prompt whose sequence may be the point.

### Configuring a local model

Local multimodal needs a **projector** — llama.cpp's `mmproj-*.gguf`, published
beside the model in its GGUF repo. Name it with `mmprojPath`:

```toml
[llm]
modelPath  = "hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
mmprojPath = "hf:unsloth/gemma-4-E4B-it-GGUF/mmproj-BF16.gguf"
```

`MMPROJ_PATH` overrides it in the environment, like every other setting. Both
shapes `modelPath` accepts work here: an `hf:ORG/REPO[@REV]/file.gguf` spec that
downloads into the HF cache, or a local path (resolved against the config file's
own directory).

Absent `mmprojPath`, the backend is text-only however capable the model is — the
main model GGUF carries no multimodal weights at all.

The projector is loaded **eagerly**, at provider construction. A bad path or a
projector built for a different model fails at startup with the filename in the
message, rather than on the first turn that attaches something.

It is deliberately *not* guessed from `modelPath` by looking for a sibling
`mmproj-*.gguf`: a projector that silently mismatched its model produces garbage
embeddings rather than an error, and naming it is one line.

### From the app-server

`turn/start` input items may carry an image:

```json
{"type": "image", "imageUrl": "data:image/png;base64,…"}
```

Only base64 `data:` URLs are accepted. A remote `https://` URL would mean this
process fetching something a client chose, which is a different decision than
carrying bytes the client already had; such items are dropped, counted, and
logged so the client can tell.

**There is no audio item** — codex's protocol, which gallium's app-server
mirrors, defines `imageUrl` and nothing for sound. `@audio:` in the REPL is the
only way in today.

`turn/steer` **refuses** images rather than dropping them: `SteerInbox` carries a
`String`, and there is no media on that path. Attach to `turn/start`.

## Models and projectors

Every Gemma 4 handles text, image and audio. Qwen3.8 handles text and image —
its projector is vision-only, no audio encoder at all. GPT-OSS and LFM2.5 are
text-only and have no projector to fetch.

| Backend | Projector | Verified here |
|---|---|---|
| `gemma4` (E4B) | `mmproj-BF16.gguf`, 946 MB | ✅ image + audio |
| `gemma4-12b` | `mmproj-BF16.gguf`, 167 MB | ✅ image; audio garbles (below) |
| `gemma4-26b` | ships one | not tested — not cached locally |
| `qwen3.8` | `mmproj-F16.gguf`, 928 MB | ✅ image; no audio encoder, refuses cleanly |
| `gpt-oss` | none | text only |
| `lfm2` | none | text only |

### Encoder-based vs encoder-free, and why it shows

The two Gemma generations reach multimodality differently, and it is visible in
results rather than only in headers:

| | `gemma4` (E4B) | `gemma4-12b` |
|---|---|---|
| Design | [dedicated encoders](https://huggingface.co/google/gemma-4-E4B) | [encoder-free](https://huggingface.co/google/gemma-4-12B) |
| Projector tensors | 1411 (478M params) | 11 (52M params) |
| Vision | `block_count = 16` | `block_count = 0` |
| Audio | `block_count = 12`, `a.conv1d`, `a.pre_encode` | `block_count = 0` |
| Projector types | `gemma4v` / `gemma4a` | `gemma4uv` / `gemma4ua` |
| Download | **946 MB** | **167 MB** |
| Transcribes "The secret codeword is zucchini." | exactly | *"the secret code word is zuki"* |

`block_count = 0` on both modalities *is* the encoder-free design in the file
format: no transformer stack, just the linear layers that put image patches and
audio waveforms into the embedding space.

So **the smaller model has the larger projector, and it is the one that hears
properly.** E4B ships two transformer stacks; 12B ships linear layers. Vision is
fine on both — 12B reads the image test correctly.

Practical read: for audio, prefer a model with a real audio encoder. For vision,
either works, and encoder-free is a much smaller download.

## What media costs

Measured on `gemma4` (E4B) with an identical text prompt, varying only the
attachment. The baseline is high because the tool catalog dominates a gallium
prompt (~1770 tokens for ten built-in tools).

| Attachment | Prompt tokens | Delta |
|---|---|---|
| none | 1774 | — |
| `number.png`, 432×304 | 1831 | **+57** |
| `number.png` upscaled to 1536×1536 | 2033 | **+259** |
| `speech.wav`, ~4 s, 16 kHz mono | 1825 | **+51** |

Image cost scales with resolution, in the budgets Gemma 4 supports (70, 140,
280, 560, 1120 visual tokens). `MtmdContextParams` exposes `image_min_tokens` /
`image_max_tokens` to cap it; gallium leaves both at the model default (`-1`).

Two consequences worth planning for:

- The context window a multimodal turn needs is sized from what the projector
  actually produced, not from the prompt string's length — `generate_with_media`
  uses `chunks.total_tokens()`.
- Encoding is not free in time either. A ~4 s clip took **~2.5 s** to encode on
  an M-series Mac (`audio slice encoded in 2456 ms`), against 8 ms for the
  decode that follows.

## How it works

### Two prompt paths

`llm_local` branches on whether the turn carries media at all.

**No media** — exactly the path that has always existed: render the chat
template, tokenize, one `LlamaBatch`, decode. Enabling mtmd changes nothing here,
and no projector is touched.

**With media** — `stage_media` → `build_prompt` → `generate_with_media`:

1. `stage_media` rewrites each message that has attachments, prefixing its
   content with one `<__media__>` marker per attachment, and collects the decoded
   bytes **in the same pass**.
2. `build_prompt` renders those staged messages through the model's own jinja
   template as usual — the markers are ordinary text and survive it.
3. `MtmdBitmap::from_buffer` takes the raw file bytes; llama.cpp sniffs image vs
   audio from the magic bytes, so a PNG and a WAV are the same call.
4. `MtmdContext::tokenize` splits the prompt around its markers into chunks —
   text runs and media.
5. `eval_chunks` walks them in order, running the projector on media chunks and
   `llama_decode` on text ones, and returns the resulting `n_past`.

Both paths then share `sample_until_done`. Everything after the first logits is
identical, and the Gemma tool-boundary and UTF-8 handling should not be
maintained twice.

### Why one ordered list, and one pass

mtmd pairs markers with bitmaps **positionally**, so ordering is correctness,
not presentation. Two things follow.

`ChatMessage.media` and `UserInput.media` are a single `Vec<MediaContent>`
rather than an images vec beside an audio vec. With two lists, something
downstream must choose an order to recombine them, and whatever it chooses is a
silent rewrite of what the user wrote — an earlier revision concatenated images
before audio, which quietly moved every image ahead of every clip.

`stage_media` then emits markers and collects bytes in **one walk** of that
list, so the Nth marker and the Nth bitmap are the Nth attachment by
construction rather than by two loops agreeing. Markers are identical strings;
position is the only thing tying an attachment to its slot.

`input.rs` asserts this in both directions — audio-first and image-first — since
an implementation with separate vecs passes one and fails the other.

### Concurrency

The `MtmdContext` sits behind a `Mutex` because `eval_chunks` is documented as
not thread-safe, while a provider is shared across turns. Only multimodal turns
contend for it, and the lock is released as soon as the prompt is built — before
the far longer sampling stretch — so one turn can encode while another samples.

### Build

The `llama-cpp-2` `mtmd` feature is enabled unconditionally, in both per-target
dependency blocks. The cost is build time only (about a minute, plus stb_image
and miniaudio); an `MtmdContext` is created solely when `mmprojPath` names a
projector, so a text-only run allocates nothing and behaves exactly as before.

## Refusals

Every "no" names the missing piece, because the fixes differ.

| Message | Means | Fix |
|---|---|---|
| `the llama.cpp backend has no multimodal projector` | no `mmprojPath` configured | set it |
| `the configured projector has no vision encoder` | wrong projector for the job | check it matches the model |
| `the configured projector has no audio encoder` | vision-only projector | use a model whose projector has audio |
| `the OpenAI backend does not carry audio` | images are fine, sound is not | use llama.cpp for audio |
| `the candle backend cannot see images as built` | candle has no mtmd path at all | use the llama.cpp backend |

The rule behind all of them: an attachment nobody looked at produces a confident
answer about something the model never received, which is indistinguishable from
a model that cannot perceive. Refusing says which it was.

## Limits

- **No audio from tools.** `ToolContent` has `Text` and `Image` only, so a tool
  can return a screenshot but not a recording.
- **No audio over the app-server**, as above — the protocol defines no item.
- **No media in `turn/steer`**, which refuses rather than dropping.
- **The native candle backend has no multimodal path.**
  `gallium-models/src/gemma4_vision.rs` compiles and is exported, but nothing
  calls it.
- **Traces record the text of a turn, not its attachments.** A base64 payload
  would dwarf the rest of the file, and a trace replays as a script of tool
  calls, which no attachment participates in. A replayed multimodal turn is
  therefore missing input the recorded one had.
- **Compaction's *estimate* ignores media.** `memory::estimate_message_tokens`
  is `content.len() / 4 + 10` and does not look at attachments. In practice this
  is bounded: `compaction_target` prefers the provider's reported peak prompt,
  which *is* accurate — `generate_with_media` reports `chunks.total_tokens()`,
  media included — and uses the estimate only as a floor. So the under-count
  matters for the first turn of a thread, before any usage has been reported.
- **llama.cpp calls its audio front end experimental** ("audio input is in
  experimental stage and may have reduced quality"), which matches what the 12B
  transcription shows.

## Testing

`testsuite/testcases/multimodal_image` and `multimodal_audio`. See
[testsuite/README.md](../testsuite/README.md) for the harness.

```bash
bash testsuite/runner.sh multimodal_image gemma4
bash testsuite/runner.sh multimodal_audio gemma4
TESTS="multimodal_image,multimodal_audio" bash testsuite/matrix_runner.sh
```

Both testcases **forbid tool use and enforce it**: the fixture sits in the
agent's cwd, so a capable agent could decode the PNG or run something over the
WAV with Bash and answer correctly without perceiving anything. `check.sh` fails
the test if any tool ran.

Three fixture lessons are baked into those testcases and worth not re-learning:

1. **Ask for something unguessable.** The image asks for a two-digit number; a
   colour is a test a blind model passes one time in six.
2. **Ask for something the modality can actually do.** The audio fixture began
   as a pure tone with "what frequency is this?" — which *nothing* passes.
   Gemma 4 12B demonstrably heard a 613 Hz tone (the token count proves the
   audio was encoded) and answered "440" anyway. Pitch estimation is not what an
   audio LLM does; transcription is.
   - That 613 Hz was itself chosen after a 440 Hz version, which `gpt-5.6-luna`
     "passed" without ever receiving a sample — A4 is what anyone guesses from
     the word "tone".
3. **Wording can decide the outcome, so measure it.** "Transcribe the speech in
   this audio clip" makes E4B answer *"you have not provided one"* — 3 runs of
   3, with the audio demonstrably encoded into the prompt. Dropping the words
   "in this audio clip" transcribes correctly, also 3 of 3.

Fixtures are committed, so a test run needs no Python.
`uv run python testsuite/fixtures/make_fixtures.py` regenerates them; the image
is standard-library-only, the speech needs macOS `say`/`afconvert` (any 16 kHz
mono WAV of the phrase works elsewhere).

## Inspecting a projector

A projector's header says what it can do. There is no gallium command for this;
the GGUF metadata keys that matter are:

```
clip.has_vision_encoder     clip.has_audio_encoder
clip.vision.projector_type  clip.audio.projector_type
clip.vision.block_count     clip.audio.block_count     # 0 ⇒ encoder-free
clip.audio.num_mel_bins     clip.vision.image_size
```

gallium logs the answers at load, which is usually enough:

```
Multimodal projector: …/mmproj-BF16.gguf
Projector supports: vision=true, audio=true
Projector audio sample rate: 16000 Hz
```

## References

- [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- `llama_cpp_2::mtmd` — the Rust wrapper (`MtmdContext`, `MtmdBitmap`,
  `MtmdInputChunks`)
- `crates/gallium-agent/src/input.rs` — `UserInput`, marker parsing
- `crates/gallium-agent/src/llm_local.rs` — `stage_media`,
  `generate_with_media`, `refuse_unsupported_media`
- [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B) /
  [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B) model cards
