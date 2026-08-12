# `asr gigaam` — GigaAM engine

> One of the engines of [trakktor's `asr` command](README.md); the behavior
> shared by all engines — audio input, model storage, device and precision,
> runtime, timestamps, output files — is documented there.

[GigaAM](https://github.com/salute-developers/GigaAM) is a family of Conformer
acoustic models with CTC or RNN-T (transducer) decoding — mainly for Russian,
and, with the multilingual checkpoints, several more languages. It is
a faithful port of the reference pipeline on the same candle runtime as
[`whisper`](whisper.md), and all the [shared behavior](README.md) — audio
input, model storage, device and precision, output — applies as-is. The engine
has no decoding knobs, no clipping, and no `--vad` flag: greedy decoding has
nothing to tune, and speech detection is already built into its chunking
(below). GigaAM does not detect the language; `--language <code>` only
annotates the output.

The shared `--runtime` flag applies too: `--runtime burn` runs the same
network on the alternative burn runtime (see the
[Runtime](README.md#runtime) section).

```sh
trakktor asr gigaam ru.mp3                       # Russian, punctuated (v3_e2e_rnnt), CPU, JSON
trakktor asr gigaam ru.mp3 --text                # readable [start --> end] lines
trakktor asr gigaam ru.mp3 --timestamps word     # per-word timings
trakktor asr gigaam ru.mp3 \
  --model multilingual_large_ctc --device metal   # largest model on GPU (f16 by default)
```

Models (downloaded on first use into `~/.trakktor/asr/gigaam/<name>.ckpt`, or
pass a path to a local `.ckpt`):

- **`v3_e2e_rnnt`** (default) — Russian **with punctuation and
  capitalization** (end-to-end model with text normalization — numbers as
  digits, sentence casing), decoded by an RNN-T (transducer) head. The more
  accurate of the two punctuated models.
- **`v3_e2e_ctc`** — the same punctuated output from a CTC decoder: it ends
  sentences more often, at some cost in word accuracy. On a GPU the two run
  equally fast (the transducer's decode loop rides behind the next chunk's
  encode); on CPU-only runs this is the lighter choice. Worth the swap when
  sentence structure matters more than wording.
- **`v3_ctc`** — Russian; normalized lowercase text without punctuation.
- **`v3_rnnt`** — Russian; like `v3_ctc` but with an RNN-T (transducer)
  decoder — usually the most accurate raw text on Russian, slightly slower.
- **`multilingual_ctc`** — multiple languages.
- **`multilingual_large_ctc`** — the largest, most accurate multilingual
  model.

The `v3_e2e_*` models emit readable, punctuated Russian. The plain `v3_ctc`
and `v3_rnnt` models emit normalized lowercase text without punctuation
(their alphabet has none) — the usual form for downstream text processing or
word-error-rate (WER) evaluation; pick `v3_rnnt` when raw-text accuracy
matters most:

```sh
trakktor asr gigaam ru.mp3 --model v3_rnnt --device metal --text
```

Unlike Whisper's autoregressive decoder, GigaAM decoding is a single encoder
pass per chunk followed by one read-back — the CTC models take an argmax over
frames, and `v3_rnnt` runs its small transducer loop on the CPU from the
encoder output — so the GPU stays busy through a chunk either way. Word
timings with `--timestamps word` are read off those same emission frames, at
no extra cost. Audio longer than ~25 seconds is split along detected speech
(voice-activity detection) into chunks, each transcribed independently and
stitched back onto the original timeline. Chunk boundaries are placed by
dynamic programming over the pauses between speech — the longer a pause, the
likelier the cut lands there — so segments tend to fall on sentence boundaries
while staying near the target length.

The whole pipeline is **streaming**: the file is decoded block by block,
speech detection runs incrementally, and each chunk is transcribed — and its
audio released — as soon as its boundaries are settled. Memory stays bounded
by a few minutes of audio regardless of the recording length, so multi-hour
files transcribe in constant memory.
