# `asr` — speech recognition

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

Transcribe (or translate) speech from an audio file. Speech recognition is
organized as **engines**, each selected as a subcommand with its own models and
flags. Three engines ship today, each documented on its own page:

- [**`asr whisper`**](whisper.md) — Whisper models: many
  languages with autodetection; can translate to English.
- [**`asr gigaam`**](gigaam.md) — GigaAM Conformer models
  (CTC and RNN-T): mainly for Russian; fast single-pass decoding.
- [**`asr vosk`**](vosk.md) — Vosk's Zipformer2 transducers
  (Alpha Cephei / k2-fsa): mainly for Russian, with offline and
  low-latency streaming models.

```sh
trakktor asr whisper talk.mp3            # JSON (default): text + timestamped segments
trakktor asr whisper talk.mp3 --text     # readable [start --> end] lines
trakktor asr whisper talk.mp3 --pretty   # indented JSON

trakktor asr gigaam ru.mp3               # GigaAM (Russian by default)
trakktor asr gigaam ru.mp3 --model multilingual_large_ctc --device metal

trakktor asr vosk ru.mp3                 # Vosk (large Russian by default)
trakktor asr vosk ru.mp3 --model small-streaming-ru --decoding greedy
```

What the engines share is documented once, right below: audio input, model
storage, device and precision, and the output shape and files. Each engine's
own models and flags live on its page.

## Audio input

The one required argument is the path to an audio file. It is decoded by a
**built-in decoder** — no ffmpeg or other external tool is needed — then
internally downmixed to mono and resampled to 16 kHz. Supported inputs: mp3,
aac (LC), vorbis, flac, alac, adpcm, and raw PCM, held in wav, aiff, caf, ogg,
mp4, or mkv containers.

```
--audio-decoder builtin|ffmpeg   # default: builtin
```

`builtin` (the default) is the pure-Rust decoder described above and needs no
external tools. `ffmpeg` instead shells out to an installed `ffmpeg`, which
decodes many more input formats — opus, HE-AAC, wma, amr, and others the
built-in decoder does not cover:

```sh
trakktor asr whisper voice.opus --audio-decoder ffmpeg
```

It requires `ffmpeg` on your `PATH`; without it the command fails with a clear
error. The default build never needs ffmpeg.

## Model storage

Models are downloaded **on first use** and reused on later runs; first-run
download progress is printed to stderr.

A checkpoint is gigabytes, so the download is built to survive the trip. It
runs over four parallel connections — measurably about three times faster than
one — and **resumes** if it is interrupted: a run that dies at 90 % costs the
remaining 10 % next time, not the whole file again. Transient failures (a
dropped connection, a stalled transfer, a 5xx) are retried, and a file only
gets its final name once it is complete and matches its published checksum, so
an interrupted download can never be mistaken for a usable model. Interrupt one
with `Ctrl-C` and start it again to see it pick up where it stopped.

The weights are large and **shared across projects**: they live under the
**model directory** — `~/.trakktor` by default — *not* in the per-project
`--work-dir`, each engine under its own path (`asr/whisper/<name>/`,
`asr/gigaam/<name>.ckpt`, `asr/vosk/<name>/`). Override the root with
`--model-dir <path>` or `TRAKKTOR_MODEL_DIR` (precedence: flag > env >
`~/.trakktor`); to reuse weights already downloaded elsewhere, point it at
that root — e.g. `TRAKKTOR_MODEL_DIR=/data/models` looks for
`/data/models/asr/whisper/<name>/`.

Which models exist, and what `--model` accepts, is each engine's own — see its
page.

## Device and precision

```
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

- **`--device metal`** runs on the macOS GPU and is several times faster than
  the CPU on a typical clip. It needs a build with the `metal` feature:

  ```sh
  cargo install --locked --git https://github.com/lymar/trakktor.git \
    --features metal trakktor
  ```

  Without that feature, `--device metal` is rejected. (The alternative burn
  runtime is the exception — it ships its own Metal backend; see below.)

- **`--precision f16`** (the default) uses about half the memory and is faster;
  **`--precision f32`** computes in full precision for reproducible results, at
  twice the weight memory. f16 is what keeps the large models within reach on a
  16 GB machine.

## Runtime

```
--runtime candle|burn   # default: candle
```

All three engines execute their network on the [candle](https://github.com/huggingface/candle)
runtime by default (as do [`text structify`](../text/README.md) and
[`text punctuate`](../text/README.md), and both TTS engines,
[`tts qwen3-tts`](../tts/qwen3-tts.md) and [`tts espeech`](../tts/espeech.md)).
An alternative [burn](https://github.com/tracel-ai/burn)
runtime is available behind the `burn` build feature and selected per run with
`--runtime burn` (candle needs nothing extra; burn brings its own Metal
backend, independent of the `metal` feature). What to know about it:

- **Parity.** Both runtimes produce the same transcription — byte-identical
  in f32 on our test material; f16 runs may swap the odd word, as with any
  half-precision kernel change (and since Whisper conditions each window on
  the previous text, on a long recording one swapped word can ripple into
  locally different, equally valid phrasing downstream).
- **Speed on Metal.** The two are close: burn measured moderately faster than
  candle for Whisper `large-v3` at a lower memory peak, and on par for GigaAM
  and Vosk.
- **A known candle defect.** candle degrades on Metal with `--precision f32`
  on Whisper `large-v3` — for full precision on the GPU use `--runtime burn`,
  which handles it correctly.
- **First-run autotuning.** The very first burn run on a machine is a few
  times slower while it autotunes its GPU kernels; trakktor announces this on
  stderr when it is about to happen, the tuning result is cached, and later
  runs are full speed.
- **The burn CPU backend** computes in f32 only, so combine `--runtime burn`
  on the CPU with `--precision f32` — and expect it to be several times
  slower than candle there (its CPU matmuls do not parallelize the way
  candle's do); the burn runtime is aimed at Metal.

## Timestamps and output shape

```
--timestamps none|segment|word   # default: segment
```

- `segment` — per-segment start/end times (the default).
- `word` — segment times plus per-word timings. Whisper derives them with an
  extra alignment pass (slower); GigaAM and Vosk read them off their emission
  frames at no extra cost.
- `none` — text only, no segment list.

The default output is a single JSON object: the transcript `text`, the
`language` (whisper detects it when omitted; gigaam and vosk only report one
when `--language` is given), the audio `duration` (seconds), the `engine`, and —
unless `--timestamps none` — a `segments` array. Each segment carries its
`id`, `start`, `end`, `text`, and (with `--timestamps word`) a `words` array;
whisper segments also carry decoder quality signals under `whisper`. A whisper
result:

```json
{
  "text": "Ask not what your country can do for you.",
  "language": "en",
  "duration": 11.0,
  "engine": { "name": "whisper", "model": "tiny" },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.6,
      "text": "Ask not what your country can do for you.",
      "whisper": {
        "avg_logprob": -0.29,
        "compression_ratio": 1.15,
        "no_speech_prob": 0.01,
        "temperature": 0.0
      }
    }
  ]
}
```

With `--timestamps word`, each segment also gets a `words` array of
`{ start, end, word, probability }` (GigaAM and Vosk words have no
`probability`):

```json
"words": [
  { "start": 0.0, "end": 0.42, "word": "Ask", "probability": 0.98 }
]
```

The `whisper` block is diagnostic: `avg_logprob` (average token
log-probability — confidence), `compression_ratio` (zlib ratio; high means
repetitive), `no_speech_prob`, and the `temperature` the accepted result was
decoded at. The GigaAM and Vosk decoders have no counterpart signals, so their
segments carry no such block. `--text` instead prints one right-aligned
`[start --> end] text` line per segment (or just the transcript with
`--timestamps none`), and errors follow the usual
`{ "error": { "code", "message" } }` contract.

While a long file decodes, a live progress line — audio position, percent,
elapsed, and a rough estimate of the time remaining — is written to **stderr**,
so stdout stays a clean JSON (or text) stream.

## Output files

By default the result goes to stdout as JSON (or text with `--text`).
`--output-format` **additionally** writes the transcript to files, without
changing what stdout prints:

```
--output-format txt|vtt|srt|tsv|json|all   # which file(s) to write
--output-dir <dir>                         # where (default: the current dir)
```

| Format | File | Contents |
|---|---|---|
| `txt` | `<audio>.txt` | one line per segment |
| `vtt` | `<audio>.vtt` | WebVTT subtitles |
| `srt` | `<audio>.srt` | SubRip (SRT) subtitles |
| `tsv` | `<audio>.tsv` | `start`⇥`end`⇥`text`, times in integer milliseconds |
| `json` | `<audio>.json` | the full JSON result (same as stdout) |
| `all` | — | every format above |

Each file is named after the audio (`talk.mp3` → `talk.srt`) and written into
`--output-dir` (created if it does not exist); the paths written are reported on
stderr. Subtitle cues are one per segment.

```sh
# Write SubRip subtitles (stdout still prints the JSON result)
trakktor asr whisper talk.mp3 --output-format srt

# Write every format into subs/
trakktor asr whisper talk.mp3 --output-format all --output-dir subs
```
