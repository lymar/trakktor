# trakktor

`trakktor` is a predictable, automation-friendly Rust CLI toolbox for coding
agents (Claude Code, OpenCode, etc.): speech-to-text and text-to-speech,
voice-activity audio editing, feeds, text structuring, and more —
machine-readable output, stable flags, and meaningful exit codes.

## Install

Requires a stable Rust toolchain **≥ 1.88** (edition 2024; nightly is only
needed for `cargo fmt`, not to build). The binary installs into `~/.cargo/bin`,
which must be on your `PATH`.

From git:

```sh
cargo install --locked --git https://github.com/lymar/trakktor.git trakktor
```

- `trakktor` (the trailing word) is the package to install — the workspace root
  is virtual, so it must be named.
- `--locked` builds with the exact dependency versions pinned in `Cargo.lock`
  (reproducible); omit it to pull the latest semver-compatible versions.

From a local clone (installs whatever is checked out):

```sh
git clone https://github.com/lymar/trakktor.git
cd trakktor
cargo install --locked --path trakktor
```

For GPU acceleration on macOS (speech recognition and text structuring), add
`--features metal` to either install command; see the `asr` section below.

Verify, then remove if needed:

```sh
trakktor --version
cargo uninstall trakktor
```

On Linux, the build pulls in `reqwest`'s default TLS, which needs OpenSSL
(`pkg-config` plus `libssl-dev`/`openssl-devel`); macOS uses the system TLS and
needs nothing extra.

## Layout

A Cargo workspace with a flat crate layout:

- `trakktor` — the CLI binary: argument parsing, configuration, output
  formatting. A thin layer over the library.
- `trakktor_core` — the library: all functionality, independent of the CLI.

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

Formatting uses unstable rustfmt features, so it requires nightly:

```sh
cargo +nightly fmt --all
```

## Output and exit codes

The default output is machine-readable JSON (`--pretty` indents it); `--text`
switches to human-readable text. Data is written to stdout, errors to stderr.

- `0` — success.
- `1` — runtime/validation error (network, parsing, I/O, bad value). Formatted
  like normal output: JSON `{ "error": { "code": "…", "message": "…" } }` by
  default, plain text with `--text`.
- `2` — usage error (unknown flag, missing argument); always text, from the
  argument parser.

Global options (usable before or after the command): `--work-dir <path>` (also
`TRAKKTOR_DIR`; default `./.trakktor`, per-project state such as feed
read-state), `--model-dir <path>` (also `TRAKKTOR_MODEL_DIR`; default
`~/.trakktor`, where model weights are cached — see `asr`), `--text`,
`--pretty`.

## `asr` — speech recognition

Transcribe (or translate) speech from an audio file. Speech recognition is
organized as **engines**, each selected as a subcommand with its own models and
flags. Three engines ship today, each documented in its own section below:

- [**`asr whisper`**](#asr-whisper--whisper-engine) — Whisper models: many
  languages with autodetection; can translate to English.
- [**`asr gigaam`**](#asr-gigaam--gigaam-engine) — GigaAM Conformer models
  (CTC and RNN-T): mainly for Russian; fast single-pass decoding.
- [**`asr vosk`**](#asr-vosk--vosk-engine) — Vosk's Zipformer2 transducers
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
own models and flags live in its section.

### Audio input

The one required argument is the path to an audio file. It is decoded by a
**built-in decoder** — no ffmpeg or other external tool is needed — then
internally downmixed to mono and resampled to 16 kHz. Supported inputs: mp3,
aac (LC), vorbis, flac, alac, and raw PCM, held in wav, aiff, caf, ogg, mp4, or
mkv containers.

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

### Model storage

Models are downloaded **on first use** and reused on later runs; first-run
download progress is printed to stderr. The weights are large and **shared
across projects**: they live under the **model directory** — `~/.trakktor` by
default — *not* in the per-project `--work-dir`, each engine under its own
path (`asr/whisper/<name>/`, `asr/gigaam/<name>.ckpt`,
`asr/vosk/<name>/`). Override the root with
`--model-dir <path>` or `TRAKKTOR_MODEL_DIR` (precedence: flag > env >
`~/.trakktor`); to reuse weights already downloaded elsewhere, point it at
that root — e.g. `TRAKKTOR_MODEL_DIR=/data/models` looks for
`/data/models/asr/whisper/<name>/`.

Which models exist, and what `--model` accepts, is each engine's own — see its
section.

### Device and precision

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

### Runtime

```
--runtime candle|burn   # default: candle
```

All three engines execute their network on the [candle](https://github.com/huggingface/candle)
runtime by default (as do `text structify`, `text punctuate`, and
`tts qwen3-tts`). An alternative [burn](https://github.com/tracel-ai/burn)
runtime is available behind the `burn` build feature and selected per run with
`--runtime burn` (candle needs nothing extra; burn brings its own Metal
backend, independent of the `metal` feature). Both runtimes produce the same
transcription — byte-identical in f32 on our test material; f16 runs may swap
the odd word, as with any half-precision kernel change (and since Whisper
conditions each window on the previous text, on a long recording one swapped
word can ripple into locally different, equally valid phrasing downstream).
On Metal the two runtimes are close: burn measured moderately faster than
candle for Whisper `large-v3` at a lower memory peak, and on par for GigaAM
and Vosk.
(One known exception: candle degrades on Metal with `--precision f32` on
Whisper `large-v3` — for full precision on the GPU use `--runtime burn`,
which handles it correctly.) The very first burn run on a machine is a few
times slower while it autotunes its GPU kernels; trakktor announces this on
stderr when it is about to happen, the tuning result is cached, and later
runs are full speed. The burn CPU backend computes in f32 only, so
combine `--runtime burn` on the CPU with `--precision f32` — and expect it to
be several times slower than candle there (its CPU matmuls do not
parallelize the way candle's do); the burn runtime is aimed at Metal.

### Timestamps and output shape

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

### Output files

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

### `asr whisper` — Whisper engine

[Whisper](https://github.com/openai/whisper) is the most versatile engine:
many languages, language autodetection, and optional translation into English.
Transcription runs window by window (30 seconds each) with a
temperature-fallback policy that guards against repetition loops, closely
following Whisper's own decoding behavior.

#### Models

```
--model <name|dir>      # default: tiny
```

Pass a **published name** — downloaded on first use into the model directory
(`~/.trakktor/asr/whisper/<name>/` by default; see
[Model storage](#model-storage)) — or a **path** to a local checkpoint
directory (one containing `config.json`).

Names, smallest to largest (larger is slower but more accurate):

- **Multilingual:** `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`,
  `large-v3` (alias `large`), `large-v3-turbo` (alias `turbo`).
- **English-only:** `tiny.en`, `base.en`, `small.en`, `medium.en` — slightly
  better on English audio.

#### Language and task

```
--language <code|name>        # e.g. en, ru, or russian; autodetected if omitted
--task transcribe|translate   # default: transcribe
```

With no `--language`, the language is detected from the first 30 seconds.
`--task translate` renders the speech as English instead of transcribing it in
the source language.

#### Clipping: transcribe only part of the audio

```
--start <time>      # begin at this offset (default: the beginning)
--end <time>        # stop at this offset (default: the end)
```

`--start`/`--end` limit transcription to a time range. Each accepts a plain
number of seconds or a `[[HH:]MM:]SS[.mmm]` clock, to millisecond precision, and
either flag works on its own:

```sh
trakktor asr whisper talk.mp3 --start 0:15 --end 5:30   # from 0:15 to 5:30
trakktor asr whisper talk.mp3 --start 90                # from 1:30 to the end
trakktor asr whisper talk.mp3 --end 1:02:03.250         # from the start to 1:02:03.250
```

They are a convenience over `--clip-timestamps` and cannot be combined with it;
use `--clip-timestamps` directly to transcribe several ranges at once.

#### Voice-activity detection (VAD)

`--vad` runs Silero voice-activity detection first and transcribes only the
speech, dropping silence, music, and noise. It is the direct remedy for
Whisper's tendency to hallucinate and loop over long non-speech stretches, and
it is faster on sparse audio. Off by default; the model is built in (nothing to
download) and runs on the CPU, so `--device metal` still accelerates the
transcription itself.

```sh
trakktor asr whisper interview.mp3 --vad
```

The detected speech is glued into one dense buffer, transcribed in a single
pass, and the timestamps are then mapped back to the original timeline — so the
output stays on the source clock while non-speech is never sent to the model.

Detection is tunable (shown with the reference defaults):

```
--vad-threshold 0.5                 # speech-probability cutoff (0..=1)
--vad-min-speech-duration-ms 250    # drop shorter detections
--vad-min-silence-duration-ms 100   # a shorter pause won't split a segment
--vad-speech-pad-ms 30              # padding kept around each speech span
--vad-max-speech-duration-s none    # force-split longer speech (none = never)
```

`--vad` combines with `--start`/`--end` (detect speech within that range) but
not with `--clip-timestamps`. When VAD is active the JSON result gains a
top-level `vad` block listing the detected speech spans (original timeline):

```json
"vad": { "speech": [ { "start": 0.48, "end": 12.3 } ] }
```

#### Decoding controls

The decoding defaults mirror the reference behavior and rarely need touching;
every flag below has a sensible default. Run `trakktor asr whisper --help` for
the complete list with defaults and exact value formats. In brief:

- **Temperature fallback** — `--temperature`,
  `--temperature-increment-on-fallback`: the schedule the decoder climbs when a
  window looks like a failure.
- **Sampling and search** — `--best-of` (trajectories at non-zero temperature),
  `--beam-size` (beam width at temperature 0), `--patience`, `--length-penalty`.
- **Failure gates** (each triggers a hotter retry) —
  `--compression-ratio-threshold` (repetition), `--logprob-threshold`
  (confidence), `--no-speech-threshold` (silence).
- **Prompting** — `--initial-prompt` (bias the first window toward domain
  vocabulary or proper nouns), `--carry-initial-prompt`,
  `--condition-on-previous-text`.
- **Token suppression** — `--suppress-tokens` (`-1` expands to a built-in
  non-speech set).
- **Word-timestamp tuning** (with `--timestamps word`) —
  `--prepend-punctuations`, `--append-punctuations`,
  `--hallucination-silence-threshold`.
- **Partial audio** — `--clip-timestamps` to transcribe only selected
  `start,end` second ranges. For a single range, the `--start`/`--end` flags
  above are usually easier.
- **Voice-activity detection** — `--vad` (with the `--vad-*` tuning flags)
  detects speech and skips non-speech before transcribing; see the VAD section
  above.

#### Examples

```sh
# Russian interview, best model on the GPU, word-level timings, indented JSON
trakktor asr whisper interview.m4a \
  --model large-v3 --device metal --language ru \
  --timestamps word --pretty

# Translate a lecture to English, readable text output
trakktor asr whisper lecture.mp3 --model medium --task translate --text

# Bias the first window with domain terms
trakktor asr whisper standup.wav \
  --initial-prompt "Kubernetes, Grafana, Prometheus, sharding"

# Subtitle one section: transcribe 0:15–5:30 and write every format into subs/
trakktor asr whisper talk.mp3 --start 0:15 --end 5:30 \
  --output-format all --output-dir subs

# Decode a format the built-in decoder does not cover
trakktor asr whisper voice.opus --audio-decoder ffmpeg --model small
```

### `asr gigaam` — GigaAM engine

[GigaAM](https://github.com/salute-developers/GigaAM) is a family of Conformer
acoustic models with CTC or RNN-T (transducer) decoding — mainly for Russian,
and, with the multilingual checkpoints, several more languages. It is
a faithful port of the reference pipeline on the same candle runtime as
`whisper`, and all the shared behavior above — audio input, model storage,
device and precision, output — applies as-is. The engine has no decoding
knobs, no clipping, and no `--vad` flag: greedy decoding has nothing to tune,
and speech detection is already built into its chunking (below). GigaAM does
not detect the language; `--language <code>` only annotates the output.

The shared `--runtime` flag applies too: `--runtime burn` runs the same
network on the alternative burn runtime (see the Runtime section above).

```sh
trakktor asr gigaam ru.mp3                       # Russian, punctuated (v3_e2e_ctc), CPU, JSON
trakktor asr gigaam ru.mp3 --text                # readable [start --> end] lines
trakktor asr gigaam ru.mp3 --timestamps word     # per-word timings
trakktor asr gigaam ru.mp3 \
  --model multilingual_large_ctc --device metal   # largest model on GPU (f16 by default)
```

Models (downloaded on first use into `~/.trakktor/asr/gigaam/<name>.ckpt`, or
pass a path to a local `.ckpt`):

- **`v3_e2e_ctc`** (default) — Russian **with punctuation and capitalization**
  (end-to-end model with text normalization — numbers as digits, sentence
  casing).
- **`v3_e2e_rnnt`** — punctuated Russian like `v3_e2e_ctc`, with an RNN-T
  (transducer) decoder; slightly slower.
- **`v3_ctc`** — Russian; normalized lowercase text without punctuation.
- **`v3_rnnt`** — Russian; like `v3_ctc` but with an RNN-T (transducer)
  decoder — usually the most accurate raw text on Russian, slightly slower.
- **`multilingual_ctc`** — multiple languages (220M).
- **`multilingual_large_ctc`** — the largest, most accurate multilingual model
  (600M).

The `v3_e2e_*` models emit readable, punctuated Russian. The plain `v3_ctc`
and `v3_rnnt` models emit normalized lowercase text without punctuation
(their alphabet has none) — the usual form for downstream text processing or
WER evaluation; pick `v3_rnnt` when raw-text accuracy matters most:

```sh
trakktor asr gigaam ru.mp3 --model v3_rnnt --device metal --text
```

Unlike Whisper's autoregressive decoder, GigaAM decoding is a single encoder
pass per chunk followed by one read-back — the CTC models take an argmax over
frames, and `v3_rnnt` runs its small transducer loop on the CPU from the
encoder output — so the GPU stays busy through a chunk either way. Audio
longer than ~25 seconds is
split along detected speech (voice-activity detection) into chunks, each
transcribed independently and stitched back onto the original timeline. Chunk
boundaries are placed by dynamic programming over the pauses between speech —
the longer a pause, the likelier the cut lands there — so segments tend to fall
on sentence boundaries while staying near the target length.

The whole pipeline is **streaming**: the file is decoded block by block,
speech detection runs incrementally, and each chunk is transcribed — and its
audio released — as soon as its boundaries are settled. Memory stays bounded
by a few minutes of audio regardless of the recording length, so multi-hour
files transcribe in constant memory.

### `asr vosk` — Vosk engine

[Vosk](https://alphacephei.com/vosk/)'s current model line (Alpha Cephei) is
a family of **Zipformer2 RNN-T transducers** trained with
[k2-fsa/icefall](https://github.com/k2-fsa/icefall) — mainly for Russian,
with more languages available. trakktor is a native port of the reference
pipeline (feature extraction, encoder, and transducer decoding follow the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) runtime), on the same
candle runtime as the other engines, and all the shared behavior above —
audio input, model storage, device and precision, output — applies as-is. The
models emit **lowercase text without punctuation** (the usual form for
downstream processing or WER evaluation). Vosk does not detect the language;
`--language <code>` only annotates the output. The shared `--runtime burn`
runs the same network on the alternative burn runtime.

```sh
trakktor asr vosk ru.mp3                           # large Russian, offline, JSON
trakktor asr vosk ru.mp3 --text                    # readable [start --> end] lines
trakktor asr vosk ru.mp3 --decoding greedy         # faster search, slightly less accurate
trakktor asr vosk ru.mp3 --model small-streaming-ru --device metal
```

Models (downloaded on first use into `~/.trakktor/asr/vosk/<name>/`, or pass a
path to a directory holding a compatible export — `encoder.onnx`,
`decoder.onnx`, `joiner.onnx`, `tokens.txt`). The size is the total download
(the fp32 encoder dominates); "large" models are ~264 MB, "small" ones ~94 MB:

| Model | Language | Type | Size |
|---|---|---|---|
| `ru` (default) | Russian | offline (full-context) | ~264 MB |
| `small-ru` | Russian | offline | ~93 MB |
| `streaming-ru` | Russian | streaming (low-latency chunked) | ~264 MB |
| `small-streaming-ru` | Russian | streaming | ~94 MB |
| `small-streaming-bn` | Bengali | streaming | ~94 MB |
| `tg` | Tajik | offline | ~264 MB |

**Languages.** The catalog covers **Russian, Bengali, and Tajik** — the models
Alpha Cephei ships in its portable Zipformer2 line so far. Vosk's other
languages are still legacy Kaldi models, which this native port does not cover.
`--model <dir>` additionally loads any compatible icefall Zipformer2 transducer
export (the same four files) from a local directory.

Two searches are available with `--decoding`: **`beam`** (the default,
modified beam search — the reference's method) and **`greedy`** (one token per
frame — faster, usually slightly less accurate). The transducer decoder runs
on the CPU from the encoder output, so a chunk is one encoder pass plus one
read-back on the GPU, like GigaAM.

**Offline** models transcribe audio up to ~25 seconds directly; longer audio
is split along detected speech into chunks (the same voice-activity
segmentation as GigaAM) and stitched back onto the original timeline. The
whole pipeline is streaming and bounded in memory regardless of file length.
**Streaming** models instead run a native chunked encoder with cached state —
low latency and low memory, the basis for future live transcription; on files,
the offline models are more accurate.

## `tts` — speech synthesis

`trakktor tts` is the counterpart to `asr`: text in, an audio file out. Like
`asr`, it is organized as a set of engines, each picked as a subcommand, and
runs entirely locally.

```bash
trakktor tts qwen3-tts "Привет! Это синтез речи." --language russian -o hello.wav

# a whole article — plain text or Markdown, any length
trakktor tts qwen3-tts --text-file article.md --language russian -o article.mp3

# or from a pipe, saying how to read it
cat notes.md | trakktor tts qwen3-tts --text-file - --text-format md -o notes.wav
```

The text comes as the positional argument, from `--text-file`, or from standard
input (`--text-file -`) — exactly one of them.

**Long text is spoken whole.** One utterance is capped at two minutes of speech
(the checkpoints would allow more, but a single derailed generation costs
minutes and the model was trained on utterances, not chapters), so the text is
split into paragraphs, each is spoken separately, and the pieces are joined
into one file:

- `--text-format <auto|txt|md>` says where paragraphs end. `txt` takes one
  paragraph per line; `md` separates them with blank lines and strips the
  markup — headings, list and quote markers, emphasis, inline code, links (the
  link text is kept, the URL is not), front matter, HTML comments. Tables and
  code blocks keep their **text**: removing markup is not the same as deciding
  what you did not want to hear. `auto` (the default) reads the file extension,
  then the text itself, and falls back to `md`.
- A paragraph still too long for one utterance is split further with
  [`text structify`](#text-structify--split-text-into-paragraphs), which is
  already a boundary model — it picks the coarsest cut that fits, so the text
  breaks as few times as possible. That model is downloaded and loaded **only**
  if some paragraph actually needs it.
- `--pause-ms <ms>` (default 500) sets the gap between paragraphs. Each piece
  is trimmed of the silence the model leaves at its edges and faded at the
  join, so the pause is exactly what you asked for and the seams do not click;
  pieces of one split paragraph get half the gap.
- Every piece is sampled from a seed derived from `--seed`, so the same run
  reproduces the same file.

The audio is always written to a file — raw samples on stdout would not survive
the machine-readable contract — and `stdout` carries the metadata:

```json
{
  "output": "hello.wav",
  "format": "wav",
  "sample_rate": 24000,
  "duration": 2.16,
  "chunks": 1,
  "language": "russian",
  "voice": { "kind": "preset", "name": "serena" },
  "engine": { "name": "qwen3-tts", "model": "0.6b-customvoice", "runtime": "candle" },
  "qwen3-tts": { "frames": 27, "sampling": "top_k", "seed": 0, "top_k": 50, "temperature": 0.9 }
}
```

`chunks` is how many pieces were spoken and stitched together, and `frames`
their total. Should a piece still run into the two-minute cap, it is cut down
and spoken again (twice at most); if even that does not help, the engine block
carries `"truncated": true` and the tail of that piece is missing.

The container follows the `--output` extension: `.wav` (32-bit float, exactly
as synthesized) and `.flac` (quantized to 24 bits) are written by the built-in
encoder, and anything else — mp3, m4a, opus, ogg — is handed to an installed
`ffmpeg` (`--audio-encoder` forces the choice, `--bitrate` sets `-b:a` for
lossy formats).

### `tts qwen3-tts` — Qwen3-TTS engine

A native port of the open **Qwen3-TTS 12 Hz** family. Three stages run in
sequence: a Qwen3 decoder ("talker") reads the text and predicts the first
codebook of every 12.5 Hz frame; a small code predictor fills that frame's
remaining 15 residual codebooks; and a causal convolutional codec decoder turns
the finished frames into a 24 kHz waveform.

| Flag | Default | Meaning |
|---|---|---|
| `--voice <name>` | `serena` | Preset timbre: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. |
| `--language <lang>` | `auto` | Target language, set independently of the voice: `russian`, `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `korean`, `italian`, `portuguese`. |
| `--model <name\|dir>` | `0.6b-customvoice` | `0.6b-customvoice` or `1.7b-customvoice` (larger, slower), or a checkpoint directory. |
| `--text-file <path\|->` | — | Read the text from a file, or from standard input with `-`. |
| `--text-format <auto\|txt\|md>` | `auto` | Where paragraphs end and whether markup is stripped. |
| `--pause-ms <ms>` | `500` | Gap between paragraphs (half that inside a split paragraph). |
| `--audio-encoder <auto\|builtin\|ffmpeg>` | `auto` | Who writes the file: the built-in wav/flac encoder, or ffmpeg for everything else. |
| `--bitrate <rate>` | — | `-b:a` for lossy ffmpeg formats, e.g. `192k`. |
| `--seed <int>` | `0` | Makes a sampled run repeatable. |
| `--temperature`, `--top-k`, `--repetition-penalty` | `0.9`, `50`, `1.05` | Sampling controls. |
| `--greedy` | off | Take the most likely code instead of sampling — deterministic, usually flatter. |
| `--precision <bf16\|f32>` | `bf16` on candle, `f32` on burn | **Runtime-dependent default.** `bf16` is the format the weights are stored in and what the reference runs — and what keeps `1.7b-customvoice` within a 16 GB machine on Metal; `f32` doubles the memory and is reproducible, and is the *only* precision the burn runtime serves. The codec always runs in full precision either way. |
| `--runtime <candle\|burn>` | `candle` | Inference runtime; burn needs the `burn` build feature and computes in f32 only. |
| `--device <cpu\|metal>` | `cpu` | Compute device; `metal` needs the `metal` build feature (burn brings its own). |

Any voice can speak any supported language: the language is a separate
conditioning token, not a property of the timbre. Russian is supported
first-class.

Generation **samples** by default, so two runs of the same text differ slightly;
`--seed` pins a run, and `--greedy` removes the randomness altogether. The
codec decoder is deterministic either way — with the frames fixed, it
reproduces the same waveform every time.

The first use downloads the checkpoint (about 2.5 GB for `0.6b-customvoice`,
4.5 GB for `1.7b-customvoice`, codec included) into the model directory; later
runs reuse it. `--device metal` is considerably faster than the CPU and is the
recommended way to run it — the CPU path is impractically slow for anything
past a short phrase.

`1.7b-customvoice` needs roughly 4.5 GB of memory in `bf16` and about twice
that in `f32`; on a 16 GB machine only `bf16` is practical for it on candle,
which is why `bf16` is the default there — including on Metal, where it is what
keeps the large model in memory. (burn is the exception: it runs the same model
in `f32` on Metal within 16 GB; see Runtime below.)

#### Runtime

The alternative [burn](https://github.com/tracel-ai/burn) runtime (see the
Runtime section under `asr`) runs this engine too, with one restriction: **it
computes in f32 only**. You need not pass `--precision` for it — the default is
runtime-dependent, `f32` whenever `--runtime burn` is selected and `bf16` on
candle. Half precision is unavailable on burn from below: its Metal backend
cannot compile `bf16` kernels, and `f16` is excluded by the model itself — so
`--precision bf16 --runtime burn` is a validation error rather than a silent
downgrade.

Both runtimes pick the same codes in `--greedy --precision f32`, and the
waveforms are indistinguishable (cosine 1.0000000000); burn additionally
reproduces byte-for-byte between its own runs. On speed the trade is the
opposite of the encoders': generation is one frame at a time over many tiny
passes, which suits candle's lower per-operation overhead, so burn is around
1.25× slower per frame and slower to load. What burn can do and candle cannot
is run `1.7b-customvoice` in **full precision on Metal** within 16 GB — for
that, `--runtime burn --precision f32`.

For **long text, candle is the runtime to use.** burn's kernels are compiled
and autotuned per shape, and the first pass over a shape it has not seen pays
for that — a page of text runs a few times slower than on candle, most of the
gap on the first run of a given machine. (The decoder used to make this worse:
its last chunk was whatever frames were left over, so nearly every paragraph
introduced a new shape and stalled for tens of seconds. It now decodes at a
rounded-up length and drops the extra samples — the network is causal, so the
waveform is unchanged to within 1e-6.)

## `vad` — voice-activity audio editing

`trakktor vad` finds the speech in an audio file with Silero voice-activity
detection and acts on it — report where the speech is, cut the silence out, or
split a recording into clips. It stands beside `asr` (this is editing, not
transcription): the same detector, used to reshape audio rather than feed a
transcription engine.

The key idea: **detection runs on 16 kHz mono (what Silero needs), but cutting
runs on the decoded original at full quality** — its own sample rate, channels,
and bit depth. Nothing is downsampled to the detector's 16 kHz; the speech times
are mapped back to the original samples, and only a few milliseconds of fade at
each join are altered. Output is a self-contained pure-Rust WAV (default) or FLAC
— no ffmpeg required.

```sh
trakktor vad timeline talk.mp3     # JSON: where the speech is, plus stats
trakktor vad cut talk.mp3          # write talk.speech.wav with the silence removed
trakktor vad split talk.mp3        # write one file per utterance
```

### timeline — report the speech, write nothing

```sh
trakktor vad timeline interview.mp3
```

Prints a JSON object: the audio `duration`, a `speech` array of `{start, end}`
intervals (seconds, original timeline), and a `stats` object — segment count,
total speech/silence seconds, speech ratio, and the longest pause. Safe and
side-effect-free. `--text` prints one `start`⇥`end` per line.

### cut — remove (or keep) the silence

```sh
trakktor vad cut interview.mp3                     # → interview.speech.wav
trakktor vad cut interview.mp3 --format flac       # smaller, lossless (integer sources)
trakktor vad cut interview.mp3 --keep non-speech   # invert: keep the non-speech
trakktor vad cut podcast.wav --max-silence-ms 400  # shorten long pauses, don't drop them
```

Concatenates the detected speech into one file, dropping non-speech, with a short
fade at each join so edits do not click. `--keep non-speech` inverts it (keep the
silence/music, drop the speech). `--max-silence-ms` collapses long pauses to a
fixed length instead of removing them, keeping the natural rhythm. The written
path is reported on stderr; stdout carries a JSON summary (`output`, `kept`,
`segments`, source and output durations).

### split — one clip per utterance

```sh
trakktor vad split lecture.mp3 --output-dir clips --min-duration-ms 500
```

Writes one file per detected span — `lecture.speech.001.wav`, `.002.wav`, … into
`--output-dir` (created if missing) — dropping any shorter than
`--min-duration-ms`. The JSON result lists each clip with its index, source
`start`/`end`, and duration.

### Detection tuning and presets

Detection is the canonical Silero algorithm, and its unit is a **speech
probability** (not decibels). `--preset` sets a base that the individual flags
override:

```
--preset tight|asr|natural    # default: tight
```

- **`tight`** (default) — aggressive silence removal (canonical Silero).
- **`asr`** — bridge long pauses, keeping speech in large chunks (a
  transcription front-end).
- **`natural`** — keep more breathing room around speech.

Override any piece directly with `--threshold`, `--min-speech-duration-ms`,
`--min-silence-duration-ms`, `--speech-pad-ms`, `--max-speech-duration-s`. Shape
the edit with `--fade-ms`, `--merge-gap-ms` (merge close spans), and `--margin-ms`
(extra audio kept around each span). `--start`/`--end` (same time format as
`asr`) define a **working window**: cutting, splitting, and inversion all happen
within `[start, end]`, so `--keep non-speech --start 1:00 --end 2:00` yields just
the silence inside that minute.

### Output format

```
--format wav|flac                 # default: wav (built-in encoder)
--audio-encoder builtin|ffmpeg    # default: builtin
--bitrate <rate>                  # e.g. 192k, for lossy ffmpeg formats
```

With the built-in pure-Rust encoder (the default), `--format wav` reproduces any
source exactly — including the float PCM that mp3/aac/vorbis decode to — so it is
the safe choice for preserving quality; `--format flac` is lossless and about
half the size, but integer-only (up to 24-bit), so a float or >24-bit source is
rejected with a message pointing back at WAV.

For anything else, `--audio-encoder ffmpeg` shells out to an installed `ffmpeg`,
and `--format` then accepts any format it writes by extension — mp3, aac, m4a,
opus, ogg, and more:

```sh
trakktor vad cut talk.mp3 --audio-encoder ffmpeg --format mp3 --bitrate 192k
trakktor vad cut talk.mp3 --audio-encoder ffmpeg --format opus
```

The cut PCM is streamed to ffmpeg and re-encoded (not stream-copied), so cuts
stay sample-accurate for any target; `--bitrate` sets ffmpeg's `-b:a` for lossy
formats. It needs `ffmpeg` on your `PATH` — the built-in path never does.

## `text` — structure text

Text-processing operations, each with a local model, fully offline. Two ship
today: `structify` (split into paragraphs) and `punctuate` (restore punctuation
and casing).

### `text structify` — split text into paragraphs

Turn an unstructured wall of text — for example a speech transcript whose line
breaks fall on engine segments rather than meaning — into readable paragraphs,
fully offline:

```sh
trakktor text structify transcript.txt            # JSON (default): model + paragraphs
trakktor text structify transcript.txt --text     # paragraphs separated by blank lines
trakktor text structify transcript.txt --pretty   # indented JSON
```

The one required argument is the path to a UTF-8 text file. Its existing line
breaks are collapsed first (a transcript's segment breaks are not paragraph
breaks), then a local **SaT** (Segment any Text) model — an XLM-RoBERTa network
that scores each position for a boundary — re-groups the text into paragraphs.
It is multilingual, Russian and English included.

The same model doubles as trakktor's splitter elsewhere: `tts` calls it to cut
a paragraph too long to speak in one utterance, picking the coarsest boundaries
that fit.

#### Models

```
--model <name|dir>      # default: sat-12l-no-limited-lookahead
```

Pass a **published name** — downloaded on first use into
`~/.trakktor/text/structify/<name>/` and reused on later runs — or a **path** to
a local checkpoint directory (one containing `config.json`). The shared XLM-R
tokenizer is downloaded once alongside. As with `asr`, weights live under the
**model directory** (`--model-dir` / `TRAKKTOR_MODEL_DIR`; default `~/.trakktor`),
not the per-project `--work-dir`.

Two families, differing in what they cut on:

- **`sat-1l-no-limited-lookahead`, `sat-3l-…`, `sat-6l-…`, `sat-9l-…`,
  `sat-12l-… (the default)`** score **paragraph** breaks and yield coarse,
  reader-style paragraphs. **Depth matters here:** shallow models give a weakly
  calibrated paragraph signal (`sat-3l` at the default `--threshold 0.5` barely
  splits at all, and has no clean paragraph threshold), so `sat-12l` is the
  default — it produces good paragraphs at `0.5`. Smaller ones are faster but
  need a lower `--threshold` and segment less cleanly. A higher `--threshold`
  gives fewer, larger paragraphs.
- **`sat-1l-sm`, `sat-3l-sm`, `sat-6l-sm`, `sat-9l-sm`, `sat-12l-sm`** score
  finer **sentence** breaks — roughly one unit per sentence, and smaller/faster.
  Use these when you want sentence-level segmentation.

```sh
# Sentence-level splitting instead of paragraphs
trakktor text structify transcript.txt --model sat-3l-sm --text
```

#### Runtime, device, and precision

```
--runtime candle|burn   # default: candle
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

`--device metal` runs on the macOS GPU and needs a build with the `metal`
feature (as for `asr`); without it, `--device metal` is rejected. `--precision
f16` (the default) uses about half the memory and is faster; `f32` computes in
full precision for reproducible results.

The shared `--runtime` flag applies too: `--runtime burn` runs the same
network on the alternative burn runtime (see the Runtime section under `asr`
— the same build feature, backends, and caveats). Both runtimes produce the
same paragraphs.

#### Segmentation controls

```
--threshold 0.5   # paragraph-boundary probability cutoff (0..=1); higher = fewer, longer paragraphs
--stride 256      # window step in tokens; a smaller stride overlaps more (steadier, slower)
--batch-size 32   # windows per forward batch — the main lever on GPU utilization
```

#### Output shape

The default output is a single JSON object: the `model` and a `paragraphs`
array. Each paragraph carries its character range `start`/`end` (code points
into the whitespace-normalized text) and its `text`:

```json
{
  "model": "sat-12l-no-limited-lookahead",
  "paragraphs": [
    { "start": 0, "end": 137, "text": "…" },
    { "start": 137, "end": 402, "text": "…" }
  ]
}
```

`--text` instead prints the paragraphs separated by a blank line.

```sh
# Paragraphs on the GPU (the default model), indented JSON
trakktor text structify transcript.txt --device metal --pretty

# From speech to paragraphs: transcribe, then structure
trakktor asr whisper talk.mp3 --timestamps none --text > transcript.txt
trakktor text structify transcript.txt --text
```

### `text punctuate` — restore punctuation and casing

Turn raw ASR output — lowercase text with no punctuation, as the Vosk and
GigaAM engines emit — into readable text: restore punctuation, capitalization
(including acronyms like `NATO` and `U.S.`), and sentence boundaries, fully
offline:

```sh
trakktor text punctuate transcript.txt          # JSON (default): model + text + sentences
trakktor text punctuate transcript.txt --text   # the restored text
trakktor text punctuate transcript.txt --pretty # indented JSON
```

The one required argument is the path to a UTF-8 text file. Its whitespace is
collapsed first, then a local multilingual **XLM-RoBERTa** model — an encoder
with a cascade of punctuation, true-casing, and sentence-boundary heads —
restores the marks and casing and splits the stream into sentences. It handles
47 languages, Russian and English included.

#### Models

```
--model <name|dir>      # default: xlmr-47lang
```

Pass a **published name** — downloaded on first use into
`~/.trakktor/text/punctuate/<name>/` and reused on later runs — or a **path** to
a local model directory. One model ships today:

- **`xlmr-47lang` (the default)** — a multilingual XLM-RoBERTa punctuator over
  47 languages, Russian and English included. It restores `.`, `,`, `?` (and
  the marks of other scripts), capitalization, and sentence boundaries. On first
  use its SentencePiece model and PyTorch weights (extracted from the published
  NeMo archive) are downloaded from Hugging Face.

#### Runtime, device, and precision

```
--runtime candle|burn   # default: candle
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

Same as `structify` (and `asr`): `--device metal` needs the `metal` feature and
`--runtime burn` the `burn` feature. Both runtimes, and both f32 devices,
produce byte-identical output; `f16` may differ by the odd sentence split on a
borderline token.

#### Windowing controls

```
--overlap 16      # token overlap between consecutive windows (for long inputs)
--batch-size 16   # windows per forward batch
```

Inputs longer than the model's window (256 tokens) are split into overlapping
windows and stitched back together; these tune that.

#### Output shape

The default output is a single JSON object: the `model`, the restored `text`
(the sentences joined by a space), and the `sentences` array.

```json
{
  "model": "xlmr-47lang",
  "text": "Hello friend, how's it going? It's snowing outside right now.",
  "sentences": [
    "Hello friend, how's it going?",
    "It's snowing outside right now."
  ]
}
```

`--text` instead prints the restored text alone.

```sh
# From raw Russian speech to punctuated paragraphs: transcribe, punctuate, structure.
# v3_rnnt gives the most accurate raw (lowercase, unpunctuated) Russian text.
trakktor asr gigaam talk.mp3 --model v3_rnnt --timestamps none --text > raw.txt
trakktor text punctuate raw.txt --text > punctuated.txt
trakktor text structify punctuated.txt --text
```

## `feed` — RSS / Atom / JSON Feed

### Discover feeds on a page

```sh
trakktor feed discover https://example.com
```

Returns the feeds declared on the page (`url`, `type`, `title`). An empty result
is success.

### Read a feed

```sh
trakktor feed read https://example.com/feed.xml             # JSON (default)
trakktor feed read https://example.com/feed.xml --all --fields all --text
```

Accepts a feed URL or a regular page (autodiscovery applies, reading the first
feed found). Each publication carries a stable `uid` and an `is_read` flag.

- By default only **unread** publications are returned; `--all` includes read
  ones.
- `uid` is each publication's **primary key** — the id you pass to `mark-read` —
  so it is **always** included, independent of `--fields`.
- `--fields <list>` selects the *additional* fields: a comma-separated list of
  `is_read,title,link,published,updated,summary,content,authors`, or the special
  values `minimal` (default, `title,link`) and `all`.
- With `--text`, `uid` is the first column and a `mark-read` hint is printed to
  stderr.

The `uid` is `hex(BLAKE3(feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key))` and is
stable across runs for the same feed + entry.

### Mark publications read

```sh
trakktor feed mark-read <uid> [<uid>...]
```

Idempotent. Read state is stored as plain files under `<work-dir>/feed/`, sharded
by uid; nothing else is needed (no database).

## Typical agent workflow

```sh
trakktor feed discover https://example.com             # find a feed
trakktor feed read https://example.com/feed.xml        # read unread items (JSON)
# … take each item's uid …
trakktor feed mark-read <uid1> <uid2>                  # mark them handled
```

On the next `read`, marked publications are no longer returned.

## `skill` — generate the agent skill

trakktor can describe itself to a coding agent as an Agent Skill. The content is
generated from the live `clap` definition, so it always matches the installed
binary — there is no hand-written reference to drift out of date.

### Show the skill

```sh
trakktor skill show          # narrative guide (JSON { "content": … } by default)
trakktor skill show --full   # plus the full command/flag/value reference
trakktor skill show --text   # the raw Markdown
```

By default the Markdown is wrapped as `{ "content": "…" }`; `--text` prints the
raw Markdown.

### Install the stub

```sh
trakktor skill install claude            # ./.claude/skills/trakktor/SKILL.md
trakktor skill install agents            # ./.agents/skills/trakktor/SKILL.md
trakktor skill install claude --global   # ~/.claude/skills/trakktor/SKILL.md
trakktor skill install claude --force    # overwrite an existing stub
```

Writes a thin discovery stub to `<dir>/skills/trakktor/SKILL.md`. The
destination is explicit: `claude` or `agents` selects the agent layout in the
current project, and `--global` (only valid with `claude`) targets `~/.claude`
in your home directory. A project install creates the whole path; a global
install requires `~/.claude` to already exist — it is never created, and the
command fails if it is missing. The stub only points back at `trakktor skill
show`, so it never goes stale between releases. An existing `SKILL.md` is left
untouched unless `--force` is given. By default the result is a single JSON
object `{ "path": "…", "status": "written" | "skipped" }`; `--text` prints it as
lines.

## Acknowledgments

trakktor ports and builds on several open-source projects, all MIT- or
Apache-licensed:

- **`asr whisper`** — [OpenAI Whisper](https://github.com/openai/whisper)
  (MIT), run on [candle](https://github.com/huggingface/candle) (Apache-2.0 OR
  MIT), with an optional alternative runtime on
  [burn](https://github.com/tracel-ai/burn) (Apache-2.0 OR MIT).
- **`asr gigaam`** — [GigaAM](https://github.com/salute-developers/GigaAM)
  (MIT): Conformer acoustic models (CTC and RNN-T) by the GigaChat team,
  pipeline ported to the same candle runtime, with the same optional burn
  runtime.
- **`asr vosk`** — [Vosk](https://alphacephei.com/vosk/) models by Alpha
  Cephei (Apache-2.0): Zipformer2 RNN-T transducers trained with
  [k2-fsa/icefall](https://github.com/k2-fsa/icefall) (Apache-2.0); the
  inference pipeline is a native port of
  [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0)
  with Kaldi-compatible fbank features from
  [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
  (Apache-2.0), on the same candle runtime with the same optional burn
  runtime.
- **`tts qwen3-tts`** — [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
  (Apache-2.0) by the Alibaba Qwen team: the 12 Hz talker, residual code
  predictor, and codec decoder, ported to the same candle runtime, with the
  same optional burn runtime. Model weights and the bundled speech tokenizer
  are downloaded at runtime. Technical
  report: [arXiv:2601.15621](https://arxiv.org/abs/2601.15621).
- **`vad`, and `asr --vad`** — [Silero-VAD](https://github.com/snakers4/silero-vad)
  (MIT): the ported speech detector behind both the audio editing commands and
  the transcription preprocessing stage.
- **`text structify`** — [SaT / wtpsplit](https://github.com/segment-any-text/wtpsplit)
  (MIT; Frohmann et al., *Segment Any Text*, EMNLP 2024), with the
  [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) (MIT)
  tokenizer, on the same candle runtime with the same optional burn runtime.
  Please cite the SaT paper if you use these models.
- **`text punctuate`** — the
  [1-800-BAD-CODE multilingual punctuation/true-casing model](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)
  (Apache-2.0), an [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (MIT) encoder with cascaded heads, its post-processing following the author's
  [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package (MIT), on
  the same candle runtime with the same optional burn runtime.

See [`NOTICE`](NOTICE) for the full third-party attributions and license
notices.
