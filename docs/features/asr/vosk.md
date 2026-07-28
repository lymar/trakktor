# `asr vosk` — Vosk engine

> One of the engines of [trakktor's `asr` command](README.md); the behavior
> shared by all engines — audio input, model storage, device and precision,
> runtime, timestamps, output files — is documented there.

[Vosk](https://alphacephei.com/vosk/)'s current model line (Alpha Cephei) is
a family of **Zipformer2 RNN-T transducers** trained with
[k2-fsa/icefall](https://github.com/k2-fsa/icefall) — mainly for Russian,
with more languages available. trakktor is a native port of the reference
pipeline (feature extraction, encoder, and transducer decoding follow the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) runtime), on the same
candle runtime as the other engines, and all the
[shared behavior](README.md) — audio input, model storage, device and
precision, output — applies as-is. The models emit **lowercase text without
punctuation** (the usual form for downstream processing or WER evaluation).
Vosk does not detect the language; `--language <code>` only annotates the
output. The shared [`--runtime burn`](README.md#runtime) runs the same network
on the alternative burn runtime.

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
segmentation as [GigaAM](gigaam.md)) and stitched back onto the original
timeline. The whole pipeline is streaming and bounded in memory regardless of
file length.
**Streaming** models instead run a native chunked encoder with cached state —
low latency and low memory, the basis for future live transcription; on files,
the offline models are more accurate.
