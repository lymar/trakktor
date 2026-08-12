# `tts qwen3-tts` — Qwen3-TTS engine

> One of the engines of [trakktor's `tts` command](README.md); the shared
> behavior — text input, paragraph splitting and joining, loudness matching,
> output containers — is documented there.

A native port of the open **Qwen3-TTS 12 Hz** family. Three stages run in
sequence: a Qwen3 decoder ("talker") reads the text and predicts the first
codebook of every 12.5 Hz frame; a small code predictor fills that frame's
remaining 15 residual codebooks; and a causal convolutional codec decoder turns
the finished frames into a 24 kHz waveform.

```bash
trakktor tts qwen3-tts "Привет! Это синтез речи." --language russian -o hello.wav
trakktor tts qwen3-tts --text-file article.md --language russian -o article.mp3
```

| Flag | Default | Meaning |
|---|---|---|
| `--voice <name>` | `serena` | Preset timbre: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. |
| `--language <lang>` | `auto` | Target language, set independently of the voice: `russian`, `english`, `german`, `spanish`, `chinese`, `japanese`, `french`, `korean`, `italian`, `portuguese`. |
| `--model <name\|dir>` | `0.6b-customvoice` | `0.6b-customvoice` or `1.7b-customvoice` (larger, slower), or a checkpoint directory. |
| `-o, --output <path>` | `speech.wav` | Where to write the audio; the container follows the extension. |
| `--text-file <path\|->` | — | Read the text from a file, or from standard input with `-`. |
| `--text-format <auto\|txt\|md>` | `auto` | Where paragraphs end and whether markup is stripped. |
| `--pause-ms <ms>` | `500` | Gap between paragraphs (half that inside a split paragraph). |
| `--levels <match\|keep>` | `match` | Bring the paragraphs to a common loudness before joining, or keep the levels the model gave them. |
| `--audio-encoder <auto\|builtin\|ffmpeg>` | `auto` | Who writes the file: the built-in wav/flac encoder, or ffmpeg for everything else. |
| `--bitrate <rate>` | — | `-b:a` for lossy ffmpeg formats, e.g. `192k`. |
| `--seed <int>` | `0` | Makes a sampled run repeatable. |
| `--temperature`, `--top-k`, `--repetition-penalty` | `0.9`, `50`, `1.05` | Sampling controls. |
| `--greedy` | off | Take the most likely code instead of sampling — deterministic, usually flatter. Cannot be combined with the sampling flags (`--seed`, `--temperature`, `--top-k`, `--repetition-penalty`). |
| `--precision <bf16\|f32>` | `bf16` on candle + Metal, `f32` elsewhere | **Runtime- and device-dependent default.** `bf16` is the format the weights are stored in and what the reference runs — and what keeps `1.7b-customvoice` within a 16 GB machine on Metal; only candle on Metal serves it (the burn runtime and candle's CPU backend compute in `f32`, and asking them for `bf16` is an error rather than a silent downgrade). `f32` doubles the memory and is reproducible. The codec always runs in full precision either way. |
| `--runtime <candle\|burn>` | `candle` | Inference runtime; burn needs the `burn` build feature and computes in f32 only. |
| `--device <cpu\|metal>` | `cpu` | Compute device; `metal` needs the `metal` build feature (burn brings its own). |

Any voice can speak any supported language: the language is a separate
conditioning token, not a property of the timbre. Russian is supported
first-class. Two presets are dialect voices — with `--language auto` (the
default) or `chinese`, `eric` speaks Sichuan Mandarin and `dylan` Beijing
Mandarin, exactly as the reference resolves them; any other explicit language
overrides the dialect.

Generation **samples** by default, so two runs of the same text differ slightly;
`--seed` pins a run, and `--greedy` removes the randomness altogether. The
codec decoder is deterministic either way — with the frames fixed, it
reproduces the same waveform every time.

The first use downloads the checkpoint into the model directory — 2.5 GB for
`0.6b-customvoice`, 4.5 GB for `1.7b-customvoice`, of which 0.7 GB is the
codec — and later runs reuse it. `--device metal` is considerably faster than
the CPU and is the recommended way to run it — the CPU path is impractically
slow for anything past a short phrase.

Resident memory is a figure of its own, though for `1.7b-customvoice` it
lands next to the download: `bf16` weights load in the format they are stored
in, so the model needs roughly 4.5 GB of memory in `bf16` and about twice
that in `f32`. On a 16 GB machine only `bf16` is practical for it on candle
with Metal, which is why `bf16` is the default there — it is what keeps the
large model in memory. On the CPU candle computes in `f32` (its CPU backend
has no `bf16` arithmetic), and burn does so everywhere; burn still runs the
same model in `f32` on Metal within 16 GB — see Runtime below.

## Runtime

The alternative [burn](https://github.com/tracel-ai/burn) runtime (see the
[Runtime section under `asr`](../asr/README.md#runtime)) runs this engine too,
with one restriction: **it
computes in f32 only**. You need not pass `--precision` for it — the default
follows the runtime and device, `f32` whenever `--runtime burn` is selected
(and on candle's CPU path) and `bf16` on candle with Metal. Half precision is
unavailable on burn from below: its Metal backend cannot compile `bf16`
kernels, and `f16` is excluded by the model itself — so `--precision bf16
--runtime burn` is a validation error rather than a silent downgrade.

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
