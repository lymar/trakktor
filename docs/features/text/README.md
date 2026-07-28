# `text` — structure text

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

Text-processing operations, each with a local model, fully offline. Two ship
today: `structify` (split into paragraphs) and `punctuate` (restore punctuation
and casing).

## `text structify` — split text into paragraphs

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

The same model doubles as trakktor's splitter elsewhere:
[`tts`](../tts/README.md) calls it to cut a paragraph too long to speak in one
utterance, picking the coarsest boundaries that fit.

### Models

```
--model <name|dir>      # default: sat-12l-no-limited-lookahead
```

Pass a **published name** — downloaded on first use into
`~/.trakktor/text/structify/<name>/` and reused on later runs — or a **path** to
a local checkpoint directory (one containing `config.json`). The shared XLM-R
tokenizer is downloaded once alongside. As with
[`asr`](../asr/README.md#model-storage), weights live under the
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

### Runtime, device, and precision

```
--runtime candle|burn   # default: candle
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

`--device metal` runs on the macOS GPU and needs a build with the `metal`
feature (as for [`asr`](../asr/README.md#device-and-precision)); without it,
`--device metal` is rejected. `--precision
f16` (the default) uses about half the memory and is faster; `f32` computes in
full precision for reproducible results.

The shared `--runtime` flag applies too: `--runtime burn` runs the same
network on the alternative burn runtime (see the
[Runtime section under `asr`](../asr/README.md#runtime)
— the same build feature, backends, and caveats). Both runtimes produce the
same paragraphs.

### Segmentation controls

```
--threshold 0.5   # paragraph-boundary probability cutoff (0..=1); higher = fewer, longer paragraphs
--stride 256      # window step in tokens; a smaller stride overlaps more (steadier, slower)
--batch-size 32   # windows per forward batch — the main lever on GPU utilization
```

### Output shape

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

## `text punctuate` — restore punctuation and casing

Turn raw ASR output — lowercase text with no punctuation, as the
[Vosk](../asr/vosk.md) and [GigaAM](../asr/gigaam.md) engines emit — into
readable text: restore punctuation, capitalization
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

### Models

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

### Runtime, device, and precision

```
--runtime candle|burn   # default: candle
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

Same as `structify` (and [`asr`](../asr/README.md#runtime)): `--device metal`
needs the `metal` feature and
`--runtime burn` the `burn` feature. Both runtimes, and both f32 devices,
produce byte-identical output; `f16` may differ by the odd sentence split on a
borderline token.

### Windowing controls

```
--overlap 16      # token overlap between consecutive windows (for long inputs)
--batch-size 16   # windows per forward batch
```

Inputs longer than the model's window (256 tokens) are split into overlapping
windows and stitched back together; these tune that.

### Output shape

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
