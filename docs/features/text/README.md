# `text` — structure and transform text

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

Text-processing operations, each with a local model, fully offline. Three ship
today: `structify` (split into paragraphs), `punctuate` (restore punctuation and
casing), and `stress` (mark the stressed vowel in Russian).

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

## `text stress` — mark the stress in Russian text

Russian does not write stress, and a speech synthesizer needs it: an unmarked
word is read by guesswork, and a pair like `все`/`всё` cannot be told apart by a
stress mark at all. This marks it, fully offline:

```sh
trakktor text stress chapter.txt              # JSON (default): text, counts, misses
trakktor text stress chapter.txt --text       # the marked text
trakktor text stress chapter.txt --pretty     # indented JSON
```

The one required argument is the path to a UTF-8 text file. The output is the
**same text**: nothing is reordered or rewritten, only marks are inserted and
`е` becomes `ё` where it belongs.

```
Он запер замок и ушел.  →  +Он з+апер зам+ок +и уш+ёл.
В чашке остыл мед.      →  В ч+ашке ост+ыл м+ёд.
```

Three layers do the work, in this order: your own dictionary, then a small
encoder that reads the sentence around a word whose spelling does not say how it
is read (`з+амки` vs `замк+и`), then a network over the word's character n-grams
that names the stressed vowel and any hidden `ё`. **A word you have marked
yourself is never re-marked**, so marking part of the text by hand and leaving
the rest to the model is the normal way to work.

The natural consumer is [`tts espeech`](../tts/espeech.md), which reads `+` as a
real input — and which calls this operation itself by default.

### Mark form

```
--marker plus|acute     # default: plus
```

`plus` writes `+` before the stressed vowel — the form the speech engines read.
`acute` writes the combining acute accent U+0301 after it, the form dictionaries
and corpora use (`он за́пер замо́к`); `ё` is left unaccented there, as it is
stressed by definition. Either form is **accepted on input** too and is never
overwritten, so the output of one run can be fed back into another.

### The letter ё

```
--yo auto|off           # default: auto
```

`auto` writes `ё` where it belongs. This is the only thing that separates
`все` from `всё` or `небо` from `нёбо`, and since `ё` is always stressed, writing
it marks the stress at the same time. It does mean the output differs from the
input in **letters**, not only in marks — so `off` turns the whole layer off and
guarantees that stripping the marks gives back your text character for
character.

### Your own dictionary

```
--dict <path>           # repeatable
```

Names, terms, rare words, and the three-way homographs the model cannot express
(`с+ела` / `сел+а` / `с+ёла`) belong here. One marked word per line, `#` starts a
comment:

```
# names and terms
кинов+арь
ф+орзац
Корол+ёв
```

The key is the line with the marks removed, lowercased and with `ё` folded into
`е` — so `Корол+ёв` covers both `Королев` and `Королёв`. An entry wins over the
model, keeps the case of the text it is applied to, and is written in before the
model runs. A malformed line is an error, not a silent skip.

This is the way to fix a word the model gets wrong. On its own it reads
`Он показал мне киноварь` as `к+иноварь`; with the entry above it reads
`кинов+арь`, and it stays that way in every later run.

### Models, runtime, device, precision

```
--model <name|dir>      # default: silero-ru
--runtime candle|burn   # default: candle
--device cpu|metal      # default: cpu
--precision f32|f16     # default: f32
```

One model ships today: **`silero-ru`**, downloaded on first use (54 MB) into
`~/.trakktor/text/stress/silero-ru/` and converted once into the form later runs
read. It is Russian only.

Unlike the other text models the default precision here is **f32**: both networks
are small enough that it costs nothing, and every decision they make is a
threshold that half precision could flip. And unlike the others, **the CPU is the
target**: marking a page of prose takes about 0.2 s there, where the GPU spends
longer starting up than computing. All runtimes and devices produce the same
text.

### Output shape

```json
{
  "model": "silero-ru",
  "marker": "plus",
  "text": "+Он з+апер зам+ок +и уш+ёл.",
  "stats": {
    "words": 5, "stressed": 5, "yo_restored": 1,
    "homographs": 1, "from_dictionary": 0
  },
  "unstressed": []
}
```

`stats.words` counts the words that can carry a stress at all — one without a
vowel is not counted anywhere. **`unstressed` is the actionable half**: when the
model is not confident it leaves the word alone rather than guessing, and those
words are listed here, deduplicated and lowercased — which is exactly the form a
`--dict` entry takes, so the list reads as a to-do. It is short: on a page of
ordinary prose a word or two, usually a proper name.

`--text` prints the marked text alone.

```sh
# from raw speech to marked-up text, ready to speak
trakktor asr gigaam talk.mp3 --model v3_rnnt --timestamps none --text > raw.txt
trakktor text punctuate raw.txt --text > punctuated.txt
trakktor text stress punctuated.txt --text > marked.txt
```
