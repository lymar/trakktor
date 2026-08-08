# `ocr paddle` — the PP-OCRv5 pipeline

> Part of [`ocr`](README.md); the shared page model, output shape and language
> rules are described there.

Three networks in sequence: a detector finds the text lines on the page and
returns a quadrangle for each, every quadrangle is straightened out of the page
into an upright crop, and a recognizer reads the crop. An optional third model
decides whether a crop is upside down before it is read.

trakktor runs PaddleOCR's own published artifacts directly — the graph and the
weights exactly as they are published — so there is no conversion step, no
Python and no ONNX runtime in the picture. The full working set for a language
is about 13 MB, downloaded on first use into `~/.trakktor/ocr/paddle/`.

## Models

```
--lang <code>              # default: ru — picks the recognizer
--det-model <name|dir>     # default: PP-OCRv5_mobile_det
--rec-model <name|dir>     # overrides what --lang chose
```

A detector serves every language: it looks for text as such and does not care
what script it is. Twelve recognizers cover the scripts between them — Eastern
Slavic, wider Cyrillic, Latin, English, Arabic, Devanagari, Korean, Thai, Greek,
Telugu, Tamil, and Chinese/Japanese. They share an architecture and differ only
in the alphabet they were trained on.

There are two detectors, and the default is the small one:

| `--det-model` | download | detection* | what it changes |
|---|---:|---:|---|
| `PP-OCRv5_mobile_det` | 4.7 MB | ~1 s | the default |
| `PP-OCRv5_server_det` | 88 MB | 12–15 s | finds short lines and superscript footnote markers the small one drops, and keeps a line whole where the small one splits it |

\* the detection stage alone, on an A4 scan at `--limit-side-len 1920`;
recognition comes on top of it either way, so the whole page went from ~14 s to
~24 s in the same measurement.

The large one is worth its time on a densely set page — footnotes, marginal
numbers, a line of two words — and not otherwise. It is also, on one measured
page, slightly more likely to lose an ordinary line of body text to
`--box-thresh`: its probability map is sharper, so a box score can land just
under the default 0.6. If a line goes missing with it, try `--box-thresh 0.4`.

Either flag also takes a **path** to a directory holding a model's
`inference.json`, `inference.pdiparams` and `config.json`, which is how to run a
model that is not in the catalog.

### The dictionary is the ceiling

A recognizer can only emit characters from its own dictionary, and the
dictionaries are not supersets of one another. The English one has no en dash,
so `pp. 358–359` comes back as `pp. 358359`; the Latin one has none either, but
does have `ß` and `ā`; the Eastern Slavic one has the dash. A character outside
the dictionary is simply missing from the output, which reads like a recognition
failure and is not one.

## Finding small type

```
--limit-side-len <px>      # default: 960
```

The page is scaled so that its longest side is at most this many pixels before
detection. **This is the single most consequential setting.** An A4 page scanned
at 300 dpi is 3508 pixels tall, so the default shrinks it nearly fourfold — and
at that size footnote-sized text stops being detected at all, silently: the
lines are absent from the result rather than wrong in it.

On a dense page, 1536 or 2048 finds those lines. The cost rises roughly with the
area, so four times the side length is four times the detection time.

## Tuning the detector

```
--thresh <p>               # default: 0.3   pixel is text above this probability
--box-thresh <p>           # default: 0.6   mean probability a box must reach
--unclip-ratio <ratio>     # default: 1.5   how far a box is expanded
--drop-score <p>           # default: 0.5   drop lines read with less confidence
```

The detector marks a shrunken core of each line rather than its full extent, so
some expansion is always needed; more of it captures tall letters and accents,
and eventually the neighbouring line. `--drop-score 0` keeps everything, which
is the setting to use when a line is missing and the question is whether it was
found at all.

```
--textline-orientation     # off by default
```

adds the classifier that turns an upside-down line around. It costs a little per
line and rarely fires on a scanned book.

## What Markdown does and does not do

`--format md` works from two things: the geometry of the lines — spacing,
indents, the width of the column, where each line ends — and the **labels** of
the layout model, which runs by default.

The geometry gives the reading order (including columns), the paragraphs, the
joining of hyphenated line breaks, and the continuation of a paragraph across a
page break. The labels give what a block *is*: document title, section heading,
abstract, footnote, running head, page number, table, formula, picture, caption.
That second half is what geometry cannot reach — a title set in capitals makes
*shorter* boxes than the text below it, so it is not found by size at all.

`--no-layout` drops back to geometry alone. Then headings are guessed from
centring and isolation rather than known; running heads, page numbers and
footnotes are separated by position and type size; tables come out as text; and
illustrations are found by looking for ink that no text box covers, which finds
pictures but also finds large tables and display formulas.

The layout model is imperfect in its own way: it is trained on Chinese and
English documents, and on a page in an unfamiliar script it is much less sure
of itself — the blocks are still in the right places, but closer to the score
floor that keeps them. A page it says nothing about is read exactly as
`--no-layout` would read it, and a line that fell in no block is never dropped.

The stage is steered from here by three flags:

```
--no-layout                # skip it: geometry alone, and 129 MB not downloaded
--layout-model <name|dir>  # which layout model to run
--layout-threshold <p>     # one score floor for every label, over the defaults
```

`--layout-threshold` is the one to reach for on a page in an unfamiliar script,
where the labels come back thin or not at all. Lowering it is not free in one
direction: a picture box covering the whole sheet passes a low floor too, and a
picture swallows the blocks inside it, so the page can end up with fewer labels
than the defaults gave it. [`ocr layout --boxes`](layout.md) shows what a value
does to a page in one picture, for a second rather than a page of reading.

## Speed

The recognizer, not the detector, is where the time goes: on a CPU the detector
takes about a second and a half for an A4 page at `--limit-side-len 1920`, and
the recognizer about a third of a second per line — so a 45-line page is around
seventeen seconds. `--device metal` moves both onto the GPU.
