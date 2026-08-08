# `ocr paddle` — the PP-OCRv5 pipeline

> Part of [`ocr`](README.md); the shared page model, output shape and language
> rules are described there.

Three networks in sequence: a detector finds the text lines on the page and
returns a quadrangle for each, every quadrangle is straightened out of the page
into an upright crop, and a recognizer reads the crop. An optional third model
decides whether a crop is upside down before it is read.

trakktor runs PaddleOCR's own published artifacts directly — the graph and the
weights exactly as they are published — so there is no conversion step, no
Python and no ONNX runtime in the picture. The models are downloaded on first
use into `~/.trakktor/ocr/paddle/`, and the run says what it is fetching, and
how much, before the first byte moves.

## Models

```
--lang <code>              # default: en — picks the recognizer
--quality <best|fast>      # default: best — which end of the catalog to read with
--det-model <name|dir>     # overrides what --quality chose
--rec-model <name|dir>     # overrides what --lang and --quality chose
```

A detector serves every language: it looks for text as such and does not care
what script it is. Thirteen recognizers cover the scripts between them — Eastern
Slavic, wider Cyrillic, Latin, English, Arabic, Devanagari, Korean, Thai, Greek,
Telugu, Tamil, and Chinese/Japanese, which has two. Eleven of them share an
architecture and differ only in the alphabet they were trained on.

**`--quality` is not a pair of model names but a rule**: for the language asked
for, take the model that reads its alphabet best, or the one that reads it
cheapest. It has to be a rule because the two ends are not the same models for
every language — the newest and largest generation PaddleOCR publishes carries
no Cyrillic at all, so "newest and largest" and "best for this page" are
different answers. Which models actually ran is in the result, under `models`.

The default is `best`: an OCR run is wanted for its accuracy, and a page that
reads badly is worth less than a page that reads slowly. The rule always moves
the detector, of which there are two:

| detector | download | what it changes |
|---|---:|---|
| `PP-OCRv5_server_det` | 88 MB | `--quality best` — finds the short line that closes a paragraph and the superscript marker of a footnote, and keeps a line whole where the small one splits it into pieces |
| `PP-OCRv5_mobile_det` | 4.7 MB | `--quality fast` — a quarter to a third off the time a page takes |

It moves the recognizer only for Chinese and Japanese, the one alphabet
PaddleOCR publishes in two sizes:

| recognizer | download | what it changes |
|---|---:|---|
| `PP-OCRv5_server_rec` | 84 MB | `--quality best` for `zh`, `ja` and `chinese_cht` |
| `PP-OCRv5_mobile_rec` | 17 MB | `--quality fast` for the same three |

Every other language has one recognizer, and `--quality` leaves it alone;
`--lang list` prints the model each code resolves to under the quality given.

Measured on an A4 page at 300 dpi, on a GPU, whole run including the layout
stage — a page of ordinary prose reads the same text either way, and the
difference is on dense pages, small print and scans:

| | `--quality fast` | `--quality best` |
|---|---:|---:|
| page of 47 lines, footnotes | 12 s, 44 lines found | 17 s, 47 lines found |
| dense page of 108 lines | 16 s, 101 found | 22 s, 107 found |

The large detector is also, on one measured page, slightly more likely to lose
an ordinary line of body text to `--box-thresh`: its probability map is sharper,
so a box score can land just under the default 0.6. If a line goes missing, try
`--box-thresh 0.4`.

`--det-model` and `--rec-model` also take a **path** to a directory holding a
model's `inference.json`, `inference.pdiparams` and `config.json`, which is how
to run a model that is not in the catalog.

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
detection. An A4 page scanned at 300 dpi is 3508 pixels tall, so the default
shrinks it nearly fourfold — and small type can stop being detected at all,
silently: the lines are absent from the result rather than wrong in it.

**How much this matters depends on the detector.** With `--quality fast` it is
the single most consequential setting: on a dense page the small detector found
101 lines at 960 and 108 at 1920. The large one is far less sensitive — 107 at
960 and 108 at 1920 on the same page — so under the default quality this is a
flag to reach for when a line is missing, not one to raise routinely. Raising it
is also more expensive there: the cost rises roughly with the area, and the
large detector is the slower network to begin with.

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

The labels also reach one stage earlier, into the lines themselves: where two
columns are set close together the detector joins a line of one column to the
line facing it in the other, and the boundary between two blocks is what takes
that line apart again and has each half read on its own. This is the one thing
the layout model changes about `--format lines` as well as about Markdown.

`--no-layout` drops back to geometry alone. Then headings are guessed from
centring and isolation rather than known; running heads, page numbers and
footnotes are separated by position and type size; tables come out as text;
illustrations are found by looking for ink that no text box covers, which finds
pictures but also finds large tables and display formulas; and a line glued
across a gutter stays glued.

The layout model is imperfect in its own way: it is trained on Chinese and
English documents, and on a page in an unfamiliar script it is much less sure
of itself — the blocks are still in the right places, but closer to the score
floor that keeps them. A page it says nothing about is read exactly as
`--no-layout` would read it, and a line that fell in no block is never dropped.

The stage is steered from here by three flags:

```
--no-layout                # skip it: geometry alone, and 130 MB not downloaded
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

With `--quality fast` the recognizer, not the detector, is where the time goes:
the small detector takes about a second for an A4 page at the default side
length, and the recognizer a fraction of a second per line, so most of a page is
its lines. `--quality best` adds a heavier detector on top of that — a few
seconds at the default side length, and around ten at `--limit-side-len 1920`,
where it becomes the larger half of the page.

`--device metal` moves everything onto the GPU. It is worth roughly a quarter of
the large detector's time, which is to say it does not turn tens of seconds into
one: the choice between the two qualities is not a choice a faster device makes
for you.
