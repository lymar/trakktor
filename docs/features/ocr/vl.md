# `ocr vl` — the PaddleOCR-VL document model

> Part of [`ocr`](README.md); the shared page model and output shape are
> described there.

A single generative model reads the page. Instead of picking each character out
of a fixed dictionary, it **writes out** what it sees, token by token — so it is
not tied to one alphabet, works out the writing system for itself, and can be
asked for a table as markup or a formula as LaTeX rather than as lines of text.

That is a different bargain from [`paddle`](paddle.md), not a better one:

| | `paddle` | `vl` |
|---|---|---|
| download | ~13 MB per language | **~2 GB**, once |
| a page | a second or two | tens of seconds |
| alphabet | the recognizer's dictionary | whatever the model knows |
| tables, formulas | lines of text | markup, LaTeX |

Reach for `vl` when the page is in a script `paddle` has no recognizer for,
when it mixes scripts line by line, or when the structure of a table or a
formula is the point. Otherwise `paddle` is the right default.

```sh
trakktor ocr vl page.png                       # JSON: pages, lines, boxes, scores
trakktor ocr vl page.png --text                # the text
trakktor ocr vl page.png --format md --out doc.md
trakktor ocr vl table.png --task table --text  # the table as markup
```

## It reads blocks, not lines

The model needs a **coherent region** of the page. Given a whole page it tends
to lose its place and repeat itself; given one isolated line it has too little
context to settle on a writing system and will happily invent an alphabet. Given
a few lines that belong together it does its best work.

So `vl` does not run alone. It borrows `paddle`'s text detector — one that
looks for text as such regardless of script — groups the lines it finds into
blocks, and reads a block at a time. The detector is what makes the arrangement
work: it finds lines in scripts that no recognizer in the classic catalog could
read.

**The default detector here is the large one** (`PP-OCRv5_server_det`, 88 MB),
where `paddle` defaults to the small one. A line the small detector misses is
not merely missing from the result here: it also changes how the lines around it
are grouped, so a footnote can come back as a fragment with its opening gone.
That is worth about three seconds on a page that already takes thirty. Pass
`--det-model PP-OCRv5_mobile_det` for the fast one.

One caveat comes with it, and it applies here as much as in `paddle`: the large
detector's probability map is sharper, so an ordinary line can score just under
the default `--box-thresh 0.6` and be dropped before the blocks are cut. If a
block comes back short of a line, try `--box-thresh 0.4` — knowing its price:
weak boxes over decorative ink also make it in, shift the blocks around them,
and can pull hallucinated lines into the result. Measured across a page set,
0.4 changed nothing on most pages, recovered a dropped line on one, and traded
lines for hallucinations on the ornate ones — which is why it is a flag and not
the default.

Two consequences worth knowing:

- **Three models are downloaded**: the 1.92 GB one, the 88 MB detector, and the
  129 MB layout model below — about 2.1 GB in all.
- **The quality of the reading depends on the grouping.** Where the grouping is
  bad — a caption wrapped around a picture, a table split into columns — the
  model gets the input it is weakest on.

### The layout model does the grouping

By default the blocks are cut along the **regions a layout model found**, not
across them. That is what a table needs: grouped by geometry alone it comes back
as strips with its columns doubled, because the rows of a table look alike and
sit near each other. Grouped by its region it goes to the model whole and is
asked for **as a table** — and a formula is asked for as a formula.

Measured on a calendar page mixing Tibetan, English and Devanagari: seven blocks
instead of sixteen, the table returned as cell markup instead of doubled
columns, and the page read in 21 seconds instead of 89.

`--no-layout` goes back to grouping by geometry alone, and the knobs below are
what shape it then. They still apply to the lines the layout model did not
claim.

The grouping is adjustable:

```
--block-lines <n>          # default: 12    most rows in one block
--block-gap <ratio>        # default: 0.8   × line height: the gap that ends a block
--block-row-gap <ratio>    # default: 1.2   × line height: word space vs column gutter
--block-overlap <ratio>    # default: 0.3   how much two rows must share sideways
--block-height <ratio>     # default: 1.5   how much line heights may differ
--block-padding <ratio>    # default: 0.35  × line height: margin around the cut
--whole-page                             # do not group: one call per page
```

`--block-height` is the one that matters most on a page that mixes writing
systems. A line of a stacking script — Tibetan, Devanagari — runs taller than an
alphabetic line of the same type size, and that ratio is the only sign the
geometry has that the two are different kinds of text. Raising it merges them
into one block, and a block that mixes scripts line by line reads **markedly
worse** than the same lines on their own: the answer comes back as fluent
nonsense in some third script. More context helps only while the block stays
coherent.

`--crops <dir>` writes out every block exactly as the model received it. When a
reading goes wrong this is the first thing to look at: it usually shows that the
model was handed something incoherent rather than that it misread something
clear. The other tell is `score`: a block the model invented sits far below the
rest of the page.

## Tasks

```
--task <ocr|table|formula|chart>     # default: ocr
```

`ocr` reads the text and is the only one that works block by block. The other
three are asked of the **page as a whole**, because a table or a formula is
itself one region and cutting it up destroys the structure that made the
question worth asking.

## Stopping

A generative reader does not fail by producing a wrong letter; it fails by not
stopping — one syllable, or one short phrase, repeated until the budget runs
out. So the loop watches for that explicitly and cuts the answer where the
repetition began.

```
--max-tokens <n>           # default: 1024   ceiling on one block's answer
--drop-score <p>           # default: 0.3    mean token probability to keep a block
```

`--max-tokens` has to be generous, and how generous depends on the script. The
model's vocabulary covers some writing systems far better than others: English
costs about a quarter of a token per character, while a script that falls back
to raw bytes can cost more than one token per character — five times the budget
for the same amount of text.

`score` in the output is the mean probability of the tokens the model chose. It
plays the part `paddle`'s character confidence plays, and behaves the same way:
near one where the reading is certain, sagging where the model is inventing.

## Text and boxes

The model returns the text of a block, and the result contract wants lines with
quadrangles. When the model returns as many lines as the detector found rows,
the two are matched up in order. When it does not — and often it does not,
because this model reflows text and joins words broken across a line break —
every line of that block gets the block's own rectangle instead. A coarse box is
reported rather than a made-up precise one.

## Device

```
--device <cpu|metal>       # default: metal where the build has it
```

This is nearly two gigabytes of weights driven by an autoregressive loop. On a
GPU that is tens of seconds a page; on a CPU it is minutes. The default is the
GPU wherever the binary was built with support for one, which is the opposite of
the rest of trakktor and deliberate: a silent fall back to the CPU here would
look like a hang.

Half precision is used on the GPU and full precision on the CPU. The weights are
published in bfloat16 and the three precisions were measured to produce the same
text, so the choice costs nothing but memory and speed.

## Runtime

```
--runtime <candle|burn>    # default: candle
```

Two independent implementations of the same networks, reading the pages the
same. `candle` is the default and the fast one. `burn` sits behind the `burn`
build feature and computes in f32 on either device — its half-precision Metal
backend cannot run this model yet — which on a GPU makes it measurably slower
than candle's f16; it is kept as a second, independently written implementation
to check the first against, not as a speed play. The detection stage runs on
candle either way. Of the two OCR engines only `vl` has a burn runtime;
`ocr paddle --runtime burn` is a validation error.

## Languages

There is no `--lang`. The model has one vocabulary for every writing system it
knows and decides for itself what it is looking at — on a page that mixes two
scripts it reads both. A flag that changed nothing would promise control that
does not exist.
