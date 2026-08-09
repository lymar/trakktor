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
| download | ~139 MB, or 13 MB at `--quality fast` | **~2 GB**, once |
| a page | some ten seconds, less at `--quality fast` | tens of seconds |
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

**The detector here is the large one** (`PP-OCRv5_server_det`, 88 MB —
`paddle` has since moved its default to the newer `PP-OCRv6_medium_det`; this
engine keeps the v5) — and there is no `--quality fast` here to trade it back
for. A line
the small detector misses is not merely missing from the result: it also changes
how the lines around it are grouped, so a footnote can come back as a fragment
with its opening gone. The saving would be about three seconds on a page that
already takes thirty, which is not a speed anybody came here for. Pass
`--det-model PP-OCRv5_mobile_det` if you want it anyway; `--model` does the
same for the VL checkpoint itself — a published name or a local directory.

One caveat comes with it, and it applies here as much as in `paddle`: the large
detector's probability map is sharper, so an ordinary line can score just under
the default `--box-thresh 0.6` and be dropped before the blocks are cut. If a
block comes back short of a line, try `--box-thresh 0.4` — knowing its price:
weak boxes over decorative ink also make it in, shift the blocks around them,
and can pull hallucinated lines into the result. Measured across a page set,
0.4 changed nothing on most pages, recovered a dropped line on one, and traded
lines for hallucinations on the ornate ones — which is why it is a flag and not
the default. The detector's other two knobs, `--thresh` and `--unclip-ratio`,
exist here as well, at the v5 defaults (0.3 and 1.5) —
[the `paddle` page](paddle.md) explains all three.

Two consequences worth knowing:

- **Three models are downloaded**: the 1.92 GB one, the 88 MB detector, and the
  130 MB layout model below — about 2.1 GB in all.
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

The regions also keep a block out of the next column. Where two columns are set
close together the detector joins a line of one to the line facing it in the
other, and that single box drags both columns into one block — which is exactly
the input blocks exist to avoid. The boundary between two regions cuts that row
apart again, and each column is read on its own. On a two-column page set with a
narrow gutter this removed every line that ran the two columns together, and
what the page lost was duplicated text rather than text: measured against the
page's own words, the reading went from 29 missing and 101 surplus to 9 and 34.

`--no-layout` goes back to grouping by geometry alone, and the knobs below are
what shape it then. They still apply to the lines the layout model did not
claim.

The stage itself takes two more:

```
--layout-model <name|dir>  # which layout model to run
--layout-threshold <p>     # one score floor for every label, over the defaults
```

The floor matters more here than in the classic engine, because a labelled
block *is* the unit this engine reads. On a page in an unfamiliar script the
model is much less sure of itself — a paragraph can score below the default —
and the page falls back to the geometry above. Lowering the floor gives the
labels back, but not for free: a picture box covering the whole sheet passes a
low floor too, and a picture swallows the blocks inside it — and a picture is
not read at all. [`ocr layout --boxes`](layout.md) shows what a value does to a
page for the price of a second.

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

`--boxes <file|dir>` draws the same thing on the page rather than beside it:
each block outlined and numbered as its crop is, over the thin outlines of the
lines the detector found. Those two layers are the two things that decide a
reading here — what was detected, and how it was grouped — and a block whose
reading was dropped is marked `DROPPED` on the page instead of quietly leaving a
hole in the text.

A block can also come out as a shape the model cannot be shown at all: the
picture processor refuses one more than two hundred times wider than it is tall,
because a hairline is not a picture of anything. Such a block is skipped and
marked the same way. It costs its own text and never the page's — the rest of
the sheet's blocks have nothing to do with it.

## Tasks

```
--task <ocr|table|formula|chart>     # default: ocr
```

`ocr` reads the text and is the only one that works block by block. The other
three are asked of the **page as a whole**, because a table or a formula is
itself one region and cutting it up destroys the structure that made the
question worth asking. What each returns: a table as markup and a formula as
LaTeX (both below); `chart` has the model describe what the chart shows, and
the description is passed through as the model wrote it.

### What a table comes back as

A table is answered with cell markup rather than with text — one tag per cell,
one per row, and separate tags for a cell merged with the one to its left, the
one above, or both. A firing log with two entries, each standing against three
stages, comes back like this — the head, then the first of its two entries:

```
<fcel>Firing<fcel>Stage<fcel>Hours<nl><fcel>First<fcel>Warming<fcel>6<nl><ucel><fcel>Soaking<fcel>2<nl><ucel><fcel>Cooling<fcel>9<nl>…
```

`--format lines` prints that as it stands. `--format md` turns it into a table
of one of two shapes:

- a **pipe table**, while nothing in the table is merged. That is most tables,
  and ordinary Markdown reads best.
- an **HTML `<table>`** with `rowspan`/`colspan`, as soon as one cell covers two
  rows or two columns. A pipe table has neither, so a merged cell could only
  come out as a blank next to the cell that carries the text — and a blank is
  what the markup already means by its own empty-cell tag. Markdown accepts
  inline HTML, so the structure survives:

```html
<table>
<tr><td>Firing</td><td>Stage</td><td>Hours</td></tr>
<tr><td rowspan="3">First</td><td>Warming</td><td>6</td></tr>
<tr><td>Soaking</td><td>2</td></tr>
<tr><td>Cooling</td><td>9</td></tr>
</table>
```

No header row is invented in the HTML shape: every cell is a `<td>`. The pipe
table has to name one — the format rules a line under the first row — and on a
table whose column heads take two rows that guess is simply wrong.

Two things are worth knowing about the markup itself. The model does not always
report a merge: the same table can come back with the row head against the first
of its rows and a merge tag under it, or with the head against the row it is
*printed* against and blanks around it, which is a merge already lost before the
Markdown is assembled. And a table cut off by `--max-tokens` ends wherever the
ceiling fell — possibly inside a number — so the Markdown marks it with a
`<!-- table cut short -->` comment, which is invisible in a rendered document
and plain in the source.

## Stopping

A generative reader does not fail by producing a wrong letter; it fails by not
stopping — one syllable, or one short phrase, repeated until the budget runs
out. So the loop watches for that explicitly and cuts the answer where the
repetition began. A line cut that way — or at the `--max-tokens` ceiling —
carries `"truncated": true` in the output.

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

## A photograph rather than a scan

This engine takes the shared [page preprocessing](photo.md) too —
`--doc-orientation --sheet --unwarp`, off by default. It matters here for the
same reason `--limit-side-len` does (its default here is 1440, against
`paddle`'s 960): this engine reads **blocks**, and blocks
are assembled from the detector's lines. A page that is not upright does not
merely read badly, it is grouped into blocks that were never on the page.

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
same and at the same precision — f16 on Metal, f32 on the CPU. `candle` is the
default. `burn` sits behind the `burn` build feature and runs its Metal
backend with tensor-op fusion turned off — fused, the f16 kernels break on
this model's mixed-precision chains — and is kept as a second, independently
written implementation to check the first against, not as a speed play. The
detection stage runs on candle either way. Of the two OCR engines only `vl`
has a burn runtime; `ocr paddle --runtime burn` is a validation error.

## Languages

There is no `--lang`. The model has one vocabulary for every writing system it
knows and decides for itself what it is looking at — on a page that mixes two
scripts it reads both. A flag that changed nothing would promise control that
does not exist.
