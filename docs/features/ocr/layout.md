# The layout stage — what is on the page

> Part of [`ocr`](README.md); the shared page model and output shape are
> described there.

A model that looks at a page and returns the **blocks it is made of**, each
with a label — document title, section heading (`paragraph_title`), paragraph,
abstract, footnote, running head (`header`), page number (`number`), table,
formula, picture (`image`), caption (`figure_title`), stamp (`seal`), chart,
among the twenty classes the model tells apart ([the full list
below](#the-twenty-labels)). It reads no text. This is the answer to *what is
on this page*, not *what does it say*.

Both OCR engines run it by default, because it is what turns a list of
recognized lines into a document. It is also available on its own:

```sh
trakktor ocr layout page.png              # JSON: blocks, labels, scores, boxes
trakktor ocr layout page.png --text       # one block per line
trakktor ocr layout p1.png p2.png         # several pages in one run
trakktor ocr layout page.png --crops ./blocks   # each block as its own image
trakktor ocr layout page.png --boxes boxes.png  # the page, with the blocks drawn on it
```

About 130 MB downloaded once, and about a second and a half a page on the
default runtime (see the table below).

## Why it exists

The Markdown assembly can work out a great deal from geometry alone: the reading
order, the columns, where a paragraph starts and ends, how to join a word broken
across a line. What geometry cannot do is say what a block *is* — and the place
it fails is the place it hurts most:

- **A title is often the smallest box on the page.** Set in capitals, it has no
  descenders, so it measures *shorter* than the body text below it. "Find the
  heading by its size" cannot work.
- **A running head and a first heading look identical** on a single page. Only
  their repetition across a document tells them apart.
- **A table looks like a paragraph** — lines of similar height, close together.
- **An illustration has to be guessed from ink** that no text box covers, which
  also finds large tables, display formulas, coloured banners and calligraphy.
- **A line can cross from one block into the next.** Where two columns are set
  close together, the line detector — which works from ink, not from structure —
  joins the end of a line in one column to the line facing it in the other. One
  box covers both, and it is read as two sentences run together.

With labels these stop being guesses. Headings become `#` levels by what they
are; running heads, page numbers and footnotes are dropped or set apart by
label; pictures are pictures. And a line that crossed a boundary between two
blocks is cut at it, each half read on its own.

## What it changes, measured

On the same pages, with the stage and with `--no-layout`:

- an article's title page: the author line no longer becomes a second-level
  heading, and the numbered section heading — which was not recognized as a
  heading at all — becomes one;
- a page whose text wraps around an illustration: six consecutive one-line
  paragraphs become one paragraph, and the caption under the picture is set as a
  caption;
- the abstract is one paragraph instead of two;
- the page number is dropped by label rather than by position;
- a two-column page set with a narrow gutter: the lines the detector had glued
  across it are cut at the boundary and each half is read on its own, so neither
  column's sentence lands in the middle of the other's paragraph. The line is
  read both ways and the cut is taken only if every piece came back, so the page
  cannot lose a character to it — on the page this was measured on, not one
  non-blank character changed.

For [`ocr vl`](vl.md) it does something else as well: the blocks handed to the
generative model are cut **along** the structure rather than across it, so a
table goes to the model whole and is asked for as a table. On a calendar page
that meant seven blocks instead of sixteen, cell markup instead of doubled
columns, and 21 seconds instead of 89.

The glued line matters more there, too. In the classic engine it puts one
column's sentence in the middle of the other's paragraph; in the generative one
it drags both columns into a single block, and the block is the picture the
model is shown. The same boundary takes it apart, one stage later — after the
detector's boxes have been joined into lines, which is where that engine would
otherwise glue the pieces back together.

## What comes out

```json
{
  "pages": [
    {
      "number": 1,
      "source": "page.png",
      "width": 2481,
      "height": 3508,
      "blocks": [
        {
          "label": "doc_title",
          "score": 0.898,
          "quad": [[775, 614], [1766, 614], [1766, 659], [775, 659]]
        }
      ]
    }
  ],
  "models": { "layout": "PP-DocLayout_plus-L" }
}
```

Blocks come back in the order the model ranked them — by confidence. Reading
order is a property of a page that has been *read*, and belongs to the engines.

Inside an engine the same blocks appear per page, in reading order, with the
lines each one caught:

```json
{
  "kind": "heading",
  "label": "doc_title",
  "score": 0.898,
  "level": 1,
  "lines": [0],
  "quad": [[775, 614], [1766, 614], [1766, 659], [775, 659]]
}
```

`label` is what the model said; `kind` is what the assembly made of it, and is
there even for a block the model never claimed. A block with no `label` is one
the geometry worked out on its own.

`kind` takes six values: `figure` — an illustration; `heading` — a heading,
carrying its `level`; `paragraph` — running text; `footnote` — a footnote;
`furniture` — a running head or a page number; `caption` — a caption.

## Its limits, and why they show up as a threshold

The model is trained on Chinese and English documents, and **its confidence
tracks how familiar the writing system is**: around 0.98 for a paragraph of
English, 0.40 to 0.69 for the same paragraph in a Tibetan book. The blocks are
in the right places either way — what changes is how many of them clear the
score floor that keeps them.

That makes `--threshold` unusually meaningful on pages in unfamiliar scripts.
Lower it to see what the model nearly said; raise it to keep only what it is
sure of. The engines start from the same per-label defaults, chosen to keep
those pages working, and take `--layout-threshold` to override them.

Lowering the floor is not free in one direction: a picture box covering the
whole sheet passes a low floor too, and a picture swallows the blocks inside it
— so a page can come back with fewer labels than the defaults gave it, or with
none. `--boxes` shows that in one picture, which is why a value is worth trying
here before it is carried into a page of reading.

Two guarantees hold whatever the model says:

- **a line that fell in no block is never dropped** — it is assembled by
  geometry exactly as it would have been without the stage;
- **a page the model says nothing about reads exactly as `--no-layout`** would
  read it.

## The twenty labels

Every string `label` can hold, in the model's own class order:

| label | what it marks |
|---|---|
| `paragraph_title` | a section heading |
| `image` | a picture |
| `text` | a paragraph of body text |
| `number` | a page number |
| `abstract` | an abstract |
| `content` | a table of contents |
| `figure_title` | a caption |
| `formula` | a display formula |
| `table` | a table |
| `reference` | a bibliography |
| `doc_title` | the document title |
| `footnote` | a footnote |
| `header` | a running head |
| `algorithm` | pseudocode, set as a listing |
| `footer` | a running foot |
| `seal` | a stamp or seal |
| `chart` | a chart |
| `formula_number` | the number beside a display formula |
| `aside_text` | text set outside the main flow, in a margin or sidebar |
| `reference_content` | the entries of a bibliography |

## Options

```
--model <name|dir>      the layout model to run
--threshold <p>         one score floor for every label, replacing the defaults
--crops <dir>           write each block as its own image
--boxes <file|dir>      write the page with every block outlined and labelled
--runtime <candle|burn> the inference runtime; `burn` needs a build with it
--device <cpu|metal>
```

`--boxes` is the fastest way to see what the model made of a page: every block
is drawn where it sits, numbered as the result numbers it and captioned with its
label, in a second colour where the score is below half.

One thing this command does *not* take is the [photo pipeline](photo.md) —
there is no `--doc-orientation`, `--sheet` or `--unwarp` here. On a photograph
the preview therefore sees the page as it was shot, unstraightened, while an
engine run given those flags marks up the straightened page.

Inside `ocr paddle` and `ocr vl` the stage takes `--no-layout` to skip it,
`--layout-model` to choose the model, and `--layout-threshold` for the same
score floor as `--threshold` here. The longer name keeps it apart from
`--thresh` and `--box-thresh`, which are the line detector's thresholds and a
different stage.

## Runtimes

Both runtimes produce identical blocks, in f32 on either device. Per page, warm,
including loading the weights:

| | seconds |
|---|---:|
| `burn`, Metal | 0.9 |
| `candle`, CPU | 1.4 |
| `candle`, Metal | 1.9 |
| `burn`, CPU | 10.0 |

The default is `candle` on the CPU. Two of those numbers are worth knowing:
candle on Metal is *slower* than on the CPU here — the network is hundreds of
small convolution kernels, and launching them costs more than running them —
and burn on Metal is the fastest of the four. A first burn/Metal run costs about
22 seconds while its kernels are tuned; the result is cached on disk.
