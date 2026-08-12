# `ocr` — read the text off page images

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

Text recognition over page images: scans, photographs of pages, screenshots.
Two engines, both fully offline once their models are downloaded.

**A PDF made from a layout program does not belong here** — its text is already
in the file, and [`convert pdf`](../convert/README.md) gets it out exactly and
about a thousand times faster. `ocr` reads page *images*: a scanned PDF is
rendered to images first (for example `pdftoppm -png scan.pdf page`), and its
pages passed in order. If you are unsure which kind of PDF you have, run
`convert pdf` — it says so, and names the pages that do need recognizing.

- [**`paddle`**](paddle.md) — a port of PaddleOCR's classic pipeline: a detector
  finds the lines, a recognizer reads each one. It reads with the best models it
  has for the language: about 139 MB — plus 130 MB for the layout stage below —
  and some ten seconds a page, or 13 MB and a quarter to a third less time
  under `--quality fast`. **The default choice.**
- [**`vl`**](vl.md) — a port of the PaddleOCR-VL document model, which writes
  out what it sees rather than picking characters from a dictionary. It works
  out the writing system itself, reads scripts `paddle` has no model for, and
  can return a table as markup or a formula as LaTeX. About 2 GB downloaded
  once, and tens of seconds per page.

Two stages sit around the engines and are shared by both:

- the [**layout stage**](layout.md) runs **by default**: a model that labels
  the blocks of a page — document title, section heading, paragraph, abstract,
  footnote, running head, page number, table, formula, picture, caption and
  more — without reading any text, for about 130 MB downloaded once and about
  a second and a half per page. `--no-layout` skips it, and `ocr layout` runs
  it alone;
- the [**page preprocessing**](photo.md) is **off by default** and is what a
  page you *photographed* needs: `--doc-orientation` turns a page shot sideways
  the right way up, `--sheet` cuts the sheet out of the frame, and `--unwarp`
  straightens it.

```sh
trakktor ocr paddle page.png                     # JSON: pages, lines, boxes, scores
trakktor ocr paddle page.png --text              # the recognized lines
trakktor ocr paddle p1.png p2.png p3.png         # one document, three pages
trakktor ocr paddle scan.png --lang en           # pick the recognizer by language
trakktor ocr paddle scan.png --format md --out doc.md
trakktor ocr paddle scan.png --format md --no-layout --out doc.md  # skip the labels

trakktor ocr vl scan.png --text                  # no --lang: the model works it out
trakktor ocr vl table.png --task table --text    # the table as markup
trakktor ocr vl page.png --no-layout --text      # blocks from geometry alone

trakktor ocr layout page.png --text              # what is on this page

trakktor ocr paddle photo.jpg --doc-orientation --sheet --unwarp  # a page you photographed
```

## Pages, not files

**The input is a sequence of pages, even when it is one file.** Every argument
contributes one page (a multi-frame TIFF contributes its first frame today),
pages are numbered from 1 in argument order, and each remembers the file it came
from. So the order of the arguments is the order of the document: three page
images passed together are one three-page document, not three runs — which is
what lets a paragraph that runs over a page break be joined back together in the
Markdown output.

## What comes out

JSON (the default) is one object for the run:

```json
{
  "pages": [
    {
      "number": 1,
      "source": "page.png",
      "width": 1240,
      "height": 900,
      "text": "…the page's lines…",
      "lines": [
        {
          "text": "The first line",
          "score": 0.983,
          "quad": [[68, 66], [668, 66], [668, 107], [68, 107]],
          "rotated": false
        }
      ],
      "blocks": [
        {
          "kind": "heading",
          "label": "doc_title",
          "score": 0.921,
          "level": 1,
          "lines": [0],
          "quad": [[68, 60], [668, 60], [668, 110], [68, 110]]
        }
      ]
    }
  ],
  "language": "ru",
  "models": {
    "detection": "PP-OCRv6_medium_det",
    "recognition": "eslav_PP-OCRv5_mobile_rec",
    "layout": "PP-DocLayout_plus-L"
  }
}
```

- `quad` is four points in page pixels, clockwise from the top left. A line on a
  scan is almost never parallel to the edge, so a box is a quadrangle rather
  than a rectangle.
- `score` is the mean probability of the characters that were kept.
- `lines` are in reading order, not in the order the detector found them.
- `models` names what actually ran, and the two are picked separately: a Russian
  page is found by the newest detector and read by an older recognizer, because
  the newest generation carries no Cyrillic. `--quality` and `--lang` move them,
  so this field is how to tell which reading a result came from.
- `blocks` are the layout stage's regions, each with its own `quad` and the
  lines it caught ([the stage's page](layout.md) explains `kind` against
  `label`). With `--no-layout` the shape is the same minus `blocks` and
  `models.layout`. `--format md` adds a top-level `markdown` field; `--out`
  writes it to a file instead (reported as `out`), with any illustrations
  saved to an `imgs` directory next to it and linked from the Markdown.

`--text` prints the lines, with the pages separated by a marker that is part of
the data (`=== page 2 · scan-02.png ===`) — without it the text of a multi-page
document could not be taken apart again.

`--format md` assembles Markdown instead: paragraphs rather than lines, a
reading order that follows columns, hyphenated line breaks joined back into
words, and page-to-page paragraph continuation. A table `vl` returned as cell
markup becomes a table there as well — a pipe table, or an HTML one when a cell
spans rows or columns and a pipe table could not say so ([the engine
page](vl.md#what-a-table-comes-back-as)). With the layout stage on (the
default) the labels supply the heading levels and the pictures; with
`--no-layout` the assembly falls back to geometry alone — the details:
[the engine page](paddle.md#what-markdown-does-and-does-not-do).

## Languages

The two engines answer this question differently, and it is the main reason to
pick one over the other.

`paddle` takes `--lang <code>`, which picks the recognizer; `--lang list` prints
every code with the model it selects, and English is the default. A recognizer
is trained on one script and can only ever emit characters from its own
dictionary. A page in a script the selected model does not cover comes back
empty or as nonsense **even though its lines were found** — the detector is
script-independent, the recognizer is not. Some writing systems have no
recognizer in this line of models at all.

`vl` has no `--lang`: one model covers every writing system it knows and decides
for itself what it is looking at, so a page that mixes two scripts is read as
one page. That is what it is for.

## Checking a result

```sh
trakktor ocr paddle page.png --crops ./crops     # what was read
trakktor ocr paddle page.png --boxes boxes.png   # where it was, and in what order
```

`--crops` writes one image per recognized line — the straightened crop the
recognizer actually read. `--boxes` writes the page itself with every reported
line outlined and numbered the way the result numbers it, in a second colour
where the reading was less than half sure. Between them they say which stage
went wrong: no outline means the detector never found the line, an outline whose
crop is blank means the recognizer could not read it.

Both flags work on all three subcommands, each with its own unit. For `vl`,
`--crops` writes out **blocks** rather than lines — the regions the model was
asked to make sense of (see [its page](vl.md#it-reads-blocks-not-lines)) — and
`--boxes` draws two layers: those blocks, numbered as their crops are, over the
thin outlines of the lines they were assembled from. For `ocr layout`, `--boxes`
labels every region as well as numbering it, which is the quickest way to see
what the model thinks is a table.

`--boxes` takes the file to write for a single page, and a directory to fill
with one `pNNN.png` per page for a longer run.

## A page you photographed

A photograph of a page differs from a scan in five ways, and three of them can
be taken out before the detector sees it — the page being sideways, the sheet
being small in a cluttered frame, and the page not being flat:

```sh
trakktor ocr paddle photo.jpg --doc-orientation --sheet --unwarp
trakktor ocr paddle photo.jpg --doc-orientation --sheet --unwarp --rectified out.png
```

All three are off by default, both engines take them, and the reported boxes
stay on **your** file however much the page was moved to read it. A photograph
of a page lying on its side goes from unreadable to reading exactly as well as
a scan of the same page. The two things this does *not* fix are uneven light and
a two-page spread. Details, numbers and the reasons: [reading a page you
photographed](photo.md).

## Finding small type

When a line is missing from the result and neither `--crops` nor `--boxes`
shows it at all, the lever is `--limit-side-len`. It decides how far the page is
scaled down before detection, and with it whether small type is found at all:
see [the engine page](paddle.md#finding-small-type). Both engines have it —
with different defaults, 960 for `paddle` and 1440 for `vl` — and
it matters to `vl` twice over — a line the detector misses there is not merely
unreported, it is a block the model never sees.
