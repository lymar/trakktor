# `ocr` — read the text off page images

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

Text recognition over page images: scans, photographs of pages, screenshots.
One engine ships today — [`paddle`](paddle.md), a port of the PP-OCRv5 pipeline —
and it runs fully offline once its models are downloaded.

```sh
trakktor ocr paddle page.png                     # JSON: pages, lines, boxes, scores
trakktor ocr paddle page.png --text              # the recognized lines
trakktor ocr paddle p1.png p2.png p3.png         # one document, three pages
trakktor ocr paddle scan.png --lang en           # pick the recognizer by language
trakktor ocr paddle scan.png --format md --out doc.md
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
      ]
    }
  ],
  "language": "ru",
  "models": {
    "detection": "PP-OCRv5_mobile_det",
    "recognition": "eslav_PP-OCRv5_mobile_rec"
  }
}
```

- `quad` is four points in page pixels, clockwise from the top left. A line on a
  scan is almost never parallel to the edge, so a box is a quadrangle rather
  than a rectangle.
- `score` is the mean probability of the characters that were kept.
- `lines` are in reading order, not in the order the detector found them.

`--text` prints the lines, with the pages separated by a marker that is part of
the data (`=== page 2 · scan-02.png ===`) — without it the text of a multi-page
document could not be taken apart again.

`--format md` assembles Markdown instead: paragraphs rather than lines, a
reading order that follows columns, hyphenated line breaks joined back into
words, and page-to-page paragraph continuation. What it does **not** do yet is
tell a heading from body text by anything other than geometry, or find
illustrations — see [the engine page](paddle.md#what-markdown-does-and-does-not-do).

## Languages

`--lang <code>` picks the recognizer; `--lang list` prints every code with the
model it selects. Russian is the default.

A recognizer is trained on one script and can only ever emit characters from its
own dictionary. A page in a script the selected model does not cover comes back
empty or as nonsense **even though its lines were found** — the detector is
script-independent, the recognizer is not. There is no Tibetan recognizer in
this line of models at all.

## Checking a result

```sh
trakktor ocr paddle page.png --crops ./crops
```

writes one image per recognized line — the straightened crop the recognizer
actually read. If a line is missing from the output, the crops say whether the
detector never found it or the recognizer could not read it.

The other lever is `--limit-side-len`. It decides how far the page is scaled
down before detection, and it is the setting that decides whether small type is
found at all: see [the engine page](paddle.md#finding-small-type).
