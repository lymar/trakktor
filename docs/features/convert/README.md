# `convert` — turn a document into Markdown

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor convert pdf` reads the text that is already in a PDF and writes it out
as Markdown. A document made from a layout program carries every character in
the file, with the font code that draws it — nothing has to be recognized, and
the answer comes out letter for letter.

```sh
trakktor convert pdf paper.pdf                    # JSON: pages + assembled Markdown
trakktor convert pdf paper.pdf --text             # just the Markdown
trakktor convert pdf paper.pdf -o paper.md        # …and write it to a file
trakktor convert pdf book.pdf --pages 3-5,12      # only these pages
```

## Which command: this or `ocr`

**A PDF from a layout program is `convert pdf`. A PDF from a scanner is
[`ocr`](../ocr/README.md)** — which reads page *images*, not PDFs, so render
the pages to images first (for example `pdftoppm -png scan.pdf page`), then
hand them to `ocr` in order.

You do not have to know which one you have. `convert pdf` classifies every page
before it reads it, converts the pages that carry text, and names the pages that
do not. If that is *every* page, it fails with `no_text_layer` and says to use
`ocr` instead (on rendered page images, as above).

The difference is worth caring about. On a hundred-page typeset document this
takes a fraction of a second and reproduces the text exactly; recognizing the
same pages as pictures takes about ten seconds each and is only ever as accurate
as the recognizer. It also costs nothing to try: no model is downloaded, nothing
runs on a GPU, and a wrong guess comes back as an error in milliseconds.

## What comes out

Headings, paragraphs, lists, tables and the reading order of a multi-column page
are worked out from the document's own structure; running heads and page numbers
are dropped. With more than one page, each page opens with a `<!-- page N -->`
comment, and a page that did not convert leaves a comment of its own —
`<!-- page 7: no text layer (scanned) — read this page with `trakktor ocr` -->`
— so a hole in the document is visible to whoever reads the source and invisible
in the rendered text.

Page numbers are the document's own, counting from one. `--pages 3-5` returns
pages numbered 3, 4 and 5. A value is a page (`3`), a range (`1-5`), or a list
of both (`1-5,12,40-`); a range left open at the end runs to the last page,
and without `--pages` every page is converted. Page 0 does not exist and is
rejected; a range that merely *runs* past the end of the document is clamped
to it, but a selection that *starts* past the end is an error.

## A text layer can be there and still be wrong

"This page has text" and "this text is right" are different claims, and only the
first one is easy. Three ways a PDF hands over confident, non-empty, wrong text
turned up in testing. One is repaired, two are reported.

**Fonts that keep their encoding to themselves — repaired.** A document typeset
in TeX embeds fonts whose encoding lives inside the font program and writes
nothing about it into the font dictionary. Read naively, such a page loses its
ligatures (`first` becomes `rst`), turns en dashes into `{` and quotes into
`\` and `"`, and — worse, because it still reads as mathematics — quietly
mangles the operators: `≤` arrives as `6`, `−` and `·` vanish. On an ordinary
typeset paper that is around 2.5 % of the words. So before reading, trakktor
takes the encoding vector out of each embedded font program and writes it into
the font dictionary where the extractor will look for it. Nothing is invented:
what gets written is what the file already said, one indirection away. The
result carries `fonts_repaired` when this happened, and on such a paper the
encoding damage measures at zero afterwards.

**Text that maps nowhere — reported.** A font may carry a character map that
points into a Unicode private use area: the file says which glyph to draw, but
not which character it is. There is nothing to recover — every other reader gets
the same nothing — so the page still converts and an issue names it:

```json
{ "code": "private_use_text", "pages": [4, 5, 7], "message": "…" }
```

This is common in documents mixing a well-supported script with a
poorly-supported one: the first extracts perfectly, the second is lost. If you
need the lost part, render those pages to images and recognize them with
`ocr`.

**Pages that are pictures pretending to be text — reported.** A page can be
drawn entirely as vector outlines and still carry a real text watermark, so it
converts to a page whose whole content is the same few words as ninety others.
Those pages are reported as needing OCR rather than handed over as content.

One kind of damage is **not** handled: a legacy font that maps another script
onto Latin code points. The text comes out as ordinary letters spelling nothing,
and there is no general way to tell — a table specific to the font is the only
cure.

## Output

```sh
trakktor convert pdf paper.pdf --pretty
```

```json
{
  "source": "paper.pdf",
  "kind": "text",
  "page_count": 86,
  "pages": [
    { "number": 1, "markdown": "## TITLE\n\n…" },
    { "number": 2, "needs_ocr": true, "reason": "vector_text" }
  ],
  "markdown": "…the whole document…",
  "pages_needing_ocr": [2],
  "issues": [{ "code": "private_use_text", "pages": [4], "message": "…" }],
  "fonts_repaired": 21,
  "out": "paper.md"
}
```

`kind` is `text`, `mixed`, `scanned` or `image`. `page_count` counts the
document's pages, not the selection's. A page's `reason` is `scanned`,
`no_text`, `vector_text` or `no_unique_text`. `pages_needing_ocr`, `issues`,
`fonts_repaired` and `out` appear only when there is something to report; an
issue's `pages` is omitted when the damage is a property of the document
rather than of particular pages.

`--text` prints the assembled Markdown and nothing else; the warnings go to
stderr, so a document with half its pages missing still pipes as a document.
With `-o` the Markdown is also written to the file, but `--text` still prints
the Markdown itself, never the path — unlike `pdf cut`, whose `--text` prints
the path it wrote.

Errors, all exit 1: `invalid_input` (missing, unreadable, or not a PDF),
`parse_failed`, `encrypted` (pass `--password` — though a document locked
with only an *owner* password cannot be converted at all: there is no user
password to pass), `no_text_layer` (use `ocr`), `invalid_options` (a
`--pages` value that is malformed or that the document cannot honour),
`io_error`.

## Cost

Nothing is downloaded and no model runs. A hundred-page typeset document takes
around a tenth of a second; a large one with complex pages, a couple of seconds.
Memory tracks the size of the document.

## Credits

The PDF engine is [pdf-inspector](https://github.com/firecrawl/pdf-inspector)
(MIT) by Firecrawl — classification and Markdown extraction in pure Rust, with
no C dependency. It is used as a library, not ported. See
[`docs/acknowledgments.md`](../../acknowledgments.md) and
[`NOTICE`](../../../NOTICE).
