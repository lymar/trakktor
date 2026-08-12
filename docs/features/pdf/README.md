# `pdf` — edit a PDF as a document

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor pdf cut` cuts a page range out of a PDF and writes it as a new,
self-contained PDF — the chapter out of a book, one paper out of a proceedings
volume. Nothing is read, recognized or re-encoded: this command edits the
document itself, with pages as the unit.

```sh
trakktor pdf cut book.pdf --pages 588-613                  # → book.cut.pdf
trakktor pdf cut book.pdf --pages 10-20 --out chapter.pdf  # name the output
trakktor pdf cut scan.pdf --pages 1,3,5 --text             # prints the path
trakktor pdf cut locked.pdf --pages 2- --password s3cret   # written decrypted
```

No text comes out of this command. To read a PDF's text, use
[`convert pdf`](../convert/README.md); to recognize a scanned page, use
[`ocr`](../ocr/README.md); to cut pages out of the file itself, use this.

## What the new file carries

Everything the kept pages use, byte for byte: embedded fonts (no re-subsetting,
no guesswork — a font program survives exactly as it was), images, shared
resources, the pages' own annotations, and the document's metadata. Attributes
that pages inherit from the page tree — media box, rotation, resources —
keep working, because the tree is edited in place rather than rebuilt.
Resources used only by the removed pages are garbage-collected, which is why
cutting a chapter out of a large book usually shrinks the file by an order of
magnitude.

Pages come out in document order; a selection is a set, not a reordering
(`--pages 20,10` means pages 10 and 20, in that order). Keeping every page
(`--pages 1-`) is allowed: that is a cleaned, decrypted copy of the document.

`--pages` is required, counting from 1 as the document does: a page (`3`), a
range (`1-5`), or a list of both (`1-5,12,40-`), where a range left open at
the end runs to the last page. Page 0 does not exist and is rejected; a range
that merely *runs* past the end of the document is clamped to it, but a
selection that *starts* past the end is an error.

## What is dropped, and why it is named

A document-level structure that points into the removed pages cannot survive
the cut whole. Leaving it half-working — an outline whose entries lead
nowhere, page labels that number positions that have moved — would be worse
than losing it, so it is dropped and **named in the output**:

| `dropped` entry | What it was |
|---|---|
| `outlines` | The bookmark tree: an entry led to a removed page. |
| `page_labels` | User-facing page numbering (`iv`, `A-7`): labels number positions, and the positions moved. |
| `struct_tree` | The logical structure tree of a tagged PDF. |
| `open_action` | The "open at page N" action, when N was removed. |
| `form_fields` | Interactive form fields, dropped field by field — a field with a widget on a kept page survives. |
| `named_destinations` | Named link targets that led to removed pages — filtered name by name, the rest survive. |

A full copy drops nothing. A link on a *kept* page whose target page was
removed stays a well-formed link that leads nowhere — the same dead end every
page cutter leaves.

## Encrypted documents

`--password` opens a document encrypted with a *user* password (RC4, AES-128,
AES-256). The result is always written without encryption — the point of
cutting pages is to use them. A document that only carries an *owner* password
(the kind that restricts printing and copying but opens everywhere) is opened
and decrypted silently, without a flag.

## Output

```sh
trakktor pdf cut manual.pdf --pages 4-6 --out manual-ops.pdf --pretty
```

```json
{
  "source": "manual.pdf",
  "page_count": 7,
  "pages": [4, 5, 6],
  "out": "manual-ops.pdf",
  "dropped": ["outlines", "page_labels", "open_action", "named_destinations"]
}
```

`page_count` counts the *source* document's pages; `pages` are the kept pages
by their source numbers, in order — page N of the result is the Nth entry.
`dropped` appears only when something was dropped. Without `--out`, the result
is `<input name>.cut.pdf` in the current directory.

`--text` prints the written path and nothing else, ready to hand to the next
command; the summary (`kept 3 of 7 pages; dropped: …`) goes to stderr either
way.

Errors, all exit 1: `invalid_input` (missing, unreadable, or not a PDF),
`parse_failed`, `encrypted` (pass `--password`), `invalid_options` (a
`--pages` value that is malformed or that the document cannot honour),
`io_error`.

## Cost

Nothing is downloaded and no model runs. Cutting 26 pages out of a
669-page, 35 MB volume takes a few seconds, and the result is 1.2 MB; memory
tracks the size of the document.

## Credits

The PDF engine is [lopdf](https://github.com/J-F-Liu/lopdf) (MIT) — parsing,
editing and writing PDF documents in pure Rust. It is used as a library, not
ported; the same parser also backs the font repair of
[`convert pdf`](../convert/README.md). See
[`docs/acknowledgments.md`](../../acknowledgments.md) and
[`NOTICE`](../../../NOTICE).
