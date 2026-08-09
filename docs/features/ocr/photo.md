# Reading a page you photographed

> Part of [`ocr`](README.md); the shared page model and output shape are
> described there.

Three steps that run before the detector and turn a photograph of a page into
something a scanner could have produced. All three are off by default and both
engines take them:

```sh
trakktor ocr paddle photo.jpg --doc-orientation --sheet --unwarp
trakktor ocr vl     photo.jpg --doc-orientation --sheet --unwarp
```

| Flag | What it does | Cost |
|---|---|---|
| `--doc-orientation` | Finds which of the four right angles the page is at and turns it upright. | 7 MB, ~0.15 s a page |
| `--sheet` | Finds the sheet in the frame and cuts it out — the desk, your fingers and the facing page stay outside. | no model, ~0.3 s a page |
| `--unwarp` | Straightens the page: the perspective of a shot taken at an angle and the curve of a page that will not lie flat. | 32 MB, ~0.7 s a page |
| `--rectified <file\|dir>` | Writes the straightened page itself, so it can be looked at. | — |

**Use all three together.** They are separate flags because they answer separate
questions, not because they are meant to be picked between: `--unwarp` on its
own can make a reading *worse* (see [below](#why-sheet-exists)), and `--sheet`
is what stops it.

They are off by default because on a scan or a rendered PDF page they do
nothing — the page is already upright and already flat — and doing nothing
costs two models and a second a page.

## What it is worth

Measured on twelve photographs of one typeset page, against a flat 300 dpi
render of the same PDF. The number is the character error rate; the render's
own rate, **0.0069**, is the floor — the recognizer has no bullet in its
dictionary, and three of them cost twelve characters out of 1751. "Reads like a
scan" means exactly 0.0069.

| The photograph | plain | with all three |
|---|---:|---:|
| straight on, or 20° off | 0.0069 | 0.0069 |
| held 12° off level | 0.0091 | 0.0069 |
| lying on its side | 0.9023 | **0.0069** |
| upside down | 0.8875 | **0.0069** |
| curled, one edge lifted | 0.3335 | **0.0069** |
| held at the edge by two fingers | 0.2850 | **0.0080** |
| small in a cluttered frame | 0.0069 | 0.0069 |
| 40° off, page running out of frame | 0.0091 | 0.0103 |
| lamp reflected off the paper | 0.0640 | 0.0640 |

Two things in that table are worth taking away.

**Perspective alone is not the problem.** A page shot 20° off the normal reads
as well as its scan with nothing turned on, and at 40° it is four characters
behind. The detector finds each line as its own quadrangle and straightens it
independently, so a page-shaped trapezoid does not trouble it. What breaks a
reading is **curvature** — a curved line stops being a quadrangle — and
**orientation**, which nothing else recovers from.

**Uneven light is not handled at all.** A lamp reflected off the paper costs
nine times the floor, and none of the three steps moves it. A hand's shadow, on
the other hand, the detector takes in its stride. If a photograph reads badly
and it is not sideways and not curved, look at the light before reaching for
these flags.

## Where the boxes are

**On the file you passed in.** The reading happens on the straightened page, but
every quadrangle in the result is carried back onto your own photograph, and
`width`/`height` are your file's.

That is worth stating because it is not free and not obvious. A page shot on its
side is read upright, and yet:

```console
$ trakktor ocr paddle sideways.jpg --doc-orientation --sheet --unwarp
{"pages": [{"source": "sideways.jpg", "width": 3072, "height": 2304,
            "lines": [{"text": "3  Keeping the Lamp",
                       "quad": [[407,1776],[435,1162],[495,1161],[466,1776]]}]}]}
```

The text came out upright; the box is a *tall* quadrangle on a *landscape*
frame, because that is where those words are in the file on disk. Everything
geometric — columns, reading order, Markdown assembly, cutting out lines and
illustrations — happens on the straightened page first; the journey back
happens only when the result is written.

This works because neither transformation had to be inverted numerically. The
sheet cut is a perspective warp, and a homography inverts in closed form; the
straightener produces a **backward** map — for each pixel of the straightened
page, where it came from — which already runs in the direction the answer is
wanted.

A page that needed none of this pays none of it: with nothing to undo, there is
no journey back.

## Why `--sheet` exists

The straightener works on the **frame**, not on the sheet in it. It shrinks the
whole picture into a fixed 712×488 box and decides the page's shape from that.
So a page that fills its photograph comes back straight, and a page lying on a
desk at the far end of a wide shot reaches the network as a third of that box
and comes back no straighter than it went in — and resampled, which is worse
than not straightening at all:

| The photograph | plain | `--unwarp` only | `--sheet --unwarp` |
|---|---:|---:|---:|
| small in a cluttered frame | 0.0069 | 0.1491 | 0.0069 |
| held at the edge by two fingers | 0.2850 | 0.1616 | 0.0080 |
| 40° off | 0.0091 | 0.3838 | 0.0103 |

Cutting the sheet out first hands the straightener a picture the page fills.
It needs no model: a threshold, the largest bright region of the frame, its four
sides, and the same perspective warp that already straightens every line of
text.

It declines when there is nothing to do. A frame whose bright region is nearly
the whole picture has no sheet to cut out — that is a scan — and a bright sliver
is not a sheet either. In both cases the page goes on untouched, which is what
makes the flag safe to leave on.

## Looking at what happened

```sh
trakktor ocr paddle photo.jpg --doc-orientation --sheet --unwarp \
    --rectified straightened.png
```

`--rectified` writes the page the reading was actually done on — a file for one
page, a directory for several, and nothing at all when the steps found nothing
to change. It is the third checking flag next to
[`--crops` and `--boxes`](README.md#checking-a-result): those say *what* was
read and *where* it was, this one says *what it was read from*.

If a photograph reads badly, that is the picture to look at first. A page with
its edges clipped, or one still visibly bent, says the sheet was not found; a
page that looks right says to look at the light and at
[`--limit-side-len`](paddle.md#finding-small-type) instead.

## What is not handled

- **Uneven light.** Shadows and reflections are left exactly as they were; see
  the table above for what a reflection costs.
- **A spread is still one page.** Photograph an open book and you get one page
  of output covering both halves, with the reading order running across the
  gutter. Splitting a spread into two pages would change what a "page" is, and
  that has not been done.
- **Anything but a right angle.** `--doc-orientation` chooses among 0°, 90°,
  180° and 270°. A page a few degrees off level is `--unwarp`'s business, and a
  page at 45° is nobody's.

## Where the models come from

Both are ports of the document pre-processor in
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): `PP-LCNet_x1_0_doc_ori`
for the orientation, [`UVDoc`](https://github.com/tanguymagne/UVDoc) for the
straightening. They are downloaded on first use and verified against a pinned
revision. Finding the sheet is trakktor's own and needs nothing downloaded.
See [Acknowledgments](../../acknowledgments.md).
