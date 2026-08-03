//! Assembling detected lines into the blocks the model is fed.
//!
//! The unit this engine reads is a **block**, and the reason is measured, not
//! aesthetic: handed a whole page the model collapses into a repeat loop,
//! handed a single mixed-script strip it hallucinates entire alphabets, and
//! handed one coherent block it reads Tibetan at about one per cent character
//! error. More context is better right up to the point where it is much worse,
//! and the block is where that curve turns.
//!
//! It is built in two passes, and the first one is not optional. The detector
//! does not return one box per line of text: a line with wide word spaces, or
//! with a change of script in the middle, comes back as **several** boxes side
//! by side. Grouping straight into blocks without joining those first cuts
//! lines in half and hands the model a fragment — which is the input it is
//! worst at, and the failure looks like a hallucination rather than like a
//! cropping mistake.
//!
//! So: boxes are joined into **rows** first, then rows into blocks.
//!
//! Both passes see the detector's reading order — top to bottom, left to right
//! within a row — which on a two-column page interleaves the columns. So blocks
//! are not accumulated one at a time: several stay **open** at once and each
//! row joins the open block it fits, which is what keeps two columns apart
//! without a separate column-finding pass.

#[cfg(test)]
mod tests;

use image::RgbImage;

use crate::ocr::page::Quad;

/// A run of lines the model reads in one go.
#[derive(Debug, Clone)]
pub struct Block {
    /// Indices into the line list this block was built from, in reading order.
    pub lines: Vec<usize>,
    /// The rectangle to cut, already padded and clamped to the page.
    pub rect: Rect,
    /// The page rows this block owns, as `(top, bottom)` in page pixels.
    ///
    /// A block's rectangle is a window onto the page, and a window catches
    /// whatever else falls inside it: the foot of the line above, the head of
    /// the line below, and — where this block is narrower than its neighbours
    /// — the middle of a line whose ends are outside the frame. The model
    /// reads those fragments and reports them as if they were text, which
    /// shows up as a half-line duplicated between two blocks. So
    /// everything outside these rows is painted out when the block is cut.
    bands: Vec<(u32, u32)>,
}

/// An axis-aligned rectangle in page pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// The rectangle as a quadrangle, for the result envelope.
    pub fn quad(&self) -> Quad {
        let (x0, y0) = (self.x as f32, self.y as f32);
        let (x1, y1) =
            ((self.x + self.width) as f32, (self.y + self.height) as f32);
        Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }
}

/// The thresholds of the assembly, all relative to the line height so that a
/// thumbnail and a 300-dpi scan behave alike.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// × line height: the vertical gap that ends a block.
    pub gap: f32,
    /// Fraction of the narrower span two boxes must share horizontally to
    /// belong together.
    pub overlap: f32,
    /// How far a line's height may differ from the block's, in either
    /// direction. Keeps a heading off the front of a paragraph and a footnote
    /// off its back.
    pub height: f32,
    /// Fraction of the shorter box two boxes must share vertically to be
    /// pieces of one row.
    pub row_overlap: f32,
    /// × line height: the widest horizontal gap that still joins two pieces of
    /// one row. It is also, by construction, the narrowest column gutter — the
    /// two questions are the same question asked from opposite sides.
    pub row_gap: f32,
    /// Most rows one block may hold. The answer grows with the block, and so
    /// does the chance of the model losing its place in it.
    pub max_lines: usize,
    /// × line height: how far the cut is grown beyond the boxes.
    ///
    /// The detector marks a shrunken core of each line. Without a margin the
    /// ascenders and descenders are shaved off — and in a stacking script
    /// (Tibetan, Devanagari) whole tiers of vowel signs and subjoined letters
    /// go with them.
    pub padding: f32,
}

/// × line height. A little over one line of leading: enough to hold a
/// paragraph together, tight enough to cut between a heading and the text.
const GAP: f32 = 0.8;

/// Two boxes belong to one block when the narrower lies this far inside the
/// wider. Low, because a short last line of a paragraph still belongs to it.
const OVERLAP: f32 = 0.3;

/// The bound that decides, more than any other, how well a mixed-script page
/// reads — because a line of a stacking script runs taller than an alphabetic
/// line of the same type size, and the ratio between them is the only signal
/// the geometry has that the two are different kinds of text.
///
/// **Measured on pages that alternate a verse in a stacking script with its
/// transliteration and its translation.** At 2.5 all three join into one block
/// and the model, handed that mixture, degenerates into a repeat loop on most
/// blocks. At 1.8 they part on one page but not on another, where the ratio
/// happens to fall just under it — and there the mixed blocks come back as
/// fluent nonsense in a third script entirely. At 1.5 they part on both, and
/// six or seven lines of every seven come back right.
///
/// That is the shape of the whole trade-off, and it runs the opposite way to
/// the intuition: more context helps only while the block stays *coherent*. A
/// block that mixes writing systems line by line is worse than the same lines
/// on their own. Prose is unaffected — the line heights inside a paragraph
/// vary by a quarter, nowhere near this.
const HEIGHT: f32 = 1.5;

/// Rows per block. Twelve rows of Tibetan is already about 700 tokens.
const MAX_LINES: usize = 12;

/// × line height.
const PADDING: f32 = 0.35;

/// Two boxes are pieces of one row when the shorter lies this far inside the
/// taller's vertical span. Generous, because ascenders and descenders make
/// neighbouring boxes of one row differ in height by half.
const ROW_OVERLAP: f32 = 0.5;

/// × line height. Below this a horizontal break is a word space; above it, a
/// column gutter.
const ROW_GAP: f32 = 1.2;

impl Default for Settings {
    fn default() -> Self {
        Self {
            gap: GAP,
            overlap: OVERLAP,
            height: HEIGHT,
            row_overlap: ROW_OVERLAP,
            row_gap: ROW_GAP,
            max_lines: MAX_LINES,
            padding: PADDING,
        }
    }
}

/// A bounding box under construction.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Bounds {
    fn of(quad: &Quad) -> Self {
        let (x0, y0, x1, y1) = quad.bounds();
        Self { x0, y0, x1, y1 }
    }

    fn join(&mut self, other: &Self) {
        self.x0 = self.x0.min(other.x0);
        self.y0 = self.y0.min(other.y0);
        self.x1 = self.x1.max(other.x1);
        self.y1 = self.y1.max(other.y1);
    }

    fn height(&self) -> f32 { (self.y1 - self.y0).max(1.0) }

    fn width(&self) -> f32 { (self.x1 - self.x0).max(0.0) }

    /// How much of the shorter box's height the two share.
    fn vertical_share(&self, other: &Self) -> f32 {
        let shared = (self.y1.min(other.y1) - self.y0.max(other.y0)).max(0.0);
        shared / self.height().min(other.height())
    }

    /// How much of the narrower box's width the two share.
    fn horizontal_share(&self, other: &Self) -> f32 {
        let shared = (self.x1.min(other.x1) - self.x0.max(other.x0)).max(0.0);
        let narrower = self.width().min(other.width());
        if narrower <= 0.0 {
            0.0
        } else {
            shared / narrower
        }
    }

    /// The horizontal distance between two boxes, zero when they overlap.
    fn horizontal_gap(&self, other: &Self) -> f32 {
        (other.x0 - self.x1).max(self.x0 - other.x1).max(0.0)
    }
}

/// One row: the pieces of one line of text, joined.
#[derive(Debug, Clone)]
struct Row {
    lines: Vec<usize>,
    bounds: Bounds,
}

/// Joins detected boxes that are pieces of one row.
///
/// Two boxes belong to one row when they sit at the same height and the gap
/// between them is no wider than a line is tall. That second test is also what
/// keeps a two-column page in two columns: a gutter is wider than a word space
/// by definition, and this is where the line is drawn.
fn rows(quads: &[Quad], settings: &Settings) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for (index, quad) in quads.iter().enumerate() {
        let bounds = Bounds::of(quad);
        let joined = rows.iter_mut().rev().take(8).find(|row| {
            row.bounds.vertical_share(&bounds) >= settings.row_overlap &&
                row.bounds.horizontal_gap(&bounds) <=
                    settings.row_gap *
                        row.bounds.height().min(bounds.height())
        });
        match joined {
            Some(row) => {
                row.lines.push(index);
                row.bounds.join(&bounds);
            },
            None => rows.push(Row {
                lines: vec![index],
                bounds,
            }),
        }
    }
    rows
}

/// One block under construction.
struct Open {
    lines: Vec<usize>,
    /// The vertical span of each row that joined, before padding.
    rows: Vec<(f32, f32)>,
    /// The first line's index, which is the block's place in reading order.
    first: usize,
    bounds: Bounds,
    heights: Vec<f32>,
}

impl Open {
    fn height(&self) -> f32 {
        let mut sorted = self.heights.clone();
        sorted.sort_by(f32::total_cmp);
        sorted[sorted.len() / 2]
    }
}

/// How much ink a page row may hold and still count as a place to cut.
///
/// A pixel is ink when it is darker than this on every channel. The number is
/// the same one the illustration finder uses; a scan's paper is never near it.
const INK: u8 = 160;

/// Ink per page row: how many pixels of the row are dark.
///
/// This is what makes the boundary between two lines findable. The detector
/// marks a *shrunken core* of each line, and on a tight setting the real ink of
/// a line — the descenders of Cyrillic, the upper tiers of Tibetan — reaches
/// tens of pixels past its box. Halfway between two boxes can therefore still
/// be inside the neighbour's ink, and a block cut there shows the model a strip
/// of the line above, which it dutifully reads.
pub fn ink_profile(bgr: &[u8], page: (u32, u32)) -> Vec<u32> {
    let stride = page.0 as usize * 3;
    (0..page.1 as usize)
        .map(|row| {
            bgr[row * stride..(row + 1) * stride]
                .chunks_exact(3)
                .filter(|pixel| pixel.iter().all(|value| *value < INK))
                .count() as u32
        })
        .collect()
}

/// Groups quadrangles into blocks.
///
/// `quads` must be in the detector's reading order; `page` is the page size in
/// pixels, which the padded rectangles are clamped to. `ink` is the page's row
/// profile from [`ink_profile`]; without it the boundary between two blocks is
/// put halfway between their boxes, which is right only when the boxes are
/// tight.
pub fn assemble(
    quads: &[Quad],
    page: (u32, u32),
    ink: Option<&[u32]>,
    settings: &Settings,
) -> Vec<Block> {
    let mut open: Vec<Open> = Vec::new();
    let mut done: Vec<Open> = Vec::new();

    let all = rows(quads, settings);
    // Every row on the page, sorted by where it starts, so a block's padding
    // can be stopped before it reaches a row that is not its own.
    let mut spans: Vec<(f32, f32)> = all
        .iter()
        .map(|row| (row.bounds.y0, row.bounds.y1))
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));

    for row in all {
        let bounds = row.bounds;
        let height = bounds.height();

        // A block whose foot is already far above this row can never take
        // another: the rows arrive in order of their tops.
        let mut still_open = Vec::with_capacity(open.len());
        for block in open.drain(..) {
            if bounds.y0 - block.bounds.y1 > settings.gap * block.height() {
                done.push(block);
            } else {
                still_open.push(block);
            }
        }
        open = still_open;

        let mut best: Option<(usize, f32)> = None;
        for (at, block) in open.iter().enumerate() {
            if block.rows.len() >= settings.max_lines {
                continue;
            }
            let unit = block.height();
            let ratio = height / unit;
            if !(1.0 / settings.height..=settings.height).contains(&ratio) {
                continue;
            }
            if block.bounds.horizontal_share(&bounds) < settings.overlap {
                continue;
            }
            let gap = bounds.y0 - block.bounds.y1;
            if gap > settings.gap * unit {
                continue;
            }
            let distance = gap.max(0.0);
            if best.is_none_or(|(_, best)| distance < best) {
                best = Some((at, distance));
            }
        }

        match best {
            Some((at, _)) => {
                let block = &mut open[at];
                block.lines.extend(&row.lines);
                block.rows.push((bounds.y0, bounds.y1));
                block.heights.push(height);
                block.bounds.join(&bounds);
            },
            None => open.push(Open {
                first: row.lines[0],
                lines: row.lines,
                rows: vec![(bounds.y0, bounds.y1)],
                bounds,
                heights: vec![height],
            }),
        }
    }
    done.extend(open);
    done.sort_by_key(|block| block.first);

    done.into_iter()
        .map(|block| {
            let margin = settings.padding * block.height();
            let bands: Vec<(f32, f32)> = block
                .rows
                .iter()
                .map(|row| grown(*row, margin, &spans, &block.rows, ink))
                .collect();
            let mut bounds = block.bounds;
            bounds.y0 = bands.iter().map(|b| b.0).fold(f32::MAX, f32::min);
            bounds.y1 = bands.iter().map(|b| b.1).fold(f32::MIN, f32::max);
            Block {
                // The sides still get the full margin: nothing of a
                // neighbouring column reaches into a block that was cut from
                // it by a gutter wider than this.
                rect: clamp(bounds, margin, 0.0, page),
                bands: bands
                    .iter()
                    .map(|(top, bottom)| {
                        (
                            top.floor().max(0.0) as u32,
                            (bottom.ceil().max(0.0) as u32).min(page.1),
                        )
                    })
                    .collect(),
                lines: block.lines,
            }
        })
        .collect()
}

/// Grows one row by `margin`, stopping short of any row that is not this
/// block's.
///
/// Without this the margin of a short row reaches into the line above or below
/// it, the frame catches a strip of that line, and the model reads the strip —
/// which comes back as half a line duplicated between two blocks.
///
/// Where to stop is the whole question. Halfway between the two boxes is the
/// answer when the boxes are tight; when they are not, the whitest page row
/// between the two lines is, and that is what the ink profile is for.
fn grown(
    row: (f32, f32),
    margin: f32,
    spans: &[(f32, f32)],
    mine: &[(f32, f32)],
    ink: Option<&[u32]>,
) -> (f32, f32) {
    let owns = |span: &(f32, f32)| {
        mine.iter().any(|own| own.0 == span.0 && own.1 == span.1)
    };
    let above = spans
        .iter()
        .filter(|span| !owns(span) && span.1 <= row.0)
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let below = spans
        .iter()
        .filter(|span| !owns(span) && span.0 >= row.1)
        .min_by(|a, b| a.0.total_cmp(&b.0));

    // The search may reach a little way into this row, and only into this row:
    // the neighbour's ink runs past the bottom of its box, so the white space
    // between two lines is often below where the box says it should be.
    // Reaching the other way would find the white space above the *neighbour*
    // and swallow it whole.
    //
    // How far is set by the **neighbour's** height, not this row's. The
    // overhang belongs to the neighbour, so its scale does too — and a tall
    // stacking line must not have its upper tier of vowel signs cut off just
    // because it is tall.
    let reach =
        |neighbour: &(f32, f32)| (neighbour.1 - neighbour.0).max(1.0) * 0.4;
    let top = match above {
        Some(above) => {
            (row.0 - margin).max(boundary(above.1, row.0 + reach(above), ink))
        },
        None => row.0 - margin,
    };
    let bottom = match below {
        Some(below) => {
            (row.1 + margin).min(boundary(row.1 - reach(below), below.0, ink))
        },
        None => row.1 + margin,
    };
    (top.min(row.1), bottom.max(row.0))
}

/// The whitest page row in `first..second` — where two lines part.
///
/// Without an ink profile, or with an empty range, it is the midpoint.
fn boundary(first: f32, second: f32, ink: Option<&[u32]>) -> f32 {
    let middle = (first + second) / 2.0;
    let Some(ink) = ink else { return middle };
    let from = first.max(0.0).floor() as usize;
    let to = (second.ceil().max(0.0) as usize).min(ink.len());
    if from >= to {
        return middle;
    }
    let (at, _) = ink[from..to]
        .iter()
        .enumerate()
        // Ties go to the row nearest the midpoint: a run of blank rows is the
        // common case, and cutting through its middle is what a reader would
        // do.
        .min_by(|(one, a), (other, b)| {
            a.cmp(b).then_with(|| {
                let distance =
                    |at: usize| ((from + at) as f32 - middle).abs().to_bits();
                distance(*one).cmp(&distance(*other))
            })
        })
        .expect("the range is not empty");
    (from + at) as f32
}

/// Grows a bounding box sideways by `sides` and vertically by `ends`, then
/// clips it to the page.
fn clamp(bounds: Bounds, sides: f32, ends: f32, page: (u32, u32)) -> Rect {
    let x0 = (bounds.x0 - sides).floor().max(0.0) as u32;
    let y0 = (bounds.y0 - ends).floor().max(0.0) as u32;
    let x1 = ((bounds.x1 + sides).ceil().max(0.0) as u32).min(page.0);
    let y1 = ((bounds.y1 + ends).ceil().max(0.0) as u32).min(page.1);
    Rect {
        x: x0.min(page.0.saturating_sub(1)),
        y: y0.min(page.1.saturating_sub(1)),
        width: x1.saturating_sub(x0).max(1),
        height: y1.saturating_sub(y0).max(1),
    }
}

/// The whole page as one block: nothing painted out, every line inside.
pub fn whole(quads: &[Quad], page: (u32, u32)) -> Block {
    Block {
        lines: (0..quads.len()).collect(),
        rect: Rect {
            x: 0,
            y: 0,
            width: page.0,
            height: page.1,
        },
        bands: vec![(0, page.1)],
    }
}

/// Cuts a block out of a page raster, painting out everything that is inside
/// the frame but not part of the block.
///
/// The page is BGR — the classic engine reads it through a pipeline that is
/// BGR end to end — and the model wants RGB, so the swap happens here, at the
/// one place the two meet.
pub fn cut(bgr: &[u8], page: (u32, u32), block: &Block) -> RgbImage {
    let rect = &block.rect;
    let stride = page.0 as usize * 3;
    let mut out = RgbImage::new(rect.width, rect.height);
    for row in 0..rect.height {
        let y = rect.y + row;
        let mine = block
            .bands
            .iter()
            .any(|(top, bottom)| y >= *top && y < *bottom);
        if !mine {
            for column in 0..rect.width {
                out.put_pixel(column, row, image::Rgb([255, 255, 255]));
            }
            continue;
        }
        let source = y as usize * stride + rect.x as usize * 3;
        for column in 0..rect.width {
            let at = source + column as usize * 3;
            out.put_pixel(
                column,
                row,
                image::Rgb([bgr[at + 2], bgr[at + 1], bgr[at]]),
            );
        }
    }
    out
}
