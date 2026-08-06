//! Drawing what a reader found back onto the page it read.
//!
//! The picture answers the first question of any reading gone wrong — *what
//! did the model find, and in what order* — faster than the result can, and
//! next to the crops it splits the blame between the stages: no outline on the
//! page means the detector missed it, an outline whose crop is blank means the
//! recognizer did.
//!
//! Nothing here is exposed as a setting. This is a diagnostic picture rather
//! than a drawing program, and three decisions carry that:
//!
//! * **The sizes come from the page, not from the caller.** An outline is about
//!   a thousandth of the shorter side and a caption about a hundredth of it, so
//!   a screenshot and a 300-dpi scan come out looking alike — the point being
//!   that the whole page is looked at at once, and a caption that is legible at
//!   that size is the one worth drawing.
//! * **A caption that does not fit its own box shrinks with it.** On a page of
//!   footnote-sized boxes a fixed size would cover the neighbours it is there
//!   to tell apart.
//! * **Every outline is drawn before any caption.** Captions are opaque, so
//!   drawing each shape whole would let a later outline cut through an earlier
//!   number; two passes keep every caption readable whatever it overlaps.

mod font;
#[cfg(test)]
mod tests;

use std::io::Cursor;

use image::{ImageFormat, RgbImage};

use crate::ocr::{error::OcrError, page::Quad};

/// Bytes per pixel of a page raster.
const CHANNELS: usize = 3;

/// A colour to draw in, red first.
pub type Rgb = [u8; 3];

/// What the reader was sure of.
pub const CONFIDENT: Rgb = [26, 105, 220];

/// …and what it was not.
pub const DOUBTFUL: Rgb = [214, 45, 45];

/// A layer that is context rather than result — the boxes a detector handed on
/// to the stage above it, whose own confidence never reaches this far.
pub const CONTEXT: Rgb = [30, 148, 84];

/// The confidence at which a box changes colour. It is a round half rather
/// than any engine's own floor, so that the colours mean the same thing across
/// the engines and whatever `--drop-score` a run was given.
pub const SURE: f32 = 0.5;

/// Captions are written in white on the shape's own colour: the plate makes
/// the number readable over ink, and one ink colour keeps it readable on every
/// plate.
const CAPTION_INK: Rgb = [255, 255, 255];

/// Cells of clearance between a caption's text and the edge of its plate.
const PADDING: usize = 1;

/// A thin outline is this fraction of the page's shorter side, and never
/// thinner than a pixel: thick enough to see on a scan, thin enough not to
/// swallow the ink it is drawn around.
const STROKE_DIVISOR: usize = 900;

/// One cell of a caption — a seventh of its height — is this fraction of the
/// shorter side, which puts a caption at roughly one per cent of the page.
const CELL_DIVISOR: usize = 600;

/// The colour a box of this confidence is drawn in.
pub fn score_color(score: f32) -> Rgb {
    if score >= SURE { CONFIDENT } else { DOUBTFUL }
}

/// How heavily a shape is outlined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    /// The stage underneath: the line boxes a generative engine's blocks were
    /// assembled from.
    Thin,
    /// The result itself.
    Thick,
}

/// One thing to draw on the page.
#[derive(Debug, Clone)]
pub struct Shape {
    /// Where it sits, in page pixels.
    pub quad: Quad,
    /// What colour to draw it in.
    pub color: Rgb,
    /// How heavily to outline it.
    pub weight: Weight,
    /// What to write at its first corner, or nothing. Drawn in the alphabet's
    /// one case, with anything outside that alphabet spelled `?`.
    pub caption: Option<String>,
}

/// Draws `shapes` over a page and encodes the result as PNG.
///
/// `page_bgr` is the page raster the engines already hold: row-major BGR,
/// `width` by `height`. A shape reaching off the page is clipped rather than
/// refused — a box is allowed to sit on the edge of the paper.
pub fn draw_png(
    page_bgr: &[u8],
    width: usize,
    height: usize,
    shapes: &[Shape],
) -> Result<Vec<u8>, OcrError> {
    let mut canvas = Canvas::new(page_bgr, width, height)?;
    for shape in shapes {
        canvas.outline(shape);
    }
    for shape in shapes {
        if let Some(caption) = &shape.caption {
            canvas.caption(shape, caption);
        }
    }
    canvas.encode()
}

/// The page being drawn on, plus the two sizes every stroke is measured in.
struct Canvas {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    /// Width of a thin outline, in pixels.
    stroke: usize,
    /// Side of one caption cell, in pixels.
    cell: usize,
}

impl Canvas {
    /// Turns the page raster into something to draw on.
    fn new(
        page_bgr: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Self, OcrError> {
        if width == 0 ||
            height == 0 ||
            page_bgr.len() < width * height * CHANNELS
        {
            return Err(OcrError::Runtime(format!(
                "a {width}x{height} page needs {} bytes of raster, got {}",
                width * height * CHANNELS,
                page_bgr.len()
            )));
        }
        let mut pixels = Vec::with_capacity(width * height * CHANNELS);
        for at in (0..width * height * CHANNELS).step_by(CHANNELS) {
            pixels.extend_from_slice(&[
                page_bgr[at + 2],
                page_bgr[at + 1],
                page_bgr[at],
            ]);
        }
        let short = width.min(height);
        Ok(Self {
            pixels,
            width,
            height,
            // An outline rounds down and a caption rounds to the nearest: a
            // hairline that grows swallows the ink it is drawn around, while a
            // caption that shrinks stops being readable at all.
            stroke: (short / STROKE_DIVISOR).max(1),
            cell: ((short + CELL_DIVISOR / 2) / CELL_DIVISOR).max(1),
        })
    }

    /// Draws a shape's four edges.
    fn outline(&mut self, shape: &Shape) {
        let width = match shape.weight {
            Weight::Thin => self.stroke,
            Weight::Thick => self.stroke * 2,
        };
        let corners = shape.quad.points;
        for at in 0..corners.len() {
            self.segment(
                corners[at],
                corners[(at + 1) % corners.len()],
                shape.color,
                width,
            );
        }
    }

    /// Draws one edge, as a run of square stamps. Aliasing is not worth
    /// avoiding here: the edges wanted are the ones that stand out.
    fn segment(
        &mut self,
        from: (f32, f32),
        to: (f32, f32),
        color: Rgb,
        width: usize,
    ) {
        let (dx, dy) = (to.0 - from.0, to.1 - from.1);
        let span = dx.abs().max(dy.abs()).ceil();
        if !span.is_finite() {
            return;
        }
        let steps = span as usize;
        for step in 0..=steps {
            let along = if steps == 0 {
                0.0
            } else {
                step as f32 / steps as f32
            };
            self.stamp(from.0 + dx * along, from.1 + dy * along, color, width);
        }
    }

    /// Paints a square of `side` pixels centred on a point.
    fn stamp(&mut self, x: f32, y: f32, color: Rgb, side: usize) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let (x, y) = (x.round() as isize, y.round() as isize);
        let first = -((side / 2) as isize);
        let last = first + side as isize - 1;
        for row in first..=last {
            for column in first..=last {
                self.pixel(x + column, y + row, color);
            }
        }
    }

    /// Writes a caption hanging off the shape's first corner, up and to the
    /// left of it.
    ///
    /// That corner is the one place on a page of text where there is usually
    /// nothing: the margin beside the line above. Written *over* the box the
    /// caption would cover the very text the box is drawn around, and written
    /// straight above it, the end of the line before. Where the corner is
    /// against the edge of the page the caption is pushed back inside it,
    /// because a caption half off the paper says nothing at all.
    fn caption(&mut self, shape: &Shape, text: &str) {
        let characters = text.chars().count();
        if characters == 0 {
            return;
        }
        let across = font::width(characters) + PADDING * 2;
        let down = font::HEIGHT + PADDING * 2;

        let (x0, _, x1, _) = shape.quad.bounds();
        let corner = shape.quad.points[0];
        if !corner.0.is_finite() || !corner.1.is_finite() || !x0.is_finite() {
            return;
        }
        // A caption too wide for its own box shrinks with it — but never past
        // half the page's own size, because a number nobody can read helps
        // nobody.
        let room = (x1 - x0).max(1.0) as usize;
        let cell = self
            .cell
            .min((room / across).max(1))
            .max(self.cell / 2)
            .max(1);
        let (width, height) = (across * cell, down * cell);

        let left = (corner.0 as isize - width as isize).max(0) as usize;
        let left = left.min(self.width.saturating_sub(width));
        let above = corner.1 as isize - height as isize;
        let top = if above >= 0 {
            above as usize
        } else {
            corner.1.max(0.0) as usize
        }
        .min(self.height.saturating_sub(height));

        self.fill(left, top, width, height, shape.color);
        let mut at = left + PADDING * cell;
        for character in text.chars() {
            self.glyph(character, at, top + PADDING * cell, cell);
            at += (font::WIDTH + font::GAP) * cell;
        }
    }

    /// Draws one character, a filled square per lit cell.
    fn glyph(&mut self, character: char, x: usize, y: usize, cell: usize) {
        for (down, bits) in font::glyph(character).iter().enumerate() {
            for across in 0..font::WIDTH {
                if bits & (1 << (font::WIDTH - 1 - across)) == 0 {
                    continue;
                }
                self.fill(
                    x + across * cell,
                    y + down * cell,
                    cell,
                    cell,
                    CAPTION_INK,
                );
            }
        }
    }

    /// Paints a filled rectangle, clipped to the page.
    fn fill(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        color: Rgb,
    ) {
        for row in y..(y + height).min(self.height) {
            for column in x..(x + width).min(self.width) {
                let at = (row * self.width + column) * CHANNELS;
                self.pixels[at..at + CHANNELS].copy_from_slice(&color);
            }
        }
    }

    /// Paints one pixel, ignoring everything outside the page.
    fn pixel(&mut self, x: isize, y: isize, color: Rgb) {
        if x < 0 ||
            y < 0 ||
            x as usize >= self.width ||
            y as usize >= self.height
        {
            return;
        }
        let at = (y as usize * self.width + x as usize) * CHANNELS;
        self.pixels[at..at + CHANNELS].copy_from_slice(&color);
    }

    /// The drawn-on page as PNG.
    fn encode(self) -> Result<Vec<u8>, OcrError> {
        let page = RgbImage::from_raw(
            self.width as u32,
            self.height as u32,
            self.pixels,
        )
        .expect("the buffer holds three bytes per pixel of the page");
        let mut png = Cursor::new(Vec::new());
        page.write_to(&mut png, ImageFormat::Png)
            .map_err(|source| {
                OcrError::Runtime(format!(
                    "cannot encode a page overlay as PNG: {source}"
                ))
            })?;
        Ok(png.into_inner())
    }
}
