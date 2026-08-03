//! Finding the illustrations on a page, with no layout model in sight.
//!
//! The method is one sentence long: paint out every text line the detector
//! found, and whatever ink is left standing is a picture. What remains is
//! grouped into connected regions, and a region big enough — and not too far
//! from square — is reported as a figure.
//!
//! **This finds ink, not meaning.** A large table, a display formula with a
//! tall brace, a stamp, a signature, a logo, a boxed sidebar: to a page of
//! pixels these are all the same thing, a coherent patch of ink that no text
//! box covers, and they all come back as "figures". Telling them apart needs a
//! model that reads the page semantically. What the pixels do buy, without
//! one, is worth having anyway: the region stops being read as text, so a
//! column that wraps around a picture keeps its own left margin instead of
//! looking like a page-wide paragraph with a ragged indent, and the picture
//! itself can be cut out and referenced from the markdown.
//!
//! Four decisions make the scan cheap and keep it from finding things that are
//! not there.
//!
//! * **The work happens on a mask a quarter of the page on each side.** Only
//!   the thresholding reads the raster; everything after it — the labelling,
//!   the merge, the measurements — runs over a sixteenth of the page, which is
//!   what keeps a large scan cheap. The downscale takes the *darkest* pixel of
//!   each block rather than their average: averaging a block that holds one
//!   thin stroke lifts it back above any sensible ink threshold, and a line
//!   drawing is nothing but thin strokes.
//! * **"Dark" means the darkest of the three channels**, not the luminance. A
//!   saturated colour is ink even where its luminance says otherwise — a yellow
//!   field on white is invisible to a luminance threshold and obvious to this
//!   one. The cost is that a pale tint counts as ink too, which is what the
//!   area and aspect filters are for.
//! * **Text is painted out before anything is labelled**, with a margin
//!   proportional to the line's own height, so an accent or a descender poking
//!   out of its box cannot seed a region. The margin is applied to the
//!   quadrangle's bounding box, which over-paints a rotated line — deliberately
//!   the safe direction.
//! * **Ink is grown slightly before regions are counted**, so that a drawing
//!   made of separate strokes comes back as one figure rather than a dozen. The
//!   boxes are then measured on the original ink, not on the grown mask, so
//!   growing costs nothing in accuracy.
//!
//! What is dropped, besides the small and the misshapen: a region that hugs
//! the edge of the page and is either thin or hollow. That is the scanner's
//! own shadow, a staple, a ruled line, and the printed frame some books put
//! around every page — none of which is a picture, all of which would
//! otherwise be the biggest "figure" on the page. A picture that genuinely
//! bleeds off the edge is solid, so it survives.

#[cfg(test)]
mod tests;

use std::io::Cursor;

use image::{ImageFormat, RgbImage};

use crate::ocr::{error::OcrError, page::Quad};

/// Bytes per pixel of a page raster, blue first.
const CHANNELS: usize = 3;

/// How much smaller than the page the working mask is, on each side.
const SCALE: usize = 4;

/// How far past its own box a text line is painted out, as a fraction of the
/// line's height.
const TEXT_MARGIN: f32 = 0.25;

/// The same margin, floored in page pixels, so a very short line still gets
/// more than a rounding's worth of clearance.
const MIN_TEXT_MARGIN: f32 = 2.0;

/// How far ink is grown before regions are counted, as a fraction of the
/// shorter side of the mask. Two pieces of ink end up in the same figure when
/// the gap between them is at most *twice* this.
const GROW: f32 = 0.015;

/// The same reach, floored in mask cells.
const MIN_GROW: usize = 2;

/// How close to the edge of the page a region counts as hugging it, as a
/// fraction of the shorter side of the mask.
const BORDER_BAND: f32 = 0.05;

/// A region touching that band is thin — and so a rule, an edge streak or a
/// staple shadow — when its shorter side is below this fraction of the shorter
/// side of the mask.
const BORDER_THIN: f32 = 0.03;

/// The same bar, floored in mask cells: at a quarter scale a hairline is one
/// cell thick, and a mask cell is worth four page pixels.
const MIN_BORDER_THIN: usize = 2;

/// A region touching the band that reaches this far across the mask in *both*
/// directions is a candidate frame.
const BORDER_SPAN: f32 = 0.8;

/// …and it is a frame rather than a full-page picture when its ink fills less
/// than this much of its own box.
const BORDER_FILL: f32 = 0.10;

/// A non-text region of the page.
///
/// Page pixels, with `x`/`y` the top-left corner. The box is a whole number of
/// mask cells on every side, so it can sit a few pixels outside the ink it was
/// measured from — never inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Figure {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Figure {
    /// The figure as a quadrangle, so a caller that orders text boxes can
    /// order a picture alongside them without a second code path.
    pub fn quad(&self) -> Quad {
        let (x0, y0) = (self.x as f32, self.y as f32);
        let (x1, y1) = (x0 + self.width as f32, y0 + self.height as f32);
        Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }
}

/// What counts as a figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Smallest area, as a fraction of the page, that counts as a figure.
    pub min_area_fraction: f32,
    /// Aspect-ratio band a figure must fall inside, as width over height.
    pub aspect_range: (f32, f32),
    /// How dark a pixel must be to count as ink, `0..=255` on the grey scale.
    /// The grey value is the darkest of the three channels, so a saturated
    /// colour is dark here even when its luminance is not.
    pub ink_threshold: u8,
}

impl Default for Settings {
    /// Half a percent of the page, a decade of aspect ratio either way, and an
    /// ink bar low enough to ignore the grey of scanned paper.
    fn default() -> Self {
        Self {
            min_area_fraction: 0.005,
            aspect_range: (0.1, 10.0),
            ink_threshold: 200,
        }
    }
}

/// The figures of one page, top to bottom and then left to right.
///
/// `page_bgr` is the page raster, row-major BGR, `width` x `height`. `text` is
/// every quadrangle the detector claimed, whose ink is excluded from the
/// search; passing an empty slice is legal and simply reports the ink of the
/// whole page, text included. A raster shorter than the size it declares
/// yields no figures rather than an error — this is a heuristic pass, and it
/// has nothing to say about a page it cannot read.
///
/// A figure may enclose a text line, because a label inside a drawing is
/// painted out of the mask while the drawing around it still bounds the box.
/// A caller that folds figures into a reading order decides for itself what to
/// do with the lines that land inside one.
pub fn find(
    page_bgr: &[u8],
    width: usize,
    height: usize,
    text: &[Quad],
    settings: &Settings,
) -> Vec<Figure> {
    if width == 0 || height == 0 || page_bgr.len() < width * height * CHANNELS {
        return Vec::new();
    }

    let (mut mask, mask_w, mask_h) =
        ink_mask(page_bgr, width, height, settings.ink_threshold);
    clear_text(&mut mask, mask_w, mask_h, width, height, text);
    clear_furniture(&mut mask, mask_w, mask_h);

    // The regions are labelled on the grown mask and then measured on the
    // ungrown one, which is what keeps the merge from inflating every box by
    // its own reach.
    let grown = grow(&mask, mask_w, mask_h, reach(mask_w, mask_h));
    let (labels, count) = label(&grown, mask_w, mask_h);

    let mut figures: Vec<Figure> = regions(&labels, &mask, count, mask_w)
        .into_iter()
        .flatten()
        .filter_map(|region| accept(&region, width, height, settings))
        .collect();
    figures.sort_by_key(|figure| (figure.y, figure.x));
    figures
}

/// Crops a figure out of the page as a PNG, ready to be written next to the
/// markdown.
///
/// The box is clamped to the page rather than rejected, because it comes from
/// a downscaled mask and its far edges can round a few pixels past the last
/// row or column. A box with nothing left after that clamp is an error — it
/// describes some other page.
pub fn crop_png(
    page_bgr: &[u8],
    width: usize,
    height: usize,
    figure: &Figure,
) -> Result<Vec<u8>, OcrError> {
    if width == 0 || height == 0 || page_bgr.len() < width * height * CHANNELS {
        return Err(OcrError::Runtime(format!(
            "a {width}x{height} page needs {} bytes of raster, got {}",
            width * height * CHANNELS,
            page_bgr.len()
        )));
    }
    let x0 = (figure.x as usize).min(width);
    let y0 = (figure.y as usize).min(height);
    let x1 = x0.saturating_add(figure.width as usize).min(width);
    let y1 = y0.saturating_add(figure.height as usize).min(height);
    if x1 <= x0 || y1 <= y0 {
        return Err(OcrError::Runtime(format!(
            "a figure {}x{} at ({}, {}) has nothing inside a {width}x{height} \
             page",
            figure.width, figure.height, figure.x, figure.y
        )));
    }

    let (crop_w, crop_h) = (x1 - x0, y1 - y0);
    let mut rgb = Vec::with_capacity(crop_w * crop_h * CHANNELS);
    for y in y0..y1 {
        for x in x0..x1 {
            let at = (y * width + x) * CHANNELS;
            rgb.extend_from_slice(&[
                page_bgr[at + 2],
                page_bgr[at + 1],
                page_bgr[at],
            ]);
        }
    }
    let crop = RgbImage::from_raw(crop_w as u32, crop_h as u32, rgb)
        .expect("the buffer holds three bytes per pixel of the crop");

    let mut png = Cursor::new(Vec::new());
    crop.write_to(&mut png, ImageFormat::Png)
        .map_err(|source| {
            OcrError::Runtime(format!(
                "cannot encode a figure as PNG: {source}"
            ))
        })?;
    Ok(png.into_inner())
}

/// One connected region: its bounds in mask cells, inclusive, and how many
/// cells of ink it holds.
#[derive(Debug, Clone, Copy)]
struct Region {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    cells: usize,
}

impl Region {
    /// Width in mask cells.
    fn width(&self) -> usize { self.x1 + 1 - self.x0 }

    /// Height in mask cells.
    fn height(&self) -> usize { self.y1 + 1 - self.y0 }
}

/// The page at a quarter scale, one flag per cell: is there any ink in it?
///
/// The reduction is a minimum, not a mean — see the note on thin strokes at
/// the top of the module — and it is a minimum over the channels as well as
/// over the block, so a cell is ink when any pixel of it is dark in any
/// channel.
fn ink_mask(
    page_bgr: &[u8],
    width: usize,
    height: usize,
    threshold: u8,
) -> (Vec<bool>, usize, usize) {
    let mask_w = width.div_ceil(SCALE);
    let mask_h = height.div_ceil(SCALE);
    let mut mask = vec![false; mask_w * mask_h];
    for y in 0..height {
        let row = y * width * CHANNELS;
        let cells = (y / SCALE) * mask_w;
        for x in 0..width {
            let at = row + x * CHANNELS;
            let darkest =
                page_bgr[at].min(page_bgr[at + 1]).min(page_bgr[at + 2]);
            if darkest <= threshold {
                mask[cells + x / SCALE] = true;
            }
        }
    }
    (mask, mask_w, mask_h)
}

/// Paints every text quadrangle out of the mask, margin included.
fn clear_text(
    mask: &mut [bool],
    mask_w: usize,
    mask_h: usize,
    width: usize,
    height: usize,
    text: &[Quad],
) {
    for quad in text {
        let (x0, y0, x1, y1) = quad.bounds();
        let margin = (quad.height() * TEXT_MARGIN).max(MIN_TEXT_MARGIN);
        let Some((from_x, to_x)) =
            span(x0 - margin, x1 + margin, width, mask_w)
        else {
            continue;
        };
        let Some((from_y, to_y)) =
            span(y0 - margin, y1 + margin, height, mask_h)
        else {
            continue;
        };
        for y in from_y..=to_y {
            mask[y * mask_w + from_x..=y * mask_w + to_x].fill(false);
        }
    }
}

/// The inclusive range of mask cells a span of page pixels covers, or `None`
/// when the span misses the page entirely. A `NaN` bound misses.
fn span(
    from: f32,
    to: f32,
    size: usize,
    cells: usize,
) -> Option<(usize, usize)> {
    if from.is_nan() ||
        to.is_nan() ||
        from > to ||
        to < 0.0 ||
        from >= size as f32
    {
        return None;
    }
    let from = from.max(0.0) as usize / SCALE;
    let to = to.min((size - 1) as f32) as usize / SCALE;
    Some((from.min(cells - 1), to.min(cells - 1)))
}

/// Paints out the regions that hug the edge of the page without being
/// pictures.
///
/// This runs before the merge, and that ordering is the point: a printed frame
/// left in the mask would be within reach of everything it encloses and would
/// swallow the page whole.
fn clear_furniture(mask: &mut [bool], mask_w: usize, mask_h: usize) {
    let (labels, count) = label(mask, mask_w, mask_h);
    let doomed: Vec<bool> = regions(&labels, mask, count, mask_w)
        .iter()
        .map(|region| {
            region.is_some_and(|region| furniture(&region, mask_w, mask_h))
        })
        .collect();
    for (cell, ink) in mask.iter_mut().enumerate() {
        let label = labels[cell] as usize;
        if *ink && label > 0 && doomed[label - 1] {
            *ink = false;
        }
    }
}

/// Whether a region is page furniture: touching the edge band, and either thin
/// enough to be a rule or hollow enough to be a frame.
fn furniture(region: &Region, mask_w: usize, mask_h: usize) -> bool {
    let short = mask_w.min(mask_h) as f32;
    let band = (BORDER_BAND * short) as usize;
    let touches = region.x0 <= band ||
        region.y0 <= band ||
        region.x1 + band + 1 >= mask_w ||
        region.y1 + band + 1 >= mask_h;
    if !touches {
        return false;
    }
    let (width, height) = (region.width(), region.height());
    let thin = width.min(height) <=
        ((BORDER_THIN * short) as usize).max(MIN_BORDER_THIN);
    let spans = width as f32 >= BORDER_SPAN * mask_w as f32 &&
        height as f32 >= BORDER_SPAN * mask_h as f32;
    let hollow =
        spans && (region.cells as f32) < BORDER_FILL * (width * height) as f32;
    thin || hollow
}

/// How far ink is grown before regions are counted, in mask cells.
fn reach(mask_w: usize, mask_h: usize) -> usize {
    ((GROW * mask_w.min(mask_h) as f32) as usize).max(MIN_GROW)
}

/// Grows the mask by `reach` cells in every direction, as two one-dimensional
/// passes — a square structuring element separates, and one pass over the rows
/// followed by one over the columns is far cheaper than a square kernel.
fn grow(mask: &[bool], width: usize, height: usize, reach: usize) -> Vec<bool> {
    if reach == 0 {
        return mask.to_vec();
    }
    let mut wide = vec![false; mask.len()];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            if mask[row + x] {
                let from = row + x.saturating_sub(reach);
                let to = row + (x + reach).min(width - 1);
                wide[from..=to].fill(true);
            }
        }
    }
    let mut grown = vec![false; mask.len()];
    for y in 0..height {
        let rows = y.saturating_sub(reach)..=(y + reach).min(height - 1);
        for x in 0..width {
            if wide[y * width + x] {
                for other in rows.clone() {
                    grown[other * width + x] = true;
                }
            }
        }
    }
    grown
}

/// Labels the eight-connected components of a mask: `0` is background, the
/// components are numbered from `1`.
///
/// The flood fill carries its own stack. A picture that fills a 300-dpi page
/// is a hundred thousand cells even at a quarter scale, and a recursive fill
/// would overflow the real one.
fn label(mask: &[bool], width: usize, height: usize) -> (Vec<u32>, usize) {
    let mut labels = vec![0u32; mask.len()];
    let mut count = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        count += 1;
        labels[start] = count;
        stack.push(start);
        while let Some(at) = stack.pop() {
            let (x, y) = (at % width, at / width);
            for other_y in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for other_x in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let neighbour = other_y * width + other_x;
                    if mask[neighbour] && labels[neighbour] == 0 {
                        labels[neighbour] = count;
                        stack.push(neighbour);
                    }
                }
            }
        }
    }
    (labels, count as usize)
}

/// The bounds and the ink count of every label, measured over `ink` alone.
///
/// The two arguments are deliberately allowed to disagree: labelling the grown
/// mask and measuring the ungrown one is how near-touching pieces share a
/// label without sharing the grown mask's inflated bounds. A label with no ink
/// under it comes back as `None`.
fn regions(
    labels: &[u32],
    ink: &[bool],
    count: usize,
    width: usize,
) -> Vec<Option<Region>> {
    let mut found: Vec<Option<Region>> = vec![None; count];
    for (cell, on) in ink.iter().enumerate() {
        let label = labels[cell] as usize;
        if !*on || label == 0 {
            continue;
        }
        let (x, y) = (cell % width, cell / width);
        match &mut found[label - 1] {
            Some(region) => {
                region.x0 = region.x0.min(x);
                region.x1 = region.x1.max(x);
                region.y0 = region.y0.min(y);
                region.y1 = region.y1.max(y);
                region.cells += 1;
            },
            slot => {
                *slot = Some(Region {
                    x0: x,
                    y0: y,
                    x1: x,
                    y1: y,
                    cells: 1,
                });
            },
        }
    }
    found
}

/// Maps a region back onto the page and applies the size and shape filters.
fn accept(
    region: &Region,
    width: usize,
    height: usize,
    settings: &Settings,
) -> Option<Figure> {
    let x = region.x0 * SCALE;
    let y = region.y0 * SCALE;
    let right = ((region.x1 + 1) * SCALE).min(width);
    let bottom = ((region.y1 + 1) * SCALE).min(height);
    if right <= x || bottom <= y {
        return None;
    }
    let figure = Figure {
        x: x as u32,
        y: y as u32,
        width: (right - x) as u32,
        height: (bottom - y) as u32,
    };

    let area = f64::from(figure.width) * f64::from(figure.height);
    if area < f64::from(settings.min_area_fraction) * (width * height) as f64 {
        return None;
    }
    let aspect = figure.width as f32 / figure.height as f32;
    let (narrowest, widest) = settings.aspect_range;
    if aspect < narrowest || aspect > widest {
        return None;
    }
    Some(figure)
}
