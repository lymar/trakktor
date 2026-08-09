//! The backward map: what the unwarper's field means, and what it is for.
//!
//! The field is 45 by 31 and the page is millions of pixels, so it is never
//! enlarged into an array — it is read where it is needed. Both things a
//! caller wants are the same lookup:
//!
//! * **the straightened page** — for every one of its pixels, ask the field
//!   where that pixel comes from and gather it;
//! * **the way back** — for a corner found on the straightened page, ask the
//!   field the same question and get the point on the photograph.
//!
//! That second one is why the port keeps the field rather than the picture.
//! A quadrangle on a straightened page is not a quadrangle on the photograph,
//! and a dense displacement has no closed-form inverse; but a *backward* map
//! needs none, because it already runs in the direction the answer is wanted.
//!
//! The map is bilinear between cells and the corners are pinned — Paddle's
//! `align_corners=True` in both the enlargement and the gather. Sampling
//! outside the photograph gives black, which is the reference's `zeros`
//! padding and is what cuts the sheet out of a frame with a table in it.

#[cfg(test)]
mod tests;

use super::unwarp::{FIELD_HEIGHT, FIELD_WIDTH};
use crate::ocr::{
    paddle::image::{CHANNELS, Page},
    page::Quad,
};

/// A page's backward map, kept at the size the network emits it.
#[derive(Debug, Clone)]
pub struct Backmap {
    /// `2 * FIELD_HEIGHT * FIELD_WIDTH`: the horizontal channel, then the
    /// vertical one, normalized to `-1..=1` over the photograph.
    field: Vec<f32>,
    /// The photograph's size, which the straightened page also has.
    width: usize,
    height: usize,
}

impl Backmap {
    /// Pairs a field with the page it was computed for.
    pub fn new(field: Vec<f32>, width: usize, height: usize) -> Self {
        debug_assert_eq!(field.len(), 2 * FIELD_HEIGHT * FIELD_WIDTH);
        Self {
            field,
            width,
            height,
        }
    }

    pub fn width(&self) -> usize { self.width }

    pub fn height(&self) -> usize { self.height }

    /// Where a point of the straightened page sits on the photograph, in
    /// pixels.
    ///
    /// Points outside the straightened page are read at its edge rather than
    /// refused: a quadrangle can touch the margin, and a caller mapping its
    /// corners back should get the margin rather than an error.
    pub fn source(&self, x: f32, y: f32) -> (f32, f32) {
        let cell = |value: f32, size: usize, cells: usize| {
            if size <= 1 {
                return 0.0;
            }
            (value / (size - 1) as f32).clamp(0.0, 1.0) * (cells - 1) as f32
        };
        let fx = cell(x, self.width, FIELD_WIDTH);
        let fy = cell(y, self.height, FIELD_HEIGHT);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(FIELD_WIDTH - 1);
        let y1 = (y0 + 1).min(FIELD_HEIGHT - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let plane = FIELD_HEIGHT * FIELD_WIDTH;
        let read = |channel: usize| {
            let at = |y: usize, x: usize| {
                self.field[channel * plane + y * FIELD_WIDTH + x]
            };
            let top = at(y0, x0) * (1.0 - tx) + at(y0, x1) * tx;
            let bottom = at(y1, x0) * (1.0 - tx) + at(y1, x1) * tx;
            top * (1.0 - ty) + bottom * ty
        };

        // From the `-1..=1` the gather reads to pixels, corners pinned.
        let to_pixels =
            |value: f32, size: usize| (value + 1.0) * (size as f32 - 1.0) / 2.0;
        (
            to_pixels(read(0), self.width),
            to_pixels(read(1), self.height),
        )
    }

    /// The straightened page.
    pub fn apply(&self, page: &Page) -> Page {
        let (width, height) = (self.width, self.height);
        let row = page.width as usize * CHANNELS;
        let (source_w, source_h) = (page.width as usize, page.height as usize);
        let mut bgr = vec![0u8; width * height * CHANNELS];

        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = self.source(x as f32, y as f32);
                let at = (y * width + x) * CHANNELS;
                // Outside the photograph is black, not the nearest edge: the
                // straightened page of a photograph with a table in it has
                // corners that genuinely have no source.
                if sx < -0.5 ||
                    sy < -0.5 ||
                    sx > source_w as f32 - 0.5 ||
                    sy > source_h as f32 - 0.5
                {
                    continue;
                }
                let x0 = sx.floor().max(0.0) as usize;
                let y0 = sy.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(source_w - 1);
                let y1 = (y0 + 1).min(source_h - 1);
                let tx = (sx - x0 as f32).clamp(0.0, 1.0);
                let ty = (sy - y0 as f32).clamp(0.0, 1.0);
                for c in 0..CHANNELS {
                    let read = |yy: usize, xx: usize| {
                        f32::from(page.bgr[yy * row + xx * CHANNELS + c])
                    };
                    let top = read(y0, x0) * (1.0 - tx) + read(y0, x1) * tx;
                    let bottom = read(y1, x0) * (1.0 - tx) + read(y1, x1) * tx;
                    let value = top * (1.0 - ty) + bottom * ty;
                    bgr[at + c] = value.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Page {
            width: width as u32,
            height: height as u32,
            bgr,
        }
    }

    /// A quadrangle of the straightened page, put back on the photograph.
    ///
    /// Only the corners are carried over. The true image of a straight edge
    /// under a dense displacement is a curve, and a text line is short enough
    /// that the chord is within a pixel of it — but a block spanning the page
    /// is not, which is why this is used on lines and blocks and not on the
    /// page outline.
    pub fn unmap(&self, quad: &Quad) -> Quad {
        let mut points = quad.points;
        for point in &mut points {
            *point = self.source(point.0, point.1);
        }
        Quad::new(points)
    }
}
