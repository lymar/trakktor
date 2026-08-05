//! Preparing a page for the layout network.
//!
//! Three steps, and each of them is different from what the rest of the domain
//! does — this network was exported by a different upstream pipeline, and
//! following the other engines' habits here silently feeds it the wrong thing:
//!
//! - **The page is squashed into a square.** 800×800, aspect ratio *not* kept,
//!   nothing padded. An A4 page comes out visibly stretched, and that is what
//!   the model was trained on. Because nothing is padded, boxes come back by a
//!   plain per-axis rescale, which is what the graph does internally.
//! - **The channel order is RGB**, not the BGR the classic pipeline is written
//!   in end to end.
//! - **The values are divided by 255**, with no mean and no standard deviation.
//!   The model's own description says `norm_type: none` with mean 0 and std 1,
//!   which reads as "no normalization at all" — but upstream's builder defaults
//!   the absent `is_scale` to true and rewrites `none` into `mean_std`, so the
//!   division happens anyway. Skipping it hands the network values a hundred
//!   times too large and it answers with noise.
//!
//! The resampler is the bicubic kernel with `a = −0.75`, which is the kernel
//! `cv2.INTER_CUBIC` uses. It is evaluated in floating point here, while
//! OpenCV evaluates it in 11-bit fixed point; the two differ by a level of
//! brightness here and there. That matters for the text detector, whose
//! binarization threshold sits right on top of such differences, and does not
//! matter for this one: its parity contract is labels and box overlap, not
//! bits.

#[cfg(test)]
mod tests;

use crate::ocr::paddle::image::Page;

/// Channels of a page raster.
const CHANNELS: usize = 3;

/// The bicubic parameter OpenCV uses. Not the more common −0.5: the difference
/// is visible, and the reference is OpenCV.
const CUBIC_A: f64 = -0.75;

/// A page prepared for the network.
pub struct Prepared {
    /// `1 × 3 × side × side`, RGB, row-major, already divided by 255.
    pub data: Vec<f32>,
    pub side: usize,
    /// The page size the boxes must come back in.
    pub page: (u32, u32),
}

impl Prepared {
    /// The `scale_factor` input the graph is fed: how much the page was
    /// squashed, height first.
    pub fn scale_factor(&self) -> (f32, f32) {
        (
            self.side as f32 / self.page.1 as f32,
            self.side as f32 / self.page.0 as f32,
        )
    }
}

/// Squashes a page into the network's square and normalizes it.
pub fn prepare(page: &Page, side: usize) -> Prepared {
    let resized = resize_cubic(
        &page.bgr,
        page.width as usize,
        page.height as usize,
        side,
        side,
    );
    // BGR → RGB and interleaved → planar, in one pass, scaling as we go.
    let plane = side * side;
    let mut data = vec![0f32; CHANNELS * plane];
    for (at, pixel) in resized.chunks_exact(CHANNELS).enumerate() {
        data[at] = f32::from(pixel[2]) / 255.0;
        data[plane + at] = f32::from(pixel[1]) / 255.0;
        data[2 * plane + at] = f32::from(pixel[0]) / 255.0;
    }
    Prepared {
        data,
        side,
        page: (page.width, page.height),
    }
}

/// The four bicubic weights for a fractional position.
fn cubic_weights(t: f64) -> [f64; 4] {
    let a = CUBIC_A;
    let w0 =
        ((a * (t + 1.0) - 5.0 * a) * (t + 1.0) + 8.0 * a) * (t + 1.0) - 4.0 * a;
    let w1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    let u = 1.0 - t;
    let w2 = ((a + 2.0) * u - (a + 3.0)) * u * u + 1.0;
    [w0, w1, w2, 1.0 - w0 - w1 - w2]
}

/// One axis of the resize: for every output sample, the leftmost of its four
/// source samples and their weights.
struct Axis {
    first: Vec<isize>,
    weights: Vec<[f64; 4]>,
}

fn axis(source: usize, target: usize) -> Axis {
    let scale = source as f64 / target as f64;
    let mut first = Vec::with_capacity(target);
    let mut weights = Vec::with_capacity(target);
    for out in 0..target {
        let position = (out as f64 + 0.5) * scale - 0.5;
        let sample = position.floor();
        first.push(sample as isize - 1);
        weights.push(cubic_weights(position - sample));
    }
    Axis { first, weights }
}

/// Bicubic resize of an 8-bit interleaved image.
///
/// Samples outside the source replicate its edge, which is what OpenCV does
/// for a resize (as opposed to the reflection it uses for a warp).
pub fn resize_cubic(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    if source_width == 0 || source_height == 0 || width == 0 || height == 0 {
        return vec![0u8; width * height * CHANNELS];
    }
    if (source_width, source_height) == (width, height) {
        return source.to_vec();
    }

    let horizontal = axis(source_width, width);
    let vertical = axis(source_height, height);
    let row = width * CHANNELS;

    // The horizontal pass is run per source row into a small ring of scratch
    // rows: an output row needs four neighbouring source rows, and successive
    // output rows overlap in three of them.
    let mut scratch: Vec<Vec<f64>> = vec![vec![0.0; row]; 4];
    let mut loaded: Option<isize> = None;

    let mut out = Vec::with_capacity(row * height);
    for y in 0..height {
        let top = vertical.first[y];
        if loaded != Some(top) {
            // Reuse what the previous output row already sampled, which on a
            // downscale is nothing and on an upscale is three rows out of four.
            let shift = match loaded {
                Some(previous) if top > previous && top - previous < 4 => {
                    (top - previous) as usize
                },
                _ => 4,
            };
            if shift < 4 {
                scratch.rotate_left(shift);
            }
            for k in (4 - shift)..4 {
                let source_row =
                    clamp_index(top + k as isize, source_height) * source_width;
                sample_row(
                    &horizontal,
                    source,
                    source_row,
                    source_width,
                    &mut scratch[k],
                );
            }
            loaded = Some(top);
        }

        let weights = vertical.weights[y];
        for x in 0..row {
            let value = weights[0] * scratch[0][x] +
                weights[1] * scratch[1][x] +
                weights[2] * scratch[2][x] +
                weights[3] * scratch[3][x];
            out.push(value.round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// One source row through the horizontal pass.
fn sample_row(
    axis: &Axis,
    source: &[u8],
    row_start: usize,
    source_width: usize,
    out: &mut [f64],
) {
    for x in 0..axis.first.len() {
        let weights = axis.weights[x];
        for c in 0..CHANNELS {
            let mut value = 0.0;
            for k in 0..4 {
                let at = clamp_index(axis.first[x] + k as isize, source_width);
                value += weights[k] *
                    f64::from(source[(row_start + at) * CHANNELS + c]);
            }
            out[x * CHANNELS + c] = value;
        }
    }
}

fn clamp_index(index: isize, len: usize) -> usize {
    index.clamp(0, len as isize - 1) as usize
}
