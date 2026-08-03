//! Straightening a detected quadrangle out of the page into an upright crop.
//!
//! What the detector hands over is four corners, not a rectangle: a line of
//! text on a scan or a photograph is rotated, and usually slightly
//! trapezoidal. What the recognizer reads is a plain upright strip. This
//! module is the bridge — one perspective warp carrying the four detected
//! corners onto the corners of a fresh rectangle, sampling the page
//! bicubically on the way.
//!
//! Four details decide whether the crop lands where the recognizer's weights
//! expect it. None of them is guessable, and none of them fails loudly: get
//! one wrong and the crop is off by a fraction of a pixel, or upside down, and
//! the recognizer answers with fluent nonsense instead of an error.
//!
//! * **The crop size truncates.** The rectangle is as wide as the longer of the
//!   two horizontal sides and as tall as the longer of the two vertical ones,
//!   with the fraction *dropped*, not rounded — a quadrangle 99.7 px across
//!   becomes a 99 px crop.
//! * **The warp reads the whole page.** Sampling a bounding-box cut-out first
//!   would be cheaper, but then the border rule replicates the *cut-out's* edge
//!   instead of the page's, and every crop that touches the margin of its own
//!   box picks up different pixels.
//! * **The sampling kernel is OpenCV's bicubic, whose free parameter is
//!   `-0.75`.** The Catmull-Rom `-0.5` that most textbooks call "the" bicubic
//!   kernel is a different filter, and the difference is not academic: on a
//!   page of ordinary text it strays by several levels per channel, where
//!   `-0.75` stays within one everywhere.
//! * **A crop at least 1.5x taller than it is wide gets a quarter turn
//!   counter-clockwise.** That is how a vertical line of text — which the
//!   detector boxes as a tall, narrow quadrangle — is laid on its side for a
//!   recognizer that only ever reads left to right.
//!
//! The four corners are expected clockwise from the top left, which is the
//! order the box builder produces; corner *i* of the quadrangle lands on
//! corner *i* of the rectangle, so passing them in another order silently
//! mirrors or transposes the crop rather than failing.

#[cfg(test)]
mod tests;

use crate::ocr::page::Quad;

/// Bytes per pixel. Crops stay in the page's own channel order (blue first),
/// because that is what the detector and the recognizer were trained on; only
/// the orientation classifier wants them the other way round.
const CHANNELS: usize = 3;

/// The free parameter of the bicubic kernel, as OpenCV sets it.
const CUBIC_A: f64 = -0.75;

/// Height-to-width ratio at which the crop is turned upright. The comparison
/// is inclusive: a crop exactly 1.5x taller than it is wide does turn.
const TURN_RATIO: f64 = 1.5;

/// A straightened text-line crop: BGR bytes, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Crop {
    pub width: usize,
    pub height: usize,
    /// `width * height * 3` bytes, row-major, blue first.
    pub bgr: Vec<u8>,
}

impl Crop {
    /// The crop a degenerate quadrangle yields — one whose sides truncate to
    /// zero, or whose corners do not span a quadrangle at all.
    ///
    /// This is a deliberate divergence from PaddleOCR, where such a box has
    /// no empty case at all: OpenCV's warp, asked for an output with a zero
    /// side, hands back the *whole source image* instead, and that full page
    /// then travels on to the recognizer as if it were a text line. Callers
    /// here are expected to drop the box.
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            bgr: Vec::new(),
        }
    }

    /// Whether the crop holds no pixels.
    pub fn is_empty(&self) -> bool { self.bgr.is_empty() }
}

/// Straightens `quad` out of the page into an upright crop.
///
/// `page_bgr` is the whole page, row-major, three bytes per pixel; the
/// quadrangle's corners are page pixels, clockwise from the top left, and may
/// legitimately sit outside the page — the detector clamps them to `0..=width`
/// inclusive, one past the last column, and a warp reads a two-pixel apron
/// around its corners anyway. Everything outside repeats the nearest edge
/// pixel.
///
/// A degenerate quadrangle, or a page whose buffer is shorter than its stated
/// size, yields [`Crop::empty`] rather than an error: cropping cannot fail
/// halfway through a page, and one unusable box out of a hundred is a box to
/// skip, not a run to abort.
pub fn rotate_crop(
    page_bgr: &[u8],
    page_width: usize,
    page_height: usize,
    quad: &Quad,
) -> Crop {
    let Some(bytes) = page_width
        .checked_mul(page_height)
        .and_then(|n| n.checked_mul(CHANNELS))
    else {
        return Crop::empty();
    };
    if bytes == 0 || page_bgr.len() < bytes {
        return Crop::empty();
    }

    let src = quad.points.map(|(x, y)| (f64::from(x), f64::from(y)));
    let (width, height) = crop_size(&src, page_width, page_height);
    if width == 0 || height == 0 {
        return Crop::empty();
    }

    // Solving from the rectangle *back* to the quadrangle gives the map the
    // sampler actually walks — inverse mapping, one destination pixel at a
    // time — without the detour through a forward transform and a matrix
    // inversion.
    let (w, h) = (width as f64, height as f64);
    let dst = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let Some(m) = homography(&dst, &src) else {
        return Crop::empty();
    };

    let mut bgr = vec![0u8; width * height * CHANNELS];
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = source_of(&m, x as f64, y as f64);
            let pixel = sample(page_bgr, page_width, page_height, sx, sy);
            let at = (y * width + x) * CHANNELS;
            bgr[at..at + CHANNELS].copy_from_slice(&pixel);
        }
    }

    let crop = Crop { width, height, bgr };
    if h / w >= TURN_RATIO {
        turn(&crop)
    } else {
        crop
    }
}

/// Size of the rectangle `quad` is straightened into: the longer of the two
/// horizontal sides by the longer of the two vertical ones, fraction dropped.
///
/// A side longer than the page's diagonal cannot belong to a quadrangle on
/// this page, so it is a caller's mistake — and it must not become an
/// allocation. Such a quadrangle, like one whose sides round down to nothing,
/// reports a zero size.
fn crop_size(
    quad: &[(f64, f64); 4],
    page_width: usize,
    page_height: usize,
) -> (usize, usize) {
    let [p0, p1, p2, p3] = *quad;
    let width = distance(p0, p1).max(distance(p2, p3));
    let height = distance(p0, p3).max(distance(p1, p2));
    let bound = (page_width as f64).hypot(page_height as f64);
    // Negated so that a NaN side, which compares false either way, fails.
    if !(width <= bound) || !(height <= bound) {
        return (0, 0);
    }
    (width as usize, height as usize)
}

/// Euclidean distance, summed and rooted in that order — the arithmetic
/// upstream's vector norm does, rather than the better-conditioned `hypot`,
/// so that a side length landing within an ulp of an integer truncates to the
/// same crop size on both sides.
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    (dx * dx + dy * dy).sqrt()
}

/// Solves the perspective transform carrying `from[i]` onto `to[i]`, returned
/// row-major as `[a, b, c, d, e, f, g, h, 1]`.
///
/// Each correspondence contributes two rows of the usual eight-unknown linear
/// system, solved by Gauss-Jordan elimination with partial pivoting. `None`
/// when the correspondences are degenerate — four corners on one line, say —
/// because the system is then singular and there is nothing to warp.
fn homography(
    from: &[(f64, f64); 4],
    to: &[(f64, f64); 4],
) -> Option<[f64; 9]> {
    let mut rows = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (u, v) = from[i];
        let (x, y) = to[i];
        rows[2 * i] = [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x, x];
        rows[2 * i + 1] = [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y, y];
    }

    // The system's entries span page coordinates and their products, so
    // "singular" has to be judged against the largest of them rather than
    // against an absolute epsilon.
    let scale = rows
        .iter()
        .flatten()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    if !(scale > 0.0) {
        return None;
    }
    let tolerance = scale * 1e-12;

    for col in 0..8 {
        let pivot = (col..8)
            .max_by(|&i, &j| rows[i][col].abs().total_cmp(&rows[j][col].abs()))
            .expect("the column has at least one remaining row");
        if !(rows[pivot][col].abs() > tolerance) {
            return None;
        }
        rows.swap(col, pivot);
        let lead = rows[col][col];
        for k in col..9 {
            rows[col][k] /= lead;
        }
        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = rows[row][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..9 {
                rows[row][k] -= factor * rows[col][k];
            }
        }
    }

    let mut m = [1.0f64; 9];
    for (i, row) in rows.iter().enumerate() {
        m[i] = row[8];
    }
    m.iter().all(|value| value.is_finite()).then_some(m)
}

/// Where destination pixel `(x, y)` reads from on the page. Pixel centers are
/// the integers themselves, as in OpenCV's warp — no half-pixel shift.
fn source_of(m: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = m[6] * x + m[7] * y + m[8];
    // A destination pixel sitting on the transform's horizon has no source
    // point at all. OpenCV folds that denominator to zero rather than to
    // infinity, which reads the page's top-left corner; matching it keeps the
    // pathological case boring instead of black.
    let (sx, sy) = if w == 0.0 {
        (0.0, 0.0)
    } else {
        (
            (m[0] * x + m[1] * y + m[2]) / w,
            (m[3] * x + m[4] * y + m[5]) / w,
        )
    };
    if sx.is_finite() && sy.is_finite() {
        (sx, sy)
    } else {
        (0.0, 0.0)
    }
}

/// One bicubic sample of the page, with the border replicated outwards.
///
/// The kernel spans four taps per axis, starting one pixel before the sample,
/// and every tap index is clamped into the page — which is exactly the
/// replicate rule, since a clamped index reads the nearest edge pixel.
fn sample(
    page: &[u8],
    width: usize,
    height: usize,
    sx: f64,
    sy: f64,
) -> [u8; CHANNELS] {
    let (fx, fy) = (sx.floor(), sy.floor());
    let wx = cubic_weights(sx - fx);
    let wy = cubic_weights(sy - fy);
    // Saturating casts: a source coordinate beyond the page still lands on a
    // clamped tap.
    let (ix, iy) = (fx as i64, fy as i64);
    let cols: [usize; 4] =
        std::array::from_fn(|k| clamp_index(ix - 1 + k as i64, width));
    let rows: [usize; 4] =
        std::array::from_fn(|k| clamp_index(iy - 1 + k as i64, height));

    let mut pixel = [0u8; CHANNELS];
    for (c, out) in pixel.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (r, row) in rows.iter().enumerate() {
            let base = row * width * CHANNELS + c;
            let mut line = 0.0;
            for (k, col) in cols.iter().enumerate() {
                line += wx[k] * f64::from(page[base + col * CHANNELS]);
            }
            acc += wy[r] * line;
        }
        *out = round_u8(acc);
    }
    pixel
}

/// The four bicubic weights for a sample sitting `t` of the way between its
/// second and third tap, `t` in `0.0..1.0`.
///
/// Written the way OpenCV writes it, last weight included: taking it as the
/// remainder of the other three, rather than evaluating the kernel a fourth
/// time, is what makes the four sum to exactly one.
fn cubic_weights(t: f64) -> [f64; 4] {
    let a = CUBIC_A;
    let (before, after) = (t + 1.0, 1.0 - t);
    let c0 = ((a * before - 5.0 * a) * before + 8.0 * a) * before - 4.0 * a;
    let c1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    let c2 = ((a + 2.0) * after - (a + 3.0)) * after * after + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// A tap index clamped into `0..len`, which is the replicate border rule.
fn clamp_index(index: i64, len: usize) -> usize {
    index.clamp(0, len.saturating_sub(1) as i64) as usize
}

/// Back to a byte: halves away from zero, then clamped.
fn round_u8(value: f64) -> u8 { value.round().clamp(0.0, 255.0) as u8 }

/// A quarter turn counter-clockwise.
///
/// The crop's rightmost column becomes the turned crop's top row, so a
/// vertical line read top-to-bottom comes out reading left-to-right. Turning
/// the other way hands the recognizer the same strip upside down, which it
/// reads as confident nonsense rather than refusing.
fn turn(crop: &Crop) -> Crop {
    let (width, height) = (crop.height, crop.width);
    let mut bgr = vec![0u8; width * height * CHANNELS];
    for y in 0..height {
        for x in 0..width {
            let from = (x * crop.width + (crop.width - 1 - y)) * CHANNELS;
            let to = (y * width + x) * CHANNELS;
            bgr[to..to + CHANNELS]
                .copy_from_slice(&crop.bgr[from..from + CHANNELS]);
        }
    }
    Crop { width, height, bgr }
}

/// Encodes a crop as a PNG, for a caller that wants to look at what the
/// recognizer was given.
pub fn to_png(crop: &Crop) -> Result<Vec<u8>, crate::ocr::error::OcrError> {
    use image::{ImageEncoder, codecs::png::PngEncoder};

    let mut rgb = Vec::with_capacity(crop.bgr.len());
    for pixel in crop.bgr.chunks_exact(3) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(
            &rgb,
            crop.width as u32,
            crop.height as u32,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| {
            crate::ocr::error::OcrError::Runtime(format!(
                "encoding a line crop: {e}"
            ))
        })?;
    Ok(png)
}
