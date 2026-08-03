//! Loading a page and preparing it for the detector.
//!
//! Three things in here are parity-critical and quietly easy to get wrong.
//!
//! **The byte order is BGR.** PaddleOCR reads its pages through OpenCV, so the
//! whole pipeline is BGR end to end, and the ImageNet normalization constants
//! are applied to blue, green and red *in that order* — which is not what the
//! numbers look like they mean. The swap happens once, here, at the decode
//! boundary. Decoding also has to hand the pipeline what OpenCV's loader
//! would: a page with an alpha channel keeps its colour values and simply
//! loses the alpha (compositing it onto white would repaint every transparent
//! pixel), a grayscale page becomes three identical channels, a 16-bit page
//! keeps its high byte, and an EXIF orientation is applied.
//!
//! **The resize is not a general-purpose resize.** OpenCV's default is a
//! two-tap bilinear in 11-bit fixed point whose support does *not* widen when
//! the image shrinks, so it aliases freely; every general-purpose resampler
//! instead scales its kernel with the ratio and anti-aliases. On a 2x
//! downscale the two differ by a mean of five levels and a maximum of nearly
//! forty — enough to move box edges by a pixel or two and to make marginal
//! detections appear and disappear. The fixed-point algorithm is reproduced
//! here exactly, coefficient rounding included.
//!
//! **The size rounding breaks ties toward the even band.** The scaled side is
//! rounded to a multiple of 32 with Python's `round`, which is round-half-to-
//! *even*: 80 becomes 64, not 96. Rounding half up instead would hand the
//! network a whole extra 32-pixel band on every page whose scaled side lands
//! exactly halfway.

#[cfg(test)]
mod tests;

use std::{borrow::Cow, cmp::Ordering, path::Path};

use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

use super::{config::Normalize, det::SIZE_MULTIPLE};
use crate::ocr::error::OcrError;

/// Bytes per pixel of a decoded page.
pub const CHANNELS: usize = 3;

/// The normalization the published detectors are exported with: ImageNet's
/// statistics, applied to blue, green and red in that order. A model that
/// declares its own overrides it — this is the fallback for one that does not.
pub const DETECTOR_NORMALIZE: Normalize = Normalize {
    scale: 1.0 / 255.0,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
};

/// Neither side of the detector's input may exceed this, whatever the
/// side-length rule works out to. It is the largest shape the published
/// artifacts are exported for.
pub const MAX_SIDE_LIMIT: usize = 4000;

/// Pages whose two sides *add up* to less than this are padded out to at
/// least [`SIZE_MULTIPLE`] per side before anything else happens. The ratio
/// rule works on proportions and cannot by itself guarantee the network's
/// floor, so a thumbnail gets a black margin instead of being smeared across
/// one.
const PAD_BELOW_SUM: usize = 64;

/// Fixed-point weights carry 11 fractional bits, so a pair of them sums to
/// exactly this.
const COEF_SCALE: f32 = 2048.0;

/// Which side `limit_side_len` constrains.
///
/// Note that neither of the first two rules works in both directions: `Max`
/// only ever shrinks a page and `Min` only ever grows one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitType {
    /// The longer side must not exceed the limit.
    Max,
    /// The shorter side must reach the limit.
    Min,
    /// The longer side becomes the limit, whichever way that is.
    Long,
}

/// A decoded page: 8-bit BGR, row-major, three bytes per pixel, rows packed
/// with no padding between them.
#[derive(Debug, Clone)]
pub struct Page {
    pub width: u32,
    pub height: u32,
    pub bgr: Vec<u8>,
}

impl Page {
    /// Decodes an image file.
    ///
    /// The format is guessed from the content rather than the extension, and
    /// an EXIF orientation is applied — a phone photo is otherwise transposed
    /// relative to what the reference pipeline reads.
    pub fn load(path: &Path) -> Result<Self, OcrError> {
        let read_error = |source: image::ImageError| OcrError::ImageRead {
            path: path.display().to_string(),
            source,
        };
        let reader = ImageReader::open(path)
            .and_then(|reader| reader.with_guessed_format())
            .map_err(|e| read_error(image::ImageError::IoError(e)))?;
        let mut decoder = reader.into_decoder().map_err(read_error)?;
        let orientation = decoder.orientation().map_err(read_error)?;
        let mut decoded =
            DynamicImage::from_decoder(decoder).map_err(read_error)?;
        decoded.apply_orientation(orientation);

        let page = Self::from_dynamic(&decoded);
        if page.width == 0 || page.height == 0 {
            return Err(OcrError::EmptyPage {
                path: path.display().to_string(),
                width: page.width,
                height: page.height,
            });
        }
        Ok(page)
    }

    /// Takes a page that is already RGB — a rasterized PDF, say — and swaps
    /// it into the pipeline's byte order.
    pub fn from_rgb8(image: &RgbImage) -> Self {
        let mut bgr = Vec::with_capacity(image.as_raw().len());
        pack8(image.as_raw(), CHANNELS, &mut bgr);
        Self {
            width: image.width(),
            height: image.height(),
            bgr,
        }
    }

    /// Narrows any decoded image to what the pipeline works in, matching
    /// OpenCV's loader: alpha dropped, grayscale tripled, 16-bit shifted down.
    pub fn from_dynamic(image: &DynamicImage) -> Self {
        let width = image.width();
        let height = image.height();
        let mut bgr =
            Vec::with_capacity(width as usize * height as usize * CHANNELS);
        match image {
            DynamicImage::ImageLuma8(buffer) => {
                pack8(buffer.as_raw(), 1, &mut bgr)
            },
            DynamicImage::ImageLumaA8(buffer) => {
                pack8(buffer.as_raw(), 2, &mut bgr)
            },
            DynamicImage::ImageRgb8(buffer) => {
                pack8(buffer.as_raw(), 3, &mut bgr)
            },
            DynamicImage::ImageRgba8(buffer) => {
                pack8(buffer.as_raw(), 4, &mut bgr)
            },
            DynamicImage::ImageLuma16(buffer) => {
                pack16(buffer.as_raw(), 1, &mut bgr)
            },
            DynamicImage::ImageLumaA16(buffer) => {
                pack16(buffer.as_raw(), 2, &mut bgr)
            },
            DynamicImage::ImageRgb16(buffer) => {
                pack16(buffer.as_raw(), 3, &mut bgr)
            },
            DynamicImage::ImageRgba16(buffer) => {
                pack16(buffer.as_raw(), 4, &mut bgr)
            },
            // Floating-point pages reach us only from formats this build does
            // not enable; go through the crate's own narrowing for those.
            other => pack8(other.to_rgb8().as_raw(), CHANNELS, &mut bgr),
        }
        Self { width, height, bgr }
    }

    /// The page's pixel count.
    pub fn pixels(&self) -> usize { self.width as usize * self.height as usize }
}

/// Packs decoded 8-bit samples into BGR. `channels` is the source layout: one
/// or two channels are grayscale (with an alpha that is dropped, not
/// composited), three or four are RGB with the same treatment of alpha.
fn pack8(samples: &[u8], channels: usize, out: &mut Vec<u8>) {
    for pixel in samples.chunks_exact(channels) {
        if channels < CHANNELS {
            out.extend_from_slice(&[pixel[0]; CHANNELS]);
        } else {
            out.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
    }
}

/// The same for a 16-bit page. The narrowing keeps the high byte — a shift,
/// not a rescale, which is what OpenCV does and which differs by a level from
/// the rounding conversion a general image library would apply.
fn pack16(samples: &[u16], channels: usize, out: &mut Vec<u8>) {
    let high = |sample: u16| (sample >> 8) as u8;
    for pixel in samples.chunks_exact(channels) {
        if channels < CHANNELS {
            out.extend_from_slice(&[high(pixel[0]); CHANNELS]);
        } else {
            out.extend_from_slice(&[
                high(pixel[2]),
                high(pixel[1]),
                high(pixel[0]),
            ]);
        }
    }
}

/// A page normalized into the tensor the detector takes.
#[derive(Debug, Clone)]
pub struct DetectorInput {
    /// `[1, 3, height, width]`, C-order, channels in BGR order.
    pub data: Vec<f32>,
    /// The size the page was resized to — both sides multiples of 32.
    pub width: usize,
    pub height: usize,
    /// Resized size over the size that went into the resize. The detector's
    /// own post-processing rescales boxes straight from the probability map
    /// to the source page and never reads these; other heads do, and they are
    /// cheap to carry.
    pub ratio_width: f64,
    pub ratio_height: f64,
}

/// Resizes a page to the detector's input size and normalizes it.
///
/// The chain is the published one: pad a tiny page, pick a scale from the
/// side-length rule, cap it at [`MAX_SIDE_LIMIT`], round both sides to a
/// multiple of 32, resize, then `(pixel * scale - mean) / std` per channel
/// into an NCHW tensor.
pub fn detector_input(
    page: &Page,
    limit_side_len: usize,
    limit_type: LimitType,
    normalize: &Normalize,
) -> DetectorInput {
    let (source, width, height) = pad(page);
    let (target_height, target_width) =
        resize_target(height, width, limit_side_len, limit_type);
    let resized = resize_linear(
        &source,
        width,
        height,
        CHANNELS,
        target_width,
        target_height,
    );

    let plane = target_width * target_height;
    let mut data = vec![0.0f32; CHANNELS * plane];
    for channel in 0..CHANNELS {
        let mean = normalize.mean[channel];
        let deviation = normalize.std[channel];
        let out = &mut data[channel * plane..(channel + 1) * plane];
        for (value, pixel) in out.iter_mut().zip(resized.chunks_exact(CHANNELS))
        {
            // Left unfused on purpose. Folding the scale into `1 / std` and
            // the mean into an offset is one multiply-add instead of three
            // operations, and it moves the last bit of every element.
            let scaled = f32::from(pixel[channel]) * normalize.scale;
            *value = (scaled - mean) / deviation;
        }
    }

    DetectorInput {
        data,
        width: target_width,
        height: target_height,
        ratio_width: target_width as f64 / width as f64,
        ratio_height: target_height as f64 / height as f64,
    }
}

/// Grows a page that is smaller than the network's floor, anchoring it at the
/// top left over a black margin. Larger pages are passed through untouched.
fn pad(page: &Page) -> (Cow<'_, [u8]>, usize, usize) {
    let width = page.width as usize;
    let height = page.height as usize;
    let (padded_width, padded_height) = padded_size(width, height);
    if (padded_width, padded_height) == (width, height) {
        return (Cow::Borrowed(&page.bgr), width, height);
    }

    let mut padded = vec![0u8; padded_width * padded_height * CHANNELS];
    let row = width * CHANNELS;
    for y in 0..height {
        let from = y * row;
        let to = y * padded_width * CHANNELS;
        padded[to..to + row].copy_from_slice(&page.bgr[from..from + row]);
    }
    (Cow::Owned(padded), padded_width, padded_height)
}

/// The size the resize actually sees. Pages are padded by the *sum* of their
/// sides, not by either side alone.
fn padded_size(width: usize, height: usize) -> (usize, usize) {
    if width + height < PAD_BELOW_SUM {
        (width.max(SIZE_MULTIPLE), height.max(SIZE_MULTIPLE))
    } else {
        (width, height)
    }
}

/// The `(height, width)` the page is resized to, from the (already padded)
/// size it comes in at.
fn resize_target(
    height: usize,
    width: usize,
    limit_side_len: usize,
    limit_type: LimitType,
) -> (usize, usize) {
    let limit = limit_side_len as f64;
    let (h, w) = (height as f64, width as f64);
    // Both rules pick the *limiting* side by comparing the two, which for
    // `Max` is the longer one and for `Min` the shorter one.
    let ratio = match limit_type {
        LimitType::Max if height.max(width) > limit_side_len => {
            if height > width { limit / h } else { limit / w }
        },
        LimitType::Min if height.min(width) < limit_side_len => {
            if height < width { limit / h } else { limit / w }
        },
        LimitType::Long => limit / h.max(w),
        _ => 1.0,
    };

    let mut target_height = (h * ratio) as usize;
    let mut target_width = (w * ratio) as usize;
    let longest = target_height.max(target_width);
    if longest > MAX_SIDE_LIMIT {
        let capped = MAX_SIDE_LIMIT as f64 / longest as f64;
        target_height = (target_height as f64 * capped) as usize;
        target_width = (target_width as f64 * capped) as usize;
    }
    (
        round_to_multiple(target_height),
        round_to_multiple(target_width),
    )
}

/// Rounds a side to a multiple of [`SIZE_MULTIPLE`], never below one.
///
/// The tie goes to the *even* multiple, which is what Python's `round` does
/// and what a `(v + 16) / 32` would get wrong for every side that is exactly
/// halfway with an even quotient: 80 rounds down to 64, 144 down to 128.
fn round_to_multiple(side: usize) -> usize {
    let bands = side / SIZE_MULTIPLE;
    let rest = side % SIZE_MULTIPLE;
    let bands = match rest.cmp(&(SIZE_MULTIPLE / 2)) {
        Ordering::Less => bands,
        Ordering::Greater => bands + 1,
        Ordering::Equal => bands + bands % 2,
    };
    (bands * SIZE_MULTIPLE).max(SIZE_MULTIPLE)
}

/// Resizes a page with [`resize_linear`].
pub fn resize_bgr(page: &Page, width: usize, height: usize) -> Vec<u8> {
    resize_linear(
        &page.bgr,
        page.width as usize,
        page.height as usize,
        CHANNELS,
        width,
        height,
    )
}

/// One axis of the resize: for each output sample, the source index it reads
/// from and the two 11-bit weights of that sample and its right/lower
/// neighbour.
struct Axis {
    index: Vec<usize>,
    low: Vec<i32>,
    high: Vec<i32>,
}

/// Builds the sampling table for one axis.
///
/// Sample centres sit at half-pixel positions, and a centre that falls
/// outside the source degenerates to the nearest edge sample rather than
/// being renormalized over a clipped support — the border behaviour a
/// convolution-style resampler gets wrong. The weights are the fractional
/// position rounded to 11 bits, ties to even.
fn axis(source: usize, target: usize) -> Axis {
    let scale = source as f64 / target as f64;
    let mut index = Vec::with_capacity(target);
    let mut low = Vec::with_capacity(target);
    let mut high = Vec::with_capacity(target);
    for out in 0..target {
        // Computed in double and kept in single, as the reference does — the
        // narrowing is observable in the coefficients.
        let mut position = ((out as f64 + 0.5) * scale - 0.5) as f32;
        let mut sample = position.floor() as isize;
        position -= sample as f32;
        if sample < 0 {
            position = 0.0;
            sample = 0;
        }
        if sample >= source as isize - 1 {
            position = 0.0;
            sample = source as isize - 1;
        }
        index.push(sample as usize);
        low.push(((1.0 - position) * COEF_SCALE).round_ties_even() as i32);
        high.push((position * COEF_SCALE).round_ties_even() as i32);
    }
    Axis { index, low, high }
}

/// Bilinear resize of an 8-bit interleaved image, in fixed point.
///
/// This is the reference pipeline's only resampler, and reproducing it
/// exactly is the highest-value piece of parity in the whole port: the
/// detector's probability map is sensitive to it everywhere at once. Two taps
/// per axis whatever the ratio, 11-bit weights, an `i32` horizontal pass and
/// a `(v + 2) >> 2` finish after the vertical one.
pub fn resize_linear(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    channels: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    if source_width == 0 || source_height == 0 || width == 0 || height == 0 {
        return vec![0u8; width * height * channels];
    }
    // At a scale of one every weight pair is (2048, 0) and the fixed-point
    // arithmetic reduces to the identity, so skip it.
    if (source_width, source_height) == (width, height) {
        return source.to_vec();
    }

    let horizontal = axis(source_width, width);
    let vertical = axis(source_height, height);
    let row = width * channels;

    // The horizontal pass runs on demand rather than over the whole source:
    // output rows read the source in ascending order and each of them needs
    // two neighbouring source rows, so two rows of scratch are enough — and
    // on a downscale most source rows are never touched at all.
    let mut top = vec![0i32; row];
    let mut bottom = vec![0i32; row];
    let mut loaded: Option<usize> = None;

    let mut out = Vec::with_capacity(row * height);
    for y in 0..height {
        let upper = vertical.index[y];
        let lower = (upper + 1).min(source_height - 1);
        if loaded != Some(upper) {
            match loaded {
                // The output row before this one read the pair one higher, so
                // its lower row is this one's upper row.
                Some(previous) if previous + 1 == upper => {
                    std::mem::swap(&mut top, &mut bottom)
                },
                _ => sample_row(
                    &horizontal,
                    source,
                    upper,
                    source_width,
                    channels,
                    &mut top,
                ),
            }
            sample_row(
                &horizontal,
                source,
                lower,
                source_width,
                channels,
                &mut bottom,
            );
            loaded = Some(upper);
        }

        let (near, far) = (vertical.low[y], vertical.high[y]);
        for x in 0..row {
            let value = ((near * (top[x] >> 4)) >> 16) +
                ((far * (bottom[x] >> 4)) >> 16);
            out.push(((value + 2) >> 2).clamp(0, 255) as u8);
        }
    }
    out
}

/// One source row through the horizontal pass, left in 11-bit fixed point for
/// the vertical one.
fn sample_row(
    axis: &Axis,
    source: &[u8],
    row: usize,
    source_width: usize,
    channels: usize,
    out: &mut [i32],
) {
    let base = row * source_width * channels;
    for x in 0..axis.index.len() {
        let left = base + axis.index[x] * channels;
        let right = base + (axis.index[x] + 1).min(source_width - 1) * channels;
        let (near, far) = (axis.low[x], axis.high[x]);
        for c in 0..channels {
            out[x * channels + c] = i32::from(source[left + c]) * near +
                i32::from(source[right + c]) * far;
        }
    }
}
