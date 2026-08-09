//! The page-orientation classifier: `PP-LCNet_x1_0_doc_ori`.
//!
//! The same backbone as the text-line classifier ([`crate::ocr::paddle::cls`])
//! and the same weights layout, differing in two places only: a square input,
//! so every strided block spends both axes rather than the height alone, and
//! four classes instead of two. Both differences live in
//! [`Backbone`](crate::ocr::paddle::lcnet::Backbone) and in the head here; the
//! thirteen blocks between them are shared code.
//!
//! **A class is the turn that puts the page right, not the turn it has taken.**
//! Upstream's own pipeline reads the label as an angle and hands it straight to
//! a counter-clockwise rotation, so a page lying on its left side answers
//! `270`. Reading it the other way round is a mistake that survives every test
//! made of an upright page and fails on every other.
//!
//! The input is prepared the way the reference does it and not the way the
//! text-line classifier does: the short side is scaled to 256 and a 224 square
//! is cut from the middle. A page is not squeezed into the square — squeezing
//! a portrait page to 1:1 makes it look like a landscape one turned, which is
//! the very question being asked.

#[cfg(test)]
mod tests;

use candle_core::{Device, Tensor};

use crate::ocr::{
    error::OcrError,
    paddle::{
        image::{CHANNELS, Page, resize_linear},
        lcnet::{Backbone, FEATURES, Stride},
        net::{Linear, Loader},
    },
};

/// The square the network reads.
pub const SIDE: usize = 224;

/// What the short side is scaled to before the square is cut out of the
/// middle.
pub const SHORT_SIDE: usize = 256;

/// The four classes, in the order the graph emits them. The value is the
/// counter-clockwise turn that puts the page upright.
pub const CLASSES: [u16; 4] = [0, 90, 180, 270];

/// How sure the classifier must be before the page is turned.
///
/// A page is normally upright, and turning an upright page wrecks it
/// completely rather than slightly — so an unsure answer leaves the page
/// alone. The reference takes the arg-max whatever it is; on a page whose
/// classes come out near even that is a coin toss with a page riding on it.
pub const THRESHOLD: f32 = 0.6;

/// The normalization the model was trained with: the ImageNet statistics,
/// applied to the `0..=1` scale.
const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const STD: [f64; 3] = [0.229, 0.224, 0.225];

/// What the classifier made of a page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// The counter-clockwise turn, in degrees, that puts the page upright.
    pub turn: u16,
    /// How sure the classifier is of it.
    pub score: f32,
}

/// The page-orientation classifier.
#[derive(Debug)]
pub struct Classifier {
    device: Device,
    backbone: Backbone,
    head: Linear,
}

impl Classifier {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let classes = loader.artifact().output_classes().ok_or_else(|| {
            OcrError::Artifact(
                "the page-orientation classifier does not declare how many \
                 classes it reads"
                    .into(),
            )
        })?;
        if classes != CLASSES.len() {
            return Err(OcrError::Artifact(format!(
                "the page-orientation classifier reads {classes} classes; \
                 this port knows the {}-class one, the four right angles",
                CLASSES.len()
            )));
        }
        Ok(Self {
            device: loader.device().clone(),
            backbone: Backbone::load(loader, Stride::Both)?,
            head: Linear::load(loader, "linear_0", FEATURES, CLASSES.len())?,
        })
    }

    /// The four probabilities for one page.
    ///
    /// The softmax is inside the exported graph, so what comes back already
    /// sums to one.
    pub fn probabilities(&self, page: &Page) -> Result<[f32; 4], OcrError> {
        let input = page_tensor(page, &self.device)?;
        let logits = self.head.forward(&self.backbone.features(&input)?)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
        let flat = probabilities.flatten_all()?.to_vec1::<f32>()?;
        let mut out = [0f32; 4];
        out.copy_from_slice(&flat[..CLASSES.len()]);
        Ok(out)
    }

    /// Which way up the page is, or `None` when the classifier is not
    /// [`THRESHOLD`] sure and the page is better left as it came.
    pub fn read(&self, page: &Page) -> Result<Option<Reading>, OcrError> {
        let probabilities = self.probabilities(page)?;
        let (at, score) = probabilities.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best, high), (at, score)| {
                if *score > high {
                    (at, *score)
                } else {
                    (best, high)
                }
            },
        );
        Ok((score >= THRESHOLD).then_some(Reading {
            turn: CLASSES[at],
            score,
        }))
    }
}

/// Turns a page into the tensor the network reads: `[1, 3, 224, 224]`, red
/// channel first.
///
/// The short side goes to 256 and the middle 224 square is cut out, which is
/// the reference's `ResizeImage(resize_short) + CropImage`. Rounding follows
/// it too: the long side is scaled by the same factor and truncated, and the
/// crop's offset is the half-difference truncated.
pub fn page_tensor(page: &Page, device: &Device) -> Result<Tensor, OcrError> {
    let (width, height) = (page.width as usize, page.height as usize);
    if width == 0 || height == 0 {
        return Err(OcrError::Runtime(
            "the page-orientation classifier was handed an empty page".into(),
        ));
    }

    let scale = SHORT_SIDE as f64 / width.min(height) as f64;
    let scaled_w = ((width as f64 * scale) as usize).max(SIDE);
    let scaled_h = ((height as f64 * scale) as usize).max(SIDE);
    let resized =
        resize_linear(&page.bgr, width, height, CHANNELS, scaled_w, scaled_h);

    let left = (scaled_w - SIDE) / 2;
    let top = (scaled_h - SIDE) / 2;

    let mut data = vec![0f32; CHANNELS * SIDE * SIDE];
    for c in 0..CHANNELS {
        let scale = (1.0 / 255.0 / STD[c]) as f32;
        let shift = (-MEAN[c] / STD[c]) as f32;
        // The page arrives blue first and the model wants red first, so the
        // channels are read back to front.
        let source = CHANNELS - 1 - c;
        for y in 0..SIDE {
            for x in 0..SIDE {
                let at = ((top + y) * scaled_w + left + x) * CHANNELS + source;
                data[(c * SIDE + y) * SIDE + x] =
                    scale * f32::from(resized[at]) + shift;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, CHANNELS, SIDE, SIDE), device)?)
}

/// Turns a page counter-clockwise by a multiple of a right angle.
///
/// A quarter turn transposes the page, so the caller gets a page of the other
/// shape back; a half turn keeps it. The turn is exact — whole pixels moved,
/// nothing resampled — which is the whole reason orientation is settled before
/// anything else touches the page.
pub fn turn(page: &Page, degrees: u16) -> Page {
    let (width, height) = (page.width as usize, page.height as usize);
    let row = width * CHANNELS;
    match degrees % 360 {
        0 => page.clone(),
        90 => {
            // Counter-clockwise: the last column becomes the first row.
            let mut bgr = vec![0u8; page.bgr.len()];
            for y in 0..height {
                for x in 0..width {
                    let from = y * row + x * CHANNELS;
                    let to = ((width - 1 - x) * height + y) * CHANNELS;
                    bgr[to..to + CHANNELS]
                        .copy_from_slice(&page.bgr[from..from + CHANNELS]);
                }
            }
            Page {
                width: page.height,
                height: page.width,
                bgr,
            }
        },
        180 => {
            let mut bgr = vec![0u8; page.bgr.len()];
            for y in 0..height {
                for x in 0..width {
                    let from = y * row + x * CHANNELS;
                    let to =
                        ((height - 1 - y) * width + width - 1 - x) * CHANNELS;
                    bgr[to..to + CHANNELS]
                        .copy_from_slice(&page.bgr[from..from + CHANNELS]);
                }
            }
            Page {
                width: page.width,
                height: page.height,
                bgr,
            }
        },
        270 => {
            let mut bgr = vec![0u8; page.bgr.len()];
            for y in 0..height {
                for x in 0..width {
                    let from = y * row + x * CHANNELS;
                    let to = (x * height + height - 1 - y) * CHANNELS;
                    bgr[to..to + CHANNELS]
                        .copy_from_slice(&page.bgr[from..from + CHANNELS]);
                }
            }
            Page {
                width: page.height,
                height: page.width,
                bgr,
            }
        },
        other => {
            debug_assert!(false, "{other} is not a right angle");
            page.clone()
        },
    }
}

/// Where a point of the turned page sat on the page before the turn.
///
/// The turn is what the pipeline applies before anything is detected, so every
/// quadrangle it finds is a quadrangle of the turned page; this is what puts
/// them back on the page the caller handed in. `width` and `height` are the
/// *original* page's.
pub fn unturn(
    point: (f32, f32),
    degrees: u16,
    width: f32,
    height: f32,
) -> (f32, f32) {
    let (x, y) = point;
    match degrees % 360 {
        0 => (x, y),
        90 => (width - 1.0 - y, x),
        180 => (width - 1.0 - x, height - 1.0 - y),
        270 => (y, height - 1.0 - x),
        _ => (x, y),
    }
}
