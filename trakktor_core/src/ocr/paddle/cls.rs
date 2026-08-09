//! The text-line orientation classifier: `PP-LCNet_x1_0_textline_ori` and its
//! narrower sibling, which differ only in how wide their channels are.
//!
//! Detection finds text lines, not their direction. A line that runs the other
//! way up the page — a caption under a rotated figure, a page scanned upside
//! down — is straightened into a crop that still reads bottom to top, and the
//! recognizer would make noise of it. This classifier looks at a crop and says
//! whether it is the right way up.
//!
//! Three of its details are its own, and each is worth stating once:
//!
//! - **It eats RGB** where the detector and the recognizer eat BGR, and it
//!   scales its input to a fixed 80x160 with no regard for the aspect ratio.
//! - **Its exported graph keeps a training dropout**, which Paddle evaluates as
//!   a plain multiplication by `1 - p`. Leaving that factor out moves the
//!   probabilities by enough to flip a borderline crop.
//! - **The decision is a threshold, not an arg-max.** A crop is turned around
//!   only when the classifier is [`THRESHOLD`] sure of it. PaddleOCR's own
//!   pipeline turns one around on a bare majority, and a majority of two
//!   classes is not evidence — a crop it cannot read either way would be
//!   flipped on noise.

#[cfg(test)]
mod tests;

use candle_core::{Device, Tensor};

use super::{
    crop::Crop,
    image::{CHANNELS, resize_linear},
    lcnet::{Backbone, FEATURES, Stride},
    net::{Linear, Loader},
    rec::readable,
};
use crate::ocr::error::OcrError;

/// The size every crop is scaled to, aspect ratio and all.
pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 80;

/// Crops per forward pass.
pub const BATCH: usize = 6;

/// How sure the classifier must be before a crop is turned around.
pub const THRESHOLD: f32 = 0.9;

/// The two classes: upright, and turned around.
const CLASSES: usize = 2;
/// The class that means "turned around". With two classes, a probability above
/// [`THRESHOLD`] is also the arg-max, so one comparison settles both.
const UPSIDE_DOWN: usize = 1;

/// The normalization the model was trained with: the ImageNet statistics,
/// applied to the `0..=1` scale.
const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const STD: [f64; 3] = [0.229, 0.224, 0.225];

/// The text-line orientation classifier.
#[derive(Debug)]
pub struct Classifier {
    device: Device,
    backbone: Backbone,
    head: Linear,
}

impl Classifier {
    /// Reads the weights.
    ///
    /// Everything but the head is [`Backbone`], which the page-orientation
    /// classifier shares — the two published text-line classifiers are that
    /// same network at two scales, and it reads its own channel widths off the
    /// file rather than out of a table.
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let classes = loader.artifact().output_classes().ok_or_else(|| {
            OcrError::Artifact(
                "the orientation classifier does not declare how many classes \
                 it reads"
                    .into(),
            )
        })?;
        if classes != CLASSES {
            return Err(OcrError::Artifact(format!(
                "the orientation classifier reads {classes} classes; this \
                 port knows the {CLASSES}-class one, upright against turned \
                 around"
            )));
        }

        Ok(Self {
            device: loader.device().clone(),
            backbone: Backbone::load(loader, Stride::Height)?,
            head: Linear::load(loader, "linear_0", FEATURES, CLASSES)?,
        })
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device { &self.device }

    /// Runs the network over a batch of normalized crops.
    ///
    /// The input is `[batch, 3, 80, 160]`, red channel first; the result is
    /// `[batch, 2]` of probabilities — as with the recognizer, the softmax is
    /// inside the exported graph.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let (batch, _, height, width) = x.dims4()?;
        if height != HEIGHT || width != WIDTH {
            return Err(OcrError::Runtime(format!(
                "the orientation classifier reads {HEIGHT}x{WIDTH} crops, got \
                 {height}x{width}"
            )));
        }

        let logits = self.head.forward(&self.backbone.features(x)?)?;
        debug_assert_eq!(logits.dim(0).ok(), Some(batch));
        Ok(candle_nn::ops::softmax_last_dim(&logits)?)
    }

    /// How sure the classifier is that each crop is upside down.
    ///
    /// A crop with no pixels is never classified, and gets a zero.
    pub fn probabilities(&self, crops: &[Crop]) -> Result<Vec<f32>, OcrError> {
        let mut scores = vec![0f32; crops.len()];
        let usable: Vec<usize> =
            (0..crops.len()).filter(|i| readable(&crops[*i])).collect();
        for group in usable.chunks(BATCH) {
            let batch: Vec<&Crop> = group.iter().map(|i| &crops[*i]).collect();
            let input = batch_tensor(&batch, &self.device)?;
            let probabilities = self.forward(&input)?;
            let flat = probabilities.flatten_all()?.to_vec1::<f32>()?;
            for (row, index) in group.iter().enumerate() {
                scores[*index] = flat[row * CLASSES + UPSIDE_DOWN];
            }
        }
        Ok(scores)
    }

    /// Which of these crops should be turned around before they are read.
    pub fn upside_down(&self, crops: &[Crop]) -> Result<Vec<bool>, OcrError> {
        Ok(self
            .probabilities(crops)?
            .into_iter()
            .map(|score| score > THRESHOLD)
            .collect())
    }
}

/// Turns a batch of crops into the tensor the network reads: `[batch, 3, 80,
/// 160]`, red channel first.
///
/// Each crop is squeezed to that size whatever its shape — no padding, no
/// letterboxing — and then normalized by the ImageNet statistics as one
/// multiply-add per channel, which is how the reference folds them.
pub fn batch_tensor(
    crops: &[&Crop],
    device: &Device,
) -> Result<Tensor, OcrError> {
    let mut data = vec![0f32; crops.len() * CHANNELS * HEIGHT * WIDTH];
    for (n, crop) in crops.iter().enumerate() {
        if !readable(crop) {
            continue;
        }
        let resized = resize_linear(
            &crop.bgr,
            crop.width,
            crop.height,
            CHANNELS,
            WIDTH,
            HEIGHT,
        );
        for c in 0..CHANNELS {
            let scale = (1.0 / 255.0 / STD[c]) as f32;
            let shift = (-MEAN[c] / STD[c]) as f32;
            // The crop arrives blue first and the model wants red first, so
            // the channels are read back to front.
            let source = CHANNELS - 1 - c;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let value =
                        f32::from(resized[(y * WIDTH + x) * CHANNELS + source]);
                    data[((n * CHANNELS + c) * HEIGHT + y) * WIDTH + x] =
                        scale * value + shift;
                }
            }
        }
    }
    let shape = (crops.len(), CHANNELS, HEIGHT, WIDTH);
    Ok(Tensor::from_vec(data, shape, device)?)
}

/// Turns a crop around: a true 180-degree flip, row order and column order
/// both reversed.
///
/// This is a deliberate divergence from the reference, which rotates about the
/// centre of the crop's *extent* rather than of its pixels and so lands one
/// pixel down and to the right of a flip, with a black row along the top and a
/// black column down the left. Reproducing that would mean feeding the
/// recognizer a crop that is a pixel off and edged in black; a flip is what
/// the operation means.
pub fn turn_around(crop: &Crop) -> Crop {
    let mut bgr = vec![0u8; crop.bgr.len()];
    let row = crop.width * CHANNELS;
    for y in 0..crop.height {
        for x in 0..crop.width {
            let from = (y * crop.width + x) * CHANNELS;
            let to = ((crop.height - 1 - y) * crop.width + crop.width - 1 - x) *
                CHANNELS;
            bgr[to..to + CHANNELS]
                .copy_from_slice(&crop.bgr[from..from + CHANNELS]);
        }
    }
    debug_assert_eq!(bgr.len(), crop.height * row);
    Crop {
        width: crop.width,
        height: crop.height,
        bgr,
    }
}
