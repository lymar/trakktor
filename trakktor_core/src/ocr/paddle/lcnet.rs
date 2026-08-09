//! The PP-LCNet v1 backbone, shared by the two orientation classifiers.
//!
//! One network, two jobs: deciding whether a text-line crop reads bottom to
//! top ([`super::cls`]), and deciding which of the four right angles a whole
//! page is at ([`crate::ocr::preprocess::orientation`]). The published
//! artifacts are the same graph to the op — same stem, same thirteen blocks,
//! same two gates, same 1280-wide closing convolution — and differ in exactly
//! two places:
//!
//! * **which axes a stride is spent on.** A page is halved along both axes at
//!   each of the four strided blocks, 224 down to 7. A text-line crop is 160
//!   pixels long and must stay that long, so the same four blocks spend the
//!   height alone, 80 down to 3.
//! * **how many classes the head reads**, which is the classifier's own
//!   business and not this module's.
//!
//! Two details of the loading are worth stating because both are silent when
//! wrong. Convolutions are numbered in the order the modules declare them, so
//! a gate's pair of convolutions pushes the pointwise convolution after it
//! along by two, while the normalizations — which gates do not have — keep
//! their own count. And the channel widths are read off the weights rather
//! than tabulated, because upstream publishes this backbone at more than one
//! scale and reading is both shorter and safer than guessing which scale a
//! directory holds.

use candle_core::Tensor;

use super::net::{
    BatchNorm, Conv, HARD_SIGMOID_SLOPE, Loader, SqueezeExcite, hardswish,
    subsample,
};
use crate::ocr::error::OcrError;

/// The width of the closing convolution, which no scale changes.
pub const FEATURES: usize = 1280;

/// What the graph's inference-time dropout multiplies the features by.
///
/// Paddle's `downgrade_in_infer` dropout is not a no-op at inference: it
/// scales by `1 - p` instead of scaling during training. The published graphs
/// carry `p = 0.2`, so the features arrive at the classifier four fifths of
/// their size. Dropping this is the single easiest way to get plausible but
/// wrong probabilities out of either model.
const DROPOUT_KEEP: f64 = 0.8;

/// Which axes a strided block spends its stride on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stride {
    /// Halve both axes, as a square input can afford.
    Both,
    /// Halve the height and keep the length, as a text line needs.
    Height,
}

impl Stride {
    fn of(self, downsample: bool) -> (usize, usize) {
        match (self, downsample) {
            (_, false) => (1, 1),
            (Self::Both, true) => (2, 2),
            (Self::Height, true) => (2, 1),
        }
    }
}

/// A convolution with batch normalization and a hard-swish, which is every
/// convolution in this backbone but the last.
#[derive(Debug)]
struct Layer {
    conv: Conv,
    /// What is left of the stride after the convolution has taken its share.
    stride: (usize, usize),
    norm: BatchNorm,
}

impl Layer {
    fn load(
        loader: &Loader,
        conv: &str,
        norm: &str,
        dims: [usize; 4],
        stride: (usize, usize),
        groups: usize,
    ) -> Result<Self, OcrError> {
        // A stride that is the same along both axes is the convolution's own;
        // the per-axis ones a text line needs are taken from its output
        // instead.
        let (strided, kept) = if stride.0 == stride.1 {
            (stride.0, (1, 1))
        } else {
            (1, stride)
        };
        Ok(Self {
            conv: Conv::load(loader, conv, dims, strided, dims[2] / 2, groups)?,
            stride: kept,
            norm: BatchNorm::load(loader, norm, dims[0])?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = subsample(&self.conv.forward(x)?, self.stride)?;
        hardswish(&self.norm.forward(&y)?)
    }
}

/// A backbone block: depthwise, optional gate, pointwise.
#[derive(Debug)]
struct Block {
    depthwise: Layer,
    excite: Option<SqueezeExcite>,
    pointwise: Layer,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.depthwise.forward(x)?;
        let y = match &self.excite {
            None => y,
            Some(excite) => excite.forward(&y)?,
        };
        self.pointwise.forward(&y)
    }
}

/// One row of the backbone's block table.
struct BlockSpec {
    kernel: usize,
    downsample: bool,
    excite: bool,
}

const BLOCKS: [BlockSpec; 13] = [
    BlockSpec {
        kernel: 3,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        downsample: true,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        downsample: true,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        downsample: true,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        downsample: true,
        excite: true,
    },
    BlockSpec {
        kernel: 5,
        downsample: false,
        excite: true,
    },
];

/// The backbone: everything up to and including the 1280-wide features a
/// classifier's own head reads.
#[derive(Debug)]
pub struct Backbone {
    stem: Layer,
    blocks: Vec<Block>,
    last: Conv,
}

impl Backbone {
    pub fn load(loader: &Loader, stride: Stride) -> Result<Self, OcrError> {
        let mut channels = out_channels(loader, "conv2d_0")?;
        let stem = Layer::load(
            loader,
            "conv2d_0",
            "batch_norm2d_0",
            [channels, 3, 3, 3],
            (2, 2),
            1,
        )?;

        let mut conv = 1;
        let mut norm = 1;
        let mut blocks = Vec::with_capacity(BLOCKS.len());
        for spec in &BLOCKS {
            let depthwise = Layer::load(
                loader,
                &format!("conv2d_{conv}"),
                &format!("batch_norm2d_{norm}"),
                [channels, 1, spec.kernel, spec.kernel],
                stride.of(spec.downsample),
                channels,
            )?;
            conv += 1;
            norm += 1;
            let excite = if spec.excite {
                let gate = SqueezeExcite::load(
                    loader,
                    &format!("conv2d_{conv}"),
                    &format!("conv2d_{}", conv + 1),
                    channels,
                    channels / 4,
                    HARD_SIGMOID_SLOPE,
                )?;
                conv += 2;
                Some(gate)
            } else {
                None
            };
            let out = out_channels(loader, &format!("conv2d_{conv}"))?;
            let pointwise = Layer::load(
                loader,
                &format!("conv2d_{conv}"),
                &format!("batch_norm2d_{norm}"),
                [out, channels, 1, 1],
                (1, 1),
                1,
            )?;
            conv += 1;
            norm += 1;
            channels = out;
            blocks.push(Block {
                depthwise,
                excite,
                pointwise,
            });
        }

        Ok(Self {
            stem,
            blocks,
            last: Conv::load(
                loader,
                &format!("conv2d_{conv}"),
                [FEATURES, channels, 1, 1],
                1,
                0,
                1,
            )?,
        })
    }

    /// The features for a batch of normalized images, `[batch, 1280]`.
    ///
    /// The inference-time dropout is applied here, because it belongs to the
    /// backbone's tail and every head that reads these features has it.
    pub fn features(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let batch = x.dim(0)?;
        let mut y = self.stem.forward(x)?;
        for block in &self.blocks {
            y = block.forward(&y)?;
        }
        let pooled = y.mean_keepdim(3)?.mean_keepdim(2)?;
        let features = hardswish(&self.last.forward(&pooled)?)?;
        Ok(features
            .affine(DROPOUT_KEEP, 0.0)?
            .reshape((batch, FEATURES))?)
    }
}

/// The out-channel count a convolution's weight declares.
fn out_channels(loader: &Loader, name: &str) -> Result<usize, OcrError> {
    let dims = loader.dims(&format!("{name}.w_0"))?;
    match dims.first() {
        Some(out) if dims.len() == 4 => Ok(*out),
        _ => Err(OcrError::Artifact(format!(
            "`{name}.w_0` is {dims:?}, which is not a convolution weight"
        ))),
    }
}
