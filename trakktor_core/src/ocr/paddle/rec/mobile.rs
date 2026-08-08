//! The small recognizer's backbone: PP-LCNetV3 in its recognition variant.
//!
//! This is the detector's backbone family with its strides rearranged: reading
//! a text line means keeping its length and spending its height. The height
//! falls 48 → 24 (the stem) → 12 → 6 → 3, while the width is halved exactly
//! once inside the blocks, by the sixth. A backbone copied from the detector
//! would halve the width four times over and leave a sequence far too short to
//! hold the text.

use candle_core::Tensor;

use super::head::Naming;
use crate::ocr::{
    error::OcrError,
    paddle::net::{
        Affine, BatchNorm, Conv, HARD_SIGMOID_SLOPE, Loader, SqueezeExcite,
        hardswish, subsample,
    },
};

/// The width this backbone hands to the head, and where the head's own weights
/// are numbered from.
///
/// The gap between the backbone's last convolution and the head's first is the
/// classification layer the exported graph declares and never runs.
pub const NAMING: Naming = Naming {
    width: 480,
    conv: 131,
    norm: 146,
    linear: 0,
};

/// One collapsed rep-layer: a convolution, its own pair of learned scalars, an
/// activation and a second pair.
///
/// Unlike the detector's, none of these layers skips its activation. Upstream
/// drops the activation of a layer whose stride is the number two, and the
/// recognition variant strides one axis at a time — a pair that is never that
/// number — so all twenty-eight of them keep it.
#[derive(Debug)]
struct RepLayer {
    conv: Conv,
    stride: (usize, usize),
    affine: Affine,
    act: Affine,
}

impl RepLayer {
    /// `affine` is the index of the layer's own pair of scalars; the
    /// activation's pair is always the one after it.
    fn load(
        loader: &Loader,
        conv: &str,
        dims: [usize; 4],
        stride: (usize, usize),
        padding: usize,
        groups: usize,
        affine: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(loader, conv, dims, 1, padding, groups)?,
            stride,
            affine: Affine::load(loader, affine)?,
            act: Affine::load(loader, affine + 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = subsample(&self.conv.forward(x)?, self.stride)?;
        let y = self.affine.forward(&y)?;
        self.act.forward(&hardswish(&y)?)
    }
}

/// A backbone block: depthwise, optional gate, pointwise.
#[derive(Debug)]
struct Block {
    depthwise: RepLayer,
    excite: Option<SqueezeExcite>,
    pointwise: RepLayer,
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

/// One row of the backbone's block table. The stride is per-axis: reading a
/// text line means keeping its length and spending its height.
struct BlockSpec {
    /// Depthwise kernel size.
    kernel: usize,
    stride: (usize, usize),
    in_channels: usize,
    out_channels: usize,
    excite: bool,
}

/// The fourteen blocks of the recognition backbone, with the channel counts its
/// scale works out to. Weight names run in step with the index: the depthwise
/// convolution of block `i` is `conv2d_{136 + 2i}`, the pointwise one follows
/// it, and the four learned-scalar blocks are `4i .. 4i + 3`.
const BLOCKS: [BlockSpec; 14] = [
    BlockSpec {
        kernel: 3,
        stride: (1, 1),
        in_channels: 16,
        out_channels: 32,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: (1, 1),
        in_channels: 32,
        out_channels: 64,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: (1, 1),
        in_channels: 64,
        out_channels: 64,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: (2, 1),
        in_channels: 64,
        out_channels: 128,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: (1, 1),
        in_channels: 128,
        out_channels: 128,
        excite: false,
    },
    // The one width reduction, and the last block of this stage to keep a 3x3
    // depthwise kernel.
    BlockSpec {
        kernel: 3,
        stride: (1, 2),
        in_channels: 128,
        out_channels: 240,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 240,
        out_channels: 240,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 240,
        out_channels: 240,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 240,
        out_channels: 240,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 240,
        out_channels: 240,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (2, 1),
        in_channels: 240,
        out_channels: 480,
        excite: true,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 480,
        out_channels: 480,
        excite: true,
    },
    BlockSpec {
        kernel: 5,
        stride: (2, 1),
        in_channels: 480,
        out_channels: 480,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: (1, 1),
        in_channels: 480,
        out_channels: 480,
        excite: false,
    },
];

/// The two blocks that carry a squeeze-and-excitation gate name their
/// convolutions out of step with everything else.
const EXCITE_CONVS: [(usize, &str, &str); 2] = [
    (10, "conv2d_96", "conv2d_97"),
    (11, "conv2d_107", "conv2d_108"),
];

/// The backbone.
#[derive(Debug)]
pub struct Net {
    stem: Conv,
    stem_norm: BatchNorm,
    blocks: Vec<Block>,
}

impl Net {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let stem = Conv::load(loader, "conv2d_0", [16, 3, 3, 3], 2, 1, 1)?;
        let stem_norm = BatchNorm::load(loader, "batch_norm2d_0", 16)?;

        let mut blocks = Vec::with_capacity(BLOCKS.len());
        for (i, spec) in BLOCKS.iter().enumerate() {
            let depthwise = RepLayer::load(
                loader,
                &format!("conv2d_{}", 136 + 2 * i),
                [spec.in_channels, 1, spec.kernel, spec.kernel],
                spec.stride,
                spec.kernel / 2,
                spec.in_channels,
                4 * i,
            )?;
            let pointwise = RepLayer::load(
                loader,
                &format!("conv2d_{}", 137 + 2 * i),
                [spec.out_channels, spec.in_channels, 1, 1],
                (1, 1),
                0,
                1,
                4 * i + 2,
            )?;
            let excite = if spec.excite {
                let (_, down, up) = EXCITE_CONVS
                    .iter()
                    .find(|(block, _, _)| *block == i)
                    .ok_or_else(|| {
                        OcrError::Artifact(format!(
                            "block {i} wants a squeeze-and-excitation gate \
                             that the model does not name"
                        ))
                    })?;
                Some(SqueezeExcite::load(
                    loader,
                    down,
                    up,
                    spec.in_channels,
                    spec.in_channels / 4,
                    HARD_SIGMOID_SLOPE,
                )?)
            } else {
                None
            };
            blocks.push(Block {
                depthwise,
                excite,
                pointwise,
            });
        }
        Ok(Self {
            stem,
            stem_norm,
            blocks,
        })
    }

    /// Runs the backbone over `[batch, 3, 48, width]` and returns the pooled
    /// feature map, `[batch, 480, 1, width / 8]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        // The stem carries no activation, unlike every convolution after it.
        let mut y = self.stem_norm.forward(&self.stem.forward(x)?)?;
        for block in &self.blocks {
            y = block.forward(&y)?;
        }
        // Three rows into one and two columns into one: the feature map
        // becomes a single row, and the sequence gets its length.
        Ok(y.avg_pool2d((3, 2))?)
    }
}
