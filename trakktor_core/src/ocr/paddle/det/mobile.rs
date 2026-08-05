//! `PP-OCRv5_mobile_det`: a PP-LCNetV3 backbone under an RSE feature-pyramid
//! neck and a differentiable-binarization head.
//!
//! The published weights are already reparameterized: what was a stack of
//! parallel branches during training is one convolution per layer here, so the
//! port implements only the collapsed form. Two consequences show up as
//! irregularities rather than as structure — a layer that strides by two skips
//! its activation entirely (and its two activation scalars are absent from the
//! file), and the head's second branch, used only by the training loss, is not
//! in the graph at all.

use candle_core::Tensor;

use crate::ocr::{
    error::OcrError,
    paddle::net::{
        Affine, BatchNorm, Conv, ConvTranspose, HARD_SIGMOID_SLOPE, Loader,
        NECK_HARD_SIGMOID_SLOPE, SqueezeExcite, hardswish, relu, upsample,
    },
};

/// One collapsed rep-layer: a convolution, its own pair of learned scalars
/// and — unless it strides by two — an activation with a second pair.
#[derive(Debug)]
struct RepLayer {
    conv: Conv,
    affine: Affine,
    act: Option<Affine>,
}

impl RepLayer {
    fn load(
        loader: &Loader,
        conv: &str,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        groups: usize,
        affine: usize,
        act: Option<usize>,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(loader, conv, dims, stride, padding, groups)?,
            affine: Affine::load(loader, affine)?,
            act: act.map(|i| Affine::load(loader, i)).transpose()?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.conv.forward(x)?;
        let y = self.affine.forward(&y)?;
        match &self.act {
            None => Ok(y),
            Some(act) => act.forward(&hardswish(&y)?),
        }
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

/// One row of the backbone's block table.
struct BlockSpec {
    /// Depthwise kernel size.
    kernel: usize,
    stride: usize,
    in_channels: usize,
    out_channels: usize,
    excite: bool,
}

/// The fourteen blocks of `PPLCNetV3(scale=0.75)`, with the channel counts the
/// scale works out to. Weight names run in step with the index: the depthwise
/// convolution of block `i` is `conv2d_{161 + 2i}`, the pointwise one follows
/// it, and the four learned-scalar blocks are `4i .. 4i + 3` — with `4i + 1`
/// absent whenever the depthwise layer strides by two.
const BLOCKS: [BlockSpec; 14] = [
    BlockSpec {
        kernel: 3,
        stride: 1,
        in_channels: 16,
        out_channels: 32,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: 2,
        in_channels: 32,
        out_channels: 48,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: 1,
        in_channels: 48,
        out_channels: 48,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: 2,
        in_channels: 48,
        out_channels: 96,
        excite: false,
    },
    BlockSpec {
        kernel: 3,
        stride: 1,
        in_channels: 96,
        out_channels: 96,
        excite: false,
    },
    // The first block of this stage keeps a 3x3 depthwise kernel; the rest of
    // the stage moves to 5x5.
    BlockSpec {
        kernel: 3,
        stride: 2,
        in_channels: 96,
        out_channels: 192,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 192,
        out_channels: 192,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 192,
        out_channels: 192,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 192,
        out_channels: 192,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 192,
        out_channels: 192,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 2,
        in_channels: 192,
        out_channels: 384,
        excite: true,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 384,
        out_channels: 384,
        excite: true,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 384,
        out_channels: 384,
        excite: false,
    },
    BlockSpec {
        kernel: 5,
        stride: 1,
        in_channels: 384,
        out_channels: 384,
        excite: false,
    },
];

/// The blocks whose output the neck consumes (the last block of each stage
/// after the first), and the width the tap convolution reduces it to.
const TAPS: [(usize, usize, usize); 4] =
    [(2, 48, 12), (4, 96, 18), (9, 192, 42), (13, 384, 360)];

/// The two blocks that carry a squeeze-and-excitation gate name their
/// convolutions out of step with everything else.
const EXCITE_CONVS: [(usize, &str, &str); 2] = [
    (10, "conv2d_96", "conv2d_97"),
    (11, "conv2d_107", "conv2d_108"),
];

/// Width the neck works in.
const NECK: usize = 96;
/// Width of each of the four pyramid branches (`NECK / 4`).
const BRANCH: usize = 24;

/// A neck layer: a convolution with no bias, gated and added back to itself.
#[derive(Debug)]
struct RseLayer {
    conv: Conv,
    excite: SqueezeExcite,
}

impl RseLayer {
    fn load(
        loader: &Loader,
        first: usize,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
    ) -> Result<Self, OcrError> {
        let reduced = out_channels / 4;
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{first}"),
                [out_channels, in_channels, kernel, kernel],
                1,
                kernel / 2,
                1,
            )?,
            excite: SqueezeExcite::load(
                loader,
                &format!("conv2d_{}", first + 1),
                &format!("conv2d_{}", first + 2),
                out_channels,
                reduced,
                NECK_HARD_SIGMOID_SLOPE,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.conv.forward(x)?;
        let gated = self.excite.forward(&y)?;
        Ok((&y + &gated)?)
    }
}

/// The network.
#[derive(Debug)]
pub struct Net {
    stem: Conv,
    stem_norm: BatchNorm,
    blocks: Vec<Block>,
    taps: Vec<Conv>,
    lateral: Vec<RseLayer>,
    branches: Vec<RseLayer>,
    head_conv: Conv,
    head_norm: BatchNorm,
    head_up1: ConvTranspose,
    head_norm2: BatchNorm,
    head_up2: ConvTranspose,
}

impl Net {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let stem = Conv::load(loader, "conv2d_0", [16, 3, 3, 3], 2, 1, 1)?;
        let stem_norm = BatchNorm::load(loader, "batch_norm2d_0", 16)?;

        let mut blocks = Vec::with_capacity(BLOCKS.len());
        for (i, spec) in BLOCKS.iter().enumerate() {
            let depthwise = RepLayer::load(
                loader,
                &format!("conv2d_{}", 161 + 2 * i),
                [spec.in_channels, 1, spec.kernel, spec.kernel],
                spec.stride,
                spec.kernel / 2,
                spec.in_channels,
                4 * i,
                // A stride-2 layer skips its activation, and the scalars for
                // that activation are not in the file.
                (spec.stride != 2).then_some(4 * i + 1),
            )?;
            let pointwise = RepLayer::load(
                loader,
                &format!("conv2d_{}", 162 + 2 * i),
                [spec.out_channels, spec.in_channels, 1, 1],
                1,
                0,
                1,
                4 * i + 2,
                Some(4 * i + 3),
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

        // The four taps reduce the backbone's stages to the widths the neck
        // expects; these are plain scaled widths, not rounded ones.
        let mut taps = Vec::with_capacity(TAPS.len());
        for (i, (_, from, to)) in TAPS.iter().enumerate() {
            taps.push(Conv::load(
                loader,
                &format!("conv2d_{}", 131 + i),
                [*to, *from, 1, 1],
                1,
                0,
                1,
            )?);
        }

        let mut lateral = Vec::with_capacity(4);
        let mut branches = Vec::with_capacity(4);
        for (i, (_, _, width)) in TAPS.iter().enumerate() {
            lateral.push(RseLayer::load(loader, 135 + 6 * i, *width, NECK, 1)?);
            branches.push(RseLayer::load(
                loader,
                138 + 6 * i,
                NECK,
                BRANCH,
                3,
            )?);
        }

        Ok(Self {
            stem,
            stem_norm,
            blocks,
            taps,
            lateral,
            branches,
            head_conv: Conv::load(
                loader,
                "conv2d_159",
                [BRANCH, NECK, 3, 3],
                1,
                1,
                1,
            )?,
            head_norm: BatchNorm::load(loader, "batch_norm_0", BRANCH)?,
            head_up1: ConvTranspose::load(
                loader,
                "conv2d_transpose_0",
                [BRANCH, BRANCH, 2, 2],
                2,
            )?,
            head_norm2: BatchNorm::load(loader, "batch_norm_1", BRANCH)?,
            head_up2: ConvTranspose::load(
                loader,
                "conv2d_transpose_1",
                [BRANCH, 1, 2, 2],
                2,
            )?,
        })
    }

    /// Runs the network over one normalized page, returning the probability
    /// map as `[1, 1, height, width]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        // The stem is a bare convolution and normalization: no activation and
        // no learned scalars follow it, which is why the graph holds 24
        // activations for 28 rep-layers rather than 25.
        let mut y = self.stem.forward(x)?;
        y = self.stem_norm.forward(&y)?;

        let mut stages = Vec::with_capacity(TAPS.len());
        for (i, block) in self.blocks.iter().enumerate() {
            y = block.forward(&y)?;
            if let Some(tap) = TAPS.iter().position(|(at, _, _)| *at == i) {
                stages.push(self.taps[tap].forward(&y)?);
            }
        }

        // Top-down pyramid: each level is its own lateral projection plus the
        // coarser level, doubled.
        let mut lateral: Vec<Tensor> = Vec::with_capacity(stages.len());
        for (layer, stage) in self.lateral.iter().zip(&stages) {
            lateral.push(layer.forward(stage)?);
        }
        let mut merged = vec![lateral[3].clone()];
        for level in (0..3).rev() {
            let coarser = upsample(merged.last().expect("a coarser level"), 2)?;
            merged.push((&lateral[level] + &coarser)?);
        }
        merged.reverse(); // finest first

        // Each level goes through its own branch and is brought back to the
        // finest scale; the concatenation is coarsest first.
        let mut branches = Vec::with_capacity(4);
        for (level, (layer, merged)) in
            self.branches.iter().zip(&merged).enumerate()
        {
            let branch = layer.forward(merged)?;
            branches.push(if level == 0 {
                branch
            } else {
                upsample(&branch, 1 << level)?
            });
        }
        branches.reverse();
        let fused = Tensor::cat(&branches, 1)?;

        let y = self.head_conv.forward(&fused)?;
        let y = relu(&self.head_norm.forward(&y)?)?;
        let y = self.head_up1.forward(&y)?;
        let y = relu(&self.head_norm2.forward(&y)?)?;
        let y = self.head_up2.forward(&y)?;
        Ok(candle_nn::ops::sigmoid(&y)?)
    }
}
