//! The PP-HGNetV2 backbone, shared by the layout model, the server text
//! detector and the server text recognizer.
//!
//! All three carry the *same* backbone — upstream spells it `PPHGNetV2-L` in
//! one configuration and `PPHGNetV2_B4` in the others, and they work out to the
//! same stem, the same four stages and the same channel counts. What differs is
//! [where the resolution is spent](Shape) and what is read off the result: the
//! layout model takes the last three stages, the detector all four, the
//! recognizer only the last.
//!
//! The structure is ordinary — convolutions, batch normalization and ReLU —
//! with two habits worth naming, because both are silent when got wrong:
//!
//! - **A block keeps everything it computes.** Its six layers run in series,
//!   and the input plus all six outputs are concatenated before being squeezed
//!   back down. That is where the wide 1×1 convolutions come from (2176 input
//!   channels in the third stage), and it is why the layers cannot be fused.
//! - **In the deeper stages a layer is two convolutions, and only the second
//!   one has an activation.** The pointwise half is bare; the depthwise half
//!   that follows it carries the ReLU.

use candle_core::Tensor;

use super::net::{ConvBn, Loader, pad_end};
use crate::ocr::error::OcrError;

/// The size the backbone's four stages divide the page by, in the
/// [`Shape::Page`] shape. The coarsest output is a thirty-second of the input,
/// so a side that is not a multiple of it would leave the pyramid's levels
/// disagreeing by a row.
pub const SIZE_MULTIPLE: usize = 32;

/// The two shapes the backbone is published in.
///
/// The weights are laid out identically and the two are numbered alike; what
/// differs is where the resolution goes. Reading a **page** halves both axes
/// together, five times over, so a line of text ends up a few pixels of a
/// feature map. Reading a **line** spends the height and keeps the length: the
/// height falls 48 → 3 while the width is halved twice, because the width *is*
/// the sequence the recognizer reads and shortening it would throw characters
/// away.
///
/// The difference amounts to one stride per stage plus the stem's second
/// strided convolution, which the line shape does not stride at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Detection and layout: a thirty-second of the page at the coarsest stage.
    Page,
    /// Recognition: a sixteenth of the height and a quarter of the length.
    Line,
}

impl Shape {
    /// The stride of the stem's second strided convolution — the one after the
    /// two-way split.
    fn stem_stride(self) -> usize {
        match self {
            Self::Page => 2,
            Self::Line => 1,
        }
    }

    /// What a stage opens with, or `None` when it opens straight into its
    /// blocks.
    fn downsample(self, stage: usize) -> Option<(usize, usize)> {
        match self {
            // The first stage inherits the stem's quarter resolution and is
            // left alone; the rest halve both axes.
            Self::Page => (stage > 0).then_some((2, 2)),
            // Every stage strides, and exactly one of them — the second —
            // spends the width instead of the height.
            Self::Line => Some(if stage == 1 { (1, 2) } else { (2, 1) }),
        }
    }
}

fn relu(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.relu()?) }

/// The stem: two strided convolutions with a two-way split in between.
///
/// The split is the only part that needs care. Two kernels of size two run over
/// the first convolution's output padded by one on the right and the bottom,
/// while the same padded tensor goes through a max pool of the same shape; the
/// two results are concatenated.
#[derive(Debug)]
struct Stem {
    first: ConvBn,
    left: ConvBn,
    left2: ConvBn,
    third: ConvBn,
    fourth: ConvBn,
}

impl Stem {
    fn load(
        loader: &Loader,
        next: &mut usize,
        bn_offset: usize,
        shape: Shape,
    ) -> Result<Self, OcrError> {
        let mut conv = |dims: [usize; 4], stride, padding, groups| {
            let at = *next;
            *next += 1;
            ConvBn::load(loader, at, bn_offset, dims, stride, padding, groups)
        };
        Ok(Self {
            first: conv([32, 3, 3, 3], 2, 1, 1)?,
            left: conv([16, 32, 2, 2], 1, 0, 1)?,
            left2: conv([32, 16, 2, 2], 1, 0, 1)?,
            third: conv([32, 64, 3, 3], shape.stem_stride(), 1, 1)?,
            fourth: conv([48, 32, 1, 1], 1, 0, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = relu(&self.first.forward(x)?)?;
        let padded = pad_end(&y, 1)?;
        let left = relu(&self.left.forward(&padded)?)?;
        let left = relu(&self.left2.forward(&pad_end(&left, 1)?)?)?;
        let right = padded.max_pool2d_with_stride(2, 1)?;
        let y = Tensor::cat(&[&right, &left], 1)?;
        let y = relu(&self.third.forward(&y)?)?;
        relu(&self.fourth.forward(&y)?)
    }
}

/// One block of a stage: six convolutions whose outputs are all kept, then
/// concatenated with the input and squeezed back down.
#[derive(Debug)]
struct HgBlock {
    layers: Vec<HgLayer>,
    squeeze: ConvBn,
    excite: ConvBn,
    residual: bool,
}

/// A layer inside a block: one convolution, or — in the deeper stages — a
/// pointwise convolution followed by a depthwise one.
#[derive(Debug)]
enum HgLayer {
    Plain(ConvBn),
    Light { point: ConvBn, depth: ConvBn },
}

impl HgLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        match self {
            Self::Plain(conv) => relu(&conv.forward(x)?),
            // The pointwise half carries no activation; only the depthwise one
            // does.
            Self::Light { point, depth } => {
                relu(&depth.forward(&point.forward(x)?)?)
            },
        }
    }
}

impl HgBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        loader: &Loader,
        next: &mut usize,
        bn_offset: usize,
        in_channels: usize,
        mid: usize,
        out: usize,
        kernel: usize,
        light: bool,
        residual: bool,
    ) -> Result<Self, OcrError> {
        const LAYERS: usize = 6;
        let pad = (kernel - 1) / 2;
        let mut layers = Vec::with_capacity(LAYERS);
        for at in 0..LAYERS {
            let from = if at == 0 { in_channels } else { mid };
            let mut conv = |dims: [usize; 4], stride, padding, groups| {
                let index = *next;
                *next += 1;
                ConvBn::load(
                    loader, index, bn_offset, dims, stride, padding, groups,
                )
            };
            layers.push(if light {
                HgLayer::Light {
                    point: conv([mid, from, 1, 1], 1, 0, 1)?,
                    depth: conv([mid, 1, kernel, kernel], 1, pad, mid)?,
                }
            } else {
                HgLayer::Plain(conv([mid, from, kernel, kernel], 1, pad, 1)?)
            });
        }
        let total = in_channels + LAYERS * mid;
        let squeeze = {
            let index = *next;
            *next += 1;
            ConvBn::load(
                loader,
                index,
                bn_offset,
                [out / 2, total, 1, 1],
                1,
                0,
                1,
            )?
        };
        let excite = {
            let index = *next;
            *next += 1;
            ConvBn::load(
                loader,
                index,
                bn_offset,
                [out, out / 2, 1, 1],
                1,
                0,
                1,
            )?
        };
        Ok(Self {
            layers,
            squeeze,
            excite,
            residual,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut kept = Vec::with_capacity(self.layers.len() + 1);
        kept.push(x.clone());
        let mut state = x.clone();
        for layer in &self.layers {
            state = layer.forward(&state)?;
            kept.push(state.clone());
        }
        let joined = Tensor::cat(&kept, 1)?;
        let y = relu(&self.squeeze.forward(&joined)?)?;
        let y = relu(&self.excite.forward(&y)?)?;
        if self.residual { Ok((y + x)?) } else { Ok(y) }
    }
}

/// One stage: an optional depthwise halving of the resolution, then its blocks.
#[derive(Debug)]
struct HgStage {
    downsample: Option<ConvBn>,
    blocks: Vec<HgBlock>,
}

/// The four stages, as the published configuration spells them. Both shapes
/// agree on every number here; they differ only in [`Shape::downsample`].
struct StageSpec {
    in_channels: usize,
    mid: usize,
    out: usize,
    blocks: usize,
    kernel: usize,
    light: bool,
}

const STAGES: [StageSpec; 4] = [
    StageSpec {
        in_channels: 48,
        mid: 48,
        out: 128,
        blocks: 1,
        kernel: 3,
        light: false,
    },
    StageSpec {
        in_channels: 128,
        mid: 96,
        out: 512,
        blocks: 1,
        kernel: 3,
        light: false,
    },
    StageSpec {
        in_channels: 512,
        mid: 192,
        out: 1024,
        blocks: 3,
        kernel: 5,
        light: true,
    },
    StageSpec {
        in_channels: 1024,
        mid: 384,
        out: 2048,
        blocks: 1,
        kernel: 5,
        light: true,
    },
];

/// The channels each stage ends on, finest first.
pub const STAGE_CHANNELS: [usize; 4] = [128, 512, 1024, 2048];

#[derive(Debug)]
pub struct Backbone {
    stem: Stem,
    stages: Vec<HgStage>,
}

impl Backbone {
    /// Loads the backbone in one of its two [shapes](Shape), advancing `next`
    /// past the convolutions it claims so that whatever the artifact numbers
    /// after it can carry on counting.
    pub fn load(
        loader: &Loader,
        next: &mut usize,
        bn_offset: usize,
        shape: Shape,
    ) -> Result<Self, OcrError> {
        let stem = Stem::load(loader, next, bn_offset, shape)?;
        let mut stages = Vec::with_capacity(STAGES.len());
        for (at, spec) in STAGES.iter().enumerate() {
            let downsample = match shape.downsample(at) {
                None => None,
                Some(stride) => {
                    let index = *next;
                    *next += 1;
                    // Depthwise, and with no activation of its own.
                    Some(ConvBn::load_axes(
                        loader,
                        index,
                        bn_offset,
                        [spec.in_channels, 1, 3, 3],
                        stride,
                        1,
                        spec.in_channels,
                    )?)
                },
            };
            let mut blocks = Vec::with_capacity(spec.blocks);
            for at in 0..spec.blocks {
                blocks.push(HgBlock::load(
                    loader,
                    next,
                    bn_offset,
                    if at == 0 { spec.in_channels } else { spec.out },
                    spec.mid,
                    spec.out,
                    spec.kernel,
                    spec.light,
                    at != 0,
                )?);
            }
            stages.push(HgStage { downsample, blocks });
        }
        Ok(Self { stem, stages })
    }

    /// The four stages' outputs, finest first.
    ///
    /// In the [page](Shape::Page) shape they are a quarter, an eighth, a
    /// sixteenth and a thirty-second of the input; in the
    /// [line](Shape::Line) one the height falls the same way from a quarter
    /// while the width stops at a quarter of its own.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>, OcrError> {
        let mut state = self.stem.forward(x)?;
        let mut out = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            if let Some(down) = &stage.downsample {
                state = down.forward(&state)?;
            }
            for block in &stage.blocks {
                state = block.forward(&state)?;
            }
            out.push(state.clone());
        }
        Ok(out)
    }
}
