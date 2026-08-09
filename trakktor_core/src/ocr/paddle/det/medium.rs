//! `PP-OCRv6_medium_det`: a PP-LCNetV4 backbone under a large-kernel pyramid
//! whose wide convolutions are split, and a plain differentiable-binarization
//! head.
//!
//! It is the same pyramid the [large v5 detector](super::server) carries —
//! four rungs down, four back up, an intra-class block on each — with two
//! changes that are the whole point of the generation:
//!
//! - **The 9×9 convolutions are split into a depthwise 9×9 and a pointwise
//!   1×1.** That is where the size goes: this detector is a fifth of the large
//!   v5 one and still reads the page at the same width.
//! - **Everything is reparameterized.** Not one batch normalization survives in
//!   the backbone or the pyramid; the published convolutions carry the folded
//!   bias instead. Only the intra-class blocks keep a normalization of their
//!   own, because theirs was never foldable — it sits after the block's
//!   residual, not before it.
//!
//! The head is the plain one, not the two-map head the large v5 detector has:
//! one stack of transposed convolutions and one sigmoid.
//!
//! **The weights are taken in the order the graph reads them**, not by their
//! numbers. Reparameterization renumbered every fused layer, so the pyramid's
//! first convolution is `conv2d_105` and the block beside it is `conv2d_56`;
//! see [`Order`](crate::ocr::paddle::net::Order). That makes the load order
//! part of the network's meaning: this file constructs its layers in exactly
//! the order the forward pass runs them, and a rearrangement that looks
//! harmless will load the wrong weights into the right shapes wherever two
//! layers happen to match.

use candle_core::Tensor;

use crate::ocr::{
    error::OcrError,
    paddle::{
        lcnetv4::{self, BlockSpec, block},
        net::{
            Banded, BatchNorm, Conv, ConvTranspose, Loader, Order, relu,
            upsample,
        },
    },
};

/// The side length the input must be a multiple of: the backbone's four stages
/// quarter and then halve three times.
pub const SIZE_MULTIPLE: usize = 32;

/// The width the pyramid works in, and the width of each of its four branches.
const NECK: usize = 256;
const BRANCH: usize = NECK / 4;
/// The kernel the pyramid's depthwise convolutions use.
const LARGE: usize = 9;
/// The intra-class block squeezes its input by this much before the cascade.
const INTRACL_REDUCE: usize = 2;
/// The three kernel sizes an intra-class cascade steps through.
const CASCADE: [usize; 3] = [7, 5, 3];

/// The stem's channels: half-width in the middle, full width out.
const STEM: (usize, usize) = (64, 128);

/// The four stages of the backbone, as the published configuration spells
/// them: kernel three throughout, a doubling at the head of each stage after
/// the first, and a gate on every second block.
const STAGE1: [BlockSpec; 2] = [
    block(128, 128, (1, 1), true),
    block(128, 128, (1, 1), false),
];
const STAGE2: [BlockSpec; 3] = [
    block(128, 256, (2, 2), false),
    block(256, 256, (1, 1), true),
    block(256, 256, (1, 1), false),
];
const STAGE3: [BlockSpec; 5] = [
    block(256, 512, (2, 2), false),
    block(512, 512, (1, 1), true),
    block(512, 512, (1, 1), false),
    block(512, 512, (1, 1), true),
    block(512, 512, (1, 1), false),
];
const STAGE4: [BlockSpec; 3] = [
    block(512, 896, (2, 2), false),
    block(896, 896, (1, 1), true),
    block(896, 896, (1, 1), false),
];

/// What each stage ends on, finest first.
const STAGE_CHANNELS: [usize; 4] = [128, 256, 512, 896];

/// One wide convolution, split in two: a depthwise kernel that carries the
/// reach and a pointwise one that carries the channels.
///
/// The depthwise half is banded for the same reason the v5 pyramid's whole 9×9
/// is — see [`Banded`] — even though a depthwise kernel asks for far less
/// scratch than a dense one. It costs nothing to keep the guard.
#[derive(Debug)]
struct Wide {
    depth: Banded,
    point: Conv,
}

impl Wide {
    fn load(
        loader: &Loader,
        order: &mut Order,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            depth: Banded::load(
                loader,
                order.take()?,
                [in_channels, 1, LARGE, LARGE],
                in_channels,
            )?,
            point: Conv::load(
                loader,
                order.take()?,
                [out_channels, in_channels, 1, 1],
                1,
                0,
                1,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        self.point.forward(&self.depth.forward(x)?)
    }
}

/// One step of an intra-class cascade: the same tensor read by a square
/// kernel, a vertical one and a horizontal one, and the three outputs added.
#[derive(Debug)]
struct Step {
    square: Banded,
    vertical: Banded,
    horizontal: Banded,
}

impl Step {
    fn load(
        loader: &Loader,
        order: &mut Order,
        inner: usize,
        kernel: usize,
    ) -> Result<Self, OcrError> {
        let mut band =
            |dims: [usize; 4]| Banded::load(loader, order.take()?, dims, 1);
        Ok(Self {
            square: band([inner, inner, kernel, kernel])?,
            vertical: band([inner, inner, kernel, 1])?,
            horizontal: band([inner, inner, 1, kernel])?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        Ok(((self.square.forward(x)? + self.vertical.forward(x)?)? +
            self.horizontal.forward(x)?)?)
    }
}

/// A cascade of three steps, wrapped in a squeeze and an expansion and added
/// back to the input.
///
/// The same block the large v5 detector carries, down to the kernel sizes.
/// PaddleOCR adapted it from I3CL — its own source says so — and the shape of
/// it, three kernel orientations summed and narrowing from seven to three, is
/// that paper's idea rather than PaddleOCR's.
#[derive(Debug)]
struct IntraCl {
    reduce: Conv,
    steps: Vec<Step>,
    expand: Conv,
    norm: BatchNorm,
}

impl IntraCl {
    fn load(
        loader: &Loader,
        order: &mut Order,
        channels: usize,
    ) -> Result<Self, OcrError> {
        let inner = channels / INTRACL_REDUCE;
        let reduce = Conv::load(
            loader,
            order.take()?,
            [inner, channels, 1, 1],
            1,
            0,
            1,
        )?;
        let mut steps = Vec::with_capacity(CASCADE.len());
        for kernel in CASCADE {
            steps.push(Step::load(loader, order, inner, kernel)?);
        }
        let expand = Conv::load(
            loader,
            order.take()?,
            [channels, inner, 1, 1],
            1,
            0,
            1,
        )?;
        let norm = BatchNorm::load(loader, order.take()?, channels)?;
        Ok(Self {
            reduce,
            steps,
            expand,
            norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut y = self.reduce.forward(x)?;
        for step in &self.steps {
            y = step.forward(&y)?;
        }
        let y = relu(&self.norm.forward(&self.expand.forward(&y)?)?)?;
        Ok((x + y)?)
    }
}

/// The head: one stack, one map.
#[derive(Debug)]
struct Head {
    conv: Banded,
    up1: ConvTranspose,
    up2: ConvTranspose,
}

impl Head {
    fn load(loader: &Loader, order: &mut Order) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Banded::load(loader, order.take()?, [BRANCH, NECK, 3, 3], 1)?,
            up1: ConvTranspose::load(
                loader,
                order.take()?,
                [BRANCH, BRANCH, 2, 2],
                2,
            )?,
            up2: ConvTranspose::load(
                loader,
                order.take()?,
                [BRANCH, 1, 2, 2],
                2,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = relu(&self.conv.forward(x)?)?;
        let y = relu(&self.up1.forward(&y)?)?;
        Ok(candle_nn::ops::sigmoid(&self.up2.forward(&y)?)?)
    }
}

/// The network.
#[derive(Debug)]
pub struct Net {
    backbone: lcnetv4::Backbone,
    /// The 1×1 projections onto the pyramid's width, **coarsest first** — the
    /// order the graph reads them in, which is the order the top-down pass
    /// needs them.
    lateral: Vec<Conv>,
    /// The wide convolutions that leave the pyramid, coarsest first.
    out: Vec<Wide>,
    /// The 3×3 strided convolutions that walk back down it, finest first.
    down: Vec<Conv>,
    /// The wide convolutions on the way out, finest first.
    lateral_out: Vec<Wide>,
    /// The intra-class blocks, **coarsest first**.
    intracl: Vec<IntraCl>,
    head: Head,
}

impl Net {
    /// Loads the network, taking every weight in the order the graph reads it.
    ///
    /// The sequence below is the forward pass written out: the backbone, then
    /// the pyramid's four projections from the coarsest rung down, then the
    /// four wide convolutions in the same direction, the three that walk back
    /// up, the four on the way out, the four intra-class blocks from the
    /// coarsest, and the head.
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let mut order = Order::new(loader);
        let stages: [&[BlockSpec]; 4] = [&STAGE1, &STAGE2, &STAGE3, &STAGE4];
        let backbone =
            lcnetv4::Backbone::load(loader, &mut order, STEM, 2, &stages)?;

        let mut lateral = Vec::with_capacity(STAGE_CHANNELS.len());
        for from in STAGE_CHANNELS.into_iter().rev() {
            lateral.push(Conv::load(
                loader,
                order.take()?,
                [NECK, from, 1, 1],
                1,
                0,
                1,
            )?);
        }

        let mut out = Vec::with_capacity(STAGE_CHANNELS.len());
        for _ in 0..STAGE_CHANNELS.len() {
            out.push(Wide::load(loader, &mut order, NECK, BRANCH)?);
        }

        let mut down = Vec::with_capacity(STAGE_CHANNELS.len() - 1);
        for _ in 1..STAGE_CHANNELS.len() {
            down.push(Conv::load(
                loader,
                order.take()?,
                [BRANCH, BRANCH, 3, 3],
                2,
                1,
                1,
            )?);
        }

        let mut lateral_out = Vec::with_capacity(STAGE_CHANNELS.len());
        for _ in 0..STAGE_CHANNELS.len() {
            lateral_out.push(Wide::load(loader, &mut order, BRANCH, BRANCH)?);
        }

        let mut intracl = Vec::with_capacity(STAGE_CHANNELS.len());
        for _ in 0..STAGE_CHANNELS.len() {
            intracl.push(IntraCl::load(loader, &mut order, BRANCH)?);
        }

        let head = Head::load(loader, &mut order)?;
        order.finish()?;

        Ok(Self {
            backbone,
            lateral,
            out,
            down,
            lateral_out,
            intracl,
            head,
        })
    }

    /// Runs the network over one normalized page, returning the probability
    /// map as `[1, 1, height, width]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let stages = self.backbone.forward(x)?;

        // Top-down: the coarsest rung is its own projection, and each finer
        // one is its projection plus the rung above it, doubled.
        let mut merged: Vec<Tensor> = Vec::with_capacity(stages.len());
        for (at, lateral) in self.lateral.iter().enumerate() {
            let projected = lateral.forward(&stages[stages.len() - 1 - at])?;
            merged.push(match merged.last() {
                None => projected,
                Some(coarser) => (projected + upsample(coarser, 2)?)?,
            });
        }

        // The wide convolutions read the merged rungs in the same order.
        let out: Vec<Tensor> = self
            .out
            .iter()
            .zip(&merged)
            .map(|(wide, merged)| wide.forward(merged))
            .collect::<Result<_, _>>()?;

        // Bottom-up: the finest rung walks back down, each step strided by two
        // and added to the rung it lands on.
        let mut aggregated = vec![out[out.len() - 1].clone()];
        for (at, down) in self.down.iter().enumerate() {
            let carried = down.forward(&aggregated[at])?;
            aggregated.push((&out[out.len() - 2 - at] + &carried)?);
        }

        // Each rung leaves through a second wide convolution and an intra-class
        // block, then is brought back to the finest scale. The blocks are held
        // coarsest first, which is the order the graph reads them; the rungs
        // are finest first.
        let mut branches = Vec::with_capacity(aggregated.len());
        for (level, aggregated) in aggregated.iter().enumerate() {
            let branch = self.lateral_out[level].forward(aggregated)?;
            let branch = self.intracl[self.intracl.len() - 1 - level]
                .forward(&branch)?;
            branches.push(if level == 0 {
                branch
            } else {
                upsample(&branch, 1 << level)?
            });
        }
        // Coarsest first, the way the head expects them.
        branches.reverse();
        let fused = Tensor::cat(&branches, 1)?;

        self.head.forward(&fused)
    }
}
