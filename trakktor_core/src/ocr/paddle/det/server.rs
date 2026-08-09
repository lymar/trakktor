//! `PP-OCRv5_server_det`: a PP-HGNetV2 backbone under a large-kernel pyramid
//! and a two-branch differentiable-binarization head.
//!
//! Same job as the small detector, nineteen times the weights, and a different
//! network at every level: the [backbone](crate::ocr::paddle::hgnet) is shared
//! with the layout model, the neck is a feature pyramid with a path-aggregation
//! pass back down it, and the head produces the probability map twice and
//! averages the two.
//!
//! Three things about it are easy to get wrong and silent when wrong:
//!
//! - **The neck's convolutions are 9×9.** Both the pyramid's outputs and the
//!   lateral pass use a kernel of nine, which is where "large-kernel" comes
//!   from and where most of the time goes; the port keeps them as they are
//!   rather than separating them, because the published weights are dense.
//! - **The intra-class blocks are not a refinement to be skipped.** Each rung
//!   of the pyramid goes through one, and each is a three-step cascade of
//!   square, vertical and horizontal kernels added together. Dropping them
//!   still produces a plausible map.
//! - **The head averages two maps, not one.** The first is the usual
//!   transposed-convolution stack; the second re-reads its own half-resolution
//!   features alongside the first map and produces a correction. Both go
//!   through a sigmoid, and the output is their mean — so a port that stops at
//!   the first map is systematically off rather than obviously broken.

use candle_core::Tensor;

use crate::ocr::{
    error::OcrError,
    paddle::{
        hgnet,
        net::{Banded, BatchNorm, Conv, ConvTranspose, Loader, relu, upsample},
    },
};

/// The width the pyramid works in.
const NECK: usize = 256;
/// Width of each of the four pyramid branches (`NECK / 4`), which is also what
/// the head reads.
const BRANCH: usize = NECK / 4;
/// The kernel the neck's own convolutions use.
const LARGE: usize = 9;
/// The intra-class block squeezes its input by this much before the cascade.
const INTRACL_REDUCE: usize = 2;

/// The published numbering of the neck's convolutions.
///
/// They are numbered in *construction* order, not in the order the forward pass
/// visits them, and construction runs level by level: at each level the lateral
/// 1×1, then the 9×9 that leaves the pyramid, then — from the second level on —
/// the 3×3 that walks back down it, then the second 9×9. Four levels, fifteen
/// convolutions, and the forward pass jumps between them.
const NECK_FIRST_CONV: usize = 81;
/// Where the four intra-class blocks start, and how many convolutions each one
/// claims.
const INTRACL_FIRST_CONV: usize = 96;
const INTRACL_CONVS: usize = 11;
/// The normalization inside the intra-class blocks continues the backbone's
/// numbering rather than starting over.
const INTRACL_FIRST_NORM: usize = 80;

/// One rung of the pyramid, as the published numbering lays it out.
struct RungConvs {
    /// The 1×1 that brings a backbone stage to the neck's width.
    lateral: usize,
    /// The 9×9 that leaves the pyramid.
    out: usize,
    /// The 3×3, stride 2, that carries the finer rung down onto this one.
    down: Option<usize>,
    /// The second 9×9, on the way out.
    lateral_out: usize,
}

/// The four rungs, finest first.
const RUNGS: [RungConvs; 4] = [
    RungConvs {
        lateral: NECK_FIRST_CONV,
        out: NECK_FIRST_CONV + 1,
        down: None,
        lateral_out: NECK_FIRST_CONV + 2,
    },
    RungConvs {
        lateral: NECK_FIRST_CONV + 3,
        out: NECK_FIRST_CONV + 4,
        down: Some(NECK_FIRST_CONV + 5),
        lateral_out: NECK_FIRST_CONV + 6,
    },
    RungConvs {
        lateral: NECK_FIRST_CONV + 7,
        out: NECK_FIRST_CONV + 8,
        down: Some(NECK_FIRST_CONV + 9),
        lateral_out: NECK_FIRST_CONV + 10,
    },
    RungConvs {
        lateral: NECK_FIRST_CONV + 11,
        out: NECK_FIRST_CONV + 12,
        down: Some(NECK_FIRST_CONV + 13),
        lateral_out: NECK_FIRST_CONV + 14,
    },
];

/// A banded convolution, loaded by its published number.
fn banded(
    loader: &Loader,
    index: usize,
    dims: [usize; 4],
) -> Result<Banded, OcrError> {
    Banded::load(loader, &format!("conv2d_{index}"), dims, 1)
}

/// A plain convolution with no bias, loaded by its published number.
fn conv(
    loader: &Loader,
    index: usize,
    dims: [usize; 4],
    stride: usize,
    padding: usize,
) -> Result<Conv, OcrError> {
    Conv::load(loader, &format!("conv2d_{index}"), dims, stride, padding, 1)
}

/// One rung's four convolutions.
#[derive(Debug)]
struct Rung {
    lateral: Conv,
    out: Banded,
    down: Option<Conv>,
    lateral_out: Banded,
}

impl Rung {
    fn load(
        loader: &Loader,
        spec: &RungConvs,
        from: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            lateral: conv(loader, spec.lateral, [NECK, from, 1, 1], 1, 0)?,
            out: banded(loader, spec.out, [BRANCH, NECK, LARGE, LARGE])?,
            down: spec
                .down
                .map(|index| conv(loader, index, [BRANCH, BRANCH, 3, 3], 2, 1))
                .transpose()?,
            lateral_out: banded(
                loader,
                spec.lateral_out,
                [BRANCH, BRANCH, LARGE, LARGE],
            )?,
        })
    }
}

/// The three kernel sizes an intra-class cascade steps through.
const CASCADE: [usize; 3] = [7, 5, 3];

/// One step of the cascade: the same tensor read by a square kernel, a vertical
/// one and a horizontal one, and the three outputs added.
#[derive(Debug)]
struct Step {
    square: Banded,
    vertical: Banded,
    horizontal: Banded,
}

impl Step {
    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        Ok(((self.square.forward(x)? + self.vertical.forward(x)?)? +
            self.horizontal.forward(x)?)?)
    }
}

/// A cascade of three steps, wrapped in a squeeze and an expansion and added
/// back to the input.
///
/// Every convolution here carries a bias, which nothing else in this network
/// does. The block is PaddleOCR's, adapted there from I3CL — its own source
/// says so, and the shape of it (three kernel orientations summed, narrowing
/// from seven to three) is that paper's idea rather than PaddleOCR's.
#[derive(Debug)]
struct IntraCl {
    reduce: Conv,
    steps: Vec<Step>,
    expand: Conv,
    norm: BatchNorm,
}

impl IntraCl {
    /// `first` is the number of this block's first convolution; the ten that
    /// follow it are in construction order: the squeeze, the expansion, the
    /// three vertical kernels, the three horizontal ones, then the three square
    /// ones — largest kernel first in each group.
    fn load(
        loader: &Loader,
        first: usize,
        norm: usize,
        channels: usize,
    ) -> Result<Self, OcrError> {
        let inner = channels / INTRACL_REDUCE;
        let mut steps = Vec::with_capacity(CASCADE.len());
        for (at, kernel) in CASCADE.into_iter().enumerate() {
            // A `k × 1` kernel pads along the height only and a `1 × k` one
            // along the width only, which a single padding argument cannot
            // say; padding by hand is what [`Banded`] does anyway.
            steps.push(Step {
                square: banded(
                    loader,
                    first + 8 + at,
                    [inner, inner, kernel, kernel],
                )?,
                vertical: banded(
                    loader,
                    first + 2 + at,
                    [inner, inner, kernel, 1],
                )?,
                horizontal: banded(
                    loader,
                    first + 5 + at,
                    [inner, inner, 1, kernel],
                )?,
            });
        }
        Ok(Self {
            reduce: conv(loader, first, [inner, channels, 1, 1], 1, 0)?,
            steps,
            expand: conv(loader, first + 1, [channels, inner, 1, 1], 1, 0)?,
            norm: BatchNorm::load(
                loader,
                &format!("batch_norm2d_{norm}"),
                channels,
            )?,
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

/// The head: the probability map, and a correction to it.
#[derive(Debug)]
struct Head {
    conv: Banded,
    norm: BatchNorm,
    up1: ConvTranspose,
    norm2: BatchNorm,
    up2: ConvTranspose,
    local: Banded,
    local_norm: BatchNorm,
    local_out: Conv,
}

/// The published numbering of the head. The gap at 141 belongs to the second
/// head the training loss needs, which the exported graph does not carry; the
/// normalization numbering has the same gap, which is why the local module's is
/// `batch_norm_4` rather than `_2`.
const HEAD_CONV: usize = 140;
const LOCAL_CONV: usize = 142;
const LOCAL_OUT_CONV: usize = 143;
const LOCAL_NORM: usize = 4;

impl Head {
    fn load(loader: &Loader) -> Result<Self, OcrError> {
        Ok(Self {
            conv: banded(loader, HEAD_CONV, [BRANCH, NECK, 3, 3])?,
            norm: BatchNorm::load(loader, "batch_norm_0", BRANCH)?,
            up1: ConvTranspose::load(
                loader,
                "conv2d_transpose_0",
                [BRANCH, BRANCH, 2, 2],
                2,
            )?,
            norm2: BatchNorm::load(loader, "batch_norm_1", BRANCH)?,
            up2: ConvTranspose::load(
                loader,
                "conv2d_transpose_1",
                [BRANCH, 1, 2, 2],
                2,
            )?,
            local: banded(loader, LOCAL_CONV, [BRANCH, BRANCH + 1, 3, 3])?,
            local_norm: BatchNorm::load(
                loader,
                &format!("batch_norm_{LOCAL_NORM}"),
                BRANCH,
            )?,
            local_out: conv(loader, LOCAL_OUT_CONV, [1, BRANCH, 1, 1], 1, 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.conv.forward(x)?;
        let y = relu(&self.norm.forward(&y)?)?;
        let y = self.up1.forward(&y)?;
        // Half resolution, and read twice: once by the transposed convolution
        // that finishes the map, once by the local module below.
        let half = relu(&self.norm2.forward(&y)?)?;
        let base = candle_nn::ops::sigmoid(&self.up2.forward(&half)?)?;

        let joined = Tensor::cat(&[&base, &upsample(&half, 2)?], 1)?;
        let local = self.local.forward(&joined)?;
        let local = relu(&self.local_norm.forward(&local)?)?;
        let local = candle_nn::ops::sigmoid(&self.local_out.forward(&local)?)?;

        Ok((base + local)?.affine(0.5, 0.0)?)
    }
}

/// The network.
#[derive(Debug)]
pub struct Net {
    backbone: hgnet::Backbone,
    rungs: Vec<Rung>,
    intracl: Vec<IntraCl>,
    head: Head,
}

impl Net {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let mut next = 0usize;
        // The backbone's own numbering ends where the neck's begins; the gap is
        // the classification convolution a detector never runs.
        let backbone =
            hgnet::Backbone::load(loader, &mut next, 0, hgnet::Shape::Page)?;

        let mut rungs = Vec::with_capacity(RUNGS.len());
        for (spec, from) in RUNGS.iter().zip(hgnet::STAGE_CHANNELS) {
            rungs.push(Rung::load(loader, spec, from)?);
        }

        let mut intracl = Vec::with_capacity(RUNGS.len());
        for at in 0..RUNGS.len() {
            intracl.push(IntraCl::load(
                loader,
                INTRACL_FIRST_CONV + at * INTRACL_CONVS,
                INTRACL_FIRST_NORM + at,
                BRANCH,
            )?);
        }

        Ok(Self {
            backbone,
            rungs,
            intracl,
            head: Head::load(loader)?,
        })
    }

    /// Runs the network over one normalized page, returning the probability
    /// map as `[1, 1, height, width]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let stages = self.backbone.forward(x)?;

        // Top-down: each rung is its own lateral projection plus the coarser
        // rung, doubled.
        let mut lateral = Vec::with_capacity(stages.len());
        for (rung, stage) in self.rungs.iter().zip(&stages) {
            lateral.push(rung.lateral.forward(stage)?);
        }
        let mut merged = vec![lateral[3].clone()];
        for level in (0..3).rev() {
            let coarser = upsample(merged.last().expect("a coarser level"), 2)?;
            merged.push((&lateral[level] + &coarser)?);
        }
        merged.reverse(); // finest first

        let mut out = Vec::with_capacity(self.rungs.len());
        for (rung, merged) in self.rungs.iter().zip(&merged) {
            out.push(rung.out.forward(merged)?);
        }

        // Bottom-up: the finest rung walks back down, each step strided by two
        // and added to the rung it lands on. The finest one keeps its own
        // output, which is why the loop starts at the second.
        let mut aggregated = vec![out[0].clone()];
        for level in 1..out.len() {
            let down = self.rungs[level]
                .down
                .as_ref()
                .expect("every rung but the finest walks down")
                .forward(&aggregated[level - 1])?;
            aggregated.push((&out[level] + &down)?);
        }

        // Each rung leaves through its second 9×9 and an intra-class block,
        // then is brought back to the finest scale; the concatenation is
        // coarsest first.
        let mut branches = Vec::with_capacity(self.rungs.len());
        for (level, aggregated) in aggregated.iter().enumerate() {
            let branch = self.rungs[level].lateral_out.forward(aggregated)?;
            let branch = self.intracl[level].forward(&branch)?;
            branches.push(if level == 0 {
                branch
            } else {
                upsample(&branch, 1 << level)?
            });
        }
        branches.reverse();
        let fused = Tensor::cat(&branches, 1)?;

        self.head.forward(&fused)
    }
}
