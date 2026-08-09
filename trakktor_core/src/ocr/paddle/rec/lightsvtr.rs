//! The newest recognizer's head: a lighter sequence encoder, and the CTC
//! projection above it.
//!
//! Where the [classic head](super::head) narrows the backbone's output to an
//! eighth, runs the transformer, widens the result back and lays it beside the
//! original before narrowing again, this one does the cheap version of the same
//! idea: two 1×1 projections of the backbone's output — one that feeds the
//! transformer and one that is simply added back at the end — and, between
//! them, a depthwise convolution seven columns wide that gives each step a
//! little of its neighbours before global attention sees it.
//!
//! The blocks themselves are [the same](super::svtr), 192 channels wide here
//! against 120 there.
//!
//! One ordering detail matters and is easy to lose: **the skip projection runs
//! first**. It is built last in upstream's source and read first by the graph,
//! and since the weights here are taken in graph order rather than by name,
//! reordering these two loads silently swaps two convolutions of identical
//! shape.

use candle_core::Tensor;

use super::svtr::{self, Width};
use crate::ocr::{
    error::OcrError,
    paddle::net::{BatchNorm, Conv, LayerNorm, Linear, Loader, Order, swish},
};

/// The width the transformer works in, which is also what the projection reads.
const NECK: usize = 192;
/// The hidden width of a block's feed-forward part (`mlp_ratio` is four).
const HIDDEN: usize = NECK * 4;
const WIDTH: Width = Width {
    dim: NECK,
    hidden: HIDDEN,
};
/// Transformer blocks.
const DEPTH: usize = 2;
/// How many columns the local convolution reads at once.
const LOCAL: usize = 7;

/// A 1×1 projection with its normalization and a sigmoid-weighted activation.
#[derive(Debug)]
struct Project {
    conv: Conv,
    norm: BatchNorm,
}

impl Project {
    fn load(
        loader: &Loader,
        order: &mut Order,
        from: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(
                loader,
                order.take()?,
                [NECK, from, 1, 1],
                1,
                0,
                1,
            )?,
            norm: BatchNorm::load(loader, order.take()?, NECK)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        swish(&self.norm.forward(&self.conv.forward(x)?)?)
    }
}

/// The depthwise convolution that mixes a step with its neighbours along the
/// sequence and nothing across the channels.
///
/// It is one row tall and seven columns wide, so it pads along the width
/// alone; a convolution takes one padding for both axes, so that padding is
/// added to the input instead.
#[derive(Debug)]
struct Local {
    conv: Conv,
    norm: BatchNorm,
}

impl Local {
    fn load(loader: &Loader, order: &mut Order) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(
                loader,
                order.take()?,
                [NECK, 1, 1, LOCAL],
                1,
                0,
                NECK,
            )?,
            norm: BatchNorm::load(loader, order.take()?, NECK)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let padded = x.pad_with_zeros(3, LOCAL / 2, LOCAL / 2)?;
        swish(&self.norm.forward(&self.conv.forward(&padded)?)?)
    }
}

/// The head.
#[derive(Debug)]
pub struct Head {
    /// Read first by the graph, added last by the forward pass.
    skip: Project,
    reduce: Project,
    local: Local,
    encoder: Vec<svtr::Block>,
    norm: LayerNorm,
    project: Linear,
    classes: usize,
}

impl Head {
    /// Loads the head above a backbone `from` channels wide, taking every
    /// weight in the order the graph reads it.
    pub fn load(
        loader: &Loader,
        order: &mut Order,
        from: usize,
        classes: usize,
    ) -> Result<Self, OcrError> {
        let skip = Project::load(loader, order, from)?;
        let reduce = Project::load(loader, order, from)?;
        let local = Local::load(loader, order)?;
        let mut encoder = Vec::with_capacity(DEPTH);
        for _ in 0..DEPTH {
            // A block is read in forward order: normalize, attend, normalize,
            // feed forward. Its second normalization therefore sits between
            // its two pairs of projections rather than beside the first.
            let norm1 = order.take()?;
            let qkv = order.take()?;
            let proj = order.take()?;
            let norm2 = order.take()?;
            let fc1 = order.take()?;
            let fc2 = order.take()?;
            encoder.push(svtr::Block::load(
                loader,
                (norm1, norm2),
                (qkv, proj, fc1, fc2),
                WIDTH,
            )?);
        }
        let norm =
            LayerNorm::load(loader, order.take()?, NECK, svtr::FINAL_EPS)?;
        let project = Linear::load(loader, order.take()?, NECK, classes)?;
        Ok(Self {
            skip,
            reduce,
            local,
            encoder,
            norm,
            project,
            classes,
        })
    }

    /// How many classes the projection reads, the blank and the space
    /// included.
    pub fn classes(&self) -> usize { self.classes }

    /// Reads a backbone's pooled output — `[batch, from, 1, steps]` — and
    /// returns the **logits** as `[batch, steps, classes]`.
    pub fn forward(&self, pooled: &Tensor) -> Result<Tensor, OcrError> {
        let skip = self.skip.forward(pooled)?;
        let z = self.reduce.forward(pooled)?;
        let z = (&z + self.local.forward(&z)?)?;

        let (batch, _, _, steps) = z.dims4()?;
        let mut sequence = z
            .reshape((batch, NECK, steps))?
            .transpose(1, 2)?
            .contiguous()?;
        for block in &self.encoder {
            sequence = block.forward(&sequence)?;
        }
        sequence = self.norm.forward(&sequence)?;

        let z = sequence
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, NECK, 1, steps))?;
        let z = (z + skip)?;

        let sequence = z
            .reshape((batch, NECK, steps))?
            .transpose(1, 2)?
            .contiguous()?;
        self.project.forward(&sequence)
    }
}
