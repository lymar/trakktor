//! Everything a recognizer does downstream of its backbone: the small
//! transformer that reads the feature map as a sequence, and the projection
//! that turns each step into a distribution over the character table.
//!
//! Upstream calls the pair `MultiHead` — `SequenceEncoder/EncoderWithSVTR`
//! followed by `CTCHead` — and publishes it unchanged above both backbones the
//! catalog carries. It is written once here because it really is one network:
//! two transformer blocks 120 channels wide, whatever fed them.
//!
//! What the two artifacts do *not* share is where the weights are numbered
//! from, so that is the only thing a caller states — see [`Naming`].
//!
//! Three details are pinned here because each one fails quietly:
//!
//! - **The width the backbone hands over is halved and halved again by name,
//!   not by shape.** The reducing convolution squeezes it to an eighth before
//!   the transformer sees it, and the fusing one reads *twice* that width
//!   because the backbone's own output is laid beside the transformer's.
//! - **The closing normalization does not share the blocks' epsilon.** Theirs
//!   is `1e-5` and this one is `1e-6`; unifying them changes the output without
//!   changing a shape.
//! - **The activation is the smooth one.** The neck uses `x * sigmoid(x)` where
//!   the backbones around it use the piecewise-linear hard swish.

use candle_core::Tensor;

use super::svtr::{self, Width};
use crate::ocr::{
    error::OcrError,
    paddle::net::{BatchNorm, Conv, LayerNorm, Linear, Loader, swish},
};

/// The width the transformer works in, which is also what the projection reads.
const NECK: usize = 120;
/// The hidden width of a block's feed-forward part.
const HIDDEN: usize = 240;
const WIDTH: Width = Width {
    dim: NECK,
    hidden: HIDDEN,
};
/// Transformer blocks.
const DEPTH: usize = 2;
/// How much of the backbone's width reaches the transformer.
const SQUEEZE: usize = 8;

/// Where an artifact numbers this part of itself, and how wide the backbone
/// under it is.
///
/// The published graphs number every convolution and every projection in
/// declaration order across the whole network, so these bases are simply
/// "however many the backbone used up" — and they are not that, quite: both
/// artifacts declare a classification layer they never run, which takes a
/// number with it and leaves a gap. The numbers are therefore read off the
/// artifact rather than derived, and each backbone states its own.
#[derive(Debug, Clone, Copy)]
pub struct Naming {
    /// Channels the backbone hands over.
    pub width: usize,
    /// The index of the first of the five convolutions (`conv2d_{n}`).
    pub conv: usize,
    /// The index of the first of their five normalizations
    /// (`batch_norm2d_{n}`).
    pub norm: usize,
    /// The index of the first projection of the first transformer block
    /// (`linear_{n}`); the eight of them run consecutively and the character
    /// projection follows.
    pub linear: usize,
}

impl Naming {
    /// The squeezed width the reducing and fusing convolutions work in.
    fn squeezed(self) -> usize { self.width / SQUEEZE }
}

/// A neck convolution: no bias, batch normalization, and a sigmoid-weighted
/// activation — the neck's activation is not the backbone's hard one.
///
/// Two of the five have a kernel one row tall and three columns wide, padded
/// along the width alone. A convolution takes one padding for both axes, so
/// that padding is added to the input instead.
#[derive(Debug)]
struct ConvNorm {
    conv: Conv,
    pad: usize,
    norm: BatchNorm,
}

impl ConvNorm {
    fn load(
        loader: &Loader,
        at: usize,
        naming: Naming,
        dims: [usize; 4],
    ) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{}", naming.conv + at),
                dims,
                1,
                0,
                1,
            )?,
            pad: dims[3] / 2,
            norm: BatchNorm::load(
                loader,
                &format!("batch_norm2d_{}", naming.norm + at),
                dims[0],
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let padded = match self.pad {
            0 => x.clone(),
            pad => x.pad_with_zeros(3, pad, pad)?,
        };
        let y = self.norm.forward(&self.conv.forward(&padded)?)?;
        swish(&y)
    }
}

/// The sequence encoder and the projection that reads it.
#[derive(Debug)]
pub struct Head {
    /// The neck narrows the backbone's width to the one the transformer works
    /// in,
    reduce: [ConvNorm; 2],
    encoder: Vec<svtr::Block>,
    norm: LayerNorm,
    /// widens what the transformer made of it back to the backbone's width so
    /// that the two can be laid side by side,
    restore: ConvNorm,
    /// and narrows the pair once more for the projection to read.
    fuse: [ConvNorm; 2],
    project: Linear,
    classes: usize,
}

impl Head {
    /// Loads the head of a recognizer whose backbone is `naming.width` wide and
    /// whose weights are numbered from `naming`.
    pub fn load(
        loader: &Loader,
        naming: Naming,
        classes: usize,
    ) -> Result<Self, OcrError> {
        let squeezed = naming.squeezed();
        // The block at `index` owns normalizations `2i` and `2i + 1` and the
        // four projections that follow the ones the blocks before it took.
        let mut encoder = Vec::with_capacity(DEPTH);
        for index in 0..DEPTH {
            let norm =
                |offset: usize| format!("layer_norm_{}", 2 * index + offset);
            let name = |offset: usize| {
                format!("linear_{}", naming.linear + 4 * index + offset)
            };
            encoder.push(svtr::Block::load(
                loader,
                (&norm(0), &norm(1)),
                (&name(0), &name(1), &name(2), &name(3)),
                WIDTH,
            )?);
        }
        Ok(Self {
            reduce: [
                ConvNorm::load(
                    loader,
                    0,
                    naming,
                    [squeezed, naming.width, 1, 3],
                )?,
                ConvNorm::load(loader, 1, naming, [NECK, squeezed, 1, 1])?,
            ],
            encoder,
            norm: LayerNorm::load(
                loader,
                "layer_norm_4",
                NECK,
                svtr::FINAL_EPS,
            )?,
            restore: ConvNorm::load(
                loader,
                2,
                naming,
                [naming.width, NECK, 1, 1],
            )?,
            fuse: [
                ConvNorm::load(
                    loader,
                    3,
                    naming,
                    [squeezed, 2 * naming.width, 1, 3],
                )?,
                ConvNorm::load(loader, 4, naming, [NECK, squeezed, 1, 1])?,
            ],
            project: Linear::load(
                loader,
                &format!("linear_{}", naming.linear + 4 * DEPTH),
                NECK,
                classes,
            )?,
            classes,
        })
    }

    /// How many classes the projection reads, the blank and the space included.
    pub fn classes(&self) -> usize { self.classes }

    /// Reads a backbone's pooled output — `[batch, width, 1, steps]` — and
    /// returns the **logits** as `[batch, steps, classes]`.
    ///
    /// The softmax is inside the exported graph but belongs to the caller here,
    /// because it is the last thing either recognizer does and stating it once
    /// beside the network keeps the two from disagreeing about it.
    pub fn forward(&self, pooled: &Tensor) -> Result<Tensor, OcrError> {
        let mut z = pooled.clone();
        for layer in &self.reduce {
            z = layer.forward(&z)?;
        }
        let (batch, _, _, steps) = z.dims4()?;
        let mut sequence = z
            .reshape((batch, NECK, steps))?
            .transpose(1, 2)?
            .contiguous()?;
        for block in &self.encoder {
            sequence = block.forward(&sequence)?;
        }
        sequence = self.norm.forward(&sequence)?;
        z = sequence
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, NECK, 1, steps))?;
        z = self.restore.forward(&z)?;
        // What the backbone produced comes first, what the transformer made of
        // it second.
        z = Tensor::cat(&[pooled, &z], 1)?;
        for layer in &self.fuse {
            z = layer.forward(&z)?;
        }

        let sequence = z
            .reshape((batch, NECK, steps))?
            .transpose(1, 2)?
            .contiguous()?;
        self.project.forward(&sequence)
    }
}
