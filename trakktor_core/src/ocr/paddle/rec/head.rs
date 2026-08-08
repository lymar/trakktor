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
//! - **Two epsilons.** The normalizations inside the blocks carry `1e-5`; the
//!   one that closes the encoder carries `1e-6`. Unifying them changes the
//!   output without changing a shape.
//! - **The activation is the smooth one.** The neck uses `x * sigmoid(x)` where
//!   the backbones around it use the piecewise-linear hard swish.

use candle_core::{D, Tensor};

use crate::ocr::{
    error::OcrError,
    paddle::net::{BatchNorm, Conv, LayerNorm, Linear, Loader, swish},
};

/// The width the transformer works in, which is also what the projection reads.
const NECK: usize = 120;
/// Attention heads and the width of each.
const HEADS: usize = 8;
const HEAD_WIDTH: usize = NECK / HEADS;
/// The hidden width of a block's feed-forward part.
const HIDDEN: usize = 240;
/// Transformer blocks.
const DEPTH: usize = 2;
/// The epsilon the four normalizations inside the blocks carry, and the
/// different one the closing normalization carries. One constant, two values,
/// and a silently wrong output if they are unified.
const BLOCK_EPS: f64 = 1e-5;
const FINAL_EPS: f64 = 1e-6;
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

/// Global self-attention over the sequence: no mask, no positional encoding.
#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
}

impl Attention {
    fn load(loader: &Loader, qkv: &str, proj: &str) -> Result<Self, OcrError> {
        Ok(Self {
            qkv: Linear::load(loader, qkv, NECK, 3 * NECK)?,
            proj: Linear::load(loader, proj, NECK, NECK)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let (batch, steps, _) = x.dims3()?;
        // One projection produces all three sequences at once, so splitting it
        // means reading the heads out of the middle of the width.
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((batch, steps, 3, HEADS, HEAD_WIDTH))?
            .permute((2, 0, 3, 1, 4))?;
        // The query is scaled before the product, not the product after it.
        let scale = (HEAD_WIDTH as f64).powf(-0.5);
        let q = qkv.get(0)?.contiguous()?.affine(scale, 0.0)?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let across = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let weights = candle_nn::ops::softmax_last_dim(&q.matmul(&across)?)?;
        let y = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, steps, NECK))?;
        self.proj.forward(&y)
    }
}

/// One transformer block: attention and a feed-forward part, each normalized
/// before it runs and added back to what went into it.
#[derive(Debug)]
struct EncoderBlock {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderBlock {
    /// The block at `index` owns normalizations `2i` and `2i + 1` and the four
    /// projections that follow the ones the blocks before it took.
    fn load(
        loader: &Loader,
        index: usize,
        naming: Naming,
    ) -> Result<Self, OcrError> {
        let norm = |offset: usize| {
            LayerNorm::load(
                loader,
                &format!("layer_norm_{}", 2 * index + offset),
                NECK,
                BLOCK_EPS,
            )
        };
        let name = |offset: usize| {
            format!("linear_{}", naming.linear + 4 * index + offset)
        };
        Ok(Self {
            norm1: norm(0)?,
            attn: Attention::load(loader, &name(0), &name(1))?,
            norm2: norm(1)?,
            fc1: Linear::load(loader, &name(2), NECK, HIDDEN)?,
            fc2: Linear::load(loader, &name(3), HIDDEN, NECK)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.attn.forward(&self.norm1.forward(x)?)?;
        let x = (x + y)?;
        let y = self.fc1.forward(&self.norm2.forward(&x)?)?;
        let y = self.fc2.forward(&swish(&y)?)?;
        Ok((&x + &y)?)
    }
}

/// The sequence encoder and the projection that reads it.
#[derive(Debug)]
pub struct Head {
    /// The neck narrows the backbone's width to the one the transformer works
    /// in,
    reduce: [ConvNorm; 2],
    encoder: Vec<EncoderBlock>,
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
        let mut encoder = Vec::with_capacity(DEPTH);
        for index in 0..DEPTH {
            encoder.push(EncoderBlock::load(loader, index, naming)?);
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
            norm: LayerNorm::load(loader, "layer_norm_4", NECK, FINAL_EPS)?,
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
