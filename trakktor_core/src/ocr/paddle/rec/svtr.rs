//! The transformer block both recognizer heads are built from.
//!
//! Upstream calls it an SVTR block and configures it two ways: 120 channels
//! wide under the classic heads, 192 under the light one that came with the
//! newest generation. Nothing else about it differs — global attention with no
//! mask and no positional encoding, a feed-forward part, each normalized
//! before it runs and added back to what went into it.
//!
//! The two constants worth naming are the ones a reader will assume and be
//! wrong about: the query is scaled **before** the product rather than the
//! product after it, and the normalization epsilon inside a block (`1e-5`) is
//! not the one that closes an encoder (`1e-6`).

use candle_core::{D, Tensor};

use crate::ocr::{
    error::OcrError,
    paddle::net::{LayerNorm, Linear, Loader, swish},
};

/// The epsilon the normalizations inside a block carry.
pub const BLOCK_EPS: f64 = 1e-5;
/// The different one that closes an encoder.
pub const FINAL_EPS: f64 = 1e-6;
/// Attention heads. Both configurations use eight.
const HEADS: usize = 8;

/// How wide a block is.
#[derive(Debug, Clone, Copy)]
pub struct Width {
    /// Channels in and out.
    pub dim: usize,
    /// The hidden width of the feed-forward part.
    pub hidden: usize,
}

impl Width {
    fn head(self) -> usize { self.dim / HEADS }
}

/// Global self-attention over the sequence: no mask, no positional encoding.
#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    width: Width,
}

impl Attention {
    fn load(
        loader: &Loader,
        qkv: &str,
        proj: &str,
        width: Width,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            qkv: Linear::load(loader, qkv, width.dim, 3 * width.dim)?,
            proj: Linear::load(loader, proj, width.dim, width.dim)?,
            width,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let (batch, steps, _) = x.dims3()?;
        let head = self.width.head();
        // One projection produces all three sequences at once, so splitting it
        // means reading the heads out of the middle of the width.
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((batch, steps, 3, HEADS, head))?
            .permute((2, 0, 3, 1, 4))?;
        // The query is scaled before the product, not the product after it.
        let scale = (head as f64).powf(-0.5);
        let q = qkv.get(0)?.contiguous()?.affine(scale, 0.0)?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let across = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let weights = candle_nn::ops::softmax_last_dim(&q.matmul(&across)?)?;
        let y = weights.matmul(&v)?.transpose(1, 2)?.reshape((
            batch,
            steps,
            self.width.dim,
        ))?;
        self.proj.forward(&y)
    }
}

/// One block: attention and a feed-forward part, each normalized before it
/// runs and added back to what went into it.
#[derive(Debug)]
pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    /// Builds a block from the five names it owns, in the order the graph
    /// reads them: the first normalization, the fused query-key-value
    /// projection, the projection that closes attention, the second
    /// normalization, and the feed-forward pair.
    pub fn load(
        loader: &Loader,
        norms: (&str, &str),
        projections: (&str, &str, &str, &str),
        width: Width,
    ) -> Result<Self, OcrError> {
        let norm =
            |name: &str| LayerNorm::load(loader, name, width.dim, BLOCK_EPS);
        Ok(Self {
            norm1: norm(norms.0)?,
            attn: Attention::load(loader, projections.0, projections.1, width)?,
            norm2: norm(norms.1)?,
            fc1: Linear::load(loader, projections.2, width.dim, width.hidden)?,
            fc2: Linear::load(loader, projections.3, width.hidden, width.dim)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.attn.forward(&self.norm1.forward(x)?)?;
        let x = (x + y)?;
        let y = self.fc1.forward(&self.norm2.forward(&x)?)?;
        let y = self.fc2.forward(&swish(&y)?)?;
        Ok((&x + &y)?)
    }
}
