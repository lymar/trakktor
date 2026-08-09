//! The PP-LCNetV4 backbone, shared by the newest detector and recognizer.
//!
//! One network in two shapes, the way [PP-HGNetV2](super::hgnet) is: the same
//! stem and the same kind of block throughout, differing only in the table of
//! blocks and where the strides fall. Reading a page halves both axes at each
//! stage; reading a text line spends the height and keeps the length.
//!
//! What is new here, and what the whole generation is built around, is that the
//! published weights are **reparameterized**. During training a block's
//! depthwise mixer is three parallel branches and every convolution carries a
//! batch normalization; what is published is the single convolution they fold
//! into, bias and all. So this module has no normalization in it at all, and
//! the price is paid elsewhere: the fused convolutions were numbered when they
//! were created, after everything else, so their names interleave with the ones
//! that survived from training and cannot be walked with a counter. They are
//! taken in the order the graph reads them
//! ([`Order`](super::net::Order)) instead.
//!
//! Two more details are worth naming because each is silent when wrong:
//!
//! - **A block's residual skips the mixer, not the block.** What is added back
//!   is the depthwise mixer's output — after the gate, if there is one — and
//!   not the block's input.
//! - **The activation is the exact GELU.** The graph says `approximate: false`,
//!   so it is the error function and not the cheaper tanh curve that stands in
//!   for it nearly everywhere else.

use candle_core::Tensor;

use super::net::{
    Conv, HARD_SIGMOID_SLOPE, Loader, Order, SqueezeExcite, relu, subsample,
};
use crate::ocr::error::OcrError;

/// The reduction a block's squeeze-and-excitation gate squeezes by.
const GATE_REDUCE: usize = 4;
/// How much wider a block's channel mixer works than the block itself.
const EXPAND: usize = 2;

/// One block of the table: kernel, channels in and out, stride per axis, and
/// whether the mixer carries a gate.
#[derive(Debug, Clone, Copy)]
pub struct BlockSpec {
    pub kernel: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: (usize, usize),
    pub gate: bool,
}

/// A convolution whose two axes may stride by different amounts.
///
/// candle takes one stride for both, so an uneven pair convolves at stride one
/// and drops the rows the strided convolution would never have computed — the
/// same values, and cheap here because every such layer is depthwise.
#[derive(Debug)]
struct Strided {
    conv: Conv,
    keep: Option<(usize, usize)>,
}

impl Strided {
    fn load(
        loader: &Loader,
        name: &str,
        dims: [usize; 4],
        stride: (usize, usize),
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        let even = stride.0 == stride.1;
        Ok(Self {
            conv: Conv::load(
                loader,
                name,
                dims,
                if even { stride.0 } else { 1 },
                padding,
                groups,
            )?,
            keep: (!even).then_some(stride),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.conv.forward(x)?;
        match self.keep {
            None => Ok(y),
            Some(stride) => subsample(&y, stride),
        }
    }
}

/// One block: a depthwise mixer with an optional gate, then a widening and
/// narrowing pair that is added back when the shapes allow.
#[derive(Debug)]
struct Block {
    mixer: Strided,
    gate: Option<SqueezeExcite>,
    expand: Conv,
    compress: Conv,
    residual: bool,
}

impl Block {
    fn load(
        loader: &Loader,
        order: &mut Order,
        spec: &BlockSpec,
    ) -> Result<Self, OcrError> {
        let pad = spec.kernel / 2;
        let mixer = Strided::load(
            loader,
            order.take()?,
            [spec.in_channels, 1, spec.kernel, spec.kernel],
            spec.stride,
            pad,
            spec.in_channels,
        )?;
        let gate = if spec.gate {
            let down = order.take()?.to_string();
            let up = order.take()?;
            Some(SqueezeExcite::load(
                loader,
                &down,
                up,
                spec.in_channels,
                spec.in_channels / GATE_REDUCE,
                HARD_SIGMOID_SLOPE,
            )?)
        } else {
            None
        };
        let hidden = spec.in_channels * EXPAND;
        let expand = Conv::load(
            loader,
            order.take()?,
            [hidden, spec.in_channels, 1, 1],
            1,
            0,
            1,
        )?;
        let compress = Conv::load(
            loader,
            order.take()?,
            [spec.out_channels, hidden, 1, 1],
            1,
            0,
            1,
        )?;
        Ok(Self {
            mixer,
            gate,
            expand,
            compress,
            residual: spec.in_channels == spec.out_channels &&
                spec.stride == (1, 1),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mixed = self.mixer.forward(x)?;
        let mixed = match &self.gate {
            None => mixed,
            Some(gate) => gate.forward(&mixed)?,
        };
        let y = self.expand.forward(&mixed)?;
        let y = self.compress.forward(&gelu(&y)?)?;
        // What comes back is the mixer's output, not the block's input.
        if self.residual {
            Ok((mixed + y)?)
        } else {
            Ok(y)
        }
    }
}

/// The exact GELU. The graph asks for `approximate: false`, which is the error
/// function rather than the tanh curve that usually stands in for it.
fn gelu(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.gelu_erf()?) }

/// The stem: a strided convolution, a two-way split, and two more.
///
/// The split is the part that needs care: two kernels of size two run over the
/// first convolution's output padded by one on the right and the bottom, while
/// the same padded tensor goes through a max pool of that shape, and the two
/// are concatenated — the pooled half first.
#[derive(Debug)]
struct Stem {
    first: Conv,
    left: Conv,
    left2: Conv,
    third: Conv,
    fourth: Conv,
}

impl Stem {
    fn load(
        loader: &Loader,
        order: &mut Order,
        mid: usize,
        out: usize,
        stride: usize,
    ) -> Result<Self, OcrError> {
        let mut conv = |dims: [usize; 4], stride, padding| {
            Conv::load(loader, order.take()?, dims, stride, padding, 1)
        };
        Ok(Self {
            first: conv([mid, 3, 3, 3], 2, 1)?,
            left: conv([mid / 2, mid, 2, 2], 1, 0)?,
            left2: conv([mid, mid / 2, 2, 2], 1, 0)?,
            third: conv([mid, 2 * mid, 3, 3], stride, 1)?,
            fourth: conv([out, mid, 1, 1], 1, 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = relu(&self.first.forward(x)?)?;
        let padded = super::net::pad_end(&y, 1)?;
        let left = relu(&self.left.forward(&padded)?)?;
        let left = relu(&self.left2.forward(&super::net::pad_end(&left, 1)?)?)?;
        let right = padded.max_pool2d_with_stride(2, 1)?;
        let y = Tensor::cat(&[&right, &left], 1)?;
        let y = relu(&self.third.forward(&y)?)?;
        relu(&self.fourth.forward(&y)?)
    }
}

/// The backbone: a stem and a list of stages, each stage a list of blocks.
#[derive(Debug)]
pub struct Backbone {
    stem: Stem,
    stages: Vec<Vec<Block>>,
}

impl Backbone {
    /// Loads the stem and the stages the table describes, taking weights in
    /// the order the graph reads them.
    ///
    /// `stem` is `(middle, out)` channels and `stem_stride` the stride of the
    /// stem's second strided convolution — two when the backbone reads a page,
    /// two again when it reads a line, because here both shapes quarter the
    /// input before their stages begin.
    pub fn load(
        loader: &Loader,
        order: &mut Order,
        stem: (usize, usize),
        stem_stride: usize,
        stages: &[&[BlockSpec]],
    ) -> Result<Self, OcrError> {
        let stem = Stem::load(loader, order, stem.0, stem.1, stem_stride)?;
        let mut loaded = Vec::with_capacity(stages.len());
        for stage in stages {
            let mut blocks = Vec::with_capacity(stage.len());
            for spec in *stage {
                blocks.push(Block::load(loader, order, spec)?);
            }
            loaded.push(blocks);
        }
        Ok(Self {
            stem,
            stages: loaded,
        })
    }

    /// Every stage's output, finest first.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>, OcrError> {
        let mut state = self.stem.forward(x)?;
        let mut out = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                state = block.forward(&state)?;
            }
            out.push(state.clone());
        }
        Ok(out)
    }
}

/// Shorthand for one row of a stage table.
pub const fn block(
    in_channels: usize,
    out_channels: usize,
    stride: (usize, usize),
    gate: bool,
) -> BlockSpec {
    BlockSpec {
        kernel: 3,
        in_channels,
        out_channels,
        stride,
        gate,
    }
}
