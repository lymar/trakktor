//! The latent autoencoder on candle — the enhancer's two ends.
//!
//! The encoder turns a mel into a 64-wide latent; the decoder turns a latent
//! back into the 160 channels the vocoder wants (the mel it reconstructs, plus
//! thirty-two the vocoder is free to use for whatever the mel does not carry).
//! Between them sits the flow model ([`cfm`](super::cfm)), which is the part
//! that actually invents anything.
//!
//! **Both ends are used, and for different things.** The decoder is on the
//! output path. The encoder is on the *input* path: what the flow model starts
//! from is the encoding of the recording as it is, mixed with noise in the
//! proportion `--temperature` names. At temperature 1 the encoder's output is
//! discarded and the walk starts from pure noise.
//!
//! The name is upstream's: the rank of the latent is minimized implicitly, by
//! ending the encoder with four 1×1 projections that have no activation between
//! them. Four linear maps in a row are one linear map — which is the point, and
//! the reason they cannot be folded into one here either: they are trained as
//! four and their product is what it is.
//!
//! Ported from resemble-enhance (MIT).

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, Module, VarBuilder};

use super::super::config::{
    IRMAE_BLOCKS, IRMAE_DILATIONS, IRMAE_GROUPS, IRMAE_HIDDEN,
    IRMAE_PROJECTIONS, LATENT, MELS, VOCODER_INPUT,
};

/// The normalization's epsilon — torch's `GroupNorm` default.
const NORM_EPS: f64 = 1e-5;

/// A width-3 convolution whose padding keeps the length, at a dilation.
fn conv3(
    in_channels: usize,
    out_channels: usize,
    dilation: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    candle_nn::conv1d(
        in_channels,
        out_channels,
        3,
        Conv1dConfig {
            // `padding="same"` for an odd kernel is the dilation times half its
            // width, which for a width of three is the dilation itself.
            padding: dilation,
            dilation,
            ..Default::default()
        },
        vb,
    )
}

/// Four dilated convolutions, each behind a normalization and a rectifier,
/// added back onto the input.
#[derive(Debug)]
struct ResBlock {
    layers: Vec<(GroupNorm, Conv1d)>,
}

impl ResBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(IRMAE_DILATIONS.len());
        for (index, &dilation) in IRMAE_DILATIONS.iter().enumerate() {
            // A `Sequential` of normalization, rectifier, convolution, three
            // times over: the normalizations are 0, 3, 6, 9 and the
            // convolutions 2, 5, 8, 11.
            let at = index * 3;
            layers.push((
                candle_nn::group_norm(
                    IRMAE_GROUPS,
                    IRMAE_HIDDEN,
                    NORM_EPS,
                    vb.pp(format!("{at}")),
                )?,
                conv3(
                    IRMAE_HIDDEN,
                    IRMAE_HIDDEN,
                    dilation,
                    vb.pp(format!("{}", at + 2)),
                )?,
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut branch = xs.clone();
        for (norm, conv) in &self.layers {
            branch = conv.forward(&norm.forward(&branch)?.gelu_erf()?)?;
        }
        xs + branch
    }
}

/// The encoder: a mel in, a latent out.
#[derive(Debug)]
pub struct Encoder {
    lift: Conv1d,
    blocks: Vec<ResBlock>,
    projections: Vec<Conv1d>,
}

impl Encoder {
    /// Loads it from a checkpoint rooted at `vb`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a missing or misshapen tensor.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let lift = conv3(MELS, IRMAE_HIDDEN, 1, vb.pp("0"))?;
        let mut blocks = Vec::with_capacity(IRMAE_BLOCKS);
        for index in 0..IRMAE_BLOCKS {
            blocks.push(ResBlock::load(vb.pp(format!("{}", index + 1)))?);
        }
        let mut projections = Vec::with_capacity(IRMAE_PROJECTIONS);
        for index in 0..IRMAE_PROJECTIONS {
            let inputs = if index == 0 { IRMAE_HIDDEN } else { LATENT };
            projections.push(candle_nn::conv1d_no_bias(
                inputs,
                LATENT,
                1,
                Conv1dConfig::default(),
                vb.pp(format!("{}", index + 1 + IRMAE_BLOCKS)),
            )?);
        }
        Ok(Self {
            lift,
            blocks,
            projections,
        })
    }

    /// `[1, MELS, frames]` in, `[1, LATENT, frames]` out.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.lift.forward(xs)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        for projection in &self.projections {
            hidden = projection.forward(&hidden)?;
        }
        hidden.tanh()
    }
}

/// The decoder: a latent in, the vocoder's conditioning out.
#[derive(Debug)]
pub struct Decoder {
    lift: Conv1d,
    blocks: Vec<ResBlock>,
    out: Conv1d,
}

impl Decoder {
    /// Loads it from a checkpoint rooted at `vb`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a missing or misshapen tensor.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let lift = conv3(LATENT, IRMAE_HIDDEN, 1, vb.pp("0"))?;
        let mut blocks = Vec::with_capacity(IRMAE_BLOCKS);
        for index in 0..IRMAE_BLOCKS {
            blocks.push(ResBlock::load(vb.pp(format!("{}", index + 1)))?);
        }
        Ok(Self {
            lift,
            blocks,
            out: candle_nn::conv1d(
                IRMAE_HIDDEN,
                VOCODER_INPUT,
                1,
                Conv1dConfig::default(),
                vb.pp(format!("{}", IRMAE_BLOCKS + 1)),
            )?,
        })
    }

    /// `[1, LATENT, frames]` in, `[1, VOCODER_INPUT, frames]` out.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.lift.forward(xs)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.out.forward(&hidden)
    }
}
