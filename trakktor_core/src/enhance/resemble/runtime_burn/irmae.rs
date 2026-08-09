//! The latent autoencoder on burn — the same network as
//! [the candle one](super::super::runtime::irmae).
//!
//! Ported from resemble-enhance (MIT).

use burn::tensor::{
    Tensor, activation, backend::Backend, module::conv1d, ops::ConvOptions,
};

use super::{Weights, weight};
use crate::enhance::{
    EnhanceError,
    resemble::config::{
        IRMAE_BLOCKS, IRMAE_DILATIONS, IRMAE_GROUPS, IRMAE_HIDDEN,
        IRMAE_PROJECTIONS, LATENT, MELS, VOCODER_INPUT,
    },
};

/// The normalization's epsilon — torch's `GroupNorm` default.
const NORM_EPS: f64 = 1e-5;

/// A one-dimensional convolution, optionally without a bias.
pub(super) struct Conv<B: Backend> {
    pub weight: Tensor<B, 3>,
    pub bias: Option<Tensor<B, 1>>,
    pub padding: usize,
    pub dilation: usize,
}

impl<B: Backend> Conv<B> {
    #[allow(
        clippy::too_many_arguments,
        reason = "a convolution is described by exactly these, and naming \
                  them in a struct would only move the list"
    )]
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_channels, in_channels, kernel],
            )?,
            bias: if bias {
                Some(weight(
                    weights,
                    device,
                    &format!("{prefix}.bias"),
                    [out_channels],
                )?)
            } else {
                None
            },
            padding: dilation * (kernel / 2),
            dilation,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([1], [self.padding], [self.dilation], 1),
        )
    }
}

/// Normalization over a fixed number of groups.
pub(super) struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    groups: usize,
}

impl<B: Backend> Norm<B> {
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
        groups: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [channels],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [channels],
            )?,
            groups,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, length] = x.dims();
        let per_group = channels / self.groups * length;
        let grouped = x.reshape([batch, self.groups, per_group]);
        let mean = grouped.clone().mean_dim(2);
        let centered = grouped - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(2);
        let normed = (centered / (variance + NORM_EPS).sqrt())
            .reshape([batch, channels, length]);
        normed * self.weight.clone().reshape([1, channels, 1]) +
            self.bias.clone().reshape([1, channels, 1])
    }
}

/// Four dilated convolutions, each behind a normalization and a rectifier,
/// added back onto the input.
struct ResBlock<B: Backend> {
    layers: Vec<(Norm<B>, Conv<B>)>,
}

impl<B: Backend> ResBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        let mut layers = Vec::with_capacity(IRMAE_DILATIONS.len());
        for (index, &dilation) in IRMAE_DILATIONS.iter().enumerate() {
            let at = index * 3;
            layers.push((
                Norm::load(
                    weights,
                    device,
                    &format!("{prefix}.{at}"),
                    IRMAE_HIDDEN,
                    IRMAE_GROUPS,
                )?,
                Conv::load(
                    weights,
                    device,
                    &format!("{prefix}.{}", at + 2),
                    IRMAE_HIDDEN,
                    IRMAE_HIDDEN,
                    3,
                    dilation,
                    true,
                )?,
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut branch = x.clone();
        for (norm, conv) in &self.layers {
            branch = conv.forward(activation::gelu(norm.forward(branch)));
        }
        x + branch
    }
}

/// The encoder: a mel in, a latent out.
pub struct Encoder<B: Backend> {
    lift: Conv<B>,
    blocks: Vec<ResBlock<B>>,
    projections: Vec<Conv<B>>,
}

impl<B: Backend> Encoder<B> {
    /// Loads it.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when a weight is missing or has the
    /// wrong shape.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        let root = "lcfm.ae.encoder";
        let mut blocks = Vec::with_capacity(IRMAE_BLOCKS);
        for index in 0..IRMAE_BLOCKS {
            blocks.push(ResBlock::load(
                weights,
                device,
                &format!("{root}.{}", index + 1),
            )?);
        }
        let mut projections = Vec::with_capacity(IRMAE_PROJECTIONS);
        for index in 0..IRMAE_PROJECTIONS {
            let inputs = if index == 0 { IRMAE_HIDDEN } else { LATENT };
            projections.push(Conv::load(
                weights,
                device,
                &format!("{root}.{}", index + 1 + IRMAE_BLOCKS),
                inputs,
                LATENT,
                1,
                1,
                false,
            )?);
        }
        Ok(Self {
            lift: Conv::load(
                weights,
                device,
                &format!("{root}.0"),
                MELS,
                IRMAE_HIDDEN,
                3,
                1,
                true,
            )?,
            blocks,
            projections,
        })
    }

    /// `[1, MELS, frames]` in, `[1, LATENT, frames]` out.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = self.lift.forward(x);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        for projection in &self.projections {
            hidden = projection.forward(hidden);
        }
        activation::tanh(hidden)
    }
}

/// The decoder: a latent in, the vocoder's conditioning out.
pub struct Decoder<B: Backend> {
    lift: Conv<B>,
    blocks: Vec<ResBlock<B>>,
    out: Conv<B>,
}

impl<B: Backend> Decoder<B> {
    /// Loads it.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when a weight is missing or has the
    /// wrong shape.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        let root = "lcfm.ae.decoder";
        let mut blocks = Vec::with_capacity(IRMAE_BLOCKS);
        for index in 0..IRMAE_BLOCKS {
            blocks.push(ResBlock::load(
                weights,
                device,
                &format!("{root}.{}", index + 1),
            )?);
        }
        Ok(Self {
            lift: Conv::load(
                weights,
                device,
                &format!("{root}.0"),
                LATENT,
                IRMAE_HIDDEN,
                3,
                1,
                true,
            )?,
            blocks,
            out: Conv::load(
                weights,
                device,
                &format!("{root}.{}", IRMAE_BLOCKS + 1),
                IRMAE_HIDDEN,
                VOCODER_INPUT,
                1,
                1,
                true,
            )?,
        })
    }

    /// `[1, LATENT, frames]` in, `[1, VOCODER_INPUT, frames]` out.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = self.lift.forward(x);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        self.out.forward(hidden)
    }
}
