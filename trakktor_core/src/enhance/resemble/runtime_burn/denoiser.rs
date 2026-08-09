//! The denoiser's UNet on burn — the same network as
//! [the candle one](super::super::runtime::denoiser), written against burn's
//! functional tensor operations.
//!
//! Two things are written out here rather than called:
//!
//! - **the resampling**. Upstream reaches for nearest-neighbour interpolation
//!   in both directions; at a factor of exactly two that is a repeat on the way
//!   up and a stride on the way down, and writing them as a reshape avoids
//!   depending on how a backend rounds an interpolation index;
//! - **the group normalization**, which burn has as a module rather than as a
//!   function and which is four lines either way.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.
//!
//! Ported from resemble-enhance (MIT).

use burn::tensor::{
    Tensor, activation,
    backend::Backend,
    module::conv2d,
    ops::{ConvOptions, PadMode},
};

use super::{Weights, weight};
use crate::enhance::{
    EnhanceError,
    resemble::{
        config::{
            BINS, GROUP_CHANNELS, MAGPHASE_EPS, UNET_ALIGN, UNET_BLOCKS,
            UNET_HIDDEN, UNET_MIDDLE,
        },
        stft::{Prediction, Spectrum},
    },
};

/// The normalization's epsilon — torch's `GroupNorm` default.
const NORM_EPS: f64 = 1e-5;

/// A convolution with its kernel and bias.
struct Conv<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
    padding: usize,
}

impl<B: Backend> Conv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_channels, in_channels, kernel, kernel],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_channels],
            )?,
            padding: kernel / 2,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1, 1], [self.padding, self.padding], [1, 1], 1),
        )
    }
}

/// Normalization over groups of [`GROUP_CHANNELS`] channels.
struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    groups: usize,
}

impl<B: Backend> Norm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
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
            groups: channels / GROUP_CHANNELS,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let per_group = channels / self.groups * height * width;
        let grouped = x.reshape([batch, self.groups, per_group]);
        let mean = grouped.clone().mean_dim(2);
        let centered = grouped - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(2);
        let normed = (centered / (variance + NORM_EPS).sqrt())
            .reshape([batch, channels, height, width]);
        normed * self.weight.clone().reshape([1, channels, 1, 1]) +
            self.bias.clone().reshape([1, channels, 1, 1])
    }
}

/// Two convolutions with a normalization and a rectifier before each, added
/// back onto the input.
struct PreactResBlock<B: Backend> {
    norm1: Norm<B>,
    conv1: Conv<B>,
    norm2: Norm<B>,
    conv2: Conv<B>,
}

impl<B: Backend> PreactResBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            norm1: Norm::load(
                weights,
                device,
                &format!("{prefix}.0"),
                channels,
            )?,
            conv1: Conv::load(
                weights,
                device,
                &format!("{prefix}.2"),
                channels,
                channels,
                3,
            )?,
            norm2: Norm::load(
                weights,
                device,
                &format!("{prefix}.3"),
                channels,
            )?,
            conv2: Conv::load(
                weights,
                device,
                &format!("{prefix}.5"),
                channels,
                channels,
                3,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let branch = activation::gelu(self.norm1.forward(x.clone()));
        let branch = self.conv1.forward(branch);
        let branch = activation::gelu(self.norm2.forward(branch));
        x + self.conv2.forward(branch)
    }
}

/// Which way a block resamples, if at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scale {
    Down,
    Same,
    Up,
}

/// One rung of the UNet.
struct UnetBlock<B: Backend> {
    pre: Conv<B>,
    res1: PreactResBlock<B>,
    res2: PreactResBlock<B>,
    scale: Scale,
}

impl<B: Backend> UnetBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        scale: Scale,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            pre: Conv::load(
                weights,
                device,
                &format!("{prefix}.pre_conv"),
                in_channels,
                out_channels,
                3,
            )?,
            res1: PreactResBlock::load(
                weights,
                device,
                &format!("{prefix}.res_block1"),
                out_channels,
            )?,
            res2: PreactResBlock::load(
                weights,
                device,
                &format!("{prefix}.res_block2"),
                out_channels,
            )?,
            scale,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 4>,
        skip: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut hidden = if self.scale == Scale::Up { grow(x) } else { x };
        if let Some(skip) = skip {
            hidden = hidden + skip;
        }
        let hidden = self
            .res2
            .forward(self.res1.forward(self.pre.forward(hidden)));
        let out = if self.scale == Scale::Down {
            shrink(hidden.clone())
        } else {
            hidden.clone()
        };
        (out, hidden)
    }
}

/// Nearest-neighbour interpolation by two on both axes: every point becomes a
/// two-by-two square.
fn grow<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    x.reshape([batch, channels, height, 1, width, 1])
        .expand([batch, channels, height, 2, width, 2])
        .reshape([batch, channels, height * 2, width * 2])
}

/// And its inverse: every other point of both axes, which is what torch's
/// nearest mode picks at a factor of one half.
fn shrink<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    x.reshape([batch, channels, height / 2, 2, width / 2, 2])
        .narrow(3, 0, 1)
        .narrow(5, 0, 1)
        .reshape([batch, channels, height / 2, width / 2])
}

/// The whole network.
pub struct Unet<B: Backend> {
    input_proj: Conv<B>,
    encoder: Vec<UnetBlock<B>>,
    middle: Vec<UnetBlock<B>>,
    decoder: Vec<UnetBlock<B>>,
    head_conv: Conv<B>,
    head_out: Conv<B>,
}

impl<B: Backend> Unet<B> {
    /// Loads the network under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when a weight is missing or has the
    /// wrong shape.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        let width = |level: usize| UNET_HIDDEN << level;
        let mut encoder = Vec::with_capacity(UNET_BLOCKS);
        for level in 0..UNET_BLOCKS {
            encoder.push(UnetBlock::load(
                weights,
                device,
                &format!("{prefix}.encoder_blocks.{level}"),
                width(level),
                width(level + 1),
                Scale::Down,
            )?);
        }
        let mut middle = Vec::with_capacity(UNET_MIDDLE);
        for index in 0..UNET_MIDDLE {
            middle.push(UnetBlock::load(
                weights,
                device,
                &format!("{prefix}.middle_blocks.{index}"),
                width(UNET_BLOCKS),
                width(UNET_BLOCKS),
                Scale::Same,
            )?);
        }
        let mut decoder = Vec::with_capacity(UNET_BLOCKS);
        for level in (0..UNET_BLOCKS).rev() {
            decoder.push(UnetBlock::load(
                weights,
                device,
                &format!("{prefix}.decoder_blocks.{}", UNET_BLOCKS - 1 - level),
                width(level + 1),
                width(level),
                Scale::Up,
            )?);
        }
        Ok(Self {
            input_proj: Conv::load(
                weights,
                device,
                &format!("{prefix}.input_proj"),
                3,
                UNET_HIDDEN,
                3,
            )?,
            encoder,
            middle,
            decoder,
            head_conv: Conv::load(
                weights,
                device,
                &format!("{prefix}.head.0"),
                UNET_HIDDEN,
                UNET_HIDDEN,
                3,
            )?,
            head_out: Conv::load(
                weights,
                device,
                &format!("{prefix}.head.2"),
                UNET_HIDDEN,
                3,
                1,
            )?,
        })
    }

    /// What the network predicts for one analysed chunk.
    pub fn predict(
        &self,
        spectrum: &Spectrum,
        device: &B::Device,
    ) -> Prediction {
        let frames = spectrum.frames;
        let plane = |values: &[f32]| -> Tensor<B, 4> {
            Tensor::from_data(
                burn::tensor::TensorData::new(
                    values.to_vec(),
                    [1, 1, BINS, frames],
                ),
                device,
            )
        };
        let input = Tensor::cat(
            vec![
                plane(&spectrum.magnitude),
                plane(&spectrum.cos),
                plane(&spectrum.sin),
            ],
            1,
        );
        let out = self.run(input);
        let mask = activation::sigmoid(out.clone().narrow(1, 0, 1));
        let real = activation::tanh(out.clone().narrow(1, 1, 1));
        let imag = activation::tanh(out.narrow(1, 2, 1));
        let size = (real.clone().powi_scalar(2) +
            imag.clone().powi_scalar(2) +
            f64::from(MAGPHASE_EPS))
        .sqrt();
        Prediction {
            mask: host(mask),
            cos: host(real / size.clone()),
            sin: host(imag / size),
        }
    }

    /// Runs the UNet over one already-assembled input.
    fn run(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, height, width] = x.dims();
        let pad = |size: usize| (UNET_ALIGN - size % UNET_ALIGN) % UNET_ALIGN;
        let mut hidden =
            x.pad([(0, pad(height)), (0, pad(width))], PadMode::Constant(0.0));
        hidden = self.input_proj.forward(hidden);

        let mut skips = Vec::with_capacity(self.encoder.len());
        for block in &self.encoder {
            let (next, skip) = block.forward(hidden, None);
            hidden = next;
            skips.push(skip);
        }
        for block in &self.middle {
            hidden = block.forward(hidden, None).0;
        }
        for (block, skip) in self.decoder.iter().zip(skips.into_iter().rev()) {
            hidden = block.forward(hidden, Some(skip)).0;
        }
        hidden = activation::gelu(self.head_conv.forward(hidden));
        hidden = self.head_out.forward(hidden);
        hidden.narrow(2, 0, height).narrow(3, 0, width)
    }
}

/// A tensor as host `f32` values.
fn host<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .expect("a stage as f32 values")
}
