//! The denoiser's network on candle: a UNet over the spectrum, three planes in
//! and three out.
//!
//! # What the three planes are
//!
//! In: a magnitude and the cosine and sine of its phase. Out: a gain, and two
//! numbers whose direction is a **rotation** of that phase. The second is the
//! interesting one — the network does not predict a phase, it predicts how far
//! to turn the one that came in, and the turn is applied by a complex multiply
//! ([`stft::apply`](super::super::stft::apply)). A real mask cannot turn
//! anything; this can.
//!
//! # Shape
//!
//! Four halvings down and four doublings back up, over both axes at once —
//! frequency and time are treated alike, as an image. Both axes are padded up
//! to a multiple of [`UNET_ALIGN`](super::super::config::UNET_ALIGN) first and
//! cropped back at the end, because four halvings need it.
//!
//! The width doubles at each step down: sixteen channels at the top, two
//! hundred and fifty-six at the bottom. Almost all of the arithmetic is
//! nevertheless at the **top**, where the resolution is: a thirty-second chunk
//! is 841 bins by 3151 frames, and thirty-two channels over that is more points
//! than two hundred and fifty-six channels over a sixteenth of it in each
//! direction.
//!
//! Ported from resemble-enhance (MIT).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, Module, VarBuilder};

use super::super::{
    config::{
        BINS, GROUP_CHANNELS, MAGPHASE_EPS, UNET_ALIGN, UNET_BLOCKS,
        UNET_HIDDEN, UNET_MIDDLE,
    },
    stft::{Prediction, Spectrum},
};

/// The normalization's epsilon — torch's `GroupNorm` default.
const NORM_EPS: f64 = 1e-5;

/// Elements above which a convolution is done in bands along time.
///
/// **candle's Metal backend returns zeros — silently — for a convolution whose
/// tensors pass about 2²⁶ elements.** Measured on this network: 848 bins by
/// 2416 frames by 32 channels (65.6 M) is right, 848 by 2528 by 32 (68.6 M) is
/// digital silence, and nothing anywhere reports a failure. A thirty-second
/// chunk is 3151 frames, so every full chunk of this engine was on the wrong
/// side of that line.
///
/// The answer is the same one the OCR detector needed: do the convolution in
/// bands. A 3×3 kernel reaches one frame either way, so bands that overlap by
/// one produce the same numbers — this is not an approximation, and the parity
/// tests are run with it on.
const CONV_BUDGET: usize = 48 << 20;

/// A convolution that knows how far its kernel reaches, so that it can be run
/// over bands of the time axis when the whole tensor would be too large.
#[derive(Debug)]
struct Conv {
    inner: Conv2d,
    reach: usize,
    out_channels: usize,
}

impl Conv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            inner: candle_nn::conv2d(
                in_channels,
                out_channels,
                kernel,
                Conv2dConfig {
                    padding: kernel / 2,
                    ..Default::default()
                },
                vb,
            )?,
            reach: kernel / 2,
            out_channels,
        })
    }

    /// The convolution, in bands along time when the tensors are large enough
    /// for the backend to get them wrong.
    ///
    /// A band takes `reach` frames of real context on each side and keeps only
    /// the outputs whose kernel sat entirely inside it, so the zero padding the
    /// convolution applies to a band's edges is used only where it is also the
    /// recording's edge. The numbers are the same as one pass.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = xs.dims4()?;
        let widest = channels.max(self.out_channels);
        let per_frame = batch * widest * height;
        let band = (CONV_BUDGET / per_frame.max(1)).clamp(1, width);
        if band >= width {
            return self.inner.forward(xs);
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < width {
            let count = band.min(width - start);
            let low = start.saturating_sub(self.reach);
            let high = (start + count + self.reach).min(width);
            let out = self
                .inner
                .forward(&xs.narrow(3, low, high - low)?.contiguous()?)?;
            pieces.push(out.narrow(3, start - low, count)?);
            start += count;
        }
        Tensor::cat(&pieces, 3)
    }
}

/// A 3×3 convolution that keeps its shape.
fn conv3x3(
    in_channels: usize,
    out_channels: usize,
    vb: VarBuilder,
) -> Result<Conv> {
    Conv::load(in_channels, out_channels, 3, vb)
}

/// A normalization with [`GROUP_CHANNELS`] channels per group.
fn norm(channels: usize, vb: VarBuilder) -> Result<GroupNorm> {
    candle_nn::group_norm(channels / GROUP_CHANNELS, channels, NORM_EPS, vb)
}

/// Nearest-neighbour resampling of both axes by the same factor.
///
/// Upstream reaches for `nn.Upsample` in both directions — a factor of two to
/// grow and a half to shrink — and torch's nearest mode is `floor(i · src/dst)`
/// either way. candle's kernel computes the same index, so one call covers
/// both; shrinking through an "upsample" reads oddly and is exactly what the
/// reference asks for.
fn resample(
    xs: &Tensor,
    numerator: usize,
    denominator: usize,
) -> Result<Tensor> {
    let (_, _, height, width) = xs.dims4()?;
    xs.upsample_nearest2d(
        height * numerator / denominator,
        width * numerator / denominator,
    )
}

/// Two convolutions with a normalization and a rectifier before each, added
/// back onto the input.
#[derive(Debug)]
struct PreactResBlock {
    norm1: GroupNorm,
    conv1: Conv,
    norm2: GroupNorm,
    conv2: Conv,
}

impl PreactResBlock {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        // The block is a `Sequential`, so its children are numbered: the two
        // normalizations are 0 and 3, the two convolutions 2 and 5, and the
        // rectifiers in between carry nothing.
        Ok(Self {
            norm1: norm(channels, vb.pp("0"))?,
            conv1: conv3x3(channels, channels, vb.pp("2"))?,
            norm2: norm(channels, vb.pp("3"))?,
            conv2: conv3x3(channels, channels, vb.pp("5"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let branch = self.norm1.forward(xs)?.gelu_erf()?;
        let branch = self.conv1.forward(&branch)?;
        let branch = self.norm2.forward(&branch)?.gelu_erf()?;
        let branch = self.conv2.forward(&branch)?;
        xs + branch
    }
}

/// Which way a block resamples, if at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    /// Halve both axes on the way out — an encoder block.
    Down,
    /// Keep them — a middle block.
    Same,
    /// Double both axes on the way in — a decoder block.
    Up,
}

/// One rung of the UNet.
#[derive(Debug)]
struct UnetBlock {
    pre: Conv,
    res1: PreactResBlock,
    res2: PreactResBlock,
    scale: Scale,
}

impl UnetBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        scale: Scale,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            pre: conv3x3(in_channels, out_channels, vb.pp("pre_conv"))?,
            res1: PreactResBlock::load(out_channels, vb.pp("res_block1"))?,
            res2: PreactResBlock::load(out_channels, vb.pp("res_block2"))?,
            scale,
        })
    }

    /// Returns what the next block takes, and what the block on the way back up
    /// adds in — which are the same tensor except in an encoder block, where
    /// the first is the halved one.
    fn forward(
        &self,
        xs: &Tensor,
        skip: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut hidden = if self.scale == Scale::Up {
            resample(xs, 2, 1)?
        } else {
            xs.clone()
        };
        if let Some(skip) = skip {
            hidden = (hidden + skip)?;
        }
        let hidden = self.pre.forward(&hidden)?;
        let hidden = self.res1.forward(&hidden)?;
        let hidden = self.res2.forward(&hidden)?;
        let out = if self.scale == Scale::Down {
            resample(&hidden, 1, 2)?
        } else {
            hidden.clone()
        };
        Ok((out, hidden))
    }
}

/// The whole network.
#[derive(Debug)]
pub struct Unet {
    input_proj: Conv,
    encoder: Vec<UnetBlock>,
    middle: Vec<UnetBlock>,
    decoder: Vec<UnetBlock>,
    head_conv: Conv,
    head_out: Conv,
    device: Device,
    dtype: DType,
}

impl Unet {
    /// Loads the network from a checkpoint rooted at `vb`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a missing or misshapen tensor.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let width = |level: usize| UNET_HIDDEN << level;
        let input_proj = conv3x3(3, UNET_HIDDEN, vb.pp("input_proj"))?;

        let mut encoder = Vec::with_capacity(UNET_BLOCKS);
        for level in 0..UNET_BLOCKS {
            encoder.push(UnetBlock::load(
                width(level),
                width(level + 1),
                Scale::Down,
                vb.pp(format!("encoder_blocks.{level}")),
            )?);
        }

        let mut middle = Vec::with_capacity(UNET_MIDDLE);
        for index in 0..UNET_MIDDLE {
            middle.push(UnetBlock::load(
                width(UNET_BLOCKS),
                width(UNET_BLOCKS),
                Scale::Same,
                vb.pp(format!("middle_blocks.{index}")),
            )?);
        }

        // Built from the bottom up: the first decoder block is the widest, and
        // it pairs with the last skip the encoder produced.
        let mut decoder = Vec::with_capacity(UNET_BLOCKS);
        for level in (0..UNET_BLOCKS).rev() {
            decoder.push(UnetBlock::load(
                width(level + 1),
                width(level),
                Scale::Up,
                vb.pp(format!("decoder_blocks.{}", UNET_BLOCKS - 1 - level)),
            )?);
        }

        Ok(Self {
            input_proj,
            encoder,
            middle,
            decoder,
            head_conv: conv3x3(UNET_HIDDEN, UNET_HIDDEN, vb.pp("head.0"))?,
            head_out: Conv::load(UNET_HIDDEN, 3, 1, vb.pp("head.2"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// The device the network is on.
    #[must_use]
    pub fn device(&self) -> &Device { &self.device }

    /// The three planes the network takes, as one `[1, 3, bins, frames]`
    /// tensor.
    fn input(&self, spectrum: &Spectrum) -> Result<Tensor> {
        let plane = |values: &[f32]| -> Result<Tensor> {
            Tensor::from_slice(
                values,
                (1, 1, BINS, spectrum.frames),
                &self.device,
            )?
            .to_dtype(self.dtype)
        };
        Tensor::cat(
            &[
                plane(&spectrum.magnitude)?,
                plane(&spectrum.cos)?,
                plane(&spectrum.sin)?,
            ],
            1,
        )
    }

    /// Runs the UNet over one already-assembled input.
    fn run(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = xs.dims4()?;
        let pad = |size: usize| (UNET_ALIGN - size % UNET_ALIGN) % UNET_ALIGN;
        let mut hidden = xs.pad_with_zeros(2, 0, pad(height))?.pad_with_zeros(
            3,
            0,
            pad(width),
        )?;
        hidden = self.input_proj.forward(&hidden)?;

        let mut skips = Vec::with_capacity(self.encoder.len());
        for block in &self.encoder {
            let (next, skip) = block.forward(&hidden, None)?;
            hidden = next;
            skips.push(skip);
        }
        for block in &self.middle {
            hidden = block.forward(&hidden, None)?.0;
        }
        for (block, skip) in self.decoder.iter().zip(skips.iter().rev()) {
            hidden = block.forward(&hidden, Some(skip))?.0;
        }

        hidden = self.head_conv.forward(&hidden)?.gelu_erf()?;
        hidden = self.head_out.forward(&hidden)?;
        hidden.narrow(2, 0, height)?.narrow(3, 0, width)
    }

    /// What the network predicts for one analysed chunk: a gain, and the
    /// direction to turn each phase by.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn predict(&self, spectrum: &Spectrum) -> Result<Prediction> {
        let out = self.run(&self.input(spectrum)?)?;
        let (mask, cos, sin) = self.split(&out)?;
        Ok(Prediction {
            mask: host(&mask)?,
            cos: host(&cos)?,
            sin: host(&sin)?,
        })
    }

    /// The three planes of the output, each turned into what it stands for:
    /// the first is squashed into a gain, the other two are squashed and then
    /// scaled to unit length so that they name a rotation and nothing else.
    fn split(&self, out: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let plane =
            |index: usize| out.narrow(1, index, 1)?.squeeze(1)?.squeeze(0);
        let mask = candle_nn::ops::sigmoid(&plane(0)?)?;
        let real = plane(1)?.tanh()?;
        let imag = plane(2)?.tanh()?;
        let size =
            ((real.sqr()? + imag.sqr()?)? + f64::from(MAGPHASE_EPS))?.sqrt()?;
        Ok((mask, (real / &size)?, (imag / &size)?))
    }

    /// Every stage of one run, for the parity tests.
    #[cfg(test)]
    pub(crate) fn stages(
        &self,
        spectrum: &Spectrum,
    ) -> Result<Vec<(String, Tensor)>> {
        let xs = self.input(spectrum)?;
        let (_, _, height, width) = xs.dims4()?;
        let pad = |size: usize| (UNET_ALIGN - size % UNET_ALIGN) % UNET_ALIGN;
        let mut hidden = xs.pad_with_zeros(2, 0, pad(height))?.pad_with_zeros(
            3,
            0,
            pad(width),
        )?;
        let mut stages = Vec::new();
        hidden = self.input_proj.forward(&hidden)?;
        stages.push(("unet_proj".to_owned(), hidden.clone()));

        let mut skips = Vec::with_capacity(self.encoder.len());
        for (index, block) in self.encoder.iter().enumerate() {
            let (next, skip) = block.forward(&hidden, None)?;
            stages.push((format!("unet_enc{index}_0"), next.clone()));
            stages.push((format!("unet_enc{index}_1"), skip.clone()));
            hidden = next;
            skips.push(skip);
        }
        for (index, block) in self.middle.iter().enumerate() {
            let (next, skip) = block.forward(&hidden, None)?;
            stages.push((format!("unet_mid{index}_0"), next.clone()));
            stages.push((format!("unet_mid{index}_1"), skip));
            hidden = next;
        }
        for (index, (block, skip)) in
            self.decoder.iter().zip(skips.iter().rev()).enumerate()
        {
            let (next, inner) = block.forward(&hidden, Some(skip))?;
            stages.push((format!("unet_dec{index}_0"), next.clone()));
            stages.push((format!("unet_dec{index}_1"), inner));
            hidden = next;
        }
        hidden = self.head_conv.forward(&hidden)?.gelu_erf()?;
        hidden = self.head_out.forward(&hidden)?;
        stages.push(("unet_head".to_owned(), hidden.clone()));
        let out = hidden.narrow(2, 0, height)?.narrow(3, 0, width)?;
        let (mask, cos, sin) = self.split(&out)?;
        stages.push(("mask".to_owned(), mask));
        stages.push(("cos_res".to_owned(), cos));
        stages.push(("sin_res".to_owned(), sin));
        Ok(stages)
    }
}

/// A tensor as host `f32` values.
fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}
