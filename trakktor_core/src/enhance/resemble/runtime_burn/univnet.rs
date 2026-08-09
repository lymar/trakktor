//! The waveform generator on burn — the same network as
//! [the candle one](super::super::runtime::univnet), with the same two
//! decisions written out: the anti-aliased resamplers as twelve shifted
//! multiplies rather than a grouped convolution, and the location-variable
//! convolution as one batched matmul per slice of conditioning frames.
//!
//! Ported from resemble-enhance (MIT), whose vocoder follows UnivNet and
//! LVCNet, with the anti-aliased activation from BigVGAN.

use burn::tensor::{
    Tensor, activation,
    backend::Backend,
    module::{conv_transpose1d, conv1d},
    ops::{ConvOptions, ConvTransposeOptions, PadMode},
};

use super::{Weights, irmae::Conv, weight};
use crate::enhance::{
    EnhanceError,
    resemble::config::{
        AMP_DILATIONS, KPNET_HIDDEN, KPNET_KERNEL, LEAKY_SLOPE, LVC_DILATIONS,
        NOISE_CHANNELS, RESAMPLE_RATIO, RESAMPLE_TAPS, SNAKE_CLAMP, STRIDES,
        UNIVNET_CHANNELS, VOCODER_INPUT, VOCODER_PAD,
    },
};

/// Largest gathered location-variable batch, in elements — the same budget the
/// candle runtime keeps, so the two split the frames the same way.
const LVC_BUDGET: usize = 16 << 20;

/// How many noise values one run over `frames` of conditioning needs.
#[must_use]
pub fn noise_len(frames: usize) -> usize {
    NOISE_CHANNELS * (frames + VOCODER_PAD)
}

/// A rectifier with a small slope below zero.
fn leaky<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    activation::leaky_relu(x, LEAKY_SLOPE)
}

/// The Snake activation: `x + sin²(αx) / β`.
struct Snake<B: Backend> {
    alpha: Tensor<B, 3>,
    beta: Tensor<B, 3>,
}

impl<B: Backend> Snake<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, EnhanceError> {
        let read = |name: &str| -> Result<Tensor<B, 3>, EnhanceError> {
            let raw: Tensor<B, 1> = weight(
                weights,
                device,
                &format!("{prefix}.{name}"),
                [channels],
            )?;
            Ok(raw
                .exp()
                .clamp(SNAKE_CLAMP.0, SNAKE_CLAMP.1)
                .reshape([1, channels, 1]))
        };
        Ok(Self {
            alpha: read("log_alpha")?,
            beta: read("log_beta")?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let wave = (x.clone() * self.alpha.clone()).sin().powi_scalar(2);
        x + wave / self.beta.clone()
    }
}

/// A half-band filter, held as its twelve taps rather than as a kernel.
struct HalfBand {
    taps: Vec<f64>,
}

impl HalfBand {
    fn load(weights: &Weights, key: &str) -> Result<Self, EnhanceError> {
        let (values, dims) = weights.parts(key)?;
        if dims != [1, 1, RESAMPLE_TAPS] {
            return Err(EnhanceError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected [1, 1, {RESAMPLE_TAPS}]"
            )));
        }
        Ok(Self {
            taps: values.into_iter().map(f64::from).collect(),
        })
    }

    /// Two-to-one interpolation, by its two polyphase branches.
    fn up<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, len] = x.dims();
        let half = RESAMPLE_TAPS / RESAMPLE_RATIO;
        let margin = half - 1;
        let span = len + 2 * margin;
        let reach = half - 1;
        let padded = x.pad([(margin, margin)], PadMode::Edge);
        let zeroed = padded.pad([(reach, reach)], PadMode::Constant(0.0));
        let branch = |parity: usize| -> Tensor<B, 3> {
            let mut sum: Option<Tensor<B, 3>> = None;
            for step in 0..half {
                let term = zeroed.clone().narrow(2, reach - step, span + reach) *
                    self.taps[RESAMPLE_RATIO * step + parity];
                sum = Some(match sum {
                    Some(total) => total + term,
                    None => term,
                });
            }
            sum.expect("a filter with taps")
        };
        let woven = Tensor::stack::<4>(vec![branch(0), branch(1)], 3)
            .reshape([batch, channels, RESAMPLE_RATIO * (span + reach)]);
        let trim =
            margin * RESAMPLE_RATIO + (RESAMPLE_TAPS - RESAMPLE_RATIO) / 2;
        woven.narrow(2, trim, len * RESAMPLE_RATIO) * RESAMPLE_RATIO as f64
    }

    /// One-to-two decimation through the same filter, likewise by polyphase.
    fn down<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, len] = x.dims();
        let half = RESAMPLE_TAPS / RESAMPLE_RATIO;
        let padded = x.pad([(half - 1, half)], PadMode::Edge);
        let padded = padded.pad([(0, 1)], PadMode::Constant(0.0));
        let span = padded.dims()[2];
        let phases = padded.reshape([
            batch,
            channels,
            span / RESAMPLE_RATIO,
            RESAMPLE_RATIO,
        ]);
        let out = len / RESAMPLE_RATIO;
        let mut sum: Option<Tensor<B, 3>> = None;
        for step in 0..half {
            for parity in 0..RESAMPLE_RATIO {
                let phase = phases
                    .clone()
                    .narrow(3, parity, 1)
                    .squeeze_dim::<3>(3)
                    .narrow(2, step, out);
                let term = phase * self.taps[RESAMPLE_RATIO * step + parity];
                sum = Some(match sum {
                    Some(total) => total + term,
                    None => term,
                });
            }
        }
        sum.expect("a filter with taps")
    }
}

/// One anti-aliased layer.
struct AmpLayer<B: Backend> {
    first: Conv<B>,
    snake: Snake<B>,
    up: HalfBand,
    down: HalfBand,
    second: Conv<B>,
}

impl<B: Backend> AmpLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
        dilation: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            first: Conv::load(
                weights,
                device,
                &format!("{prefix}.0"),
                channels,
                channels,
                3,
                dilation,
                true,
            )?,
            snake: Snake::load(
                weights,
                device,
                &format!("{prefix}.1.act"),
                channels,
            )?,
            up: HalfBand::load(
                weights,
                &format!("{prefix}.1.upsample.filter"),
            )?,
            down: HalfBand::load(
                weights,
                &format!("{prefix}.1.downsample.lowpass.filter"),
            )?,
            second: Conv::load(
                weights,
                device,
                &format!("{prefix}.2"),
                channels,
                channels,
                3,
                1,
                true,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.first.forward(x);
        let hidden = self.up.up(hidden);
        let hidden = self.snake.forward(hidden);
        let hidden = self.down.down(hidden);
        self.second.forward(hidden)
    }
}

/// Three of those, added back onto the input.
struct AmpBlock<B: Backend> {
    layers: Vec<AmpLayer<B>>,
}

impl<B: Backend> AmpBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, EnhanceError> {
        let mut layers = Vec::with_capacity(AMP_DILATIONS.len());
        for (index, &dilation) in AMP_DILATIONS.iter().enumerate() {
            layers.push(AmpLayer::load(
                weights,
                device,
                &format!("{prefix}.{index}"),
                channels,
                dilation,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        x + hidden
    }
}

/// The network that emits a convolution kernel per conditioning frame.
///
/// As in [the candle runtime](super::super::runtime::univnet), the kernels are
/// never produced whole: the one convolution upstream predicts all four layers
/// with is split into four at load time, and each layer's kernels are produced
/// a slice of frames at a time inside the loop that consumes them.
struct KernelPredictor<B: Backend> {
    input: Conv<B>,
    residual: Vec<(Conv<B>, Conv<B>)>,
    kernels: Vec<Conv<B>>,
    biases: Conv<B>,
}

impl<B: Backend> KernelPredictor<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, EnhanceError> {
        let layers = LVC_DILATIONS.len();
        let per_layer = channels * 2 * channels * KPNET_KERNEL;
        let bias_channels = 2 * channels * layers;
        let mut residual = Vec::with_capacity(3);
        for index in 0..3 {
            let at = format!("{prefix}.residual_convs.{index}");
            residual.push((
                Conv::load(
                    weights,
                    device,
                    &format!("{at}.1"),
                    KPNET_HIDDEN,
                    KPNET_HIDDEN,
                    KPNET_KERNEL,
                    1,
                    true,
                )?,
                Conv::load(
                    weights,
                    device,
                    &format!("{at}.3"),
                    KPNET_HIDDEN,
                    KPNET_HIDDEN,
                    KPNET_KERNEL,
                    1,
                    true,
                )?,
            ));
        }
        // The one convolution, read once and cut into four along its slowest
        // axis. Its padding is taken over by the caller, which pads the frames
        // it hands in.
        let full: Tensor<B, 3> = weight(
            weights,
            device,
            &format!("{prefix}.kernel_conv.weight"),
            [per_layer * layers, KPNET_HIDDEN, KPNET_KERNEL],
        )?;
        let full_bias: Tensor<B, 1> = weight(
            weights,
            device,
            &format!("{prefix}.kernel_conv.bias"),
            [per_layer * layers],
        )?;
        let mut kernels = Vec::with_capacity(layers);
        for index in 0..layers {
            kernels.push(Conv {
                weight: full.clone().narrow(0, index * per_layer, per_layer),
                bias: Some(full_bias.clone().narrow(
                    0,
                    index * per_layer,
                    per_layer,
                )),
                padding: 0,
                dilation: 1,
            });
        }
        Ok(Self {
            input: Conv::load(
                weights,
                device,
                &format!("{prefix}.input_conv.0"),
                VOCODER_INPUT,
                KPNET_HIDDEN,
                5,
                1,
                true,
            )?,
            residual,
            kernels,
            biases: Conv::load(
                weights,
                device,
                &format!("{prefix}.bias_conv"),
                KPNET_HIDDEN,
                bias_channels,
                KPNET_KERNEL,
                1,
                true,
            )?,
        })
    }

    /// Reads the conditioning into the representation the kernels are made
    /// from, padded by the one frame of context the width-three convolution
    /// needs at each end, and the biases, which are small enough to keep whole.
    fn forward(
        &self,
        cond: Tensor<B, 3>,
        channels: usize,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let frames = cond.dims()[2];
        let mut hidden = leaky(self.input.forward(cond));
        for (first, second) in &self.residual {
            let branch = leaky(first.forward(hidden.clone()));
            let branch = leaky(second.forward(branch));
            hidden = hidden + branch;
        }
        let layers = LVC_DILATIONS.len();
        let out_channels = 2 * channels;
        let biases = self.biases.forward(hidden.clone()).reshape([
            layers,
            out_channels,
            frames,
        ]);
        let per_layer = (0..layers)
            .map(|index| {
                biases
                    .clone()
                    .narrow(0, index, 1)
                    .squeeze_dim::<2>(0)
                    .swap_dims(0, 1)
                    .reshape([frames, 1, out_channels])
            })
            .collect();
        (hidden.pad([(1, 1)], PadMode::Constant(0.0)), per_layer)
    }

    /// One layer's kernels for a range of frames, as `[frames, taps · in,
    /// out]`.
    fn slice(
        &self,
        layer: usize,
        hidden: &Tensor<B, 3>,
        start: usize,
        count: usize,
        channels: usize,
    ) -> Tensor<B, 3> {
        let out_channels = 2 * channels;
        self.kernels[layer]
            .forward(hidden.clone().narrow(2, start, count + 2))
            .reshape([channels, out_channels, KPNET_KERNEL, count])
            .permute([3, 2, 0, 1])
            .reshape([count, KPNET_KERNEL * channels, out_channels])
    }
}

/// One upsampling block.
struct LvcBlock<B: Backend> {
    up_weight: Tensor<B, 3>,
    up_bias: Tensor<B, 1>,
    up_options: ConvTransposeOptions<1>,
    amp: AmpBlock<B>,
    predictor: KernelPredictor<B>,
    convs: Vec<Conv<B>>,
    hop: usize,
    channels: usize,
}

impl<B: Backend> LvcBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
        stride: usize,
        hop: usize,
    ) -> Result<Self, EnhanceError> {
        let mut convs = Vec::with_capacity(LVC_DILATIONS.len());
        for (index, &dilation) in LVC_DILATIONS.iter().enumerate() {
            convs.push(Conv::load(
                weights,
                device,
                &format!("{prefix}.conv_blocks.{index}.1"),
                channels,
                channels,
                3,
                dilation,
                true,
            )?);
        }
        Ok(Self {
            up_weight: weight(
                weights,
                device,
                &format!("{prefix}.convt_pre.1.weight"),
                [channels, channels, 2 * stride],
            )?,
            up_bias: weight(
                weights,
                device,
                &format!("{prefix}.convt_pre.1.bias"),
                [channels],
            )?,
            up_options: ConvTransposeOptions::new(
                [stride],
                [stride / 2 + stride % 2],
                [stride % 2],
                [1],
                1,
            ),
            amp: AmpBlock::load(
                weights,
                device,
                &format!("{prefix}.amp_block"),
                channels,
            )?,
            predictor: KernelPredictor::load(
                weights,
                device,
                &format!("{prefix}.kernel_predictor"),
                channels,
            )?,
            convs,
            hop,
            channels,
        })
    }

    fn forward(&self, x: Tensor<B, 3>, cond: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = conv_transpose1d(
            leaky(x),
            self.up_weight.clone(),
            Some(self.up_bias.clone()),
            self.up_options.clone(),
        );
        hidden = self.amp.forward(hidden);
        let (conditioning, biases) =
            self.predictor.forward(cond, self.channels);
        for (index, (conv, bias)) in self.convs.iter().zip(&biases).enumerate()
        {
            let branch = leaky(conv.forward(leaky(hidden.clone())));
            let out = self.variable_conv(branch, index, &conditioning, bias);
            let gate =
                activation::sigmoid(out.clone().narrow(1, 0, self.channels));
            let value =
                activation::tanh(out.narrow(1, self.channels, self.channels));
            hidden = hidden + gate * value;
        }
        hidden
    }

    /// The convolution whose kernel changes with the conditioning frame. Both
    /// the kernels and the gathered taps are produced a slice of frames at a
    /// time, under one budget.
    fn variable_conv(
        &self,
        x: Tensor<B, 3>,
        layer: usize,
        conditioning: &Tensor<B, 3>,
        bias: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_, channels, len] = x.dims();
        let frames = len / self.hop;
        let out_channels = 2 * channels;
        let taps = KPNET_KERNEL;
        let padded = x.pad([(taps / 2, taps / 2)], PadMode::Constant(0.0));

        let per_frame =
            self.hop * taps * channels + taps * channels * out_channels;
        let slice = (LVC_BUDGET / per_frame.max(1)).clamp(1, frames);
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < frames {
            let count = slice.min(frames - start);
            let span = count * self.hop;
            let gathered: Vec<Tensor<B, 3>> = (0..taps)
                .map(|tap| {
                    padded
                        .clone()
                        .narrow(2, start * self.hop + tap, span)
                        .reshape([channels, count, self.hop])
                        .permute([1, 2, 0])
                })
                .collect();
            let features = Tensor::stack::<4>(gathered, 2).reshape([
                count,
                self.hop,
                taps * channels,
            ]);
            let kernel = self.predictor.slice(
                layer,
                conditioning,
                start,
                count,
                channels,
            );
            let piece =
                features.matmul(kernel) + bias.clone().narrow(0, start, count);
            pieces.push(piece.permute([2, 0, 1]).reshape([out_channels, span]));
            start += count;
        }
        Tensor::cat(pieces, 1).reshape([1, out_channels, len])
    }
}

/// The vocoder.
pub struct UnivNet<B: Backend> {
    pre_weight: Tensor<B, 3>,
    pre_bias: Tensor<B, 1>,
    blocks: Vec<LvcBlock<B>>,
    post_weight: Tensor<B, 3>,
    post_bias: Tensor<B, 1>,
}

impl<B: Backend> UnivNet<B> {
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
        let mut blocks = Vec::with_capacity(STRIDES.len());
        let mut hop = 1;
        for (index, &stride) in STRIDES.iter().enumerate() {
            hop *= stride;
            blocks.push(LvcBlock::load(
                weights,
                device,
                &format!("vocoder.blocks.{index}"),
                UNIVNET_CHANNELS,
                stride,
                hop,
            )?);
        }
        Ok(Self {
            pre_weight: weight(
                weights,
                device,
                "vocoder.conv_pre.weight",
                [UNIVNET_CHANNELS, NOISE_CHANNELS, 7],
            )?,
            pre_bias: weight(
                weights,
                device,
                "vocoder.conv_pre.bias",
                [UNIVNET_CHANNELS],
            )?,
            blocks,
            post_weight: weight(
                weights,
                device,
                "vocoder.conv_post.1.weight",
                [1, UNIVNET_CHANNELS, 7],
            )?,
            post_bias: weight(
                weights,
                device,
                "vocoder.conv_post.1.bias",
                [1],
            )?,
        })
    }

    /// Turns conditioning into a waveform.
    pub fn forward(
        &self,
        cond: Tensor<B, 3>,
        noise: &[f32],
        device: &B::Device,
    ) -> Vec<f32> {
        let frames = cond.dims()[2];
        let padded_frames = frames + VOCODER_PAD;
        let cond = cond.pad([(0, VOCODER_PAD)], PadMode::Constant(0.0));
        let start: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(
                noise.to_vec(),
                [1, NOISE_CHANNELS, padded_frames],
            ),
            device,
        );
        let mut hidden = conv1d(
            start.pad([(3, 3)], PadMode::Reflect),
            self.pre_weight.clone(),
            Some(self.pre_bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        for block in &self.blocks {
            hidden = block.forward(hidden, cond.clone());
        }
        hidden = leaky(hidden);
        hidden = activation::tanh(conv1d(
            hidden.pad([(3, 3)], PadMode::Reflect),
            self.post_weight.clone(),
            Some(self.post_bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        ));
        let hop: usize = STRIDES.iter().product();
        hidden
            .narrow(2, 0, frames * hop)
            .into_data()
            .into_vec::<f32>()
            .expect("the waveform as f32 values")
    }
}
