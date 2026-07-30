//! The Silero networks on burn.
//!
//! A port of the candle networks in
//! [`runtime::net`](crate::tts::silero::runtime::net) with the same numerical
//! semantics, written for burn rather than as a mirror. What is the same, and
//! deliberately so:
//!
//! - the **time-major, batch-free** layout, for the same reason (one utterance,
//!   thousands of frames, no axis worth carrying);
//! - the convolutions **as matrix multiplications** — an `im2col` and one large
//!   product for the full ones, `k` shifted multiply-adds for the depthwise
//!   one. On a GPU backend this matters more, not less: one big product is one
//!   kernel with a shape the autotuner will have seen before;
//! - the **dead half of the decoder's shortened stage is not run**, and its
//!   weights are not loaded.
//!
//! What differs: attention goes through burn's own fused scaled-dot-product
//! kernel rather than the tiled loop candle needs, because the kernel already
//! does the tiling and does it better than a loop over tensor ops can.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load; a panic past
//! loading is a bug, not a data condition.
//!
//! Ported from Silero TTS and Vocos (both MIT).

use burn::{
    prelude::Int,
    tensor::{
        Tensor, TensorData, activation,
        backend::Backend,
        module::{attention, linear},
        ops::AttentionModuleOptions,
    },
};

use super::Weights;
use crate::tts::silero::{
    config::{
        Config, FFN_KERNEL, HEADS, NORM_EPS, PITCH_KERNEL, SHORTEN_FACTOR,
        VOCODER_KERNEL,
    },
    error::SileroError,
};

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, SileroError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(SileroError::Checkpoint(format!(
            "{key}: shape {dims:?}, expected {shape:?}"
        )));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// A linear layer, stored the way burn's `linear` wants it: `[in, out]`.
struct Dense<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> Dense<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Self, SileroError> {
        let key = format!("{prefix}.weight");
        let (values, dims) = weights.parts(&key)?;
        if dims != [outputs, inputs] {
            return Err(SileroError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected [{outputs}, {inputs}]"
            )));
        }
        Ok(Self {
            weight: Tensor::from_data(
                TensorData::new(
                    transpose(&values, outputs, inputs),
                    [inputs, outputs],
                ),
                device,
            ),
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [outputs],
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        linear(xs, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// A same-padded convolution over time, folded into one matrix.
struct TimeConv<B: Backend> {
    /// `[kernel · in, out]`.
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    kernel: usize,
}

impl<B: Backend> TimeConv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        inputs: usize,
        outputs: usize,
        kernel: usize,
    ) -> Result<Self, SileroError> {
        let key = format!("{prefix}.weight");
        let (values, dims) = weights.parts(&key)?;
        if dims != [outputs, inputs, kernel] {
            return Err(SileroError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected [{outputs}, {inputs}, \
                 {kernel}]"
            )));
        }
        // `[out, in, k]` → `[k · in, out]`, the order the stacked windows
        // below arrive in.
        let mut folded = vec![0f32; values.len()];
        for out in 0..outputs {
            for input in 0..inputs {
                for tap in 0..kernel {
                    folded[(tap * inputs + input) * outputs + out] =
                        values[(out * inputs + input) * kernel + tap];
                }
            }
        }
        Ok(Self {
            weight: Tensor::from_data(
                TensorData::new(folded, [kernel * inputs, outputs]),
                device,
            ),
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [outputs],
            )?,
            kernel,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [time, _] = xs.dims();
        let stacked = if self.kernel == 1 {
            xs
        } else {
            let padded = pad_time(xs, self.kernel / 2);
            let [_, channels] = padded.dims();
            Tensor::cat(
                (0..self.kernel)
                    .map(|tap| {
                        padded.clone().slice([tap..tap + time, 0..channels])
                    })
                    .collect(),
                1,
            )
        };
        stacked.matmul(self.weight.clone()) + self.bias.clone().unsqueeze()
    }
}

/// A depthwise convolution over time.
struct DepthwiseConv<B: Backend> {
    /// `[kernel, channels]`.
    taps: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    kernel: usize,
}

impl<B: Backend> DepthwiseConv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
        kernel: usize,
    ) -> Result<Self, SileroError> {
        let key = format!("{prefix}.weight");
        let (values, dims) = weights.parts(&key)?;
        if dims != [channels, 1, kernel] {
            return Err(SileroError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected [{channels}, 1, {kernel}]"
            )));
        }
        Ok(Self {
            taps: Tensor::from_data(
                TensorData::new(
                    transpose(&values, channels, kernel),
                    [kernel, channels],
                ),
                device,
            ),
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [channels],
            )?,
            kernel,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [time, channels] = xs.dims();
        let padded = pad_time(xs, self.kernel / 2);
        let mut out: Option<Tensor<B, 2>> = None;
        for tap in 0..self.kernel {
            let shifted = padded.clone().slice([tap..tap + time, 0..channels]);
            let taps = self.taps.clone().slice([tap..tap + 1, 0..channels]);
            let term = shifted * taps;
            out = Some(match out {
                None => term,
                Some(sum) => sum + term,
            });
        }
        out.expect("a convolution has at least one tap") +
            self.bias.clone().unsqueeze()
    }
}

/// Pads a `[time, channels]` block with `pad` zero frames at each end.
fn pad_time<B: Backend>(xs: Tensor<B, 2>, pad: usize) -> Tensor<B, 2> {
    if pad == 0 {
        return xs;
    }
    let [_, channels] = xs.dims();
    let device = xs.device();
    let edge = Tensor::zeros([pad, channels], &device);
    Tensor::cat(vec![edge.clone(), xs, edge], 0)
}

/// Transposes a row-major `[rows, cols]` matrix, in blocks.
fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const TILE: usize = 64;
    let mut out = vec![0f32; values.len()];
    for row0 in (0..rows).step_by(TILE) {
        for col0 in (0..cols).step_by(TILE) {
            for row in row0..(row0 + TILE).min(rows) {
                for col in col0..(col0 + TILE).min(cols) {
                    out[col * rows + row] = values[row * cols + col];
                }
            }
        }
    }
    out
}

/// Normalization over the last axis, with a learned scale and shift.
struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> Norm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        size: usize,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [size],
            )?,
            bias: weight(weights, device, &format!("{prefix}.bias"), [size])?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = xs.clone().mean_dim(1);
        let centered = xs - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(1);
        centered / (variance + NORM_EPS).sqrt() *
            self.weight.clone().unsqueeze() +
            self.bias.clone().unsqueeze()
    }
}

/// The learned positional table.
struct Positional<B: Backend> {
    table: Tensor<B, 2>,
    scale: Tensor<B, 1>,
}

impl<B: Backend> Positional<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        positions: usize,
        dim: usize,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            table: weight(
                weights,
                device,
                &format!("{prefix}.pe"),
                [positions, dim],
            )?,
            scale: weight(weights, device, &format!("{prefix}.scale"), [1])?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [time, dim] = xs.dims();
        let table = self.table.clone().slice([0..time, 0..dim]);
        xs + table * self.scale.clone().unsqueeze()
    }
}

/// Multi-head self-attention over time.
struct Attention<B: Backend> {
    in_proj: Dense<B>,
    out_proj: Dense<B>,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
    ) -> Result<Self, SileroError> {
        let key = format!("{prefix}.in_proj_weight");
        let (values, dims) = weights.parts(&key)?;
        if dims != [3 * dim, dim] {
            return Err(SileroError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected [{}, {dim}]",
                3 * dim
            )));
        }
        Ok(Self {
            in_proj: Dense {
                weight: Tensor::from_data(
                    TensorData::new(
                        transpose(&values, 3 * dim, dim),
                        [dim, 3 * dim],
                    ),
                    device,
                ),
                bias: weight(
                    weights,
                    device,
                    &format!("{prefix}.in_proj_bias"),
                    [3 * dim],
                )?,
            },
            out_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.out_proj"),
                dim,
                dim,
            )?,
            head_dim: dim / HEADS,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [time, dim] = xs.dims();
        let projected = self.in_proj.forward(xs);
        let split = |offset: usize| -> Tensor<B, 4> {
            projected
                .clone()
                .slice([0..time, offset * dim..(offset + 1) * dim])
                .reshape([1, time, HEADS, self.head_dim])
                .swap_dims(1, 2)
        };
        let context = attention(
            split(0),
            split(1),
            split(2),
            None,
            None,
            AttentionModuleOptions::default(),
        );
        self.out_proj
            .forward(context.swap_dims(1, 2).reshape([time, dim]))
    }
}

/// One block of the acoustic model.
struct FftBlock<B: Backend> {
    attention: Attention<B>,
    conv: TimeConv<B>,
    project: Dense<B>,
    norm1: Norm<B>,
    norm2: Norm<B>,
}

impl<B: Backend> FftBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
        inner: usize,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            attention: Attention::load(
                weights,
                device,
                &format!("{prefix}.self_attn"),
                dim,
            )?,
            conv: TimeConv::load(
                weights,
                device,
                &format!("{prefix}.conv1"),
                dim,
                inner,
                FFN_KERNEL,
            )?,
            project: Dense::load(
                weights,
                device,
                &format!("{prefix}.conv2"),
                inner,
                dim,
            )?,
            norm1: Norm::load(
                weights,
                device,
                &format!("{prefix}.norm1"),
                dim,
            )?,
            norm2: Norm::load(
                weights,
                device,
                &format!("{prefix}.norm2"),
                dim,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let attended =
            self.norm1.forward(xs.clone() + self.attention.forward(xs));
        let hidden = activation::relu(self.conv.forward(attended.clone()));
        self.norm2.forward(attended + self.project.forward(hidden))
    }
}

/// A stack of blocks over a positional table.
struct ForwardTransformer<B: Backend> {
    positional: Positional<B>,
    layers: Vec<FftBlock<B>>,
    norm: Norm<B>,
}

impl<B: Backend> ForwardTransformer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        positions: usize,
        dim: usize,
        inner: usize,
        layers: usize,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            positional: Positional::load(
                weights,
                device,
                &format!("{prefix}.pos_encoder"),
                positions,
                dim,
            )?,
            layers: (0..layers)
                .map(|index| {
                    FftBlock::load(
                        weights,
                        device,
                        &format!("{prefix}.layers.{index}"),
                        dim,
                        inner,
                    )
                })
                .collect::<Result<_, _>>()?,
            norm: Norm::load(weights, device, &format!("{prefix}.norm"), dim)?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut hidden = self.positional.forward(xs);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.norm.forward(hidden)
    }
}

/// A head that predicts one number per symbol.
pub struct SeriesPredictor<B: Backend> {
    embedding: Tensor<B, 2>,
    speakers: Tensor<B, 2>,
    types: Option<Tensor<B, 2>>,
    transformer: ForwardTransformer<B>,
    out: Dense<B>,
}

impl<B: Backend> SeriesPredictor<B> {
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        config: &Config,
        with_types: bool,
    ) -> Result<Self, SileroError> {
        let dim = config.predictor_dim;
        Ok(Self {
            embedding: weight(
                weights,
                device,
                &format!("{prefix}.embedding.weight"),
                [config.symbols, dim],
            )?,
            speakers: weight(
                weights,
                device,
                &format!("{prefix}.speaker_embedding.weight"),
                [config.speaker_slots, dim],
            )?,
            types: with_types
                .then(|| {
                    weight(
                        weights,
                        device,
                        &format!("{prefix}.type_embedding.weight"),
                        [config.utterance_types, dim],
                    )
                })
                .transpose()?,
            transformer: ForwardTransformer::load(
                weights,
                device,
                &format!("{prefix}.transformer"),
                config.positions,
                dim,
                config.predictor_ff_inner,
                config.predictor_layers,
            )?,
            out: Dense::load(
                weights,
                device,
                &format!("{prefix}.lin"),
                dim,
                1,
            )?,
        })
    }

    /// Predicts one value per symbol.
    pub fn forward(
        &self,
        ids: Tensor<B, 1, Int>,
        speaker: usize,
        types: Option<Tensor<B, 1, Int>>,
    ) -> Tensor<B, 1> {
        let mut hidden = self.embedding.clone().select(0, ids);
        hidden = hidden + speaker_row(&self.speakers, speaker);
        if let (Some(table), Some(types)) = (&self.types, types) {
            hidden = hidden + table.clone().select(0, types);
        }
        let hidden = self.transformer.forward(hidden);
        let [time, _] = hidden.dims();
        self.out.forward(hidden).reshape([time])
    }
}

/// The decoder: full resolution, a third of it, full again.
struct HourGlass<B: Backend> {
    positional: Positional<B>,
    pre: FftBlock<B>,
    shortened: FftBlock<B>,
    post: FftBlock<B>,
    upsample: Dense<B>,
    norm: Norm<B>,
}

impl<B: Backend> HourGlass<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        config: &Config,
    ) -> Result<Self, SileroError> {
        let dim = config.dim;
        let inner = config.ff_inner;
        Ok(Self {
            positional: Positional::load(
                weights,
                device,
                &format!("{prefix}.pos_encoder"),
                config.positions,
                dim,
            )?,
            pre: FftBlock::load(
                weights,
                device,
                &format!("{prefix}.pre_vanilla_layers.0"),
                dim,
                inner,
            )?,
            // Index one, not zero: the published forward applies both to the
            // same input and keeps this one.
            shortened: FftBlock::load(
                weights,
                device,
                &format!("{prefix}.shorten_layers.1"),
                dim,
                inner,
            )?,
            post: FftBlock::load(
                weights,
                device,
                &format!("{prefix}.post_vanilla_layers.0"),
                dim,
                inner,
            )?,
            upsample: Dense::load(
                weights,
                device,
                &format!("{prefix}.upsample.proj"),
                dim,
                dim * SHORTEN_FACTOR,
            )?,
            norm: Norm::load(weights, device, &format!("{prefix}.norm"), dim)?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let [time, dim] = xs.dims();
        let hidden = self.pre.forward(self.positional.forward(xs));
        let padded_len = time.div_ceil(SHORTEN_FACTOR) * SHORTEN_FACTOR;
        let residual = if padded_len == time {
            hidden
        } else {
            let device = hidden.device();
            let tail = Tensor::zeros([padded_len - time, dim], &device);
            Tensor::cat(vec![hidden, tail], 0)
        };
        let pooled = residual
            .clone()
            .reshape([padded_len / SHORTEN_FACTOR, SHORTEN_FACTOR, dim])
            .mean_dim(1)
            .reshape([padded_len / SHORTEN_FACTOR, dim]);
        let shortened = self.shortened.forward(pooled);
        let upsampled =
            self.upsample.forward(shortened).reshape([padded_len, dim]);
        let joined = (upsampled + residual).slice([0..time, 0..dim]);
        self.norm.forward(self.post.forward(joined))
    }
}

/// The acoustic model.
pub struct Acoustic<B: Backend> {
    embedding: Tensor<B, 2>,
    speakers: Tensor<B, 2>,
    encoder: ForwardTransformer<B>,
    pitch_proj: TimeConv<B>,
    decoder: HourGlass<B>,
    out: Dense<B>,
}

impl<B: Backend> Acoustic<B> {
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        config: &Config,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            embedding: weight(
                weights,
                device,
                "tacotron.embedding.weight",
                [config.symbols, config.dim],
            )?,
            speakers: weight(
                weights,
                device,
                "tacotron.speaker_embedding.weight",
                [config.speaker_slots, config.dim],
            )?,
            encoder: ForwardTransformer::load(
                weights,
                device,
                "tacotron.encoder",
                config.positions,
                config.dim,
                config.ff_inner,
                config.encoder_layers,
            )?,
            pitch_proj: TimeConv::load(
                weights,
                device,
                "tacotron.pitch_proj",
                1,
                config.dim,
                PITCH_KERNEL,
            )?,
            decoder: HourGlass::load(
                weights,
                device,
                "tacotron.decoder",
                config,
            )?,
            out: Dense::load(
                weights,
                device,
                "tacotron.lin",
                config.dim,
                config.mel_channels,
            )?,
        })
    }

    /// Encodes the symbols, expands them to frames, and decodes a mel.
    pub fn forward(
        &self,
        ids: Tensor<B, 1, Int>,
        speaker: usize,
        pitch: Tensor<B, 1>,
        expansion: Tensor<B, 1, Int>,
    ) -> Tensor<B, 2> {
        let [time] = pitch.dims();
        let embedded = self.embedding.clone().select(0, ids);
        let encoded = self.encoder.forward(embedded) +
            speaker_row(&self.speakers, speaker);
        let projected = self.pitch_proj.forward(pitch.reshape([time, 1]));
        let expanded = (encoded + projected).select(0, expansion);
        self.out.forward(self.decoder.forward(expanded))
    }
}

/// One ConvNeXt block of the vocoder.
struct ConvNeXt<B: Backend> {
    dwconv: DepthwiseConv<B>,
    norm: Norm<B>,
    pwconv1: Dense<B>,
    pwconv2: Dense<B>,
    gamma: Tensor<B, 1>,
}

impl<B: Backend> ConvNeXt<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
        inner: usize,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            dwconv: DepthwiseConv::load(
                weights,
                device,
                &format!("{prefix}.dwconv"),
                dim,
                VOCODER_KERNEL,
            )?,
            norm: Norm::load(weights, device, &format!("{prefix}.norm"), dim)?,
            pwconv1: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv1"),
                dim,
                inner,
            )?,
            pwconv2: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv2"),
                inner,
                dim,
            )?,
            gamma: weight(weights, device, &format!("{prefix}.gamma"), [dim])?,
        })
    }

    fn forward(&self, xs: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = self.norm.forward(self.dwconv.forward(xs.clone()));
        let hidden = activation::gelu(self.pwconv1.forward(hidden));
        let hidden = self.pwconv2.forward(hidden);
        xs + hidden * self.gamma.clone().unsqueeze()
    }
}

/// The vocoder.
pub struct Vocoder<B: Backend> {
    embed: TimeConv<B>,
    norm: Norm<B>,
    blocks: Vec<ConvNeXt<B>>,
    final_norm: Norm<B>,
    out: Dense<B>,
}

impl<B: Backend> Vocoder<B> {
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        config: &Config,
    ) -> Result<Self, SileroError> {
        Ok(Self {
            embed: TimeConv::load(
                weights,
                device,
                "vocoder.backbone.embed",
                config.mel_channels,
                config.vocoder_dim,
                VOCODER_KERNEL,
            )?,
            norm: Norm::load(
                weights,
                device,
                "vocoder.backbone.norm",
                config.vocoder_dim,
            )?,
            blocks: (0..config.vocoder_layers)
                .map(|index| {
                    ConvNeXt::load(
                        weights,
                        device,
                        &format!("vocoder.backbone.convnext.{index}"),
                        config.vocoder_dim,
                        config.vocoder_ff_inner,
                    )
                })
                .collect::<Result<_, _>>()?,
            final_norm: Norm::load(
                weights,
                device,
                "vocoder.backbone.final_layer_norm",
                config.vocoder_dim,
            )?,
            out: Dense::load(
                weights,
                device,
                "vocoder.head.out",
                config.vocoder_dim,
                config.n_fft + 2,
            )?,
        })
    }

    /// Runs the backbone and the head over `[frames, mel]`.
    pub fn forward(&self, mel: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut hidden = self.norm.forward(self.embed.forward(mel));
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        self.out.forward(self.final_norm.forward(hidden))
    }
}

/// One speaker's embedding row, shaped to broadcast over time.
fn speaker_row<B: Backend>(
    table: &Tensor<B, 2>,
    speaker: usize,
) -> Tensor<B, 2> {
    let [_, dim] = table.dims();
    table.clone().slice([speaker..speaker + 1, 0..dim])
}
