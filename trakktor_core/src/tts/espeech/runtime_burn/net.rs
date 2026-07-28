//! The DiT and the vocoder on burn.
//!
//! A port of the candle networks in [`runtime`](crate::tts::espeech::runtime)
//! with the same numerical semantics, written idiomatically for burn rather
//! than as a mirror. What differs, and why:
//!
//! - **Attention runs through burn's fused scaled-dot-product `attention`.**
//!   One utterance is one unpadded block over which every position attends to
//!   every other — the kernel's simplest and fastest path.
//! - **The context projection is folded at load** exactly as on candle: the
//!   input matrix is split into the columns that read the solver state and the
//!   columns that read the conditioning and the text, so the second half is
//!   projected once per utterance instead of once per step.
//! - **The text encoder runs once per utterance**, for each guidance branch,
//!   and its result is kept for the whole solve.
//! - The rotary rotation pairs neighbouring channels, which burn has no
//!   primitive for, so it is spelled out through a five-dimensional view.
//!
//! The networks are written for any element type, so the casts that hold the
//! normalization statistics in full precision are spelled out even though this
//! runtime's backends are all f32 today and skip them.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.
//!
//! Ported from F5-TTS and Vocos (both MIT).

use burn::{
    prelude::Int,
    tensor::{
        DType, Tensor, TensorData, activation,
        backend::Backend,
        module::{attention, conv1d, embedding, linear},
        ops::{AttentionModuleOptions, ConvOptions},
    },
};

use super::Weights;
use crate::tts::espeech::{
    config::{DIM_HEAD, DitConfig, ROPE_THETA, TIME_SCALE, VocoderConfig},
    error::EspeechError,
    runtime::model_err,
};

/// Positions the sinusoidal text embedding is tabulated for, as upstream.
const MAX_TEXT_POS: usize = 8192;

/// Epsilon of every normalization in both networks.
const NORM_EPS: f64 = 1e-6;

/// Ceiling on the vocoder's predicted magnitudes, as upstream.
const MAX_MAGNITUDE: f32 = 1e2;

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, EspeechError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Blocked out-of-place transpose of a row-major `[rows, cols]` matrix.
fn transpose_2d(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const TILE: usize = 64;
    let mut out = vec![0.0f32; values.len()];
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

/// Reads an `[out, in]` checkpoint matrix as burn's `[in, out]` layout.
fn matrix<B: Backend>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor<B, 2>, EspeechError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
        ));
    }
    Ok(Tensor::from_data(
        TensorData::new(
            transpose_2d(&values, out_dim, in_dim),
            [in_dim, out_dim],
        ),
        device,
    ))
}

/// A dense projection with a bias.
struct Dense<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> Dense<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, EspeechError> {
        Ok(Self {
            weight: matrix(
                weights,
                device,
                &format!("{prefix}.weight"),
                out_dim,
                in_dim,
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_dim],
            )?,
        })
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// A convolution over time, padded to keep the length.
struct Conv<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    padding: usize,
    groups: usize,
}

impl<B: Backend> Conv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self, EspeechError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_channels, in_channels / groups, kernel],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_channels],
            )?,
            padding: kernel / 2,
            groups,
        })
    }

    /// Convolves `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [self.padding], [1], self.groups),
        )
    }
}

/// Normalization over the last axis with no learned scale or shift.
fn layer_norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dtype = x.dtype();
    let x = x.cast(DType::F32);
    let mean = x.clone().mean_dim(D - 1);
    let centered = x - mean;
    let variance = centered.clone().powi_scalar(2).mean_dim(D - 1);
    (centered / (variance + NORM_EPS).sqrt()).cast(dtype)
}

/// Normalization over the last axis with a learned scale and shift.
struct AffineNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> AffineNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        size: usize,
    ) -> Result<Self, EspeechError> {
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

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, channels] = x.dims();
        layer_norm(x) * self.weight.clone().reshape([1, 1, channels]) +
            self.bias.clone().reshape([1, 1, channels])
    }
}

/// The GELU the reference asks for in the feed-forward: its `tanh`
/// approximation, which differs from the exact one by about a thousandth —
/// enough to matter against a golden trace.
fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    const COEFF: f64 = 0.044_715;
    let scale = (2.0 / std::f64::consts::PI).sqrt();
    let inner = (x.clone() + x.clone().powi_scalar(3) * COEFF) * scale;
    x * (inner.tanh() + 1.0) * 0.5
}

/// The Mish activation, `x · tanh(softplus(x))`, in the form that does not
/// overflow for large inputs.
fn mish<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let positive = x.clone().clamp_min(0.0);
    let softplus = positive + (-x.clone().abs()).exp().add_scalar(1.0).log();
    x * softplus.tanh()
}

/// Global response normalization, over time rather than over channels.
struct Grn<B: Backend> {
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
}

impl<B: Backend> Grn<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, EspeechError> {
        let read = |name: &str| -> Result<Tensor<B, 1>, EspeechError> {
            weight::<B, 3>(
                weights,
                device,
                &format!("{prefix}.{name}"),
                [1, 1, channels],
            )
            .map(|tensor| tensor.reshape([channels]))
        };
        Ok(Self {
            gamma: read("gamma")?,
            beta: read("beta")?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, channels] = x.dims();
        let energy = x.clone().powi_scalar(2).sum_dim(1).sqrt();
        let relative = energy.clone() / (energy.mean_dim(2) + 1e-6);
        let scaled = x.clone() * relative;
        scaled * self.gamma.clone().reshape([1, 1, channels]) +
            self.beta.clone().reshape([1, 1, channels]) +
            x
    }
}

/// A ConvNeXt block, in the two shapes the reference uses: the text encoder's
/// (with global response normalization) and the vocoder's (with a layer scale).
struct ConvNeXtBlock<B: Backend> {
    dwconv: Conv<B>,
    norm: AffineNorm<B>,
    pwconv1: Dense<B>,
    pwconv2: Dense<B>,
    grn: Option<Grn<B>>,
    gamma: Option<Tensor<B, 1>>,
}

impl<B: Backend> ConvNeXtBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
        inner: usize,
        with_grn: bool,
    ) -> Result<Self, EspeechError> {
        Ok(Self {
            dwconv: Conv::load(
                weights,
                device,
                &format!("{prefix}.dwconv"),
                dim,
                dim,
                7,
                dim,
            )?,
            norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm"),
                dim,
            )?,
            pwconv1: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv1"),
                inner,
                dim,
            )?,
            pwconv2: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv2"),
                dim,
                inner,
            )?,
            grn: with_grn
                .then(|| {
                    Grn::load(weights, device, &format!("{prefix}.grn"), inner)
                })
                .transpose()?,
            gamma: (!with_grn)
                .then(|| {
                    weight::<B, 1>(
                        weights,
                        device,
                        &format!("{prefix}.gamma"),
                        [dim],
                    )
                })
                .transpose()?,
        })
    }

    /// Applies the block to `x`, shaped `[batch, time, channels]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let hidden = self.dwconv.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        let hidden = self.norm.forward(hidden);
        let hidden = activation::gelu(self.pwconv1.forward(hidden));
        let hidden = match &self.grn {
            Some(grn) => grn.forward(hidden),
            None => hidden,
        };
        let hidden = self.pwconv2.forward(hidden);
        let hidden = match &self.gamma {
            Some(gamma) => {
                let [_, _, channels] = hidden.dims();
                hidden * gamma.clone().reshape([1, 1, channels])
            },
            None => hidden,
        };
        residual + hidden
    }
}

/// One DiT block: modulated self-attention, then a modulated feed-forward.
struct DitBlock<B: Backend> {
    modulation: Dense<B>,
    to_q: Dense<B>,
    to_k: Dense<B>,
    to_v: Dense<B>,
    to_out: Dense<B>,
    ff_in: Dense<B>,
    ff_out: Dense<B>,
    heads: usize,
    dim: usize,
}

impl<B: Backend> DitBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &DitConfig,
    ) -> Result<Self, EspeechError> {
        Ok(Self {
            modulation: Dense::load(
                weights,
                device,
                &format!("{prefix}.attn_norm.linear"),
                cfg.dim * 6,
                cfg.dim,
            )?,
            to_q: Dense::load(
                weights,
                device,
                &format!("{prefix}.attn.to_q"),
                cfg.dim,
                cfg.dim,
            )?,
            to_k: Dense::load(
                weights,
                device,
                &format!("{prefix}.attn.to_k"),
                cfg.dim,
                cfg.dim,
            )?,
            to_v: Dense::load(
                weights,
                device,
                &format!("{prefix}.attn.to_v"),
                cfg.dim,
                cfg.dim,
            )?,
            to_out: Dense::load(
                weights,
                device,
                &format!("{prefix}.attn.to_out.0"),
                cfg.dim,
                cfg.dim,
            )?,
            ff_in: Dense::load(
                weights,
                device,
                &format!("{prefix}.ff.ff.0.0"),
                cfg.ff_inner,
                cfg.dim,
            )?,
            ff_out: Dense::load(
                weights,
                device,
                &format!("{prefix}.ff.ff.2"),
                cfg.dim,
                cfg.ff_inner,
            )?,
            heads: cfg.heads,
            dim: cfg.dim,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        time: Tensor<B, 2>,
        rope: &Rope<B>,
    ) -> Tensor<B, 3> {
        let modulation = self.modulation.forward(activation::silu(time));
        let [branches, _] = modulation.dims();
        let part = |index: usize| -> Tensor<B, 3> {
            modulation
                .clone()
                .slice([0..branches, index * self.dim..(index + 1) * self.dim])
                .reshape([branches, 1, self.dim])
        };
        let (shift_attn, scale_attn, gate_attn) = (part(0), part(1), part(2));
        let (shift_ff, scale_ff, gate_ff) = (part(3), part(4), part(5));

        let normed = layer_norm(x.clone()) * (scale_attn + 1.0) + shift_attn;
        let attended = self.attend(normed, rope);
        let x = x + attended * gate_attn;

        let normed = layer_norm(x.clone()) * (scale_ff + 1.0) + shift_ff;
        let hidden = gelu_tanh(self.ff_in.forward(normed));
        x + self.ff_out.forward(hidden) * gate_ff
    }

    /// Bidirectional self-attention with rotary positions on queries and keys.
    /// There is no mask: a single utterance is one unpadded sequence.
    fn attend(&self, x: Tensor<B, 3>, rope: &Rope<B>) -> Tensor<B, 3> {
        let [batch, frames, dim] = x.dims();
        let split = |projected: Tensor<B, 3>| -> Tensor<B, 4> {
            projected
                .reshape([batch, frames, self.heads, DIM_HEAD])
                .swap_dims(1, 2)
        };
        let query = rope.apply(split(self.to_q.forward(x.clone())));
        let key = rope.apply(split(self.to_k.forward(x.clone())));
        let value = split(self.to_v.forward(x));

        let dtype = query.dtype();
        let context = attention(
            query.cast(DType::F32),
            key.cast(DType::F32),
            value.cast(DType::F32),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .cast(dtype);
        self.to_out
            .forward(context.swap_dims(1, 2).reshape([batch, frames, dim]))
    }
}

/// The rotary tables, and the rotation that pairs neighbouring channels.
struct Rope<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

impl<B: Backend> Rope<B> {
    fn new(device: &B::Device, positions: usize) -> Self {
        let half = DIM_HEAD / 2;
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for position in 0..positions {
            for index in 0..half {
                let inverse =
                    1.0 / ROPE_THETA.powf(2.0 * index as f64 / DIM_HEAD as f64);
                let angle = position as f64 * inverse;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            cos: Tensor::from_data(
                TensorData::new(cos, [positions, half]),
                device,
            ),
            sin: Tensor::from_data(
                TensorData::new(sin, [positions, half]),
                device,
            ),
        }
    }

    /// Rotates `x`, shaped `[batch, heads, frames, DIM_HEAD]`.
    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, frames, dim] = x.dims();
        let half = dim / 2;
        let cos = self
            .cos
            .clone()
            .slice([0..frames, 0..half])
            .reshape([1, 1, frames, half, 1]);
        let sin = self
            .sin
            .clone()
            .slice([0..frames, 0..half])
            .reshape([1, 1, frames, half, 1]);
        // The pairs are neighbouring channels, so the last axis splits in two
        // rather than the tensor splitting in halves.
        let paired = x.reshape([batch, heads, frames, half, 2]);
        let even = paired.clone().slice([
            0..batch,
            0..heads,
            0..frames,
            0..half,
            0..1,
        ]);
        let odd = paired.slice([0..batch, 0..heads, 0..frames, 0..half, 1..2]);
        let rotated_even =
            even.clone() * cos.clone() - odd.clone() * sin.clone();
        let rotated_odd = odd * cos + even * sin;
        Tensor::cat(vec![rotated_even, rotated_odd], 4)
            .reshape([batch, heads, frames, dim])
    }
}

/// The DiT on burn.
pub struct Dit<B: Backend> {
    cfg: DitConfig,
    time_mlp: (Dense<B>, Dense<B>),
    text_embed: Tensor<B, 2>,
    text_blocks: Vec<ConvNeXtBlock<B>>,
    text_pos: Tensor<B, 2>,
    proj_state: Tensor<B, 2>,
    proj_context: Tensor<B, 2>,
    proj_bias: Tensor<B, 1>,
    conv_pos: (Conv<B>, Conv<B>),
    blocks: Vec<DitBlock<B>>,
    norm_out: Dense<B>,
    proj_out: Dense<B>,
    rope: Rope<B>,
}

impl<B: Backend> Dit<B> {
    /// Loads the network with the geometry derived from the checkpoint.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: DitConfig,
    ) -> Result<Self, EspeechError> {
        let time_mlp = (
            Dense::load(
                weights,
                device,
                "transformer.time_embed.time_mlp.0",
                cfg.dim,
                cfg.time_dim,
            )?,
            Dense::load(
                weights,
                device,
                "transformer.time_embed.time_mlp.2",
                cfg.dim,
                cfg.dim,
            )?,
        );
        let text_embed = weight(
            weights,
            device,
            "transformer.text_embed.text_embed.weight",
            [cfg.vocab_size + 1, cfg.text_dim],
        )?;
        let text_blocks = (0..cfg.text_conv_layers)
            .map(|index| {
                ConvNeXtBlock::load(
                    weights,
                    device,
                    &format!("transformer.text_embed.text_blocks.{index}"),
                    cfg.text_dim,
                    cfg.text_ff_inner,
                    true,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The input matrix is split by what it reads, so the context half can
        // be projected once per utterance.
        let (values, dims) =
            weights.parts("transformer.input_embed.proj.weight")?;
        let width = cfg.mel_channels * 2 + cfg.text_dim;
        if dims != [cfg.dim, width] {
            return Err(model_err(
                "transformer.input_embed.proj.weight",
                format!("shape {dims:?}, expected {:?}", [cfg.dim, width]),
            ));
        }
        let transposed = transpose_2d(&values, cfg.dim, width);
        let state_columns = cfg.mel_channels;
        let mut state = Vec::with_capacity(state_columns * cfg.dim);
        let mut context = Vec::with_capacity((width - state_columns) * cfg.dim);
        for row in 0..width {
            let slice = &transposed[row * cfg.dim..(row + 1) * cfg.dim];
            if row < state_columns {
                state.extend_from_slice(slice);
            } else {
                context.extend_from_slice(slice);
            }
        }

        Ok(Self {
            cfg,
            time_mlp,
            text_embed,
            text_blocks,
            text_pos: sinusoidal_positions(device, cfg.text_dim, MAX_TEXT_POS),
            proj_state: Tensor::from_data(
                TensorData::new(state, [state_columns, cfg.dim]),
                device,
            ),
            proj_context: Tensor::from_data(
                TensorData::new(context, [width - state_columns, cfg.dim]),
                device,
            ),
            proj_bias: weight(
                weights,
                device,
                "transformer.input_embed.proj.bias",
                [cfg.dim],
            )?,
            conv_pos: (
                Conv::load(
                    weights,
                    device,
                    "transformer.input_embed.conv_pos_embed.conv1d.0",
                    cfg.dim,
                    cfg.dim,
                    cfg.conv_pos_kernel,
                    cfg.conv_pos_groups,
                )?,
                Conv::load(
                    weights,
                    device,
                    "transformer.input_embed.conv_pos_embed.conv1d.2",
                    cfg.dim,
                    cfg.dim,
                    cfg.conv_pos_kernel,
                    cfg.conv_pos_groups,
                )?,
            ),
            blocks: (0..cfg.depth)
                .map(|index| {
                    DitBlock::load(
                        weights,
                        device,
                        &format!("transformer.transformer_blocks.{index}"),
                        &cfg,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            norm_out: Dense::load(
                weights,
                device,
                "transformer.norm_out.linear",
                cfg.dim * 2,
                cfg.dim,
            )?,
            proj_out: Dense::load(
                weights,
                device,
                "transformer.proj_out",
                cfg.mel_channels,
                cfg.dim,
            )?,
            rope: Rope::new(device, MAX_TEXT_POS),
        })
    }

    /// The geometry this was loaded with.
    pub fn config(&self) -> &DitConfig { &self.cfg }

    /// Encodes the text for one guidance branch.
    pub fn encode_text(
        &self,
        ids: Tensor<B, 2, Int>,
        keep: Tensor<B, 3>,
        drop: bool,
    ) -> Tensor<B, 3> {
        let [_, frames] = ids.dims();
        let ids = if drop { ids.zeros_like() } else { ids };
        let mut hidden = embedding(self.text_embed.clone(), ids);
        hidden = hidden +
            self.text_pos
                .clone()
                .slice([0..frames, 0..self.cfg.text_dim])
                .reshape([1, frames, self.cfg.text_dim]);
        // Positions past the end of the text are held at zero, before the
        // blocks and after every one of them.
        hidden = hidden * keep.clone();
        for block in &self.text_blocks {
            hidden = block.forward(hidden) * keep.clone();
        }
        hidden
    }

    /// Projects everything that does not change between solver steps.
    pub fn project_context(
        &self,
        cond: Tensor<B, 3>,
        text: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let context = Tensor::cat(vec![cond, text], 2);
        linear(
            context,
            self.proj_context.clone(),
            Some(self.proj_bias.clone()),
        )
    }

    /// Predicts the flow at time `t` for state `x`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Tensor<B, 3>,
        t: f32,
    ) -> Tensor<B, 3> {
        let [branches, _, _] = context.dims();
        let time = self.embed_time(x.device(), t, branches);

        let mut hidden = context + linear(x, self.proj_state.clone(), None);
        hidden = self.position_embed(hidden.clone()) + hidden;

        for block in &self.blocks {
            hidden = block.forward(hidden, time.clone(), &self.rope);
        }

        // The final modulation states the scale before the shift — the blocks
        // state the shift first.
        let modulation = self.norm_out.forward(activation::silu(time));
        let scale = modulation
            .clone()
            .slice([0..branches, 0..self.cfg.dim])
            .reshape([branches, 1, self.cfg.dim]);
        let shift = modulation
            .slice([0..branches, self.cfg.dim..self.cfg.dim * 2])
            .reshape([branches, 1, self.cfg.dim]);
        let normed = layer_norm(hidden) * (scale + 1.0) + shift;
        self.proj_out.forward(normed)
    }

    /// The convolutional position embedding.
    fn position_embed(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = mish(self.conv_pos.0.forward(x.swap_dims(1, 2)));
        mish(self.conv_pos.1.forward(hidden)).swap_dims(1, 2)
    }

    /// Embeds the timestep and repeats it for every guidance branch.
    fn embed_time(
        &self,
        device: B::Device,
        t: f32,
        branches: usize,
    ) -> Tensor<B, 2> {
        let half = self.cfg.time_dim / 2;
        let step = (10_000f64).ln() / (half - 1) as f64;
        let mut values = Vec::with_capacity(self.cfg.time_dim);
        let angles: Vec<f64> = (0..half)
            .map(|index| {
                TIME_SCALE * f64::from(t) * (-step * index as f64).exp()
            })
            .collect();
        values.extend(angles.iter().map(|angle| angle.sin() as f32));
        values.extend(angles.iter().map(|angle| angle.cos() as f32));
        let embedded: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(values, [1, self.cfg.time_dim]),
            &device,
        );
        let hidden = self.time_mlp.0.forward(embedded);
        let hidden = self.time_mlp.1.forward(activation::silu(hidden));
        if branches == 1 {
            hidden
        } else {
            Tensor::cat(vec![hidden; branches], 0)
        }
    }
}

/// The vocoder on burn: the network only, with the inverse transform left to
/// the shared host-side code.
pub struct Vocoder<B: Backend> {
    cfg: VocoderConfig,
    embed: Conv<B>,
    norm: AffineNorm<B>,
    blocks: Vec<ConvNeXtBlock<B>>,
    final_norm: AffineNorm<B>,
    out: Dense<B>,
}

impl<B: Backend> Vocoder<B> {
    /// Loads the network with the geometry derived from its checkpoint.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: VocoderConfig,
    ) -> Result<Self, EspeechError> {
        Ok(Self {
            cfg,
            embed: Conv::load(
                weights,
                device,
                "backbone.embed",
                cfg.mel_channels,
                cfg.dim,
                7,
                1,
            )?,
            norm: AffineNorm::load(weights, device, "backbone.norm", cfg.dim)?,
            blocks: (0..cfg.layers)
                .map(|index| {
                    ConvNeXtBlock::load(
                        weights,
                        device,
                        &format!("backbone.convnext.{index}"),
                        cfg.dim,
                        cfg.ff_inner,
                        false,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            final_norm: AffineNorm::load(
                weights,
                device,
                "backbone.final_layer_norm",
                cfg.dim,
            )?,
            out: Dense::load(
                weights,
                device,
                "head.out",
                cfg.n_fft + 2,
                cfg.dim,
            )?,
        })
    }

    /// Runs the network over a mel shaped `[1, frames, mel]` and returns the
    /// magnitudes and phases the inverse transform needs.
    pub fn spectrum(
        &self,
        mel: Tensor<B, 3>,
    ) -> Result<(Vec<f32>, Vec<f32>), EspeechError> {
        let hidden = self.embed.forward(mel.swap_dims(1, 2)).swap_dims(1, 2);
        let mut hidden = self.norm.forward(hidden);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        let hidden = self.final_norm.forward(hidden);
        let out = self.out.forward(hidden);

        let [batch, frames, _] = out.dims();
        let bins = self.cfg.n_fft / 2 + 1;
        let magnitude = out
            .clone()
            .slice([0..batch, 0..frames, 0..bins])
            .exp()
            .clamp_max(MAX_MAGNITUDE);
        let phase = out.slice([0..batch, 0..frames, bins..bins * 2]);
        Ok((into_values(magnitude)?, into_values(phase)?))
    }
}

/// Reads a tensor back to the host as f32 values.
pub fn into_values<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, EspeechError> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| model_err("reading a tensor", format!("{e:?}")))
}

/// The sinusoidal positions the text embedding is offset by.
fn sinusoidal_positions<B: Backend>(
    device: &B::Device,
    dim: usize,
    positions: usize,
) -> Tensor<B, 2> {
    let half = dim / 2;
    let mut values = Vec::with_capacity(positions * dim);
    for position in 0..positions {
        let angles: Vec<f64> = (0..half)
            .map(|index| {
                let inverse =
                    1.0 / ROPE_THETA.powf(2.0 * index as f64 / dim as f64);
                position as f64 * inverse
            })
            .collect();
        values.extend(angles.iter().map(|angle| angle.cos() as f32));
        values.extend(angles.iter().map(|angle| angle.sin() as f32));
    }
    Tensor::from_data(TensorData::new(values, [positions, dim]), device)
}
