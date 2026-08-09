//! The networks on burn.
//!
//! A port of the candle network in [`runtime::net`](super::super::runtime::net)
//! with the same numerical semantics, written for burn rather than mirrored
//! operation by operation. Two differences are worth naming:
//!
//! - the encoder's attention runs through burn's fused scaled-dot-product
//!   kernel, which takes the gated position bias as its additive mask;
//! - burn has no group normalization, so the backbone's is written out — a
//!   reshape into groups, statistics over the group, and the per-channel affine
//!   put back.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.

use burn::tensor::{
    DType, Int, Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, conv1d, linear},
    ops::{AttentionModuleOptions, ConvOptions},
};

use super::Weights;
use crate::enhance::unipase::{
    config::{
        ATTENTION_HEADS, BACKBONE_DIM, BACKBONE_FF, BACKBONE_LAYERS,
        BACKBONE_NORM_EPS, CONV_LAYERS, CONV_POS, CONV_POS_GROUPS, ENCODER_DIM,
        ENCODER_LAYERS, ENCODER_NORM_EPS, FFN_DIM, GREP_DIM, POS_NET_ATTN,
        POS_NET_GROUPS, POS_NET_RES, TAP_ACOUSTIC, TAP_NORM_EPS, TAP_PHONETIC,
        bins, head_dim,
    },
    error::UnipaseError,
    runtime::net::relative_buckets,
};

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, UnipaseError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err_str(
            key,
            &format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// A checkpoint error naming the tensor it is about.
fn model_err_str(key: &str, what: &str) -> UnipaseError {
    UnipaseError::Checkpoint(format!("{key}: {what}"))
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
) -> Result<Tensor<B, 2>, UnipaseError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err_str(
            key,
            &format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
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
    ) -> Result<Self, UnipaseError> {
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

/// A convolution over time.
struct Conv<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
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
        stride: usize,
        padding: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_channels, in_channels / groups, kernel],
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
            stride,
            padding,
            groups,
        })
    }

    /// Convolves `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([self.stride], [self.padding], [1], self.groups),
        )
    }
}

/// Normalization over the last axis with a learned scale and shift.
struct AffineNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
}

impl<B: Backend> AffineNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        size: usize,
        eps: f64,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [size],
            )?,
            bias: weight(weights, device, &format!("{prefix}.bias"), [size])?,
            eps,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, channels] = x.dims();
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(2);
        let normed = centered / (variance + self.eps).sqrt();
        (normed * self.weight.clone().reshape([1, 1, channels]) +
            self.bias.clone().reshape([1, 1, channels]))
        .cast(dtype)
    }
}

/// Group normalization over `[batch, channels, time]`.
struct GroupNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    groups: usize,
    eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, UnipaseError> {
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
            groups: POS_NET_GROUPS,
            eps: BACKBONE_NORM_EPS,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let per_group = channels / self.groups * time;
        let grouped = x.reshape([batch, self.groups, per_group]);
        let mean = grouped.clone().mean_dim(2);
        let centered = grouped - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(2);
        let normed = (centered / (variance + self.eps).sqrt())
            .reshape([batch, channels, time]);
        normed * self.weight.clone().reshape([1, channels, 1]) +
            self.bias.clone().reshape([1, channels, 1])
    }
}

/// The convolutional feature extractor.
struct ConvExtractor<B: Backend> {
    blocks: Vec<(Conv<B>, AffineNorm<B>)>,
}

impl<B: Backend> ConvExtractor<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, UnipaseError> {
        let mut blocks = Vec::with_capacity(CONV_LAYERS.len());
        let mut in_channels = 1;
        for (index, &(channels, kernel, stride)) in
            CONV_LAYERS.iter().enumerate()
        {
            blocks.push((
                Conv::load(
                    weights,
                    device,
                    &format!("{prefix}.{index}.0"),
                    in_channels,
                    channels,
                    kernel,
                    stride,
                    0,
                    1,
                    false,
                )?,
                AffineNorm::load(
                    weights,
                    device,
                    &format!("{prefix}.{index}.2.1"),
                    channels,
                    ENCODER_NORM_EPS,
                )?,
            ));
            in_channels = channels;
        }
        Ok(Self { blocks })
    }

    /// Takes `[1, samples]` to `[1, channels, frames]`.
    fn forward(&self, samples: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, len] = samples.dims();
        let mut hidden = samples.reshape([batch, 1, len]);
        for (conv, norm) in &self.blocks {
            hidden = conv.forward(hidden);
            hidden = norm.forward(hidden.swap_dims(1, 2)).swap_dims(1, 2);
            hidden = activation::gelu(hidden);
        }
        hidden
    }
}

/// Self-attention with the gated relative position bias.
struct Attention<B: Backend> {
    q_proj: Dense<B>,
    k_proj: Dense<B>,
    v_proj: Dense<B>,
    out_proj: Dense<B>,
    grep_linear: Dense<B>,
    grep_a: Tensor<B, 4>,
    rel_bias: Option<Tensor<B, 2>>,
}

impl<B: Backend> Attention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        first: bool,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            q_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.q_proj"),
                ENCODER_DIM,
                ENCODER_DIM,
            )?,
            k_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.k_proj"),
                ENCODER_DIM,
                ENCODER_DIM,
            )?,
            v_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.v_proj"),
                ENCODER_DIM,
                ENCODER_DIM,
            )?,
            out_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.out_proj"),
                ENCODER_DIM,
                ENCODER_DIM,
            )?,
            grep_linear: Dense::load(
                weights,
                device,
                &format!("{prefix}.grep_linear"),
                GREP_DIM,
                head_dim(),
            )?,
            grep_a: weight(
                weights,
                device,
                &format!("{prefix}.grep_a"),
                [1, ATTENTION_HEADS, 1, 1],
            )?,
            rel_bias: if first {
                Some(weight(
                    weights,
                    device,
                    &format!("{prefix}.relative_attention_bias.weight"),
                    [
                        crate::enhance::unipase::config::NUM_BUCKETS,
                        ATTENTION_HEADS,
                    ],
                )?)
            } else {
                None
            },
        })
    }

    /// The bias table this layer contributes, `[1, heads, frames, frames]`.
    fn position_bias(
        &self,
        frames: usize,
        device: &B::Device,
    ) -> Option<Tensor<B, 4>> {
        let table = self.rel_bias.as_ref()?;
        let buckets: Vec<i32> = relative_buckets(frames)
            .into_iter()
            .map(|bucket| bucket as i32)
            .collect();
        let index = Tensor::<B, 1, Int>::from_data(
            TensorData::new(buckets, [frames * frames]),
            device,
        );
        Some(
            table
                .clone()
                .select(0, index)
                .reshape([1, frames, frames, ATTENTION_HEADS])
                .permute([0, 3, 1, 2]),
        )
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, frames, _] = hidden.dims();
        let heads = ATTENTION_HEADS;
        let dim = head_dim();
        let split = |x: Tensor<B, 3>| -> Tensor<B, 4> {
            x.reshape([batch, frames, heads, dim]).swap_dims(1, 2)
        };
        let query = split(self.q_proj.forward(hidden.clone()));
        let key = split(self.k_proj.forward(hidden.clone()));
        let value = split(self.v_proj.forward(hidden.clone()));

        // The gate reads the layer's input split into heads, not the projected
        // query.
        let gate = self.gate(split(hidden), batch, frames, heads);
        let mask = bias * gate;

        let context = attention(
            query,
            key,
            value,
            None,
            Some(mask),
            AttentionModuleOptions::default(),
        );
        self.out_proj.forward(context.swap_dims(1, 2).reshape([
            batch,
            frames,
            heads * dim,
        ]))
    }

    /// The per-query scale on the shared bias, `[batch, heads, frames, 1]`.
    fn gate(
        &self,
        hidden: Tensor<B, 4>,
        batch: usize,
        frames: usize,
        heads: usize,
    ) -> Tensor<B, 4> {
        let raw = self.grep_linear.forward(hidden);
        let halves = raw
            .reshape([batch, heads, frames, 2, GREP_DIM / 2])
            .sum_dim(4)
            .reshape([batch, heads, frames, 2]);
        let squashed = activation::sigmoid(halves);
        let gate_a = squashed.clone().narrow(3, 0, 1);
        let gate_b = squashed.narrow(3, 1, 1);
        gate_a * (gate_b * self.grep_a.clone() - 1.0) + 2.0
    }
}

/// One transformer layer, pre-norm.
struct EncoderLayer<B: Backend> {
    attention: Attention<B>,
    attn_norm: AffineNorm<B>,
    fc1: Dense<B>,
    fc2: Dense<B>,
    final_norm: AffineNorm<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        first: bool,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            attention: Attention::load(
                weights,
                device,
                &format!("{prefix}.self_attn"),
                first,
            )?,
            attn_norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.self_attn_layer_norm"),
                ENCODER_DIM,
                ENCODER_NORM_EPS,
            )?,
            fc1: Dense::load(
                weights,
                device,
                &format!("{prefix}.fc1"),
                FFN_DIM,
                ENCODER_DIM,
            )?,
            fc2: Dense::load(
                weights,
                device,
                &format!("{prefix}.fc2"),
                ENCODER_DIM,
                FFN_DIM,
            )?,
            final_norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.final_layer_norm"),
                ENCODER_DIM,
                ENCODER_NORM_EPS,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let attended = self
            .attention
            .forward(self.attn_norm.forward(hidden.clone()), bias);
        let hidden = hidden + attended;
        let normed = self.final_norm.forward(hidden.clone());
        let ffn = self.fc2.forward(activation::gelu(self.fc1.forward(normed)));
        hidden + ffn
    }
}

/// The encoder.
pub struct Encoder<B: Backend> {
    extractor: ConvExtractor<B>,
    post_norm: AffineNorm<B>,
    proj: Dense<B>,
    mask_emb: Tensor<B, 1>,
    pos_conv: Conv<B>,
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Encoder<B> {
    /// Loads the encoder from the converted checkpoint.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, UnipaseError> {
        let (last, _, _) = CONV_LAYERS[CONV_LAYERS.len() - 1];
        Ok(Self {
            extractor: ConvExtractor::load(
                weights,
                device,
                "wavlm.feature_extractor.conv_layers",
            )?,
            post_norm: AffineNorm::load(
                weights,
                device,
                "wavlm.layer_norm",
                last,
                ENCODER_NORM_EPS,
            )?,
            proj: Dense::load(
                weights,
                device,
                "wavlm.post_extract_proj",
                ENCODER_DIM,
                last,
            )?,
            mask_emb: weight(weights, device, "wavlm.mask_emb", [ENCODER_DIM])?,
            pos_conv: Conv::load(
                weights,
                device,
                "wavlm.encoder.pos_conv.0",
                ENCODER_DIM,
                ENCODER_DIM,
                CONV_POS,
                1,
                CONV_POS / 2,
                CONV_POS_GROUPS,
                true,
            )?,
            layers: (0..ENCODER_LAYERS)
                .map(|index| {
                    EncoderLayer::load(
                        weights,
                        device,
                        &format!("wavlm.encoder.layers.{index}"),
                        index == 0,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// The two tapped representations of one aligned window.
    pub fn features(
        &self,
        samples: Tensor<B, 2>,
        lost: &[bool],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let extracted = self.extractor.forward(samples).swap_dims(1, 2);
        let normed = self.post_norm.forward(extracted);
        let projected = self.proj.forward(normed);
        let masked = self.apply_mask(projected, lost, device);
        let mut hidden = self.positional(masked);

        let frames = hidden.dims()[1];
        let bias = self.layers[0]
            .attention
            .position_bias(frames, device)
            .expect("the first layer carries the bias table");

        let mut acoustic = None;
        let mut phonetic = None;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, bias.clone());
            match index + 1 {
                TAP_ACOUSTIC => acoustic = Some(hidden.clone()),
                TAP_PHONETIC => phonetic = Some(hidden.clone()),
                _ => {},
            }
        }
        (
            normalize_tap(acoustic.expect("the acoustic tap")),
            normalize_tap(phonetic.expect("the deep tap")),
        )
    }

    fn apply_mask(
        &self,
        hidden: Tensor<B, 3>,
        lost: &[bool],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        if !lost.iter().any(|&flag| flag) {
            return hidden;
        }
        let frames = hidden.dims()[1];
        let keep: Vec<f32> = (0..frames)
            .map(|frame| f32::from(!lost.get(frame).copied().unwrap_or(false)))
            .collect();
        let keep = Tensor::<B, 3>::from_data(
            TensorData::new(keep, [1, frames, 1]),
            device,
        );
        let fill = keep.clone().neg() + 1.0;
        let embedding = self.mask_emb.clone().reshape([1, 1, ENCODER_DIM]);
        hidden * keep + fill * embedding
    }

    fn positional(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let frames = hidden.dims()[1];
        let conv = self.pos_conv.forward(hidden.clone().swap_dims(1, 2));
        let conv = activation::gelu(conv.narrow(2, 0, frames));
        hidden + conv.swap_dims(1, 2)
    }
}

/// The tap normalization: over time and channels together, no affine.
fn normalize_tap<B: Backend>(hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, frames, dim] = hidden.dims();
    let dtype = hidden.dtype();
    let flat = hidden.reshape([batch, frames * dim]).cast(DType::F32);
    let mean = flat.clone().mean_dim(1);
    let centered = flat - mean;
    let variance = centered.clone().powi_scalar(2).mean_dim(1);
    (centered / (variance + TAP_NORM_EPS).sqrt())
        .reshape([batch, frames, dim])
        .cast(dtype)
}

/// A ConvNeXt block of the Vocos backbone.
struct ConvNeXtBlock<B: Backend> {
    dwconv: Conv<B>,
    norm: AffineNorm<B>,
    pwconv1: Dense<B>,
    pwconv2: Dense<B>,
    gamma: Tensor<B, 1>,
}

impl<B: Backend> ConvNeXtBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            dwconv: Conv::load(
                weights,
                device,
                &format!("{prefix}.dwconv"),
                BACKBONE_DIM,
                BACKBONE_DIM,
                7,
                1,
                3,
                BACKBONE_DIM,
                true,
            )?,
            norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm"),
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
            )?,
            pwconv1: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv1"),
                BACKBONE_FF,
                BACKBONE_DIM,
            )?,
            pwconv2: Dense::load(
                weights,
                device,
                &format!("{prefix}.pwconv2"),
                BACKBONE_DIM,
                BACKBONE_FF,
            )?,
            gamma: weight(
                weights,
                device,
                &format!("{prefix}.gamma"),
                [BACKBONE_DIM],
            )?,
        })
    }

    /// `hidden` is `[1, channels, frames]`.
    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let mixed = self.dwconv.forward(hidden);
        let mixed = self.norm.forward(mixed.swap_dims(1, 2));
        let mixed = activation::gelu(self.pwconv1.forward(mixed));
        let mixed = self.pwconv2.forward(mixed);
        let mixed = mixed * self.gamma.clone().reshape([1, 1, BACKBONE_DIM]);
        residual + mixed.swap_dims(1, 2)
    }
}

/// A residual block of the positional network.
struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv<B>,
    norm2: GroupNorm<B>,
    conv2: Conv<B>,
}

impl<B: Backend> ResBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, UnipaseError> {
        let conv = |name: &str| {
            Conv::load(
                weights,
                device,
                &format!("{prefix}.{name}"),
                BACKBONE_DIM,
                BACKBONE_DIM,
                3,
                1,
                1,
                1,
                true,
            )
        };
        Ok(Self {
            norm1: GroupNorm::load(
                weights,
                device,
                &format!("{prefix}.norm1"),
                BACKBONE_DIM,
            )?,
            conv1: conv("conv1")?,
            norm2: GroupNorm::load(
                weights,
                device,
                &format!("{prefix}.norm2"),
                BACKBONE_DIM,
            )?,
            conv2: conv("conv2")?,
        })
    }

    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let branch = swish(self.norm1.forward(hidden.clone()));
        let branch = self.conv1.forward(branch);
        let branch = swish(self.norm2.forward(branch));
        hidden + self.conv2.forward(branch)
    }
}

/// The reference's `nonlinearity`: `x · σ(x)`.
fn swish<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.clone() * activation::sigmoid(x)
}

/// The single-headed attention in the middle of the positional network.
struct AttnBlock<B: Backend> {
    norm: GroupNorm<B>,
    query: Conv<B>,
    key: Conv<B>,
    value: Conv<B>,
    out: Conv<B>,
}

impl<B: Backend> AttnBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, UnipaseError> {
        let point = |name: &str| {
            Conv::load(
                weights,
                device,
                &format!("{prefix}.{name}"),
                BACKBONE_DIM,
                BACKBONE_DIM,
                1,
                1,
                0,
                1,
                true,
            )
        };
        Ok(Self {
            norm: GroupNorm::load(
                weights,
                device,
                &format!("{prefix}.norm"),
                BACKBONE_DIM,
            )?,
            query: point("q")?,
            key: point("k")?,
            value: point("v")?,
            out: point("proj_out")?,
        })
    }

    fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.norm.forward(hidden.clone());
        let query = self.query.forward(normed.clone());
        let key = self.key.forward(normed.clone());
        let value = self.value.forward(normed);
        let scale = 1.0 / (BACKBONE_DIM as f64).sqrt();
        let scores = query.swap_dims(1, 2).matmul(key) * scale;
        let weights = activation::softmax(scores, 2);
        let context = value.matmul(weights.swap_dims(1, 2));
        hidden + self.out.forward(context)
    }
}

/// One of the two Vocos backbones.
pub struct Backbone<B: Backend> {
    embed: Conv<B>,
    res_in: Vec<ResBlock<B>>,
    attn: Vec<AttnBlock<B>>,
    res_out: Vec<ResBlock<B>>,
    pos_norm: GroupNorm<B>,
    norm: AffineNorm<B>,
    blocks: Vec<ConvNeXtBlock<B>>,
    final_norm: AffineNorm<B>,
}

impl<B: Backend> Backbone<B> {
    /// Loads a backbone from the checkpoint's `decoder.` subtree.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, UnipaseError> {
        let half = POS_NET_RES / 2;
        let res = |from: usize, count: usize| {
            (0..count)
                .map(|index| {
                    ResBlock::load(
                        weights,
                        device,
                        &format!("{prefix}.pos_net.{}", from + index),
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            embed: Conv::load(
                weights,
                device,
                &format!("{prefix}.embed"),
                BACKBONE_DIM,
                BACKBONE_DIM,
                7,
                1,
                3,
                1,
                true,
            )?,
            res_in: res(0, half)?,
            attn: (0..POS_NET_ATTN)
                .map(|index| {
                    AttnBlock::load(
                        weights,
                        device,
                        &format!("{prefix}.pos_net.{}", half + index),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            res_out: res(half + POS_NET_ATTN, half)?,
            pos_norm: GroupNorm::load(
                weights,
                device,
                &format!("{prefix}.pos_net.{}", POS_NET_RES + POS_NET_ATTN),
                BACKBONE_DIM,
            )?,
            norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm"),
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
            )?,
            blocks: (0..BACKBONE_LAYERS)
                .map(|index| {
                    ConvNeXtBlock::load(
                        weights,
                        device,
                        &format!("{prefix}.convnext.{index}"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            final_norm: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.final_layer_norm"),
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
            )?,
        })
    }

    /// `hidden` is `[1, channels, frames]`, and so is the result.
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = self.embed.forward(hidden);
        for block in &self.res_in {
            hidden = block.forward(hidden);
        }
        for block in &self.attn {
            hidden = block.forward(hidden);
        }
        for block in &self.res_out {
            hidden = block.forward(hidden);
        }
        hidden = self.pos_norm.forward(hidden);
        hidden = self.norm.forward(hidden.swap_dims(1, 2)).swap_dims(1, 2);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        self.final_norm
            .forward(hidden.swap_dims(1, 2))
            .swap_dims(1, 2)
    }
}

/// The adapter.
pub struct Adapter<B: Backend> {
    proj: Dense<B>,
    backbone: Backbone<B>,
    head: Dense<B>,
}

impl<B: Backend> Adapter<B> {
    /// Loads the adapter.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            proj: Dense::load(
                weights,
                device,
                "adapter.proj",
                BACKBONE_DIM,
                BACKBONE_DIM,
            )?,
            backbone: Backbone::load(weights, device, "adapter.decoder")?,
            head: Dense::load(
                weights,
                device,
                "adapter.head",
                BACKBONE_DIM,
                BACKBONE_DIM,
            )?,
        })
    }

    /// Both taps are `[1, frames, dim]`; so is the result.
    pub fn forward(
        &self,
        acoustic: Tensor<B, 3>,
        phonetic: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let summed = self.proj.forward(phonetic) + acoustic;
        let hidden = self.backbone.forward(summed.swap_dims(1, 2));
        self.head.forward(hidden.swap_dims(1, 2))
    }
}

/// The vocoder.
pub struct Vocoder<B: Backend> {
    backbone: Backbone<B>,
    out: Dense<B>,
}

impl<B: Backend> Vocoder<B> {
    /// Loads the vocoder.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, UnipaseError> {
        Ok(Self {
            backbone: Backbone::load(weights, device, "vocoder.decoder")?,
            out: Dense::load(
                weights,
                device,
                "vocoder.head.out",
                2 * bins(),
                BACKBONE_DIM,
            )?,
        })
    }

    /// `hidden` is `[1, frames, dim]`. The result is the head's raw
    /// `[2 · bins, frames]` output as host values.
    pub fn spectrum(&self, hidden: Tensor<B, 3>) -> Vec<f32> {
        let backbone = self.backbone.forward(hidden.swap_dims(1, 2));
        let raw = self
            .out
            .forward(backbone.swap_dims(1, 2))
            .swap_dims(1, 2)
            .cast(DType::F32);
        raw.into_data()
            .into_vec()
            .expect("the head's output as f32 values")
    }
}
