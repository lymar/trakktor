//! The codec decoder on burn: frames of codes in, a 24 kHz waveform out.
//!
//! A port of the candle decoder in `runtime::codec`, stage for stage, and the
//! runtime's strictest test: with the codes fixed the network is purely
//! feed-forward, so its output is the one thing that has to reproduce.
//!
//! Written for burn rather than mirrored from candle in two places:
//!
//! - **Every codebook of a group is one table.** The group's codebooks are
//!   concatenated at load and its entries divided by their usage counts there
//!   too, so dequantizing a chunk is one gather and one sum instead of a gather
//!   and an add per codebook.
//! - **The activation's parameters are exponentiated at load.** They are stored
//!   as logarithms and never change, so the exponentials — and the guard added
//!   to the divisor — are folded into the loaded tensors.
//!
//! Attention over the frame axis keeps the reference's sliding window, built as
//! an additive mask exactly as the candle path builds it.

use burn::tensor::{
    Int, Tensor, TensorData,
    activation::{self, silu},
    backend::Backend,
    module::{attention, conv_transpose1d, conv1d, embedding, linear},
    ops::{AttentionModuleOptions, ConvOptions, ConvTransposeOptions, PadMode},
};

use super::{
    Weights,
    net::{RmsNorm, matrix, rope_values, rows_of, weight},
};
use crate::tts::qwen3_tts::{
    chunking::{chunk_plan, window_visible},
    config::CodecConfig,
    error::Qwen3TtsError,
    runtime::model_err,
};

/// Guard on the divisor when turning accumulated codebook sums into entries.
const CLUSTER_USAGE_EPS: f32 = 1e-5;
/// The guard the reference adds to the Snake activation's divisor.
const SNAKE_EPS: f32 = 1e-9;
/// The dilations the reference gives the three residual units of every block.
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];
/// Normalization epsilon of the ConvNeXt blocks.
const CONVNEXT_NORM_EPS: f32 = 1e-6;

/// A 1-D convolution padded so the output never depends on future samples.
///
/// The reference pads the input on the left by the dilated receptive field
/// minus the stride; with the unit strides the decoder uses, that is exactly
/// `(kernel - 1) * dilation` and no padding is needed on the right.
struct CausalConv1d<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    left_pad: usize,
    dilation: usize,
    groups: usize,
}

impl<B: Backend> CausalConv1d<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, Qwen3TtsError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.conv.weight"),
                [out_channels, in_channels / groups, kernel_size],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.conv.bias"),
                [out_channels],
            )?,
            left_pad: (kernel_size - 1) * dilation,
            dilation,
            groups,
        })
    }

    /// Convolves `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let padded = x.pad((self.left_pad, 0, 0, 0), PadMode::Constant(0.0));
        conv1d(
            padded,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [0], [self.dilation], self.groups),
        )
    }
}

/// A transposed 1-D convolution trimmed on the right so it stays causal.
struct CausalConvTranspose1d<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    stride: usize,
    right_trim: usize,
}

impl<B: Backend> CausalConvTranspose1d<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    ) -> Result<Self, Qwen3TtsError> {
        Ok(Self {
            // Transposed convolutions store weights as [in, out, kernel].
            weight: weight(
                weights,
                device,
                &format!("{prefix}.conv.weight"),
                [in_channels, out_channels, kernel_size],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.conv.bias"),
                [out_channels],
            )?,
            stride,
            right_trim: kernel_size - stride,
        })
    }

    /// Upsamples `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = conv_transpose1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvTransposeOptions::new([self.stride], [0], [0], [1], 1),
        );
        if self.right_trim == 0 {
            return out;
        }
        let keep = out.dims()[2] - self.right_trim;
        out.narrow(2, 0, keep)
    }
}

/// The Snake activation with separate frequency and magnitude parameters:
/// `x + sin²(x·eᵃ) / eᵇ`.
///
/// Both parameters are stored as logarithms; they never change, so the
/// exponentials and the guard the reference adds to the divisor are folded in
/// at load.
struct SnakeBeta<B: Backend> {
    /// `eᵃ`, shaped `[1, channels, 1]`.
    alpha: Tensor<B, 3>,
    /// `eᵇ + ε`, shaped `[1, channels, 1]`.
    beta: Tensor<B, 3>,
}

impl<B: Backend> SnakeBeta<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, Qwen3TtsError> {
        let exponentiate = |name: &str,
                            guard: f32|
         -> Result<Tensor<B, 3>, Qwen3TtsError> {
            let (values, dims) = weights.parts(&format!("{prefix}.{name}"))?;
            if dims != [channels] {
                return Err(model_err(
                    &format!("{prefix}.{name}"),
                    format!("shape {dims:?}, expected {:?}", [channels]),
                ));
            }
            let values: Vec<f32> =
                values.iter().map(|v| v.exp() + guard).collect();
            Ok(Tensor::from_data(
                TensorData::new(values, [1, channels, 1]),
                device,
            ))
        };
        Ok(Self {
            alpha: exponentiate("alpha", 0.0)?,
            beta: exponentiate("beta", SNAKE_EPS)?,
        })
    }

    /// Applies the activation to `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let sine = (x.clone() * self.alpha.clone()).sin();
        x + sine.clone() * sine / self.beta.clone()
    }
}

/// One residual vector-quantization group: a stack of codebooks whose
/// dequantized entries are summed, then projected back to the codec width.
struct QuantizerGroup<B: Backend> {
    /// Every codebook of the group, one after another, already divided by its
    /// usage counts. Row `book · size + code` is that codebook's entry.
    entries: Tensor<B, 2>,
    count: usize,
    size: usize,
    /// The 1×1 convolution projecting the summed entries out, held as a matrix
    /// because a 1×1 convolution is a matrix multiply.
    output_proj: Tensor<B, 2>,
}

impl<B: Backend> QuantizerGroup<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        count: usize,
        cfg: &CodecConfig,
    ) -> Result<Self, Qwen3TtsError> {
        let mut entries =
            Vec::with_capacity(count * cfg.codebook_size * cfg.quantizer_dim);
        for index in 0..count {
            let book = format!("{prefix}.vq.layers.{index}._codebook");
            let sums = rows_of(
                weights,
                &format!("{book}.embedding_sum"),
                cfg.codebook_size,
                cfg.quantizer_dim,
            )?;
            let (usage, dims) =
                weights.parts(&format!("{book}.cluster_usage"))?;
            if dims != [cfg.codebook_size] {
                return Err(model_err(
                    &format!("{book}.cluster_usage"),
                    format!(
                        "shape {dims:?}, expected {:?}",
                        [cfg.codebook_size]
                    ),
                ));
            }
            // Entries are stored as running sums; dividing by the usage count
            // recovers the centroid.
            for (entry, count) in sums.chunks(cfg.quantizer_dim).zip(&usage) {
                let divisor = count.max(CLUSTER_USAGE_EPS);
                entries.extend(entry.iter().map(|value| value / divisor));
            }
        }

        let (proj, dims) =
            weights.parts(&format!("{prefix}.output_proj.weight"))?;
        if dims != [cfg.codebook_dim, cfg.quantizer_dim, 1] {
            return Err(model_err(
                &format!("{prefix}.output_proj.weight"),
                format!(
                    "shape {dims:?}, expected {:?}",
                    [cfg.codebook_dim, cfg.quantizer_dim, 1]
                ),
            ));
        }
        Ok(Self {
            entries: Tensor::from_data(
                TensorData::new(
                    entries,
                    [count * cfg.codebook_size, cfg.quantizer_dim],
                ),
                device,
            ),
            count,
            size: cfg.codebook_size,
            output_proj: Tensor::from_data(
                TensorData::new(
                    super::transpose_2d(
                        &proj,
                        cfg.codebook_dim,
                        cfg.quantizer_dim,
                    ),
                    [cfg.quantizer_dim, cfg.codebook_dim],
                ),
                device,
            ),
        })
    }

    /// Dequantizes `codes`, laid out codebook-major with `frames` per codebook,
    /// into `[1, frames, codebook_dim]`.
    fn decode(
        &self,
        codes: &[u32],
        frames: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let rows: Vec<i64> = codes
            .iter()
            .enumerate()
            .map(|(index, &code)| {
                ((index / frames) * self.size + code as usize) as i64
            })
            .collect();
        let picked = embedding(
            self.entries.clone(),
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(rows, [1, self.count * frames]),
                device,
            ),
        );
        let dim = self.entries.dims()[1];
        let summed = picked
            .reshape([self.count, frames, dim])
            .sum_dim(0)
            .reshape([1, frames, dim]);
        linear(summed, self.output_proj.clone(), None)
    }
}

/// The split quantizer: one semantic codebook plus the acoustic residuals,
/// each group projected separately and then summed.
struct Quantizer<B: Backend> {
    semantic: QuantizerGroup<B>,
    acoustic: QuantizerGroup<B>,
    num_semantic: usize,
}

impl<B: Backend> Quantizer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &CodecConfig,
    ) -> Result<Self, Qwen3TtsError> {
        let num_semantic = cfg.num_semantic_quantizers;
        Ok(Self {
            semantic: QuantizerGroup::load(
                weights,
                device,
                "decoder.quantizer.rvq_first",
                num_semantic,
                cfg,
            )?,
            acoustic: QuantizerGroup::load(
                weights,
                device,
                "decoder.quantizer.rvq_rest",
                cfg.num_quantizers - num_semantic,
                cfg,
            )?,
            num_semantic,
        })
    }

    /// Dequantizes every codebook of `codes`, laid out codebook-major.
    fn decode(
        &self,
        codes: &[u32],
        frames: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let split = self.num_semantic * frames;
        self.semantic.decode(&codes[..split], frames, device) +
            self.acoustic.decode(&codes[split..], frames, device)
    }
}

/// A ConvNeXt block: depthwise convolution, normalization, and a widening
/// pointwise pair, added back onto the input through a learned scale.
struct ConvNextBlock<B: Backend> {
    dwconv: CausalConv1d<B>,
    norm_weight: Tensor<B, 3>,
    norm_bias: Tensor<B, 3>,
    pwconv1: (Tensor<B, 2>, Tensor<B, 1>),
    pwconv2: (Tensor<B, 2>, Tensor<B, 1>),
    gamma: Tensor<B, 3>,
}

impl<B: Backend> ConvNextBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
    ) -> Result<Self, Qwen3TtsError> {
        let vector = |name: &str| -> Result<Tensor<B, 3>, Qwen3TtsError> {
            Ok(weight::<B, 1>(weights, device, name, [dim])?
                .reshape([1, 1, dim]))
        };
        Ok(Self {
            // Depthwise: one group per channel.
            dwconv: CausalConv1d::load(
                weights,
                device,
                &format!("{prefix}.dwconv"),
                dim,
                dim,
                7,
                1,
                dim,
            )?,
            norm_weight: vector(&format!("{prefix}.norm.weight"))?,
            norm_bias: vector(&format!("{prefix}.norm.bias"))?,
            pwconv1: (
                matrix(
                    weights,
                    device,
                    &format!("{prefix}.pwconv1.weight"),
                    4 * dim,
                    dim,
                )?,
                weight(
                    weights,
                    device,
                    &format!("{prefix}.pwconv1.bias"),
                    [4 * dim],
                )?,
            ),
            pwconv2: (
                matrix(
                    weights,
                    device,
                    &format!("{prefix}.pwconv2.weight"),
                    dim,
                    4 * dim,
                )?,
                weight(
                    weights,
                    device,
                    &format!("{prefix}.pwconv2.bias"),
                    [dim],
                )?,
            ),
            gamma: vector(&format!("{prefix}.gamma"))?,
        })
    }

    /// Runs the block over `x`, shaped `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.dwconv.forward(x.clone()).swap_dims(1, 2);
        // Layer normalization over the channel axis.
        let mean = hidden.clone().mean_dim(2);
        let centered = hidden - mean;
        let variance = (centered.clone() * centered.clone()).mean_dim(2);
        let normed = centered / variance.add_scalar(CONVNEXT_NORM_EPS).sqrt();
        let hidden = normed * self.norm_weight.clone() + self.norm_bias.clone();

        let hidden = activation::gelu(linear(
            hidden,
            self.pwconv1.0.clone(),
            Some(self.pwconv1.1.clone()),
        ));
        let hidden = linear(
            hidden,
            self.pwconv2.0.clone(),
            Some(self.pwconv2.1.clone()),
        );
        x + (hidden * self.gamma.clone()).swap_dims(1, 2)
    }
}

/// A residual unit: two activated convolutions added back onto the input.
struct ResidualUnit<B: Backend> {
    act1: SnakeBeta<B>,
    conv1: CausalConv1d<B>,
    act2: SnakeBeta<B>,
    conv2: CausalConv1d<B>,
}

impl<B: Backend> ResidualUnit<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dim: usize,
        dilation: usize,
    ) -> Result<Self, Qwen3TtsError> {
        Ok(Self {
            act1: SnakeBeta::load(
                weights,
                device,
                &format!("{prefix}.act1"),
                dim,
            )?,
            conv1: CausalConv1d::load(
                weights,
                device,
                &format!("{prefix}.conv1"),
                dim,
                dim,
                7,
                dilation,
                1,
            )?,
            act2: SnakeBeta::load(
                weights,
                device,
                &format!("{prefix}.act2"),
                dim,
            )?,
            conv2: CausalConv1d::load(
                weights,
                device,
                &format!("{prefix}.conv2"),
                dim,
                dim,
                1,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.conv1.forward(self.act1.forward(x.clone()));
        x + self.conv2.forward(self.act2.forward(hidden))
    }
}

/// One block of the convolutional decoder: activate, upsample, then three
/// dilated residual units.
struct DecoderBlock<B: Backend> {
    act: SnakeBeta<B>,
    upsample: CausalConvTranspose1d<B>,
    units: Vec<ResidualUnit<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        rate: usize,
    ) -> Result<Self, Qwen3TtsError> {
        let units = RESIDUAL_DILATIONS
            .iter()
            .enumerate()
            .map(|(index, &dilation)| {
                // The units follow the activation and the upsample in the
                // block.
                ResidualUnit::load(
                    weights,
                    device,
                    &format!("{prefix}.{}", index + 2),
                    out_dim,
                    dilation,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            act: SnakeBeta::load(
                weights,
                device,
                &format!("{prefix}.0"),
                in_dim,
            )?,
            upsample: CausalConvTranspose1d::load(
                weights,
                device,
                &format!("{prefix}.1"),
                in_dim,
                out_dim,
                2 * rate,
                rate,
            )?,
            units,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut hidden = self.upsample.forward(self.act.forward(x));
        for unit in &self.units {
            hidden = unit.forward(hidden);
        }
        hidden
    }
}

/// One transformer layer of the pre-decoder stack: windowed self-attention and
/// a gated feed-forward, each added back through its own learned scale.
struct TransformerLayer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
    /// `q`, `k` and `v` glued into one `[hidden, 3·heads·dim]`.
    qkv: Tensor<B, 2>,
    o_proj: Tensor<B, 2>,
    /// `gate` and `up` glued into one `[hidden, 2·intermediate]`.
    gate_up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    attn_scale: Tensor<B, 3>,
    mlp_scale: Tensor<B, 3>,
    num_heads: usize,
    head_dim: usize,
    intermediate: usize,
}

impl<B: Backend> TransformerLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &CodecConfig,
    ) -> Result<Self, Qwen3TtsError> {
        let hidden = cfg.hidden_size;
        let inner = cfg.num_attention_heads * cfg.head_dim;
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let part = |name: &str,
                    out: usize|
         -> Result<(Vec<f32>, usize), Qwen3TtsError> {
            Ok((rows_of(weights, name, out, hidden)?, out))
        };
        let qkv = super::net::glue_columns(
            &[
                part(&format!("{attention}.q_proj.weight"), inner)?,
                part(&format!("{attention}.k_proj.weight"), inner)?,
                part(&format!("{attention}.v_proj.weight"), inner)?,
            ],
            hidden,
        );
        let gate_up = super::net::glue_columns(
            &[
                part(
                    &format!("{mlp}.gate_proj.weight"),
                    cfg.intermediate_size,
                )?,
                part(&format!("{mlp}.up_proj.weight"), cfg.intermediate_size)?,
            ],
            hidden,
        );
        let scale = |name: &str| -> Result<Tensor<B, 3>, Qwen3TtsError> {
            Ok(weight::<B, 1>(weights, device, name, [hidden])?
                .reshape([1, 1, hidden]))
        };
        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                cfg.rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                cfg.rms_norm_eps,
            )?,
            qkv: Tensor::from_data(
                TensorData::new(qkv, [hidden, 3 * inner]),
                device,
            ),
            o_proj: matrix(
                weights,
                device,
                &format!("{attention}.o_proj.weight"),
                hidden,
                inner,
            )?,
            gate_up: Tensor::from_data(
                TensorData::new(gate_up, [hidden, 2 * cfg.intermediate_size]),
                device,
            ),
            down: matrix(
                weights,
                device,
                &format!("{mlp}.down_proj.weight"),
                hidden,
                cfg.intermediate_size,
            )?,
            attn_scale: scale(&format!(
                "{prefix}.self_attn_layer_scale.scale"
            ))?,
            mlp_scale: scale(&format!("{prefix}.mlp_layer_scale.scale"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            intermediate: cfg.intermediate_size,
        })
    }

    /// Runs the layer over `x`, shaped `[batch, time, hidden]`.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        mask: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let inner = self.num_heads * self.head_dim;

        let normed = self.input_layernorm.forward(x.clone());
        let projected = linear(normed, self.qkv.clone(), None);
        let head = |slice: Tensor<B, 3>| -> Tensor<B, 4> {
            slice
                .reshape([batch, seq, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let query = super::net::apply_rope(
            head(projected.clone().narrow(2, 0, inner)),
            cos,
            sin,
        );
        let key = super::net::apply_rope(
            head(projected.clone().narrow(2, inner, inner)),
            cos,
            sin,
        );
        let value = head(projected.narrow(2, 2 * inner, inner));

        let context = attention(
            query,
            key,
            value,
            None,
            Some(mask.clone()),
            AttentionModuleOptions::default(),
        );
        let attended = context.swap_dims(1, 2).reshape([batch, seq, inner]);
        let x = x + linear(attended, self.o_proj.clone(), None) *
            self.attn_scale.clone();

        let normed = self.post_attention_layernorm.forward(x.clone());
        let projected = linear(normed, self.gate_up.clone(), None);
        let gated = silu(projected.clone().narrow(2, 0, self.intermediate)) *
            projected.narrow(2, self.intermediate, self.intermediate);
        x + linear(gated, self.down.clone(), None) * self.mlp_scale.clone()
    }
}

/// The codec decoder.
pub struct CodecDecoder<B: Backend> {
    device: B::Device,
    cfg: CodecConfig,
    quantizer: Quantizer<B>,
    pre_conv: CausalConv1d<B>,
    input_proj: (Tensor<B, 2>, Tensor<B, 1>),
    layers: Vec<TransformerLayer<B>>,
    norm: RmsNorm<B>,
    output_proj: (Tensor<B, 2>, Tensor<B, 1>),
    upsample: Vec<(CausalConvTranspose1d<B>, ConvNextBlock<B>)>,
    head_conv: CausalConv1d<B>,
    blocks: Vec<DecoderBlock<B>>,
    tail_act: SnakeBeta<B>,
    tail_conv: CausalConv1d<B>,
}

impl<B: Backend> CodecDecoder<B> {
    /// Loads the decoder half of a codec checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when a tensor is missing or does
    /// not match the declared geometry.
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &CodecConfig,
    ) -> Result<Self, Qwen3TtsError> {
        let transformer = "decoder.pre_transformer";
        let bias_pair = |prefix: &str,
                         out: usize,
                         in_dim: usize|
         -> Result<
            (Tensor<B, 2>, Tensor<B, 1>),
            Qwen3TtsError,
        > {
            Ok((
                matrix(
                    weights,
                    device,
                    &format!("{prefix}.weight"),
                    out,
                    in_dim,
                )?,
                weight(weights, device, &format!("{prefix}.bias"), [out])?,
            ))
        };

        let layers = (0..cfg.num_hidden_layers)
            .map(|index| {
                TransformerLayer::load(
                    weights,
                    device,
                    &format!("{transformer}.layers.{index}"),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let upsample = cfg
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(index, &ratio)| {
                let prefix = format!("decoder.upsample.{index}");
                Ok((
                    CausalConvTranspose1d::load(
                        weights,
                        device,
                        &format!("{prefix}.0"),
                        cfg.latent_dim,
                        cfg.latent_dim,
                        ratio,
                        ratio,
                    )?,
                    ConvNextBlock::load(
                        weights,
                        device,
                        &format!("{prefix}.1"),
                        cfg.latent_dim,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, Qwen3TtsError>>()?;

        let blocks = cfg
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(index, &rate)| {
                DecoderBlock::load(
                    weights,
                    device,
                    &format!("decoder.decoder.{}.block", index + 1),
                    cfg.decoder_dim >> index,
                    cfg.decoder_dim >> (index + 1),
                    rate,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tail_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
        let tail_index = cfg.upsample_rates.len() + 1;

        Ok(Self {
            device: device.clone(),
            cfg: cfg.clone(),
            quantizer: Quantizer::load(weights, device, cfg)?,
            pre_conv: CausalConv1d::load(
                weights,
                device,
                "decoder.pre_conv",
                cfg.codebook_dim,
                cfg.latent_dim,
                3,
                1,
                1,
            )?,
            input_proj: bias_pair(
                &format!("{transformer}.input_proj"),
                cfg.hidden_size,
                cfg.latent_dim,
            )?,
            layers,
            norm: RmsNorm::load(
                weights,
                device,
                &format!("{transformer}.norm.weight"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            output_proj: bias_pair(
                &format!("{transformer}.output_proj"),
                cfg.latent_dim,
                cfg.hidden_size,
            )?,
            upsample,
            head_conv: CausalConv1d::load(
                weights,
                device,
                "decoder.decoder.0",
                cfg.latent_dim,
                cfg.decoder_dim,
                7,
                1,
                1,
            )?,
            blocks,
            tail_act: SnakeBeta::load(
                weights,
                device,
                &format!("decoder.decoder.{tail_index}"),
                tail_dim,
            )?,
            tail_conv: CausalConv1d::load(
                weights,
                device,
                &format!("decoder.decoder.{}", tail_index + 1),
                tail_dim,
                1,
                7,
                1,
                1,
            )?,
        })
    }

    /// The sample rate of the waveform this decoder produces.
    pub(super) fn sample_rate(&self) -> u32 { self.cfg.output_sample_rate }

    /// Decodes `frames` — one row of `num_quantizers` codes per frame — into a
    /// mono waveform.
    ///
    /// Long inputs are split into chunks carrying a left context, exactly as
    /// the reference splits them, so the result does not depend on how many
    /// frames are decoded at once.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when a frame does not carry the
    /// expected number of codes, or the backend cannot return the waveform.
    pub(super) fn decode(
        &self,
        frames: &[Vec<u32>],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let quantizers = self.cfg.num_quantizers;
        for (index, frame) in frames.iter().enumerate() {
            if frame.len() != quantizers {
                return Err(model_err(
                    "decoding the codec frames",
                    format!(
                        "frame {index} carries {} codes, expected {quantizers}",
                        frame.len()
                    ),
                ));
            }
        }

        let upsample = self.cfg.decode_upsample_rate;
        let total = frames.len();
        let mut wave: Vec<f32> = Vec::with_capacity(total * upsample);
        for chunk in chunk_plan(total) {
            // Lay the chunk's codes out codebook-major: the quantizer indexes
            // whole rows.
            let span = chunk.span();
            let start = chunk.context_start();
            let mut codes = vec![0u32; quantizers * span];
            for (offset, frame) in frames[start..chunk.end].iter().enumerate() {
                for (book, &code) in frame.iter().enumerate() {
                    codes[book * span + offset] = code;
                }
            }
            let mut decoded = self.forward(&codes, span)?;
            // Drop the samples the context produced; they only prime the
            // convolutions.
            wave.extend_from_slice(
                &decoded.split_off(chunk.context * upsample),
            );
        }
        Ok(wave)
    }

    /// Runs the network over one chunk of codes, laid out codebook-major.
    fn forward(
        &self,
        codes: &[u32],
        frames: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        let hidden = self.quantizer.decode(codes, frames, &self.device);
        let hidden = self.pre_conv.forward(hidden.swap_dims(1, 2));

        // The attention stack sees a causal window, so a frame never depends
        // on frames after it.
        let (cos, sin) = self.rope(frames);
        let mask = self.window_mask(frames);
        let mut hidden = linear(
            hidden.swap_dims(1, 2),
            self.input_proj.0.clone(),
            Some(self.input_proj.1.clone()),
        );
        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin, &mask);
        }
        let hidden = self.norm.forward(hidden);
        let hidden = linear(
            hidden,
            self.output_proj.0.clone(),
            Some(self.output_proj.1.clone()),
        );

        let mut hidden = hidden.swap_dims(1, 2);
        for (upsample, convnext) in &self.upsample {
            hidden = convnext.forward(upsample.forward(hidden));
        }

        let mut wave = self.head_conv.forward(hidden);
        for block in &self.blocks {
            wave = block.forward(wave);
        }
        let wave = self.tail_conv.forward(self.tail_act.forward(wave));
        wave.clamp(-1.0, 1.0)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| model_err("reading the waveform", format!("{e:?}")))
    }

    /// The rotary tables for `frames` positions.
    fn rope(&self, frames: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let dim = self.cfg.head_dim;
        let (cos, sin) = rope_values(dim, frames, self.cfg.rope_theta);
        let table = |values: Vec<f32>| {
            Tensor::<B, 1>::from_data(
                TensorData::new(values, [frames * dim]),
                &self.device,
            )
            .reshape([1, 1, frames, dim])
        };
        (table(cos), table(sin))
    }

    /// The additive attention mask: a frame may attend to itself and to the
    /// `window - 1` frames before it, and to nothing after it.
    fn window_mask(&self, frames: usize) -> Tensor<B, 4> {
        let window = self.cfg.sliding_window;
        let mut mask = vec![0f32; frames * frames];
        for query in 0..frames {
            for key in 0..frames {
                if !window_visible(query, key, window) {
                    mask[query * frames + key] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_data(
            TensorData::new(mask, [1, 1, frames, frames]),
            &self.device,
        )
    }
}

#[cfg(test)]
mod tests;
