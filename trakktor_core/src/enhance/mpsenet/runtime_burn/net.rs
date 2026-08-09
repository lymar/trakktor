//! The network, on burn.
//!
//! The same architecture as the candle runtime's, written against burn's
//! functional tensor operations. Two things are simpler here and one is not:
//!
//! - burn's `conv2d` takes a stride, a padding and a dilation **per axis**, so
//!   the dense blocks' `(2, 3)` kernels with their time-only dilation are
//!   ordinary two-dimensional convolutions rather than a sum of frequency ones.
//!   Their padding is still asymmetric — the whole dilation at the front of the
//!   time axis and nothing at the back, which is what makes the blocks causal —
//!   so it is applied to the tensor first and the convolution pads by nothing;
//! - attention is the fused primitive, which does not build the score matrix;
//! - there is no recurrence, so the bidirectional unit is written out, with the
//!   two directions stacked into a leading axis and stepped together.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.
//!
//! Ported from MP-SENet (MIT).

use burn::tensor::{
    Tensor, activation,
    backend::Backend,
    module::{attention, conv2d, linear},
    ops::{AttentionModuleOptions, ConvOptions, PadMode},
};

use super::Weights;
use crate::enhance::{
    EnhanceError,
    mpsenet::config::{
        ATTENTION_HEADS, BINS, CHANNELS, DENSE_DEPTH, GRU_HIDDEN,
        INSTANCE_NORM_EPS, LEAKY_SLOPE, MASK_BETA, NORM_EPS, TS_BLOCKS,
    },
};

/// Largest attention score matrix computed at once, in elements — the same
/// budget the candle runtime keeps, so the two split the batch the same way.
const ATTENTION_BUDGET: usize = 32 << 20;

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, EnhanceError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err_str(
            key,
            &format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(
        burn::tensor::TensorData::new(values, shape),
        device,
    ))
}

/// A checkpoint error naming the tensor it is about.
fn model_err_str(key: &str, what: &str) -> EnhanceError {
    EnhanceError::Checkpoint(format!("{key}: {what}"))
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
) -> Result<Tensor<B, 2>, EnhanceError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err_str(
            key,
            &format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
        ));
    }
    Ok(Tensor::from_data(
        burn::tensor::TensorData::new(
            transpose_2d(&values, out_dim, in_dim),
            [in_dim, out_dim],
        ),
        device,
    ))
}

/// A rectifier with a small slope below zero.
fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let positive = activation::relu(x.clone());
    let negative = x - positive.clone();
    positive + negative * LEAKY_SLOPE
}

/// A parametric rectifier with one slope per channel.
struct Prelu<B: Backend> {
    slope: Tensor<B, 4>,
}

impl<B: Backend> Prelu<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            slope: weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.weight"),
                [CHANNELS],
            )?
            .reshape([1, CHANNELS, 1, 1]),
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let positive = activation::relu(x.clone());
        let negative = x - positive.clone();
        positive + negative * self.slope.clone()
    }
}

/// Normalization over one channel of one window — every frame and every bin of
/// it — with a learned scale and shift. See the candle runtime's note: unlike
/// a batch norm this keeps no running statistics, so it is a computation and
/// not a stored affine map.
struct InstanceNorm<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 4>,
}

impl<B: Backend> InstanceNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        let plane = |name: &str| -> Result<Tensor<B, 4>, EnhanceError> {
            Ok(weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.{name}"),
                [CHANNELS],
            )?
            .reshape([1, CHANNELS, 1, 1]))
        };
        Ok(Self {
            weight: plane("weight")?,
            bias: plane("bias")?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mean = x.clone().mean_dim(3).mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(3).mean_dim(2);
        let normed = centered / (variance + INSTANCE_NORM_EPS).sqrt();
        normed * self.weight.clone() + self.bias.clone()
    }
}

/// A convolution over frequency alone — every `(1, k)` kernel in the network.
struct FreqConv<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
    stride: usize,
    padding: usize,
}

impl<B: Backend> FreqConv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_channels, in_channels, 1, kernel],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_channels],
            )?,
            stride,
            padding,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1, self.stride], [0, self.padding], [1, 1], 1),
        )
    }
}

/// A dense block's `(2, 3)` convolution, dilated over time alone and padded at
/// the front of it — which is what makes the block causal.
struct TimeFreqConv<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
    dilation: usize,
}

impl<B: Backend> TimeFreqConv<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        dilation: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [CHANNELS, in_channels, 2, 3],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [CHANNELS],
            )?,
            dilation,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let padded =
            x.pad([(self.dilation, 0), (1, 1)], PadMode::Constant(0.0));
        conv2d(
            padded,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1, 1], [0, 0], [self.dilation, 1], 1),
        )
    }
}

/// Convolution, instance norm, parametric rectifier.
struct DenseLayer<B: Backend> {
    conv: TimeFreqConv<B>,
    norm: InstanceNorm<B>,
    prelu: Prelu<B>,
}

impl<B: Backend> DenseLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_channels: usize,
        dilation: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            conv: TimeFreqConv::load(
                weights,
                device,
                &format!("{prefix}.1"),
                in_channels,
                dilation,
            )?,
            norm: InstanceNorm::load(weights, device, &format!("{prefix}.2"))?,
            prelu: Prelu::load(weights, device, &format!("{prefix}.3"))?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.prelu.forward(self.norm.forward(self.conv.forward(x)))
    }
}

/// Four convolutions, each of which sees everything the ones before it
/// produced as well as the block's input; the dilation doubles from layer to
/// layer.
struct DenseBlock<B: Backend> {
    layers: Vec<DenseLayer<B>>,
}

impl<B: Backend> DenseBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            layers: (0..DENSE_DEPTH)
                .map(|index| {
                    DenseLayer::load(
                        weights,
                        device,
                        &format!("{prefix}.dense_block.{index}"),
                        CHANNELS * (index + 1),
                        1 << index,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut skip = x.clone();
        let mut out = x;
        for layer in &self.layers {
            out = layer.forward(skip.clone());
            skip = Tensor::cat(vec![out.clone(), skip], 1);
        }
        out
    }
}

/// The encoder: a pointwise lift into 64 channels, a dense block, and one
/// stride-2 convolution that halves the frequency axis.
struct Encoder<B: Backend> {
    lift: FreqConv<B>,
    lift_norm: InstanceNorm<B>,
    lift_prelu: Prelu<B>,
    dense: DenseBlock<B>,
    down: FreqConv<B>,
    down_norm: InstanceNorm<B>,
    down_prelu: Prelu<B>,
}

impl<B: Backend> Encoder<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        let root = "dense_encoder";
        Ok(Self {
            lift: FreqConv::load(
                weights,
                device,
                &format!("{root}.dense_conv_1.0"),
                2,
                CHANNELS,
                1,
                1,
                0,
            )?,
            lift_norm: InstanceNorm::load(
                weights,
                device,
                &format!("{root}.dense_conv_1.1"),
            )?,
            lift_prelu: Prelu::load(
                weights,
                device,
                &format!("{root}.dense_conv_1.2"),
            )?,
            dense: DenseBlock::load(
                weights,
                device,
                &format!("{root}.dense_block"),
            )?,
            down: FreqConv::load(
                weights,
                device,
                &format!("{root}.dense_conv_2.0"),
                CHANNELS,
                CHANNELS,
                3,
                2,
                1,
            )?,
            down_norm: InstanceNorm::load(
                weights,
                device,
                &format!("{root}.dense_conv_2.1"),
            )?,
            down_prelu: Prelu::load(
                weights,
                device,
                &format!("{root}.dense_conv_2.2"),
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self
            .lift_prelu
            .forward(self.lift_norm.forward(self.lift.forward(x)));
        let x = self.dense.forward(x);
        self.down_prelu
            .forward(self.down_norm.forward(self.down.forward(x)))
    }
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
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [CHANNELS],
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [CHANNELS],
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powi_scalar(2).mean_dim(2);
        let normed = centered / (variance + NORM_EPS).sqrt();
        normed * self.weight.clone().reshape([1, 1, CHANNELS]) +
            self.bias.clone().reshape([1, 1, CHANNELS])
    }
}

/// A gated recurrent unit that runs both ways at once. See the candle
/// runtime's note on why the two directions are stacked rather than looped
/// twice.
struct BiGru<B: Backend> {
    /// `[2, input, 3·hidden]`, the directions stacked.
    weight_ih: Tensor<B, 3>,
    /// `[2, hidden, 3·hidden]`.
    weight_hh: Tensor<B, 3>,
    /// `[2, 1, 3·hidden]`.
    bias_ih: Tensor<B, 3>,
    bias_hh: Tensor<B, 3>,
    hidden: usize,
}

impl<B: Backend> BiGru<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        input: usize,
        hidden: usize,
    ) -> Result<Self, EnhanceError> {
        let pair = |name: &str,
                    rows: usize,
                    cols: usize|
         -> Result<Tensor<B, 3>, EnhanceError> {
            let forward = matrix(
                weights,
                device,
                &format!("{prefix}.{name}_l0"),
                rows,
                cols,
            )?;
            let backward = matrix(
                weights,
                device,
                &format!("{prefix}.{name}_l0_reverse"),
                rows,
                cols,
            )?;
            Ok(Tensor::stack(vec![forward, backward], 0))
        };
        let biases = |name: &str| -> Result<Tensor<B, 3>, EnhanceError> {
            let forward = weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.{name}_l0"),
                [3 * hidden],
            )?;
            let backward = weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.{name}_l0_reverse"),
                [3 * hidden],
            )?;
            Ok(Tensor::stack::<2>(vec![forward, backward], 0).reshape([
                2,
                1,
                3 * hidden,
            ]))
        };
        Ok(Self {
            weight_ih: pair("weight_ih", 3 * hidden, input)?,
            weight_hh: pair("weight_hh", 3 * hidden, hidden)?,
            bias_ih: biases("bias_ih")?,
            bias_hh: biases("bias_hh")?,
            hidden,
        })
    }

    /// Runs `[batch, steps, input]` and returns `[batch, steps, 2·hidden]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, steps, input] = x.dims();
        let device = x.device();
        // The backward direction reads the sequence reversed, so both
        // directions can then be stepped with the same index.
        let both = Tensor::stack::<4>(vec![x.clone(), x.flip([1])], 0);
        let projected = both
            .reshape([2, batch * steps, input])
            .matmul(self.weight_ih.clone())
            .add(self.bias_ih.clone())
            .reshape([2, batch, steps, 3 * self.hidden]);

        let mut state = Tensor::<B, 3>::zeros([2, batch, self.hidden], &device);
        let mut outputs = Vec::with_capacity(steps);
        for step in 0..steps {
            let gates_x = projected.clone().narrow(2, step, 1).reshape([
                2,
                batch,
                3 * self.hidden,
            ]);
            let gates_h = state
                .clone()
                .matmul(self.weight_hh.clone())
                .add(self.bias_hh.clone());
            let gate = |xs: &Tensor<B, 3>, index: usize| -> Tensor<B, 3> {
                xs.clone().narrow(2, index * self.hidden, self.hidden)
            };
            let reset =
                activation::sigmoid(gate(&gates_x, 0) + gate(&gates_h, 0));
            let update =
                activation::sigmoid(gate(&gates_x, 1) + gate(&gates_h, 1));
            // The candidate applies the reset gate to the *biased* recurrent
            // projection, which is torch's convention and not every library's.
            let candidate =
                (gate(&gates_x, 2) + reset * gate(&gates_h, 2)).tanh();
            state = (-update.clone() + 1.0) * candidate + update * state;
            outputs.push(state.clone());
        }

        let stacked = Tensor::stack::<4>(outputs, 2);
        let forward = stacked.clone().narrow(0, 0, 1).reshape([
            batch,
            steps,
            self.hidden,
        ]);
        let backward = stacked
            .narrow(0, 1, 1)
            .reshape([batch, steps, self.hidden])
            .flip([1]);
        Tensor::cat(vec![forward, backward], 2)
    }
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
    ) -> Result<Self, EnhanceError> {
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

/// Multi-head self-attention over `[batch, steps, channels]`.
struct Attention<B: Backend> {
    in_proj: Tensor<B, 2>,
    in_bias: Tensor<B, 1>,
    out_proj: Dense<B>,
}

impl<B: Backend> Attention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            in_proj: matrix(
                weights,
                device,
                &format!("{prefix}.in_proj_weight"),
                3 * CHANNELS,
                CHANNELS,
            )?,
            in_bias: weight(
                weights,
                device,
                &format!("{prefix}.in_proj_bias"),
                [3 * CHANNELS],
            )?,
            out_proj: Dense::load(
                weights,
                device,
                &format!("{prefix}.out_proj"),
                CHANNELS,
                CHANNELS,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, steps, dim] = x.dims();
        let head = dim / ATTENTION_HEADS;
        let projected =
            linear(x, self.in_proj.clone(), Some(self.in_bias.clone()))
                .reshape([batch, steps, 3, dim]);
        let split = |which: usize| -> Tensor<B, 4> {
            projected
                .clone()
                .narrow(2, which, 1)
                .reshape([batch, steps, ATTENTION_HEADS, head])
                .swap_dims(1, 2)
        };
        let query = split(0);
        let key = split(1);
        let value = split(2);

        // The fused kernel does not build the score matrix, but the batch is
        // still walked in pieces — the same ones the candle runtime uses — so
        // that a long window costs the same wherever it runs.
        let per_entry = ATTENTION_HEADS * steps * steps;
        let chunk = (ATTENTION_BUDGET / per_entry.max(1)).clamp(1, batch);
        let mut pieces = Vec::with_capacity(batch.div_ceil(chunk));
        for start in (0..batch).step_by(chunk) {
            let take = chunk.min(batch - start);
            pieces.push(attention(
                query.clone().narrow(0, start, take),
                key.clone().narrow(0, start, take),
                value.clone().narrow(0, start, take),
                None,
                None,
                AttentionModuleOptions::default(),
            ));
        }
        let context = if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            Tensor::cat(pieces, 0)
        };
        self.out_proj
            .forward(context.swap_dims(1, 2).reshape([batch, steps, dim]))
    }
}

/// One transformer over `[batch, steps, channels]`.
struct Transformer<B: Backend> {
    norm1: AffineNorm<B>,
    attention: Attention<B>,
    norm2: AffineNorm<B>,
    gru: BiGru<B>,
    project: Dense<B>,
    norm3: AffineNorm<B>,
}

impl<B: Backend> Transformer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            norm1: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm1"),
            )?,
            attention: Attention::load(
                weights,
                device,
                &format!("{prefix}.attention"),
            )?,
            norm2: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm2"),
            )?,
            gru: BiGru::load(
                weights,
                device,
                &format!("{prefix}.ffn.gru"),
                CHANNELS,
                GRU_HIDDEN,
            )?,
            project: Dense::load(
                weights,
                device,
                &format!("{prefix}.ffn.linear"),
                CHANNELS,
                2 * GRU_HIDDEN,
            )?,
            norm3: AffineNorm::load(
                weights,
                device,
                &format!("{prefix}.norm3"),
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attended = self.attention.forward(self.norm1.forward(x.clone()));
        let x = x + attended;
        let recurrent = self.gru.forward(self.norm2.forward(x.clone()));
        let projected = self.project.forward(leaky_relu(recurrent));
        self.norm3.forward(x + projected)
    }
}

/// One two-stage block: a transformer across frequency and one across time.
struct TsBlock<B: Backend> {
    across_frequency: Transformer<B>,
    across_time: Transformer<B>,
}

impl<B: Backend> TsBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        index: usize,
    ) -> Result<Self, EnhanceError> {
        let prefix = format!("TSTransformer.{index}");
        Ok(Self {
            across_frequency: Transformer::load(
                weights,
                device,
                &format!("{prefix}.time_transformer"),
            )?,
            across_time: Transformer::load(
                weights,
                device,
                &format!("{prefix}.freq_transformer"),
            )?,
        })
    }

    /// `x` is `[1, channels, time, frequency]`, and so is the result.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, channels, time, freq] = x.dims();
        // `[1, c, t, f]` → `[t, f, c]`: time batches, frequency is the
        // sequence.
        let over_freq = x.reshape([channels, time, freq]).permute([1, 2, 0]);
        let over_freq =
            over_freq.clone() + self.across_frequency.forward(over_freq);
        // The same tensor read the other way.
        let over_time = over_freq.permute([1, 0, 2]);
        let over_time = over_time.clone() + self.across_time.forward(over_time);
        over_time
            .permute([2, 1, 0])
            .reshape([1, channels, time, freq])
    }
}

/// The upsampler both decoders end with: a convolution to twice the channels,
/// unwoven into twice the bins.
struct SubPixel<B: Backend> {
    conv: FreqConv<B>,
}

impl<B: Backend> SubPixel<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            conv: FreqConv::load(
                weights,
                device,
                &format!("{prefix}.conv"),
                CHANNELS,
                2 * CHANNELS,
                3,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(x);
        let [batch, _, time, freq] = out.dims();
        out.reshape([batch, 2, CHANNELS, time, freq])
            .permute([0, 2, 3, 4, 1])
            .reshape([batch, CHANNELS, time, freq * 2])
    }
}

/// The head both decoders share.
struct DecoderStem<B: Backend> {
    dense: DenseBlock<B>,
    up: SubPixel<B>,
    norm: InstanceNorm<B>,
    prelu: Prelu<B>,
}

impl<B: Backend> DecoderStem<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        root: &str,
        conv: &str,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            dense: DenseBlock::load(
                weights,
                device,
                &format!("{root}.dense_block"),
            )?,
            up: SubPixel::load(weights, device, &format!("{root}.{conv}.0"))?,
            norm: InstanceNorm::load(
                weights,
                device,
                &format!("{root}.{conv}.1"),
            )?,
            prelu: Prelu::load(weights, device, &format!("{root}.{conv}.2"))?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.dense.forward(x);
        self.prelu.forward(self.norm.forward(self.up.forward(x)))
    }
}

/// The magnitude decoder: a gain per bin, between zero and `β`.
struct MaskDecoder<B: Backend> {
    stem: DecoderStem<B>,
    out: FreqConv<B>,
    slope: Tensor<B, 2>,
}

impl<B: Backend> MaskDecoder<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            stem: DecoderStem::load(
                weights,
                device,
                "mask_decoder",
                "mask_conv",
            )?,
            out: FreqConv::load(
                weights,
                device,
                "mask_decoder.mask_conv.3",
                CHANNELS,
                1,
                2,
                1,
                0,
            )?,
            slope: weight(
                weights,
                device,
                "mask_decoder.lsigmoid.slope",
                [BINS, 1],
            )?,
        })
    }

    /// The result is `[bins, frames]`.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let raw = self.out.forward(self.stem.forward(x));
        let [_, _, time, freq] = raw.dims();
        let raw = raw.reshape([time, freq]).swap_dims(0, 1);
        activation::sigmoid(raw * self.slope.clone()) * MASK_BETA
    }
}

/// The phase decoder: two components per bin, which the host turns into an
/// angle.
struct PhaseDecoder<B: Backend> {
    stem: DecoderStem<B>,
    real: FreqConv<B>,
    imag: FreqConv<B>,
}

impl<B: Backend> PhaseDecoder<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        let head = |name: &str| -> Result<FreqConv<B>, EnhanceError> {
            FreqConv::load(
                weights,
                device,
                &format!("phase_decoder.{name}"),
                CHANNELS,
                1,
                2,
                1,
                0,
            )
        };
        Ok(Self {
            stem: DecoderStem::load(
                weights,
                device,
                "phase_decoder",
                "phase_conv",
            )?,
            real: head("phase_conv_r")?,
            imag: head("phase_conv_i")?,
        })
    }

    /// Both results are `[bins, frames]`.
    fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = self.stem.forward(x);
        let plane = |xs: Tensor<B, 4>| -> Tensor<B, 2> {
            let [_, _, time, freq] = xs.dims();
            xs.reshape([time, freq]).swap_dims(0, 1)
        };
        (
            plane(self.real.forward(hidden.clone())),
            plane(self.imag.forward(hidden)),
        )
    }
}

/// The whole network.
pub struct Mpsenet<B: Backend> {
    encoder: Encoder<B>,
    blocks: Vec<TsBlock<B>>,
    mask: MaskDecoder<B>,
    phase: PhaseDecoder<B>,
}

impl<B: Backend> Mpsenet<B> {
    /// Loads the converted checkpoint onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when a tensor is missing or has a
    /// shape this network does not have.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            encoder: Encoder::load(weights, device)?,
            blocks: (0..TS_BLOCKS)
                .map(|index| TsBlock::load(weights, device, index))
                .collect::<Result<Vec<_>, _>>()?,
            mask: MaskDecoder::load(weights, device)?,
            phase: PhaseDecoder::load(weights, device)?,
        })
    }

    /// What the network predicts for one window: a mask and the two components
    /// of a phase, each `[bins, frames]` in row-major order.
    pub fn predict(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
        device: &B::Device,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let plane = |values: &[f32]| -> Tensor<B, 2> {
            Tensor::from_data(
                burn::tensor::TensorData::new(values.to_vec(), [BINS, frames]),
                device,
            )
            .swap_dims(0, 1)
        };
        let input = Tensor::stack::<3>(vec![plane(magnitude), plane(phase)], 0)
            .reshape([1, 2, frames, BINS]);

        let mut hidden = self.encoder.forward(input);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        let mask = self.mask.forward(hidden.clone());
        let (real, imag) = self.phase.forward(hidden);
        (host(mask), host(real), host(imag))
    }
}

/// A rank-2 tensor as f32 values on the host.
fn host<B: Backend>(x: Tensor<B, 2>) -> Vec<f32> {
    x.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("a stage as f32 values")
}
