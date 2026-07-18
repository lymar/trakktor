//! The GigaAM Conformer encoder and CTC head on burn.
//!
//! A port of the candle network in `runtime::net` with the same numerical
//! semantics, written idiomatically for burn rather than as a mirror: linear
//! layers and attention use burn's dedicated primitives (`module::linear`,
//! fused scaled-dot-product `module::attention`), and the depthwise
//! convolution is a native grouped `conv1d` — burn's convolution kernels do
//! not suffer candle's one-launch-per-group pathology, so the shifted-sum
//! workaround of the candle runtime is unnecessary (and slower) here. Layer
//! normalization keeps its statistics in `f32` (as candle does), and rotary
//! tables are computed on the host with the reference formula.
//!
//! Weight tensors are built directly from the checkpoint's `state_dict` —
//! there are no burn records involved.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.

use std::collections::HashMap;

use burn::tensor::{
    Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, conv1d, linear},
    ops::{AttentionModuleOptions, ConvOptions},
};
use candle_core::Tensor as CandleTensor;

use crate::asr::gigaam::{
    config::{Attention, ConvNorm, EncoderConfig, Subsampling},
    error::GigaamError,
    runtime::{model_err, tensor_to_f32_parts},
};

/// The checkpoint's tensors, keyed by `state_dict` name.
pub(super) type StateDict = HashMap<String, CandleTensor>;

/// Reads a checkpoint tensor of the given shape as a burn tensor (in the
/// backend's compute dtype).
fn weight<B: Backend, const D: usize>(
    map: &StateDict,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, GigaamError> {
    let (values, dims) = tensor_to_f32_parts(map, key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Reads a `[out, in]`-shaped checkpoint weight transposed into burn's
/// `[in, out]` linear layout. `trailing_one` accepts the `[out, in, 1]` shape
/// of a 1×1 convolution, which is the same per-frame linear map.
fn weight_transposed<B: Backend>(
    map: &StateDict,
    device: &B::Device,
    key: &str,
    out_dim: usize,
    in_dim: usize,
    trailing_one: bool,
) -> Result<Tensor<B, 2>, GigaamError> {
    let (values, dims) = tensor_to_f32_parts(map, key)?;
    let expected: &[usize] = if trailing_one {
        &[out_dim, in_dim, 1]
    } else {
        &[out_dim, in_dim]
    };
    if dims != expected {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {expected:?}"),
        ));
    }
    let mut transposed = vec![0.0f32; values.len()];
    for o in 0..out_dim {
        for i in 0..in_dim {
            transposed[i * out_dim + o] = values[o * in_dim + i];
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(transposed, [in_dim, out_dim]),
        device,
    ))
}

/// A linear layer with candle/PyTorch semantics: `y = x·Wᵀ + b`. The
/// checkpoint's `[out, in]` weight is transposed once at load into burn's
/// `[in, out]` layout, and the forward pass is burn's `linear` primitive.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [in, out]
    bias: Tensor<B, 1>,   // [out]
}

impl<B: Backend> Linear<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, GigaamError> {
        Self::load_shaped(map, device, prefix, in_dim, out_dim, false)
    }

    /// Loads a 1×1 `Conv1d` checkpoint weight (`[out, in, 1]`) as a linear —
    /// the identical per-frame map, without the layout shuffling of a
    /// convolution over `[B, C, T]`.
    fn load_conv1x1(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, GigaamError> {
        Self::load_shaped(map, device, prefix, in_dim, out_dim, true)
    }

    fn load_shaped(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        trailing_one: bool,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            weight: weight_transposed(
                map,
                device,
                &format!("{prefix}.weight"),
                out_dim,
                in_dim,
                trailing_one,
            )?,
            bias: weight(map, device, &format!("{prefix}.bias"), [out_dim])?,
        })
    }

    /// `[B, T, in]` -> `[B, T, out]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// A 1-D convolution holding its checkpoint weights.
struct Conv<B: Backend> {
    weight: Tensor<B, 3>, // [out, in, k]
    bias: Tensor<B, 1>,   // [out]
    stride: usize,
    padding: usize,
}

impl<B: Backend> Conv<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            weight: weight(
                map,
                device,
                &format!("{prefix}.weight"),
                [out_ch, in_ch, kernel],
            )?,
            bias: weight(map, device, &format!("{prefix}.bias"), [out_ch])?,
            stride,
            padding,
        })
    }

    /// `[B, in, T]` -> `[B, out, T']`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([self.stride], [self.padding], [1], 1),
        )
    }
}

/// Two strided 1-D convolutions with ReLU, subsampling time by 4.
struct Conv1dSubsampling<B: Backend> {
    convs: Vec<Conv<B>>,
}

impl<B: Backend> Conv1dSubsampling<B> {
    fn load(
        cfg: &EncoderConfig,
        map: &StateDict,
        device: &B::Device,
    ) -> Result<Self, GigaamError> {
        let stages = (cfg.subsampling_factor as f64).log2() as usize;
        let padding = (cfg.subs_kernel_size - 1) / 2;
        let mut convs = Vec::with_capacity(stages);
        let mut in_ch = cfg.n_mels;
        for s in 0..stages {
            // Sequential indices: conv at 0, 2, 4, ... (ReLU in between).
            convs.push(Conv::load(
                map,
                device,
                &format!("encoder.pre_encode.conv.{}", s * 2),
                cfg.d_model,
                in_ch,
                cfg.subs_kernel_size,
                2,
                padding,
            )?);
            in_ch = cfg.d_model;
        }
        Ok(Self { convs })
    }

    /// `[B, n_mels, T]` -> `[B, T', d_model]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for conv in &self.convs {
            x = activation::relu(conv.forward(x));
        }
        // [B, d_model, T'] -> [B, T', d_model]
        x.swap_dims(1, 2)
    }
}

/// Rotary position embedding tables, computed on the host in `f32` with the
/// reference formula and uploaded shaped for broadcasting over
/// `[B, T, n_heads, d_head]`.
struct Rotary {
    dim: usize,
    base: f64,
}

impl Rotary {
    fn new(dim: usize, base: f64) -> Self { Self { dim, base } }

    /// Builds `cos`/`sin` of shape `[1, length, 1, dim]` for the given length.
    fn tables<B: Backend>(
        &self,
        length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half = self.dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                1.0 / (self.base.powf((2 * i) as f64 / self.dim as f64)) as f32
            })
            .collect();
        // freqs [length, half]; emb = cat([freqs, freqs]) -> [length, dim].
        let mut cos = Vec::with_capacity(length * self.dim);
        let mut sin = Vec::with_capacity(length * self.dim);
        for t in 0..length {
            for j in 0..self.dim {
                let freq = t as f32 * inv_freq[j % half];
                cos.push(freq.cos());
                sin.push(freq.sin());
            }
        }
        let shape = [1, length, 1, self.dim];
        (
            Tensor::from_data(TensorData::new(cos, shape), device),
            Tensor::from_data(TensorData::new(sin, shape), device),
        )
    }
}

/// `rotate_half`: `[-x2, x1]` where `x1, x2` are the two halves of the last
/// dim.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let d = x.dims()[3];
    let x1 = x.clone().narrow(3, 0, d / 2);
    let x2 = x.narrow(3, d / 2, d / 2);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

/// Applies rotary embeddings to `x` `[B, T, n_heads, d_head]`.
fn apply_rotary<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    x.clone() * cos.clone() + rotate_half(x) * sin.clone()
}

/// Rotary multi-head self-attention. The reference applies rotary to the block
/// input *before* the q/k/v projections; q and k therefore share the same
/// rotated input, v uses the raw input. The attended context is computed by
/// burn's fused scaled-dot-product `attention` (its default scale is the
/// reference's `1/sqrt(d_head)`).
struct RotaryAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> RotaryAttention<B> {
    fn load(
        cfg: &EncoderConfig,
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, GigaamError> {
        let d = cfg.d_model;
        let lin = |name: &str| -> Result<Linear<B>, GigaamError> {
            Linear::load(map, device, &format!("{prefix}.{name}"), d, d)
        };
        Ok(Self {
            q: lin("linear_q")?,
            k: lin("linear_k")?,
            v: lin("linear_v")?,
            out: lin("linear_out")?,
            n_heads: cfg.n_heads,
            d_head: cfg.d_head(),
        })
    }

    /// `[B, T, D]` -> `[B, n_heads, T, d_head]`.
    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, t, _] = x.dims();
        x.reshape([b, t, self.n_heads, self.d_head]).swap_dims(1, 2)
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        // Rotary is applied per-head to the input, then flattened back.
        let x_heads = x.clone().reshape([b, t, self.n_heads, self.d_head]);
        let rotated = apply_rotary(x_heads, cos, sin).reshape([b, t, d]);

        let q = self.split_heads(self.q.forward(rotated.clone()));
        let k = self.split_heads(self.k.forward(rotated));
        let v = self.split_heads(self.v.forward(x));

        let out =
            attention(q, k, v, None, None, AttentionModuleOptions::default());
        let out = out.swap_dims(1, 2).reshape([b, t, d]);
        self.out.forward(out)
    }
}

/// Position-wise feed-forward: `linear2(silu(linear1(x)))`.
struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            linear1: Linear::load(
                map,
                device,
                &format!("{prefix}.linear1"),
                d_model,
                d_ff,
            )?,
            linear2: Linear::load(
                map,
                device,
                &format!("{prefix}.linear2"),
                d_ff,
                d_model,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear2
            .forward(activation::silu(self.linear1.forward(x)))
    }
}

/// Layer normalization over the last dim, mirroring candle's numerics: the
/// statistics are computed in `f32` (also for `f16` models), the result is
/// cast back to the compute dtype, then scaled and shifted.
pub(super) struct LayerNorm<B: Backend> {
    pub(super) gamma: Tensor<B, 3>, // [1, 1, d]
    pub(super) beta: Tensor<B, 3>,  // [1, 1, d]
    pub(super) eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        d: usize,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            gamma: weight::<B, 1>(
                map,
                device,
                &format!("{prefix}.weight"),
                [d],
            )?
            .reshape([1, 1, d]),
            beta: weight::<B, 1>(map, device, &format!("{prefix}.bias"), [d])?
                .reshape([1, 1, d]),
            eps: 1e-5,
        })
    }

    /// `[B, T, d]` -> `[B, T, d]`, normalized over `d`.
    pub(super) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = x.dtype();
        let x = x.cast(burn::tensor::DType::F32);
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = (centered.clone() * centered.clone()).mean_dim(2);
        let normed = centered / (var.add_scalar(self.eps)).sqrt();
        normed.cast(dtype) * self.gamma.clone() + self.beta.clone()
    }
}

/// Inference-mode batch normalization over channels of `[B, C, T]`.
struct BatchNorm<B: Backend> {
    mean: Tensor<B, 3>,   // [1, C, 1]
    var: Tensor<B, 3>,    // [1, C, 1]
    weight: Tensor<B, 3>, // [1, C, 1]
    bias: Tensor<B, 3>,   // [1, C, 1]
    eps: f64,
}

impl<B: Backend> BatchNorm<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        c: usize,
    ) -> Result<Self, GigaamError> {
        let get = |name: &str| -> Result<Tensor<B, 3>, GigaamError> {
            Ok(
                weight::<B, 1>(map, device, &format!("{prefix}.{name}"), [c])?
                    .reshape([1, c, 1]),
            )
        };
        Ok(Self {
            mean: get("running_mean")?,
            var: get("running_var")?,
            weight: get("weight")?,
            bias: get("bias")?,
            eps: 1e-5,
        })
    }

    /// `[B, C, T]` -> `[B, C, T]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = (x - self.mean.clone()) /
            (self.var.clone().add_scalar(self.eps)).sqrt();
        normed * self.weight.clone() + self.bias.clone()
    }
}

/// Channel normalization inside the convolution module.
enum ConvNormLayer<B: Backend> {
    Batch(BatchNorm<B>),
    Layer(LayerNorm<B>),
}

/// A depthwise 1-D convolution (one independent filter per channel), as a
/// native grouped convolution with one group per channel.
pub(super) struct DepthwiseConv1d<B: Backend> {
    pub(super) weight: Tensor<B, 3>, // [C, 1, K]
    pub(super) bias: Tensor<B, 1>,   // [C]
    pub(super) padding: usize,
}

impl<B: Backend> DepthwiseConv1d<B> {
    fn load(
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
        channels: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            weight: weight(
                map,
                device,
                &format!("{prefix}.weight"),
                [channels, 1, kernel_size],
            )?,
            bias: weight(map, device, &format!("{prefix}.bias"), [channels])?,
            padding,
        })
    }

    /// `[B, C, T]` -> `[B, C, T]` (padding keeps the length for the odd kernels
    /// used here).
    pub(super) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let channels = self.weight.dims()[0];
        conv1d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [self.padding], [1], channels),
        )
    }
}

/// Conformer convolution module: pointwise → GLU → depthwise → norm → SiLU →
/// pointwise. The pointwise 1×1 convolutions are per-frame linear maps and run
/// as `linear` on `[B, T, D]`, so only the depthwise convolution (and the
/// batch-norm variant) sees the `[B, D, T]` layout.
struct ConformerConv<B: Backend> {
    pointwise1: Linear<B>,
    depthwise: DepthwiseConv1d<B>,
    norm: ConvNormLayer<B>,
    pointwise2: Linear<B>,
}

impl<B: Backend> ConformerConv<B> {
    fn load(
        cfg: &EncoderConfig,
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, GigaamError> {
        let d = cfg.d_model;
        let pointwise1 = Linear::load_conv1x1(
            map,
            device,
            &format!("{prefix}.pointwise_conv1"),
            d,
            d * 2,
        )?;
        let depthwise = DepthwiseConv1d::load(
            map,
            device,
            &format!("{prefix}.depthwise_conv"),
            d,
            cfg.conv_kernel_size,
            (cfg.conv_kernel_size - 1) / 2,
        )?;
        // The attribute is named `batch_norm` even for the LayerNorm variant.
        let norm = match cfg.conv_norm {
            ConvNorm::BatchNorm => ConvNormLayer::Batch(BatchNorm::load(
                map,
                device,
                &format!("{prefix}.batch_norm"),
                d,
            )?),
            ConvNorm::LayerNorm => ConvNormLayer::Layer(LayerNorm::load(
                map,
                device,
                &format!("{prefix}.batch_norm"),
                d,
            )?),
        };
        let pointwise2 = Linear::load_conv1x1(
            map,
            device,
            &format!("{prefix}.pointwise_conv2"),
            d,
            d,
        )?;
        Ok(Self {
            pointwise1,
            depthwise,
            norm,
            pointwise2,
        })
    }

    /// `[B, T, D]` -> `[B, T, D]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.pointwise1.forward(x); // [B, T, 2D]
        let x = activation::glu(x, 2); // [B, T, D]
        let x = self.depthwise.forward(x.swap_dims(1, 2)); // [B, D, T]
        let x = match &self.norm {
            ConvNormLayer::Batch(bn) => bn.forward(x).swap_dims(1, 2),
            // LayerNorm normalizes over channels, i.e. on [B, T, D].
            ConvNormLayer::Layer(ln) => ln.forward(x.swap_dims(1, 2)),
        };
        let x = activation::silu(x);
        self.pointwise2.forward(x)
    }
}

/// One Conformer block.
struct ConformerLayer<B: Backend> {
    norm_ff1: LayerNorm<B>,
    ff1: FeedForward<B>,
    norm_conv: LayerNorm<B>,
    conv: ConformerConv<B>,
    norm_attn: LayerNorm<B>,
    attn: RotaryAttention<B>,
    norm_ff2: LayerNorm<B>,
    ff2: FeedForward<B>,
    norm_out: LayerNorm<B>,
}

impl<B: Backend> ConformerLayer<B> {
    fn load(
        cfg: &EncoderConfig,
        map: &StateDict,
        device: &B::Device,
        prefix: &str,
    ) -> Result<Self, GigaamError> {
        let d = cfg.d_model;
        let d_ff = d * 4;
        let norm = |name: &str| -> Result<LayerNorm<B>, GigaamError> {
            LayerNorm::load(map, device, &format!("{prefix}.{name}"), d)
        };
        Ok(Self {
            norm_ff1: norm("norm_feed_forward1")?,
            ff1: FeedForward::load(
                map,
                device,
                &format!("{prefix}.feed_forward1"),
                d,
                d_ff,
            )?,
            norm_conv: norm("norm_conv")?,
            conv: ConformerConv::load(
                cfg,
                map,
                device,
                &format!("{prefix}.conv"),
            )?,
            norm_attn: norm("norm_self_att")?,
            attn: RotaryAttention::load(
                cfg,
                map,
                device,
                &format!("{prefix}.self_attn"),
            )?,
            norm_ff2: norm("norm_feed_forward2")?,
            ff2: FeedForward::load(
                map,
                device,
                &format!("{prefix}.feed_forward2"),
                d,
                d_ff,
            )?,
            norm_out: norm("norm_out")?,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        // FF1 (half-step).
        let ff = self.ff1.forward(self.norm_ff1.forward(x.clone()));
        let x = x + ff.mul_scalar(0.5);
        // Self-attention.
        let attn =
            self.attn
                .forward(self.norm_attn.forward(x.clone()), cos, sin);
        let x = x + attn;
        // Convolution.
        let conv = self.conv.forward(self.norm_conv.forward(x.clone()));
        let x = x + conv;
        // FF2 (half-step).
        let ff = self.ff2.forward(self.norm_ff2.forward(x.clone()));
        let x = x + ff.mul_scalar(0.5);
        self.norm_out.forward(x)
    }
}

/// The Conformer encoder.
pub struct ConformerEncoder<B: Backend> {
    subsampling: Conv1dSubsampling<B>,
    rotary: Rotary,
    layers: Vec<ConformerLayer<B>>,
}

impl<B: Backend> ConformerEncoder<B> {
    pub fn load(
        cfg: &EncoderConfig,
        map: &StateDict,
        device: &B::Device,
    ) -> Result<Self, GigaamError> {
        assert_eq!(
            cfg.subsampling,
            Subsampling::Conv1d,
            "only conv1d subsampling is implemented"
        );
        assert_eq!(
            cfg.attention,
            Attention::Rotary,
            "only rotary attention is implemented"
        );
        let subsampling = Conv1dSubsampling::load(cfg, map, device)?;
        let rotary = Rotary::new(cfg.d_head(), 5000.0);
        let layers = (0..cfg.n_layers)
            .map(|i| {
                ConformerLayer::load(
                    cfg,
                    map,
                    device,
                    &format!("encoder.layers.{i}"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            subsampling,
            rotary,
            layers,
        })
    }

    /// `[B, n_mels, T]` log-mel features -> `[B, T', d_model]` encoded output.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = self.subsampling.forward(mel); // [B, T', D]
        let seq_len = x.dims()[1];
        let (cos, sin) = self.rotary.tables::<B>(seq_len, &x.device());
        for layer in &self.layers {
            x = layer.forward(x, &cos, &sin);
        }
        x
    }
}

/// CTC head: a 1×1 convolution over channels, i.e. a per-frame linear map to
/// class scores. Decoding argmaxes raw logits (the softmax normalization
/// cannot change the argmax), so no log-softmax is applied.
pub struct CtcHead<B: Backend> {
    head: Linear<B>,
}

impl<B: Backend> CtcHead<B> {
    pub fn load(
        d_model: usize,
        num_classes: usize,
        map: &StateDict,
        device: &B::Device,
    ) -> Result<Self, GigaamError> {
        Ok(Self {
            head: Linear::load_conv1x1(
                map,
                device,
                "head.decoder_layers.0",
                d_model,
                num_classes,
            )?,
        })
    }

    /// `[B, T', d_model]` -> `[B, T', num_classes]` logits.
    pub fn logits(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.head.forward(x)
    }
}
