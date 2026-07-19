//! The GigaAM Conformer encoder and CTC head on candle.
//!
//! A faithful port of `gigaam.encoder.ConformerEncoder` and the CTC head.
//! Geometry (channel width, layer count, subsampling and normalization
//! kinds, attention kind) is passed in as [`EncoderConfig`], so one
//! implementation serves every published checkpoint. The RNN-T head is not
//! here: it always runs on the CPU, shared across runtimes (see
//! [`rnnt`](crate::asr::gigaam::rnnt)).
//!
//! Weight names follow the checkpoint's `state_dict` layout
//! (`encoder.*`, `head.*`).

use candle_core::{D, Module, ModuleT, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder, conv1d, layer_norm,
    linear,
};

use crate::asr::gigaam::config::{
    Attention, ConvNorm, EncoderConfig, Subsampling,
};

/// Two strided 1-D convolutions with ReLU, subsampling time by 4.
struct Conv1dSubsampling {
    convs: Vec<Conv1d>,
}

impl Conv1dSubsampling {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let stages = (cfg.subsampling_factor as f64).log2() as usize;
        let padding = (cfg.subs_kernel_size - 1) / 2;
        let conv_cfg = Conv1dConfig {
            padding,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let mut convs = Vec::with_capacity(stages);
        let mut in_ch = cfg.n_mels;
        for s in 0..stages {
            // Sequential indices: conv at 0, 2, 4, ... (ReLU in between).
            let conv = conv1d(
                in_ch,
                cfg.d_model,
                cfg.subs_kernel_size,
                conv_cfg,
                vb.pp(format!("conv.{}", s * 2)),
            )?;
            convs.push(conv);
            in_ch = cfg.d_model;
        }
        Ok(Self { convs })
    }

    /// `[B, n_mels, T]` -> `[B, T', d_model]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for conv in &self.convs {
            x = conv.forward(&x)?.relu()?;
        }
        // [B, d_model, T'] -> [B, T', d_model]
        x.transpose(1, 2)?.contiguous()
    }
}

/// Rotary position embedding tables (`cos`/`sin`), shaped for broadcasting
/// over `[B, T, n_heads, d_head]`.
struct Rotary {
    dim: usize,
    base: f64,
}

impl Rotary {
    fn new(dim: usize, base: f64) -> Self { Self { dim, base } }

    /// Builds `cos`/`sin` of shape `[1, length, 1, dim]` for the given length.
    fn tables(
        &self,
        length: usize,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = self.dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                1.0 / (self.base.powf((2 * i) as f64 / self.dim as f64)) as f32
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, length as u32, device)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((length, 1))?;
        // freqs [length, half]; emb = cat([freqs, freqs]) -> [length, dim].
        let freqs = t.broadcast_mul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = emb.cos()?.reshape((1, length, 1, self.dim))?;
        let sin = emb.sin()?.reshape((1, length, 1, self.dim))?;
        Ok((cos, sin))
    }
}

/// `rotate_half`: `[-x2, x1]` where `x1, x2` are the two halves of the last
/// dim.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Applies rotary embeddings to `x` `[B, T, n_heads, d_head]`.
fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
}

/// Rotary multi-head self-attention. The reference applies rotary to the block
/// input *before* the q/k/v projections; q and k therefore share the same
/// rotated input, v uses the raw input.
struct RotaryAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_heads: usize,
    d_head: usize,
}

impl RotaryAttention {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q: linear(d, d, vb.pp("linear_q"))?,
            k: linear(d, d, vb.pp("linear_k"))?,
            v: linear(d, d, vb.pp("linear_v"))?,
            out: linear(d, d, vb.pp("linear_out"))?,
            n_heads: cfg.n_heads,
            d_head: cfg.d_head(),
        })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        // [B, T, D] -> [B, n_heads, T, d_head]
        x.reshape((b, t, self.n_heads, self.d_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        // Rotary is applied per-head to the input, then flattened back.
        let x_heads = x.reshape((b, t, self.n_heads, self.d_head))?;
        let rotated = apply_rotary(&x_heads, cos, sin)?.reshape((b, t, d))?;

        let q = self.split_heads(&self.q.forward(&rotated)?)?;
        let k = self.split_heads(&self.k.forward(&rotated)?)?;
        let v = self.split_heads(&self.v.forward(x)?)?;

        let scale = 1.0 / (self.d_head as f64).sqrt();
        let qk = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&qk)?;
        let out = attn.matmul(&v)?; // [B, n_heads, T, d_head]
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        self.out.forward(&out)
    }
}

/// Position-wise feed-forward: `linear2(silu(linear1(x)))`.
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(d_model, d_ff, vb.pp("linear1"))?,
            linear2: linear(d_ff, d_model, vb.pp("linear2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear2.forward(&self.linear1.forward(x)?.silu()?)
    }
}

/// Channel normalization inside the convolution module.
enum ConvNormLayer {
    Batch(candle_nn::BatchNorm),
    Layer(LayerNorm),
}

/// A depthwise 1-D convolution (one independent filter per channel).
///
/// candle's grouped `conv1d` splits the input into `groups` slices and runs a
/// separate convolution per group; with one group per channel that is hundreds
/// of tiny kernel launches per layer, which dominates the encoder on Metal.
/// A depthwise convolution is instead a small sum of shifted,
/// per-channel-scaled copies of the (padded) input — `kernel_size` broadcast
/// multiplies rather than `channels` convolutions — so it runs as a handful of
/// elementwise ops.
struct DepthwiseConv1d {
    /// Per-tap channel weights, `kernel_size` tensors of shape `[1, C, 1]`.
    taps: Vec<Tensor>,
    bias: Tensor, // [1, C, 1]
    padding: usize,
}

impl DepthwiseConv1d {
    fn load(
        channels: usize,
        kernel_size: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Checkpoint weight: [C, 1, K].
        let weight = vb.get((channels, 1, kernel_size), "weight")?;
        let bias = vb.get(channels, "bias")?.reshape((1, channels, 1))?;
        // Split into K per-tap channel vectors, each broadcastable over
        // [B,C,T].
        let mut taps = Vec::with_capacity(kernel_size);
        for k in 0..kernel_size {
            let tap = weight
                .narrow(2, k, 1)? // [C, 1, 1]
                .reshape((1, channels, 1))?
                .contiguous()?;
            taps.push(tap);
        }
        Ok(Self {
            taps,
            bias,
            padding,
        })
    }

    /// `[B, C, T]` -> `[B, C, T]` (padding keeps the length for the odd kernels
    /// used here).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t) = x.dims3()?;
        // Zero-pad both ends along time so every tap window stays in range.
        let x = x.pad_with_zeros(2, self.padding, self.padding)?;
        let mut acc: Option<Tensor> = None;
        for (k, tap) in self.taps.iter().enumerate() {
            let window = x.narrow(2, k, t)?; // [B, C, T]
            let term = window.broadcast_mul(tap)?;
            acc = Some(match acc {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        acc.expect("kernel_size >= 1").broadcast_add(&self.bias)
    }
}

/// Conformer convolution module: pointwise → GLU → depthwise → norm → SiLU →
/// pointwise.
struct ConformerConv {
    pointwise1: Conv1d,
    depthwise: DepthwiseConv1d,
    norm: ConvNormLayer,
    pointwise2: Conv1d,
}

impl ConformerConv {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let pw = Conv1dConfig::default();
        let pointwise1 = conv1d(d, d * 2, 1, pw, vb.pp("pointwise_conv1"))?;
        let depthwise = DepthwiseConv1d::load(
            d,
            cfg.conv_kernel_size,
            (cfg.conv_kernel_size - 1) / 2,
            vb.pp("depthwise_conv"),
        )?;
        let norm = match cfg.conv_norm {
            ConvNorm::BatchNorm => ConvNormLayer::Batch(candle_nn::batch_norm(
                d,
                1e-5,
                vb.pp("batch_norm"),
            )?),
            // The attribute is named `batch_norm` even for the LayerNorm
            // variant.
            ConvNorm::LayerNorm => {
                ConvNormLayer::Layer(layer_norm(d, 1e-5, vb.pp("batch_norm"))?)
            },
        };
        let pointwise2 = conv1d(d, d, 1, pw, vb.pp("pointwise_conv2"))?;
        Ok(Self {
            pointwise1,
            depthwise,
            norm,
            pointwise2,
        })
    }

    /// `[B, T, D]` -> `[B, T, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // [B, D, T]
        let x = self.pointwise1.forward(&x)?; // [B, 2D, T]
        let x = glu_dim1(&x)?; // [B, D, T]
        let x = self.depthwise.forward(&x)?;
        let x = match &self.norm {
            ConvNormLayer::Batch(bn) => bn.forward_t(&x, false)?,
            ConvNormLayer::Layer(ln) => {
                // LayerNorm over channels: normalize on [B, T, D].
                ln.forward(&x.transpose(1, 2)?.contiguous()?)?
                    .transpose(1, 2)?
                    .contiguous()?
            },
        };
        let x = x.silu()?;
        let x = self.pointwise2.forward(&x)?;
        x.transpose(1, 2)?.contiguous()
    }
}

/// Gated linear unit along the channel dim (dim 1) of `[B, 2D, T]`.
fn glu_dim1(x: &Tensor) -> Result<Tensor> {
    let c = x.dim(1)?;
    let a = x.narrow(1, 0, c / 2)?;
    let b = x.narrow(1, c / 2, c / 2)?;
    a * candle_nn::ops::sigmoid(&b)?
}

/// One Conformer block.
struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_attn: LayerNorm,
    attn: RotaryAttention,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let d_ff = d * 4;
        Ok(Self {
            norm_ff1: layer_norm(d, 1e-5, vb.pp("norm_feed_forward1"))?,
            ff1: FeedForward::load(d, d_ff, vb.pp("feed_forward1"))?,
            norm_conv: layer_norm(d, 1e-5, vb.pp("norm_conv"))?,
            conv: ConformerConv::load(cfg, vb.pp("conv"))?,
            norm_attn: layer_norm(d, 1e-5, vb.pp("norm_self_att"))?,
            attn: RotaryAttention::load(cfg, vb.pp("self_attn"))?,
            norm_ff2: layer_norm(d, 1e-5, vb.pp("norm_feed_forward2"))?,
            ff2: FeedForward::load(d, d_ff, vb.pp("feed_forward2"))?,
            norm_out: layer_norm(d, 1e-5, vb.pp("norm_out"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // FF1 (half-step).
        let ff = self.ff1.forward(&self.norm_ff1.forward(x)?)?;
        let x = (x + (ff * 0.5)?)?;
        // Self-attention.
        let attn = self.attn.forward(&self.norm_attn.forward(&x)?, cos, sin)?;
        let x = (x + attn)?;
        // Convolution.
        let conv = self.conv.forward(&self.norm_conv.forward(&x)?)?;
        let x = (x + conv)?;
        // FF2 (half-step).
        let ff = self.ff2.forward(&self.norm_ff2.forward(&x)?)?;
        let x = (x + (ff * 0.5)?)?;
        self.norm_out.forward(&x)
    }
}

/// The Conformer encoder.
pub struct ConformerEncoder {
    subsampling: Conv1dSubsampling,
    rotary: Rotary,
    layers: Vec<ConformerLayer>,
}

impl ConformerEncoder {
    pub fn load(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
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
        let subsampling = Conv1dSubsampling::load(cfg, vb.pp("pre_encode"))?;
        let rotary = Rotary::new(cfg.d_head(), 5000.0);
        let layers = (0..cfg.n_layers)
            .map(|i| ConformerLayer::load(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling,
            rotary,
            layers,
        })
    }

    /// `[B, n_mels, T]` log-mel features -> `[B, T', d_model]` encoded output.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.subsampling.forward(mel)?; // [B, T', D]
        let seq_len = x.dim(1)?;
        let (cos, sin) = self.rotary.tables(seq_len, x.device())?;
        let cos = cos.to_dtype(x.dtype())?;
        let sin = sin.to_dtype(x.dtype())?;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }
        Ok(x)
    }
}

/// CTC head: a 1×1 convolution over channels, i.e. a per-frame linear map to
/// class log-probabilities. Implemented as a linear over the last dim.
pub struct CtcHead {
    weight: Tensor, // [num_classes, d_model]
    bias: Tensor,   // [num_classes]
}

impl CtcHead {
    pub fn load(
        d_model: usize,
        num_classes: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Checkpoint stores a Conv1d weight [num_classes, d_model, 1].
        let weight =
            vb.get((num_classes, d_model, 1), "decoder_layers.0.weight")?;
        let weight = weight.reshape((num_classes, d_model))?;
        let bias = vb.get(num_classes, "decoder_layers.0.bias")?;
        Ok(Self { weight, bias })
    }

    /// `[B, T', d_model]` -> `[B, T', num_classes]` log-probabilities.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x
            .broadcast_matmul(&self.weight.t()?)?
            .broadcast_add(&self.bias)?;
        candle_nn::ops::log_softmax(&logits, D::Minus1)
    }

    /// `[B, T', d_model]` -> `[B, T', num_classes]` logits without log-softmax
    /// (argmax is unaffected, so decoding can skip the normalization).
    pub fn logits(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.weight.t()?)?
            .broadcast_add(&self.bias)
    }
}
