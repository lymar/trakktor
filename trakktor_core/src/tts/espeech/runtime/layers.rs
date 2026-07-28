//! Primitives the two networks share, none of which the tree already had:
//! affine-free normalization, the two flavours of GELU the reference uses side
//! by side, Mish, and the ConvNeXt blocks.
//!
//! The GELU distinction is not pedantry. The reference builds the DiT's
//! feed-forward with the `tanh` approximation and every other GELU with the
//! exact error function; they differ by about 1e-3, which is far above the
//! tolerance a parity check runs at.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

/// Normalization over the last axis with no learned scale or shift.
///
/// A struct rather than a function so the unit affine can be cached: candle's
/// fused layer-norm kernel — one launch instead of the six the written-out form
/// takes — wants a scale and a shift, and creating those on every call would
/// cost what the fusion saves. In full precision the fused kernel runs; in half
/// precision the written-out form does, with its statistics explicitly gathered
/// in `f32`, as the reference computes them.
#[derive(Debug)]
pub struct PlainNorm {
    unit: Tensor,
    zero: Tensor,
    eps: f64,
}

impl PlainNorm {
    /// Prepares a norm over a last axis of `size`.
    pub fn new(
        size: usize,
        eps: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Ok(Self {
            unit: Tensor::ones(size, DType::F32, device)?,
            zero: Tensor::zeros(size, DType::F32, device)?,
            eps,
        })
    }

    /// Normalizes `xs` over its last axis.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.dtype() == DType::F32 {
            return candle_nn::ops::layer_norm(
                xs,
                &self.unit,
                &self.zero,
                self.eps as f32,
            );
        }
        layer_norm_slow(xs, self.eps)
    }
}

/// The written-out normalization, for inputs the fused kernel is not used on.
fn layer_norm_slow(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered
        .broadcast_div(&(variance + eps)?.sqrt()?)?
        .to_dtype(dtype)
}

/// Normalization over the last axis with a learned scale and shift.
#[derive(Debug)]
pub struct AffineNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl AffineNorm {
    /// Loads the scale and shift vectors.
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps,
        })
    }

    /// Normalizes `xs` over its last axis and applies the scale and shift.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The fused kernel folds the affine in as well; the fallback keeps the
        // half-precision path on reference semantics (statistics in f32).
        if xs.dtype() == DType::F32 && self.weight.dtype() == DType::F32 {
            return candle_nn::ops::layer_norm(
                xs,
                &self.weight,
                &self.bias,
                self.eps as f32,
            );
        }
        layer_norm_slow(xs, self.eps)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

/// The Mish activation, `x · tanh(softplus(x))`.
///
/// `softplus` is evaluated in the form that does not overflow for large inputs,
/// which is what torch's own implementation does.
pub fn mish(xs: &Tensor) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs32 = xs.to_dtype(DType::F32)?;
    let zeros = xs32.zeros_like()?;
    let positive = xs32.maximum(&zeros)?;
    let softplus =
        (positive + xs32.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
    (xs32 * softplus.tanh()?)?.to_dtype(dtype)
}

/// A depthwise-then-pointwise ConvNeXt block, in the two shapes the reference
/// uses.
///
/// The difference between them is the global response normalization the V2
/// block adds and the layer scale the vocoder's block adds instead; everything
/// else — a depthwise convolution over time, a normalization, and a two-layer
/// pointwise mixer — is the same.
#[derive(Debug)]
pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: AffineNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    /// The V2 block's global response normalization.
    grn: Option<Grn>,
    /// The vocoder block's per-channel scale on the residual branch.
    gamma: Option<Tensor>,
    /// Whether the pointwise mixer's activation is the exact GELU (the
    /// reference's default) rather than the `tanh` approximation.
    exact_gelu: bool,
}

impl ConvNeXtBlock {
    /// Loads a block with global response normalization (the text encoder's).
    pub fn load_v2(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: depthwise(dim, 7, vb.pp("dwconv"))?,
            norm: AffineNorm::load(dim, 1e-6, vb.pp("norm"))?,
            pwconv1: candle_nn::linear(dim, inner, vb.pp("pwconv1"))?,
            pwconv2: candle_nn::linear(inner, dim, vb.pp("pwconv2"))?,
            grn: Some(Grn::load(inner, vb.pp("grn"))?),
            gamma: None,
            exact_gelu: true,
        })
    }

    /// Loads a block with a layer scale (the vocoder's).
    pub fn load_scaled(
        dim: usize,
        inner: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            dwconv: depthwise(dim, 7, vb.pp("dwconv"))?,
            norm: AffineNorm::load(dim, 1e-6, vb.pp("norm"))?,
            pwconv1: candle_nn::linear(dim, inner, vb.pp("pwconv1"))?,
            pwconv2: candle_nn::linear(inner, dim, vb.pp("pwconv2"))?,
            grn: None,
            gamma: Some(vb.get(dim, "gamma")?),
            exact_gelu: true,
        })
    }

    /// Applies the block to `xs`, shaped `[batch, time, channels]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.dwconv.forward(&xs.transpose(1, 2)?.contiguous()?)?;
        let hidden =
            self.norm.forward(&hidden.transpose(1, 2)?.contiguous()?)?;
        let hidden = self.pwconv1.forward(&hidden)?;
        let hidden = if self.exact_gelu {
            hidden.gelu_erf()?
        } else {
            hidden.gelu()?
        };
        let hidden = match &self.grn {
            Some(grn) => grn.forward(&hidden)?,
            None => hidden,
        };
        let hidden = self.pwconv2.forward(&hidden)?;
        let hidden = match &self.gamma {
            Some(gamma) => hidden.broadcast_mul(gamma)?,
            None => hidden,
        };
        residual + hidden
    }
}

/// Global response normalization: it rescales each channel by how strong it is
/// across time relative to the other channels, so one channel cannot dominate.
#[derive(Debug)]
pub struct Grn {
    gamma: Tensor,
    beta: Tensor,
}

/// The guard the reference adds to the divisor.
const GRN_EPS: f64 = 1e-6;

impl Grn {
    /// Loads the per-channel scale and shift.
    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // Both are stored with two leading singleton axes.
            gamma: vb.get((1, 1, channels), "gamma")?.reshape(channels)?,
            beta: vb.get((1, 1, channels), "beta")?.reshape(channels)?,
        })
    }

    /// Applies the normalization to `xs`, shaped `[batch, time, channels]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs32 = xs.to_dtype(DType::F32)?;
        // The channel norm runs over time, not over channels.
        let energy = xs32.sqr()?.sum_keepdim(1)?.sqrt()?;
        let relative = energy
            .broadcast_div(&(energy.mean_keepdim(D::Minus1)? + GRN_EPS)?)?;
        let scaled = xs32.broadcast_mul(&relative)?;
        let scaled = scaled
            .broadcast_mul(&self.gamma.to_dtype(DType::F32)?)?
            .broadcast_add(&self.beta.to_dtype(DType::F32)?)?;
        (scaled + xs32)?.to_dtype(dtype)
    }
}

/// A depthwise convolution over time, padded to keep the length.
pub fn depthwise(
    channels: usize,
    kernel: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding: kernel / 2,
        groups: channels,
        ..Default::default()
    };
    let weight = vb.get((channels, 1, kernel), "weight")?;
    let bias = vb.get(channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// A grouped convolution over time, padded to keep the length.
pub fn grouped(
    channels: usize,
    kernel: usize,
    groups: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding: kernel / 2,
        groups,
        ..Default::default()
    };
    let weight = vb.get((channels, channels / groups, kernel), "weight")?;
    let bias = vb.get(channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// A plain convolution over time, padded to keep the length.
pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding: kernel / 2,
        ..Default::default()
    };
    let weight = vb.get((out_channels, in_channels, kernel), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}
