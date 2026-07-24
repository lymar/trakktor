//! Primitives shared by the networks of this engine, none of which the rest of
//! the tree already had: RMS normalization, the causal convolutions the codec
//! is built from, the Snake activation, and layer scaling.
//!
//! Every one of them follows the reference elementwise, including where it
//! forces a computation into full precision — normalization statistics are
//! gathered in `f32` even when the weights are half precision, exactly as
//! upstream does, so a half-precision run does not drift differently.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module,
    VarBuilder,
};

/// Root-mean-square normalization with a learned per-channel gain.
///
/// Normalizes over the last axis. The statistics are computed in `f32` and the
/// result cast back before the gain is applied, matching the reference.
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Loads the gain vector.
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    /// Normalizes `xs` over its last axis and applies the gain.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// A 1-D convolution padded so the output never depends on future samples.
///
/// The reference pads the input on the left by the dilated receptive field
/// minus the stride; with the unit strides the decoder uses, that is exactly
/// `(kernel - 1) * dilation` and no padding is needed on the right.
#[derive(Debug)]
pub struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    /// Loads a causal convolution. `groups` of `channels` makes it depthwise.
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups,
            ..Default::default()
        };
        let weight = vb.get(
            (out_channels, in_channels / groups, kernel_size),
            "conv.weight",
        )?;
        let bias = vb.get(out_channels, "conv.bias")?;
        Ok(Self {
            conv: Conv1d::new(weight, Some(bias), cfg),
            left_pad: (kernel_size - 1) * dilation,
        })
    }

    /// Convolves `xs`, shaped `[batch, channels, time]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padded = xs.pad_with_zeros(D::Minus1, self.left_pad, 0)?;
        self.conv.forward(&padded)
    }
}

/// A transposed 1-D convolution trimmed on the right so it stays causal.
///
/// The reference drops `kernel - stride` trailing samples; when the kernel
/// equals the stride nothing is trimmed and the layer is a plain upsample.
#[derive(Debug)]
pub struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    right_trim: usize,
}

impl CausalConvTranspose1d {
    /// Loads a transposed convolution with the given kernel and stride.
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        // Transposed convolutions store weights as [in, out, kernel].
        let weight =
            vb.get((in_channels, out_channels, kernel_size), "conv.weight")?;
        let bias = vb.get(out_channels, "conv.bias")?;
        Ok(Self {
            conv: ConvTranspose1d::new(weight, Some(bias), cfg),
            right_trim: kernel_size - stride,
        })
    }

    /// Upsamples `xs`, shaped `[batch, channels, time]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        if self.right_trim == 0 {
            return Ok(out);
        }
        let keep = out.dim(D::Minus1)? - self.right_trim;
        out.narrow(D::Minus1, 0, keep)
    }
}

/// The Snake activation with separate frequency and magnitude parameters:
/// `x + sin²(x·eᵃ) / eᵇ`.
///
/// Both parameters are stored as logarithms and exponentiated here, and the
/// division is guarded exactly as upstream guards it.
#[derive(Debug)]
pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

/// The guard the reference adds to the divisor.
const SNAKE_EPS: f64 = 1e-9;

impl SnakeBeta {
    /// Loads the per-channel parameters.
    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(channels, "alpha")?,
            beta: vb.get(channels, "beta")?,
        })
    }

    /// Applies the activation to `xs`, shaped `[batch, channels, time]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Line the per-channel parameters up with [batch, channels, time].
        let alpha = self.alpha.exp()?.reshape((1, (), 1))?;
        let beta = self.beta.exp()?.reshape((1, (), 1))?;
        let sin = xs.broadcast_mul(&alpha)?.sin()?;
        let scaled = sin.sqr()?.broadcast_div(&(beta + SNAKE_EPS)?)?;
        xs + scaled
    }
}

/// A learned per-channel scale on a residual branch, holding the branch near
/// zero at initialization.
#[derive(Debug)]
pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    /// Loads the scale vector.
    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get(channels, "scale")?,
        })
    }

    /// Scales `xs` along its last axis.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

/// Builds the cosine and sine tables for rotary position embeddings.
///
/// Returns two `[positions, dim / 2]` tensors; callers decide how to lay the
/// halves out, which differs between the codec (plain rotary) and the talker
/// (sectioned).
pub fn rope_tables(
    dim: usize,
    positions: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let inv_freq: Vec<f32> = (0..dim / 2)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?;
    let steps: Vec<f32> = (0..positions).map(|p| p as f32).collect();
    let steps = Tensor::from_vec(steps, (positions, 1), device)?;
    let freqs = steps.broadcast_mul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Rotates the halves of the last axis: `[a, b] → [-b, a]`.
pub fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    let first = xs.narrow(D::Minus1, 0, half)?;
    let second = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[second.neg()?, first], D::Minus1)
}

#[cfg(test)]
mod tests;
