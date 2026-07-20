//! The Zipformer2 encoder on candle.
//!
//! A faithful port of the icefall Zipformer2 at inference time (training-only
//! modules — balancers, whiteners, dropout — are identities and do not
//! exist): the Conv2d+ConvNeXt subsampling front end, six encoder stacks with
//! per-stack temporal downsampling, the shared attention-weights module
//! consumed by two self-attention passes and the nonlin-attention module,
//! SwooshL/SwooshR feed-forwards, the (chunk-causal) depthwise convolution
//! modules, BiasNorm, learned bypasses, and the final ×2 downsample; the
//! joiner's encoder projection is folded in so the output is already in the
//! joint dimension. Both the full-context (offline) forward and the chunked
//! streaming forward with cached state are implemented; batch is always 1.
//!
//! candle-specific choices:
//!
//! - depthwise convolutions run as a small sum of shifted, per-channel-scaled
//!   copies instead of grouped `conv1d`/`conv2d` (grouped convolutions launch
//!   one kernel per group on this runtime, hundreds per layer);
//! - the relative→absolute shift of the positional scores is the flatten +
//!   offset-narrow + reshape "skew" (a view-shaped copy), not a gather;
//! - BiasNorm reduces in `f32` regardless of the compute dtype (the squared
//!   mean overflows the top of `f16` otherwise), like the layer norms of the
//!   sibling engines.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::Linear;

use crate::asr::vosk::weights::{ModelWeights, StackConfig, ZipformerConfig};

/// Loads a `[out, in]` linear with bias from the canonical map.
fn linear(
    w: &ModelWeights,
    prefix: &str,
    out_dim: usize,
    in_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Linear> {
    let weight = tensor(w, &format!("{prefix}.weight"), device, dtype)?
        .reshape((out_dim, in_dim))?;
    let bias = tensor(w, &format!("{prefix}.bias"), device, dtype)?;
    Ok(Linear::new(weight, Some(bias)))
}

/// Uploads a canonical tensor to the device at the compute dtype.
fn tensor(
    w: &ModelWeights,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let raw = w
        .get_any(name)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let dims = if raw.dims.is_empty() {
        vec![1]
    } else {
        raw.dims.clone()
    };
    Tensor::from_slice(&raw.data, dims, device)?.to_dtype(dtype)
}

/// `SwooshL(x) = softplus(x − 4) − 0.08·x − 0.035`, in the overflow-stable
/// softplus form.
fn swoosh_l(x: &Tensor) -> Result<Tensor> {
    let y = (x - 4.0)?;
    (softplus(&y)? - (x * 0.08)?)? - 0.035
}

/// `SwooshR(x) = softplus(x − 1) − 0.08·x − 0.313261687`.
fn swoosh_r(x: &Tensor) -> Result<Tensor> {
    let y = (x - 1.0)?;
    (softplus(&y)? - (x * 0.08)?)? - 0.313261687
}

/// `softplus(y) = max(y, 0) + log1p(exp(−|y|))`.
fn softplus(y: &Tensor) -> Result<Tensor> {
    let relu = y.relu()?;
    let log1p = (y.abs()?.neg()?.exp()? + 1.0)?.log()?;
    relu + log1p
}

/// BiasNorm: `x · (mean((x − bias)², last) ^ −½ · exp(log_scale))`, the
/// reduction in `f32`.
struct BiasNorm {
    bias: Tensor, // [C], f32
    scale: f64,   // exp(log_scale)
    dtype: DType,
}

impl BiasNorm {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let bias = tensor(w, &format!("{prefix}.bias"), device, DType::F32)?;
        let log_scale = w
            .get_any(&format!("{prefix}.log_scale"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self {
            bias,
            scale: f64::from(log_scale.data[0]).exp(),
            dtype,
        })
    }

    /// `[.., C]` -> `[.., C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let centered = xf.broadcast_sub(&self.bias)?;
        let scales =
            ((centered.sqr()?.mean_keepdim(D::Minus1)?.powf(-0.5)?) *
                self.scale)?
                .to_dtype(self.dtype)?;
        x.broadcast_mul(&scales)
    }
}

/// A learned per-channel bypass: `orig + (x − orig)·scale`.
struct Bypass {
    scale: Tensor, // [C]
}

impl Bypass {
    fn load(
        w: &ModelWeights,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            scale: tensor(w, name, device, dtype)?,
        })
    }

    fn forward(&self, orig: &Tensor, x: &Tensor) -> Result<Tensor> {
        orig + (x - orig)?.broadcast_mul(&self.scale)?
    }
}

/// Feed-forward: `out_proj(SwooshL(in_proj(x)))`.
struct FeedForward {
    in_proj: Linear,
    out_proj: Linear,
}

impl FeedForward {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        dim: usize,
        ff_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            in_proj: linear(
                w,
                &format!("{prefix}.in_proj"),
                ff_dim,
                dim,
                device,
                dtype,
            )?,
            out_proj: linear(
                w,
                &format!("{prefix}.out_proj"),
                dim,
                ff_dim,
                device,
                dtype,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out_proj.forward(&swoosh_l(&self.in_proj.forward(x)?)?)
    }
}

/// The shared attention-weights module: computes per-head attention
/// probabilities `[heads, tgt, src]` from the layer input and the positional
/// embedding, consumed by both self-attention passes and (first head only)
/// the nonlin-attention module.
struct AttentionWeights {
    in_proj: Linear,
    linear_pos: Tensor, // [pos_dim, p·h] — bias-less, pre-transposed
    num_heads: usize,
    query_head_dim: usize,
    pos_head_dim: usize,
}

impl AttentionWeights {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        stack: &StackConfig,
        pos_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (h, q, p) =
            (stack.num_heads, stack.query_head_dim, stack.pos_head_dim);
        let d = stack.encoder_dim;
        let linear_pos =
            tensor(w, &format!("{prefix}.linear_pos.weight"), device, dtype)?
                .reshape((p * h, pos_dim))?
                .t()?
                .contiguous()?;
        Ok(Self {
            in_proj: linear(
                w,
                &format!("{prefix}.in_proj"),
                (2 * q + p) * h,
                d,
                device,
                dtype,
            )?,
            linear_pos,
            num_heads: h,
            query_head_dim: q,
            pos_head_dim: p,
        })
    }

    /// `x`: `[T, C]`; `pos_emb`: `[pos_len, pos_dim]` with
    /// `pos_len = left + 2·T − 1`; `cached_key`: streaming left-context keys
    /// `[left, q·h]` (empty offline). `mask_bias`: optional `[src]` additive
    /// mask. Returns the attention weights `[h, T, src]`
    /// (`src = left + T`) and the updated key cache.
    fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        cached_key: Option<&Tensor>,
        mask_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (t, _) = x.dims2()?;
        let (h, q, p) =
            (self.num_heads, self.query_head_dim, self.pos_head_dim);
        let projected = self.in_proj.forward(x)?; // [T, (2q+p)·h]
        let queries = projected.narrow(1, 0, q * h)?;
        let keys_new = projected.narrow(1, q * h, q * h)?.contiguous()?;
        let pos_q = projected.narrow(1, 2 * q * h, p * h)?;

        // Streaming: prepend the cached left-context keys, keep the tail.
        let (keys, new_cache) = match cached_key {
            None => (keys_new, None),
            Some(cache) => {
                let left = cache.dim(0)?;
                let full = Tensor::cat(&[cache, &keys_new], 0)?;
                let tail =
                    full.narrow(0, full.dim(0)? - left, left)?.contiguous()?;
                (full, Some(tail))
            },
        };
        let src = keys.dim(0)?;

        // [T, h, q] -> [h, T, q]; keys transposed for the matmul.
        let qh = queries
            .reshape((t, h, q))?
            .permute((1, 0, 2))?
            .contiguous()?;
        let kh = keys
            .reshape((src, h, q))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let mut scores = qh.matmul(&kh)?; // [h, T, src]

        // Positional scores: project the positional embedding, dot with the
        // positional queries, then shift relative → absolute.
        let pos_len = pos_emb.dim(0)?;
        let pos = pos_emb
            .matmul(&self.linear_pos)? // [pos_len, p·h]
            .reshape((pos_len, h, p))?
            .permute((1, 2, 0))?
            .contiguous()?; // [h, p, pos_len]
        let ph = pos_q.reshape((t, h, p))?.permute((1, 0, 2))?.contiguous()?; // [h, T, p]
        let pos_scores = ph.matmul(&pos)?; // [h, T, pos_len]
        let pos_scores = rel_shift(&pos_scores, src)?;
        scores = (scores + pos_scores)?;

        if let Some(bias) = mask_bias {
            scores = scores.broadcast_add(&bias.reshape((1, 1, src))?)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        Ok((weights, new_cache))
    }
}

/// The relative→absolute "skew": `out[h, i, j] = in[h, i, (T−1) − i + j]`
/// for an input `[h, T, n]` (`n = left + 2T − 1`), producing `[h, T, src]`.
/// Flatten the last two axes, drop the first `T−1` elements, reshape to rows
/// of `n − 1`, and take the first `src` columns.
fn rel_shift(pos_scores: &Tensor, src: usize) -> Result<Tensor> {
    let (h, t, n) = pos_scores.dims3()?;
    debug_assert!(src < n || t == 1);
    if t == 1 {
        // One query row: the slice is direct (and n − 1 may be < src).
        return pos_scores.narrow(2, 0, src);
    }
    let flat = pos_scores.reshape((h, t * n))?;
    let shifted = flat.narrow(1, t - 1, t * (n - 1))?;
    shifted.reshape((h, t, n - 1))?.narrow(2, 0, src)
}

/// Simple attention over precomputed weights.
struct SelfAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    value_head_dim: usize,
}

impl SelfAttention {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        stack: &StackConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (h, v, d) =
            (stack.num_heads, stack.value_head_dim, stack.encoder_dim);
        Ok(Self {
            in_proj: linear(
                w,
                &format!("{prefix}.in_proj"),
                h * v,
                d,
                device,
                dtype,
            )?,
            out_proj: linear(
                w,
                &format!("{prefix}.out_proj"),
                d,
                h * v,
                device,
                dtype,
            )?,
            num_heads: h,
            value_head_dim: v,
        })
    }

    /// `x`: `[T, C]`; `attn`: `[h, T, src]`; `cached_val`: streaming left
    /// values `[left, h·v]`. Returns `[T, C]` and the updated cache.
    fn forward(
        &self,
        x: &Tensor,
        attn: &Tensor,
        cached_val: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (t, _) = x.dims2()?;
        let (h, v) = (self.num_heads, self.value_head_dim);
        let values_new = self.in_proj.forward(x)?; // [T, h·v]
        let (values, new_cache) = match cached_val {
            None => (values_new, None),
            Some(cache) => {
                let left = cache.dim(0)?;
                let full = Tensor::cat(&[cache, &values_new], 0)?;
                let tail =
                    full.narrow(0, full.dim(0)? - left, left)?.contiguous()?;
                (full, Some(tail))
            },
        };
        let src = values.dim(0)?;
        let vh = values
            .reshape((src, h, v))?
            .permute((1, 0, 2))?
            .contiguous()?; // [h, src, v]
        let out = attn.matmul(&vh)?; // [h, T, v]
        let out = out.permute((1, 0, 2))?.reshape((t, h * v))?;
        Ok((self.out_proj.forward(&out)?, new_cache))
    }
}

/// The nonlin-attention module: a gated hidden state attended with the first
/// head's weights.
struct NonlinAttention {
    in_proj: Linear,
    out_proj: Linear,
    hidden: usize,
}

impl NonlinAttention {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        stack: &StackConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (hidden, d) = (stack.nonlin_hidden, stack.encoder_dim);
        Ok(Self {
            in_proj: linear(
                w,
                &format!("{prefix}.in_proj"),
                3 * hidden,
                d,
                device,
                dtype,
            )?,
            out_proj: linear(
                w,
                &format!("{prefix}.out_proj"),
                d,
                hidden,
                device,
                dtype,
            )?,
            hidden,
        })
    }

    /// `x`: `[T, C]`; `attn_head0`: `[1, T, src]`; `cached`: streaming left
    /// hidden values `[left, hidden]`. Returns `[T, C]` and the new cache.
    fn forward(
        &self,
        x: &Tensor,
        attn_head0: &Tensor,
        cached: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let hidden = self.hidden;
        let projected = self.in_proj.forward(x)?; // [T, 3·hidden]
        let s = projected.narrow(1, 0, hidden)?;
        let v = projected.narrow(1, hidden, hidden)?;
        let y = projected.narrow(1, 2 * hidden, hidden)?;
        let gated = (v * s.tanh()?)?; // [T, hidden]

        let (values, new_cache) = match cached {
            None => (gated, None),
            Some(cache) => {
                let left = cache.dim(0)?;
                let full = Tensor::cat(&[cache, &gated], 0)?;
                let tail =
                    full.narrow(0, full.dim(0)? - left, left)?.contiguous()?;
                (full, Some(tail))
            },
        };
        // Single-head attention with the first head's weights.
        let attended = attn_head0
            .matmul(&values.unsqueeze(0)?)? // [1, T, hidden]
            .squeeze(0)?;
        let out = (attended * y)?;
        Ok((self.out_proj.forward(&out)?, new_cache))
    }
}

/// A depthwise 1-D convolution as a sum of shifted, per-channel-scaled
/// copies over `[C, T]` input (see the module docs).
struct DepthwiseTaps {
    /// `kernel_size` per-tap channel columns `[C, 1]`.
    taps: Vec<Tensor>,
    bias: Tensor, // [C, 1]
}

impl DepthwiseTaps {
    fn load(
        w: &ModelWeights,
        weight_name: &str,
        bias_name: &str,
        channels: usize,
        kernel: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let weight = tensor(w, weight_name, device, dtype)?
            .reshape((channels, kernel))?;
        let bias =
            tensor(w, bias_name, device, dtype)?.reshape((channels, 1))?;
        let mut taps = Vec::with_capacity(kernel);
        for k in 0..kernel {
            taps.push(weight.narrow(1, k, 1)?.contiguous()?); // [C, 1]
        }
        Ok(Self { taps, bias })
    }

    /// Convolves `[C, padded_t]` input, producing `[C, out_t]` where
    /// `out_t = padded_t − kernel + 1` (the caller pads).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, padded_t) = x.dims2()?;
        let out_t = padded_t - self.taps.len() + 1;
        let mut acc: Option<Tensor> = None;
        for (k, tap) in self.taps.iter().enumerate() {
            let window = x.narrow(1, k, out_t)?;
            let term = window.broadcast_mul(tap)?;
            acc = Some(match acc {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        acc.expect("kernel >= 1").broadcast_add(&self.bias)
    }
}

/// The depthwise stage of a convolution module: plain symmetric-padded for
/// offline models, chunk-causal (a causal half-kernel plus an edge-scaled
/// chunkwise kernel with a carried cache) for streaming ones.
enum Depthwise {
    Plain {
        conv: DepthwiseTaps,
        pad: usize,
    },
    Causal {
        causal: DepthwiseTaps,
        chunkwise: DepthwiseTaps,
        /// Precomputed `1 + left_edge + right_edge` for the fixed chunk
        /// length, `[C, chunk]`.
        chunk_scale: Tensor,
        left_pad: usize,
    },
}

impl Depthwise {
    /// Offline forward over `[C, T]` (same length out).
    fn forward_plain(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Depthwise::Plain { conv, pad } => {
                let padded = x.pad_with_zeros(1, *pad, *pad)?;
                conv.forward(&padded)
            },
            Depthwise::Causal { .. } => unreachable!("causal model offline"),
        }
    }

    /// Streaming forward over `[C, chunk]` with the `[C, left_pad]` cache.
    fn forward_causal(
        &self,
        x: &Tensor,
        cache: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let Depthwise::Causal {
            causal,
            chunkwise,
            chunk_scale,
            left_pad,
        } = self
        else {
            unreachable!("offline model streaming");
        };
        let full = Tensor::cat(&[cache, x], 1)?; // [C, left_pad + chunk]
        let new_cache = full
            .narrow(1, full.dim(1)? - left_pad, *left_pad)?
            .contiguous()?;
        // The causal half-kernel consumes the cache as its left padding.
        let x_causal = causal.forward(&full)?; // [C, chunk]
        // The chunkwise kernel sees the chunk alone, zero-padded.
        let k = chunkwise.taps.len();
        let x_chunk =
            chunkwise.forward(&x.pad_with_zeros(1, k / 2, k / 2)?)?;
        let scaled = (x_chunk * chunk_scale)?;
        Ok(((scaled + x_causal)?, new_cache))
    }
}

/// The convolution module: `in_proj` → GLU gate → depthwise → SwooshR →
/// `out_proj`.
struct ConvModule {
    in_proj: Linear,
    out_proj: Linear,
    depthwise: Depthwise,
}

impl ConvModule {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        stack: &StackConfig,
        streaming_chunk: Option<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (d, k) = (stack.encoder_dim, stack.cnn_kernel);
        let depthwise = match streaming_chunk {
            None => Depthwise::Plain {
                conv: DepthwiseTaps::load(
                    w,
                    &format!("{prefix}.depthwise_conv.weight"),
                    &format!("{prefix}.depthwise_conv.bias"),
                    d,
                    k,
                    device,
                    dtype,
                )?,
                pad: k / 2,
            },
            Some(chunk) => {
                let causal = DepthwiseTaps::load(
                    w,
                    &format!("{prefix}.depthwise_conv.causal_conv.weight"),
                    &format!("{prefix}.depthwise_conv.causal_conv.bias"),
                    d,
                    k / 2 + 1,
                    device,
                    dtype,
                )?;
                let chunkwise = DepthwiseTaps::load(
                    w,
                    &format!("{prefix}.depthwise_conv.chunkwise_conv.weight"),
                    &format!("{prefix}.depthwise_conv.chunkwise_conv.bias"),
                    d,
                    k,
                    device,
                    dtype,
                )?;
                let scale_raw = w
                    .get(
                        &format!(
                            "{prefix}.depthwise_conv.chunkwise_conv_scale"
                        ),
                        &[2, d, k],
                    )
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let chunk_scale = chunk_scale_for(
                    &scale_raw.data,
                    d,
                    k,
                    chunk,
                    device,
                    dtype,
                )?;
                Depthwise::Causal {
                    causal,
                    chunkwise,
                    chunk_scale,
                    left_pad: k / 2,
                }
            },
        };
        Ok(Self {
            in_proj: linear(
                w,
                &format!("{prefix}.in_proj"),
                2 * d,
                d,
                device,
                dtype,
            )?,
            out_proj: linear(
                w,
                &format!("{prefix}.out_proj"),
                d,
                d,
                device,
                dtype,
            )?,
            depthwise,
        })
    }

    /// `x`: `[T, C]`; streaming passes the depthwise cache. Returns the
    /// module output and the updated cache.
    fn forward(
        &self,
        x: &Tensor,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let projected = self.in_proj.forward(x)?; // [T, 2C]
        let c = projected.dim(1)? / 2;
        let gate = candle_nn::ops::sigmoid(&projected.narrow(1, c, c)?)?;
        let gated = (projected.narrow(1, 0, c)? * gate)?; // [T, C]
        let tc = gated.t()?.contiguous()?; // [C, T]
        let (convolved, new_cache) = match cache {
            None => (self.depthwise.forward_plain(&tc)?, None),
            Some(cache) => {
                let (out, cache) = self.depthwise.forward_causal(&tc, cache)?;
                (out, Some(cache))
            },
        };
        let back = convolved.t()?.contiguous()?; // [T, C]
        Ok((self.out_proj.forward(&swoosh_r(&back)?)?, new_cache))
    }
}

/// Builds the `1 + left + right` chunk scale `[C, chunk]` from the raw
/// `[2, C, K]` edge parameter, for a fixed per-stack chunk length (the
/// reference `_get_chunk_scale`).
fn chunk_scale_for(
    raw: &[f32],
    channels: usize,
    kernel: usize,
    chunk: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut scale = vec![1.0f32; channels * chunk];
    let left = &raw[..channels * kernel];
    let right = &raw[channels * kernel..];
    for c in 0..channels {
        for j in 0..chunk {
            let mut v = 0.0f32;
            if chunk < kernel {
                // left_edge[:, :chunk], right_edge[:, −chunk:]
                v += left[c * kernel + j];
                v += right[c * kernel + (kernel - chunk) + j];
            } else {
                if j < kernel {
                    v += left[c * kernel + j];
                }
                if j >= chunk - kernel {
                    v += right[c * kernel + (j - (chunk - kernel))];
                }
            }
            scale[c * chunk + j] += v;
        }
    }
    Tensor::from_slice(&scale, (channels, chunk), device)?.to_dtype(dtype)
}

/// One encoder layer.
struct EncoderLayer {
    attn_weights: AttentionWeights,
    self_attn1: SelfAttention,
    self_attn2: SelfAttention,
    feed_forward1: FeedForward,
    feed_forward2: FeedForward,
    feed_forward3: FeedForward,
    nonlin_attention: NonlinAttention,
    conv_module1: ConvModule,
    conv_module2: ConvModule,
    norm: BiasNorm,
    bypass: Bypass,
    bypass_mid: Bypass,
}

/// The cached streaming state of one layer, all on the device.
pub struct LayerState {
    cached_key: Tensor,    // [left, q·h]
    cached_nonlin: Tensor, // [left, 3·dim/4]
    cached_val1: Tensor,   // [left, v·h]
    cached_val2: Tensor,   // [left, v·h]
    cached_conv1: Tensor,  // [C, k/2]
    cached_conv2: Tensor,  // [C, k/2]
}

impl EncoderLayer {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        stack: &StackConfig,
        pos_dim: usize,
        streaming_chunk: Option<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let d = stack.encoder_dim;
        let ff_dim = |name: &str| -> Result<usize> {
            w.get_any(&format!("{prefix}.{name}.in_proj.bias"))
                .map(|t| t.dims[0])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        };
        Ok(Self {
            attn_weights: AttentionWeights::load(
                w,
                &format!("{prefix}.self_attn_weights"),
                stack,
                pos_dim,
                device,
                dtype,
            )?,
            self_attn1: SelfAttention::load(
                w,
                &format!("{prefix}.self_attn1"),
                stack,
                device,
                dtype,
            )?,
            self_attn2: SelfAttention::load(
                w,
                &format!("{prefix}.self_attn2"),
                stack,
                device,
                dtype,
            )?,
            feed_forward1: FeedForward::load(
                w,
                &format!("{prefix}.feed_forward1"),
                d,
                ff_dim("feed_forward1")?,
                device,
                dtype,
            )?,
            feed_forward2: FeedForward::load(
                w,
                &format!("{prefix}.feed_forward2"),
                d,
                ff_dim("feed_forward2")?,
                device,
                dtype,
            )?,
            feed_forward3: FeedForward::load(
                w,
                &format!("{prefix}.feed_forward3"),
                d,
                ff_dim("feed_forward3")?,
                device,
                dtype,
            )?,
            nonlin_attention: NonlinAttention::load(
                w,
                &format!("{prefix}.nonlin_attention"),
                stack,
                device,
                dtype,
            )?,
            conv_module1: ConvModule::load(
                w,
                &format!("{prefix}.conv_module1"),
                stack,
                streaming_chunk,
                device,
                dtype,
            )?,
            conv_module2: ConvModule::load(
                w,
                &format!("{prefix}.conv_module2"),
                stack,
                streaming_chunk,
                device,
                dtype,
            )?,
            norm: BiasNorm::load(w, &format!("{prefix}.norm"), dtype, device)?,
            bypass: Bypass::load(
                w,
                &format!("{prefix}.bypass.bypass_scale"),
                dtype,
                device,
            )?,
            bypass_mid: Bypass::load(
                w,
                &format!("{prefix}.bypass_mid.bypass_scale"),
                dtype,
                device,
            )?,
        })
    }

    /// One layer pass over `[T, C]`. `state` carries the streaming caches
    /// (`None` offline); `mask_bias` masks unfilled left context.
    fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        state: Option<&mut LayerState>,
        mask_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let orig = x.clone();
        // Read the caches out of the state (tensor clones share storage).
        let (mut key, mut nonlin, mut val1, mut val2, mut conv1, mut conv2) =
            (None, None, None, None, None, None);
        if let Some(state) = &state {
            key = Some(state.cached_key.clone());
            nonlin = Some(state.cached_nonlin.clone());
            val1 = Some(state.cached_val1.clone());
            val2 = Some(state.cached_val2.clone());
            conv1 = Some(state.cached_conv1.clone());
            conv2 = Some(state.cached_conv2.clone());
        }

        let (attn, new_key) =
            self.attn_weights
                .forward(x, pos_emb, key.as_ref(), mask_bias)?;
        let attn_head0 = attn.narrow(0, 0, 1)?;

        let mut x = (x + self.feed_forward1.forward(x)?)?;

        let (na, new_nonlin) =
            self.nonlin_attention
                .forward(&x, &attn_head0, nonlin.as_ref())?;
        x = (x + na)?;

        let (sa1, new_val1) =
            self.self_attn1.forward(&x, &attn, val1.as_ref())?;
        x = (x + sa1)?;

        let (cv1, new_conv1) = self.conv_module1.forward(&x, conv1.as_ref())?;
        x = (x + cv1)?;

        x = (&x + self.feed_forward2.forward(&x)?)?;
        x = self.bypass_mid.forward(&orig, &x)?;

        let (sa2, new_val2) =
            self.self_attn2.forward(&x, &attn, val2.as_ref())?;
        x = (x + sa2)?;

        let (cv2, new_conv2) = self.conv_module2.forward(&x, conv2.as_ref())?;
        x = (x + cv2)?;

        x = (&x + self.feed_forward3.forward(&x)?)?;
        x = self.norm.forward(&x)?;
        x = self.bypass.forward(&orig, &x)?;

        if let Some(state) = state {
            state.cached_key = new_key.expect("streaming attn cache");
            state.cached_nonlin = new_nonlin.expect("streaming nonlin cache");
            state.cached_val1 = new_val1.expect("streaming val1 cache");
            state.cached_val2 = new_val2.expect("streaming val2 cache");
            state.cached_conv1 = new_conv1.expect("streaming conv1 cache");
            state.cached_conv2 = new_conv2.expect("streaming conv2 cache");
        }
        Ok(x)
    }
}

/// One encoder stack: optional temporal downsampling around a layer run.
struct Stack {
    config: StackConfig,
    /// Folded `softmax(downsample.bias)`; present when `downsample > 1`.
    downsample_weights: Option<Tensor>, // [ds, 1]
    out_combiner: Option<Bypass>,
    layers: Vec<EncoderLayer>,
}

impl Stack {
    fn load(
        w: &ModelWeights,
        s: usize,
        stack: &StackConfig,
        pos_dim: usize,
        streaming_chunk50: Option<usize>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let ds = stack.downsample;
        let streaming_chunk = streaming_chunk50.map(|c| c / ds);
        let layers = (0..stack.num_layers)
            .map(|l| {
                EncoderLayer::load(
                    w,
                    &StackConfig::layer_prefix(s, ds, l),
                    stack,
                    pos_dim,
                    streaming_chunk,
                    device,
                    dtype,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (downsample_weights, out_combiner) = if ds > 1 {
            let weights = tensor(
                w,
                &format!("encoder.encoders.{s}.downsample.weights"),
                device,
                dtype,
            )?
            .reshape((ds, 1))?;
            let combiner = Bypass::load(
                w,
                &format!("encoder.encoders.{s}.out_combiner.bypass_scale"),
                dtype,
                device,
            )?;
            (Some(weights), Some(combiner))
        } else {
            (None, None)
        };
        Ok(Self {
            config: stack.clone(),
            downsample_weights,
            out_combiner,
            layers,
        })
    }

    /// Runs the stack over `[T, C]` at the outer rate, handling the
    /// downsample → layers → upsample → combine wrap. `states` holds this
    /// stack's per-layer streaming state; `pos_dim` the positional width.
    fn forward(
        &self,
        x: &Tensor,
        pos_dim: usize,
        mut states: Option<&mut [LayerState]>,
        left_context: usize,
        mask_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ds = self.config.downsample;
        let orig = x.clone();
        let mut inner = if ds > 1 {
            downsample(x, self.downsample_weights.as_ref().expect("ds"), ds)?
        } else {
            x.clone()
        };
        let t = inner.dim(0)?;
        let pos_emb = pos_embedding(
            t,
            left_context,
            pos_dim,
            inner.device(),
            inner.dtype(),
        )?;
        for (l, layer) in self.layers.iter().enumerate() {
            let state = states.as_deref_mut().map(|s| &mut s[l]);
            inner = layer.forward(&inner, &pos_emb, state, mask_bias)?;
        }
        if ds > 1 {
            let up = upsample(&inner, ds, orig.dim(0)?)?;
            self.out_combiner.as_ref().expect("ds").forward(&orig, &up)
        } else {
            Ok(inner)
        }
    }
}

/// `SimpleDownsample`: pad time to a multiple of `ds` by repeating the last
/// frame, then a weighted sum of each `ds` group (`[T, C]` → `[⌈T/ds⌉, C]`).
fn downsample(x: &Tensor, weights: &Tensor, ds: usize) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    let out_t = t.div_ceil(ds);
    let pad = out_t * ds - t;
    let x = if pad > 0 {
        let last = x.narrow(0, t - 1, 1)?;
        let mut parts = vec![x.clone()];
        for _ in 0..pad {
            parts.push(last.clone());
        }
        Tensor::cat(&parts, 0)?
    } else {
        x.clone()
    };
    let grouped = x.reshape((out_t, ds, c))?;
    grouped.broadcast_mul(&weights.reshape((1, ds, 1))?)?.sum(1)
}

/// `SimpleUpsample`: repeat each frame `ds` times, truncated to `out_t`.
fn upsample(x: &Tensor, ds: usize, out_t: usize) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    let up = x
        .unsqueeze(1)?
        .broadcast_as((t, ds, c))?
        .reshape((t * ds, c))?;
    up.narrow(0, 0, out_t)
}

/// `convert_num_channels`: truncate or zero-pad the channel axis.
fn convert_channels(x: &Tensor, channels: usize) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    if channels <= c {
        x.narrow(1, 0, channels)
    } else {
        let zeros = Tensor::zeros((t, channels - c), x.dtype(), x.device())?;
        Tensor::cat(&[x, &zeros], 1)
    }
}

/// The compact relative positional embedding `[left + 2T − 1, pos_dim]`,
/// computed on the CPU by the reference formula and uploaded.
fn pos_embedding(
    t: usize,
    left_context: usize,
    pos_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total = t + left_context;
    let len = total + t - 1;
    let compression = (pos_dim as f64).sqrt();
    let length_scale = pos_dim as f64 / (2.0 * std::f64::consts::PI);
    let half = pos_dim / 2;
    let mut pe = vec![0.0f32; len * pos_dim];
    for (row, slot) in pe.chunks_exact_mut(pos_dim).enumerate() {
        // x runs −(total−1) ..= t−1.
        let x = row as f64 - (total - 1) as f64;
        let compressed = compression *
            x.signum() *
            ((x.abs() + compression).ln() - compression.ln());
        let atan = (compressed / length_scale).atan();
        for f in 0..half {
            let phase = atan * (f + 1) as f64;
            slot[2 * f] = phase.cos() as f32;
            slot[2 * f + 1] = phase.sin() as f32;
        }
        slot[pos_dim - 1] = 1.0;
    }
    Tensor::from_slice(&pe, (len, pos_dim), device)?.to_dtype(dtype)
}

/// The convolutional front end (`encoder_embed`): three Conv2d stages with
/// SwooshR, one ConvNeXt block, the output linear, and BiasNorm.
struct Embed {
    conv0_w: Tensor, // [8, 1, 3, 3]
    conv0_b: Tensor,
    conv4_w: Tensor, // [32, 8, 3, 3]
    conv4_b: Tensor,
    conv7_w: Tensor, // [128, 32, 3, 3]
    conv7_b: Tensor,
    convnext_dw: Vec<Tensor>, // 49 taps [1, 128, 1, 1]
    convnext_dw_b: Tensor,    // [1, 128, 1, 1]
    convnext_pw1: Tensor,     // [384, 128, 1, 1] as [384, 128]
    convnext_pw1_b: Tensor,   // [384]
    convnext_pw2: Tensor,     // [128, 384]
    convnext_pw2_b: Tensor,   // [128]
    out: Linear,
    out_norm: BiasNorm,
}

/// The streaming cache of the front end: the ConvNeXt left context
/// `[128, 3, freq]`.
pub struct EmbedState {
    cached: Tensor,
}

impl Embed {
    fn load(
        w: &ModelWeights,
        config: &ZipformerConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let d0 = config.stacks[0].encoder_dim;
        let out_width = (((config.feature_dim - 1) / 2) - 1) / 2;
        let dw = tensor(
            w,
            "encoder_embed.convnext.depthwise_conv.weight",
            device,
            dtype,
        )?
        .reshape((128, 49))?;
        let mut convnext_dw = Vec::with_capacity(49);
        for k in 0..49 {
            convnext_dw.push(
                dw.narrow(1, k, 1)?.reshape((1, 128, 1, 1))?.contiguous()?,
            );
        }
        Ok(Self {
            conv0_w: tensor(w, "encoder_embed.conv.0.weight", device, dtype)?,
            conv0_b: tensor(w, "encoder_embed.conv.0.bias", device, dtype)?
                .reshape((1, 8, 1, 1))?,
            conv4_w: tensor(w, "encoder_embed.conv.4.weight", device, dtype)?,
            conv4_b: tensor(w, "encoder_embed.conv.4.bias", device, dtype)?
                .reshape((1, 32, 1, 1))?,
            conv7_w: tensor(w, "encoder_embed.conv.7.weight", device, dtype)?,
            conv7_b: tensor(w, "encoder_embed.conv.7.bias", device, dtype)?
                .reshape((1, 128, 1, 1))?,
            convnext_dw,
            convnext_dw_b: tensor(
                w,
                "encoder_embed.convnext.depthwise_conv.bias",
                device,
                dtype,
            )?
            .reshape((1, 128, 1, 1))?,
            convnext_pw1: tensor(
                w,
                "encoder_embed.convnext.pointwise_conv1.weight",
                device,
                dtype,
            )?
            .reshape((384, 128))?,
            convnext_pw1_b: tensor(
                w,
                "encoder_embed.convnext.pointwise_conv1.bias",
                device,
                dtype,
            )?,
            convnext_pw2: tensor(
                w,
                "encoder_embed.convnext.pointwise_conv2.weight",
                device,
                dtype,
            )?
            .reshape((128, 384))?,
            convnext_pw2_b: tensor(
                w,
                "encoder_embed.convnext.pointwise_conv2.bias",
                device,
                dtype,
            )?,
            out: linear(
                w,
                "encoder_embed.out",
                d0,
                128 * out_width,
                device,
                dtype,
            )?,
            out_norm: BiasNorm::load(
                w,
                "encoder_embed.out_norm",
                dtype,
                device,
            )?,
        })
    }

    /// The three strided conv stages: `[1, 1, T, feat]` →
    /// `[1, 128, (T−7)/2, out_width]`.
    fn conv_stages(&self, x: &Tensor) -> Result<Tensor> {
        // Stage 1: k3, pad (0, 1).
        let x = x.pad_with_zeros(3, 1, 1)?;
        let x = x.conv2d(&self.conv0_w, 0, 1, 1, 1)?;
        let x = swoosh_r(&x.broadcast_add(&self.conv0_b)?)?;
        // Stage 2: k3, stride 2, no pad.
        let x = x.conv2d(&self.conv4_w, 0, 2, 1, 1)?;
        let x = swoosh_r(&x.broadcast_add(&self.conv4_b)?)?;
        // Stage 3: k3, stride (1, 2): full stride-1 conv, then every second
        // frequency column.
        let x = x.conv2d(&self.conv7_w, 0, 1, 1, 1)?;
        let x = swoosh_r(&x.broadcast_add(&self.conv7_b)?)?;
        let freq = x.dim(3)?;
        let idx: Vec<u32> = (0..freq).step_by(2).map(|i| i as u32).collect();
        let idx = Tensor::from_slice(&idx, (idx.len(),), x.device())?;
        x.index_select(&idx, 3)
    }

    /// The ConvNeXt block over `[1, 128, T, F]`; `pad_time` pads
    /// symmetrically (offline), otherwise the input already carries the
    /// 3-frame left context and loses the 3-frame right margin (streaming),
    /// so the output is 6 frames shorter than the input.
    fn convnext(&self, x: &Tensor, pad_time: bool) -> Result<Tensor> {
        let bypass_t = if pad_time {
            x.clone()
        } else {
            x.narrow(2, 3, x.dim(2)? - 6)?
        };
        let padded = if pad_time {
            x.pad_with_zeros(2, 3, 3)?.pad_with_zeros(3, 3, 3)?
        } else {
            x.pad_with_zeros(3, 3, 3)?
        };
        // Depthwise 7×7 as 49 shifted scaled copies.
        let (_, _, pt, pf) = padded.dims4()?;
        let out_t = pt - 6;
        let out_f = pf - 6;
        let mut acc: Option<Tensor> = None;
        for (k, tap) in self.convnext_dw.iter().enumerate() {
            let (dt, df) = (k / 7, k % 7);
            let window = padded.narrow(2, dt, out_t)?.narrow(3, df, out_f)?;
            let term = window.broadcast_mul(tap)?;
            acc = Some(match acc {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        let dw = acc.expect("taps").broadcast_add(&self.convnext_dw_b)?;
        // Pointwise 1×1 convs as matmuls over the channel axis.
        let (b, c, t, f) = dw.dims4()?;
        let flat = dw.permute((0, 2, 3, 1))?.reshape((b * t * f, c))?;
        let h = flat
            .matmul(&self.convnext_pw1.t()?)?
            .broadcast_add(&self.convnext_pw1_b)?;
        let h = swoosh_l(&h)?;
        let out = h
            .matmul(&self.convnext_pw2.t()?)?
            .broadcast_add(&self.convnext_pw2_b)?;
        let out = out.reshape((b, t, f, 128))?.permute((0, 3, 1, 2))?;
        bypass_t + out
    }

    /// Offline: `[T, feat]` features → `[T50, d0]`, `T50 = (T − 7) / 2`.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (t, feat) = features.dims2()?;
        let x = features.reshape((1, 1, t, feat))?;
        let x = self.conv_stages(&x)?;
        let x = self.convnext(&x, true)?;
        self.output(&x)
    }

    /// Streaming: consumes a window whose conv output covers
    /// `chunk50 + 3` frames, the cache supplying the left context. Returns
    /// `[chunk50, d0]` and updates the cache.
    fn forward_streaming(
        &self,
        features: &Tensor,
        state: &mut EmbedState,
    ) -> Result<Tensor> {
        let (t, feat) = features.dims2()?;
        let x = features.reshape((1, 1, t, feat))?;
        let x = self.conv_stages(&x)?; // [1, 128, (T−7)/2, F]
        let conv_t = x.dim(2)?;
        // Prepend the cached 3 frames. The new cache is the 3 frames right
        // after this step's output span (the reference `x[:, :, T:T+3]` of
        // the concatenated tensor, `T = conv_t − 3` output frames): the
        // 13-frame feature overlap regenerates the right margin next step.
        let cached = state.cached.unsqueeze(0)?;
        let with_left = Tensor::cat(&[&cached, &x], 2)?;
        state.cached = with_left
            .narrow(2, conv_t - 3, 3)?
            .squeeze(0)?
            .contiguous()?;
        let x = self.convnext(&with_left, false)?;
        self.output(&x)
    }

    /// Shared tail: flatten channels×freq, linear, BiasNorm → `[T', d0]`.
    fn output(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = x.dims4()?;
        debug_assert_eq!(b, 1);
        let flat = x.permute((0, 2, 1, 3))?.reshape((t, c * f))?;
        let out = self.out.forward(&flat)?;
        self.out_norm.forward(&out)
    }
}

/// The complete streaming state: per-layer caches (flat, layer-major), the
/// front-end cache, and the processed-frame counter.
pub struct NetState {
    layers: Vec<LayerState>,
    embed: EmbedState,
    /// 50 Hz frames processed so far (drives the left-context mask).
    processed: usize,
}

/// The Zipformer2 encoder network.
pub struct ZipformerNet {
    embed: Embed,
    stacks: Vec<Stack>,
    downsample_output: Tensor, // [2, 1]
    encoder_proj: Linear,
    config: ZipformerConfig,
    dtype: DType,
    device: Device,
    /// Per-stack positional embedding for streaming, precomputed once. In a
    /// streaming run the window size and left-context length are fixed, so the
    /// positional table is identical for every window of a given stack;
    /// recomputing it per window (host-side trig plus an upload) is pure
    /// waste. Empty for offline models, whose chunk length varies.
    streaming_pos: Vec<Tensor>,
}

impl ZipformerNet {
    /// Loads the network onto `device` at `dtype`.
    pub fn load(
        w: &ModelWeights,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let config = w.config.clone();
        let chunk50 = config.streaming.as_ref().map(|s| s.shift_frames / 2);
        let embed = Embed::load(w, &config, device, dtype)?;
        let stacks = config
            .stacks
            .iter()
            .enumerate()
            .map(|(s, stack)| {
                Stack::load(w, s, stack, config.pos_dim, chunk50, device, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        let downsample_output =
            tensor(w, "encoder.downsample_output.weights", device, dtype)?
                .reshape((2, 1))?;
        let encoder_proj = linear(
            w,
            "encoder_proj",
            config.joiner_dim,
            config.encoder_out_dim,
            device,
            dtype,
        )?;
        // Streaming: one positional table per stack (window size chunk50/ds,
        // left context left_context[s]), computed once here and reused for
        // every window.
        let streaming_pos = match &config.streaming {
            Some(streaming) => {
                let chunk50 = streaming.shift_frames / 2;
                config
                    .stacks
                    .iter()
                    .enumerate()
                    .map(|(s, stack)| {
                        pos_embedding(
                            chunk50.div_ceil(stack.downsample),
                            streaming.left_context[s],
                            config.pos_dim,
                            device,
                            dtype,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            },
            None => Vec::new(),
        };
        Ok(Self {
            embed,
            stacks,
            downsample_output,
            encoder_proj,
            config,
            dtype,
            device: device.clone(),
            streaming_pos,
        })
    }

    /// The model geometry.
    pub fn config(&self) -> &ZipformerConfig { &self.config }

    /// The compute device.
    pub fn device(&self) -> &Device { &self.device }

    /// Offline forward: `[T, feat]` features → `[T25, joiner_dim]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = self.embed.forward(features)?; // [T50, d0]
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            x = convert_channels(&x, stack.config.encoder_dim)?;
            x = stack.forward(&x, self.config.pos_dim, None, 0, None)?;
            outputs.push(x.clone());
        }
        let full = self.full_dim_output(&outputs)?;
        let down = downsample(&full, &self.downsample_output, 2)?;
        self.encoder_proj.forward(&down)
    }

    /// A fresh streaming state (zeros everywhere).
    pub fn init_state(&self) -> Result<NetState> {
        let streaming = self
            .config
            .streaming
            .as_ref()
            .expect("streaming state of a streaming model");
        let mut layers = Vec::new();
        for (s, stack) in self.config.stacks.iter().enumerate() {
            let left = streaming.left_context[s];
            let zeros = |rows: usize, cols: usize| {
                Tensor::zeros((rows, cols), self.dtype, &self.device)
            };
            for _ in 0..stack.num_layers {
                let _ = s;
                layers.push(LayerState {
                    cached_key: zeros(
                        left,
                        stack.num_heads * stack.query_head_dim,
                    )?,
                    cached_nonlin: zeros(left, stack.nonlin_hidden)?,
                    cached_val1: zeros(
                        left,
                        stack.num_heads * stack.value_head_dim,
                    )?,
                    cached_val2: zeros(
                        left,
                        stack.num_heads * stack.value_head_dim,
                    )?,
                    cached_conv1: zeros(
                        stack.encoder_dim,
                        stack.cnn_kernel / 2,
                    )?,
                    cached_conv2: zeros(
                        stack.encoder_dim,
                        stack.cnn_kernel / 2,
                    )?,
                });
            }
        }
        let out_width = (((self.config.feature_dim - 1) / 2) - 1) / 2;
        let embed = EmbedState {
            cached: Tensor::zeros(
                (128, 3, out_width),
                self.dtype,
                &self.device,
            )?,
        };
        Ok(NetState {
            layers,
            embed,
            processed: 0,
        })
    }

    /// Streaming forward: one `[window_frames, feat]` feature window →
    /// `[chunk50/2, joiner_dim]`, advancing `state`.
    pub fn forward_streaming(
        &self,
        features: &Tensor,
        state: &mut NetState,
    ) -> Result<Tensor> {
        let streaming = self
            .config
            .streaming
            .as_ref()
            .expect("streaming forward of a streaming model");
        let chunk50 = streaming.shift_frames / 2;
        let mut x = self.embed.forward_streaming(features, &mut state.embed)?;
        debug_assert_eq!(x.dim(0)?, chunk50);

        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.stacks.len());
        let mut layer_base = 0usize;
        for (s, stack) in self.stacks.iter().enumerate() {
            let ds = stack.config.downsample;
            let left = streaming.left_context[s];
            x = convert_channels(&x, stack.config.encoder_dim)?;
            let mask = self.mask_bias(state.processed, s)?;
            let states =
                &mut state.layers[layer_base..layer_base + stack.layers.len()];
            let orig = x.clone();
            let _ = left;
            let mut inner = if ds > 1 {
                downsample(
                    &x,
                    stack.downsample_weights.as_ref().expect("ds"),
                    ds,
                )?
            } else {
                x.clone()
            };
            // The positional table is window-invariant in streaming; use the
            // one precomputed at load for this stack.
            let pos_emb = &self.streaming_pos[s];
            for (l, layer) in stack.layers.iter().enumerate() {
                inner = layer.forward(
                    &inner,
                    pos_emb,
                    Some(&mut states[l]),
                    mask.as_ref(),
                )?;
            }
            x = if ds > 1 {
                let up = upsample(&inner, ds, orig.dim(0)?)?;
                stack
                    .out_combiner
                    .as_ref()
                    .expect("ds")
                    .forward(&orig, &up)?
            } else {
                inner
            };
            outputs.push(x.clone());
            layer_base += stack.layers.len();
        }
        state.processed += chunk50;

        let full = self.full_dim_output(&outputs)?;
        let down = downsample(&full, &self.downsample_output, 2)?;
        self.encoder_proj.forward(&down)
    }

    /// The additive attention mask over `[left/ds + chunk/ds]` source
    /// positions of stack `s`: −1000 on cached slots not yet filled (the
    /// reference builds the top-rate mask from the processed counter and
    /// decimates it per stack).
    fn mask_bias(&self, processed: usize, s: usize) -> Result<Option<Tensor>> {
        let streaming = self.config.streaming.as_ref().expect("streaming");
        let left_top =
            streaming.left_context[0] * self.config.stacks[0].downsample; // == left_context_frames at 50 Hz
        if processed >= left_top {
            return Ok(None);
        }
        let chunk50 = streaming.shift_frames / 2;
        let ds = self.config.stacks[s].downsample;
        // Top-rate mask: entry i of the left region is masked iff
        // i < left_top − processed; the chunk region is never masked. The
        // stack sees its decimated view: left_context[s] cached slots plus
        // the stack's chunk (ceil, matching the padded downsample).
        let masked_head = left_top - processed;
        let src = streaming.left_context[s] + chunk50.div_ceil(ds);
        let mut bias = vec![0.0f32; src];
        for (i, slot) in bias.iter_mut().enumerate() {
            if i * ds < masked_head {
                *slot = -1000.0;
            }
        }
        Ok(Some(
            Tensor::from_slice(&bias, (src,), &self.device)?
                .to_dtype(self.dtype)?,
        ))
    }

    /// `_get_full_dim_output`: the last stack's output, with higher channels
    /// taken from the most recent stack that has them.
    fn full_dim_output(&self, outputs: &[Tensor]) -> Result<Tensor> {
        let dims: Vec<usize> =
            self.config.stacks.iter().map(|s| s.encoder_dim).collect();
        let out_dim = self.config.encoder_out_dim;
        let mut pieces: Vec<Tensor> = vec![outputs[outputs.len() - 1].clone()];
        let mut cur = dims[dims.len() - 1];
        for i in (0..dims.len() - 1).rev() {
            if dims[i] > cur {
                pieces.push(outputs[i].narrow(1, cur, dims[i] - cur)?);
                cur = dims[i];
            }
        }
        debug_assert_eq!(cur, out_dim);
        if pieces.len() == 1 {
            return Ok(pieces.remove(0));
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        Tensor::cat(&refs, 1)
    }
}
