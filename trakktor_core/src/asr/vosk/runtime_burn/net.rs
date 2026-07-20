//! The Zipformer2 encoder on burn.
//!
//! A port of the candle network in [`runtime::net`](super::super::runtime::net)
//! with the same numerical semantics, written idiomatically for burn rather
//! than as a mirror: linear projections use `module::linear`, and the
//! depthwise convolutions (the two per-layer conv modules and the ConvNeXt
//! 7×7) are native grouped `conv1d`/`conv2d` — burn's convolution kernels do
//! not suffer candle's one-launch-per-group pathology, so the shifted-sum
//! workaround of the candle runtime is unnecessary here. The attention is
//! still assembled from matmuls (Zipformer shares one weight matrix across two
//! self-attention passes and the nonlin-attention module and adds a relative
//! positional term, so the fused SDPA primitive does not apply). Batch is
//! always 1; the encoder body works on rank-2 `[T, C]` tensors.
//!
//! Weight tensors are built directly from the extracted (canonical) map. burn
//! operations panic on shape mismatch, so the geometry is validated at load
//! (`weights::validate`); a panic past loading is a bug, not a data
//! condition.

use burn::tensor::{
    Tensor, TensorData,
    activation::{sigmoid, softmax},
    backend::Backend,
    module::{conv1d, conv2d},
    ops::{ConvOptions, PadMode},
};

use crate::asr::vosk::weights::{ModelWeights, StackConfig, ZipformerConfig};

/// The canonical tensor map plus device/const helpers.
struct Loader<'a, B: Backend> {
    w: &'a ModelWeights,
    device: &'a B::Device,
}

impl<'a, B: Backend> Loader<'a, B> {
    fn new(w: &'a ModelWeights, device: &'a B::Device) -> Self {
        Self { w, device }
    }

    /// A tensor of the given rank from a canonical name (row-major).
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
    ) -> Tensor<B, D> {
        let raw = self.w.get_any(name).expect("validated weight");
        debug_assert_eq!(raw.data.len(), shape.iter().product::<usize>());
        Tensor::from_data(TensorData::new(raw.data.clone(), shape), self.device)
    }

    /// A scalar canonical value (a rank-0 buffer stored as one element).
    fn scalar(&self, name: &str) -> f32 {
        self.w.get_any(name).expect("validated weight").data[0]
    }
}

/// A linear layer `y = x·Wᵀ + b`, holding the weight already transposed to
/// burn's `[in, out]` layout.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [in, out]
    bias: Tensor<B, 2>,   // [1, out]
}

impl<B: Backend> Linear<B> {
    fn load(
        l: &Loader<B>,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Self {
        let weight = l
            .tensor::<2>(&format!("{prefix}.weight"), [out_dim, in_dim])
            .swap_dims(0, 1);
        let bias = l
            .tensor::<1>(&format!("{prefix}.bias"), [out_dim])
            .reshape([1, out_dim]);
        Self { weight, bias }
    }

    /// `[T, in]` -> `[T, out]`.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.clone()) + self.bias.clone()
    }
}

/// `softplus(y) = relu(y) + log1p(exp(−|y|))`.
fn softplus<B: Backend, const D: usize>(y: Tensor<B, D>) -> Tensor<B, D> {
    let relu = y.clone().clamp_min(0.0);
    let log1p = (y.abs().neg().exp() + 1.0).log();
    relu + log1p
}

/// `SwooshL(x) = softplus(x − 4) − 0.08·x − 0.035`.
fn swoosh_l<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    softplus(x.clone() - 4.0) - x * 0.08 - 0.035
}

/// `SwooshR(x) = softplus(x − 1) − 0.08·x − 0.313261687`.
fn swoosh_r<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    softplus(x.clone() - 1.0) - x * 0.08 - 0.313_261_687
}

/// BiasNorm over the last dim: `x · (mean((x − bias)², −1)^−½ ·
/// exp(log_scale))`.
struct BiasNorm<B: Backend> {
    bias: Tensor<B, 2>, // [1, C]
    scale: f64,
}

impl<B: Backend> BiasNorm<B> {
    fn load(l: &Loader<B>, prefix: &str, channels: usize) -> Self {
        Self {
            bias: l
                .tensor::<1>(&format!("{prefix}.bias"), [channels])
                .reshape([1, channels]),
            scale: f64::from(l.scalar(&format!("{prefix}.log_scale"))).exp(),
        }
    }

    /// `[T, C]` -> `[T, C]`.
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let centered = x.clone() - self.bias.clone();
        let mean_sq = (centered.clone() * centered).mean_dim(1); // [T, 1]
        let scales = mean_sq.powf_scalar(-0.5) * self.scale;
        x * scales
    }
}

/// A learned per-channel bypass: `orig + (x − orig)·scale`.
struct Bypass<B: Backend> {
    scale: Tensor<B, 2>, // [1, C]
}

impl<B: Backend> Bypass<B> {
    fn load(l: &Loader<B>, name: &str, channels: usize) -> Self {
        Self {
            scale: l.tensor::<1>(name, [channels]).reshape([1, channels]),
        }
    }

    fn forward(&self, orig: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        orig.clone() + (x - orig) * self.scale.clone()
    }
}

/// Feed-forward: `out_proj(SwooshL(in_proj(x)))`.
struct FeedForward<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn load(l: &Loader<B>, prefix: &str, dim: usize, ff_dim: usize) -> Self {
        Self {
            in_proj: Linear::load(l, &format!("{prefix}.in_proj"), ff_dim, dim),
            out_proj: Linear::load(
                l,
                &format!("{prefix}.out_proj"),
                dim,
                ff_dim,
            ),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.out_proj.forward(swoosh_l(self.in_proj.forward(x)))
    }
}

/// The streaming caches of one layer.
pub struct LayerState<B: Backend> {
    cached_key: Tensor<B, 2>,    // [left, q·h]
    cached_nonlin: Tensor<B, 2>, // [left, hidden]
    cached_val1: Tensor<B, 2>,   // [left, v·h]
    cached_val2: Tensor<B, 2>,   // [left, v·h]
    cached_conv1: Tensor<B, 2>,  // [C, k/2]
    cached_conv2: Tensor<B, 2>,  // [C, k/2]
}

/// The shared attention-weights module.
struct AttentionWeights<B: Backend> {
    in_proj: Linear<B>,
    linear_pos: Tensor<B, 2>, // [pos_dim, p·h]
    num_heads: usize,
    query_head_dim: usize,
    pos_head_dim: usize,
}

impl<B: Backend> AttentionWeights<B> {
    fn load(
        l: &Loader<B>,
        prefix: &str,
        stack: &StackConfig,
        pos_dim: usize,
    ) -> Self {
        let (h, q, p) =
            (stack.num_heads, stack.query_head_dim, stack.pos_head_dim);
        let linear_pos = l
            .tensor::<2>(
                &format!("{prefix}.linear_pos.weight"),
                [p * h, pos_dim],
            )
            .swap_dims(0, 1); // [pos_dim, p·h]
        Self {
            in_proj: Linear::load(
                l,
                &format!("{prefix}.in_proj"),
                (2 * q + p) * h,
                stack.encoder_dim,
            ),
            linear_pos,
            num_heads: h,
            query_head_dim: q,
            pos_head_dim: p,
        }
    }

    /// Returns the attention weights `[h, T, src]` and, streaming, the new
    /// key cache.
    fn forward(
        &self,
        x: Tensor<B, 2>,
        pos_emb: &Tensor<B, 2>,
        cached_key: Option<Tensor<B, 2>>,
        mask_bias: Option<&Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 2>>) {
        let t = x.dims()[0];
        let (h, q, p) =
            (self.num_heads, self.query_head_dim, self.pos_head_dim);
        let projected = self.in_proj.forward(x); // [T, (2q+p)·h]
        let queries = projected.clone().narrow(1, 0, q * h);
        let keys_new = projected.clone().narrow(1, q * h, q * h);
        let pos_q = projected.narrow(1, 2 * q * h, p * h);

        let (keys, new_cache) = match cached_key {
            None => (keys_new, None),
            Some(cache) => {
                let left = cache.dims()[0];
                let full = Tensor::cat(vec![cache, keys_new], 0);
                let tail = full.clone().narrow(0, full.dims()[0] - left, left);
                (full, Some(tail))
            },
        };
        let src = keys.dims()[0];

        let qh = queries.reshape([t, h, q]).swap_dims(0, 1); // [h, T, q]
        let kh = keys.reshape([src, h, q]).swap_dims(0, 1).swap_dims(1, 2); // [h, q, src]
        let mut scores = qh.matmul(kh); // [h, T, src]

        let pos_len = pos_emb.dims()[0];
        let pos = pos_emb.clone().matmul(self.linear_pos.clone()); // [pos_len, p·h]
        let pos = pos.reshape([pos_len, h, p]).swap_dims(0, 1).swap_dims(1, 2); // [h, p, pos_len]
        let ph = pos_q.reshape([t, h, p]).swap_dims(0, 1); // [h, T, p]
        let pos_scores = ph.matmul(pos); // [h, T, pos_len]
        scores = scores + rel_shift(pos_scores, src);

        if let Some(bias) = mask_bias {
            scores = scores + bias.clone().reshape([1, 1, src]);
        }
        (softmax(scores, 2), new_cache)
    }
}

/// The relative→absolute skew of `[h, T, n]` positional scores into
/// `[h, T, src]` (`out[h,i,j] = in[h,i,(T−1)−i+j]`).
fn rel_shift<B: Backend>(pos_scores: Tensor<B, 3>, src: usize) -> Tensor<B, 3> {
    let [h, t, n] = pos_scores.dims();
    if t == 1 {
        return pos_scores.narrow(2, 0, src);
    }
    let flat = pos_scores.reshape([h, t * n]);
    let shifted = flat.narrow(1, t - 1, t * (n - 1));
    shifted.reshape([h, t, n - 1]).narrow(2, 0, src)
}

/// Simple attention over precomputed weights.
struct SelfAttention<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    value_head_dim: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn load(l: &Loader<B>, prefix: &str, stack: &StackConfig) -> Self {
        let (h, v, d) =
            (stack.num_heads, stack.value_head_dim, stack.encoder_dim);
        Self {
            in_proj: Linear::load(l, &format!("{prefix}.in_proj"), h * v, d),
            out_proj: Linear::load(l, &format!("{prefix}.out_proj"), d, h * v),
            num_heads: h,
            value_head_dim: v,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        attn: &Tensor<B, 3>,
        cached_val: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let t = x.dims()[0];
        let (h, v) = (self.num_heads, self.value_head_dim);
        let values_new = self.in_proj.forward(x); // [T, h·v]
        let (values, new_cache) = match cached_val {
            None => (values_new, None),
            Some(cache) => {
                let left = cache.dims()[0];
                let full = Tensor::cat(vec![cache, values_new], 0);
                let tail = full.clone().narrow(0, full.dims()[0] - left, left);
                (full, Some(tail))
            },
        };
        let src = values.dims()[0];
        let vh = values.reshape([src, h, v]).swap_dims(0, 1); // [h, src, v]
        let out = attn.clone().matmul(vh); // [h, T, v]
        let out = out.swap_dims(0, 1).reshape([t, h * v]);
        (self.out_proj.forward(out), new_cache)
    }
}

/// The nonlin-attention module.
struct NonlinAttention<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    hidden: usize,
}

impl<B: Backend> NonlinAttention<B> {
    fn load(l: &Loader<B>, prefix: &str, stack: &StackConfig) -> Self {
        let (hidden, d) = (stack.nonlin_hidden, stack.encoder_dim);
        Self {
            in_proj: Linear::load(
                l,
                &format!("{prefix}.in_proj"),
                3 * hidden,
                d,
            ),
            out_proj: Linear::load(l, &format!("{prefix}.out_proj"), d, hidden),
            hidden,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        attn_head0: &Tensor<B, 3>,
        cached: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let hidden = self.hidden;
        let projected = self.in_proj.forward(x); // [T, 3·hidden]
        let s = projected.clone().narrow(1, 0, hidden);
        let v = projected.clone().narrow(1, hidden, hidden);
        let y = projected.narrow(1, 2 * hidden, hidden);
        let gated = v * s.tanh(); // [T, hidden]

        let (values, new_cache) = match cached {
            None => (gated, None),
            Some(cache) => {
                let left = cache.dims()[0];
                let full = Tensor::cat(vec![cache, gated], 0);
                let tail = full.clone().narrow(0, full.dims()[0] - left, left);
                (full, Some(tail))
            },
        };
        let attended = attn_head0
            .clone()
            .matmul(values.unsqueeze::<3>()) // [1, T, hidden]
            .squeeze_dim::<2>(0);
        let out = attended * y;
        (self.out_proj.forward(out), new_cache)
    }
}

/// The depthwise stage of a convolution module.
enum Depthwise<B: Backend> {
    Plain {
        weight: Tensor<B, 3>, // [C, 1, k]
        bias: Tensor<B, 1>,   // [C]
        pad: usize,
    },
    Causal {
        causal_w: Tensor<B, 3>, // [C, 1, half]
        causal_b: Tensor<B, 1>,
        chunk_w: Tensor<B, 3>, // [C, 1, k]
        chunk_b: Tensor<B, 1>,
        chunk_scale: Tensor<B, 2>, // [C, chunk]
        left_pad: usize,
    },
}

impl<B: Backend> Depthwise<B> {
    /// Offline forward over `[C, T]` (same length out).
    fn forward_plain(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let Depthwise::Plain { weight, bias, pad } = self else {
            unreachable!("causal model offline");
        };
        let (c, t) = (x.dims()[0], x.dims()[1]);
        let x = x.reshape([1, c, t]);
        let out = conv1d(
            x,
            weight.clone(),
            Some(bias.clone()),
            ConvOptions::new([1], [*pad], [1], c),
        );
        out.reshape([c, t])
    }

    /// Streaming forward over `[C, chunk]` with the `[C, left_pad]` cache.
    fn forward_causal(
        &self,
        x: Tensor<B, 2>,
        cache: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let Depthwise::Causal {
            causal_w,
            causal_b,
            chunk_w,
            chunk_b,
            chunk_scale,
            left_pad,
        } = self
        else {
            unreachable!("offline model streaming");
        };
        let c = x.dims()[0];
        let chunk = x.dims()[1];
        let full = Tensor::cat(vec![cache, x.clone()], 1); // [C, left_pad + chunk]
        let full_t = full.dims()[1];
        let new_cache = full.clone().narrow(1, full_t - left_pad, *left_pad);
        // Causal half-kernel over the padded input (valid, no padding).
        let x_causal = conv1d(
            full.reshape([1, c, full_t]),
            causal_w.clone(),
            Some(causal_b.clone()),
            ConvOptions::new([1], [0], [1], c),
        )
        .reshape([c, chunk]);
        // Chunkwise kernel over the chunk alone, symmetric-padded.
        let k = chunk_w.dims()[2];
        let x_chunk = conv1d(
            x.reshape([1, c, chunk]),
            chunk_w.clone(),
            Some(chunk_b.clone()),
            ConvOptions::new([1], [k / 2], [1], c),
        )
        .reshape([c, chunk]);
        let scaled = x_chunk * chunk_scale.clone();
        (scaled + x_causal, new_cache)
    }
}

/// The convolution module.
struct ConvModule<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    depthwise: Depthwise<B>,
}

impl<B: Backend> ConvModule<B> {
    fn load(
        l: &Loader<B>,
        prefix: &str,
        stack: &StackConfig,
        streaming_chunk: Option<usize>,
    ) -> Self {
        let (d, k) = (stack.encoder_dim, stack.cnn_kernel);
        let depthwise = match streaming_chunk {
            None => Depthwise::Plain {
                weight: l.tensor::<3>(
                    &format!("{prefix}.depthwise_conv.weight"),
                    [d, 1, k],
                ),
                bias: l
                    .tensor::<1>(&format!("{prefix}.depthwise_conv.bias"), [d]),
                pad: k / 2,
            },
            Some(chunk) => {
                let half = k / 2 + 1;
                let scale_raw = l.w.get_any(&format!(
                    "{prefix}.depthwise_conv.chunkwise_conv_scale"
                ));
                let chunk_scale = chunk_scale_for(
                    &scale_raw.expect("validated").data,
                    d,
                    k,
                    chunk,
                    l.device,
                );
                Depthwise::Causal {
                    causal_w: l.tensor::<3>(
                        &format!("{prefix}.depthwise_conv.causal_conv.weight"),
                        [d, 1, half],
                    ),
                    causal_b: l.tensor::<1>(
                        &format!("{prefix}.depthwise_conv.causal_conv.bias"),
                        [d],
                    ),
                    chunk_w: l.tensor::<3>(
                        &format!(
                            "{prefix}.depthwise_conv.chunkwise_conv.weight"
                        ),
                        [d, 1, k],
                    ),
                    chunk_b: l.tensor::<1>(
                        &format!("{prefix}.depthwise_conv.chunkwise_conv.bias"),
                        [d],
                    ),
                    chunk_scale,
                    left_pad: k / 2,
                }
            },
        };
        Self {
            in_proj: Linear::load(l, &format!("{prefix}.in_proj"), 2 * d, d),
            out_proj: Linear::load(l, &format!("{prefix}.out_proj"), d, d),
            depthwise,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        cache: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let projected = self.in_proj.forward(x); // [T, 2C]
        let c = projected.dims()[1] / 2;
        let gate = sigmoid(projected.clone().narrow(1, c, c));
        let gated = projected.narrow(1, 0, c) * gate; // [T, C]
        let tc = gated.swap_dims(0, 1); // [C, T]
        let (convolved, new_cache) = match cache {
            None => (self.depthwise.forward_plain(tc), None),
            Some(cache) => {
                let (out, cache) = self.depthwise.forward_causal(tc, cache);
                (out, Some(cache))
            },
        };
        let back = convolved.swap_dims(0, 1); // [T, C]
        (self.out_proj.forward(swoosh_r(back)), new_cache)
    }
}

/// Builds the `1 + left + right` chunk scale `[C, chunk]` from the raw
/// `[2, C, K]` edge parameter (the reference `_get_chunk_scale`).
fn chunk_scale_for<B: Backend>(
    raw: &[f32],
    channels: usize,
    kernel: usize,
    chunk: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut scale = vec![1.0f32; channels * chunk];
    let left = &raw[..channels * kernel];
    let right = &raw[channels * kernel..];
    for c in 0..channels {
        for j in 0..chunk {
            let mut v = 0.0f32;
            if chunk < kernel {
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
    Tensor::from_data(TensorData::new(scale, [channels, chunk]), device)
}

/// One encoder layer.
struct EncoderLayer<B: Backend> {
    attn_weights: AttentionWeights<B>,
    self_attn1: SelfAttention<B>,
    self_attn2: SelfAttention<B>,
    feed_forward1: FeedForward<B>,
    feed_forward2: FeedForward<B>,
    feed_forward3: FeedForward<B>,
    nonlin_attention: NonlinAttention<B>,
    conv_module1: ConvModule<B>,
    conv_module2: ConvModule<B>,
    norm: BiasNorm<B>,
    bypass: Bypass<B>,
    bypass_mid: Bypass<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn load(
        l: &Loader<B>,
        prefix: &str,
        stack: &StackConfig,
        pos_dim: usize,
        streaming_chunk: Option<usize>,
    ) -> Self {
        let d = stack.encoder_dim;
        let ff_dim = |name: &str| {
            l.w.get_any(&format!("{prefix}.{name}.in_proj.bias"))
                .expect("validated")
                .dims[0]
        };
        Self {
            attn_weights: AttentionWeights::load(
                l,
                &format!("{prefix}.self_attn_weights"),
                stack,
                pos_dim,
            ),
            self_attn1: SelfAttention::load(
                l,
                &format!("{prefix}.self_attn1"),
                stack,
            ),
            self_attn2: SelfAttention::load(
                l,
                &format!("{prefix}.self_attn2"),
                stack,
            ),
            feed_forward1: FeedForward::load(
                l,
                &format!("{prefix}.feed_forward1"),
                d,
                ff_dim("feed_forward1"),
            ),
            feed_forward2: FeedForward::load(
                l,
                &format!("{prefix}.feed_forward2"),
                d,
                ff_dim("feed_forward2"),
            ),
            feed_forward3: FeedForward::load(
                l,
                &format!("{prefix}.feed_forward3"),
                d,
                ff_dim("feed_forward3"),
            ),
            nonlin_attention: NonlinAttention::load(
                l,
                &format!("{prefix}.nonlin_attention"),
                stack,
            ),
            conv_module1: ConvModule::load(
                l,
                &format!("{prefix}.conv_module1"),
                stack,
                streaming_chunk,
            ),
            conv_module2: ConvModule::load(
                l,
                &format!("{prefix}.conv_module2"),
                stack,
                streaming_chunk,
            ),
            norm: BiasNorm::load(l, &format!("{prefix}.norm"), d),
            bypass: Bypass::load(
                l,
                &format!("{prefix}.bypass.bypass_scale"),
                d,
            ),
            bypass_mid: Bypass::load(
                l,
                &format!("{prefix}.bypass_mid.bypass_scale"),
                d,
            ),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        pos_emb: &Tensor<B, 2>,
        state: Option<&mut LayerState<B>>,
        mask_bias: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let orig = x.clone();
        let caches = state.as_ref().map(|s| {
            (
                s.cached_key.clone(),
                s.cached_nonlin.clone(),
                s.cached_val1.clone(),
                s.cached_val2.clone(),
                s.cached_conv1.clone(),
                s.cached_conv2.clone(),
            )
        });
        let (key, nonlin, val1, val2, conv1c, conv2c) = match caches {
            Some((a, b, c, d, e, f)) => {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f))
            },
            None => (None, None, None, None, None, None),
        };

        let (attn, new_key) =
            self.attn_weights
                .forward(x.clone(), pos_emb, key, mask_bias);
        let attn_head0 = attn.clone().narrow(0, 0, 1);

        let mut x = x.clone() + self.feed_forward1.forward(x);

        let (na, new_nonlin) =
            self.nonlin_attention
                .forward(x.clone(), &attn_head0, nonlin);
        x = x + na;

        let (sa1, new_val1) = self.self_attn1.forward(x.clone(), &attn, val1);
        x = x + sa1;

        let (cv1, new_conv1) = self.conv_module1.forward(x.clone(), conv1c);
        x = x + cv1;

        x = x.clone() + self.feed_forward2.forward(x);
        x = self.bypass_mid.forward(orig.clone(), x);

        let (sa2, new_val2) = self.self_attn2.forward(x.clone(), &attn, val2);
        x = x + sa2;

        let (cv2, new_conv2) = self.conv_module2.forward(x.clone(), conv2c);
        x = x + cv2;

        x = x.clone() + self.feed_forward3.forward(x);
        x = self.norm.forward(x);
        x = self.bypass.forward(orig, x);

        if let Some(state) = state {
            state.cached_key = new_key.expect("streaming key cache");
            state.cached_nonlin = new_nonlin.expect("streaming nonlin cache");
            state.cached_val1 = new_val1.expect("streaming val1 cache");
            state.cached_val2 = new_val2.expect("streaming val2 cache");
            state.cached_conv1 = new_conv1.expect("streaming conv1 cache");
            state.cached_conv2 = new_conv2.expect("streaming conv2 cache");
        }
        x
    }
}

/// One encoder stack.
struct Stack<B: Backend> {
    config: StackConfig,
    downsample_weights: Option<Tensor<B, 2>>, // [ds, 1]
    out_combiner: Option<Bypass<B>>,
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Stack<B> {
    fn load(
        l: &Loader<B>,
        s: usize,
        stack: &StackConfig,
        pos_dim: usize,
        streaming_chunk50: Option<usize>,
    ) -> Self {
        let ds = stack.downsample;
        let streaming_chunk = streaming_chunk50.map(|c| c / ds);
        let layers = (0..stack.num_layers)
            .map(|layer| {
                EncoderLayer::load(
                    l,
                    &StackConfig::layer_prefix(s, ds, layer),
                    stack,
                    pos_dim,
                    streaming_chunk,
                )
            })
            .collect();
        let (downsample_weights, out_combiner) = if ds > 1 {
            let weights = l
                .tensor::<1>(
                    &format!("encoder.encoders.{s}.downsample.weights"),
                    [ds],
                )
                .reshape([ds, 1]);
            let combiner = Bypass::load(
                l,
                &format!("encoder.encoders.{s}.out_combiner.bypass_scale"),
                stack.encoder_dim,
            );
            (Some(weights), Some(combiner))
        } else {
            (None, None)
        };
        Self {
            config: stack.clone(),
            downsample_weights,
            out_combiner,
            layers,
        }
    }
}

/// `SimpleDownsample`: `[T, C]` → `[⌈T/ds⌉, C]`, padding by repeating the last
/// frame, then a weighted group sum.
fn downsample<B: Backend>(
    x: Tensor<B, 2>,
    weights: &Tensor<B, 2>,
    ds: usize,
) -> Tensor<B, 2> {
    let (t, c) = (x.dims()[0], x.dims()[1]);
    let out_t = t.div_ceil(ds);
    let pad = out_t * ds - t;
    let x = if pad > 0 {
        let last = x.clone().narrow(0, t - 1, 1);
        let mut parts = vec![x];
        for _ in 0..pad {
            parts.push(last.clone());
        }
        Tensor::cat(parts, 0)
    } else {
        x
    };
    let grouped = x.reshape([out_t, ds, c]);
    (grouped * weights.clone().reshape([1, ds, 1]))
        .sum_dim(1)
        .reshape([out_t, c])
}

/// `SimpleUpsample`: repeat each frame `ds` times, truncated to `out_t`.
fn upsample<B: Backend>(
    x: Tensor<B, 2>,
    ds: usize,
    out_t: usize,
) -> Tensor<B, 2> {
    let (t, c) = (x.dims()[0], x.dims()[1]);
    let up = x.reshape([t, 1, c]).expand([t, ds, c]).reshape([t * ds, c]);
    up.narrow(0, 0, out_t)
}

/// `convert_num_channels`: truncate or zero-pad the channel axis.
fn convert_channels<B: Backend>(
    x: Tensor<B, 2>,
    channels: usize,
) -> Tensor<B, 2> {
    let (t, c) = (x.dims()[0], x.dims()[1]);
    if channels <= c {
        x.narrow(1, 0, channels)
    } else {
        let zeros = Tensor::zeros([t, channels - c], &x.device());
        Tensor::cat(vec![x, zeros], 1)
    }
}

/// The compact relative positional embedding `[left + 2T − 1, pos_dim]`,
/// computed on the host by the reference formula and uploaded.
fn pos_embedding<B: Backend>(
    t: usize,
    left_context: usize,
    pos_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let total = t + left_context;
    let len = total + t - 1;
    let compression = (pos_dim as f64).sqrt();
    let length_scale = pos_dim as f64 / (2.0 * std::f64::consts::PI);
    let half = pos_dim / 2;
    let mut pe = vec![0.0f32; len * pos_dim];
    for (row, slot) in pe.chunks_exact_mut(pos_dim).enumerate() {
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
    Tensor::from_data(TensorData::new(pe, [len, pos_dim]), device)
}

/// The convolutional front end (`encoder_embed`).
struct Embed<B: Backend> {
    conv0_w: Tensor<B, 4>,
    conv0_b: Tensor<B, 1>,
    conv4_w: Tensor<B, 4>,
    conv4_b: Tensor<B, 1>,
    conv7_w: Tensor<B, 4>,
    conv7_b: Tensor<B, 1>,
    convnext_dw_w: Tensor<B, 4>, // [128, 1, 7, 7]
    convnext_dw_b: Tensor<B, 1>,
    convnext_pw1: Linear<B>,
    convnext_pw2: Linear<B>,
    out: Linear<B>,
    out_norm: BiasNorm<B>,
    out_width: usize,
}

/// The streaming cache of the front end: the ConvNeXt left context
/// `[128, 3, freq]`.
pub struct EmbedState<B: Backend> {
    cached: Tensor<B, 3>,
}

impl<B: Backend> Embed<B> {
    fn load(l: &Loader<B>, config: &ZipformerConfig) -> Self {
        let d0 = config.stacks[0].encoder_dim;
        let out_width = (((config.feature_dim - 1) / 2) - 1) / 2;
        Self {
            conv0_w: l.tensor::<4>("encoder_embed.conv.0.weight", [8, 1, 3, 3]),
            conv0_b: l.tensor::<1>("encoder_embed.conv.0.bias", [8]),
            conv4_w: l
                .tensor::<4>("encoder_embed.conv.4.weight", [32, 8, 3, 3]),
            conv4_b: l.tensor::<1>("encoder_embed.conv.4.bias", [32]),
            conv7_w: l
                .tensor::<4>("encoder_embed.conv.7.weight", [128, 32, 3, 3]),
            conv7_b: l.tensor::<1>("encoder_embed.conv.7.bias", [128]),
            convnext_dw_w: l.tensor::<4>(
                "encoder_embed.convnext.depthwise_conv.weight",
                [128, 1, 7, 7],
            ),
            convnext_dw_b: l.tensor::<1>(
                "encoder_embed.convnext.depthwise_conv.bias",
                [128],
            ),
            convnext_pw1: Linear::load(
                l,
                "encoder_embed.convnext.pointwise_conv1",
                384,
                128,
            ),
            convnext_pw2: Linear::load(
                l,
                "encoder_embed.convnext.pointwise_conv2",
                128,
                384,
            ),
            out: Linear::load(l, "encoder_embed.out", d0, 128 * out_width),
            out_norm: BiasNorm::load(l, "encoder_embed.out_norm", d0),
            out_width,
        }
    }

    /// The three strided conv stages: `[1, 1, T, feat]` →
    /// `[1, 128, (T−7)/2, out_width]`.
    fn conv_stages(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Stage 1: k3, pad (0, 1).
        let x = conv2d(
            x,
            self.conv0_w.clone(),
            Some(self.conv0_b.clone()),
            ConvOptions::new([1, 1], [0, 1], [1, 1], 1),
        );
        let x = swoosh_r(x);
        // Stage 2: k3, stride 2, no pad.
        let x = conv2d(
            x,
            self.conv4_w.clone(),
            Some(self.conv4_b.clone()),
            ConvOptions::new([2, 2], [0, 0], [1, 1], 1),
        );
        let x = swoosh_r(x);
        // Stage 3: k3, stride (1, 2), no pad.
        let x = conv2d(
            x,
            self.conv7_w.clone(),
            Some(self.conv7_b.clone()),
            ConvOptions::new([1, 2], [0, 0], [1, 1], 1),
        );
        swoosh_r(x)
    }

    /// The ConvNeXt block over `[1, 128, T, F]`. Offline pads time
    /// symmetrically; streaming carries the 3-frame left context and drops
    /// the 3-frame right margin, so the output is 6 frames shorter.
    fn convnext(&self, x: Tensor<B, 4>, pad_time: bool) -> Tensor<B, 4> {
        let bypass = if pad_time {
            x.clone()
        } else {
            let t = x.dims()[2];
            x.clone().narrow(2, 3, t - 6)
        };
        let padded = if pad_time {
            x.pad([(3, 3), (3, 3)], PadMode::Constant(0.0))
        } else {
            x.pad([(0, 0), (3, 3)], PadMode::Constant(0.0))
        };
        // Native depthwise 7×7.
        let dw = conv2d(
            padded,
            self.convnext_dw_w.clone(),
            Some(self.convnext_dw_b.clone()),
            ConvOptions::new([1, 1], [0, 0], [1, 1], 128),
        );
        // Pointwise 1×1 convs as matmuls over the channel axis.
        let [b, c, t, f] = dw.dims();
        let flat = dw.swap_dims(1, 3).swap_dims(1, 2).reshape([b * t * f, c]);
        let h = swoosh_l(self.convnext_pw1.forward(flat));
        let out = self.convnext_pw2.forward(h);
        let out = out.reshape([b, t, f, 128]).swap_dims(1, 3).swap_dims(2, 3);
        bypass + out
    }

    /// Flatten channels×freq, linear, BiasNorm → `[T', d0]`.
    fn output(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [_, c, t, f] = x.dims();
        let flat = x.swap_dims(1, 2).reshape([t, c * f]);
        self.out_norm.forward(self.out.forward(flat))
    }

    /// Offline: `[T, feat]` → `[(T−7)/2, d0]`.
    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let (t, feat) = (features.dims()[0], features.dims()[1]);
        let x = features.reshape([1, 1, t, feat]);
        let x = self.conv_stages(x);
        let x = self.convnext(x, true);
        self.output(x)
    }

    /// Streaming: a window whose conv output covers `chunk50 + 3` frames.
    fn forward_streaming(
        &self,
        features: Tensor<B, 2>,
        state: &mut EmbedState<B>,
    ) -> Tensor<B, 2> {
        let (t, feat) = (features.dims()[0], features.dims()[1]);
        let x = features.reshape([1, 1, t, feat]);
        let x = self.conv_stages(x); // [1, 128, (T−7)/2, F]
        let conv_t = x.dims()[2];
        let cached = state.cached.clone().unsqueeze::<4>();
        let with_left = Tensor::cat(vec![cached, x], 2);
        state.cached = with_left
            .clone()
            .narrow(2, conv_t - 3, 3)
            .squeeze_dim::<3>(0);
        let x = self.convnext(with_left, false);
        self.output(x)
    }
}

/// The complete streaming state.
pub struct NetState<B: Backend> {
    layers: Vec<LayerState<B>>,
    embed: EmbedState<B>,
    processed: usize,
}

/// The Zipformer2 encoder network on burn.
pub struct ZipformerNet<B: Backend> {
    embed: Embed<B>,
    stacks: Vec<Stack<B>>,
    downsample_output: Tensor<B, 2>, // [2, 1]
    encoder_proj: Linear<B>,
    config: ZipformerConfig,
    device: B::Device,
    /// Per-stack positional embedding for streaming, precomputed once (the
    /// window size and left context are fixed, so it is identical for every
    /// window). Empty for offline models, whose chunk length varies.
    streaming_pos: Vec<Tensor<B, 2>>,
}

impl<B: Backend> ZipformerNet<B> {
    /// Loads the network onto `device`.
    pub fn load(w: &ModelWeights, device: &B::Device) -> Self {
        let l = Loader::new(w, device);
        let config = w.config.clone();
        let chunk50 = config.streaming.as_ref().map(|s| s.shift_frames / 2);
        let embed = Embed::load(&l, &config);
        let stacks = config
            .stacks
            .iter()
            .enumerate()
            .map(|(s, stack)| {
                Stack::load(&l, s, stack, config.pos_dim, chunk50)
            })
            .collect();
        let downsample_output = l
            .tensor::<1>("encoder.downsample_output.weights", [2])
            .reshape([2, 1]);
        let encoder_proj = Linear::load(
            &l,
            "encoder_proj",
            config.joiner_dim,
            config.encoder_out_dim,
        );
        let streaming_pos = match &config.streaming {
            Some(streaming) => {
                let chunk50 = streaming.shift_frames / 2;
                config
                    .stacks
                    .iter()
                    .enumerate()
                    .map(|(s, stack)| {
                        pos_embedding::<B>(
                            chunk50.div_ceil(stack.downsample),
                            streaming.left_context[s],
                            config.pos_dim,
                            device,
                        )
                    })
                    .collect()
            },
            None => Vec::new(),
        };
        Self {
            embed,
            stacks,
            downsample_output,
            encoder_proj,
            config,
            device: device.clone(),
            streaming_pos,
        }
    }

    /// The model geometry.
    pub fn config(&self) -> &ZipformerConfig { &self.config }

    /// Offline forward: `[T, feat]` → `[T25, joiner_dim]`.
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.embed.forward(features);
        let mut outputs = Vec::with_capacity(self.stacks.len());
        for stack in &self.stacks {
            x = convert_channels(x, stack.config.encoder_dim);
            x = self.run_stack(stack, x, None, 0, None, None);
            outputs.push(x.clone());
        }
        let full = self.full_dim_output(outputs);
        let down = downsample(full, &self.downsample_output, 2);
        self.encoder_proj.forward(down)
    }

    /// Runs one stack (downsample → layers → upsample → combine). `pos` is the
    /// precomputed positional table (streaming, window-invariant); when `None`
    /// (offline) it is computed here for the chunk's length.
    fn run_stack(
        &self,
        stack: &Stack<B>,
        x: Tensor<B, 2>,
        mut states: Option<&mut [LayerState<B>]>,
        left_context: usize,
        pos: Option<&Tensor<B, 2>>,
        mask_bias: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let ds = stack.config.downsample;
        let orig = x.clone();
        let mut inner = if ds > 1 {
            downsample(x, stack.downsample_weights.as_ref().expect("ds"), ds)
        } else {
            x
        };
        let computed;
        let pos_emb = match pos {
            Some(pos) => pos,
            None => {
                let t = inner.dims()[0];
                computed = pos_embedding::<B>(
                    t,
                    left_context,
                    self.config.pos_dim,
                    &self.device,
                );
                &computed
            },
        };
        for (layer_idx, layer) in stack.layers.iter().enumerate() {
            let state = states.as_deref_mut().map(|s| &mut s[layer_idx]);
            inner = layer.forward(inner, pos_emb, state, mask_bias);
        }
        if ds > 1 {
            let up = upsample(inner, ds, orig.dims()[0]);
            stack.out_combiner.as_ref().expect("ds").forward(orig, up)
        } else {
            inner
        }
    }

    /// A fresh streaming state.
    pub fn init_state(&self) -> NetState<B> {
        let streaming = self
            .config
            .streaming
            .as_ref()
            .expect("streaming state of a streaming model");
        let mut layers = Vec::new();
        let zeros = |rows: usize, cols: usize| {
            Tensor::<B, 2>::zeros([rows, cols], &self.device)
        };
        for (s, stack) in self.config.stacks.iter().enumerate() {
            let left = streaming.left_context[s];
            for _ in 0..stack.num_layers {
                layers.push(LayerState {
                    cached_key: zeros(
                        left,
                        stack.num_heads * stack.query_head_dim,
                    ),
                    cached_nonlin: zeros(left, stack.nonlin_hidden),
                    cached_val1: zeros(
                        left,
                        stack.num_heads * stack.value_head_dim,
                    ),
                    cached_val2: zeros(
                        left,
                        stack.num_heads * stack.value_head_dim,
                    ),
                    cached_conv1: zeros(
                        stack.encoder_dim,
                        stack.cnn_kernel / 2,
                    ),
                    cached_conv2: zeros(
                        stack.encoder_dim,
                        stack.cnn_kernel / 2,
                    ),
                });
            }
        }
        NetState {
            layers,
            embed: EmbedState {
                cached: Tensor::zeros(
                    [128, 3, self.embed.out_width],
                    &self.device,
                ),
            },
            processed: 0,
        }
    }

    /// Streaming forward: one window → `[chunk50/2, joiner_dim]`.
    pub fn forward_streaming(
        &self,
        features: Tensor<B, 2>,
        state: &mut NetState<B>,
    ) -> Tensor<B, 2> {
        let streaming =
            self.config.streaming.as_ref().expect("streaming model");
        let chunk50 = streaming.shift_frames / 2;
        let mut x = self.embed.forward_streaming(features, &mut state.embed);

        let mut outputs = Vec::with_capacity(self.stacks.len());
        let mut layer_base = 0usize;
        for (s, stack) in self.stacks.iter().enumerate() {
            let left = streaming.left_context[s];
            x = convert_channels(x, stack.config.encoder_dim);
            let mask = self.mask_bias(state.processed, s);
            let count = stack.layers.len();
            let states = &mut state.layers[layer_base..layer_base + count];
            x = self.run_stack(
                stack,
                x,
                Some(states),
                left,
                Some(&self.streaming_pos[s]),
                mask.as_ref(),
            );
            outputs.push(x.clone());
            layer_base += count;
        }
        state.processed += chunk50;

        let full = self.full_dim_output(outputs);
        let down = downsample(full, &self.downsample_output, 2);
        self.encoder_proj.forward(down)
    }

    /// The additive attention mask over stack `s` (−1000 on unfilled cache
    /// slots).
    fn mask_bias(&self, processed: usize, s: usize) -> Option<Tensor<B, 2>> {
        let streaming = self.config.streaming.as_ref().expect("streaming");
        let left_top =
            streaming.left_context[0] * self.config.stacks[0].downsample;
        if processed >= left_top {
            return None;
        }
        let chunk50 = streaming.shift_frames / 2;
        let ds = self.config.stacks[s].downsample;
        // The stack's decimated view: left_context[s] cached slots plus the
        // stack's chunk (ceil, matching the padded downsample).
        let masked_head = left_top - processed;
        let src = streaming.left_context[s] + chunk50.div_ceil(ds);
        let mut bias = vec![0.0f32; src];
        for (i, slot) in bias.iter_mut().enumerate() {
            if i * ds < masked_head {
                *slot = -1000.0;
            }
        }
        let bias: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(bias, [src]), &self.device);
        Some(bias.reshape([1, src]))
    }

    /// `_get_full_dim_output`.
    fn full_dim_output(&self, outputs: Vec<Tensor<B, 2>>) -> Tensor<B, 2> {
        let dims: Vec<usize> =
            self.config.stacks.iter().map(|s| s.encoder_dim).collect();
        let mut pieces = vec![outputs[outputs.len() - 1].clone()];
        let mut cur = dims[dims.len() - 1];
        for i in (0..dims.len() - 1).rev() {
            if dims[i] > cur {
                pieces.push(outputs[i].clone().narrow(1, cur, dims[i] - cur));
                cur = dims[i];
            }
        }
        debug_assert_eq!(cur, self.config.encoder_out_dim);
        if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            Tensor::cat(pieces, 1)
        }
    }

    /// Reads an encoder output `[T', joiner_dim]` back as flat `f32` with its
    /// frame count.
    pub fn read_back(encoded: Tensor<B, 2>) -> (Vec<f32>, usize) {
        let frames = encoded.dims()[0];
        let data = encoded
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 readback");
        (data, frames)
    }
}
