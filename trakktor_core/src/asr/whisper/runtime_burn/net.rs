//! The Whisper network on burn.
//!
//! A port of the candle network in `runtime::net` (itself adapted from the
//! candle project's Whisper implementation, Apache-2.0 OR MIT) with the same
//! numerical semantics, written idiomatically for burn rather than as a
//! mirror:
//!
//! - attention runs through burn's fused scaled-dot-product `attention` (its
//!   default scale is the reference's split `(d_head)^-0.25` pair) everywhere
//!   raw scores are not needed; the manual matmul path survives only in the
//!   timing forward, which must capture pre-softmax cross-QK;
//! - the decoder self-attention cache lives in the fused kernel's native
//!   `(batch, head, positions, d_head)` layout and grows by `cat` — no
//!   pre-scaled transposed-K trick, the kernel handles both internally;
//! - cross-attention K/V are projected once per session from the *unreplicated*
//!   audio features (batch 1), and beam-search rows fold into the query axis of
//!   the fused call, so K/V are never replicated across the group;
//! - the per-step q/k/v projections are one fused linear (the checkpoint's
//!   three matrices are concatenated at load; the bias of the bias-less key
//!   projection is zero-filled);
//! - the tied token embedding is stored once, transposed to `[n_state,
//!   n_vocab]`: the layout serves both the embedding gather and the vocabulary
//!   projection.
//!
//! Layer normalization keeps its statistics in `f32` (as candle does), and
//! the encoder's sinusoidal positions are computed on the host with the
//! reference formula.
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
use crate::asr::whisper::{
    error::WhisperError, model::ModelDims, tokenizer::TokenId,
};

/// Reads a checkpoint tensor of the given shape as a burn tensor (in the
/// backend's compute dtype).
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, WhisperError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(WhisperError::InvalidModel(format!(
            "{key}: shape {dims:?}, expected {shape:?}"
        )));
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

/// Reads a `[out, in]`-shaped checkpoint weight as row-major f32 values,
/// shape-checked.
fn linear_values(
    weights: &Weights,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, WhisperError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(WhisperError::InvalidModel(format!(
            "{key}: shape {dims:?}, expected {:?}",
            [out_dim, in_dim]
        )));
    }
    Ok(values)
}

/// A linear layer with candle/PyTorch semantics: `y = x·Wᵀ + b`. The
/// checkpoint's `[out, in]` weight is transposed once at load into burn's
/// `[in, out]` layout, and the forward pass is burn's `linear` primitive.
pub(super) struct Linear<B: Backend> {
    pub(super) weight: Tensor<B, 2>, // [in, out]
    pub(super) bias: Option<Tensor<B, 1>>, // [out]
}

impl<B: Backend> Linear<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
    ) -> Result<Self, WhisperError> {
        let values = linear_values(
            weights,
            &format!("{prefix}.weight"),
            out_dim,
            in_dim,
        )?;
        let transposed = transpose_2d(&values, out_dim, in_dim);
        let bias = if with_bias {
            Some(weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_dim],
            )?)
        } else {
            None
        };
        Ok(Self {
            weight: Tensor::from_data(
                TensorData::new(transposed, [in_dim, out_dim]),
                device,
            ),
            bias,
        })
    }

    /// `[B, T, in]` -> `[B, T, out]`.
    pub(super) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(x, self.weight.clone(), self.bias.clone())
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
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        d: usize,
    ) -> Result<Self, WhisperError> {
        Ok(Self {
            gamma: weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.weight"),
                [d],
            )?
            .reshape([1, 1, d]),
            beta: weight::<B, 1>(
                weights,
                device,
                &format!("{prefix}.bias"),
                [d],
            )?
            .reshape([1, 1, d]),
            eps: 1e-5,
        })
    }

    /// `[B, T, d]` -> `[B, T, d]`, normalized over `d`.
    pub(super) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = (centered.clone() * centered.clone()).mean_dim(2);
        let normed = centered / (var.add_scalar(self.eps)).sqrt();
        normed.cast(dtype) * self.gamma.clone() + self.beta.clone()
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
    #[allow(clippy::too_many_arguments)]
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self, WhisperError> {
        Ok(Self {
            weight: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [out_ch, in_ch, kernel],
            )?,
            bias: weight(weights, device, &format!("{prefix}.bias"), [out_ch])?,
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

/// The MLP half of a residual block: `fc2(gelu(fc1(ln(x))))`, with the
/// reference's exact (erf) GELU.
struct Mlp<B: Backend> {
    ln: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        n_state: usize,
    ) -> Result<Self, WhisperError> {
        let n_mlp = n_state * 4;
        Ok(Self {
            ln: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.final_layer_norm"),
                n_state,
            )?,
            fc1: Linear::load(
                weights,
                device,
                &format!("{prefix}.fc1"),
                n_state,
                n_mlp,
                true,
            )?,
            fc2: Linear::load(
                weights,
                device,
                &format!("{prefix}.fc2"),
                n_mlp,
                n_state,
                true,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2
            .forward(activation::gelu(self.fc1.forward(self.ln.forward(x))))
    }
}

/// `[B, T, d]` -> `[B, n_head, T, d_head]`.
fn split_heads<B: Backend>(x: Tensor<B, 3>, n_head: usize) -> Tensor<B, 4> {
    let [b, t, d] = x.dims();
    x.reshape([b, t, n_head, d / n_head]).swap_dims(1, 2)
}

/// `[B, n_head, T, d_head]` -> `[B, T, d]`.
fn merge_heads<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 3> {
    let [b, h, t, dh] = x.dims();
    x.swap_dims(1, 2).reshape([b, t, h * dh])
}

/// The additive causal bias for a multi-token step over a cached prefix:
/// query row `i` may attend key columns `..= offset + i`.
fn causal_bias<B: Backend>(
    offset: usize,
    n_step: usize,
    k_len: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut bias = vec![0.0f32; n_step * k_len];
    for row in 0..n_step {
        for col in (offset + row + 1)..k_len {
            bias[row * k_len + col] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_data(TensorData::new(bias, [1, 1, n_step, k_len]), device)
}

/// Self-attention with the checkpoint's three projections fused into one
/// `[d, 3d]` linear (the key projection's missing bias is zero-filled, which
/// is the same arithmetic). The attended context comes from burn's fused
/// scaled-dot-product `attention`.
pub(super) struct SelfAttention<B: Backend> {
    pub(super) qkv: Linear<B>, // [d, 3d]: columns are [q | k | v]
    pub(super) out: Linear<B>,
    pub(super) n_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, WhisperError> {
        let d = n_state;
        let q =
            linear_values(weights, &format!("{prefix}.q_proj.weight"), d, d)?;
        let k =
            linear_values(weights, &format!("{prefix}.k_proj.weight"), d, d)?;
        let v =
            linear_values(weights, &format!("{prefix}.v_proj.weight"), d, d)?;
        let (q_t, k_t, v_t) = (
            transpose_2d(&q, d, d),
            transpose_2d(&k, d, d),
            transpose_2d(&v, d, d),
        );
        // Row i of the fused [d, 3d] weight is [q_t[i], k_t[i], v_t[i]].
        let mut fused = Vec::with_capacity(3 * d * d);
        for row in 0..d {
            fused.extend_from_slice(&q_t[row * d..(row + 1) * d]);
            fused.extend_from_slice(&k_t[row * d..(row + 1) * d]);
            fused.extend_from_slice(&v_t[row * d..(row + 1) * d]);
        }
        let (q_bias, dims) = weights.parts(&format!("{prefix}.q_proj.bias"))?;
        let (v_bias, v_dims) =
            weights.parts(&format!("{prefix}.v_proj.bias"))?;
        if dims != [d] || v_dims != [d] {
            return Err(WhisperError::InvalidModel(format!(
                "{prefix}: q/v bias shapes {dims:?}/{v_dims:?}, expected [{d}]"
            )));
        }
        let mut bias = q_bias;
        bias.extend(std::iter::repeat_n(0.0f32, d)); // k_proj has no bias
        bias.extend(v_bias);
        Ok(Self {
            qkv: Linear {
                weight: Tensor::from_data(
                    TensorData::new(fused, [d, 3 * d]),
                    device,
                ),
                bias: Some(Tensor::from_data(
                    TensorData::new(bias, [3 * d]),
                    device,
                )),
            },
            out: Linear::load(
                weights,
                device,
                &format!("{prefix}.out_proj"),
                d,
                d,
                true,
            )?,
            n_head,
        })
    }

    /// The fused projection split into per-head q/k/v.
    fn project(
        &self,
        x: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let d = self.qkv.weight.dims()[0];
        let qkv = self.qkv.forward(x); // [B, T, 3d]
        (
            split_heads(qkv.clone().narrow(2, 0, d), self.n_head),
            split_heads(qkv.clone().narrow(2, d, d), self.n_head),
            split_heads(qkv.narrow(2, 2 * d, d), self.n_head),
        )
    }

    /// One-shot self-attention over `x` alone: the encoder (no mask) and the
    /// timing forward (causal). With equal query/key lengths the kernel's
    /// causal mode matches the reference's lower-triangular mask exactly.
    pub(super) fn forward_plain(
        &self,
        x: Tensor<B, 3>,
        causal: bool,
    ) -> Tensor<B, 3> {
        let (q, k, v) = self.project(x);
        let ctx = attention(
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                is_causal: causal,
                ..Default::default()
            },
        );
        self.out.forward(merge_heads(ctx))
    }

    /// Incremental self-attention: appends the fed positions' K/V to the
    /// session cache and attends over the whole cached prefix.
    ///
    /// The first call of a session feeds the whole initial sequence with an
    /// empty cache (query and key lengths equal — the kernel's causal mode);
    /// later calls feed one token, which attends the full prefix unmasked. A
    /// multi-token step over a non-empty cache (legal for the contract,
    /// never produced by the decoding loop) gets an explicit additive bias.
    pub(super) fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: &mut Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let n_step = x.dims()[1];
        let (q, k_new, v_new) = self.project(x);
        let (k, v) = match cache.take() {
            Some((k_prev, v_prev)) => (
                Tensor::cat(vec![k_prev, k_new], 2),
                Tensor::cat(vec![v_prev, v_new], 2),
            ),
            None => (k_new, v_new),
        };
        *cache = Some((k.clone(), v.clone()));
        let ctx = if n_step == 1 {
            // A single query attends the whole prefix, itself included.
            attention(q, k, v, None, None, AttentionModuleOptions::default())
        } else if offset == 0 {
            attention(
                q,
                k,
                v,
                None,
                None,
                AttentionModuleOptions {
                    is_causal: true,
                    ..Default::default()
                },
            )
        } else {
            let k_len = k.dims()[2];
            let bias = causal_bias::<B>(offset, n_step, k_len, &q.device());
            attention(
                q,
                k,
                v,
                None,
                Some(bias),
                AttentionModuleOptions::default(),
            )
        };
        self.out.forward(merge_heads(ctx))
    }
}

/// Cross-attention over encoded audio. K/V are projected from the
/// *unreplicated* features (batch 1); every decode step folds its batch rows
/// into the query axis of the fused call, so the 1500-frame K/V are shared by
/// the whole group instead of being replicated per beam.
pub(super) struct CrossAttention<B: Backend> {
    pub(super) q: Linear<B>,
    pub(super) k: Linear<B>,
    pub(super) v: Linear<B>,
    pub(super) out: Linear<B>,
    pub(super) n_head: usize,
}

impl<B: Backend> CrossAttention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, WhisperError> {
        let d = n_state;
        let lin = |name: &str, with_bias: bool| {
            Linear::load(
                weights,
                device,
                &format!("{prefix}.{name}"),
                d,
                d,
                with_bias,
            )
        };
        Ok(Self {
            q: lin("q_proj", true)?,
            k: lin("k_proj", false)?,
            v: lin("v_proj", true)?,
            out: lin("out_proj", true)?,
            n_head,
        })
    }

    /// Projects the session-constant K/V from `[1, frames, d]` features.
    pub(super) fn project_kv(
        &self,
        features: &Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (
            split_heads(self.k.forward(features.clone()), self.n_head),
            split_heads(self.v.forward(features.clone()), self.n_head),
        )
    }

    /// Cross-attention of a step: the batch rows of `x` fold into the query
    /// axis, attending the shared batch-1 K/V.
    pub(super) fn forward_folded(
        &self,
        x: Tensor<B, 3>,
        kv: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let (h, dh) = (self.n_head, d / self.n_head);
        // [b, h, t, dh] -> [h, b, t, dh] -> [1, h, b*t, dh]: rows stay
        // (batch, position) row-major, heads are preserved.
        let q = split_heads(self.q.forward(x), h).swap_dims(0, 1).reshape([
            1,
            h,
            b * t,
            dh,
        ]);
        let ctx = attention(
            q,
            kv.0.clone(),
            kv.1.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        );
        let ctx = ctx.reshape([h, b, t, dh]).swap_dims(0, 1); // [b, h, t, dh]
        self.out.forward(merge_heads(ctx))
    }

    /// The timing forward: manual attention that returns the raw pre-softmax
    /// scores `[1, head, tokens, frames]` (scaled, as the reference captures
    /// them) alongside the attended output. Batch is always 1 here.
    pub(super) fn forward_capture(
        &self,
        x: Tensor<B, 3>,
        kv: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let d = x.dims()[2];
        let dh = d / self.n_head;
        let q = split_heads(self.q.forward(x), self.n_head);
        let scores = q
            .matmul(kv.0.clone().swap_dims(2, 3))
            .mul_scalar((dh as f64).powf(-0.5));
        let captured = scores.clone();
        let dtype = scores.dtype();
        let w = activation::softmax(scores.cast(DType::F32), 3).cast(dtype);
        let ctx = w.matmul(kv.1.clone());
        (self.out.forward(merge_heads(ctx)), captured)
    }
}

/// One encoder block: pre-LN self-attention and the MLP.
struct EncoderBlock<B: Backend> {
    attn_ln: LayerNorm<B>,
    attn: SelfAttention<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> EncoderBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, WhisperError> {
        Ok(Self {
            attn_ln: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.self_attn_layer_norm"),
                n_state,
            )?,
            attn: SelfAttention::load(
                weights,
                device,
                &format!("{prefix}.self_attn"),
                n_state,
                n_head,
            )?,
            mlp: Mlp::load(weights, device, prefix, n_state)?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self
            .attn
            .forward_plain(self.attn_ln.forward(x.clone()), false);
        let x = x + attn;
        let mlp = self.mlp.forward(x.clone());
        x + mlp
    }
}

/// The per-layer decoding-session state: the growing self-attention K/V and
/// the session-constant batch-1 cross-attention K/V.
pub(super) struct LayerKv<B: Backend> {
    pub(super) self_kv: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    pub(super) cross_kv: (Tensor<B, 4>, Tensor<B, 4>),
}

/// One decoder block: self-attention, cross-attention, MLP.
struct DecoderBlock<B: Backend> {
    attn_ln: LayerNorm<B>,
    attn: SelfAttention<B>,
    cross_ln: LayerNorm<B>,
    cross: CrossAttention<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> DecoderBlock<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        n_state: usize,
        n_head: usize,
    ) -> Result<Self, WhisperError> {
        Ok(Self {
            attn_ln: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.self_attn_layer_norm"),
                n_state,
            )?,
            attn: SelfAttention::load(
                weights,
                device,
                &format!("{prefix}.self_attn"),
                n_state,
                n_head,
            )?,
            cross_ln: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.encoder_attn_layer_norm"),
                n_state,
            )?,
            cross: CrossAttention::load(
                weights,
                device,
                &format!("{prefix}.encoder_attn"),
                n_state,
                n_head,
            )?,
            mlp: Mlp::load(weights, device, prefix, n_state)?,
        })
    }

    /// Session forward over the fed positions, growing the self-K/V cache.
    fn forward_session(
        &self,
        x: Tensor<B, 3>,
        kv: &mut LayerKv<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        let attn = self.attn.forward_cached(
            self.attn_ln.forward(x.clone()),
            &mut kv.self_kv,
            offset,
        );
        let x = x + attn;
        let cross = self
            .cross
            .forward_folded(self.cross_ln.forward(x.clone()), &kv.cross_kv);
        let x = x + cross;
        let mlp = self.mlp.forward(x.clone());
        x + mlp
    }

    /// The timing forward: stateless, causal self-attention, and raw
    /// cross-attention scores captured.
    fn forward_capture(
        &self,
        x: Tensor<B, 3>,
        cross_kv: &(Tensor<B, 4>, Tensor<B, 4>),
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let attn = self
            .attn
            .forward_plain(self.attn_ln.forward(x.clone()), true);
        let x = x + attn;
        let (cross, captured) = self
            .cross
            .forward_capture(self.cross_ln.forward(x.clone()), cross_kv);
        let x = x + cross;
        let mlp = self.mlp.forward(x.clone());
        (x + mlp, captured)
    }
}

/// The reference's sinusoidal positional embeddings, row-major
/// `[length, channels]` with `[sin | cos]` halves per row.
pub(super) fn sinusoids(length: usize, channels: usize) -> Vec<f32> {
    let max_timescale = 10000f32;
    let increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<f32> = (0..channels / 2)
        .map(|i| (i as f32 * -increment).exp())
        .collect();
    let mut out = Vec::with_capacity(length * channels);
    for position in 0..length {
        for &inv in &inv_timescales {
            out.push((position as f32 * inv).sin());
        }
        for &inv in &inv_timescales {
            out.push((position as f32 * inv).cos());
        }
    }
    out
}

/// The audio encoder: two convolutions with exact GELU, sinusoidal
/// positions, transformer blocks, and a final layer norm.
pub struct AudioEncoder<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    positions: Tensor<B, 3>, // [1, n_audio_ctx, d]
    blocks: Vec<EncoderBlock<B>>,
    ln_post: LayerNorm<B>,
}

impl<B: Backend> AudioEncoder<B> {
    pub(super) fn load(
        weights: &Weights,
        dims: &ModelDims,
        device: &B::Device,
    ) -> Result<Self, WhisperError> {
        let d = dims.n_audio_state;
        let conv1 = Conv::load(
            weights,
            device,
            "model.encoder.conv1",
            d,
            dims.n_mels,
            3,
            1,
            1,
        )?;
        let conv2 =
            Conv::load(weights, device, "model.encoder.conv2", d, d, 3, 2, 1)?;
        let positions = Tensor::from_data(
            TensorData::new(
                sinusoids(dims.n_audio_ctx, d),
                [1, dims.n_audio_ctx, d],
            ),
            device,
        );
        let blocks = (0..dims.n_audio_layer)
            .map(|i| {
                EncoderBlock::load(
                    weights,
                    device,
                    &format!("model.encoder.layers.{i}"),
                    d,
                    dims.n_audio_head,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ln_post =
            LayerNorm::load(weights, device, "model.encoder.layer_norm", d)?;
        Ok(Self {
            conv1,
            conv2,
            positions,
            blocks,
            ln_post,
        })
    }

    /// `[B, n_mels, 3000]` log-mel input -> `[B, n_audio_ctx, d]` features.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = activation::gelu(self.conv1.forward(mel));
        let x = activation::gelu(self.conv2.forward(x));
        let x = x.swap_dims(1, 2);
        let t = x.dims()[1];
        let mut x = x + self.positions.clone().narrow(1, 0, t);
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.ln_post.forward(x)
    }
}

/// The text decoder: the tied token embedding (stored transposed, serving
/// both the lookup and the vocabulary projection), learned positions,
/// transformer blocks with cross-attention, and a final layer norm.
pub struct TextDecoder<B: Backend> {
    embedding_t: Tensor<B, 2>, // [d, n_vocab]
    positions: Tensor<B, 3>,   // [1, n_text_ctx, d]
    blocks: Vec<DecoderBlock<B>>,
    ln: LayerNorm<B>,
    n_state: usize,
}

impl<B: Backend> TextDecoder<B> {
    pub(super) fn load(
        weights: &Weights,
        dims: &ModelDims,
        device: &B::Device,
    ) -> Result<Self, WhisperError> {
        let d = dims.n_text_state;
        let embedding = linear_values(
            weights,
            "model.decoder.embed_tokens.weight",
            dims.n_vocab,
            d,
        )?;
        let embedding_t = Tensor::from_data(
            TensorData::new(
                transpose_2d(&embedding, dims.n_vocab, d),
                [d, dims.n_vocab],
            ),
            device,
        );
        let positions = weight::<B, 2>(
            weights,
            device,
            "model.decoder.embed_positions.weight",
            [dims.n_text_ctx, d],
        )?
        .reshape([1, dims.n_text_ctx, d]);
        let blocks = (0..dims.n_text_layer)
            .map(|i| {
                DecoderBlock::load(
                    weights,
                    device,
                    &format!("model.decoder.layers.{i}"),
                    d,
                    dims.n_text_head,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ln =
            LayerNorm::load(weights, device, "model.decoder.layer_norm", d)?;
        Ok(Self {
            embedding_t,
            positions,
            blocks,
            ln,
            n_state: d,
        })
    }

    /// Starts a decoding session: batch-1 cross-attention K/V of every layer,
    /// empty self-attention caches.
    pub(super) fn begin_session(
        &self,
        features: &Tensor<B, 3>,
    ) -> Vec<LayerKv<B>> {
        self.blocks
            .iter()
            .map(|block| LayerKv {
                self_kv: None,
                cross_kv: block.cross.project_kv(features),
            })
            .collect()
    }

    /// Token embeddings plus positions for the fed step.
    fn embed(
        &self,
        tokens: &[TokenId],
        n_batch: usize,
        offset: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let n_step = tokens.len() / n_batch;
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        let ids = Tensor::<B, 1, Int>::from_data(
            TensorData::new(ids, [tokens.len()]),
            device,
        );
        // Gather columns of the transposed embedding, then shape the rows
        // back into (batch, position, d).
        let embedded = self
            .embedding_t
            .clone()
            .select(1, ids)
            .swap_dims(0, 1)
            .reshape([n_batch, n_step, self.n_state]);
        embedded + self.positions.clone().narrow(1, offset, n_step)
    }

    /// Session forward: `tokens` are the fed positions (`n_batch` rows,
    /// row-major), placed at `offset` after the cached prefix. Returns hidden
    /// states for the fed positions.
    pub(super) fn forward_session(
        &self,
        tokens: &[TokenId],
        n_batch: usize,
        offset: usize,
        caches: &mut [LayerKv<B>],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let mut x = self.embed(tokens, n_batch, offset, device);
        for (block, kv) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward_session(x, kv, offset);
        }
        self.ln.forward(x)
    }

    /// One-shot forward over a complete batch-1 sequence, returning hidden
    /// states and each layer's raw pre-softmax cross-attention scores.
    pub(super) fn forward_capture(
        &self,
        tokens: &[TokenId],
        features: &Tensor<B, 3>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let mut x = self.embed(tokens, 1, 0, device);
        let mut captured = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let cross_kv = block.cross.project_kv(features);
            let (next, qk) = block.forward_capture(x, &cross_kv);
            x = next;
            captured.push(qk);
        }
        (self.ln.forward(x), captured)
    }

    /// Logits over the vocabulary, tied to the token embedding.
    pub(super) fn logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(hidden, self.embedding_t.clone(), None)
    }
}
