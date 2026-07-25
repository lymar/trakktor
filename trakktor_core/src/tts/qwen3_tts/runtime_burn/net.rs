//! The talker and its code predictor on burn.
//!
//! A port of the candle network in `runtime::talker` with the same numerical
//! semantics, written idiomatically for burn rather than as a mirror. What
//! differs, and why:
//!
//! - **Attention runs through burn's fused scaled-dot-product `attention`**. A
//!   frame's whole prompt arrives as one causal block and every later step is a
//!   single query over the cache, which is exactly the kernel's two fast paths.
//! - **The projections of a layer are fused at load**: `q`/`k`/`v` become one
//!   matrix and `gate`/`up` another, so a decode step issues two matmuls per
//!   layer instead of five. Generation is one token at a time, where kernel
//!   launches, not arithmetic, set the pace.
//! - **The key/value cache is preallocated and written in place**, doubling
//!   when it runs out, instead of being rebuilt by concatenation every step.
//! - **All sixteen codebook tables are one table.** The talker's own codec
//!   embedding is followed by the predictor's fifteen, so folding a finished
//!   frame is a single gather plus a sum rather than sixteen gathers and
//!   fifteen adds.
//! - Grouped-query attention repeats the key/value heads before the fused call;
//!   RMS normalization keeps its statistics in f32, as candle does.
//!
//! The networks are written for any element type, so the casts that hold the
//! normalization statistics and the attention scores in full precision are
//! spelled out even though this runtime's backends are all f32 today and skip
//! them.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.

use burn::tensor::{
    DType, Int, Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, embedding, linear},
    ops::AttentionModuleOptions,
};

use super::{Weights, transpose_2d};
use crate::tts::qwen3_tts::{
    config::{CodePredictorConfig, TalkerConfig},
    error::Qwen3TtsError,
    prompt::Position,
    runtime::model_err,
};

/// Positions a freshly grown key/value cache makes room for. The code
/// predictor needs sixteen and the talker as many as it generates, so the
/// floor is set for the small case and doubling covers the large one.
pub(super) const CACHE_MIN_CAPACITY: usize = 32;

/// Reads a checkpoint tensor of the given shape as a burn tensor.
pub(super) fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, Qwen3TtsError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Reads an `[out, in]` checkpoint matrix as burn's `[in, out]` layout.
pub(super) fn matrix<B: Backend>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor<B, 2>, Qwen3TtsError> {
    let values = rows_of(weights, key, out_dim, in_dim)?;
    Ok(Tensor::from_data(
        TensorData::new(
            transpose_2d(&values, out_dim, in_dim),
            [in_dim, out_dim],
        ),
        device,
    ))
}

/// Reads an `[out, in]` checkpoint matrix as row-major values, shape-checked.
pub(super) fn rows_of(
    weights: &Weights,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, Qwen3TtsError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
        ));
    }
    Ok(values)
}

/// Glues several `[out_i, in]` checkpoint matrices into one `[in, Σ out_i]`
/// weight whose column blocks are the originals, in order.
pub(super) fn glue_columns(
    blocks: &[(Vec<f32>, usize)],
    in_dim: usize,
) -> Vec<f32> {
    let total: usize = blocks.iter().map(|(_, out)| *out).sum();
    let mut glued = vec![0.0f32; in_dim * total];
    let mut offset = 0;
    for (values, out) in blocks {
        let transposed = transpose_2d(values, *out, in_dim);
        for row in 0..in_dim {
            let at = row * total + offset;
            glued[at..at + out]
                .copy_from_slice(&transposed[row * out..(row + 1) * out]);
        }
        offset += out;
    }
    glued
}

/// Root-mean-square normalization with a learned per-channel gain.
///
/// Normalizes over the last axis, with the statistics in f32 and the result
/// cast back before the gain is applied — the same order candle uses.
pub(super) struct RmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    size: usize,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        key: &str,
        size: usize,
        eps: f64,
    ) -> Result<Self, Qwen3TtsError> {
        Ok(Self {
            weight: weight(weights, device, key, [size])?,
            size,
            eps,
        })
    }

    pub(super) fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let variance = (x.clone() * x.clone()).mean_dim(D - 1);
        let normed = x / variance.add_scalar(self.eps).sqrt();
        let mut shape = [1usize; D];
        shape[D - 1] = self.size;
        normed.cast(dtype) * self.weight.clone().reshape(shape)
    }
}

/// The cosine and sine tables of the rotary positions, grown as a run needs
/// them and sliced per step.
struct RopeCache<B: Backend> {
    cos: Option<Tensor<B, 2>>,
    sin: Option<Tensor<B, 2>>,
    len: usize,
    dim: usize,
    theta: f64,
}

impl<B: Backend> RopeCache<B> {
    fn new(dim: usize, theta: f64) -> Self {
        Self {
            cos: None,
            sin: None,
            len: 0,
            dim,
            theta,
        }
    }

    /// The tables covering `seq` positions from `start`, shaped `[1, 1, seq,
    /// dim]` with the half-table laid across both halves of the head, as the
    /// reference lays it.
    fn slice(
        &mut self,
        start: usize,
        seq: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        self.reserve(start + seq, device);
        let take = |table: &Option<Tensor<B, 2>>| {
            table
                .clone()
                .expect("reserved")
                .narrow(0, start, seq)
                .reshape([1, 1, seq, self.dim])
        };
        (take(&self.cos), take(&self.sin))
    }

    fn reserve(&mut self, needed: usize, device: &B::Device) {
        if needed <= self.len {
            return;
        }
        let len = needed.max(self.len * 2).max(CACHE_MIN_CAPACITY);
        let (cos, sin) = rope_values(self.dim, len, self.theta);
        self.cos = Some(Tensor::from_data(
            TensorData::new(cos, [len, self.dim]),
            device,
        ));
        self.sin = Some(Tensor::from_data(
            TensorData::new(sin, [len, self.dim]),
            device,
        ));
        self.len = len;
    }
}

/// The cosine and sine of every rotary position up to `len`, with the
/// half-table laid across both halves of the head, as the reference lays it.
///
/// Frequencies are computed in `f64` and rounded once, matching the reference.
pub(super) fn rope_values(
    dim: usize,
    len: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; len * dim];
    let mut sin = vec![0.0f32; len * dim];
    for position in 0..len {
        for index in 0..half {
            let freq =
                position as f64 / theta.powf(2.0 * index as f64 / dim as f64);
            let (sine, cosine) = (freq.sin() as f32, freq.cos() as f32);
            cos[position * dim + index] = cosine;
            cos[position * dim + half + index] = cosine;
            sin[position * dim + index] = sine;
            sin[position * dim + half + index] = sine;
        }
    }
    (cos, sin)
}

/// Rotates the halves of the last axis: `[a, b] → [-b, a]`.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let first = x.clone().narrow(3, 0, half);
    let second = x.narrow(3, half, half);
    Tensor::cat(vec![-second, first], 3)
}

/// Applies rotary positions to `x`, shaped `[batch, heads, time, head_dim]`.
pub(super) fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let rotated = rotate_half(x.clone());
    x * cos.clone() + rotated * sin.clone()
}

/// Repeats key/value heads so every query head has a partner.
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    if groups == 1 {
        return x;
    }
    let [batch, heads, seq, dim] = x.dims();
    x.reshape([batch, heads, 1, seq, dim])
        .repeat_dim(2, groups)
        .reshape([batch, heads * groups, seq, dim])
}

/// A preallocated key/value cache, written in place and grown by doubling.
pub(super) struct KvCache<B: Backend> {
    keys: Option<Tensor<B, 4>>,
    values: Option<Tensor<B, 4>>,
    heads: usize,
    dim: usize,
    len: usize,
    capacity: usize,
}

impl<B: Backend> KvCache<B> {
    pub(super) fn new(heads: usize, dim: usize) -> Self {
        Self {
            keys: None,
            values: None,
            heads,
            dim,
            len: 0,
            capacity: 0,
        }
    }

    /// Forgets the cached positions, keeping the buffer for the next run.
    ///
    /// Nothing past `len` is ever read, so the stale values need not be
    /// cleared — and the code predictor starts a fresh run on every frame, so
    /// releasing its buffers here would mean reallocating and zeroing them
    /// twelve times a second.
    pub(super) fn clear(&mut self) { self.len = 0; }

    /// Appends this step's keys and values and returns the whole cached prefix.
    pub(super) fn append(
        &mut self,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq = keys.dims()[2];
        let end = self.len + seq;
        self.reserve(end, &keys.device());

        let span = [0..1, 0..self.heads, self.len..end, 0..self.dim];
        let stored_keys = self
            .keys
            .take()
            .expect("reserved")
            .slice_assign(span.clone(), keys);
        let stored_values = self
            .values
            .take()
            .expect("reserved")
            .slice_assign(span, values);
        self.len = end;

        let prefix = (
            stored_keys.clone().narrow(2, 0, end),
            stored_values.clone().narrow(2, 0, end),
        );
        self.keys = Some(stored_keys);
        self.values = Some(stored_values);
        prefix
    }

    fn reserve(&mut self, needed: usize, device: &B::Device) {
        if needed <= self.capacity {
            return;
        }
        let capacity = needed.max(self.capacity * 2).max(CACHE_MIN_CAPACITY);
        let shape = [1, self.heads, capacity, self.dim];
        let grow = |old: Option<Tensor<B, 4>>, len: usize| -> Tensor<B, 4> {
            let fresh = Tensor::<B, 4>::zeros(shape, device);
            match old {
                Some(old) if len > 0 => fresh.slice_assign(
                    [0..1, 0..self.heads, 0..len, 0..self.dim],
                    old.narrow(2, 0, len),
                ),
                _ => fresh,
            }
        };
        self.keys = Some(grow(self.keys.take(), self.len));
        self.values = Some(grow(self.values.take(), self.len));
        self.capacity = capacity;
    }
}

/// Geometry an attention layer needs, shared by the talker and the predictor.
#[derive(Debug, Clone, Copy)]
struct AttentionShape {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

/// One decoder layer: normalized grouped-query attention, then a gated
/// feed-forward, each added back onto the residual stream.
struct DecoderLayer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
    /// `q`, `k` and `v` glued into one `[hidden, (heads + 2·kv_heads)·dim]`.
    qkv: Tensor<B, 2>,
    o_proj: Tensor<B, 2>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
    /// `gate` and `up` glued into one `[hidden, 2·intermediate]`.
    gate_up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    intermediate: usize,
    shape: AttentionShape,
}

impl<B: Backend> DecoderLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
        shape: AttentionShape,
    ) -> Result<Self, Qwen3TtsError> {
        let inner = shape.num_heads * shape.head_dim;
        let kv_inner = shape.num_kv_heads * shape.head_dim;
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");

        let qkv = glue_columns(
            &[
                (
                    rows_of(
                        weights,
                        &format!("{attention}.q_proj.weight"),
                        inner,
                        hidden,
                    )?,
                    inner,
                ),
                (
                    rows_of(
                        weights,
                        &format!("{attention}.k_proj.weight"),
                        kv_inner,
                        hidden,
                    )?,
                    kv_inner,
                ),
                (
                    rows_of(
                        weights,
                        &format!("{attention}.v_proj.weight"),
                        kv_inner,
                        hidden,
                    )?,
                    kv_inner,
                ),
            ],
            hidden,
        );
        let gate_up = glue_columns(
            &[
                (
                    rows_of(
                        weights,
                        &format!("{mlp}.gate_proj.weight"),
                        intermediate,
                        hidden,
                    )?,
                    intermediate,
                ),
                (
                    rows_of(
                        weights,
                        &format!("{mlp}.up_proj.weight"),
                        intermediate,
                        hidden,
                    )?,
                    intermediate,
                ),
            ],
            hidden,
        );

        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                shape.eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                shape.eps,
            )?,
            qkv: Tensor::from_data(
                TensorData::new(qkv, [hidden, inner + 2 * kv_inner]),
                device,
            ),
            o_proj: matrix(
                weights,
                device,
                &format!("{attention}.o_proj.weight"),
                hidden,
                inner,
            )?,
            // Normalization is per head, over the head dimension.
            q_norm: RmsNorm::load(
                weights,
                device,
                &format!("{attention}.q_norm.weight"),
                shape.head_dim,
                shape.eps,
            )?,
            k_norm: RmsNorm::load(
                weights,
                device,
                &format!("{attention}.k_norm.weight"),
                shape.head_dim,
                shape.eps,
            )?,
            gate_up: Tensor::from_data(
                TensorData::new(gate_up, [hidden, 2 * intermediate]),
                device,
            ),
            down: matrix(
                weights,
                device,
                &format!("{mlp}.down_proj.weight"),
                hidden,
                intermediate,
            )?,
            intermediate,
            shape,
        })
    }

    /// Runs the layer, appending this step's keys and values to `cache`.
    ///
    /// A block of more than one position only ever arrives on an empty cache —
    /// the prompt, and the predictor's two priming positions — so plain causal
    /// masking is all the kernel needs.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        cache: &mut KvCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let shape = self.shape;
        let inner = shape.num_heads * shape.head_dim;
        let kv_inner = shape.num_kv_heads * shape.head_dim;
        debug_assert!(seq == 1 || cache.len == 0);

        let normed = self.input_layernorm.forward(x.clone());
        let projected = linear(normed, self.qkv.clone(), None);
        let head = |slice: Tensor<B, 3>, heads: usize| -> Tensor<B, 4> {
            slice.reshape([batch, seq, heads, shape.head_dim])
        };
        // The per-head normalization sees the head dimension as the last axis,
        // so it runs before the heads are transposed out.
        let query = self
            .q_norm
            .forward(head(
                projected.clone().narrow(2, 0, inner),
                shape.num_heads,
            ))
            .swap_dims(1, 2);
        let key = self
            .k_norm
            .forward(head(
                projected.clone().narrow(2, inner, kv_inner),
                shape.num_kv_heads,
            ))
            .swap_dims(1, 2);
        let value = head(
            projected.narrow(2, inner + kv_inner, kv_inner),
            shape.num_kv_heads,
        )
        .swap_dims(1, 2);

        let query = apply_rope(query, cos, sin);
        let key = apply_rope(key, cos, sin);
        let (key, value) = cache.append(key, value);

        let groups = shape.num_heads / shape.num_kv_heads;
        let dtype = x.dtype();
        // Scores and softmax in full precision, as the reference computes them.
        let context = attention(
            query.cast(DType::F32),
            repeat_kv(key, groups).cast(DType::F32),
            repeat_kv(value, groups).cast(DType::F32),
            None,
            None,
            AttentionModuleOptions {
                is_causal: seq > 1,
                ..Default::default()
            },
        )
        .cast(dtype);
        let attended = context.swap_dims(1, 2).reshape([batch, seq, inner]);
        let x = x + linear(attended, self.o_proj.clone(), None);

        let normed = self.post_attention_layernorm.forward(x.clone());
        let projected = linear(normed, self.gate_up.clone(), None);
        let gated = activation::silu(projected.clone().narrow(
            2,
            0,
            self.intermediate,
        )) * projected.narrow(
            2,
            self.intermediate,
            self.intermediate,
        );
        x + linear(gated, self.down.clone(), None)
    }
}

/// A stack of decoder layers with its own cache and rotary tables.
struct Stack<B: Backend> {
    layers: Vec<DecoderLayer<B>>,
    norm: RmsNorm<B>,
    cache: Vec<KvCache<B>>,
    rope: RopeCache<B>,
    position: usize,
}

impl<B: Backend> Stack<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        count: usize,
        hidden: usize,
        intermediate: usize,
        shape: AttentionShape,
        theta: f64,
    ) -> Result<Self, Qwen3TtsError> {
        let layers = (0..count)
            .map(|index| {
                DecoderLayer::load(
                    weights,
                    device,
                    &format!("{prefix}.layers.{index}"),
                    hidden,
                    intermediate,
                    shape,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            norm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.norm.weight"),
                hidden,
                shape.eps,
            )?,
            cache: (0..count)
                .map(|_| KvCache::new(shape.num_kv_heads, shape.head_dim))
                .collect(),
            rope: RopeCache::new(shape.head_dim, theta),
            position: 0,
        })
    }

    fn reset(&mut self) {
        for cache in &mut self.cache {
            cache.clear();
        }
        self.position = 0;
    }

    /// Runs `embeds` (`[1, time, hidden]`) through the stack and returns the
    /// normalized state of its last position.
    fn forward(&mut self, embeds: Tensor<B, 3>) -> Tensor<B, 3> {
        let seq = embeds.dims()[1];
        let (cos, sin) = self.rope.slice(self.position, seq, &embeds.device());

        let mut hidden = embeds;
        for (layer, cache) in self.layers.iter().zip(self.cache.iter_mut()) {
            hidden = layer.forward(hidden, &cos, &sin, cache);
        }
        self.position += seq;
        self.norm.forward(hidden).narrow(1, seq - 1, 1)
    }
}

/// The code predictor: fills codebooks 1.. of a frame, one pass per codebook.
struct CodePredictor<B: Backend> {
    stack: Stack<B>,
    /// One output head per residual codebook, in burn's `[hidden, vocab]`
    /// layout.
    heads: Vec<Tensor<B, 2>>,
    /// Present only when the backbone is wider than the predictor.
    projection: Option<(Tensor<B, 2>, Tensor<B, 1>)>,
}

impl<B: Backend> CodePredictor<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &CodePredictorConfig,
        talker_hidden: usize,
        residuals: usize,
    ) -> Result<Self, Qwen3TtsError> {
        let shape = AttentionShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        };
        let heads = (0..residuals)
            .map(|index| {
                matrix(
                    weights,
                    device,
                    &format!("talker.code_predictor.lm_head.{index}.weight"),
                    cfg.vocab_size,
                    cfg.hidden_size,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The checkpoint carries a projection only when the widths differ;
        // otherwise the backbone state is handed over untouched.
        let projection = if cfg.hidden_size == talker_hidden {
            None
        } else {
            const PREFIX: &str =
                "talker.code_predictor.small_to_mtp_projection";
            Some((
                matrix(
                    weights,
                    device,
                    &format!("{PREFIX}.weight"),
                    cfg.hidden_size,
                    talker_hidden,
                )?,
                weight(
                    weights,
                    device,
                    &format!("{PREFIX}.bias"),
                    [cfg.hidden_size],
                )?,
            ))
        };

        Ok(Self {
            stack: Stack::load(
                weights,
                device,
                "talker.code_predictor.model",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.intermediate_size,
                shape,
                cfg.rope_theta,
            )?,
            heads,
            projection,
        })
    }

    /// Projects a backbone state into the predictor's width.
    fn project(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.projection {
            None => x,
            Some((weight, bias)) => {
                linear(x, weight.clone(), Some(bias.clone()))
            },
        }
    }
}

/// The two-layer projection that lifts text embeddings into the talker's width.
struct TextProjection<B: Backend> {
    fc1: (Tensor<B, 2>, Tensor<B, 1>),
    fc2: (Tensor<B, 2>, Tensor<B, 1>),
}

impl<B: Backend> TextProjection<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        text_hidden: usize,
        hidden: usize,
    ) -> Result<Self, Qwen3TtsError> {
        const PREFIX: &str = "talker.text_projection";
        Ok(Self {
            fc1: (
                matrix(
                    weights,
                    device,
                    &format!("{PREFIX}.linear_fc1.weight"),
                    text_hidden,
                    text_hidden,
                )?,
                weight(
                    weights,
                    device,
                    &format!("{PREFIX}.linear_fc1.bias"),
                    [text_hidden],
                )?,
            ),
            fc2: (
                matrix(
                    weights,
                    device,
                    &format!("{PREFIX}.linear_fc2.weight"),
                    hidden,
                    text_hidden,
                )?,
                weight(
                    weights,
                    device,
                    &format!("{PREFIX}.linear_fc2.bias"),
                    [hidden],
                )?,
            ),
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = activation::silu(linear(
            x,
            self.fc1.0.clone(),
            Some(self.fc1.1.clone()),
        ));
        linear(hidden, self.fc2.0.clone(), Some(self.fc2.1.clone()))
    }
}

/// The talker together with its code predictor.
pub struct Talker<B: Backend> {
    device: B::Device,
    cfg: TalkerConfig,
    text_embedding: Tensor<B, 2>,
    text_projection: TextProjection<B>,
    /// The talker's codec table followed by the predictor's fifteen, so a
    /// finished frame is one gather. Row `codec_offset(k) + code` is code
    /// `code` of codebook `k`.
    codec_embedding: Tensor<B, 2>,
    stack: Stack<B>,
    /// The codebook-0 head, in burn's `[hidden, vocab]` layout.
    codec_head: Tensor<B, 2>,
    predictor: CodePredictor<B>,
    /// The text track's padding, embedded once: every generated frame reads it
    /// and it never changes.
    pad_embed: Tensor<B, 3>,
    /// The talker's state for the frame the code predictor is filling.
    state: Option<Tensor<B, 3>>,
}

impl<B: Backend> Talker<B> {
    /// Loads the talker and its code predictor from a checkpoint's weights.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when a tensor is missing or does
    /// not match the declared geometry.
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &TalkerConfig,
        pad_token_id: u32,
    ) -> Result<Self, Qwen3TtsError> {
        let shape = AttentionShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        };
        let residuals = cfg.num_code_groups - 1;
        let predictor_vocab = cfg.code_predictor.vocab_size;

        // One table for every codebook: the talker's own, then the predictor's.
        let mut tables = rows_of(
            weights,
            "talker.model.codec_embedding.weight",
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        tables.reserve(residuals * predictor_vocab * cfg.hidden_size);
        for index in 0..residuals {
            tables.extend_from_slice(&rows_of(
                weights,
                &format!(
                    "talker.code_predictor.model.codec_embedding.{index}.\
                     weight"
                ),
                predictor_vocab,
                cfg.hidden_size,
            )?);
        }
        let codebook_rows = cfg.vocab_size + residuals * predictor_vocab;
        let codec_embedding = Tensor::from_data(
            TensorData::new(tables, [codebook_rows, cfg.hidden_size]),
            device,
        );

        let text_embedding = weight(
            weights,
            device,
            "talker.model.text_embedding.weight",
            [cfg.text_vocab_size, cfg.text_hidden_size],
        )?;
        let text_projection = TextProjection::load(
            weights,
            device,
            cfg.text_hidden_size,
            cfg.hidden_size,
        )?;
        let pad_embed = text_projection.forward(embedding(
            text_embedding.clone(),
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![i64::from(pad_token_id)], [1, 1]),
                device,
            ),
        ));

        Ok(Self {
            device: device.clone(),
            cfg: cfg.clone(),
            text_embedding,
            text_projection,
            codec_embedding,
            stack: Stack::load(
                weights,
                device,
                "talker.model",
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.intermediate_size,
                shape,
                cfg.rope_theta,
            )?,
            codec_head: matrix(
                weights,
                device,
                "talker.codec_head.weight",
                cfg.vocab_size,
                cfg.hidden_size,
            )?,
            predictor: CodePredictor::load(
                weights,
                device,
                &cfg.code_predictor,
                cfg.hidden_size,
                residuals,
            )?,
            pad_embed,
            state: None,
        })
    }

    /// The first row of codebook `index`'s table inside the shared one.
    fn codec_offset(&self, index: usize) -> usize {
        match index {
            0 => 0,
            residual => {
                self.cfg.vocab_size +
                    (residual - 1) * self.cfg.code_predictor.vocab_size
            },
        }
    }

    /// Starts a run and returns the codebook-0 logits of the first frame.
    pub(super) fn prime(
        &mut self,
        positions: &[Position],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        self.stack.reset();

        // Every position drives the text track; only the opening role, a
        // prefix, leaves the codec track silent.
        let text_ids: Vec<i64> = positions
            .iter()
            .filter_map(|position| position.text.map(i64::from))
            .collect();
        let codec_ids: Vec<i64> = positions
            .iter()
            .filter_map(|position| position.codec.map(i64::from))
            .collect();
        let lead = positions.len() - codec_ids.len();

        let text = self.text_projection.forward(embedding(
            self.text_embedding.clone(),
            indices(text_ids, &self.device),
        ));
        let voiced = embedding(
            self.codec_embedding.clone(),
            indices(codec_ids, &self.device),
        );
        let hidden = self.cfg.hidden_size;
        let codec = if lead == 0 {
            voiced
        } else {
            Tensor::cat(
                vec![Tensor::zeros([1, lead, hidden], &self.device), voiced],
                1,
            )
        };

        self.step(text + codec)
    }

    /// Folds a finished frame back into the talker and takes one step.
    pub(super) fn advance(
        &mut self,
        frame: &[u32],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        // The finished frame folds into one embedding — the sum of every
        // codebook's — and, with the text track's padding, becomes the next
        // input.
        let ids: Vec<i64> = frame
            .iter()
            .enumerate()
            .map(|(index, &code)| {
                (self.codec_offset(index) + code as usize) as i64
            })
            .collect();
        let folded =
            embedding(self.codec_embedding.clone(), indices(ids, &self.device))
                .sum_dim(1);
        self.step(folded + self.pad_embed.clone())
    }

    /// Runs one talker pass over `embeds` and scores codebook 0.
    fn step(
        &mut self,
        embeds: Tensor<B, 3>,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        let state = self.stack.forward(embeds);
        let logits = linear(state.clone(), self.codec_head.clone(), None);
        self.state = Some(state);
        host_logits(logits)
    }

    /// Generates the residual codebooks of one frame.
    pub(super) fn predict_residuals(
        &mut self,
        first_code: u32,
        pick: &mut dyn FnMut(&[f32], usize) -> u32,
    ) -> Result<Vec<u32>, Qwen3TtsError> {
        let state = self.state.clone().ok_or_else(|| {
            model_err("predicting the residual codebooks", "no primed state")
        })?;
        let table = self.codec_embedding.clone();
        let device = self.device.clone();
        let offsets: Vec<usize> = (0..self.cfg.num_code_groups)
            .map(|k| self.codec_offset(k))
            .collect();
        let first_embed = embedding(
            table.clone(),
            indices(vec![i64::from(first_code)], &device),
        );

        self.predictor.stack.reset();
        // The predictor is primed with the backbone state and the frame's
        // first code, so its first output scores codebook 1.
        let primed = self
            .predictor
            .project(Tensor::cat(vec![state, first_embed], 1));
        let mut hidden = self.predictor.stack.forward(primed);

        let residuals = self.predictor.heads.len();
        let mut codes = Vec::with_capacity(residuals);
        for step in 0..residuals {
            let logits = linear(
                hidden.clone(),
                self.predictor.heads[step].clone(),
                None,
            );
            let code = pick(&host_logits(logits)?, step);
            codes.push(code);

            // The last code needs no follow-up pass.
            if step + 1 == residuals {
                break;
            }
            let row = (offsets[step + 1] + code as usize) as i64;
            let embed = embedding(table.clone(), indices(vec![row], &device));
            let projected = self.predictor.project(embed);
            hidden = self.predictor.stack.forward(projected);
        }
        Ok(codes)
    }
}

/// Turns a list of table rows into `[1, ids]` indices on the device.
fn indices<B: Backend>(ids: Vec<i64>, device: &B::Device) -> Tensor<B, 2, Int> {
    let len = ids.len();
    Tensor::from_data(TensorData::new(ids, [1, len]), device)
}

/// Pulls a `[1, 1, vocab]` logits tensor to the host in full precision.
fn host_logits<B: Backend>(
    logits: Tensor<B, 3>,
) -> Result<Vec<f32>, Qwen3TtsError> {
    logits
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| model_err("reading the logits", format!("{e:?}")))
}
