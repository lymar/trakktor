//! The punctuation network on burn.
//!
//! A port of the candle network in `runtime::net` (itself adapted from
//! `candle-transformers`, Apache-2.0 OR MIT), written idiomatically for burn
//! and specialized to the invariant the window driver guarantees: **every
//! window in a batch has the same length**, so sequence padding never exists.
//! That removes whole stages of the candle path (as in the structify burn
//! port):
//!
//! - attention runs through burn's fused scaled-dot-product `attention` with
//!   **no mask** — a full window has nothing to mask — and **always in f32**
//!   (the flash kernel's f16 path loses accuracy on XLM-R's outlier-heavy
//!   activations; the casts around it are cheap);
//! - the position ids of a full window are the constant range `[pad + 1, …)`,
//!   so the position and token-type embeddings collapse into one table
//!   precomputed at load and broadcast-added to the word embeddings;
//! - the per-layer q/k/v projections are one fused linear.
//!
//! On top of the encoder sit the four `ConditionedPCSDecoder` heads, wired in
//! the same cascade as the candle path: `post`/`pre` punctuation from the
//! encoder output, `seg` (sentence boundary) conditioned on the argmax post
//! punctuation embedding, and `cap` (per-character casing) conditioned on the
//! right-shifted boundary prediction. LayerNorm statistics are kept in f32 and
//! GELU is the exact (erf) form, matching the candle path.

use burn::tensor::{
    DType, Int, Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, embedding, linear},
    ops::AttentionModuleOptions,
};

use super::Weights;
use crate::punctuate::{
    error::PunctuateError,
    model::Config,
    runtime::{RawOutputs, model_err},
};

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, PunctuateError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Reads a `[out, in]`-shaped checkpoint weight as row-major f32 values,
/// shape-checked.
fn linear_values(
    weights: &Weights,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, PunctuateError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
        ));
    }
    Ok(values)
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

/// Concatenates the q/k/v `[d, d]` checkpoint matrices into one transposed
/// `[d, 3·d]` weight whose column blocks are `[q | k | v]`.
fn glue_qkv(q: &[f32], k: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut glued = vec![0.0f32; d * 3 * d];
    for (block, values) in [q, k, v].into_iter().enumerate() {
        let transposed = transpose_2d(values, d, d);
        for row in 0..d {
            let src = &transposed[row * d..(row + 1) * d];
            let at = row * 3 * d + block * d;
            glued[at..at + d].copy_from_slice(src);
        }
    }
    glued
}

/// The precomputed per-position embedding rows: position embedding (from
/// `offset`, the first non-padding position id) plus the single token-type row.
fn position_type_table(
    pos: &[f32],
    type_row: &[f32],
    offset: usize,
    d: usize,
) -> Vec<f32> {
    pos[offset * d..]
        .iter()
        .zip(type_row.iter().cycle())
        .map(|(&p, &t)| p + t)
        .collect()
}

/// A linear layer with candle/PyTorch semantics (`y = x·Wᵀ + b`); the
/// checkpoint's `[out, in]` weight is transposed once at load into burn's
/// `[in, out]` layout.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>, // [in, out]
    bias: Tensor<B, 1>,   // [out]
}

impl<B: Backend> Linear<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, PunctuateError> {
        let values = linear_values(
            weights,
            &format!("{prefix}.weight"),
            out_dim,
            in_dim,
        )?;
        let transposed = transpose_2d(&values, out_dim, in_dim);
        Ok(Self {
            weight: Tensor::from_data(
                TensorData::new(transposed, [in_dim, out_dim]),
                device,
            ),
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_dim],
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// Layer normalization over the last dim; statistics in f32, matching candle.
struct LayerNorm<B: Backend> {
    gamma: Tensor<B, 3>, // [1, 1, d]
    beta: Tensor<B, 3>,  // [1, 1, d]
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        d: usize,
        eps: f64,
    ) -> Result<Self, PunctuateError> {
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
            eps,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = (centered.clone() * centered.clone()).mean_dim(2);
        let normed = centered / (var.add_scalar(self.eps)).sqrt();
        normed.cast(dtype) * self.gamma.clone() + self.beta.clone()
    }
}

/// `[B, T, d]` -> `[B, heads, T, d_head]` for the fused attention call.
fn split_heads<B: Backend>(
    x: Tensor<B, 3>,
    n_heads: usize,
    d_head: usize,
) -> Tensor<B, 4> {
    let [b, s, _] = x.dims();
    x.reshape([b, s, n_heads, d_head]).swap_dims(1, 2)
}

/// Bidirectional self-attention over full windows: one fused q/k/v projection,
/// then burn's fused scaled-dot-product `attention` without any mask, in f32.
struct SelfAttention<B: Backend> {
    qkv: Linear<B>, // [d, 3d]: columns are [q | k | v]
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &Config,
    ) -> Result<Self, PunctuateError> {
        let d = cfg.hidden_size;
        let part = |name: &str| -> Result<Vec<f32>, PunctuateError> {
            linear_values(weights, &format!("{prefix}.{name}.weight"), d, d)
        };
        let glued =
            glue_qkv(&part("query")?, &part("key")?, &part("value")?, d);
        let mut bias = Vec::with_capacity(3 * d);
        for name in ["query", "key", "value"] {
            let (values, dims) =
                weights.parts(&format!("{prefix}.{name}.bias"))?;
            if dims != [d] {
                return Err(model_err(
                    &format!("{prefix}.{name}.bias"),
                    format!("shape {dims:?}, expected {:?}", [d]),
                ));
            }
            bias.extend_from_slice(&values);
        }
        Ok(Self {
            qkv: Linear {
                weight: Tensor::from_data(
                    TensorData::new(glued, [d, 3 * d]),
                    device,
                ),
                bias: Tensor::from_data(TensorData::new(bias, [3 * d]), device),
            },
            n_heads: cfg.num_attention_heads,
            d_head: cfg.head_dim(),
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let qkv = self.qkv.forward(x); // [B, T, 3d]
        let q =
            split_heads(qkv.clone().narrow(2, 0, d), self.n_heads, self.d_head);
        let k =
            split_heads(qkv.clone().narrow(2, d, d), self.n_heads, self.d_head);
        let v = split_heads(qkv.narrow(2, 2 * d, d), self.n_heads, self.d_head);
        // The fused call always computes in f32 (see module docs).
        let dtype = q.dtype();
        let ctx = attention(
            q.cast(DType::F32),
            k.cast(DType::F32),
            v.cast(DType::F32),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .cast(dtype);
        ctx.swap_dims(1, 2).reshape([b, s, d])
    }
}

/// A dense projection with a residual add and layer norm.
struct DenseNorm<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> DenseNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        eps: f64,
    ) -> Result<Self, PunctuateError> {
        Ok(Self {
            dense: Linear::load(
                weights,
                device,
                &format!("{prefix}.dense"),
                in_dim,
                out_dim,
            )?,
            layer_norm: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.LayerNorm"),
                out_dim,
                eps,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        residual: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.layer_norm
            .forward(self.dense.forward(hidden) + residual)
    }
}

/// One transformer layer.
struct Layer<B: Backend> {
    attention: SelfAttention<B>,
    attention_output: DenseNorm<B>,
    intermediate: Linear<B>,
    output: DenseNorm<B>,
}

impl<B: Backend> Layer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &Config,
    ) -> Result<Self, PunctuateError> {
        Ok(Self {
            attention: SelfAttention::load(
                weights,
                device,
                &format!("{prefix}.attention.self"),
                cfg,
            )?,
            attention_output: DenseNorm::load(
                weights,
                device,
                &format!("{prefix}.attention.output"),
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
            intermediate: Linear::load(
                weights,
                device,
                &format!("{prefix}.intermediate.dense"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?,
            output: DenseNorm::load(
                weights,
                device,
                &format!("{prefix}.output"),
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self.attention.forward(x.clone());
        let x = self.attention_output.forward(attn, x);
        // Exact GELU (erf), matching the reference `hidden_act = "gelu"`.
        let inter = activation::gelu(self.intermediate.forward(x.clone()));
        self.output.forward(inter, x)
    }
}

/// A two-layer classification head (`Linear → ReLU → Linear`).
struct Head<B: Backend> {
    first: Linear<B>,
    last: Linear<B>,
}

impl<B: Backend> Head<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        mid_dim: usize,
        out_dim: usize,
    ) -> Result<Self, PunctuateError> {
        Ok(Self {
            first: Linear::load(
                weights,
                device,
                &format!("{prefix}._linears.0"),
                in_dim,
                mid_dim,
            )?,
            last: Linear::load(
                weights,
                device,
                &format!("{prefix}._linears.1"),
                mid_dim,
                out_dim,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.last.forward(activation::relu(self.first.forward(x)))
    }
}

/// The loaded punctuation network on burn.
pub struct PunctModel<B: Backend> {
    word_embeddings: Tensor<B, 2>, // [vocab, d]
    pos_type: Tensor<B, 2>,        // [max_positions − offset, d]
    embeddings_norm: LayerNorm<B>,
    layers: Vec<Layer<B>>,
    punct_emb: Tensor<B, 2>, // [post_classes, emb]
    punct_head_post: Head<B>,
    punct_head_pre: Head<B>,
    seg_head: Head<B>,
    cap_head: Head<B>,
    device: B::Device,
}

impl<B: Backend> PunctModel<B> {
    /// Loads the encoder (`bert_model.*`) and the four heads (`_decoder.*`).
    pub(super) fn load(
        weights: &Weights,
        cfg: &Config,
        device: &B::Device,
    ) -> Result<Self, PunctuateError> {
        let d = cfg.hidden_size;
        let emb = "bert_model.embeddings";

        let word_embeddings = weight(
            weights,
            device,
            &format!("{emb}.word_embeddings.weight"),
            [cfg.vocab_size, d],
        )?;

        let pos_key = format!("{emb}.position_embeddings.weight");
        let (pos, pos_dims) = weights.parts(&pos_key)?;
        if pos_dims != [cfg.max_position_embeddings, d] {
            return Err(model_err(
                &pos_key,
                format!(
                    "shape {pos_dims:?}, expected {:?}",
                    [cfg.max_position_embeddings, d]
                ),
            ));
        }
        let type_key = format!("{emb}.token_type_embeddings.weight");
        let (token_type, type_dims) = weights.parts(&type_key)?;
        if type_dims != [cfg.type_vocab_size, d] {
            return Err(model_err(
                &type_key,
                format!(
                    "shape {type_dims:?}, expected {:?}",
                    [cfg.type_vocab_size, d]
                ),
            ));
        }
        let offset = cfg.pad_token_id as usize + 1;
        if cfg.max_position_embeddings <= offset {
            return Err(model_err(
                &pos_key,
                format!("no positions past the padding offset {offset}"),
            ));
        }
        let table = position_type_table(&pos, &token_type[..d], offset, d);
        let pos_type = Tensor::from_data(
            TensorData::new(table, [cfg.max_position_embeddings - offset, d]),
            device,
        );

        let embeddings_norm = LayerNorm::load(
            weights,
            device,
            &format!("{emb}.LayerNorm"),
            d,
            cfg.layer_norm_eps,
        )?;

        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                Layer::load(
                    weights,
                    device,
                    &format!("bert_model.encoder.layer.{i}"),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let decoder = "_decoder";
        let punct_emb = weight(
            weights,
            device,
            &format!("{decoder}._punct_emb.weight"),
            [cfg.punct_post_classes, cfg.emb_dim],
        )?;
        let punct_head_post = Head::load(
            weights,
            device,
            &format!("{decoder}._punct_head_post"),
            d,
            cfg.punct_head_intermediate,
            cfg.punct_post_classes,
        )?;
        let punct_head_pre = Head::load(
            weights,
            device,
            &format!("{decoder}._punct_head_pre"),
            d,
            cfg.punct_head_intermediate,
            cfg.punct_pre_classes,
        )?;
        let seg_head = Head::load(
            weights,
            device,
            &format!("{decoder}._seg_head"),
            d + cfg.emb_dim,
            cfg.seg_head_intermediate,
            2,
        )?;
        let cap_head = Head::load(
            weights,
            device,
            &format!("{decoder}._cap_head"),
            d + 1,
            cfg.cap_head_intermediate,
            cfg.cap_classes,
        )?;

        Ok(Self {
            word_embeddings,
            pos_type,
            embeddings_norm,
            layers,
            punct_emb,
            punct_head_post,
            punct_head_pre,
            seg_head,
            cap_head,
            device: device.clone(),
        })
    }

    fn encode(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, s] = input_ids.dims();
        let [_, d] = self.word_embeddings.dims();
        let x = embedding(self.word_embeddings.clone(), input_ids);
        let pos = self.pos_type.clone().narrow(0, 0, s).reshape([1, s, d]);
        let mut hidden = self.embeddings_norm.forward(x + pos);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        hidden
    }

    /// Forwards a batch of equal-length windows and returns the raw head
    /// outputs pulled to the host.
    pub(crate) fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RawOutputs, PunctuateError> {
        let [b, s] = input_ids.dims();
        let hidden = self.encode(input_ids); // (B, T, D)

        let post_logits = self.punct_head_post.forward(hidden.clone()); // (B,T,17)
        let pre_logits = self.punct_head_pre.forward(hidden.clone()); // (B,T,2)
        let post_ids = post_logits.clone().argmax(2).reshape([b, s]); // (B,T) Int

        let embs = embedding(self.punct_emb.clone(), post_ids); // (B,T,emb)
        let seg_input = Tensor::cat(vec![hidden.clone(), embs], 2); // (B,T,D+emb)
        let seg_logits = self.seg_head.forward(seg_input); // (B,T,2)
        let seg_ids = seg_logits.clone().argmax(2).reshape([b, s]); // (B,T) Int

        let seg_shift = self.shift_boundaries(seg_ids, b, s); // (B,T,1) f
        let cap_input = Tensor::cat(vec![hidden, seg_shift], 2); // (B,T,D+1)
        let cap_logits = self.cap_head.forward(cap_input); // (B,T,cap)

        // The threshold decisions are made on f32 values on both runtimes, so
        // the softmax/sigmoid are computed in f32 also on the f16 backend
        // (matching the candle path, which upcasts the logits first).
        let seg_prob1 = activation::softmax(seg_logits.cast(DType::F32), 2)
            .narrow(2, 1, 1)
            .reshape([b, s]); // (B,T) f32
        let cap_prob = activation::sigmoid(cap_logits.cast(DType::F32)); // (B,T,cap)

        Ok(RawOutputs {
            pre: host_argmax(pre_logits, b, s)?,
            post: host_argmax(post_logits, b, s)?,
            seg_prob1: host_2d(seg_prob1, b, s)?,
            cap_prob: host_3d(cap_prob, b, s)?,
        })
    }

    /// `seg_ids` `(B, T)` Int → `(B, T, 1)` compute-dtype boundary feature:
    /// force column 0 to 1, then shift right by one (pad a 0 on the left, drop
    /// the last).
    fn shift_boundaries(
        &self,
        seg_ids: Tensor<B, 2, Int>,
        b: usize,
        s: usize,
    ) -> Tensor<B, 3> {
        let ones = Tensor::<B, 2, Int>::ones([b, 1], &self.device);
        let forced = if s > 1 {
            Tensor::cat(vec![ones, seg_ids.narrow(1, 1, s - 1)], 1)
        } else {
            ones
        };
        let zeros = Tensor::<B, 2, Int>::zeros([b, 1], &self.device);
        let shifted = if s > 1 {
            Tensor::cat(vec![zeros, forced.narrow(1, 0, s - 1)], 1)
        } else {
            zeros
        };
        shifted.float().reshape([b, s, 1])
    }
}

/// Argmax over the last dim of a `(B, T, C)` logits tensor, pulled to the host
/// as `[B][T]` label ids.
fn host_argmax<B: Backend>(
    logits: Tensor<B, 3>,
    b: usize,
    s: usize,
) -> Result<Vec<Vec<u32>>, PunctuateError> {
    let [_, _, c] = logits.dims();
    // Upcast before pulling to the host: on the f16 backend the tensor is f16,
    // and the exact upcast leaves the argmax unchanged.
    let flat = logits
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| model_err("reading logits", format!("{e:?}")))?;
    Ok((0..b)
        .map(|bi| {
            (0..s)
                .map(|si| {
                    let row = &flat[(bi * s + si) * c..(bi * s + si + 1) * c];
                    let mut best = 0usize;
                    for (i, &v) in row.iter().enumerate() {
                        if v > row[best] {
                            best = i;
                        }
                    }
                    best as u32
                })
                .collect()
        })
        .collect())
}

/// Pulls a `(B, T)` f32 tensor to `[B][T]`.
fn host_2d<B: Backend>(
    tensor: Tensor<B, 2>,
    b: usize,
    s: usize,
) -> Result<Vec<Vec<f32>>, PunctuateError> {
    let flat = tensor
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| model_err("reading outputs", format!("{e:?}")))?;
    Ok((0..b)
        .map(|bi| flat[bi * s..(bi + 1) * s].to_vec())
        .collect())
}

/// Pulls a `(B, T, C)` f32 tensor to `[B][T][C]`.
fn host_3d<B: Backend>(
    tensor: Tensor<B, 3>,
    b: usize,
    s: usize,
) -> Result<Vec<Vec<Vec<f32>>>, PunctuateError> {
    let [_, _, c] = tensor.dims();
    let flat = tensor
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| model_err("reading outputs", format!("{e:?}")))?;
    Ok((0..b)
        .map(|bi| {
            (0..s)
                .map(|si| {
                    flat[(bi * s + si) * c..(bi * s + si + 1) * c].to_vec()
                })
                .collect()
        })
        .collect())
}
