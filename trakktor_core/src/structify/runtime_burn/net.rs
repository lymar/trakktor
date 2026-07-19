//! The SaT network on burn.
//!
//! A port of the candle network in `runtime::net` (itself adapted from the
//! candle project's XLM-RoBERTa model, Apache-2.0 OR MIT) with the same
//! numerical semantics, written idiomatically for burn rather than as a
//! mirror, and
//! specialized to the one invariant the window pipeline guarantees: **every
//! window is full** (`block = min(n_tokens, MAX_BLOCK)`, the last window is
//! pinned to the end), so sequence padding never exists. That invariant
//! removes whole stages of the candle path:
//!
//! - attention runs through burn's fused scaled-dot-product `attention` (its
//!   default scale is the reference's `1/sqrt(d_head)`) with **no mask at all**
//!   — a full window has nothing to mask. The fused call always computes in
//!   f32, also on the f16 backend: the flash kernel's f16 path loses the
//!   boundary signal on XLM-R's outlier-heavy activations, its f32 path is
//!   exact, and the casts around it are cheap;
//! - the position ids of a full window are the constant range `[pad + 1, pad +
//!   1 + seq)` (XLM-R counts non-padding tokens from `padding_idx + 1`), so the
//!   position and token-type embeddings collapse into one table precomputed at
//!   load and broadcast-added to the word embeddings — no per-batch
//!   `ne`/`cumsum`/gather chain;
//! - the per-layer q/k/v projections are one fused linear (the checkpoint's
//!   three matrices are concatenated at load; in XLM-R all three carry biases);
//! - only the boundary row of the classifier is loaded ([`NEWLINE_INDEX`]): the
//!   base models' `Linear(768 → 111)` head becomes `Linear(768 → 1)`. The rows
//!   of a linear map are independent, so the boundary logit is exactly the same
//!   value.
//!
//! Layer normalization keeps its statistics in `f32` (as candle does), and
//! GELU is burn's exact (erf) form, matching the reference
//! `hidden_act = "gelu"`.
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

use super::Weights;
use crate::structify::{
    error::StructifyError,
    runtime::{NEWLINE_INDEX, model_err, net::Config},
};

/// Reads a checkpoint tensor of the given shape as a burn tensor (in the
/// backend's compute dtype).
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, StructifyError> {
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
) -> Result<Vec<f32>, StructifyError> {
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
/// `[d, 3·d]` weight whose column blocks are `[q | k | v]` — the fused
/// projection computes all three in a single matmul.
pub(super) fn glue_qkv(q: &[f32], k: &[f32], v: &[f32], d: usize) -> Vec<f32> {
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

/// The precomputed per-position embedding rows: position embedding (starting
/// at `offset`, the first non-padding position id) plus the single token-type
/// embedding row, summed on the host. `pos` is the `[max_positions, d]`
/// position table, `type_row` the `[d]` token-type row; returns
/// `max_positions − offset` rows.
pub(super) fn position_type_table(
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

/// A linear layer with candle/PyTorch semantics: `y = x·Wᵀ + b`. The
/// checkpoint's `[out, in]` weight is transposed once at load into burn's
/// `[in, out]` layout, and the forward pass is burn's `linear` primitive.
pub(super) struct Linear<B: Backend> {
    pub(super) weight: Tensor<B, 2>, // [in, out]
    pub(super) bias: Tensor<B, 1>,   // [out]
}

impl<B: Backend> Linear<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, StructifyError> {
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

    /// `[B, T, in]` -> `[B, T, out]`.
    pub(super) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// Layer normalization over the last dim, mirroring candle's numerics: the
/// statistics are computed in `f32` (also for `f16` models), the result is
/// cast back to the compute dtype, then scaled and shifted.
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
    ) -> Result<Self, StructifyError> {
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

    /// `[B, T, d]` -> `[B, T, d]`, normalized over `d`.
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

/// Bidirectional self-attention over full windows: one fused q/k/v
/// projection, then burn's fused scaled-dot-product `attention` without any
/// mask.
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
    ) -> Result<Self, StructifyError> {
        let d = cfg.hidden_size;
        let part = |name: &str| -> Result<Vec<f32>, StructifyError> {
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
            d_head: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    /// `[B, T, d]` -> `[B, T, d]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let qkv = self.qkv.forward(x); // [B, T, 3d]
        let q =
            split_heads(qkv.clone().narrow(2, 0, d), self.n_heads, self.d_head);
        let k =
            split_heads(qkv.clone().narrow(2, d, d), self.n_heads, self.d_head);
        let v = split_heads(qkv.narrow(2, 2 * d, d), self.n_heads, self.d_head);
        // The fused call always computes in f32: the flash kernel's f16 path
        // loses the boundary signal on XLM-R's outlier-heavy activations
        // (measured: boundary probabilities collapse and paragraphs vanish),
        // while its f32 path is exact. On the f16 backend the casts are cheap
        // next to the matmuls; on the f32 backends they are no-ops.
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

/// A dense projection with a residual add and layer norm — the shape shared by
/// the attention output and the FFN output.
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
    ) -> Result<Self, StructifyError> {
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

/// One transformer layer: self-attention and the GELU FFN, each with its
/// residual + layer norm.
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
    ) -> Result<Self, StructifyError> {
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

/// The loaded SaT network on burn.
pub struct SatModel<B: Backend> {
    word_embeddings: Tensor<B, 2>, // [vocab, d]
    /// Position + token-type embedding rows for every usable position of a
    /// full window (`[max_positions − offset, d]`).
    pos_type: Tensor<B, 2>,
    embeddings_norm: LayerNorm<B>,
    layers: Vec<Layer<B>>,
    classifier: Linear<B>, // [d, 1]: the boundary row only
}

impl<B: Backend> SatModel<B> {
    /// Loads the encoder (`roberta.*`) and the boundary row of the
    /// token-classification head (`classifier.*`).
    pub(super) fn load(
        weights: &Weights,
        cfg: &Config,
        device: &B::Device,
    ) -> Result<Self, StructifyError> {
        let d = cfg.hidden_size;
        let emb = "roberta.embeddings";

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
        // Position ids of a full window count from `padding_idx + 1`; the
        // single token-type row (id 0) folds into the same table.
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
                    &format!("roberta.encoder.layer.{i}"),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Only the boundary label's row of the classifier is ever read;
        // slicing it out at load shrinks the head to `Linear(d → 1)`.
        let head =
            linear_values(weights, "classifier.weight", cfg.num_labels, d)?;
        let row = &head[NEWLINE_INDEX * d..(NEWLINE_INDEX + 1) * d];
        let (head_bias, bias_dims) = weights.parts("classifier.bias")?;
        if bias_dims != [cfg.num_labels] {
            return Err(model_err(
                "classifier.bias",
                format!("shape {bias_dims:?}, expected {:?}", [cfg.num_labels]),
            ));
        }
        let classifier = Linear {
            weight: Tensor::from_data(
                TensorData::new(row.to_vec(), [d, 1]),
                device,
            ),
            bias: Tensor::from_data(
                TensorData::new(vec![head_bias[NEWLINE_INDEX]], [1]),
                device,
            ),
        };

        Ok(Self {
            word_embeddings,
            pos_type,
            embeddings_norm,
            layers,
            classifier,
        })
    }

    /// Runs a batch of full windows and returns the per-position **boundary
    /// logit** as an `(batch, seq)` f32 tensor (`CLS`/`SEP` positions still
    /// included).
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [b, s] = input_ids.dims();
        let [_, d] = self.word_embeddings.dims();
        let x = embedding(self.word_embeddings.clone(), input_ids);
        let pos = self.pos_type.clone().narrow(0, 0, s).reshape([1, s, d]);
        let mut hidden = self.embeddings_norm.forward(x + pos);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        let logits = self.classifier.forward(hidden); // [B, T, 1]
        logits.reshape([b, s]).cast(DType::F32)
    }
}
