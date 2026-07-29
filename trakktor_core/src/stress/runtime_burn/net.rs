//! The two networks on burn.
//!
//! A port of the candle networks in `runtime::net`, written idiomatically for
//! burn and specialized to what this feature guarantees:
//!
//! - the accentor's bags arrive as a padded rectangle with a mask, so the whole
//!   batch is one gather, one masked sum and one divide;
//! - the homograph encoder runs through burn's fused scaled-dot-product
//!   `attention` with **no mask** — which is not an optimization here but the
//!   reference's own behaviour: it pads a batch of sentences and lets the
//!   encoder attend to the padding — and **always in f32** (the flash kernel's
//!   f16 path loses accuracy on outlier-heavy activations; the casts around it
//!   are cheap);
//! - the position ids are always `0..seq` and every token is of type 0, so the
//!   two embedding tables collapse into one, precomputed at load and
//!   broadcast-added to the word embeddings;
//! - the per-layer q/k/v projections are one fused linear.
//!
//! LayerNorm statistics are kept in f32 and GELU is the exact (erf) form,
//! matching the candle path; so are the thresholds, which are decided on f32
//! values on both runtimes.

use burn::tensor::{
    DType, Int, Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, embedding, linear},
    ops::AttentionModuleOptions,
};

use super::Weights;
use crate::stress::{
    error::StressError,
    model::{
        AccentorConfig, Config, HomographConfig, HomographContext, WordScores,
        pick, scores, softmax,
    },
    runtime::model_err,
};

/// Reads a checkpoint tensor of the given shape as a burn tensor.
fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, StressError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Reads an `[out, in]`-shaped weight as row-major f32 values, shape-checked.
fn linear_values(
    weights: &Weights,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, StressError> {
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

/// Concatenates the q/k/v `[d, d]` matrices into one transposed `[d, 3·d]`
/// weight whose column blocks are `[q | k | v]`.
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

/// A linear layer with candle/PyTorch semantics (`y = x·Wᵀ + b`); the stored
/// `[out, in]` weight is transposed once at load into burn's `[in, out]`.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> Linear<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, StressError> {
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

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// Layer normalization over the last dim; statistics in f32.
struct LayerNorm<B: Backend> {
    gamma: Tensor<B, 3>,
    beta: Tensor<B, 3>,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        d: usize,
        eps: f64,
    ) -> Result<Self, StressError> {
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

/// One of the accentor's heads: four dense layers with ReLU between them.
struct Head<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> Head<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &AccentorConfig,
        classes: usize,
    ) -> Result<Self, StressError> {
        let widths = [
            cfg.dim,
            cfg.hidden[0],
            cfg.hidden[1],
            cfg.hidden[2],
            classes,
        ];
        // The reference's `Sequential` interleaves activations, so the dense
        // layers sit at the even indices.
        let layers = [0usize, 2, 4, 6]
            .into_iter()
            .enumerate()
            .map(|(step, index)| {
                Linear::load(
                    weights,
                    device,
                    &format!("{prefix}.{index}"),
                    widths[step],
                    widths[step + 1],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index + 1 < self.layers.len() {
                x = activation::relu(x);
            }
        }
        x
    }
}

/// Bidirectional self-attention over the batch, without a mask.
struct SelfAttention<B: Backend> {
    qkv: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> SelfAttention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &HomographConfig,
    ) -> Result<Self, StressError> {
        let d = cfg.hidden_size;
        let part = |name: &str| -> Result<Vec<f32>, StressError> {
            linear_values(weights, &format!("{prefix}.{name}.weight"), d, d)
        };
        let glued =
            glue_qkv(&part("query")?, &part("key")?, &part("value")?, d);
        let mut bias = Vec::with_capacity(3 * d);
        for name in ["query", "key", "value"] {
            let key = format!("{prefix}.{name}.bias");
            let (values, dims) = weights.parts(&key)?;
            if dims != [d] {
                return Err(model_err(
                    &key,
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
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn split(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, s, _] = x.dims();
        x.reshape([b, s, self.heads, self.head_dim]).swap_dims(1, 2)
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let qkv = self.qkv.forward(x);
        let q = self.split(qkv.clone().narrow(2, 0, d));
        let k = self.split(qkv.clone().narrow(2, d, d));
        let v = self.split(qkv.narrow(2, 2 * d, d));
        let dtype = q.dtype();
        // The fused call always computes in f32 (see module docs).
        let context = attention(
            q.cast(DType::F32),
            k.cast(DType::F32),
            v.cast(DType::F32),
            None,
            None,
            AttentionModuleOptions::default(),
        )
        .cast(dtype);
        context.swap_dims(1, 2).reshape([b, s, d])
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
    ) -> Result<Self, StressError> {
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
        cfg: &HomographConfig,
    ) -> Result<Self, StressError> {
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
        let attention = self.attention.forward(x.clone());
        let x = self.attention_output.forward(attention, x);
        // Exact GELU (erf), matching the reference's `gelu`.
        let intermediate =
            activation::gelu(self.intermediate.forward(x.clone()));
        self.output.forward(intermediate, x)
    }
}

/// Both loaded networks.
pub struct StressNet<B: Backend> {
    ngrams: Tensor<B, 2>,
    stress: Head<B>,
    yo: Head<B>,
    word_embeddings: Tensor<B, 2>,
    position_type: Tensor<B, 2>,
    embeddings_norm: LayerNorm<B>,
    layers: Vec<Layer<B>>,
    head_first: Linear<B>,
    head_last: Linear<B>,
}

impl<B: Backend> StressNet<B> {
    /// Loads the converted weights onto `device`.
    pub(super) fn load(
        weights: &Weights,
        cfg: &Config,
        device: &B::Device,
    ) -> Result<Self, StressError> {
        let accentor = cfg.accentor;
        let homograph = cfg.homograph;
        let d = homograph.hidden_size;

        // The published word-embedding table is quantized to bytes; it is
        // widened here, once: `scale · (q − zero_point)`.
        let key = "homograph.word_embeddings.q";
        let (quantized, dims) = weights.bytes(key)?;
        if dims != [homograph.vocab_size, d] {
            return Err(model_err(
                key,
                format!(
                    "shape {dims:?}, expected {:?}",
                    [homograph.vocab_size, d]
                ),
            ));
        }
        let scale = weights.scalar("homograph.word_embeddings.scale")?;
        let zero_point =
            weights.scalar("homograph.word_embeddings.zero_point")?;
        let table: Vec<f32> = quantized
            .into_iter()
            // Undo the bias that let the signed bytes travel as unsigned.
            .map(|byte| scale * ((f32::from(byte) - 128.0) - zero_point))
            .collect();
        let word_embeddings = Tensor::from_data(
            TensorData::new(table, [homograph.vocab_size, d]),
            device,
        );

        // Positions are always `0..seq` and every token is of type 0, so the
        // two tables add up into one before anything runs.
        let (positions, dims) =
            weights.parts("homograph.embeddings.position_embeddings.weight")?;
        if dims != [homograph.max_position_embeddings, d] {
            return Err(model_err(
                "homograph.embeddings.position_embeddings.weight",
                format!(
                    "shape {dims:?}, expected {:?}",
                    [homograph.max_position_embeddings, d]
                ),
            ));
        }
        let (types, dims) = weights
            .parts("homograph.embeddings.token_type_embeddings.weight")?;
        if dims != [homograph.type_vocab_size, d] {
            return Err(model_err(
                "homograph.embeddings.token_type_embeddings.weight",
                format!(
                    "shape {dims:?}, expected {:?}",
                    [homograph.type_vocab_size, d]
                ),
            ));
        }
        let combined: Vec<f32> = positions
            .iter()
            .zip(types[..d].iter().cycle())
            .map(|(position, kind)| position + kind)
            .collect();
        let position_type = Tensor::from_data(
            TensorData::new(combined, [homograph.max_position_embeddings, d]),
            device,
        );

        let layers = (0..homograph.num_hidden_layers)
            .map(|index| {
                Layer::load(
                    weights,
                    device,
                    &format!("homograph.layer.{index}"),
                    &homograph,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            ngrams: weight(
                weights,
                device,
                "accentor.embedding.weight",
                [accentor.ngrams, accentor.dim],
            )?,
            stress: Head::load(
                weights,
                device,
                "accentor.stress_clf",
                &accentor,
                accentor.stress_classes,
            )?,
            yo: Head::load(
                weights,
                device,
                "accentor.yo_clf",
                &accentor,
                accentor.yo_classes,
            )?,
            word_embeddings,
            position_type,
            embeddings_norm: LayerNorm::load(
                weights,
                device,
                "homograph.embeddings.LayerNorm",
                d,
                homograph.layer_norm_eps,
            )?,
            layers,
            head_first: Linear::load(
                weights,
                device,
                "homograph.head.0",
                2 * d,
                homograph.head_hidden,
            )?,
            head_last: Linear::load(
                weights,
                device,
                "homograph.head.1",
                homograph.head_hidden,
                1,
            )?,
        })
    }

    /// Scores a padded batch of n-gram bags.
    pub(super) fn accentor(
        &self,
        indices: Tensor<B, 2, Int>,
        mask: Tensor<B, 2>,
    ) -> Result<Vec<WordScores>, StressError> {
        let [batch, width] = indices.dims();
        let dim = self.ngrams.dims()[1];
        let rows = embedding(self.ngrams.clone(), indices);
        let mask = mask.reshape([batch, width, 1]).cast(rows.dtype());
        let total = (rows * mask.clone()).sum_dim(1).reshape([batch, dim]);
        let count = mask.sum_dim(1).reshape([batch, 1]);
        let mean = total / count;

        let stress = host_2d(self.stress.forward(mean.clone()))?;
        let yo = host_2d(self.yo.forward(mean))?;
        Ok(stress
            .iter()
            .zip(&yo)
            .map(|(stress, yo)| scores(&softmax(stress), &softmax(yo)))
            .collect())
    }

    /// Picks a variant for each context of a padded batch.
    pub(super) fn homographs(
        &self,
        ids: Tensor<B, 2, Int>,
        contexts: &[HomographContext],
    ) -> Result<Vec<usize>, StressError> {
        let [_, seq] = ids.dims();
        let [_, d] = self.word_embeddings.dims();
        let words = embedding(self.word_embeddings.clone(), ids);
        let positions = self
            .position_type
            .clone()
            .narrow(0, 0, seq)
            .reshape([1, seq, d]);
        let mut hidden = self.embeddings_norm.forward(words + positions);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }

        // The head reads the opening marker's position and the mean of the
        // word's positions, concatenated.
        let features: Vec<Tensor<B, 2>> = contexts
            .iter()
            .enumerate()
            .map(|(index, context)| {
                let row = hidden.clone().narrow(0, index, 1).reshape([seq, d]);
                let marker = row.clone().narrow(0, context.start, 1);
                let width =
                    context.end.saturating_sub(context.start + 1).max(1);
                let word = row
                    .narrow(0, context.start + 1, width)
                    .mean_dim(0)
                    .reshape([1, d]);
                Tensor::cat(vec![marker, word], 1)
            })
            .collect();
        let features = Tensor::cat(features, 0);
        let hidden = activation::relu(self.head_first.forward(features));
        let logits = self.head_last.forward(hidden);
        Ok(host_2d(logits)?
            .into_iter()
            .map(|row| pick(row[0]))
            .collect())
    }
}

/// Pulls a `(rows, cols)` tensor to the host as f32 rows. The upcast is exact,
/// and the thresholds downstream are decided on these numbers on both runtimes.
fn host_2d<B: Backend>(
    tensor: Tensor<B, 2>,
) -> Result<Vec<Vec<f32>>, StressError> {
    let [rows, cols] = tensor.dims();
    let flat = tensor
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| model_err("reading the output", format!("{e:?}")))?;
    Ok(flat.chunks(cols).take(rows).map(<[f32]>::to_vec).collect())
}
