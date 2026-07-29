//! The two networks on candle.
//!
//! **The accentor** looks a word up as the bag of its character n-grams: the
//! rows those n-grams hit are averaged into one 16-wide vector, and two small
//! MLPs read it — one naming the stressed vowel, one naming the letter `е` that
//! is really `ё`. The bags of a batch are padded into a rectangle and averaged
//! with a mask, so the whole batch is three tensor ops rather than a loop.
//!
//! **The homograph solver** is a stock `rubert-tiny`-class BERT encoder (three
//! layers, hidden 312, exact GELU, f32 LayerNorm statistics) over a sentence in
//! which the ambiguous word is wrapped in marker tokens. Its head reads the
//! opening marker's position and the mean of the word's positions, and answers
//! with one logit: which of the two spellings this sentence wants.
//!
//! Two details are the reference's, not ours, and both are load-bearing:
//! the word-embedding table is quantized to bytes and dequantized here
//! (`w = scale · (q − zero_point)`), and the batch runs **without an attention
//! mask** — the reference pads and lets the encoder look at the padding.

use std::{collections::HashMap, path::Path};

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Module, ops::softmax_last_dim};

use super::model_err;
use crate::stress::{
    error::StressError,
    model::{
        Config, HomographConfig, HomographContext, WordScores, scores, softmax,
    },
};

/// The converted weights, held on the compute device.
struct Weights {
    tensors: HashMap<String, Tensor>,
    dtype: DType,
}

impl Weights {
    fn open(
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, StressError> {
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self { tensors, dtype })
    }

    /// A tensor in its stored form, shape-checked.
    fn raw(&self, key: &str, shape: &[usize]) -> Result<&Tensor, StressError> {
        let tensor = self
            .tensors
            .get(key)
            .ok_or_else(|| model_err(key, "not in the weights"))?;
        if tensor.dims() != shape {
            return Err(model_err(
                key,
                format!("shape {:?}, expected {shape:?}", tensor.dims()),
            ));
        }
        Ok(tensor)
    }

    /// A tensor converted to the compute dtype.
    fn get(&self, key: &str, shape: &[usize]) -> Result<Tensor, StressError> {
        self.raw(key, shape)?
            .to_dtype(self.dtype)
            .map_err(|e| model_err(key, e))
    }

    /// The one scalar values stored as a length-one tensor.
    fn scalar(&self, key: &str) -> Result<f64, StressError> {
        let tensor = self.raw(key, &[1])?;
        let value = tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok(f64::from(value[0]))
    }
}

/// A dense layer with PyTorch's `[out, in]` weight layout.
struct Linear {
    inner: candle_nn::Linear,
}

impl Linear {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Self, StressError> {
        let weight =
            weights.get(&format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias = weights.get(&format!("{prefix}.bias"), &[out_dim])?;
        Ok(Self {
            inner: candle_nn::Linear::new(weight, Some(bias)),
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

/// One of the accentor's heads: four dense layers with ReLU between them.
struct Head {
    layers: Vec<Linear>,
}

impl Head {
    fn load(
        weights: &Weights,
        prefix: &str,
        cfg: &crate::stress::model::AccentorConfig,
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
                    &format!("{prefix}.{index}"),
                    widths[step],
                    widths[step + 1],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if index + 1 < self.layers.len() {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}

/// The n-gram accentor.
struct Accentor {
    table: Tensor,
    stress: Head,
    yo: Head,
}

impl Accentor {
    fn load(weights: &Weights, cfg: &Config) -> Result<Self, StressError> {
        let accentor = cfg.accentor;
        Ok(Self {
            table: weights.get(
                "accentor.embedding.weight",
                &[accentor.ngrams, accentor.dim],
            )?,
            stress: Head::load(
                weights,
                "accentor.stress_clf",
                &accentor,
                accentor.stress_classes,
            )?,
            yo: Head::load(
                weights,
                "accentor.yo_clf",
                &accentor,
                accentor.yo_classes,
            )?,
        })
    }

    /// `indices` is `(batch, width)` of embedding rows, `mask` is
    /// `(batch, width, 1)` with a one per real row.
    fn forward(
        &self,
        indices: &Tensor,
        mask: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (batch, width) = indices.dims2()?;
        let dim = self.table.dim(1)?;
        let rows = self
            .table
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((batch, width, dim))?;
        let mask = mask.to_dtype(rows.dtype())?;
        let total = rows.broadcast_mul(&mask)?.sum(1)?;
        let count = mask.sum(1)?;
        let mean = total.broadcast_div(&count)?;
        Ok((self.stress.forward(&mean)?, self.yo.forward(&mean)?))
    }
}

/// The encoder's input embeddings.
struct Embeddings {
    words: Tensor,
    positions: candle_nn::Embedding,
    token_type: Tensor,
    layer_norm: candle_nn::LayerNorm,
}

impl Embeddings {
    fn load(
        weights: &Weights,
        cfg: &HomographConfig,
    ) -> Result<Self, StressError> {
        // The published table is quantized to bytes; it is widened once, here.
        let quantized = weights.raw(
            "homograph.word_embeddings.q",
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        let scale = weights.scalar("homograph.word_embeddings.scale")?;
        let zero_point =
            weights.scalar("homograph.word_embeddings.zero_point")?;
        let words = quantized
            .to_dtype(DType::F32)
            // Undo the bias that let the signed bytes travel as unsigned, then
            // dequantize: `scale · (q − zero_point)`.
            .and_then(|t| t.affine(1.0, -128.0))
            .and_then(|t| t.affine(scale, -scale * zero_point))
            .and_then(|t| t.to_dtype(weights.dtype))
            .map_err(|e| model_err("dequantizing the word embeddings", e))?;

        let positions = weights.get(
            "homograph.embeddings.position_embeddings.weight",
            &[cfg.max_position_embeddings, cfg.hidden_size],
        )?;
        let token_type = weights.get(
            "homograph.embeddings.token_type_embeddings.weight",
            &[cfg.type_vocab_size, cfg.hidden_size],
        )?;
        let layer_norm = candle_nn::LayerNorm::new(
            weights.get(
                "homograph.embeddings.LayerNorm.weight",
                &[cfg.hidden_size],
            )?,
            weights.get(
                "homograph.embeddings.LayerNorm.bias",
                &[cfg.hidden_size],
            )?,
            cfg.layer_norm_eps,
        );
        Ok(Self {
            words,
            positions: candle_nn::Embedding::new(positions, cfg.hidden_size),
            // Every token is of type 0; the row is added to every position.
            token_type: token_type
                .narrow(0, 0, 1)
                .map_err(|e| model_err("token type row", e))?,
            layer_norm,
        })
    }

    /// `(batch, seq)` ids → `(batch, seq, hidden)`.
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let (_, seq) = ids.dims2()?;
        let words = self
            .words
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((ids.dim(0)?, seq, self.words.dim(1)?))?;
        // The reference's position ids are simply `0..seq`.
        let positions = Tensor::arange(0u32, seq as u32, ids.device())?;
        let positions = self.positions.forward(&positions)?.unsqueeze(0)?;
        let embeddings = words
            .broadcast_add(&positions)?
            .broadcast_add(&self.token_type.unsqueeze(0)?)?;
        self.layer_norm.forward(&embeddings)
    }
}

/// Bidirectional self-attention, without a mask.
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(
        weights: &Weights,
        prefix: &str,
        cfg: &HomographConfig,
    ) -> Result<Self, StressError> {
        let all = cfg.hidden_size;
        Ok(Self {
            query: Linear::load(weights, &format!("{prefix}.query"), all, all)?,
            key: Linear::load(weights, &format!("{prefix}.key"), all, all)?,
            value: Linear::load(weights, &format!("{prefix}.value"), all, all)?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn split(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }

    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let q = self.split(&self.query.forward(hidden)?)?;
        let k = self.split(&self.key.forward(hidden)?)?;
        let v = self.split(&self.value.forward(hidden)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = softmax_last_dim(&scores)?;
        let (b, _, s, _) = probs.dims4()?;
        probs
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, self.heads * self.head_dim))
    }
}

/// A dense projection with a residual add and layer norm.
struct DenseNorm {
    dense: Linear,
    layer_norm: candle_nn::LayerNorm,
}

impl DenseNorm {
    fn load(
        weights: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        eps: f64,
    ) -> Result<Self, StressError> {
        Ok(Self {
            dense: Linear::load(
                weights,
                &format!("{prefix}.dense"),
                in_dim,
                out_dim,
            )?,
            layer_norm: candle_nn::LayerNorm::new(
                weights
                    .get(&format!("{prefix}.LayerNorm.weight"), &[out_dim])?,
                weights.get(&format!("{prefix}.LayerNorm.bias"), &[out_dim])?,
                eps,
            ),
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        residual: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden = self.dense.forward(hidden)?;
        self.layer_norm.forward(&(hidden + residual)?)
    }
}

/// One transformer layer.
struct Layer {
    attention: SelfAttention,
    attention_output: DenseNorm,
    intermediate: Linear,
    output: DenseNorm,
}

impl Layer {
    fn load(
        weights: &Weights,
        prefix: &str,
        cfg: &HomographConfig,
    ) -> Result<Self, StressError> {
        Ok(Self {
            attention: SelfAttention::load(
                weights,
                &format!("{prefix}.attention.self"),
                cfg,
            )?,
            attention_output: DenseNorm::load(
                weights,
                &format!("{prefix}.attention.output"),
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
            intermediate: Linear::load(
                weights,
                &format!("{prefix}.intermediate.dense"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?,
            output: DenseNorm::load(
                weights,
                &format!("{prefix}.output"),
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let attention = self.attention.forward(hidden)?;
        let hidden = self.attention_output.forward(&attention, hidden)?;
        // Exact GELU (erf), matching the reference's `gelu`.
        let intermediate = self.intermediate.forward(&hidden)?.gelu_erf()?;
        self.output.forward(&intermediate, &hidden)
    }
}

/// Both loaded networks.
pub(super) struct StressNet {
    accentor: Accentor,
    embeddings: Embeddings,
    layers: Vec<Layer>,
    head_first: Linear,
    head_last: Linear,
}

impl StressNet {
    /// Loads the converted weights onto `device` at `dtype`.
    pub fn load(
        path: &Path,
        cfg: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, StressError> {
        let weights = Weights::open(path, device, dtype)?;
        let homograph = cfg.homograph;
        let layers = (0..homograph.num_hidden_layers)
            .map(|index| {
                Layer::load(
                    &weights,
                    &format!("homograph.layer.{index}"),
                    &homograph,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            accentor: Accentor::load(&weights, cfg)?,
            embeddings: Embeddings::load(&weights, &homograph)?,
            layers,
            head_first: Linear::load(
                &weights,
                "homograph.head.0",
                2 * homograph.hidden_size,
                homograph.head_hidden,
            )?,
            head_last: Linear::load(
                &weights,
                "homograph.head.1",
                homograph.head_hidden,
                1,
            )?,
        })
    }

    /// Scores a padded batch of n-gram bags.
    pub fn accentor(
        &self,
        indices: &Tensor,
        mask: &Tensor,
    ) -> Result<Vec<WordScores>, StressError> {
        let (stress, yo) = self
            .accentor
            .forward(indices, mask)
            .map_err(|e| model_err("the accentor forward", e))?;
        let pull = |logits: Tensor| -> Result<Vec<Vec<f32>>, StressError> {
            logits
                .to_dtype(DType::F32)
                .and_then(|t| t.to_vec2::<f32>())
                .map_err(|e| model_err("reading the accentor output", e))
        };
        let stress = pull(stress)?;
        let yo = pull(yo)?;
        Ok(stress
            .iter()
            .zip(&yo)
            .map(|(stress, yo)| scores(&softmax(stress), &softmax(yo)))
            .collect())
    }

    /// Picks a variant for each context of a padded batch.
    pub fn homographs(
        &self,
        ids: &Tensor,
        contexts: &[HomographContext],
    ) -> Result<Vec<usize>, StressError> {
        let hidden = self
            .encode(ids)
            .map_err(|e| model_err("the homograph forward", e))?;
        let logits = self
            .decide(&hidden, contexts)
            .map_err(|e| model_err("the homograph head", e))?;
        Ok(logits.into_iter().map(crate::stress::model::pick).collect())
    }

    fn encode(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden = self.embeddings.forward(ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    /// The head reads the opening marker's position and the mean of the word's
    /// positions, concatenated.
    fn decide(
        &self,
        hidden: &Tensor,
        contexts: &[HomographContext],
    ) -> candle_core::Result<Vec<f32>> {
        let mut features = Vec::with_capacity(contexts.len());
        for (index, context) in contexts.iter().enumerate() {
            let row = hidden.narrow(0, index, 1)?.squeeze(0)?;
            let marker = row.narrow(0, context.start, 1)?;
            let width = context.end.saturating_sub(context.start + 1).max(1);
            let word =
                row.narrow(0, context.start + 1, width)?.mean_keepdim(0)?;
            features.push(Tensor::cat(&[marker, word], 1)?);
        }
        let features = Tensor::cat(&features, 0)?;
        let hidden = self.head_first.forward(&features)?.relu()?;
        self.head_last
            .forward(&hidden)?
            .to_dtype(DType::F32)?
            .squeeze(D::Minus1)?
            .to_vec1::<f32>()
    }
}
