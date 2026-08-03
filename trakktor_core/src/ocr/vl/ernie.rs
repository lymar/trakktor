//! The decoder: an ERNIE-4.5 transformer that writes the answer token by
//! token.
//!
//! Conventional in shape — grouped-query attention, a gated feed-forward,
//! RMS normalization — with two things worth naming.
//!
//! **The head is wider than the stream.** Sixteen heads of 128 make 2048, but
//! the residual stream is 1024: the queries are projected up and the output
//! back down, so `head_dim` cannot be derived from `hidden_size / heads` and is
//! read from the config instead.
//!
//! **Positions are three-dimensional.** A text token carries the same index in
//! all three axes; a patch of the picture carries its row in one and its column
//! in another. The head dimension is split between the axes in the proportion
//! `[16, 24, 24]`, so the first 32 channels of each half rotate with the
//! sequence, the next 48 with the row, the last 48 with the column.

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, ops::softmax_last_dim};

use super::config::ModelConfig;

/// The keys and values one layer has seen so far, `[1, kv heads, seen, dim]`.
type LayerCache = Option<(Tensor, Tensor)>;

/// The growing state of one generation.
pub struct Cache {
    layers: Vec<LayerCache>,
}

impl Cache {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: vec![None; layers],
        }
    }

    /// How many positions are already in the cache.
    pub fn len(&self) -> usize {
        match self.layers.first().and_then(Option::as_ref) {
            Some((keys, _)) => keys.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

/// Root-mean-square normalization with a learned gain, statistics in `f32`.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// A linear layer without a bias — this decoder has none anywhere.
fn linear(
    rows: usize,
    cols: usize,
    vb: VarBuilder,
    name: &str,
) -> Result<Linear> {
    Ok(Linear::new(vb.get((rows, cols), name)?, None))
}

/// Repeats key/value heads so every query head has a partner.
fn repeat_kv(xs: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(xs.clone());
    }
    let (batch, heads, seq, dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, heads, groups, seq, dim))?
        .reshape((batch, heads * groups, seq, dim))
}

/// Rotates the halves of the last axis: `[a, b] → [-b, a]`.
fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    let first = xs.narrow(D::Minus1, 0, half)?;
    let second = xs.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[second.neg()?, first], D::Minus1)
}

/// The three-axis rotary tables.
///
/// Built per forward pass rather than cached, because the positions of a
/// prefill are not a prefix of anything: the picture's patches carry row and
/// column indices, and the text after them resumes from the larger of the two.
pub struct Rotary {
    inverse: Vec<f32>,
    /// How the head dimension is split between the axes.
    section: Vec<usize>,
    head_dim: usize,
}

impl Rotary {
    fn new(cfg: &ModelConfig) -> Self {
        let head_dim = cfg.head_dim;
        Self {
            inverse: (0..head_dim / 2)
                .map(|i| {
                    (1.0 / cfg
                        .rope_theta
                        .powf(2.0 * i as f64 / head_dim as f64))
                        as f32
                })
                .collect(),
            section: cfg.mrope_section.clone(),
            head_dim,
        }
    }

    /// Cosine and sine for `positions`, laid out `[3, seq]` and already
    /// collapsed onto the sections: the result is `[1, 1, seq, head_dim]`,
    /// ready to broadcast over `[1, heads, seq, head_dim]`.
    fn tables(
        &self,
        positions: &[[i64; 3]],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let seq = positions.len();
        let half = self.head_dim / 2;
        let mut angles = vec![0f32; 3 * seq * self.head_dim];
        for (index, axes) in positions.iter().enumerate() {
            for (axis, &position) in axes.iter().enumerate() {
                let base = (axis * seq + index) * self.head_dim;
                for (i, frequency) in self.inverse.iter().enumerate() {
                    let angle = position as f32 * frequency;
                    angles[base + i] = angle;
                    angles[base + half + i] = angle;
                }
            }
        }
        let angles = Tensor::from_vec(angles, (3, seq, self.head_dim), device)?;

        // Each section of the head dimension takes its angles from one axis,
        // and the split repeats across the two halves of the head.
        let mut chunks = Vec::with_capacity(self.section.len() * 2);
        let mut at = 0;
        for (index, size) in
            self.section.iter().chain(self.section.iter()).enumerate()
        {
            chunks.push(angles.i((index % 3, .., at..at + size))?);
            at += size;
        }
        let angles =
            Tensor::cat(&chunks, 1)?.reshape((1, 1, seq, self.head_dim))?;
        Ok((
            angles.cos()?.to_dtype(dtype)?,
            angles.sin()?.to_dtype(dtype)?,
        ))
    }
}

/// One decoder layer.
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl DecoderLayer {
    fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let inner = cfg.num_attention_heads * cfg.head_dim;
        let kv_inner = cfg.num_key_value_heads * cfg.head_dim;
        let attention = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        Ok(Self {
            input_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            q_proj: linear(
                inner,
                cfg.hidden_size,
                attention.pp("q_proj"),
                "weight",
            )?,
            k_proj: linear(
                kv_inner,
                cfg.hidden_size,
                attention.pp("k_proj"),
                "weight",
            )?,
            v_proj: linear(
                kv_inner,
                cfg.hidden_size,
                attention.pp("v_proj"),
                "weight",
            )?,
            o_proj: linear(
                cfg.hidden_size,
                inner,
                attention.pp("o_proj"),
                "weight",
            )?,
            gate_proj: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                mlp.pp("gate_proj"),
                "weight",
            )?,
            up_proj: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                mlp.pp("up_proj"),
                "weight",
            )?,
            down_proj: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp.pp("down_proj"),
                "weight",
            )?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let normed = self.input_layernorm.forward(xs)?;
        let split = |projected: Tensor, heads: usize| -> Result<Tensor> {
            projected
                .reshape((batch, seq, heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let rope = |xs: &Tensor| -> Result<Tensor> {
            (xs.broadcast_mul(cos)? + rotate_half(xs)?.broadcast_mul(sin)?)?
                .contiguous()
        };

        let query = rope(&split(self.q_proj.forward(&normed)?, self.heads)?)?;
        let key = rope(&split(self.k_proj.forward(&normed)?, self.kv_heads)?)?;
        let value = split(self.v_proj.forward(&normed)?, self.kv_heads)?;

        let (key, value) = match cache.take() {
            None => (key, value),
            Some((past_k, past_v)) => (
                Tensor::cat(&[&past_k, &key], 2)?.contiguous()?,
                Tensor::cat(&[&past_v, &value], 2)?.contiguous()?,
            ),
        };
        *cache = Some((key.clone(), value.clone()));

        let groups = self.heads / self.kv_heads;
        let keys = repeat_kv(&key, groups)?;
        let values = repeat_kv(&value, groups)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let mut weights =
            (query.matmul(&keys.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(mask) = mask {
            weights = weights.broadcast_add(mask)?;
        }
        let weights = softmax_last_dim(&weights.to_dtype(DType::F32)?)?
            .to_dtype(values.dtype())?;
        let attended = weights.matmul(&values)?.transpose(1, 2)?.reshape((
            batch,
            seq,
            self.heads * self.head_dim,
        ))?;
        let xs = (xs + self.o_proj.forward(&attended)?)?;

        let normed = self.post_attention_layernorm.forward(&xs)?;
        let gated = (self.gate_proj.forward(&normed)?.silu()? *
            self.up_proj.forward(&normed)?)?;
        xs + self.down_proj.forward(&gated)?
    }
}

/// Builds the additive causal mask for `seq` new positions arriving after
/// `past` cached ones.
fn causal_mask(
    seq: usize,
    past: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total = past + seq;
    let mut mask = vec![0f32; seq * total];
    for query in 0..seq {
        for key in 0..total {
            if key > past + query {
                mask[query * total + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, seq, total), device)?.to_dtype(dtype)
}

/// The decoder and its output head.
pub struct Decoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: Rotary,
    layer_count: usize,
}

impl Decoder {
    pub fn load(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let model = vb.pp("model");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                cfg,
                model.pp(format!("layers.{index}")),
            )?);
        }
        Ok(Self {
            embed_tokens: Embedding::new(
                model.get(
                    (cfg.vocab_size, cfg.hidden_size),
                    "embed_tokens.weight",
                )?,
                cfg.hidden_size,
            ),
            layers,
            norm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                model.pp("norm"),
            )?,
            // Not tied to the input embedding: the checkpoint carries both.
            lm_head: linear(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.clone(),
                "lm_head.weight",
            )?,
            rotary: Rotary::new(cfg),
            layer_count: cfg.num_hidden_layers,
        })
    }

    pub fn layer_count(&self) -> usize { self.layer_count }

    /// Looks token ids up in the embedding table.
    pub fn embed(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), device)?;
        self.embed_tokens.forward(&ids)
    }

    /// Runs the decoder over `inputs`, shaped `[1, seq, hidden]`, and returns
    /// the logits of the **last** position only, as a flat `[vocab]` — the
    /// only ones a greedy decoder ever looks at, and 103 424 numbers per
    /// position is not a row to materialize needlessly.
    pub fn forward(
        &self,
        inputs: &Tensor,
        positions: &[[i64; 3]],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq, _) = inputs.dims3()?;
        let past = cache.len();
        let device = inputs.device();
        let (cos, sin) =
            self.rotary.tables(positions, device, inputs.dtype())?;
        let mask = if seq > 1 {
            Some(causal_mask(seq, past, device, inputs.dtype())?)
        } else {
            None
        };

        let mut hidden = inputs.clone();
        for (layer, slot) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward(&hidden, &cos, &sin, mask.as_ref(), slot)?;
        }
        let last = hidden.i((.., seq - 1.., ..))?.contiguous()?;
        self.lm_head
            .forward(&self.norm.forward(&last)?)?
            .flatten_all()
    }
}
