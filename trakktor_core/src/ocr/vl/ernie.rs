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
//!
//! The decode step is built to move one token cheaply, because a page pays for
//! it thousands of times: the KV cache is preallocated once per generation and
//! written in place, the rotary tables are computed once for every position
//! the answer could reach, and grouped-query attention folds the queries into
//! the key heads' shape — a reshape — instead of copying the cached keys out
//! to the queries'.

#[cfg(test)]
mod tests;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, ops::softmax_last_dim};

use super::config::ModelConfig;

/// The keys and values already seen, in one preallocated buffer per layer.
///
/// Growing a cache by concatenation re-copies everything seen so far on every
/// step, which prices an n-token answer at n² copied positions. These buffers
/// are sized once — for the prompt plus the longest answer the limits allow —
/// and each step writes its one new column in place.
pub struct Cache {
    /// One `[1, kv heads, capacity, head dim]` tensor per layer.
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    /// Positions filled so far, the same in every layer.
    len: usize,
    capacity: usize,
}

impl Cache {
    /// Allocates a cache able to hold `capacity` positions.
    pub fn new(
        cfg: &ModelConfig,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let shape = (1, cfg.num_key_value_heads, capacity, cfg.head_dim);
        let mut keys = Vec::with_capacity(cfg.num_hidden_layers);
        let mut values = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            keys.push(Tensor::zeros(shape, dtype, device)?);
            values.push(Tensor::zeros(shape, dtype, device)?);
        }
        Ok(Self {
            keys,
            values,
            len: 0,
            capacity,
        })
    }

    /// How many positions are already in the cache.
    pub fn len(&self) -> usize { self.len }

    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Writes one layer's new keys and values, `[1, kv heads, seq, dim]`, at
    /// the append position, and returns everything seen so far including
    /// them — views into the buffers, nothing copied.
    fn push(
        &self,
        layer: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq = key.dim(2)?;
        if self.len + seq > self.capacity {
            candle_core::bail!(
                "the answer outgrew its cache: position {} + {seq} of {}",
                self.len,
                self.capacity
            );
        }
        self.keys[layer].slice_set(key, 2, self.len)?;
        self.values[layer].slice_set(value, 2, self.len)?;
        let total = self.len + seq;
        Ok((
            self.keys[layer].narrow(2, 0, total)?,
            self.values[layer].narrow(2, 0, total)?,
        ))
    }

    /// Marks `seq` more positions filled, once every layer has pushed them.
    fn advance(&mut self, seq: usize) { self.len += seq; }
}

/// Root-mean-square normalization with a learned gain; the fused kernel keeps
/// the statistics in `f32` whatever the activations are.
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
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32)
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

/// The three-axis rotary tables.
struct Rotary {
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

    /// Cosine and sine for `positions`, one row per position, laid out
    /// `[seq, head_dim / 2]`: the rotation pairs channel `i` with channel
    /// `i + head_dim / 2`, so each angle is carried once. Within a row the
    /// channels are split between the three axes by section.
    fn tables(
        &self,
        positions: &[[i64; 3]],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let seq = positions.len();
        let half = self.head_dim / 2;
        let mut angles = vec![0f32; seq * half];
        for (row, axes) in positions.iter().enumerate() {
            let slot = &mut angles[row * half..(row + 1) * half];
            let mut at = 0;
            for (&size, &position) in self.section.iter().zip(axes.iter()) {
                for (channel, angle) in
                    slot[at..at + size].iter_mut().enumerate()
                {
                    *angle = position as f32 * self.inverse[at + channel];
                }
                at += size;
            }
        }
        let angles = Tensor::from_vec(angles, (seq, half), device)?;
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
        cache: &Cache,
        layer: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let normed = self.input_layernorm.forward(xs)?;
        let split = |projected: Tensor, heads: usize| -> Result<Tensor> {
            projected
                .reshape((batch, seq, heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let rope = |xs: &Tensor| candle_nn::rotary_emb::rope(xs, cos, sin);

        let query = rope(&split(self.q_proj.forward(&normed)?, self.heads)?)?;
        let key = rope(&split(self.k_proj.forward(&normed)?, self.kv_heads)?)?;
        let value = split(self.v_proj.forward(&normed)?, self.kv_heads)?;
        let (keys, values) = cache.push(layer, &key, &value)?;
        let total = keys.dim(2)?;

        // Grouped-query attention, without the copy it is usually paid for
        // with. Instead of repeating the two cached key heads out to sixteen,
        // the sixteen query heads are folded into two groups of eight — a
        // pure reshape, because the grouping is contiguous — and each group
        // scores against its key head as one matrix. The cache is read where
        // it lies.
        let groups = self.heads / self.kv_heads;
        let query = query.reshape((
            batch,
            self.kv_heads,
            groups * seq,
            self.head_dim,
        ))?;
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let mut weights = (query.matmul(&keys.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = mask {
            // The mask rows are per query position; unfold the group axis so
            // it can broadcast over them, then fold it back.
            weights = weights
                .reshape((batch, self.kv_heads, groups, seq, total))?
                .broadcast_add(mask)?
                .reshape((batch, self.kv_heads, groups * seq, total))?;
        }
        let weights = softmax_last_dim(&weights.to_dtype(DType::F32)?)?
            .to_dtype(values.dtype())?;
        let attended = weights
            .matmul(&values)?
            .reshape((batch, self.heads, seq, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * self.head_dim))?;
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
        })
    }

    /// Looks token ids up in the embedding table.
    pub fn embed(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), device)?;
        self.embed_tokens.forward(&ids)
    }

    /// The rotary rows for `positions` — built once per generation, because
    /// every position an answer can reach is known before its first token:
    /// the prompt's are laid out by the picture, and the text after it counts
    /// up by one. The caller slices rows off as the decode advances.
    pub fn tables(
        &self,
        positions: &[[i64; 3]],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.rotary.tables(positions, device, dtype)
    }

    /// Runs the decoder over `inputs`, shaped `[1, seq, hidden]`, with `cos`
    /// and `sin` the rotary rows of exactly these `seq` positions. Returns
    /// the logits of the **last** position only, as a flat `[vocab]` — the
    /// only ones a greedy decoder ever looks at, and 103 424 numbers per
    /// position is not a row to materialize needlessly.
    pub fn forward(
        &self,
        inputs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq, _) = inputs.dims3()?;
        let past = cache.len();
        let mask = if seq > 1 {
            Some(causal_mask(seq, past, inputs.device(), inputs.dtype())?)
        } else {
            None
        };

        let mut hidden = inputs.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                cos,
                sin,
                mask.as_ref(),
                cache,
                index,
            )?;
        }
        cache.advance(seq);
        let last = hidden.i((.., seq - 1.., ..))?.contiguous()?;
        self.lm_head
            .forward(&self.norm.forward(&last)?)?
            .flatten_all()
    }
}
