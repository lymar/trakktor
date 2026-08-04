//! The ERNIE-4.5 decoder on burn.
//!
//! A port of the candle network in
//! [`runtime::ernie`](super::super::runtime::ernie) with the same numerical
//! semantics and the same three commitments that make a decode step cheap: the
//! key/value cache is **preallocated once per generation** — for the prompt
//! plus the longest answer the limits allow — and written in place; the rotary
//! rows are precomputed for every position the answer can reach and sliced per
//! step; and grouped-query attention **folds the queries into the key heads'
//! shape** — a reshape, because the grouping is contiguous — instead of copying
//! the cached keys out to the queries' shape on every step.
//!
//! The prefill is the one place the two paths part: a whole prompt arriving
//! on an empty cache is exactly the fused scaled-dot-product kernel's causal
//! fast path, so the key/value heads are repeated once — cheap, because it
//! happens once per picture, not once per token — and the fused call runs in
//! f32. The per-token path keeps its matmuls in the compute dtype and lifts
//! only the softmax to f32, as the candle decoder does.

#[cfg(test)]
mod tests;

use burn::tensor::{
    DType, Tensor, activation,
    backend::Backend,
    module::{attention, embedding, linear},
    ops::AttentionModuleOptions,
};

use super::{RmsNorm, Weights, apply_rope, indices, matrix, repeat_kv, weight};
use crate::ocr::{error::OcrError, vl::config::ModelConfig};

/// The keys and values already seen, in one preallocated buffer per layer.
///
/// Growing a cache by concatenation re-copies everything seen so far on every
/// step, which prices an n-token answer at n² copied positions. These buffers
/// are sized once — the driver knows every position an answer can reach
/// before its first token — and each step writes its one new column in place.
pub struct Cache<B: Backend> {
    /// One `[1, kv heads, capacity, head dim]` tensor per layer, taken out
    /// while written so the buffer is uniquely held and the write is in
    /// place.
    keys: Vec<Option<Tensor<B, 4>>>,
    values: Vec<Option<Tensor<B, 4>>>,
    /// Positions filled so far, the same in every layer.
    len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Cache<B> {
    /// Allocates a cache able to hold `capacity` positions.
    pub fn new(cfg: &ModelConfig, capacity: usize, device: &B::Device) -> Self {
        let shape = [1, cfg.num_key_value_heads, capacity, cfg.head_dim];
        let buffers = || {
            (0..cfg.num_hidden_layers)
                .map(|_| Some(Tensor::zeros(shape, device)))
                .collect()
        };
        Self {
            keys: buffers(),
            values: buffers(),
            len: 0,
            capacity,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    /// How many positions are already in the cache.
    pub fn len(&self) -> usize { self.len }

    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Writes one layer's new keys and values, `[1, kv heads, seq, dim]`, at
    /// the append position, and returns everything seen so far including
    /// them — views into the buffers, nothing copied.
    fn push(
        &mut self,
        layer: usize,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq = key.dims()[2];
        let total = self.len + seq;
        // The driver sizes the cache for the longest answer it allows, so
        // outgrowing it is a bug, not a data condition.
        assert!(
            total <= self.capacity,
            "the answer outgrew its cache: position {} + {seq} of {}",
            self.len,
            self.capacity
        );

        let span = [0..1, 0..self.kv_heads, self.len..total, 0..self.head_dim];
        let stored_keys = self.keys[layer]
            .take()
            .expect("allocated")
            .slice_assign(span.clone(), key);
        let stored_values = self.values[layer]
            .take()
            .expect("allocated")
            .slice_assign(span, value);

        let seen = (
            stored_keys.clone().narrow(2, 0, total),
            stored_values.clone().narrow(2, 0, total),
        );
        self.keys[layer] = Some(stored_keys);
        self.values[layer] = Some(stored_values);
        seen
    }

    /// Marks `seq` more positions filled, once every layer has pushed them.
    fn advance(&mut self, seq: usize) { self.len += seq; }
}

/// A linear projection without a bias — this decoder has none anywhere.
struct Projection<B: Backend>(Tensor<B, 2>);

impl<B: Backend> Projection<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        key: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self(matrix(weights, device, key, out_dim, in_dim)?))
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(x, self.0.clone(), None)
    }
}

/// One decoder layer.
struct DecoderLayer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
    q_proj: Projection<B>,
    k_proj: Projection<B>,
    v_proj: Projection<B>,
    o_proj: Projection<B>,
    gate_proj: Projection<B>,
    up_proj: Projection<B>,
    down_proj: Projection<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> DecoderLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &ModelConfig,
    ) -> Result<Self, OcrError> {
        let inner = cfg.num_attention_heads * cfg.head_dim;
        let kv_inner = cfg.num_key_value_heads * cfg.head_dim;
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.input_layernorm.weight"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights,
                device,
                &format!("{prefix}.post_attention_layernorm.weight"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            q_proj: Projection::load(
                weights,
                device,
                &format!("{attention}.q_proj.weight"),
                inner,
                cfg.hidden_size,
            )?,
            k_proj: Projection::load(
                weights,
                device,
                &format!("{attention}.k_proj.weight"),
                kv_inner,
                cfg.hidden_size,
            )?,
            v_proj: Projection::load(
                weights,
                device,
                &format!("{attention}.v_proj.weight"),
                kv_inner,
                cfg.hidden_size,
            )?,
            o_proj: Projection::load(
                weights,
                device,
                &format!("{attention}.o_proj.weight"),
                cfg.hidden_size,
                inner,
            )?,
            gate_proj: Projection::load(
                weights,
                device,
                &format!("{mlp}.gate_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?,
            up_proj: Projection::load(
                weights,
                device,
                &format!("{mlp}.up_proj.weight"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?,
            down_proj: Projection::load(
                weights,
                device,
                &format!("{mlp}.down_proj.weight"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
        cache: &mut Cache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let normed = self.input_layernorm.forward(x.clone());
        let split = |projected: Tensor<B, 3>, heads: usize| -> Tensor<B, 4> {
            projected
                .reshape([batch, seq, heads, self.head_dim])
                .swap_dims(1, 2)
        };

        let query = apply_rope(
            split(self.q_proj.forward(normed.clone()), self.heads),
            cos,
            sin,
        );
        let key = apply_rope(
            split(self.k_proj.forward(normed.clone()), self.kv_heads),
            cos,
            sin,
        );
        let value = split(self.v_proj.forward(normed), self.kv_heads);
        let prefill = cache.is_empty() && seq > 1;
        let (keys, values) = cache.push(layer, key, value);
        let groups = self.heads / self.kv_heads;
        let dtype = x.dtype();

        let attended = if prefill {
            // The whole prompt on an empty cache: the fused kernel's causal
            // fast path. Repeating the two key/value heads out to sixteen is
            // a copy the per-token path refuses to pay, but here it happens
            // once per picture. The call runs in f32 — the fused kernel's
            // half-precision path has bitten an earlier port.
            attention(
                query.cast(DType::F32),
                repeat_kv(keys, groups).cast(DType::F32),
                repeat_kv(values, groups).cast(DType::F32),
                None,
                None,
                AttentionModuleOptions {
                    is_causal: true,
                    ..Default::default()
                },
            )
            .cast(dtype)
        } else {
            // One token against the whole cache. Instead of repeating the two
            // cached key heads out to sixteen, the sixteen query heads are
            // folded into two groups of eight — a pure reshape, because the
            // grouping is contiguous — and each group scores against its key
            // head as one matrix. The cache is read where it lies.
            let folded = query.reshape([
                batch,
                self.kv_heads,
                groups * seq,
                self.head_dim,
            ]);
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let scores = folded.matmul(keys.swap_dims(2, 3)).mul_scalar(scale);
            let weights =
                activation::softmax(scores.cast(DType::F32), 3).cast(dtype);
            weights.matmul(values).reshape([
                batch,
                self.heads,
                seq,
                self.head_dim,
            ])
        };

        let attended = attended.swap_dims(1, 2).reshape([
            batch,
            seq,
            self.heads * self.head_dim,
        ]);
        let x = x + self.o_proj.forward(attended);

        let normed = self.post_attention_layernorm.forward(x.clone());
        let gated = activation::silu(self.gate_proj.forward(normed.clone())) *
            self.up_proj.forward(normed);
        x + self.down_proj.forward(gated)
    }
}

/// The decoder and its output head.
pub struct Decoder<B: Backend> {
    embed_tokens: Tensor<B, 2>,
    layers: Vec<DecoderLayer<B>>,
    norm: RmsNorm<B>,
    /// In burn's `[hidden, vocab]` layout.
    lm_head: Tensor<B, 2>,
}

impl<B: Backend> Decoder<B> {
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &ModelConfig,
    ) -> Result<Self, OcrError> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| {
                DecoderLayer::load(
                    weights,
                    device,
                    &format!("model.layers.{index}"),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed_tokens: weight(
                weights,
                device,
                "model.embed_tokens.weight",
                [cfg.vocab_size, cfg.hidden_size],
            )?,
            layers,
            norm: RmsNorm::load(
                weights,
                device,
                "model.norm.weight",
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?,
            // Not tied to the input embedding: the checkpoint carries both.
            lm_head: matrix(
                weights,
                device,
                "lm_head.weight",
                cfg.vocab_size,
                cfg.hidden_size,
            )?,
        })
    }

    /// Looks token ids up in the embedding table, `[1, seq, hidden]`.
    pub fn embed(&self, tokens: &[u32], device: &B::Device) -> Tensor<B, 3> {
        embedding(self.embed_tokens.clone(), indices(tokens, device))
    }

    /// Runs the decoder over `inputs`, shaped `[1, seq, hidden]`, with `cos`
    /// and `sin` the rotary rows of exactly these `seq` positions, shaped
    /// `[1, 1, seq, head_dim]`. Returns the logits of the **last** position
    /// only, as `[1, 1, vocab]` — the only ones a greedy decoder ever looks
    /// at, and 103 424 numbers per position is not a row to materialize
    /// needlessly.
    pub fn forward(
        &self,
        inputs: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        cache: &mut Cache<B>,
    ) -> Tensor<B, 3> {
        let seq = inputs.dims()[1];
        let mut hidden = inputs;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(hidden, &cos, &sin, cache, index);
        }
        cache.advance(seq);
        let last = self.norm.forward(hidden.narrow(1, seq - 1, 1));
        linear(last, self.lm_head.clone(), None)
    }
}
