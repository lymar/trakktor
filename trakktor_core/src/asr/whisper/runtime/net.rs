//! The Whisper network on candle.
//!
//! Adapted from the candle project's Whisper implementation
//! (`candle-transformers`, Apache-2.0 OR MIT), with the changes this engine
//! needs:
//!
//! - an **incremental** decoder self-attention cache: each forward appends the
//!   fed positions' K/V, positional embeddings and the causal mask are offset
//!   accordingly (the original recomputed the whole prefix each step);
//! - **cache reordering** by batch index, for beam search;
//! - optional capture of **raw pre-softmax cross-attention scores**, needed for
//!   word-level timing;
//! - **exact GELU** (`gelu_erf`) everywhere: the reference uses the erf form,
//!   not the tanh approximation;
//! - model geometry comes from [`ModelDims`]; tracing instrumentation is
//!   dropped.
//!
//! Weight names follow the safetensors layout of the published checkpoints
//! (`model.encoder.*` / `model.decoder.*`).

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder,
    embedding, linear, linear_no_bias,
};

use super::super::model::ModelDims;

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

/// Sinusoidal positional embeddings of the audio encoder.
pub(super) fn sinusoids(
    length: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment =
        max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales =
        Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time =
        (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

/// Multi-head attention with the reference's scaling: `(d_head)^-0.25`
/// applied to the queries and keys separately.
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    /// Self-attention: accumulated K/V of all cached positions.
    /// Cross-attention: K/V of the encoded audio, computed once per session.
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            kv_cache: None,
        })
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        x.reshape((n_batch, n_ctx, self.n_head, n_state / self.n_head))?
            .transpose(1, 2)
    }

    /// Scaled dot-product attention. Returns the output projection and,
    /// when `capture_qk` is set, the raw pre-softmax scores
    /// `(n_batch, n_head, n_q, n_k)` before the mask is applied — exactly
    /// what word-level timing consumes.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        capture_qk: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (_, _, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.contiguous()?.matmul(&k.contiguous()?)?;
        let captured = if capture_qk { Some(qk.clone()) } else { None };
        if let Some(mask) = mask {
            qk = qk.broadcast_add(mask)?;
        }
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        Ok((self.out.forward(&wv)?, captured))
    }

    /// One-shot attention: K/V projected from `kv_src`, no cache involved.
    /// Serves the encoder (`kv_src = x`, no mask) and the timing forward.
    fn forward_stateless(
        &self,
        x: &Tensor,
        kv_src: &Tensor,
        mask: Option<&Tensor>,
        capture_qk: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(kv_src)?;
        let v = self.value.forward(kv_src)?;
        self.attention(&q, &k, &v, mask, capture_qk)
    }

    /// Incremental self-attention: appends the fed positions' K/V to the
    /// session cache and attends over the whole cached prefix.
    ///
    /// The cache holds K/V already in attention layout — K as
    /// `(batch, head, d_head, positions)` (transposed and pre-scaled), V as
    /// `(batch, head, positions, d_head)` — so a new step only reshapes its own
    /// token(s) and appends them, instead of re-reshaping and transpose-copying
    /// the whole growing prefix every step (that transpose-copy dominated long
    /// windows). The arithmetic matches [`attention`](Self::attention).
    fn forward_self_cached(
        &mut self,
        x: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let n_state = self.query.weight().dim(0)?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let k_new =
            (self.reshape_head(&self.key.forward(x)?)?.transpose(2, 3)? *
                scale)?
                .contiguous()?;
        let v_new = self.reshape_head(&self.value.forward(x)?)?.contiguous()?;
        let (k, v) = match self.kv_cache.take() {
            Some((k_prev, v_prev)) => (
                Tensor::cat(&[&k_prev, &k_new], 3)?,
                Tensor::cat(&[&v_prev, &v_new], 2)?,
            ),
            None => (k_new, v_new),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let q = (self.reshape_head(&self.query.forward(x)?)? * scale)?
            .contiguous()?;
        let qk = q.matmul(&k)?.broadcast_add(mask)?;
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        self.out.forward(&wv)
    }

    /// Cross-attention over encoded audio: K/V computed once per session,
    /// then reused from the cache.
    fn forward_cross_cached(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
    ) -> Result<Tensor> {
        let n_state = self.query.weight().dim(0)?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        // The audio K/V are constant for the whole decode, so project and
        // reshape them into attention layout once (K transposed and pre-scaled
        // to `(batch, head, d_head, frames)`, V as `(batch, head, frames,
        // d_head)`) and reuse across steps. Reshaping the 1500-frame K/V on
        // every single-token step was the dominant decode cost; the arithmetic
        // is otherwise identical to [`attention`](Self::attention).
        let (k, v) = match &self.kv_cache {
            Some((k, v)) => (k.clone(), v.clone()),
            None => {
                let k = (self
                    .reshape_head(&self.key.forward(xa)?)?
                    .transpose(2, 3)? *
                    scale)?
                    .contiguous()?;
                let v = self
                    .reshape_head(&self.value.forward(xa)?)?
                    .contiguous()?;
                self.kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            },
        };
        let q = (self.reshape_head(&self.query.forward(x)?)? * scale)?
            .contiguous()?;
        let qk = q.matmul(&k)?;
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        self.out.forward(&wv)
    }

    fn reset_cache(&mut self) { self.kv_cache = None; }

    /// Reorders the cached sequences along the batch dimension.
    fn rearrange_cache(&mut self, indices: &Tensor) -> Result<()> {
        if let Some((k, v)) = self.kv_cache.take() {
            self.kv_cache = Some((
                k.index_select(indices, 0)?,
                v.index_select(indices, 0)?,
            ));
        }
        Ok(())
    }
}

/// One transformer block: self-attention, optional cross-attention, MLP.
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(
        n_state: usize,
        n_head: usize,
        ca: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn =
            MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross = MultiHeadAttention::load(
                n_state,
                n_head,
                vb.pp("encoder_attn"),
            )?;
            let cross_ln =
                layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross, cross_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(x)?)?
                .gelu_erf()?,
        )
    }

    /// Encoder block: stateless self-attention without a mask.
    fn forward_encoder(&self, x: &Tensor) -> Result<Tensor> {
        let ln = self.attn_ln.forward(x)?;
        let attn = self.attn.forward_stateless(&ln, &ln, None, false)?.0;
        let x = (x + attn)?;
        let mlp = self.mlp(&x)?;
        x + mlp
    }

    /// Decoder block inside a session: cached self- and cross-attention.
    fn forward_decoder_session(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let attn = self
            .attn
            .forward_self_cached(&self.attn_ln.forward(x)?, mask)?;
        let mut x = (x + attn)?;
        if let Some((cross, cross_ln)) = &mut self.cross_attn {
            let ca = cross.forward_cross_cached(&cross_ln.forward(&x)?, xa)?;
            x = (&x + ca)?;
        }
        let mlp = self.mlp(&x)?;
        x + mlp
    }

    /// Decoder block outside any session (timing): stateless throughout,
    /// returning this block's raw cross-attention scores.
    fn forward_decoder_stateless(
        &self,
        x: &Tensor,
        xa: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let ln = self.attn_ln.forward(x)?;
        let attn = self.attn.forward_stateless(&ln, &ln, Some(mask), false)?.0;
        let mut x = (x + attn)?;
        let mut captured = None;
        if let Some((cross, cross_ln)) = &self.cross_attn {
            let (ca, qk) = cross.forward_stateless(
                &cross_ln.forward(&x)?,
                xa,
                None,
                true,
            )?;
            captured = qk;
            x = (&x + ca)?;
        }
        let mlp = self.mlp(&x)?;
        Ok(((x + mlp)?, captured))
    }

    fn reset_cache(&mut self) {
        self.attn.reset_cache();
        if let Some((cross, _)) = &mut self.cross_attn {
            cross.reset_cache();
        }
    }

    fn rearrange_cache(&mut self, indices: &Tensor) -> Result<()> {
        self.attn.rearrange_cache(indices)?;
        if let Some((cross, _)) = &mut self.cross_attn {
            cross.rearrange_cache(indices)?;
        }
        Ok(())
    }
}

/// The audio encoder: two convolutions with exact GELU, sinusoidal
/// positions, transformer blocks, and a final layer norm.
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    pub fn load(vb: VarBuilder, dims: &ModelDims) -> Result<Self> {
        let n_state = dims.n_audio_state;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(dims.n_mels, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        // Sinusoids are derived in f32; cast to the weight dtype so the
        // positional add matches the (possibly f16) activations.
        let positional_embedding =
            sinusoids(dims.n_audio_ctx, n_state, vb.device())?
                .to_dtype(vb.dtype())?;
        let blocks = (0..dims.n_audio_layer)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_state,
                    dims.n_audio_head,
                    false,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    /// `(n_batch, n_mels, 3000)` log-mel input to
    /// `(n_batch, n_audio_ctx, n_audio_state)` features.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?;
        let (_n_batch, seq_len, _hidden) = x.dims3()?;
        let positional_embedding =
            self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward_encoder(&x)?;
        }
        self.ln_post.forward(&x)
    }
}

/// The text decoder: learned token and positional embeddings, transformer
/// blocks with cross-attention, a final layer norm, and logits tied to the
/// token embedding.
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    pub fn load(vb: VarBuilder, dims: &ModelDims) -> Result<Self> {
        let n_state = dims.n_text_state;
        let n_ctx = dims.n_text_ctx;
        let token_embedding =
            embedding(dims.n_vocab, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding =
            vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..dims.n_text_layer)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_state,
                    dims.n_text_head,
                    true,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| {
                (0..n_ctx)
                    .map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 })
            })
            .collect();
        // The additive causal mask (0 / -inf) rides with the attention
        // scores, so it must share their dtype.
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?
            .to_dtype(vb.dtype())?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    /// Session forward: `tokens` are the fed positions `(n_batch, n_step)`,
    /// placed at `offset` after the cached prefix. Returns hidden states for
    /// the fed positions.
    pub fn forward_session(
        &mut self,
        tokens: &Tensor,
        xa: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (_n_batch, n_step) = tokens.dims2()?;
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding =
            self.positional_embedding.narrow(0, offset, n_step)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        // Rows are the fed query positions, columns the whole visible prefix.
        let mask = self
            .mask
            .i((offset..offset + n_step, 0..offset + n_step))?
            .contiguous()?;
        for block in self.blocks.iter_mut() {
            x = block.forward_decoder_session(&x, xa, &mask)?;
        }
        self.ln.forward(&x)
    }

    /// One-shot forward over a complete sequence, returning hidden states
    /// and each layer's raw pre-softmax cross-attention scores.
    pub fn forward_stateless(
        &self,
        tokens: &Tensor,
        xa: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (_n_batch, n_ctx) = tokens.dims2()?;
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding =
            self.positional_embedding.narrow(0, 0, n_ctx)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        let mask = self.mask.i((0..n_ctx, 0..n_ctx))?.contiguous()?;
        let mut cross_qks = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter() {
            let (next, qk) = block.forward_decoder_stateless(&x, xa, &mask)?;
            x = next;
            cross_qks
                .push(qk.expect("decoder blocks always carry cross-attention"));
        }
        Ok((self.ln.forward(&x)?, cross_qks))
    }

    /// Logits over the vocabulary, tied to the token embedding.
    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let n_batch = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(n_batch)?;
        x.matmul(&w.t()?)
    }

    pub fn reset_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_cache();
        }
    }

    pub fn rearrange_cache(&mut self, indices: &Tensor) -> Result<()> {
        for block in self.blocks.iter_mut() {
            block.rearrange_cache(indices)?;
        }
        Ok(())
    }
}
