//! The talker and its code predictor.
//!
//! The talker is a Qwen3 decoder — grouped-query attention with per-head
//! normalization of the queries and keys, a gated feed-forward, and rotary
//! positions — driven one frame at a time over a growing key/value cache. It is
//! fed two tracks summed channel-wise: text embeddings projected into its width
//! and codec-token embeddings.
//!
//! Its head predicts only codebook 0 of a frame. The **code predictor** then
//! fills the remaining codebooks: a small decoder of the same shape that runs
//! once per codebook, each step reading the previous code through its own
//! embedding table and scoring the next through its own head. The finished
//! frame is folded back into a single embedding — the sum of all its
//! codebooks — and becomes the talker's next input.
//!
//! # Positions
//!
//! The reference gives the talker a three-section rotary embedding, but every
//! section is handed the same position ids: the sections exist for models that
//! also consume images, and speech synthesis has no second modality. With the
//! sections equal, the interleaving the reference performs is the identity, so
//! plain rotary positions are computed here instead.

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops::softmax};

use super::layers::{RmsNorm, rope_tables, rotate_half};
use crate::tts::qwen3_tts::config::{CodePredictorConfig, TalkerConfig};

/// Cached keys and values of one attention layer.
type LayerCache = Option<(Tensor, Tensor)>;

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

/// Applies rotary positions to `xs`, shaped `[batch, heads, time, head_dim]`.
fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(xs)?;
    xs.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
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
#[derive(Debug)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    shape: AttentionShape,
}

impl DecoderLayer {
    fn load(
        hidden: usize,
        intermediate: usize,
        shape: AttentionShape,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner = shape.num_heads * shape.head_dim;
        let kv_inner = shape.num_kv_heads * shape.head_dim;
        let no_bias =
            |rows: usize, cols: usize, name: &str| -> Result<Linear> {
                Ok(Linear::new(vb.get((rows, cols), name)?, None))
            };
        Ok(Self {
            input_layernorm: RmsNorm::load(
                hidden,
                shape.eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::load(
                hidden,
                shape.eps,
                vb.pp("post_attention_layernorm"),
            )?,
            q_proj: no_bias(inner, hidden, "self_attn.q_proj.weight")?,
            k_proj: no_bias(kv_inner, hidden, "self_attn.k_proj.weight")?,
            v_proj: no_bias(kv_inner, hidden, "self_attn.v_proj.weight")?,
            o_proj: no_bias(hidden, inner, "self_attn.o_proj.weight")?,
            // Normalization is per head, over the head dimension.
            q_norm: RmsNorm::load(
                shape.head_dim,
                shape.eps,
                vb.pp("self_attn.q_norm"),
            )?,
            k_norm: RmsNorm::load(
                shape.head_dim,
                shape.eps,
                vb.pp("self_attn.k_norm"),
            )?,
            gate_proj: no_bias(intermediate, hidden, "mlp.gate_proj.weight")?,
            up_proj: no_bias(intermediate, hidden, "mlp.up_proj.weight")?,
            down_proj: no_bias(hidden, intermediate, "mlp.down_proj.weight")?,
            shape,
        })
    }

    /// Runs the layer, appending this step's keys and values to `cache`.
    ///
    /// `mask` is `None` when a single position attends to everything cached.
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let shape = self.shape;
        let normed = self.input_layernorm.forward(xs)?;

        let split = |projected: Tensor, heads: usize| -> Result<Tensor> {
            projected
                .reshape((batch, seq, heads, shape.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        // The per-head normalization runs before the heads are transposed out,
        // so it sees the head dimension as the last axis either way.
        let query = self.q_norm.forward(
            &self.q_proj.forward(&normed)?.reshape((
                batch,
                seq,
                shape.num_heads,
                shape.head_dim,
            ))?,
        )?;
        let key = self.k_norm.forward(
            &self.k_proj.forward(&normed)?.reshape((
                batch,
                seq,
                shape.num_kv_heads,
                shape.head_dim,
            ))?,
        )?;
        let query = query.transpose(1, 2)?.contiguous()?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value = split(self.v_proj.forward(&normed)?, shape.num_kv_heads)?;

        let query = apply_rope(&query, cos, sin)?;
        let key = apply_rope(&key, cos, sin)?;

        // Grow the cache with this step's keys and values.
        let (key, value) = match cache.take() {
            None => (key, value),
            Some((past_k, past_v)) => (
                Tensor::cat(&[&past_k, &key], 2)?.contiguous()?,
                Tensor::cat(&[&past_v, &value], 2)?.contiguous()?,
            ),
        };
        *cache = Some((key.clone(), value.clone()));

        let groups = shape.num_heads / shape.num_kv_heads;
        let key = repeat_kv(&key, groups)?;
        let value = repeat_kv(&value, groups)?;

        let scale = 1f64 / (shape.head_dim as f64).sqrt();
        let mut weights = (query.matmul(&key.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = mask {
            weights = weights.broadcast_add(mask)?;
        }
        let weights = softmax(&weights.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(value.dtype())?;
        let attended = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq,
            shape.num_heads * shape.head_dim,
        ))?;
        let xs = (xs + self.o_proj.forward(&attended)?)?;

        let normed = self.post_attention_layernorm.forward(&xs)?;
        let gated = (self.gate_proj.forward(&normed)?.silu()? *
            self.up_proj.forward(&normed)?)?;
        xs + self.down_proj.forward(&gated)?
    }
}

/// Builds the additive causal mask for a block of `seq` new positions arriving
/// after `past` cached ones.
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

/// The code predictor: fills codebooks 1.. of a frame, one pass per codebook.
#[derive(Debug)]
struct CodePredictor {
    cfg: CodePredictorConfig,
    /// One embedding table per residual codebook.
    embeddings: Vec<Tensor>,
    /// One output head per residual codebook.
    heads: Vec<Tensor>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// Present only when the backbone is wider than the predictor.
    projection: Option<Linear>,
    device: Device,
    dtype: DType,
}

impl CodePredictor {
    fn load(
        cfg: &CodePredictorConfig,
        talker_hidden: usize,
        groups: usize,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let shape = AttentionShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        };
        let model = vb.pp("model");
        let residuals = groups - 1;

        let mut embeddings = Vec::with_capacity(residuals);
        let mut heads = Vec::with_capacity(residuals);
        for index in 0..residuals {
            embeddings.push(
                model
                    .pp("codec_embedding")
                    .pp(index)
                    .get((cfg.vocab_size, talker_hidden), "weight")?,
            );
            heads.push(
                vb.pp("lm_head")
                    .pp(index)
                    .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            );
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                cfg.hidden_size,
                cfg.intermediate_size,
                shape,
                model.pp("layers").pp(index),
            )?);
        }

        // The checkpoint carries a projection only when the widths differ;
        // otherwise the backbone state is handed over untouched.
        let projection = if cfg.hidden_size == talker_hidden {
            None
        } else {
            Some(Linear::new(
                vb.get(
                    (cfg.hidden_size, talker_hidden),
                    "small_to_mtp_projection.weight",
                )?,
                Some(vb.get(cfg.hidden_size, "small_to_mtp_projection.bias")?),
            ))
        };

        Ok(Self {
            cfg: cfg.clone(),
            embeddings,
            heads,
            layers,
            norm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                model.pp("norm"),
            )?,
            projection,
            device,
            dtype,
        })
    }

    /// Projects a backbone state into the predictor's width.
    fn project(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.projection {
            None => Ok(xs.clone()),
            Some(projection) => projection.forward(xs),
        }
    }

    /// Runs the stack over `embeds`, returning the last position's state.
    fn forward(
        &self,
        embeds: &Tensor,
        cache: &mut [LayerCache],
        position: usize,
    ) -> Result<Tensor> {
        let seq = embeds.dim(1)?;
        let (cos, sin) = rope_for(
            self.cfg.head_dim,
            position,
            seq,
            self.cfg.rope_theta,
            &self.device,
            self.dtype,
        )?;
        let mask = if seq > 1 {
            Some(causal_mask(seq, position, &self.device, self.dtype)?)
        } else {
            None
        };

        let mut hidden = self.project(embeds)?;
        for (layer, cache) in self.layers.iter().zip(cache.iter_mut()) {
            hidden =
                layer.forward(&hidden, &cos, &sin, mask.as_ref(), cache)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        hidden.narrow(1, seq - 1, 1)
    }

    /// Generates the residual codebooks of one frame.
    ///
    /// `backbone_state` is the talker's state for the frame and `first_embed`
    /// the embedding of the codebook-0 code it already picked.
    fn generate<P>(
        &self,
        backbone_state: &Tensor,
        first_embed: &Tensor,
        mut pick: P,
    ) -> Result<Vec<u32>>
    where
        P: FnMut(&Tensor, usize) -> Result<u32>,
    {
        let mut cache: Vec<LayerCache> = vec![None; self.layers.len()];
        let mut position = 0usize;

        // The predictor is primed with the backbone state and the frame's
        // first code, so its first output scores codebook 1.
        let primed = Tensor::cat(&[backbone_state, first_embed], 1)?;
        let mut state = self.forward(&primed, &mut cache, position)?;
        position += primed.dim(1)?;

        let mut codes = Vec::with_capacity(self.heads.len());
        for step in 0..self.heads.len() {
            let logits = state
                .squeeze(1)?
                .matmul(&self.heads[step].t()?.to_dtype(state.dtype())?)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            let code = pick(&logits, step)?;
            codes.push(code);

            // The last code needs no follow-up pass.
            if step + 1 == self.heads.len() {
                break;
            }
            let embed = self.embeddings[step]
                .i(code as usize)?
                .reshape((1, 1, ()))?
                .to_dtype(self.dtype)?;
            state = self.forward(&embed, &mut cache, position)?;
            position += 1;
        }
        Ok(codes)
    }

    /// Embeds a residual code with the table belonging to its codebook.
    fn embed(&self, index: usize, code: u32) -> Result<Tensor> {
        self.embeddings[index]
            .i(code as usize)?
            .reshape((1, 1, ()))?
            .to_dtype(self.dtype)
    }
}

/// Builds rotary tables covering `seq` positions starting at `start`.
fn rope_for(
    head_dim: usize,
    start: usize,
    seq: usize,
    theta: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = rope_tables(head_dim, start + seq, theta, device)?;
    let cos = cos.narrow(0, start, seq)?;
    let sin = sin.narrow(0, start, seq)?;
    // The reference lays the table across both halves of the head.
    let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?.to_dtype(dtype)?;
    let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?.to_dtype(dtype)?;
    Ok((
        cos.unsqueeze(0)?.unsqueeze(0)?,
        sin.unsqueeze(0)?.unsqueeze(0)?,
    ))
}

/// The two-layer projection that lifts text embeddings into the talker's width.
#[derive(Debug)]
struct TextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl TextProjection {
    fn load(text_hidden: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(
                vb.get((text_hidden, text_hidden), "linear_fc1.weight")?,
                Some(vb.get(text_hidden, "linear_fc1.bias")?),
            ),
            fc2: Linear::new(
                vb.get((hidden, text_hidden), "linear_fc2.weight")?,
                Some(vb.get(hidden, "linear_fc2.bias")?),
            ),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.silu()?)
    }
}

/// The talker together with its code predictor.
#[derive(Debug)]
pub struct Talker {
    cfg: TalkerConfig,
    device: Device,
    dtype: DType,
    text_embedding: Tensor,
    codec_embedding: Tensor,
    text_projection: TextProjection,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: Tensor,
    predictor: CodePredictor,
    cache: Vec<LayerCache>,
    position: usize,
}

impl Talker {
    /// Loads the talker and its code predictor from a checkpoint.
    pub fn load(
        cfg: &TalkerConfig,
        vb: VarBuilder,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = vb.pp("talker");
        let model = vb.pp("model");
        let shape = AttentionShape {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                cfg.hidden_size,
                cfg.intermediate_size,
                shape,
                model.pp("layers").pp(index),
            )?);
        }

        let predictor = CodePredictor::load(
            &cfg.code_predictor,
            cfg.hidden_size,
            cfg.num_code_groups,
            vb.pp("code_predictor"),
            device.clone(),
            dtype,
        )?;

        Ok(Self {
            cfg: cfg.clone(),
            text_embedding: model
                .pp("text_embedding")
                .get((cfg.text_vocab_size, cfg.text_hidden_size), "weight")?,
            codec_embedding: model
                .pp("codec_embedding")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            text_projection: TextProjection::load(
                cfg.text_hidden_size,
                cfg.hidden_size,
                vb.pp("text_projection"),
            )?,
            norm: RmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                model.pp("norm"),
            )?,
            codec_head: vb
                .pp("codec_head")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            layers,
            predictor,
            cache: vec![None; cfg.num_hidden_layers],
            position: 0,
            device,
            dtype,
        })
    }

    /// The geometry this talker was built for.
    pub fn config(&self) -> &TalkerConfig { &self.cfg }

    /// Forgets the cached keys and values, readying the talker for a new run.
    pub fn reset(&mut self) {
        self.cache = vec![None; self.layers.len()];
        self.position = 0;
    }

    /// Embeds text token ids and projects them into the talker's width,
    /// yielding `[1, ids, hidden]`.
    pub fn embed_text(&self, ids: &[u32]) -> Result<Tensor> {
        let index = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        let embedded = self
            .text_embedding
            .index_select(&index, 0)?
            .reshape((1, ids.len(), self.cfg.text_hidden_size))?;
        self.text_projection.forward(&embedded)
    }

    /// Embeds codec token ids, yielding `[1, ids, hidden]`.
    pub fn embed_codec(&self, ids: &[u32]) -> Result<Tensor> {
        let index = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        self.codec_embedding.index_select(&index, 0)?.reshape((
            1,
            ids.len(),
            self.cfg.hidden_size,
        ))
    }

    /// Runs `embeds` (`[1, time, hidden]`) through the stack, returning the
    /// codebook-0 logits of the last position and that position's state.
    pub fn forward(&mut self, embeds: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq = embeds.dim(1)?;
        let (cos, sin) = rope_for(
            self.cfg.head_dim,
            self.position,
            seq,
            self.cfg.rope_theta,
            &self.device,
            self.dtype,
        )?;
        let mask = if seq > 1 {
            Some(causal_mask(seq, self.position, &self.device, self.dtype)?)
        } else {
            None
        };

        let mut hidden = embeds.to_dtype(self.dtype)?;
        for (layer, cache) in self.layers.iter().zip(self.cache.iter_mut()) {
            hidden =
                layer.forward(&hidden, &cos, &sin, mask.as_ref(), cache)?;
        }
        self.position += seq;

        let hidden = self.norm.forward(&hidden)?;
        let last = hidden.narrow(1, seq - 1, 1)?;
        let logits = last
            .squeeze(1)?
            .matmul(&self.codec_head.t()?.to_dtype(last.dtype())?)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        Ok((logits, last))
    }

    /// Fills the residual codebooks of one frame; see
    /// [`CodePredictor::generate`].
    pub fn predict_residuals<P>(
        &self,
        backbone_state: &Tensor,
        first_code: u32,
        pick: P,
    ) -> Result<Vec<u32>>
    where
        P: FnMut(&Tensor, usize) -> Result<u32>,
    {
        let first_embed = self.embed_codec(&[first_code])?;
        self.predictor.generate(backbone_state, &first_embed, pick)
    }

    /// Folds a finished frame into the single embedding the talker reads next:
    /// the sum of every codebook's embedding.
    pub fn fold_frame(&self, frame: &[u32]) -> Result<Tensor> {
        let mut summed = self.embed_codec(&frame[..1])?;
        for (index, &code) in frame[1..].iter().enumerate() {
            summed = (summed + self.predictor.embed(index, code)?)?;
        }
        Ok(summed)
    }
}
