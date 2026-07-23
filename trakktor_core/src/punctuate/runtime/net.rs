//! The punctuation network on candle.
//!
//! A stock bidirectional XLM-RoBERTa encoder (the same architecture the
//! structify port already runs, adapted from `candle-transformers`,
//! Apache-2.0 OR MIT) followed by the cascade of four classification heads of
//! `ConditionedPCSDecoder`:
//!
//! 1. two punctuation heads read the encoder output — `post` (17 classes) and
//!    `pre` (2 classes);
//! 2. the argmax post prediction is embedded and concatenated to the encoder
//!    output to condition the sentence-boundary (`seg`) head (2 classes);
//! 3. the argmax seg prediction is shifted right by one (with the first slot
//!    forced to a boundary) and concatenated to the encoder output to condition
//!    the per-character true-casing (`cap`) head (16 per-character classes).
//!
//! Every head is a two-layer MLP (`Linear → ReLU → Linear`). The forward runs
//! over full windows with **no attention mask** — the driver batches windows of
//! equal length, so padding never exists. Exact GELU and f32 LayerNorm
//! statistics match the structify port; the final thresholds (`softmax`/
//! `sigmoid`) are computed in f32.
//!
//! Weight names follow the NeMo checkpoint: `bert_model.*` for the encoder,
//! `_decoder.*` for the heads.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm,
    linear,
    ops::{sigmoid, softmax_last_dim},
};

use crate::punctuate::{model::Config, runtime::RawOutputs};

struct Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    padding_idx: u32,
}

impl Embeddings {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            word_embeddings: embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("word_embeddings"),
            )?,
            position_embeddings: embedding(
                cfg.max_position_embeddings,
                cfg.hidden_size,
                vb.pp("position_embeddings"),
            )?,
            token_type_embeddings: embedding(
                cfg.type_vocab_size,
                cfg.hidden_size,
                vb.pp("token_type_embeddings"),
            )?,
            layer_norm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
            padding_idx: cfg.pad_token_id,
        })
    }

    /// `input_ids` is `(batch, seq)`. XLM-R position ids start at
    /// `padding_idx + 1` and count only non-padding tokens (the reference's
    /// `cumsum(mask) * mask + padding_idx`); the single token-type is 0.
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;

        let mask = input_ids.ne(self.padding_idx)?.to_dtype(DType::F32)?;
        let position_ids = (mask.cumsum(1)? * &mask)?
            .affine(1.0, f64::from(self.padding_idx))?
            .to_dtype(DType::U32)?;
        let position_embeddings =
            self.position_embeddings.forward(&position_ids)?;

        // The single token type is 0, so the ids are just zeros of the input's
        // shape (and integer dtype).
        let token_type_embeddings = self
            .token_type_embeddings
            .forward(&input_ids.zeros_like()?)?;

        let embeddings = input_embeddings
            .add(&token_type_embeddings)?
            .add(&position_embeddings)?;
        self.layer_norm.forward(&embeddings)
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let all = cfg.hidden_size;
        Ok(Self {
            query: linear(all, all, vb.pp("query"))?,
            key: linear(all, all, vb.pp("key"))?,
            value: linear(all, all, vb.pp("value"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `(batch, seq, hidden)` → `(batch, heads, seq, head_dim)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }

    /// Full bidirectional attention with no mask (windows are never padded).
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let q = self.split_heads(&self.query.forward(hidden)?)?;
        let k = self.split_heads(&self.key.forward(hidden)?)?;
        let v = self.split_heads(&self.value.forward(hidden)?)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let probs = softmax_last_dim(&scores)?;

        let (b, _, s, _) = probs.dims4()?;
        probs
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))
    }
}

/// A dense projection with a residual add and layer norm — shared by the
/// attention output and the FFN output.
struct DenseNorm {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl DenseNorm {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            dense: linear(in_dim, out_dim, vb.pp("dense"))?,
            layer_norm: layer_norm(out_dim, eps, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let hidden = self.dense.forward(hidden)?;
        self.layer_norm.forward(&(hidden + residual)?)
    }
}

struct Layer {
    attention: SelfAttention,
    attention_output: DenseNorm,
    intermediate: Linear,
    output: DenseNorm,
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = vb.pp("attention");
        Ok(Self {
            attention: SelfAttention::load(attn.pp("self"), cfg)?,
            attention_output: DenseNorm::load(
                attn.pp("output"),
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
            intermediate: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("intermediate").pp("dense"),
            )?,
            output: DenseNorm::load(
                vb.pp("output"),
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let attn = self.attention.forward(hidden)?;
        let hidden = self.attention_output.forward(&attn, hidden)?;
        // Exact GELU (erf), matching the reference `hidden_act = "gelu"`.
        let intermediate = self.intermediate.forward(&hidden)?.gelu_erf()?;
        self.output.forward(&intermediate, &hidden)
    }
}

/// A two-layer classification head: `Linear → ReLU → Linear`, matching NeMo's
/// `ClassificationHead` (no activation on the last layer).
struct Head {
    first: Linear,
    last: Linear,
}

impl Head {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        mid_dim: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let linears = vb.pp("_linears");
        Ok(Self {
            first: linear(in_dim, mid_dim, linears.pp("0"))?,
            last: linear(mid_dim, out_dim, linears.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(x)?.relu()?;
        self.last.forward(&x)
    }
}

/// The loaded punctuation network.
pub struct PunctModel {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    punct_emb: Embedding,
    punct_head_post: Head,
    punct_head_pre: Head,
    seg_head: Head,
    cap_head: Head,
    device: Device,
    dtype: DType,
}

impl PunctModel {
    /// Loads the encoder (`bert_model.*`) and the four heads (`_decoder.*`).
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let bert = vb.pp("bert_model");
        let embeddings = Embeddings::load(bert.pp("embeddings"), cfg)?;
        let encoder = bert.pp("encoder");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(encoder.pp(format!("layer.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        let decoder = vb.pp("_decoder");
        let punct_emb = embedding(
            cfg.punct_post_classes,
            cfg.emb_dim,
            decoder.pp("_punct_emb"),
        )?;
        let punct_head_post = Head::load(
            decoder.pp("_punct_head_post"),
            cfg.hidden_size,
            cfg.punct_head_intermediate,
            cfg.punct_post_classes,
        )?;
        let punct_head_pre = Head::load(
            decoder.pp("_punct_head_pre"),
            cfg.hidden_size,
            cfg.punct_head_intermediate,
            cfg.punct_pre_classes,
        )?;
        let seg_head = Head::load(
            decoder.pp("_seg_head"),
            cfg.hidden_size + cfg.emb_dim,
            cfg.seg_head_intermediate,
            2,
        )?;
        let cap_head = Head::load(
            decoder.pp("_cap_head"),
            cfg.hidden_size + 1,
            cfg.cap_head_intermediate,
            cfg.cap_classes,
        )?;

        Ok(Self {
            embeddings,
            layers,
            punct_emb,
            punct_head_post,
            punct_head_pre,
            seg_head,
            cap_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Runs the encoder over `(batch, seq)` `U32` ids.
    fn encode(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    /// Forwards a batch of equal-length windows and returns the raw head
    /// outputs pulled to the host: argmax pre/post ids, the full-stop
    /// probability, and the per-character upper-case probabilities.
    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<RawOutputs> {
        let (_, seq) = input_ids.dims2()?;
        let hidden = self.encode(input_ids)?; // (B, T, D)

        // Punctuation heads read the raw encoder output.
        let post_logits = self.punct_head_post.forward(&hidden)?; // (B,T,17)
        let pre_logits = self.punct_head_pre.forward(&hidden)?; // (B,T,2)
        let post_ids = post_logits.argmax(candle_core::D::Minus1)?; // (B,T) U32

        // Condition the seg head on the predicted post punctuation.
        let embs = self.punct_emb.forward(&post_ids)?; // (B,T,emb)
        let seg_input = Tensor::cat(&[&hidden, &embs], 2)?; // (B,T,D+emb)
        let seg_logits = self.seg_head.forward(&seg_input)?; // (B,T,2)
        let seg_ids = seg_logits.argmax(candle_core::D::Minus1)?; // (B,T) U32

        // Force the first slot to a boundary, then shift right by one to mark
        // the beginning of each sentence, and condition the cap head on it.
        let seg_shift = self.shift_boundaries(&seg_ids, seq)?; // (B,T,1)
        let cap_input = Tensor::cat(&[&hidden, &seg_shift], 2)?; // (B,T,D+1)
        let cap_logits = self.cap_head.forward(&cap_input)?; // (B,T,cap)

        // Final host-side outputs.
        let pre_ids = pre_logits.argmax(candle_core::D::Minus1)?;
        let seg_prob1 = softmax_last_dim(&seg_logits.to_dtype(DType::F32)?)?
            .narrow(2, 1, 1)?
            .squeeze(2)?; // (B,T) f32
        let cap_prob = sigmoid(&cap_logits.to_dtype(DType::F32)?)?; // (B,T,cap)

        Ok(RawOutputs {
            pre: pre_ids.to_vec2::<u32>()?,
            post: post_ids.to_vec2::<u32>()?,
            seg_prob1: seg_prob1.to_vec2::<f32>()?,
            cap_prob: cap_prob.to_vec3::<f32>()?,
        })
    }

    /// `seg_ids` `(B, T)` → `(B, T, 1)` compute-dtype boundary feature: force
    /// column 0 to 1, then shift right by one (pad a 0 on the left, drop the
    /// last), as `ConditionedPCSDecoder` does.
    fn shift_boundaries(&self, seg_ids: &Tensor, seq: usize) -> Result<Tensor> {
        let (b, _) = seg_ids.dims2()?;
        let ones = Tensor::ones((b, 1), DType::U32, &self.device)?;
        let forced = if seq > 1 {
            Tensor::cat(&[&ones, &seg_ids.narrow(1, 1, seq - 1)?], 1)?
        } else {
            ones
        };
        let zeros = Tensor::zeros((b, 1), DType::U32, &self.device)?;
        let shifted = if seq > 1 {
            Tensor::cat(&[&zeros, &forced.narrow(1, 0, seq - 1)?], 1)?
        } else {
            zeros
        };
        shifted.to_dtype(self.dtype)?.unsqueeze(2)
    }
}
