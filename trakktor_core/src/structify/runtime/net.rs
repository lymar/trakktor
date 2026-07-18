//! The SaT network on candle.
//!
//! SaT (`SubwordXLMForTokenClassification`) is a standard bidirectional
//! XLM-RoBERTa encoder with a per-token linear classifier on top. This is a
//! faithful port of that architecture, adapted from the candle project's
//! XLM-RoBERTa model (`candle-transformers`, Apache-2.0 OR MIT), reduced to the
//! encoder-only forward this feature needs:
//!
//! - no cross-attention, no incremental cache (one full pass per window);
//! - the token-classification head (`classifier`) applied to every position,
//!   from which position 0's logit is the boundary score
//!   (`Constants.NEWLINE_INDEX = 0` in the reference);
//! - **exact GELU** (`gelu_erf`), matching the reference `hidden_act = "gelu"`;
//! - the additive attention mask is built in the compute dtype with that
//!   dtype's most-negative finite value, so the f16 path never produces `-inf`.
//!
//! Weight names follow the layout of the published `sat-*` checkpoints
//! (`roberta.*` for the encoder, `classifier.*` for the head), shared by the
//! `-sm` (safetensors) and base (`pytorch_model.bin`) families.

use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm,
    linear, ops::softmax_last_dim,
};

/// The boundary label: position 0 of the classifier output (the reference's
/// `Constants.NEWLINE_INDEX`).
pub const NEWLINE_INDEX: usize = 0;

/// Geometry of an XLM-RoBERTa token-classification model.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: u32,
    pub layer_norm_eps: f64,
    pub num_labels: usize,
}

/// The additive attention mask for a `(batch, seq)` keep/drop mask: `0.0` at
/// kept positions, the dtype's most-negative finite value at dropped ones,
/// shaped `(batch, 1, 1, seq)` to broadcast over heads and query positions.
///
/// A finite floor (not `-inf`) keeps softmax well-defined even in f16.
fn additive_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let m = mask.to_dtype(dtype)?.unsqueeze(1)?.unsqueeze(1)?;
    let floor = match dtype {
        DType::F16 => -65504.0f64,
        DType::BF16 => -3.0e38f64,
        _ => f32::MIN as f64,
    };
    // (1 - m) is 1.0 at dropped positions, 0.0 at kept ones; scale to the
    // floor.
    (1.0 - m)?.affine(floor, 0.0)
}

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

    /// `input_ids`/`token_type_ids` are `(batch, seq)`. XLM-R positions start
    /// at `padding_idx + 1` and count only non-padding tokens (padding
    /// keeps `padding_idx`), exactly as the reference computes them.
    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids)?;

        // Position ids are computed in f32 (exact for integers up to the block
        // length) regardless of the compute dtype, then cast to indices.
        let mask = input_ids.ne(self.padding_idx)?.to_dtype(DType::F32)?;
        let position_ids = (mask.cumsum(1)? * &mask)?
            .affine(1.0, self.padding_idx as f64)?
            .to_dtype(DType::U32)?;
        let position_embeddings =
            self.position_embeddings.forward(&position_ids)?;

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
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    /// `(batch, seq, hidden)` → `(batch, heads, seq, head_dim)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        x.reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let q = self.split_heads(&self.query.forward(hidden)?)?;
        let k = self.split_heads(&self.key.forward(hidden)?)?;
        let v = self.split_heads(&self.value.forward(hidden)?)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores =
            (q.matmul(&k.transpose(2, 3)?)? * scale)?.broadcast_add(mask)?;
        let probs = softmax_last_dim(&scores)?;

        let (b, _, s, _) = probs.dims4()?;
        probs
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, s, self.num_heads * self.head_dim))
    }
}

/// A dense projection with a residual add and layer norm — the shape shared by
/// the attention output and the FFN output.
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

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attn = self.attention.forward(hidden, mask)?;
        let hidden = self.attention_output.forward(&attn, hidden)?;
        // Exact GELU (erf), matching the reference `hidden_act = "gelu"`.
        let intermediate = self.intermediate.forward(&hidden)?.gelu_erf()?;
        self.output.forward(&intermediate, &hidden)
    }
}

/// The loaded SaT network.
pub struct SatModel {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    classifier: Linear,
    dtype: DType,
}

impl SatModel {
    /// Loads the encoder (`roberta.*`) and the token-classification head
    /// (`classifier.*`) from `vb`.
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let roberta = vb.pp("roberta");
        let embeddings = Embeddings::load(roberta.pp("embeddings"), cfg)?;
        let encoder = roberta.pp("encoder");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(encoder.pp(format!("layer.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let classifier =
            linear(cfg.hidden_size, cfg.num_labels, vb.pp("classifier"))?;
        Ok(Self {
            embeddings,
            layers,
            classifier,
            dtype: vb.dtype(),
        })
    }

    /// Runs a batch of windows and returns the per-token **boundary logit**
    /// (label [`NEWLINE_INDEX`]) as an `(batch, seq)` f32 tensor.
    ///
    /// `input_ids` is `(batch, seq)` `U32`; `attention_mask` is `(batch, seq)`
    /// with `1.0` for real tokens and `0.0` for padding.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let token_type_ids = input_ids.zeros_like()?;
        let mut hidden = self.embeddings.forward(input_ids, &token_type_ids)?;
        let mask = additive_mask(attention_mask, self.dtype)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask)?;
        }
        let logits = self.classifier.forward(&hidden)?;
        logits
            .narrow(2, NEWLINE_INDEX, 1)?
            .squeeze(2)?
            .to_dtype(DType::F32)
    }
}
