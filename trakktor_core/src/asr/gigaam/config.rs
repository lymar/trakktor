//! Per-model configuration, embedded in the binary.
//!
//! Each supported checkpoint has a small JSON config (geometry, vocabulary, and
//! download coordinates) under `assets/`, generated from the reference model.
//! The runtime reads the config for the requested model, then loads the
//! matching `.ckpt` weights.

use super::{error::GigaamError, feature::MelConfig, tokenizer::Tokenizer};

/// Subsampling front-end kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 1-D strided convolutions over the feature axis.
    Conv1d,
    /// 2-D strided convolutions over the (time, feature) plane, then a linear.
    Conv2d,
}

/// Convolution-module normalization kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvNorm {
    /// `BatchNorm1d` over channels.
    BatchNorm,
    /// `LayerNorm` over channels.
    LayerNorm,
}

/// Self-attention position-encoding kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    /// Rotary position embeddings applied before the q/k/v projections.
    Rotary,
    /// Transformer-XL relative-position attention.
    RelPos,
}

/// Decoding head kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelClass {
    /// CTC head with greedy collapse decoding.
    Ctc,
    /// RNN-Transducer head.
    Rnnt,
}

/// RNN-T head geometry (prediction and joint networks).
#[derive(Debug, Clone, Copy)]
pub struct RnntConfig {
    /// Width of the prediction network (embedding and LSTM hidden size).
    pub pred_hidden: usize,
    /// Number of LSTM layers in the prediction network.
    pub pred_rnn_layers: usize,
    /// Width of the joint network's hidden layer.
    pub joint_hidden: usize,
}

/// Conformer encoder geometry.
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub n_mels: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub subsampling: Subsampling,
    pub subs_kernel_size: usize,
    pub subsampling_factor: usize,
    pub conv_kernel_size: usize,
    pub conv_norm: ConvNorm,
    pub attention: Attention,
}

impl EncoderConfig {
    /// Head dimension, `d_model / n_heads`.
    pub fn d_head(&self) -> usize { self.d_model / self.n_heads }
}

/// Where to fetch a checkpoint and how to verify it.
#[derive(Debug, Clone)]
pub struct DownloadSpec {
    pub ckpt: String,
    pub url: String,
    pub md5: String,
}

/// The token-to-text mapping embedded in a model config.
#[derive(Debug, Clone)]
pub enum TokenizerConfig {
    /// Character-wise vocabulary.
    Charwise { vocab: Vec<String> },
    /// SentencePiece pieces with the unknown-token id.
    SentencePiece { pieces: Vec<String>, unk_id: u32 },
}

impl TokenizerConfig {
    /// Builds the runtime tokenizer.
    pub fn build(&self) -> Tokenizer {
        match self {
            TokenizerConfig::Charwise { vocab } => {
                Tokenizer::charwise(vocab.clone())
            },
            TokenizerConfig::SentencePiece { pieces, unk_id } => {
                Tokenizer::sentencepiece(pieces.clone(), *unk_id)
            },
        }
    }
}

/// Everything the runtime needs for one model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_class: ModelClass,
    pub mel: MelConfig,
    pub encoder: EncoderConfig,
    /// RNN-T head geometry; present exactly for RNN-T models.
    pub rnnt: Option<RnntConfig>,
    pub tokenizer: TokenizerConfig,
    pub blank_id: u32,
    pub num_classes: usize,
    pub download: DownloadSpec,
}

/// The published model names this build supports.
pub const KNOWN_MODELS: &[&str] = &[
    "v3_ctc",
    "v3_rnnt",
    "v3_e2e_ctc",
    "v3_e2e_rnnt",
    "multilingual_ctc",
    "multilingual_large_ctc",
];

/// Returns the embedded config for a known model name.
pub fn config_for(name: &str) -> Option<ModelConfig> {
    let raw = match name {
        "v3_ctc" => include_str!("assets/v3_ctc.json"),
        "v3_rnnt" => include_str!("assets/v3_rnnt.json"),
        "v3_e2e_ctc" => include_str!("assets/v3_e2e_ctc.json"),
        "v3_e2e_rnnt" => include_str!("assets/v3_e2e_rnnt.json"),
        "multilingual_ctc" => include_str!("assets/multilingual_ctc.json"),
        "multilingual_large_ctc" => {
            include_str!("assets/multilingual_large_ctc.json")
        },
        _ => return None,
    };
    Some(parse(raw).expect("embedded config is valid"))
}

// NB: every entry in `KNOWN_MODELS` must have a matching `assets/<name>.json`.

/// Parses a model config from its JSON.
fn parse(raw: &str) -> Result<ModelConfig, GigaamError> {
    let v: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| GigaamError::InvalidModel(format!("config: {e}")))?;
    let err = |m: &str| GigaamError::InvalidModel(format!("config: {m}"));

    let u = |val: &serde_json::Value, k: &str| -> Result<usize, GigaamError> {
        val[k]
            .as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| err(&format!("missing `{k}`")))
    };
    let s = |val: &serde_json::Value, k: &str| -> Result<String, GigaamError> {
        val[k]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| err(&format!("missing `{k}`")))
    };

    let mel_v = &v["mel"];
    let mel = MelConfig {
        n_fft: u(mel_v, "n_fft")?,
        hop_length: u(mel_v, "hop_length")?,
        n_mels: u(mel_v, "n_mels")?,
        center: mel_v["center"].as_bool().ok_or_else(|| err("mel.center"))?,
    };

    let enc_v = &v["encoder"];
    let subsampling = match s(enc_v, "subsampling")?.as_str() {
        "conv1d" => Subsampling::Conv1d,
        "conv2d" => Subsampling::Conv2d,
        other => return Err(err(&format!("subsampling `{other}`"))),
    };
    let conv_norm = match s(enc_v, "conv_norm")?.as_str() {
        "layer_norm" => ConvNorm::LayerNorm,
        "batch_norm" => ConvNorm::BatchNorm,
        other => return Err(err(&format!("conv_norm `{other}`"))),
    };
    let attention = match s(enc_v, "attention")?.as_str() {
        "rotary" => Attention::Rotary,
        "rel_pos" => Attention::RelPos,
        other => return Err(err(&format!("attention `{other}`"))),
    };
    let encoder = EncoderConfig {
        n_mels: mel.n_mels,
        d_model: u(enc_v, "d_model")?,
        n_layers: u(enc_v, "n_layers")?,
        n_heads: u(enc_v, "n_heads")?,
        subsampling,
        subs_kernel_size: u(enc_v, "subs_kernel_size")?,
        subsampling_factor: u(enc_v, "subsampling_factor")?,
        conv_kernel_size: u(enc_v, "conv_kernel_size")?,
        conv_norm,
        attention,
    };

    let model_class = match s(&v, "model_class")?.as_str() {
        "ctc" => ModelClass::Ctc,
        "rnnt" => ModelClass::Rnnt,
        other => return Err(err(&format!("model_class `{other}`"))),
    };
    let rnnt = match model_class {
        ModelClass::Ctc => None,
        ModelClass::Rnnt => {
            let rnnt_v = &v["rnnt"];
            Some(RnntConfig {
                pred_hidden: u(rnnt_v, "pred_hidden")?,
                pred_rnn_layers: u(rnnt_v, "pred_rnn_layers")?,
                joint_hidden: u(rnnt_v, "joint_hidden")?,
            })
        },
    };

    let strings = |val: &serde_json::Value,
                   k: &str|
     -> Result<Vec<String>, GigaamError> {
        val[k].as_array().ok_or_else(|| err(k)).map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap_or_default().to_string())
                .collect()
        })
    };
    let tok_v = &v["tokenizer"];
    let tokenizer = match s(tok_v, "kind")?.as_str() {
        "charwise" => TokenizerConfig::Charwise {
            vocab: strings(tok_v, "vocab")?,
        },
        "sentencepiece" => TokenizerConfig::SentencePiece {
            pieces: strings(tok_v, "pieces")?,
            unk_id: u(tok_v, "unk_id")? as u32,
        },
        other => return Err(err(&format!("tokenizer.kind `{other}`"))),
    };

    let dl_v = &v["download"];
    let download = DownloadSpec {
        ckpt: s(dl_v, "ckpt")?,
        url: s(dl_v, "url")?,
        md5: s(dl_v, "md5")?,
    };

    Ok(ModelConfig {
        model_name: s(&v, "model_name")?,
        model_class,
        mel,
        encoder,
        rnnt,
        tokenizer,
        blank_id: u(&v, "blank_id")? as u32,
        num_classes: u(&v, "num_classes")?,
        download,
    })
}
