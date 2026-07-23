//! Geometry, head dimensions, and label tables of the punctuation model.
//!
//! The first (and, for now, only) model is
//! `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`: a stock
//! `xlm-roberta-base` encoder with a cascade of four classification heads. The
//! values here are fixed properties of that published checkpoint; every weight
//! is shape-checked against them at load, so a mismatch surfaces as
//! `invalid_model` rather than a silent wrong result.

/// Maximum sequence length the model was trained for (including BOS/EOS).
pub const MAX_LENGTH: usize = 256;

/// The cap (true-casing) head predicts this many per-character labels per
/// subtoken; characters past it are left lowercase (`max_subword_length`).
pub const MAX_SUBWORD_LEN: usize = 16;

/// Sentence-boundary output threshold on `softmax(seg_logits)[FULLSTOP]`,
/// matching the exported graph (`> 0.05`). The **internal** boundary that
/// conditions the cap head uses `argmax` (i.e. `0.5`); both are reproduced.
pub const SEG_THRESHOLD: f32 = 0.05;

/// Post-punctuation labels (the class after a subtoken), in id order. Index 0
/// is [`NULL_LABEL`] ("no punctuation"); index 1 is [`ACRONYM_LABEL`] (a period
/// after every character, e.g. `am` → `a.m.`). The rest are literal marks; for
/// Latin/Cyrillic only `.`, `,`, `?` occur.
pub const POST_LABELS: [&str; 17] = [
    "<NULL>",
    "<ACRONYM>",
    ".",
    ",",
    "?",
    "？",
    "，",
    "。",
    "、",
    "・",
    "।",
    "؟",
    "،",
    ";",
    "።",
    "፣",
    "፧",
];

/// Pre-punctuation labels (the class before a subtoken), in id order. Index 0
/// is [`NULL_LABEL`]; index 1 is the Spanish inverted question mark.
pub const PRE_LABELS: [&str; 2] = ["<NULL>", "¿"];

/// The label meaning "predict nothing" (id 0 in both pre and post tables).
pub const NULL_LABEL: &str = "<NULL>";

/// The post label meaning "period after every character of this subtoken".
pub const ACRONYM_LABEL: &str = "<ACRONYM>";

/// Geometry of the XLM-RoBERTa encoder and the four heads. Fixed for the one
/// supported model; kept as a struct so future models can supply their own.
#[derive(Debug, Clone)]
pub struct Config {
    // Encoder (stock xlm-roberta-base).
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: u32,
    pub layer_norm_eps: f64,

    // Heads (`ConditionedPCSDecoder`).
    /// Post-punctuation classes (also the punctuation-embedding vocabulary).
    pub punct_post_classes: usize,
    /// Pre-punctuation classes.
    pub punct_pre_classes: usize,
    /// Per-character cap predictions per subtoken (`max_subword_length`).
    pub cap_classes: usize,
    /// Dimension of the punctuation embedding fed to the seg head.
    pub emb_dim: usize,
    /// Hidden width of the two punctuation heads.
    pub punct_head_intermediate: usize,
    /// Hidden width of the seg head.
    pub seg_head_intermediate: usize,
    /// Hidden width of the cap head.
    pub cap_head_intermediate: usize,
}

impl Config {
    /// The configuration of
    /// `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`.
    #[must_use]
    pub fn xlmr_47lang() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            vocab_size: 250002,
            max_position_embeddings: 514,
            type_vocab_size: 1,
            pad_token_id: 1,
            layer_norm_eps: 1e-5,
            punct_post_classes: 17,
            punct_pre_classes: 2,
            cap_classes: MAX_SUBWORD_LEN,
            emb_dim: 4,
            punct_head_intermediate: 256,
            seg_head_intermediate: 128,
            cap_head_intermediate: 128,
        }
    }

    /// Head-dimension of the self-attention (`hidden_size / heads`).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
