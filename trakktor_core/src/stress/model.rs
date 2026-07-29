//! The geometry of the two networks and the seam the runtimes plug into.
//!
//! Everything above this line is string work; everything below it is tensors.
//! The seam is deliberately narrow: a runtime scores words and picks homograph
//! variants, and the decision rules (the confidence threshold, the marking
//! itself) live once, above it.

use super::error::StressError;

/// The converted weights inside a model directory.
pub(super) const WEIGHTS_FILE: &str = "model.safetensors";

/// Geometry of the accentor: the n-gram embedding table and its two heads.
#[derive(Debug, Clone, Copy)]
pub struct AccentorConfig {
    /// Rows of the n-gram embedding table.
    pub ngrams: usize,
    /// Width of an n-gram embedding.
    pub dim: usize,
    /// The two hidden widths shared by both heads.
    pub hidden: [usize; 3],
    /// Classes of the stress head — how many vowels it can point at.
    pub stress_classes: usize,
    /// Classes of the `ё` head — zero means "no `ё`".
    pub yo_classes: usize,
}

/// Geometry of the homograph solver: a `rubert-tiny`-class encoder with a
/// binary head over the marker and word embeddings.
#[derive(Debug, Clone, Copy)]
pub struct HomographConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    /// Hidden width of the classification head.
    pub head_hidden: usize,
}

impl HomographConfig {
    /// Width of one attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Everything the runtimes need to know about the shapes they load.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub accentor: AccentorConfig,
    pub homograph: HomographConfig,
}

impl Config {
    /// The published `silero-ru` model. The weights are shape-checked against
    /// this at load, so a different checkpoint fails loudly instead of
    /// producing quiet nonsense.
    #[must_use]
    pub fn silero_ru() -> Self {
        Self {
            accentor: AccentorConfig {
                ngrams: 126_523,
                dim: 16,
                hidden: [64, 128, 64],
                stress_classes: 10,
                yo_classes: 7,
            },
            homograph: HomographConfig {
                vocab_size: 83_830,
                hidden_size: 312,
                num_attention_heads: 12,
                intermediate_size: 600,
                num_hidden_layers: 3,
                max_position_embeddings: 2048,
                type_vocab_size: 2,
                layer_norm_eps: 1e-12,
                head_hidden: 256,
            },
        }
    }
}

/// What the accentor predicts for one word: which vowel carries the stress and
/// which letter `е` is really `ё`, each with the confidence behind it.
#[derive(Debug, Clone, Copy, Default)]
pub struct WordScores {
    /// Index of the stressed vowel among the word's vowels.
    pub stress_id: usize,
    /// Its softmax probability.
    pub stress_prob: f32,
    /// One-based index of the letter `е` that is really `ё`; zero means none.
    pub yo_id: usize,
    /// Its softmax probability.
    pub yo_prob: f32,
}

/// One homograph to decide: the tokenized sentence with the word wrapped in
/// markers, and where those markers landed.
#[derive(Debug, Clone)]
pub struct HomographContext {
    /// Token ids, `[CLS] … [HOMO] word [/HOMO] … [SEP]`.
    pub ids: Vec<u32>,
    /// Position of the opening marker.
    pub start: usize,
    /// Position of the closing marker.
    pub end: usize,
}

/// The loaded networks, behind which candle and burn are interchangeable.
///
/// Both calls are batched, and both hand back host-side values: the runtimes
/// own the arithmetic, nothing else.
pub trait StressModel {
    /// Scores a batch of words, each given as the embedding rows its n-grams
    /// hit (repeats included — the rows are averaged).
    ///
    /// # Errors
    ///
    /// Returns [`StressError`] on a backend failure.
    fn accentor(
        &self,
        bags: &[Vec<u32>],
    ) -> Result<Vec<WordScores>, StressError>;

    /// Picks a variant (0 or 1) for each homograph in the batch.
    ///
    /// The contexts are padded to the longest one and run **without an
    /// attention mask**, exactly as the reference does — the padding is part of
    /// its arithmetic, so masking it away would change the answers.
    ///
    /// # Errors
    ///
    /// Returns [`StressError`] on a backend failure.
    fn homographs(
        &self,
        contexts: &[HomographContext],
        pad: u32,
    ) -> Result<Vec<usize>, StressError>;
}

/// Turns the accentor's two probability rows into a [`WordScores`] — shared by
/// both runtimes so the argmax rule lives once.
#[must_use]
pub(super) fn scores(stress: &[f32], yo: &[f32]) -> WordScores {
    let best = |row: &[f32]| {
        row.iter().enumerate().fold(
            (0usize, f32::MIN),
            |best, (index, value)| {
                if *value > best.1 {
                    (index, *value)
                } else {
                    best
                }
            },
        )
    };
    let (stress_id, stress_prob) = best(stress);
    let (yo_id, yo_prob) = best(yo);
    WordScores {
        stress_id,
        stress_prob,
        yo_id,
        yo_prob,
    }
}

/// Which spelling one homograph logit picks.
///
/// The reference rounds the sigmoid, and PyTorch rounds halves to even — so a
/// logit of exactly zero picks the first spelling, and the rule is simply
/// "positive picks the second".
#[must_use]
pub(super) fn pick(logit: f32) -> usize { usize::from(logit > 0.0) }

/// Softmax over one row, in f32 — the thresholds downstream are decided on
/// these numbers, so they are never computed in half precision.
#[must_use]
pub(super) fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::MIN, f32::max);
    let exponentials: Vec<f32> =
        row.iter().map(|value| (value - max).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}
