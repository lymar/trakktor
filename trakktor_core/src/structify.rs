//! Text structuring into paragraphs (`text structify`).
//!
//! Turns an unstructured wall of text — typically an ASR transcript whose line
//! breaks fall on engine segments rather than meaning — into readable
//! paragraphs, offline. The neural layer is SaT (Segment any Text): an
//! XLM-RoBERTa encoder that scores each position for a boundary, ported to
//! candle (the runtime already used by Whisper and the VAD).
//!
//! The pipeline: normalize whitespace, tokenize, run the model over overlapping
//! windows, average the overlaps, map per-token boundary scores to characters,
//! and cut paragraphs where the score crosses a threshold. Boundaries are
//! character offsets into the normalized text, so a future mode can carry
//! per-word metadata (e.g. ASR timestamps) through the same segmentation.
//!
//! # Credits
//!
//! The models and the inference pipeline are ported from **SaT** ("Segment Any
//! Text: A Universal Approach for Robust, Efficient and Adaptable Sentence
//! Segmentation", Frohmann, Sterner, Vulić, Minixhofer, and Schedl, EMNLP 2024)
//! and its reference implementation **wtpsplit**, MIT-licensed (© 2024 Benjamin
//! Minixhofer, Markus Frohmann, Igor Sterner). The `sat-*` model weights are
//! published under the MIT license at
//! <https://huggingface.co/segment-any-text>. The XLM-RoBERTa network is
//! adapted from `candle-transformers` (Apache-2.0 OR MIT); tokenization uses
//! `FacebookAI/xlm-roberta-base` (MIT).

pub mod download;
mod error;
pub mod runtime;
#[cfg(feature = "structify-burn")]
pub mod runtime_burn;
pub mod segment;
mod tokenizer;

pub use download::{
    KNOWN_MODELS, ResolvedModel, resolve_model, resolve_tokenizer,
};
pub use error::StructifyError;
pub use runtime::{BoundaryModel, Precision, SatRuntime};
#[cfg(feature = "structify-burn")]
pub use runtime_burn::SatBurnRuntime;
pub use segment::Weighting;
pub use tokenizer::XlmrTokenizer;

/// Collapses every run of whitespace (including newlines) to a single space and
/// trims the ends — the canonical form segmentation runs on.
///
/// ASR line breaks mark engine segments, not paragraphs, and SaT treats
/// newlines as spaces anyway; normalizing gives one text whose character
/// offsets are stable for the paragraph ranges (and, later, word metadata).
#[must_use]
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The model a segmentation run uses unless told otherwise: the deepest
/// full-context base model, whose paragraph signal is the best calibrated of
/// the family (shallower ones need a lower threshold and cut less cleanly).
///
/// Every caller — the command and the other features that segment text — takes
/// its default from here rather than spelling it out again, so the choice
/// cannot drift between them.
pub const DEFAULT_MODEL: &str = "sat-12l-no-limited-lookahead";

/// Tunables for one segmentation run.
#[derive(Debug, Clone, Copy)]
pub struct StructifyOptions {
    /// Boundary probability above which a paragraph break is placed.
    pub threshold: f32,
    /// Window step in tokens; smaller overlaps more (slower, steadier).
    pub stride: usize,
    /// Windows per forward batch — the main GPU-utilization lever.
    pub batch_size: usize,
    /// Overlap-averaging weight profile.
    pub weighting: Weighting,
}

impl Default for StructifyOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            stride: 256,
            batch_size: 32,
            weighting: Weighting::Uniform,
        }
    }
}

/// A paragraph as a half-open character range `[start, end)` in the normalized
/// text, plus that (whitespace-trimmed) text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paragraph {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// The result of one run: the model used and the paragraphs found.
#[derive(Debug, Clone)]
pub struct StructifyOutput {
    pub model: String,
    pub paragraphs: Vec<Paragraph>,
}

/// A loaded structifier: a boundary model on either runtime plus the XLM-R
/// tokenizer.
pub struct Structifier {
    runtime: Box<dyn BoundaryModel>,
    tokenizer: XlmrTokenizer,
}

impl Structifier {
    /// Bundles a loaded runtime and tokenizer.
    #[must_use]
    pub fn new(
        runtime: Box<dyn BoundaryModel>,
        tokenizer: XlmrTokenizer,
    ) -> Self {
        Self { runtime, tokenizer }
    }

    /// The model's boundary probability for every character of
    /// already-[`normalize`]d text, before any threshold is applied.
    ///
    /// This is the whole cost of segmentation — one pass over the network.
    /// Everything downstream (where to cut, and at which threshold) is
    /// arithmetic over the returned vector, so a caller that tries several
    /// thresholds pays for the model exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError`] on a tokenizer or backend failure.
    pub fn boundary_probs(
        &self,
        text: &str,
        options: &StructifyOptions,
    ) -> Result<Vec<f32>, StructifyError> {
        let n_chars = text.chars().count();
        if n_chars == 0 {
            return Ok(Vec::new());
        }

        let tokenized = self.tokenizer.encode(text)?;
        let token_logits = self.runtime.token_boundary_logits(
            &tokenized.ids,
            self.tokenizer.cls_id,
            self.tokenizer.sep_id,
            options.stride,
            options.batch_size,
            options.weighting,
        )?;

        Ok(segment::char_probs(
            n_chars,
            &tokenized.offsets,
            &token_logits,
        ))
    }

    /// Segments already-[`normalize`]d text into paragraphs.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError`] on a tokenizer or backend failure.
    pub fn paragraphs(
        &self,
        text: &str,
        options: &StructifyOptions,
    ) -> Result<Vec<Paragraph>, StructifyError> {
        if !(0.0..=1.0).contains(&options.threshold) {
            // The boundary scores are probabilities; a threshold outside
            // their range silently means "never cut" or "cut everywhere",
            // which is an input mistake, not a request.
            return Err(StructifyError::InvalidOptions(format!(
                "--threshold must be between 0 and 1, got {}",
                options.threshold
            )));
        }
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }

        let char_probs = self.boundary_probs(text, options)?;
        let spans =
            segment::paragraph_spans(&chars, &char_probs, options.threshold);

        Ok(spans
            .into_iter()
            .filter_map(|span| {
                let trimmed = segment::trim_span(&chars, span);
                (trimmed.end > trimmed.start).then(|| Paragraph {
                    start: trimmed.start,
                    end: trimmed.end,
                    text: chars[trimmed.start..trimmed.end].iter().collect(),
                })
            })
            .collect())
    }

    /// Splits already-[`normalize`]d text into pieces that each fit `budget`,
    /// cutting as few times as possible.
    ///
    /// `cost` measures a piece in whatever unit `budget` counts — characters,
    /// tokens of some other model, estimated seconds of speech. The caller owns
    /// that unit; segmentation only has to know when a piece is too big.
    ///
    /// Text that already fits comes back whole, without touching the network.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError`] on a tokenizer or backend failure.
    pub fn split_to_budget(
        &self,
        text: &str,
        budget: usize,
        cost: &dyn Fn(&str) -> usize,
        options: &StructifyOptions,
    ) -> Result<Vec<String>, StructifyError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if cost(text) <= budget {
            return Ok(vec![text.to_owned()]);
        }

        let chars: Vec<char> = text.chars().collect();
        let char_probs = self.boundary_probs(text, options)?;
        Ok(segment::split_to_budget(
            &chars,
            &char_probs,
            options.threshold,
            budget,
            cost,
        ))
    }
}
