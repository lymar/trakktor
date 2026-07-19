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
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
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

        let char_probs =
            segment::char_probs(chars.len(), &tokenized.offsets, &token_logits);
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
}
