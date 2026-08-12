//! Punctuation, true-casing, and sentence-boundary restoration
//! (`text punctuate`).
//!
//! Turns a raw ASR-style transcript — lowercase text without punctuation — into
//! readable text: it restores punctuation, capitalization (including acronyms
//! like `NATO` and `U.S.`), and splits the stream into sentences on predicted
//! full stops, offline. The neural layer is a multilingual XLM-RoBERTa encoder
//! with a cascade of four classification heads (punctuation before and after
//! each subtoken, per-character true-casing, sentence boundaries), ported to
//! candle (the runtime already used by Whisper and structify) and burn.
//!
//! The pipeline: normalize whitespace, tokenize with the model's own
//! SentencePiece model, run the network over overlapping windows, stitch the
//! overlaps, and rebuild the text from the subtoken pieces with the predicted
//! marks and casing applied.
//!
//! # Credits
//!
//! The model and pipeline are ported from
//! `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` (Apache-2.0), a
//! multilingual punctuation/true-casing/sentence-boundary model built on
//! `xlm-roberta-base`. The post-processing follows the author's `punctuators`
//! package (MIT). Tokenization rebuilds the model's SentencePiece vocabulary on
//! the `tokenizers` crate.

mod decode;
pub mod download;
mod error;
pub mod model;
pub mod runtime;
#[cfg(feature = "punctuate-burn")]
pub mod runtime_burn;
pub mod segment;
mod spe;
mod tokenizer;

pub use download::{ResolvedModel, known_model_names, resolve_model};
pub use error::PunctuateError;
pub use model::MAX_LENGTH;
pub use runtime::{Precision, PunctCapSegModel, PunctRuntime};
#[cfg(feature = "punctuate-burn")]
pub use runtime_burn::PunctBurnRuntime;
pub use segment::TokenPred;
pub use tokenizer::SpeTokenizer;

/// Content tokens per window, leaving room for the BOS and EOS the model frames
/// each window with.
const MAX_CONTENT: usize = MAX_LENGTH - 2;

/// Collapses every run of whitespace (including newlines) to a single space and
/// trims the ends — the canonical form the model runs on.
///
/// ASR line breaks mark engine segments, not sentences; normalizing gives one
/// text that matches SentencePiece's own `remove_extra_whitespaces`.
#[must_use]
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Tunables for one punctuation run.
#[derive(Debug, Clone, Copy)]
pub struct PunctuateOptions {
    /// Overlap between consecutive windows, in tokens (the seam is split in
    /// half between the two windows).
    pub overlap: usize,
    /// Windows per forward batch.
    pub batch_size: usize,
    /// Split the stream into sentences on predicted full stops.
    pub apply_sbd: bool,
}

impl Default for PunctuateOptions {
    fn default() -> Self {
        Self {
            overlap: 16,
            batch_size: 16,
            apply_sbd: true,
        }
    }
}

/// A loaded punctuator: a network on either runtime plus the SentencePiece
/// tokenizer.
pub struct Punctuator {
    runtime: Box<dyn PunctCapSegModel>,
    tokenizer: SpeTokenizer,
}

impl Punctuator {
    /// Bundles a loaded runtime and tokenizer.
    #[must_use]
    pub fn new(
        runtime: Box<dyn PunctCapSegModel>,
        tokenizer: SpeTokenizer,
    ) -> Self {
        Self { runtime, tokenizer }
    }

    /// Restores punctuation and casing on already-[`normalize`]d text and
    /// returns the sentences (a single element when `apply_sbd` is off).
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError`] on a tokenizer or backend failure.
    pub fn sentences(
        &self,
        text: &str,
        options: &PunctuateOptions,
    ) -> Result<Vec<String>, PunctuateError> {
        if options.overlap >= MAX_CONTENT {
            // An overlap that swallows the whole window would make the
            // windowing loop stand still: the next window starts where the
            // previous one did.
            return Err(PunctuateError::InvalidOptions(format!(
                "--overlap must be below {MAX_CONTENT} tokens (the window \
                 minus its two markers), got {}",
                options.overlap
            )));
        }
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self.tokenizer.encode(text)?;
        let merged = segment::windowed_predictions(
            &ids,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            MAX_CONTENT,
            options.overlap,
            options.batch_size,
            |batch| self.runtime.forward_batch(batch),
        )?;
        Ok(decode::reconstruct(
            &merged,
            &self.tokenizer,
            options.apply_sbd,
        ))
    }
}
