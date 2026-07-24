//! The text tokenizer, rebuilt on the `tokenizers` crate.
//!
//! The checkpoints ship the byte-level BPE as raw `vocab.json` and `merges.txt`
//! rather than a packaged tokenizer, so the equivalent is assembled here: the
//! BPE model, the splitting pattern the reference pre-tokenizes with, and the
//! byte-level mapping that turns the pieces into bytes.
//!
//! Only the spoken text passes through it. The control tokens that frame the
//! prompt live outside the base vocabulary as added tokens, and the checkpoint
//! states their ids in its own config, so the prompt is assembled from those
//! ids directly instead of by tokenizing markup.

use std::path::Path;

use tokenizers::{
    SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

use super::error::Qwen3TtsError;

/// The pattern the reference splits text on before applying BPE. Keeping it
/// identical matters: a different split yields different tokens, and the model
/// was trained on these.
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The byte-level BPE over the spoken text.
pub struct TextTokenizer {
    inner: Tokenizer,
}

impl TextTokenizer {
    /// Builds the tokenizer from a checkpoint's `vocab.json` and `merges.txt`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::Tokenizer`] when either file is missing or
    /// cannot be assembled into a tokenizer.
    pub fn load(vocab: &Path, merges: &Path) -> Result<Self, Qwen3TtsError> {
        let fail = |what: &str, e: String| {
            Qwen3TtsError::Tokenizer(format!("{what}: {e}"))
        };

        let bpe = BPE::from_file(
            vocab.to_str().ok_or_else(|| {
                fail("vocab path", "not valid UTF-8".to_owned())
            })?,
            merges.to_str().ok_or_else(|| {
                fail("merges path", "not valid UTF-8".to_owned())
            })?,
        )
        .build()
        .map_err(|e| fail("building the BPE model", e.to_string()))?;

        let mut inner = Tokenizer::new(bpe);
        let split = Split::new(
            SplitPattern::Regex(SPLIT_PATTERN.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| fail("building the split pattern", e.to_string()))?;
        // The split runs first; the byte-level stage then maps bytes to the
        // printable alphabet the vocabulary is written in. It must not apply
        // its own splitting on top, and must not prepend a space.
        inner.with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));

        Ok(Self { inner })
    }

    /// Tokenizes `text` into base-vocabulary ids.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::Tokenizer`] on a tokenizer failure.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Qwen3TtsError> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            Qwen3TtsError::Tokenizer(format!("encoding text: {e}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}
