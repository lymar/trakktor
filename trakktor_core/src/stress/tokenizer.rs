//! The word-piece tokenizer of the homograph solver, rebuilt on the
//! `tokenizers` crate.
//!
//! The reference ships a trimmed copy of Hugging Face's `BertTokenizer`: clean
//! the text, space out CJK characters, split on whitespace and on punctuation,
//! then WordPiece — with **no** lowercasing and **no** accent stripping. The
//! crate already implements exactly that pair (`BertNormalizer` +
//! `BertPreTokenizer`), so the tokenizer is assembled rather than rewritten.
//!
//! The reference protects its `[HOMO]` / `[/HOMO]` markers with `never_split`.
//! We do not need that machinery: the markers are always inserted surrounded by
//! spaces, so the context is encoded piece by piece and the marker ids are
//! spliced in — which is the same tokenization, by construction.

use std::path::Path;

use tokenizers::{
    Tokenizer, models::wordpiece::WordPiece, normalizers::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
};

use super::{error::StressError, tables::VOCAB_FILE};

/// The special tokens the solver frames its input with.
const CLS: &str = "[CLS]";
const SEP: &str = "[SEP]";
const PAD: &str = "[PAD]";
const HOMO_START: &str = "[HOMO]";
const HOMO_END: &str = "[/HOMO]";

/// The loaded tokenizer plus the ids the solver needs by name.
pub(super) struct WordPieces {
    inner: Tokenizer,
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    pub homo_start: u32,
    pub homo_end: u32,
}

impl WordPieces {
    /// Builds the tokenizer from a model directory's `vocab.txt`.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::Tokenizer`] if the vocabulary cannot be read or
    /// is missing a special token.
    pub fn load(dir: &Path) -> Result<Self, StressError> {
        let path = dir.join(VOCAB_FILE);
        let path = path.to_str().ok_or_else(|| {
            StressError::Tokenizer(format!(
                "{} is not valid UTF-8",
                path.display()
            ))
        })?;
        let model = WordPiece::from_file(path).build().map_err(|e| {
            StressError::Tokenizer(format!("reading {path}: {e}"))
        })?;
        let mut inner = Tokenizer::new(model);
        inner.with_normalizer(Some(BertNormalizer::new(
            true,        // clean the text
            true,        // space out CJK characters
            Some(false), // keep accents
            false,       // keep case
        )));
        inner.with_pre_tokenizer(Some(BertPreTokenizer));

        let id = |token: &str| -> Result<u32, StressError> {
            inner.token_to_id(token).ok_or_else(|| {
                StressError::Tokenizer(format!(
                    "the vocabulary is missing `{token}`"
                ))
            })
        };
        Ok(Self {
            cls: id(CLS)?,
            sep: id(SEP)?,
            pad: id(PAD)?,
            homo_start: id(HOMO_START)?,
            homo_end: id(HOMO_END)?,
            inner,
        })
    }

    /// Tokenizes one piece of text; special tokens are spliced in by the
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::Tokenizer`] on a tokenizer failure.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, StressError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self.inner.encode(text, false).map_err(|e| {
            StressError::Tokenizer(format!("encoding text: {e}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}
