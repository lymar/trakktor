//! The XLM-RoBERTa tokenizer for SaT, via the Hugging Face `tokenizers` crate.
//!
//! SaT models ship no tokenizer of their own — all use
//! `facebookAI/xlm-roberta-base`. We load its `tokenizer.json` and tokenize
//! **without** special tokens (they are added per window later), keeping each
//! token's character span for the token→character boundary mapping.

use std::path::Path;

use tokenizers::Tokenizer;

use super::error::StructifyError;

/// A tokenized text: subword ids and each token's character span.
pub struct Tokenized {
    /// Subword token ids (no `CLS`/`SEP`).
    pub ids: Vec<u32>,
    /// Half-open character span `[c0, c1)` per token, in **code points** of
    /// the input text. The `tokenizers` crate reports offsets in
    /// **bytes**, so they are converted here to code-point indices —
    /// matching the reference, whose character-indexed boundary mapping is
    /// what the pipeline reproduces.
    pub offsets: Vec<(usize, usize)>,
}

/// The loaded XLM-R tokenizer plus the special-token ids used to frame windows.
pub struct XlmrTokenizer {
    inner: Tokenizer,
    /// `<s>` — prepended to every window.
    pub cls_id: u32,
    /// `</s>` — appended to every window.
    pub sep_id: u32,
    /// `<pad>` — padding id (unused while windows stay full, kept for
    /// clarity).
    pub pad_id: u32,
}

impl XlmrTokenizer {
    /// Loads `tokenizer.json` from `path`.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::Tokenizer`] if the file cannot be read or a
    /// required special token is missing.
    pub fn load(path: &Path) -> Result<Self, StructifyError> {
        let inner = Tokenizer::from_file(path).map_err(|e| {
            StructifyError::Tokenizer(format!(
                "loading {}: {e}",
                path.display()
            ))
        })?;
        let id = |token: &str| -> Result<u32, StructifyError> {
            inner.token_to_id(token).ok_or_else(|| {
                StructifyError::Tokenizer(format!(
                    "tokenizer is missing the `{token}` token"
                ))
            })
        };
        Ok(Self {
            cls_id: id("<s>")?,
            sep_id: id("</s>")?,
            pad_id: id("<pad>")?,
            inner,
        })
    }

    /// Tokenizes `text` without special tokens, returning ids and char spans.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::Tokenizer`] on a tokenizer failure.
    pub fn encode(&self, text: &str) -> Result<Tokenized, StructifyError> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            StructifyError::Tokenizer(format!("encoding text: {e}"))
        })?;
        let byte_to_char = byte_to_char_map(text);
        let offsets = encoding
            .get_offsets()
            .iter()
            .map(|&(b0, b1)| (byte_to_char[b0], byte_to_char[b1]))
            .collect();
        Ok(Tokenized {
            ids: encoding.get_ids().to_vec(),
            offsets,
        })
    }
}

/// A lookup from a byte index (`0..=text.len()`) to the number of code points
/// before it. Token offsets fall on character boundaries, but every byte is
/// filled (a continuation byte maps to its character's index) so the lookup is
/// total.
fn byte_to_char_map(text: &str) -> Vec<usize> {
    let mut map = vec![0usize; text.len() + 1];
    let mut char_index = 0usize;
    for (byte_index, ch) in text.char_indices() {
        for byte in byte_index..byte_index + ch.len_utf8() {
            map[byte] = char_index;
        }
        char_index += 1;
    }
    map[text.len()] = char_index;
    map
}
