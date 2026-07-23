//! The SentencePiece tokenizer, rebuilt on the `tokenizers` crate.
//!
//! The punctuation model ships its own `sp.model` — a SentencePiece **Unigram**
//! model whose ids are the model's embedding rows directly (control tokens at
//! `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`), incompatible with the stock
//! XLM-R `tokenizer.json`. Rather than depend on a native SentencePiece
//! library, we parse the protobuf (`spe`) and assemble an equivalent
//! `tokenizers::Tokenizer`:
//!
//! - model — `Unigram(vocab, unk_id, byte_fallback = false)`;
//! - normalizer — `Precompiled(precompiled_charsmap)` (the `nmt_nfkc` map);
//! - pre-tokenizer — `Metaspace('▁', Always, split)` (SentencePiece's
//!   `add_dummy_prefix` + whitespace→`▁`).
//!
//! This reproduces `SentencePieceProcessor.EncodeAsIds` id-for-id (verified on
//! real transcripts). The output text is rebuilt from the piece strings (a
//! leading `▁` marks a word start), so no character offsets are kept.
//!
//! One known divergence: SentencePiece never matches its CONTROL pieces
//! (`<s>`, `<pad>`, `</s>`) against input text, while the rebuilt vocabulary
//! can — text containing such a literal string tokenizes to the control id
//! (as Hugging Face's own converted tokenizers also do). ASR transcripts
//! never contain these strings, and the model simply scores the odd token.

use std::path::Path;

use tokenizers::{
    Tokenizer,
    models::unigram::Unigram,
    normalizers::Precompiled,
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
};

use super::{error::PunctuateError, spe};

/// SentencePiece's word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
pub const WORD_PREFIX: char = '▁';

/// The special-token piece strings SentencePiece models use for framing.
const BOS_PIECE: &str = "<s>";
const EOS_PIECE: &str = "</s>";
const PAD_PIECE: &str = "<pad>";

/// The loaded tokenizer plus the special-token ids used to frame windows.
pub struct SpeTokenizer {
    inner: Tokenizer,
    /// The piece string for each id, indexed by id — the reconstruction reads
    /// these directly (equivalent to `IdToPiece`).
    pieces: Vec<String>,
    /// `<s>` — prepended to every window.
    pub bos_id: u32,
    /// `</s>` — appended to every window.
    pub eos_id: u32,
    /// `<pad>` — padding id (unused while windows are length-bucketed, kept
    /// for clarity).
    pub pad_id: u32,
}

impl SpeTokenizer {
    /// Builds the tokenizer from an `sp.model` file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::Tokenizer`] if the file cannot be read, the
    /// protobuf is malformed, or a required special token is missing.
    pub fn load(path: &Path) -> Result<Self, PunctuateError> {
        let raw = std::fs::read(path).map_err(|e| {
            PunctuateError::Tokenizer(format!(
                "reading {}: {e}",
                path.display()
            ))
        })?;
        let model = spe::parse(&raw)?;
        let pieces: Vec<String> = model
            .pieces
            .iter()
            .map(|(piece, _)| piece.clone())
            .collect();

        let unigram = Unigram::from(model.pieces, Some(model.unk_id), false)
            .map_err(|e| {
                PunctuateError::Tokenizer(format!("building unigram: {e}"))
            })?;
        let mut inner = Tokenizer::new(unigram);
        let precompiled = Precompiled::from(&model.precompiled_charsmap)
            .map_err(|e| {
                PunctuateError::Tokenizer(format!("normalizer: {e}"))
            })?;
        inner.with_normalizer(Some(precompiled));
        inner.with_pre_tokenizer(Some(Metaspace::new(
            WORD_PREFIX,
            PrependScheme::Always,
            true,
        )));

        let id = |piece: &str| -> Result<u32, PunctuateError> {
            inner.token_to_id(piece).ok_or_else(|| {
                PunctuateError::Tokenizer(format!(
                    "tokenizer is missing the `{piece}` token"
                ))
            })
        };
        Ok(Self {
            bos_id: id(BOS_PIECE)?,
            eos_id: id(EOS_PIECE)?,
            pad_id: id(PAD_PIECE)?,
            pieces,
            inner,
        })
    }

    /// Tokenizes `text` (no special tokens; BOS/EOS are added per window).
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::Tokenizer`] on a tokenizer failure.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, PunctuateError> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            PunctuateError::Tokenizer(format!("encoding text: {e}"))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The piece string for a token id (equivalent to `IdToPiece`).
    #[must_use]
    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map_or("", String::as_str)
    }
}
