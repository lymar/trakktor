//! Token-to-text mapping.
//!
//! GigaAM models decode token ids to text one of two ways:
//!
//! - **character-wise** (`v3_ctc`, `multilingual_*_ctc`): the vocabulary is a
//!   list of single characters (plus space), and decoding is a plain join.
//!   These models emit lowercase text without punctuation.
//! - **SentencePiece** (`v3_e2e_ctc` and other end-to-end models): the
//!   vocabulary is a list of subword pieces, and decoding follows the
//!   SentencePiece detokenization rules — pieces are concatenated, the `▁`
//!   word-boundary marker becomes a space, leading spaces are dropped, and the
//!   unknown piece expands to a surface string. These models emit cased text
//!   with punctuation.

#[cfg(test)]
mod tests;

/// The SentencePiece word-boundary marker (U+2581, "▁").
pub const SPACE_MARKER: char = '\u{2581}';

/// The surface string an unknown SentencePiece token decodes to (the library
/// default `unk_surface`, " ⁇ ", U+2047 flanked by spaces).
const UNK_SURFACE: &str = " \u{2047} ";

/// A token-to-text decoder: either character-wise or SentencePiece.
#[derive(Debug, Clone)]
pub enum Tokenizer {
    /// Character-wise: token id → its single-character string.
    Charwise { vocab: Vec<String> },
    /// SentencePiece: token id → subword piece, decoded with the space-marker
    /// rules. `unk_id` is the id of the unknown piece.
    SentencePiece { pieces: Vec<String>, unk_id: u32 },
}

impl Tokenizer {
    /// Builds a character-wise tokenizer.
    pub fn charwise(vocab: Vec<String>) -> Self { Self::Charwise { vocab } }

    /// Builds a SentencePiece tokenizer.
    pub fn sentencepiece(pieces: Vec<String>, unk_id: u32) -> Self {
        Self::SentencePiece { pieces, unk_id }
    }

    /// Number of tokens (excluding the CTC blank).
    pub fn len(&self) -> usize {
        match self {
            Tokenizer::Charwise { vocab } => vocab.len(),
            Tokenizer::SentencePiece { pieces, .. } => pieces.len(),
        }
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// The CTC blank id, one past the last vocabulary entry.
    pub fn blank_id(&self) -> u32 { self.len() as u32 }

    /// The raw string of a single token id — the vocabulary entry (character or
    /// SentencePiece piece, `▁` marker included). Word-splitting for timestamps
    /// consumes this; final text goes through [`decode`](Self::decode).
    pub fn id_to_str(&self, id: u32) -> &str {
        match self {
            Tokenizer::Charwise { vocab } => &vocab[id as usize],
            Tokenizer::SentencePiece { pieces, .. } => &pieces[id as usize],
        }
    }

    /// Decodes token ids to text.
    ///
    /// Character-wise: a plain join. SentencePiece: pieces are concatenated
    /// with `▁` rendered as a space, leading spaces (word-start markers at the
    /// very beginning) dropped, and the unknown piece expanded to its surface
    /// string. Reproduces `SentencePieceProcessor.decode`.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::Charwise { vocab } => {
                let mut text = String::new();
                for &id in ids {
                    text.push_str(&vocab[id as usize]);
                }
                text
            },
            Tokenizer::SentencePiece { pieces, unk_id } => {
                let mut text = String::new();
                for &id in ids {
                    if id == *unk_id {
                        text.push_str(UNK_SURFACE);
                        continue;
                    }
                    for ch in pieces[id as usize].chars() {
                        if ch == SPACE_MARKER {
                            // A space marker at the very start of the output is
                            // a leading word boundary and is dropped; elsewhere
                            // it renders as a space.
                            if !text.is_empty() {
                                text.push(' ');
                            }
                        } else {
                            text.push(ch);
                        }
                    }
                }
                text
            },
        }
    }
}
