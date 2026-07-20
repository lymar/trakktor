//! Token-to-text mapping from the model's `tokens.txt`.
//!
//! Every model of the line ships a SentencePiece BPE vocabulary as a plain
//! `tokens.txt`: one `<piece> <id>` pair per line, `▁` marking a word start.
//! Ids `0`..`2` are the specials `<blk>` (the transducer blank), `<sos/eos>`,
//! and `<unk>`; the table may also carry disambiguation symbols (`#0`, …)
//! past the acoustic vocabulary, which decoding never emits.
//!
//! Decoding follows the reference runtime: emitted pieces are concatenated
//! with `▁` rendered as a space and a leading space dropped. The reference
//! never emits the blank or `<unk>` (the latter is folded into the blank), so
//! no surface form for them is needed.

#[cfg(test)]
mod tests;

use super::error::VoskError;

/// The SentencePiece word-boundary marker (U+2581, "▁").
pub const SPACE_MARKER: char = '\u{2581}';

/// A token table parsed from a model's `tokens.txt`.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Piece strings indexed by token id.
    pieces: Vec<String>,
    /// Id of the `<unk>` piece, if the table has one.
    unk_id: Option<u32>,
}

impl Tokenizer {
    /// Parses `tokens.txt` content: one `<piece> <id>` per line, the piece
    /// and id separated by a single space. Ids must be dense from 0 (the
    /// blank) upward; lines are not required to be sorted.
    pub fn parse(text: &str) -> Result<Self, VoskError> {
        let bad = |line: &str| {
            VoskError::InvalidModel(format!("tokens.txt: bad line `{line}`"))
        };
        let mut entries: Vec<(u32, String)> = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            // The piece itself may be a space; split at the *last* space.
            let (piece, id) = line.rsplit_once(' ').ok_or_else(|| bad(line))?;
            let id: u32 = id.parse().map_err(|_| bad(line))?;
            entries.push((id, piece.to_string()));
        }
        entries.sort_by_key(|(id, _)| *id);
        let mut pieces = Vec::with_capacity(entries.len());
        for (id, piece) in entries {
            if id as usize != pieces.len() {
                return Err(VoskError::InvalidModel(format!(
                    "tokens.txt: ids are not dense at {id}"
                )));
            }
            pieces.push(piece);
        }
        if pieces.is_empty() {
            return Err(VoskError::InvalidModel("tokens.txt: empty".into()));
        }
        let unk_id = pieces.iter().position(|p| p == "<unk>").map(|i| i as u32);
        Ok(Self { pieces, unk_id })
    }

    /// Number of entries in the table (may exceed the model's vocabulary by
    /// trailing disambiguation symbols).
    pub fn len(&self) -> usize { self.pieces.len() }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool { self.pieces.is_empty() }

    /// Id of the `<unk>` piece, folded into the blank by decoding.
    pub fn unk_id(&self) -> Option<u32> { self.unk_id }

    /// The raw piece string of a token id, `▁` marker included.
    /// Word-splitting for timestamps consumes this; final text goes through
    /// [`decode`](Self::decode).
    pub fn id_to_str(&self, id: u32) -> &str { &self.pieces[id as usize] }

    /// Decodes emitted token ids to text: pieces are concatenated with `▁`
    /// rendered as a space, and a word-start marker at the very beginning of
    /// the output is dropped (the reference `Convert`).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            for ch in self.pieces[id as usize].chars() {
                if ch == SPACE_MARKER {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                } else {
                    text.push(ch);
                }
            }
        }
        text
    }
}
