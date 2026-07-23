//! A minimal reader for a SentencePiece `ModelProto`.
//!
//! The punctuation model ships its own `sp.model` (a SentencePiece **Unigram**
//! model). To build an equivalent tokenizer on the `tokenizers` crate without a
//! native SentencePiece dependency, we need three things out of the protobuf:
//! the piece/score vocabulary, the id of the unknown piece, and the
//! `precompiled_charsmap` that drives the `nmt_nfkc` normalization. This parses
//! exactly those fields and ignores the rest.
//!
//! `ModelProto` (relevant fields):
//! - field 1, repeated `SentencePiece` — `piece` (field 1, string), `score`
//!   (field 2, `float`), `type` (field 3, enum);
//! - field 3, `NormalizerSpec` — `precompiled_charsmap` (field 2, bytes).
//!
//! Wire types: 0 varint, 1 64-bit, 2 length-delimited, 5 32-bit.

use super::error::PunctuateError;

/// The SentencePiece piece type that marks the unknown token (`UNKNOWN = 2`).
const TYPE_UNKNOWN: u64 = 2;

/// The parts of a SentencePiece model the tokenizer needs.
pub(super) struct SpeModel {
    /// `(piece, log-probability score)` per vocabulary id, in id order.
    pub(super) pieces: Vec<(String, f64)>,
    /// The id of the unknown piece (the one with `type = UNKNOWN`).
    pub(super) unk_id: usize,
    /// The normalizer's precompiled character map (`nmt_nfkc`); empty if the
    /// model carries none.
    pub(super) precompiled_charsmap: Vec<u8>,
}

/// A byte cursor over a protobuf message.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }

    fn at_end(&self) -> bool { self.pos >= self.buf.len() }

    /// Reads a base-128 varint.
    fn varint(&mut self) -> Result<u64, PunctuateError> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self.buf.get(self.pos).ok_or_else(truncated)?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(malformed("varint too long"));
            }
        }
    }

    /// Reads a `(field number, wire type)` tag.
    fn tag(&mut self) -> Result<(u64, u64), PunctuateError> {
        let key = self.varint()?;
        Ok((key >> 3, key & 0x7))
    }

    /// Reads a length-delimited field's bytes (wire type 2).
    fn bytes(&mut self) -> Result<&'a [u8], PunctuateError> {
        let len = self.varint()? as usize;
        let end = self.pos.checked_add(len).ok_or_else(truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(slice)
    }

    /// Reads a 32-bit little-endian value (wire type 5).
    fn fixed32(&mut self) -> Result<u32, PunctuateError> {
        let end = self.pos + 4;
        let slice = self.buf.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    /// Skips a field of the given wire type.
    fn skip(&mut self, wire: u64) -> Result<(), PunctuateError> {
        match wire {
            0 => {
                self.varint()?;
            },
            1 => self.pos += 8,
            2 => {
                self.bytes()?;
            },
            5 => self.pos += 4,
            other => return Err(malformed(&format!("wire type {other}"))),
        }
        if self.pos > self.buf.len() {
            return Err(truncated());
        }
        Ok(())
    }
}

/// Parses the pieces, unknown id, and precompiled char map out of an
/// `sp.model`.
///
/// # Errors
///
/// Returns [`PunctuateError::Tokenizer`] if the protobuf is truncated or
/// malformed, or if it declares no unknown piece.
pub(super) fn parse(bytes: &[u8]) -> Result<SpeModel, PunctuateError> {
    let mut reader = Reader::new(bytes);
    let mut pieces = Vec::new();
    let mut unk_id = None;
    let mut precompiled_charsmap = Vec::new();

    while !reader.at_end() {
        let (field, wire) = reader.tag()?;
        match (field, wire) {
            // pieces (repeated message)
            (1, 2) => {
                let (piece, score, kind) = parse_piece(reader.bytes()?)?;
                if kind == TYPE_UNKNOWN && unk_id.is_none() {
                    unk_id = Some(pieces.len());
                }
                pieces.push((piece, score));
            },
            // normalizer_spec (message)
            (3, 2) => {
                precompiled_charsmap = parse_normalizer(reader.bytes()?)?;
            },
            _ => reader.skip(wire)?,
        }
    }

    let unk_id = unk_id
        .ok_or_else(|| malformed("no unknown piece (UNKNOWN type) found"))?;
    if pieces.is_empty() {
        return Err(malformed("empty vocabulary"));
    }
    Ok(SpeModel {
        pieces,
        unk_id,
        precompiled_charsmap,
    })
}

/// Parses one `SentencePiece` submessage into `(piece, score, type)`.
fn parse_piece(bytes: &[u8]) -> Result<(String, f64, u64), PunctuateError> {
    let mut reader = Reader::new(bytes);
    let mut piece = None;
    let mut score = 0f32;
    let mut kind = 1u64; // NORMAL
    while !reader.at_end() {
        let (field, wire) = reader.tag()?;
        match (field, wire) {
            (1, 2) => {
                let raw = reader.bytes()?;
                piece = Some(
                    std::str::from_utf8(raw)
                        .map_err(|_| malformed("piece is not UTF-8"))?
                        .to_string(),
                );
            },
            (2, 5) => score = f32::from_bits(reader.fixed32()?),
            (3, 0) => kind = reader.varint()?,
            _ => reader.skip(wire)?,
        }
    }
    let piece = piece.ok_or_else(|| malformed("piece without a string"))?;
    Ok((piece, f64::from(score), kind))
}

/// Parses a `NormalizerSpec` submessage, returning its `precompiled_charsmap`.
fn parse_normalizer(bytes: &[u8]) -> Result<Vec<u8>, PunctuateError> {
    let mut reader = Reader::new(bytes);
    let mut charsmap = Vec::new();
    while !reader.at_end() {
        let (field, wire) = reader.tag()?;
        match (field, wire) {
            (2, 2) => charsmap = reader.bytes()?.to_vec(),
            _ => reader.skip(wire)?,
        }
    }
    Ok(charsmap)
}

fn truncated() -> PunctuateError {
    PunctuateError::Tokenizer("sp.model: truncated protobuf".into())
}

fn malformed(detail: &str) -> PunctuateError {
    PunctuateError::Tokenizer(format!("sp.model: {detail}"))
}

#[cfg(test)]
mod tests;
