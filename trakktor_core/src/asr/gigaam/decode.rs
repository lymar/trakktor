//! Greedy decoding of encoder outputs.
//!
//! CTC greedy decoding collapses the per-frame argmax into a token sequence,
//! recording the frame each token is emitted at; those frames give word-level
//! timestamps. (RNN-T greedy decoding is added on top.)

#[cfg(test)]
mod tests;

use super::tokenizer::{self, Tokenizer};

/// A recognized word and its time span in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// The result of decoding one chunk: the text, the emitted token ids, and the
/// encoder frame each token was emitted at.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub token_frames: Vec<usize>,
}

/// CTC greedy collapse over per-frame argmax `labels`: emit a token when it is
/// not the blank and differs from the previous frame's label, within the first
/// `length` frames. Returns the token ids and the frames they were emitted at.
pub fn ctc_greedy(
    labels: &[u32],
    length: usize,
    blank_id: u32,
) -> (Vec<u32>, Vec<usize>) {
    let end = length.min(labels.len());
    let mut ids = Vec::new();
    let mut frames = Vec::new();
    let mut prev: Option<u32> = None;
    for (t, &label) in labels.iter().take(end).enumerate() {
        if label != blank_id && Some(label) != prev {
            ids.push(label);
            frames.push(t);
        }
        prev = Some(label);
    }
    (ids, frames)
}

/// Decodes one chunk from its argmax labels: collapses, then joins to text.
pub fn decode_chunk(
    tokenizer: &Tokenizer,
    labels: &[u32],
    length: usize,
) -> Decoded {
    let (token_ids, token_frames) =
        ctc_greedy(labels, length, tokenizer.blank_id());
    let text = tokenizer.decode(&token_ids);
    Decoded {
        text,
        token_ids,
        token_frames,
    }
}

/// Seconds per encoder frame for a chunk of `n_samples` audio that produced
/// `enc_len` encoder frames (the reference `compute_frame_shift`).
pub fn frame_shift(
    n_samples: usize,
    enc_len: usize,
    sample_rate: usize,
) -> f64 {
    n_samples as f64 / sample_rate as f64 / enc_len as f64
}

/// Groups tokens into words, assigning each word the time span of its tokens'
/// frames. A word boundary is a space (character-wise) or a `▁`-prefixed piece
/// (SentencePiece), matching the reference `frames_to_words`.
pub fn frames_to_words(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    token_frames: &[usize],
    frame_shift: f64,
) -> Vec<Word> {
    let mut words = Vec::new();
    let mut chars: Vec<String> = Vec::new();
    let mut frames: Vec<usize> = Vec::new();

    let mut commit = |chars: &mut Vec<String>, frames: &mut Vec<usize>| {
        if chars.is_empty() {
            return;
        }
        let text = chars.concat();
        let text = text.trim().to_string();
        if !text.is_empty() {
            let start = frames[0] as f64 * frame_shift;
            let end = (frames[frames.len() - 1] + 1) as f64 * frame_shift;
            words.push(Word { text, start, end });
        }
        chars.clear();
        frames.clear();
    };

    for (&id, &frame) in token_ids.iter().zip(token_frames) {
        let piece = tokenizer.id_to_str(id);
        if let Some(rest) = piece.strip_prefix(tokenizer::SPACE_MARKER) {
            // A `▁` marks the start of a new word (SentencePiece).
            commit(&mut chars, &mut frames);
            if !rest.is_empty() {
                chars.push(rest.to_string());
                frames.push(frame);
            }
        } else if piece == " " {
            // An explicit space separates words (character-wise).
            commit(&mut chars, &mut frames);
        } else {
            chars.push(piece.to_string());
            frames.push(frame);
        }
    }
    commit(&mut chars, &mut frames);
    words
}
