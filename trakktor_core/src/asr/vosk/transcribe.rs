//! Per-chunk transcription, word building, and the result types.
//!
//! For an offline model one chunk is one full-context encoder pass with a
//! single device synchronization at its end (the read-back of the projected
//! encoder output), followed by the CPU transducer search. The orchestration
//! over a whole recording — long-form segmentation for offline models, the
//! chunked session for streaming ones — lives in [`stream`](super::stream).

use super::{
    constants::ENCODER_FRAME_S,
    decode::{DecodeState, Decoding, Emissions, TransducerHead},
    error::VoskError,
    feature::FbankExtractor,
    runtime::EncoderSeam,
    tokenizer::{SPACE_MARKER, Tokenizer},
};

/// A recognized word and its time span in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// One transcribed segment on the original timeline.
#[derive(Debug, Clone)]
pub struct Segment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<Word>,
}

/// A complete transcription.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub segments: Vec<Segment>,
    pub duration: f64,
}

/// Transcription options.
#[derive(Debug, Clone, Copy)]
pub struct TranscribeOptions {
    /// Compute per-word timestamps from the emission frames.
    pub word_timestamps: bool,
    /// The transducer search.
    pub decoding: Decoding,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            word_timestamps: false,
            decoding: Decoding::Beam {
                max_active: super::decode::DEFAULT_MAX_ACTIVE,
            },
        }
    }
}

/// Progress of a transcription run: audio seconds processed, and the total
/// when it is known up front (a stream may not know its length).
#[derive(Debug, Clone, Copy)]
pub struct TranscribeProgress {
    pub processed_seconds: f64,
    pub total_seconds: Option<f64>,
}

/// The per-chunk text and (optional) word timings.
pub(crate) struct ChunkResult {
    pub(crate) text: String,
    pub(crate) words: Vec<Word>,
}

/// Runs one offline chunk through feature extraction, the encoder, and the
/// transducer search.
pub(crate) fn transcribe_chunk(
    model: &dyn EncoderSeam,
    head: &TransducerHead,
    tokenizer: &Tokenizer,
    fbank: &FbankExtractor,
    chunk: &[f32],
    options: &TranscribeOptions,
) -> Result<ChunkResult, VoskError> {
    let features = fbank.compute(chunk);
    let (encoded, frames) = model.encode(&features)?;
    let mut state = DecodeState::new(head, options.decoding);
    state.decode_block(head, &encoded, frames);
    let emitted = state.finish(head);
    Ok(emissions_to_chunk(tokenizer, &emitted, 0.0, options))
}

/// Renders emissions into text plus (optionally) words, shifting frame
/// times by `offset_s` onto the original timeline.
pub(crate) fn emissions_to_chunk(
    tokenizer: &Tokenizer,
    emitted: &Emissions,
    offset_s: f64,
    options: &TranscribeOptions,
) -> ChunkResult {
    let text = tokenizer.decode(&emitted.token_ids);
    let words = if options.word_timestamps {
        tokens_to_words(
            tokenizer,
            &emitted.token_ids,
            &emitted.token_frames,
            offset_s,
        )
    } else {
        Vec::new()
    };
    ChunkResult { text, words }
}

/// Groups emitted tokens into words along the `▁` word-start markers,
/// assigning each word the time span of its tokens' emission frames (each
/// frame is [`ENCODER_FRAME_S`] long), shifted by `offset_s`.
pub(crate) fn tokens_to_words(
    tokenizer: &Tokenizer,
    ids: &[u32],
    frames: &[usize],
    offset_s: f64,
) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut chars: Vec<char> = Vec::new();
    let mut char_frames: Vec<usize> = Vec::new();

    let mut commit = |chars: &mut Vec<char>, frames: &mut Vec<usize>| {
        if chars.is_empty() {
            return;
        }
        let text: String = chars.drain(..).collect();
        let start = frames[0];
        let end = frames[frames.len() - 1];
        frames.clear();
        words.push(Word {
            text,
            start: offset_s + start as f64 * ENCODER_FRAME_S,
            end: offset_s + (end + 1) as f64 * ENCODER_FRAME_S,
        });
    };

    for (&id, &frame) in ids.iter().zip(frames) {
        let piece = tokenizer.id_to_str(id);
        if let Some(rest) = piece.strip_prefix(SPACE_MARKER) {
            commit(&mut chars, &mut char_frames);
            for ch in rest.chars() {
                chars.push(ch);
                char_frames.push(frame);
            }
        } else {
            for ch in piece.chars() {
                chars.push(ch);
                char_frames.push(frame);
            }
        }
    }
    commit(&mut chars, &mut char_frames);
    words
}
