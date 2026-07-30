//! Per-chunk transcription result types and rendering.
//!
//! One chunk is one encoder forward pass with a single device synchronization
//! at its end — the CTC path reads back per-frame argmax labels, the RNN-T
//! path reads back the encoder output for its sequential decode loop on the
//! CPU — so the device stays busy through a chunk. The orchestration over a
//! whole recording (short single-chunk audio, streamed long-form
//! segmentation, and the decode-behind-encode pipelining) lives in
//! [`stream`](super::stream).

use super::{
    constants::SAMPLE_RATE,
    decode::{Word, frames_to_words},
    runtime::Emissions,
    tokenizer::Tokenizer,
};

/// One transcribed segment (one chunk) on the original timeline.
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
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Compute per-word timestamps from the CTC frame indices.
    pub word_timestamps: bool,
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

/// Renders one chunk's decoded emissions into its text and word timings;
/// `samples` is the chunk's PCM length, the scale from encoder frames to
/// seconds.
///
/// Words are always resolved, whatever the caller asked for in the output: a
/// chunk hears past the span it owns, and their timings are what tells its own
/// words from the neighbour's context. They cost a walk over the emitted
/// tokens, the frames being a by-product of decoding either head.
pub(crate) fn chunk_result(
    tokenizer: &Tokenizer,
    emitted: &Emissions,
    samples: usize,
) -> ChunkResult {
    let text = tokenizer.decode(&emitted.token_ids);
    let shift = samples as f64 / SAMPLE_RATE as f64 / emitted.enc_frames as f64;
    let words = frames_to_words(
        tokenizer,
        &emitted.token_ids,
        &emitted.token_frames,
        shift,
    );
    ChunkResult { text, words }
}
