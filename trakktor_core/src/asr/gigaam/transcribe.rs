//! Per-chunk transcription and the result types.
//!
//! One chunk is one encoder forward pass with a single device synchronization
//! at its end — the CTC path reads back per-frame argmax labels, the RNN-T
//! path reads back the encoder output and runs its sequential decode loop on
//! the CPU — so the device stays busy through a chunk. The orchestration over
//! a whole recording (short single-chunk audio and streamed long-form
//! segmentation) lives in [`stream`](super::stream).

use super::{
    constants::SAMPLE_RATE,
    decode::{Word, frames_to_words},
    error::GigaamError,
    runtime::AsrModel,
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

/// Runs one chunk through feature extraction, the encoder, the model's head,
/// and greedy decoding.
pub(crate) fn transcribe_chunk(
    model: &dyn AsrModel,
    tokenizer: &Tokenizer,
    chunk: &[f32],
    options: &TranscribeOptions,
) -> Result<ChunkResult, GigaamError> {
    let mel = model.feature().log_mel(chunk);
    let emitted = model.emissions(&mel)?;

    let text = tokenizer.decode(&emitted.token_ids);
    let words = if options.word_timestamps {
        let shift =
            chunk.len() as f64 / SAMPLE_RATE as f64 / emitted.enc_frames as f64;
        frames_to_words(
            tokenizer,
            &emitted.token_ids,
            &emitted.token_frames,
            shift,
        )
    } else {
        Vec::new()
    };
    Ok(ChunkResult { text, words })
}
