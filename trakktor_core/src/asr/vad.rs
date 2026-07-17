//! Voice-activity detection (VAD) with Silero-VAD.
//!
//! An engine-independent preprocessing stage for ASR: it finds the speech in a
//! 16 kHz mono signal and drops non-speech (silence, music, noise) before the
//! audio reaches a transcription engine. Removing non-speech is the direct
//! remedy for the failure mode Whisper is most prone to — hallucinating and
//! looping over long non-speech stretches — and it is faster and yields
//! cleaner segment boundaries.
//!
//! The stage is a faithful port of the original Silero-VAD v5: the [`model`]
//! reproduces the network op for op, and [`segment`] reproduces the canonical
//! `get_speech_timestamps` state machine. The model is tiny and runs on the
//! CPU. Two ways to feed the detected speech into an engine are provided; the
//! reusable half (building a dense speech buffer with a time map) lives in
//! [`collapse`], the other (feeding speech spans as clip ranges) is a plain
//! list of segments the caller passes on.
//!
//! The module depends only on candle and the standard library, with no ties to
//! any particular engine.

mod assets;
pub mod collapse;
mod error;
pub mod model;
pub mod segment;

pub use collapse::{Collapsed, TimeMapping};
pub use error::VadError;
pub use model::Vad;
pub use segment::{SpeechSegment, VadOptions, speech_timestamps};

/// Audio sample rate the model expects, in Hz. The engine input is always
/// 16 kHz mono, so the 8 kHz Silero path is out of scope.
pub(crate) const SAMPLE_RATE: usize = 16_000;

/// New audio samples consumed per VAD window (32 ms at 16 kHz). One speech
/// probability is produced per window.
pub(crate) const WINDOW_SIZE: usize = 512;

/// Samples of the previous window prepended as context to each window. The
/// first window's context is zeros.
pub(crate) const CONTEXT_SIZE: usize = 64;

/// Detects speech in 16 kHz mono f32 PCM, returning the speech segments in
/// seconds on the original timeline.
///
/// Loads the embedded Silero-VAD model on the CPU, computes the per-window
/// speech probabilities, and runs the speech-timestamp state machine.
///
/// # Errors
///
/// Returns [`VadError`] if the model cannot be loaded or run.
pub fn detect_speech(
    audio: &[f32],
    options: &VadOptions,
) -> Result<Vec<SpeechSegment>, VadError> {
    let vad = Vad::load()?;
    let probs = vad.probabilities(audio)?;
    Ok(speech_timestamps(&probs, audio.len(), options))
}
