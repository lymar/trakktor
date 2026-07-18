//! Voice-activity detection (VAD) with Silero-VAD.
//!
//! Finds the speech in a 16 kHz mono signal and reports it as segments, so
//! non-speech (silence, music, noise) can be dropped. It is a shared building
//! block used two ways: as an ASR preprocessing stage (removing non-speech is
//! the direct remedy for the failure mode Whisper is most prone to —
//! hallucinating and looping over long non-speech stretches — and it is faster
//! and yields cleaner segment boundaries), and as the basis of standalone audio
//! editing (cutting silence, splitting into clips, reporting a speech
//! timeline).
//!
//! The detector is a faithful port of the original Silero-VAD v5: the [`model`]
//! reproduces the network op for op, and [`segment`] reproduces the canonical
//! `get_speech_timestamps` state machine. The model is tiny and runs on the
//! CPU. For the ASR use, detected speech is fed into an engine by collapsing it
//! into a dense speech buffer with a time map back to the original timeline;
//! that reusable half lives in [`collapse`].
//!
//! The module depends only on candle and the standard library, with no ties to
//! any particular engine or feature.

mod assets;
pub mod collapse;
pub mod edit;
mod error;
pub mod model;
pub mod segment;

pub use collapse::{Collapsed, TimeMapping};
pub use edit::{EditOptions, Keep};
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
