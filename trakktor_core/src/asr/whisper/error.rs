//! Typed errors for the Whisper engine.
//!
//! Each variant corresponds to a stable error `code` resolved at the CLI
//! boundary; the `#[error]` messages are human-facing and may change without
//! affecting the external contract.

/// Errors returned by the Whisper engine.
#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    /// The audio could not be decoded to PCM (`audio_decode_failed`).
    ///
    /// Covers a failure to launch the external decoder, a non-zero exit, or a
    /// truncated PCM stream.
    #[error("failed to decode audio: {0}")]
    AudioDecode(String),
}
