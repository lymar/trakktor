//! Typed errors for the Vosk engine.
//!
//! Each variant corresponds to a stable error `code` resolved at the CLI
//! boundary; the `#[error]` messages are human-facing and may change without
//! affecting the external contract.

/// Errors returned by the Vosk engine.
#[derive(Debug, thiserror::Error)]
pub enum VoskError {
    /// The audio could not be decoded to PCM (`audio_decode_failed`).
    #[error("failed to decode audio: {0}")]
    AudioDecode(String),

    /// The model files are malformed or unsupported (`model_unavailable`).
    ///
    /// Covers an unknown model name, missing or corrupt model files, a
    /// geometry the pipeline cannot serve, and backend failures while loading
    /// or running the network.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching the model files failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// Options are inconsistent (`invalid_options`).
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// The voice-activity detection stage failed (`vad_failed`).
    ///
    /// Long-form transcription with an offline model segments audio along
    /// detected speech; this covers a failure to load or run the VAD model.
    #[error("voice-activity detection failed: {0}")]
    Vad(String),

    /// Writing a requested output file failed (`io_error`).
    #[error("failed to write output file: {0}")]
    Io(String),

    /// The home directory could not be determined (`no_home_dir`).
    #[error(
        "could not determine the home directory; set --model-dir or \
         TRAKKTOR_MODEL_DIR"
    )]
    HomeDirUnknown,
}

impl From<crate::download::DownloadError> for VoskError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}
