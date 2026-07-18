//! Typed errors for the GigaAM engine.
//!
//! Each variant corresponds to a stable error `code` resolved at the CLI
//! boundary; the `#[error]` messages are human-facing and may change without
//! affecting the external contract.

/// Errors returned by the GigaAM engine.
#[derive(Debug, thiserror::Error)]
pub enum GigaamError {
    /// The audio could not be decoded to PCM (`audio_decode_failed`).
    #[error("failed to decode audio: {0}")]
    AudioDecode(String),

    /// The model checkpoint is malformed or unsupported (`model_unavailable`).
    ///
    /// Covers a geometry the pipeline cannot serve, an unknown model name, and
    /// backend failures while loading or running the network.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching a model checkpoint failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// Decoding options are inconsistent (`invalid_options`).
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// The requested language is not supported by the model
    /// (`unsupported_language`).
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// The voice-activity detection stage failed (`vad_failed`).
    ///
    /// Long-form transcription segments audio along detected speech; this
    /// covers a failure to load or run the VAD model.
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
