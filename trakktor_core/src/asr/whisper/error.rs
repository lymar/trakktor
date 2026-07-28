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

    /// The requested language is not supported (`unsupported_language`).
    ///
    /// Raised for a language that is unknown altogether, and for one that is
    /// known but lies outside the language set of the selected model.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// The model checkpoint is malformed or unsupported (`model_unavailable`).
    ///
    /// Covers geometry the pipeline cannot serve (e.g. an unexpected mel-band
    /// count) and backend failures while loading or running the network.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Decoding options are inconsistent (`invalid_options`).
    ///
    /// For example, a beam size combined with best-of sampling, patience
    /// without a beam, or a length penalty outside `[0, 1]`.
    #[error("invalid decoding options: {0}")]
    InvalidOptions(String),

    /// Fetching a model checkpoint failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// Writing a requested output file failed (`io_error`).
    #[error("failed to write output file: {0}")]
    Io(String),

    /// The home directory could not be determined (`no_home_dir`).
    ///
    /// Raised when the default model directory (`~/.trakktor`) is needed but no
    /// home directory is known.
    #[error(
        "could not determine the home directory; set --model-dir or \
         TRAKKTOR_MODEL_DIR"
    )]
    HomeDirUnknown,
}

impl From<crate::download::DownloadError> for WhisperError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}
