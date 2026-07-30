//! Typed errors for `tts silero`.
//!
//! Each variant maps to a stable error `code` at the CLI boundary; the
//! `#[error]` messages are human-facing and may change without affecting the
//! external contract.

/// Errors returned by the Silero engine.
#[derive(Debug, thiserror::Error)]
pub enum SileroError {
    /// Nothing was left to speak (`text_empty`).
    ///
    /// Also raised when the text held no symbol of the model's alphabet — a
    /// line of Latin, say, which this frontend removes entirely.
    #[error("the text to speak is empty")]
    TextEmpty,

    /// A piece does not fit the model's window even after splitting
    /// (`text_too_long`).
    #[error(
        "one piece would need {frames} mel frames, past the model's {limit} \
         ({seconds:.0} s); shorten the paragraph"
    )]
    TextTooLong {
        frames: usize,
        limit: usize,
        seconds: f64,
    },

    /// The requested model is published under a licence that is not enabled
    /// (`model_license_restricted`).
    #[error(
        "model `{model}` is published under {license}; pass \
         --allow-non-commercial-models to use it, or choose one of: {allowed}"
    )]
    LicenseRestricted {
        model: String,
        license: &'static str,
        allowed: String,
    },

    /// The model name is unknown, or the directory is not a converted model
    /// (`model_unavailable`).
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching or converting the model failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// The checkpoint does not hold what the port expects
    /// (`model_unavailable`).
    #[error("invalid checkpoint: {0}")]
    Checkpoint(String),

    /// The requested voice is not one this model speaks with
    /// (`unsupported_voice`).
    #[error("unknown voice `{voice}`; this model speaks with: {known}")]
    UnknownVoice { voice: String, known: String },

    /// The requested option combination cannot be served (`invalid_options`).
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// Reading the input text or writing the audio failed (`io_error`).
    #[error("{0}")]
    Io(String),
}

impl From<crate::download::DownloadError> for SileroError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}

impl From<crate::torch_package::TorchPackageError> for SileroError {
    fn from(error: crate::torch_package::TorchPackageError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}

impl From<crate::audio::AudioError> for SileroError {
    fn from(error: crate::audio::AudioError) -> Self {
        Self::Io(error.to_string())
    }
}
