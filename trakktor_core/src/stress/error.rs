//! Typed errors for `text stress`.
//!
//! Each variant maps to a stable error `code` at the CLI boundary; the
//! `#[error]` messages are human-facing and may change without affecting the
//! external contract.

/// Errors returned by the stress feature.
#[derive(Debug, thiserror::Error)]
pub enum StressError {
    /// The model is malformed or unsupported (`model_unavailable`).
    ///
    /// Covers a checkpoint whose weights do not match the declared geometry, a
    /// table the conversion could not find, and backend failures while loading
    /// or running the networks.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching or converting the model failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// The word-piece tokenizer could not be built from the model's vocabulary
    /// or applied (`model_unavailable`).
    #[error("tokenizer unavailable: {0}")]
    Tokenizer(String),

    /// The requested option combination cannot be served by this build or
    /// backend (`invalid_options`) — e.g. the burn runtime in a build without
    /// the `burn` feature, or `f16` on the burn CPU backend.
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// A user dictionary could not be parsed (`invalid_options`).
    #[error("invalid dictionary: {0}")]
    Dictionary(String),

    /// Reading the input text or a dictionary failed (`io_error`).
    #[error("failed to read input: {0}")]
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

impl From<crate::download::DownloadError> for StressError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}

impl From<crate::torch_package::TorchPackageError> for StressError {
    fn from(error: crate::torch_package::TorchPackageError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}
