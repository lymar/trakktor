//! Typed errors for `text structify`.
//!
//! Each variant maps to a stable error `code` at the CLI boundary; the
//! `#[error]` messages are human-facing and may change without affecting the
//! external contract.

/// Errors returned by the structify feature.
#[derive(Debug, thiserror::Error)]
pub enum StructifyError {
    /// The model checkpoint is malformed or unsupported (`invalid_model`).
    ///
    /// Covers geometry the pipeline cannot serve and backend failures while
    /// loading or running the network.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching a model checkpoint or the tokenizer failed
    /// (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// The tokenizer could not be loaded or applied (`model_unavailable`).
    #[error("tokenizer unavailable: {0}")]
    Tokenizer(String),

    /// The requested option combination cannot be served by this build or
    /// backend (`invalid_options`) — e.g. the burn runtime in a build without
    /// the `burn` feature, or `f16` on the burn CPU backend.
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// Reading the input text failed (`io_error`).
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

impl From<crate::download::DownloadError> for StructifyError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}
