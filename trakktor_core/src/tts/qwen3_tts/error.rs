//! Typed errors for the Qwen3-TTS engine.
//!
//! Each variant maps to a stable error `code` at the CLI boundary; the
//! `#[error]` messages are human-facing and may change without affecting the
//! external contract.

/// Errors returned by the Qwen3-TTS engine.
#[derive(Debug, thiserror::Error)]
pub enum Qwen3TtsError {
    /// The text to synthesize is empty (`text_empty`).
    #[error("the text to synthesize is empty")]
    TextEmpty,

    /// The model checkpoint is malformed or unsupported (`model_unavailable`).
    ///
    /// Covers geometry the pipeline cannot serve and backend failures while
    /// loading or running the network.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Fetching a model checkpoint or the tokenizer failed
    /// (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(String),

    /// The text tokenizer could not be loaded or applied
    /// (`model_unavailable`).
    #[error("tokenizer unavailable: {0}")]
    Tokenizer(String),

    /// The requested voice is not available on the selected model
    /// (`unsupported_voice`).
    #[error("unsupported voice: {0}")]
    UnsupportedVoice(String),

    /// The requested target language is not one the model knows
    /// (`unsupported_language`).
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// The requested option combination cannot be served by this build or
    /// backend (`invalid_options`) — e.g. the burn runtime in a build without
    /// the `burn` feature, or mutually exclusive voice flags.
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// Writing the output audio failed (`io_error`).
    #[error("io error: {0}")]
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

impl From<crate::download::DownloadError> for Qwen3TtsError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}
