//! Typed errors for the OCR domain.
//!
//! Each variant maps to a stable error `code` at the CLI boundary; the
//! `#[error]` messages are human-facing and may change without affecting the
//! external contract.

/// Errors returned by the OCR engines.
#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    /// The input file is not an image this build can decode (`invalid_input`).
    #[error("cannot read image `{path}`: {source}")]
    ImageRead {
        path: String,
        #[source]
        source: image::ImageError,
    },

    /// No input page was given (`invalid_input`).
    #[error("no input pages given")]
    NoPages,

    /// An input page is not there (`invalid_input`).
    #[error("no such page image: {path}")]
    PageNotFound { path: String },

    /// A page is empty or degenerate (`invalid_input`).
    #[error("page `{path}` is {width}x{height}; needs at least 1x1 pixels")]
    EmptyPage {
        path: String,
        width: u32,
        height: u32,
    },

    /// The model name is unknown (`model_unavailable`).
    #[error("unknown model `{name}`; known models: {known}")]
    UnknownModel { name: String, known: String },

    /// The requested language has no recognizer in the catalog
    /// (`unsupported_language`).
    #[error(
        "no recognizer covers language `{lang}`; supported languages: {known}"
    )]
    UnsupportedLanguage { lang: String, known: String },

    /// Fetching a model failed (`model_unavailable`).
    #[error("model download failed: {0}")]
    ModelDownload(#[from] crate::download::DownloadError),

    /// A published artifact does not hold what the port expects
    /// (`model_unavailable`).
    #[error("invalid model artifact: {0}")]
    Artifact(String),

    /// The model directory is missing a file (`model_unavailable`).
    #[error("model file `{path}` is missing or unreadable: {source}")]
    ModelFile {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The requested option combination cannot be served
    /// (`invalid_options`).
    #[error("{0}")]
    InvalidOptions(String),

    /// Writing a result file failed (`io_error`).
    #[error("cannot write `{path}`: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The neural runtime failed (`runtime_error`).
    #[error("OCR runtime failed: {0}")]
    Runtime(String),
}

#[cfg(feature = "ocr-runtime")]
impl From<candle_core::Error> for OcrError {
    fn from(e: candle_core::Error) -> Self { Self::Runtime(e.to_string()) }
}
