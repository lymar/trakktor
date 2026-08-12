//! Typed errors for the PDF document operations.
//!
//! Each variant corresponds to a stable error `code`. The variant → `code` →
//! exit-code mapping is performed at the CLI boundary; changing an `#[error]`
//! message must never change the external `code`. The vocabulary matches
//! `convert` on purpose — the same situations are named the same way — but the
//! type is this domain's own.

/// Errors returned by the PDF document operations.
#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    /// The input file is missing, unreadable, or is not a PDF at all
    /// (`invalid_input`).
    #[error("{0}")]
    InvalidInput(String),

    /// The file is a PDF but its structure could not be parsed
    /// (`parse_failed`).
    #[error("this PDF could not be parsed: {0}")]
    ParseFailed(String),

    /// The document is encrypted and no usable password was given
    /// (`encrypted`).
    #[error(
        "this PDF is encrypted; pass the user password with --password (the \
         result is written decrypted)"
    )]
    Encrypted,

    /// A flag combination or value the feature cannot honour, such as a page
    /// range that does not parse or points past the end (`invalid_options`).
    #[error("{0}")]
    InvalidOptions(String),

    /// A filesystem error while writing the result (`io_error`).
    #[error("could not write {path}: {source}")]
    Io {
        /// The path being written.
        path: String,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },
}

impl From<crate::pages::SelectionError> for PdfError {
    fn from(err: crate::pages::SelectionError) -> Self {
        PdfError::InvalidOptions(err.to_string())
    }
}
