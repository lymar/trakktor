//! Typed errors for the conversion feature.
//!
//! Each variant corresponds to a stable error `code`. The variant → `code` →
//! exit-code mapping is performed at the CLI boundary; changing an `#[error]`
//! message must never change the external `code`.

/// Errors returned by the conversion operations.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
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
        "this PDF is encrypted; pass the password with --password (an owner \
         password unlocks nothing here — the text stays out of reach)"
    )]
    Encrypted,

    /// Not one page of the document carries text to convert (`no_text_layer`).
    /// The message names the command that can read such a document.
    #[error(
        "this PDF has no text layer to convert — its pages are {what}. Read \
         them with `trakktor ocr` instead: it recognizes the page as a picture"
    )]
    NoTextLayer {
        /// What the pages turned out to be, in the plural and in words:
        /// "scanned images", "text drawn as vector outlines", and so on.
        what: String,
    },

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
