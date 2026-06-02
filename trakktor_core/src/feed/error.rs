//! Typed errors for the feed feature.
//!
//! Each variant corresponds to a stable error `code`. The variant → `code` →
//! exit-code mapping is performed at the CLI boundary; changing an `#[error]`
//! message must never change the external `code`.

use crate::http::HttpError;

/// Errors returned by feed operations.
#[derive(Debug, thiserror::Error)]
pub enum FeedError {
    /// Transport/protocol error while fetching a feed or page.
    ///
    /// Covers the `invalid_url`, `fetch_failed`, `http_error` and `too_large`
    /// codes (discriminated by the inner [`HttpError`]).
    #[error(transparent)]
    Http(#[from] HttpError),

    /// Content was recognized as a feed but could not be parsed
    /// (`parse_failed`). The parser error is preserved as the source.
    #[error("content was recognized as a feed but could not be parsed: {0}")]
    ParseFailed(#[source] feedparser_rs::FeedError),

    /// No feed could be found for the given page URL (`feed_not_found`).
    #[error("no feed found at URL")]
    FeedNotFound,

    /// A `mark-read` argument was not a valid uid (`invalid_uid`).
    ///
    /// A valid uid is 64 lowercase hex characters (BLAKE3 output).
    #[error("invalid uid: {0}")]
    InvalidUid(String),

    /// An unknown field name was passed to `--fields` (`invalid_field`).
    #[error("unknown field name: {0}")]
    InvalidField(String),

    /// A filesystem error while working with the read-state store
    /// (`io_error`).
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),
}
