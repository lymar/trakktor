//! Feeds: discover, read, and mark-read for RSS/Atom/JSON Feed.
//!
//! Full specification: `../trakktor_project/docs/features/feed/design.md`.
//! Key cross-cutting rules: `conventions/{cli,output,http,error-handling,
//! working-directory}.md`; uid + read-state design: ADR-0001.
//!
//! This module exposes the three operations behind `trakktor feed …`; the CLI
//! crate formats their return values.

pub mod discover;
pub mod error;
pub mod model;
pub mod read;
pub mod store;
pub mod uid;

use std::path::Path;

pub use error::FeedError;
pub use model::{
    Author, ContentBlock, DiscoveredFeed, Field, MarkReadSummary, Publication,
    parse_fields,
};

use crate::http::{HttpClient, HttpError};

/// Discovers the feeds declared on a web page (`feed discover`, design.md §3).
///
/// Fetches `page_url` as HTML and returns the feeds found via
/// `<link rel="alternate">` or, if none are declared, via typical paths. An
/// empty result is success (exit 0); only `read` raises `feed_not_found`.
///
/// # Errors
///
/// Returns a [`FeedError`] for an invalid URL or a transport/HTTP failure.
pub fn discover(page_url: &str) -> Result<Vec<DiscoveredFeed>, FeedError> {
    let client = HttpClient::new()?;
    let body = client.get_bytes(page_url)?;
    let base = url::Url::parse(page_url).map_err(|_| {
        FeedError::Http(HttpError::InvalidUrl(page_url.to_string()))
    })?;
    discover::discover_from_page(&client, &base, &body)
}

/// Reads a feed and returns its publications (`feed read`, design.md §2).
///
/// `url` may be a feed or a regular page (autodiscovery applies). When `all`
/// is false, only unread publications are returned. `work_dir` is the resolved
/// trakktor working directory (the read-state store lives at `work_dir/feed`).
///
/// # Errors
///
/// Returns a [`FeedError`] for invalid URLs, transport/HTTP/parse failures, a
/// missing feed, or store I/O.
pub fn read(
    url: &str,
    all: bool,
    work_dir: &Path,
) -> Result<Vec<Publication>, FeedError> {
    let client = HttpClient::new()?;
    let store = store::ReadStore::new(work_dir);
    read::read_feed(&client, &store, url, all)
}

/// Marks publications as read by uid; idempotent (`feed mark-read`, §2/§6).
///
/// # Errors
///
/// Returns [`FeedError::InvalidUid`] for a malformed uid or [`FeedError::Io`]
/// on store failures.
pub fn mark_read(
    uids: &[String],
    work_dir: &Path,
) -> Result<MarkReadSummary, FeedError> {
    let store = store::ReadStore::new(work_dir);
    store.mark_read(uids)
}
