//! The `feed read` pipeline.
//!
//! Fetch the URL, decide whether it is a feed or a page, parse the feed, then
//! turn each identifiable entry into a [`Publication`] carrying its stable uid
//! and read flag. Unread filtering is applied last.

use chrono::{DateTime, SecondsFormat, Utc};
use feedparser_rs::{Content, Entry, FeedVersion, ParsedFeed, Person};
use url::Url;

use crate::{
    feed::{
        discover::discover_from_page,
        error::FeedError,
        model::{Author, ContentBlock, Publication},
        store::ReadStore,
        uid::{ItemKeyTag, canonicalize_feed_key, compute_uid},
    },
    http::{HttpClient, HttpError},
};

/// Reads the feed at `url`, returning its publications.
///
/// If `url` is a feed it is used directly; otherwise the body is treated as
/// HTML and the first autodiscovered feed is read. The `feed_key` for uid
/// computation is the URL trakktor actually fetches, canonicalized, before the
/// feed's own redirects.
///
/// When `all` is false, only unread publications are returned.
///
/// # Errors
///
/// Returns a [`FeedError`] for invalid URLs, transport/HTTP failures, parse
/// failures, a missing feed, or store I/O.
pub fn read_feed(
    client: &HttpClient,
    store: &ReadStore,
    url: &str,
    all: bool,
) -> Result<Vec<Publication>, FeedError> {
    let body = client.get_bytes(url)?;

    // feedparser detects the format from the first significant byte; an
    // unrecognized format (`Unknown`) means "this is not a feed". `parse`
    // rarely errors thanks to the bozo pattern — when it does, the content was
    // feed-shaped but unparseable (`parse_failed`).
    let parsed_input =
        feedparser_rs::parse(&body).map_err(FeedError::ParseFailed)?;

    let (feed_key, parsed) = if parsed_input.version != FeedVersion::Unknown {
        // The URL pointed directly at a feed.
        let feed_key = canonicalize_feed_key(url).ok_or_else(|| {
            FeedError::Http(HttpError::InvalidUrl(url.to_string()))
        })?;
        (feed_key, parsed_input)
    } else {
        // Not a feed: autodiscover on the already-downloaded body and read the
        // first feed found.
        let base = Url::parse(url).map_err(|_| {
            FeedError::Http(HttpError::InvalidUrl(url.to_string()))
        })?;
        let first = discover_from_page(client, &base, &body)?
            .into_iter()
            .next()
            .ok_or(FeedError::FeedNotFound)?;
        let feed_key = canonicalize_feed_key(&first.url).ok_or_else(|| {
            FeedError::Http(HttpError::InvalidUrl(first.url.clone()))
        })?;
        let feed_body = client.get_bytes(&first.url)?;
        let parsed =
            feedparser_rs::parse(&feed_body).map_err(FeedError::ParseFailed)?;
        (feed_key, parsed)
    };

    build_publications(&feed_key, &parsed, store, all)
}

/// Maps a parsed feed to publications, computing uids and read flags and
/// applying the unread filter.
///
/// Entries without a usable identity (no `id`, `link`, or `title`+date) are
/// skipped. Publication order follows the feed.
///
/// # Errors
///
/// Returns [`FeedError::Io`] if the read-state store cannot be queried.
pub fn build_publications(
    feed_key: &str,
    parsed: &ParsedFeed,
    store: &ReadStore,
    all: bool,
) -> Result<Vec<Publication>, FeedError> {
    let mut publications = Vec::new();
    for entry in &parsed.entries {
        let Some((tag, item_key)) = entry_item_key(entry) else {
            continue; // Unidentifiable → never emitted.
        };
        let uid = compute_uid(feed_key, tag, &item_key);
        let is_read = store.is_read(&uid)?;
        if !all && is_read {
            continue;
        }
        publications.push(build_publication(uid, is_read, entry));
    }
    Ok(publications)
}

/// Selects the `(tag, item_key)` for an entry's uid via the fallback chain
/// `id` → `link` → derived (`title` + date). Returns `None` when the entry is
/// unidentifiable.
fn entry_item_key(entry: &Entry) -> Option<(ItemKeyTag, String)> {
    if let Some(id) = present(entry.id.as_deref()) {
        return Some((ItemKeyTag::Id, id.to_string()));
    }
    if let Some(link) = present(entry.link.as_deref()) {
        return Some((ItemKeyTag::Link, link.to_string()));
    }
    // Derived requires both a title and a date (published, else updated).
    let title = present(entry.title.as_deref())?;
    let date = entry.published.or(entry.updated)?;
    Some((
        ItemKeyTag::Derived,
        format!("{title}\0{}", format_rfc3339(date)),
    ))
}

/// Builds a [`Publication`] from an entry. Absent/empty source values become
/// `None`/empty.
fn build_publication(uid: String, is_read: bool, entry: &Entry) -> Publication {
    Publication {
        uid,
        is_read,
        title: present(entry.title.as_deref()).map(str::to_string),
        link: present(entry.link.as_deref()).map(str::to_string),
        published: entry.published.map(format_rfc3339),
        updated: entry.updated.map(format_rfc3339),
        summary: present(entry.summary.as_deref()).map(str::to_string),
        content: entry.content.iter().filter_map(content_block).collect(),
        authors: map_authors(entry),
    }
}

/// Maps a content block, dropping blocks with an empty body.
fn content_block(content: &Content) -> Option<ContentBlock> {
    if content.value.is_empty() {
        return None;
    }
    Some(ContentBlock {
        mime: present(content.content_type.as_deref()).map(str::to_string),
        value: content.value.clone(),
    })
}

/// Maps an entry's authors: the structured `authors` list, or a single entry
/// from the flat `author` string as a fallback.
fn map_authors(entry: &Entry) -> Vec<Author> {
    let mut authors: Vec<Author> =
        entry.authors.iter().filter_map(person_to_author).collect();
    if authors.is_empty() &&
        let Some(name) = present(entry.author.as_deref())
    {
        authors.push(Author {
            name: Some(name.to_string()),
            email: None,
            uri: None,
        });
    }
    authors
}

/// Maps one [`Person`] to an [`Author`], dropping persons with no usable field.
fn person_to_author(person: &Person) -> Option<Author> {
    let name = present(person.name.as_deref()).map(str::to_string);
    let email = present(person.email.as_deref()).map(str::to_string);
    let uri = present(person.uri.as_deref()).map(str::to_string);
    if name.is_none() && email.is_none() && uri.is_none() {
        None
    } else {
        Some(Author { name, email, uri })
    }
}

/// Treats an empty string as absent. feedparser already trims text, so the
/// stored value is used verbatim — important for uid determinism.
fn present(value: Option<&str>) -> Option<&str> {
    value.filter(|s| !s.is_empty())
}

/// Formats a timestamp as RFC 3339, normalized to UTC with a `Z` suffix and
/// second precision.
fn format_rfc3339(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests;
