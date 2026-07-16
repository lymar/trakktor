//! Output data model for the feed feature.
//!
//! These types are deliberately output-agnostic: the CLI crate turns them into
//! text or JSON. Optional/empty values mean "the source had no such value" and
//! are omitted by the renderer.

use crate::feed::error::FeedError;

/// A feed discovered on a web page (`feed discover`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFeed {
    /// Absolute URL of the feed.
    pub url: String,
    /// MIME type (from the `<link type>` attribute, or derived from the
    /// recognized format for feeds found via typical paths).
    pub mime: Option<String>,
    /// Feed title (the `<link title>` attribute, or the feed's own title).
    pub title: Option<String>,
}

/// One author of a publication (`{ name, email?, uri? }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Author {
    /// Display name.
    pub name: Option<String>,
    /// Email address.
    pub email: Option<String>,
    /// URI / homepage.
    pub uri: Option<String>,
}

/// One content block of a publication (`{ type?, value }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentBlock {
    /// MIME type of the block (e.g. `text/html`); omitted when unknown.
    pub mime: Option<String>,
    /// Block body.
    pub value: String,
}

/// A publication returned by `feed read`.
///
/// Every publication carries a stable [`Publication::uid`] and an
/// [`Publication::is_read`] flag. The remaining fields mirror the parsed
/// entry; absent values are `None`/empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Publication {
    /// Stable identifier of this feed + entry.
    pub uid: String,
    /// Whether the uid is present in the read-state store.
    pub is_read: bool,
    /// Entry title.
    pub title: Option<String>,
    /// Entry link.
    pub link: Option<String>,
    /// Publication date, RFC 3339 normalized to UTC.
    pub published: Option<String>,
    /// Last-update date, RFC 3339 normalized to UTC.
    pub updated: Option<String>,
    /// Short description (may contain HTML; not sanitized).
    pub summary: Option<String>,
    /// Full content blocks.
    pub content: Vec<ContentBlock>,
    /// Authors.
    pub authors: Vec<Author>,
}

/// Summary returned by `feed mark-read`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkReadSummary {
    /// Number of uids newly marked as read.
    pub marked: usize,
    /// Number of uids that were already marked (idempotent no-ops).
    pub already_read: usize,
}

/// A selectable display field of `feed read` (`--fields`). `uid` is not among
/// these — it is the record's primary key and is always emitted, independent of
/// `--fields`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    /// `is_read`
    IsRead,
    /// `title`
    Title,
    /// `link`
    Link,
    /// `published`
    Published,
    /// `updated`
    Updated,
    /// `summary`
    Summary,
    /// `content`
    Content,
    /// `authors`
    Authors,
}

impl Field {
    /// The field's stable key, as used in JSON output and `--fields`.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Field::IsRead => "is_read",
            Field::Title => "title",
            Field::Link => "link",
            Field::Published => "published",
            Field::Updated => "updated",
            Field::Summary => "summary",
            Field::Content => "content",
            Field::Authors => "authors",
        }
    }

    /// All fields, in their canonical order.
    ///
    /// This is the `all` special value of `--fields`.
    #[must_use]
    pub fn all() -> &'static [Field] {
        &[
            Field::IsRead,
            Field::Title,
            Field::Link,
            Field::Published,
            Field::Updated,
            Field::Summary,
            Field::Content,
            Field::Authors,
        ]
    }

    /// The default `minimal` display set: `title,link`. `uid` is always emitted
    /// separately, so it is not part of the selectable fields.
    #[must_use]
    pub fn minimal() -> &'static [Field] { &[Field::Title, Field::Link] }

    fn from_name(name: &str) -> Option<Field> {
        Field::all().iter().copied().find(|f| f.key() == name)
    }
}

/// Parses the `--fields` value into an ordered list of display fields.
///
/// Accepts the special values `minimal` (the default, `title,link`) and `all`,
/// or a comma-separated list of field names; order is preserved for an explicit
/// list. `uid` is always emitted separately, so it is not a selectable field —
/// a literal `uid` token is accepted but ignored.
///
/// # Errors
///
/// Returns [`FeedError::InvalidField`] for any unknown field name.
pub fn parse_fields(spec: &str) -> Result<Vec<Field>, FeedError> {
    match spec {
        "minimal" => Ok(Field::minimal().to_vec()),
        "all" => Ok(Field::all().to_vec()),
        list => list
            .split(',')
            .map(str::trim)
            .filter(|name| *name != "uid")
            .map(|name| {
                Field::from_name(name)
                    .ok_or_else(|| FeedError::InvalidField(name.to_string()))
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests;
