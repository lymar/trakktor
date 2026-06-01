//! Output data model for the feed feature (design.md §3, §4).
//!
//! These types are deliberately output-agnostic: the CLI crate turns them into
//! text or JSON (see `conventions/output.md`). Optional/empty values mean "the
//! source had no such value" and are omitted by the renderer.

use crate::feed::error::FeedError;

/// A feed discovered on a web page (`feed discover`, design.md §3).
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

/// One author of a publication (design.md §4 — `{ name, email?, uri? }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Author {
    /// Display name.
    pub name: Option<String>,
    /// Email address.
    pub email: Option<String>,
    /// URI / homepage.
    pub uri: Option<String>,
}

/// One content block of a publication (design.md §4 — `{ type?, value }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentBlock {
    /// MIME type of the block (e.g. `text/html`); omitted when unknown.
    pub mime: Option<String>,
    /// Block body.
    pub value: String,
}

/// A publication returned by `feed read` (design.md §4).
///
/// Every publication carries a stable [`Publication::uid`] (§5) and an
/// [`Publication::is_read`] flag (§6). The remaining fields mirror the parsed
/// entry; absent values are `None`/empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Publication {
    /// Stable identifier of this feed + entry (design.md §5).
    pub uid: String,
    /// Whether the uid is present in the read-state store (design.md §6).
    pub is_read: bool,
    /// Entry title.
    pub title: Option<String>,
    /// Entry link.
    pub link: Option<String>,
    /// Publication date, RFC 3339 normalized to UTC.
    pub published: Option<String>,
    /// Last-update date, RFC 3339 normalized to UTC.
    pub updated: Option<String>,
    /// Short description (may contain HTML; not sanitized — design.md §8).
    pub summary: Option<String>,
    /// Full content blocks.
    pub content: Vec<ContentBlock>,
    /// Authors.
    pub authors: Vec<Author>,
}

/// Summary returned by `feed mark-read` (design.md §2, cli.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkReadSummary {
    /// Number of uids newly marked as read.
    pub marked: usize,
    /// Number of uids that were already marked (idempotent no-ops).
    pub already_read: usize,
}

/// A selectable output field of `feed read` (design.md §4, `--fields`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    /// `uid`
    Uid,
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
            Field::Uid => "uid",
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

    /// All fields, in the canonical order of the design.md §4 table.
    ///
    /// This is the `all` special value of `--fields`.
    #[must_use]
    pub fn all() -> &'static [Field] {
        &[
            Field::Uid,
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

    /// The default `minimal` field set: `uid,title,link` (design.md §4).
    #[must_use]
    pub fn minimal() -> &'static [Field] {
        &[Field::Uid, Field::Title, Field::Link]
    }

    fn from_name(name: &str) -> Option<Field> {
        Field::all().iter().copied().find(|f| f.key() == name)
    }
}

/// Parses the `--fields` value into an ordered field list (design.md §4).
///
/// Accepts the special values `minimal` (the default, `uid,title,link`) and
/// `all`, or a comma-separated list of field names. Order is preserved for an
/// explicit list.
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
            .map(|name| {
                let name = name.trim();
                Field::from_name(name)
                    .ok_or_else(|| FeedError::InvalidField(name.to_string()))
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_is_default_set() {
        assert_eq!(
            parse_fields("minimal").unwrap(),
            vec![Field::Uid, Field::Title, Field::Link]
        );
    }

    #[test]
    fn all_covers_every_field() {
        assert_eq!(parse_fields("all").unwrap(), Field::all().to_vec());
    }

    #[test]
    fn explicit_list_preserves_order() {
        assert_eq!(
            parse_fields("link,uid,title").unwrap(),
            vec![Field::Link, Field::Uid, Field::Title]
        );
    }

    #[test]
    fn list_tolerates_surrounding_spaces() {
        assert_eq!(
            parse_fields("uid, summary").unwrap(),
            vec![Field::Uid, Field::Summary]
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = parse_fields("uid,bogus").unwrap_err();
        assert!(
            matches!(err, FeedError::InvalidField(name) if name == "bogus")
        );
    }
}
