//! Output formatting: text (default) and JSON (`--json`).
//!
//! Implements `conventions/output.md`. Data goes to stdout; errors go to
//! stderr. In JSON, lists are arrays and a single result is an object; absent
//! values are omitted (never `null`). In text, lists are one record per line
//! with tab-separated fields and composite values collapsed onto one line.

use serde_json::{Map, Value, json};
use trakktor_core::{
    feed::{
        Author, ContentBlock, DiscoveredFeed, FeedError, Field,
        MarkReadSummary, Publication,
    },
    http::HttpError,
};

// ---------------------------------------------------------------------------
// feed discover
// ---------------------------------------------------------------------------

/// Prints discovered feeds (design.md §3; output.md examples).
pub fn print_discover(feeds: &[DiscoveredFeed], json: bool, pretty: bool) {
    if json {
        let array = feeds
            .iter()
            .map(|feed| {
                let mut object = Map::new();
                object.insert("url".into(), Value::String(feed.url.clone()));
                insert_opt(&mut object, "type", feed.mime.as_deref());
                insert_opt(&mut object, "title", feed.title.as_deref());
                Value::Object(object)
            })
            .collect();
        print_json(&Value::Array(array), pretty);
        return;
    }

    for feed in feeds {
        let mut line = feed.url.clone();
        if let Some(mime) = &feed.mime {
            line.push('\t');
            line.push_str(mime);
        }
        if let Some(title) = &feed.title {
            line.push('\t');
            line.push_str(title);
        }
        println!("{line}");
    }
}

// ---------------------------------------------------------------------------
// feed read
// ---------------------------------------------------------------------------

/// Prints publications, projected to the selected fields (design.md §4).
pub fn print_read(
    publications: &[Publication],
    selection: &[Field],
    json: bool,
    pretty: bool,
) {
    if json {
        let array = publications
            .iter()
            .map(|publication| publication_to_json(publication, selection))
            .collect();
        print_json(&Value::Array(array), pretty);
        return;
    }

    for publication in publications {
        println!("{}", text_line(publication, selection));
    }
}

/// Renders one publication as a single text line: selected fields in order,
/// tab-separated. Every cell is collapsed (output.md) so newlines/tabs in any
/// field — author names included — cannot break the one-record-per-line layout.
fn text_line(publication: &Publication, selection: &[Field]) -> String {
    selection
        .iter()
        .map(|field| collapse(&cell(publication, *field)))
        .collect::<Vec<_>>()
        .join("\t")
}

/// Builds the JSON object for one publication, including only selected fields
/// that are present (output.md: absent values are omitted).
fn publication_to_json(
    publication: &Publication,
    selection: &[Field],
) -> Value {
    let mut object = Map::new();
    for field in selection {
        let key = field.key();
        match field {
            Field::Uid => {
                object
                    .insert(key.into(), Value::String(publication.uid.clone()));
            },
            Field::IsRead => {
                object.insert(key.into(), Value::Bool(publication.is_read));
            },
            Field::Title => {
                insert_opt(&mut object, key, publication.title.as_deref())
            },
            Field::Link => {
                insert_opt(&mut object, key, publication.link.as_deref())
            },
            Field::Published => {
                insert_opt(&mut object, key, publication.published.as_deref());
            },
            Field::Updated => {
                insert_opt(&mut object, key, publication.updated.as_deref());
            },
            Field::Summary => {
                insert_opt(&mut object, key, publication.summary.as_deref());
            },
            Field::Content => {
                if !publication.content.is_empty() {
                    object.insert(
                        key.into(),
                        content_to_json(&publication.content),
                    );
                }
            },
            Field::Authors => {
                if !publication.authors.is_empty() {
                    object.insert(
                        key.into(),
                        authors_to_json(&publication.authors),
                    );
                }
            },
        }
    }
    Value::Object(object)
}

/// `content` as an array of `{ type?, value }` (design.md §4).
fn content_to_json(blocks: &[ContentBlock]) -> Value {
    let array = blocks
        .iter()
        .map(|block| {
            let mut object = Map::new();
            insert_opt(&mut object, "type", block.mime.as_deref());
            object.insert("value".into(), Value::String(block.value.clone()));
            Value::Object(object)
        })
        .collect();
    Value::Array(array)
}

/// `authors` as an array of `{ name?, email?, uri? }` (design.md §4).
fn authors_to_json(authors: &[Author]) -> Value {
    let array = authors
        .iter()
        .map(|author| {
            let mut object = Map::new();
            insert_opt(&mut object, "name", author.name.as_deref());
            insert_opt(&mut object, "email", author.email.as_deref());
            insert_opt(&mut object, "uri", author.uri.as_deref());
            Value::Object(object)
        })
        .collect();
    Value::Array(array)
}

/// Renders one field of a publication as a text cell. The caller collapses
/// newlines/tabs to spaces uniformly (output.md), so this only shapes composite
/// values: content blocks join with spaces, authors join with commas.
fn cell(publication: &Publication, field: Field) -> String {
    match field {
        Field::Uid => publication.uid.clone(),
        Field::IsRead => bool_text(publication.is_read).to_string(),
        Field::Title => publication.title.clone().unwrap_or_default(),
        Field::Link => publication.link.clone().unwrap_or_default(),
        Field::Published => publication.published.clone().unwrap_or_default(),
        Field::Updated => publication.updated.clone().unwrap_or_default(),
        Field::Summary => publication.summary.clone().unwrap_or_default(),
        Field::Content => publication
            .content
            .iter()
            .map(|block| block.value.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        Field::Authors => publication
            .authors
            .iter()
            .map(author_display)
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// Text display of an author: name, else email, else uri.
fn author_display(author: &Author) -> String {
    author
        .name
        .clone()
        .or_else(|| author.email.clone())
        .or_else(|| author.uri.clone())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// feed mark-read
// ---------------------------------------------------------------------------

/// Prints the mark-read summary (design.md §2; output.md single-object form).
pub fn print_mark_read(summary: &MarkReadSummary, json: bool, pretty: bool) {
    if json {
        let value = json!({
            "marked": summary.marked,
            "already_read": summary.already_read,
        });
        print_json(&value, pretty);
    } else {
        println!("marked: {}", summary.marked);
        println!("already_read: {}", summary.already_read);
    }
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Emits an error to stderr (output.md). JSON form:
/// `{ "error": { "code", "message" } }`; text form: a plain message.
pub fn emit_error(err: &FeedError, json: bool, pretty: bool) {
    if json {
        let value = json!({
            "error": { "code": error_code(err), "message": err.to_string() },
        });
        let rendered = render_json(&value, pretty);
        eprintln!("{rendered}");
    } else {
        eprintln!("error: {err}");
    }
}

/// Maps a [`FeedError`] to its stable `code` (design.md §9). The variant →
/// `code` mapping lives at the bin boundary (error-handling.md).
fn error_code(err: &FeedError) -> &'static str {
    match err {
        FeedError::Http(http) => match http {
            HttpError::InvalidUrl(_) => "invalid_url",
            HttpError::Fetch(_) => "fetch_failed",
            HttpError::Status(_) => "http_error",
            HttpError::TooLarge { .. } => "too_large",
        },
        FeedError::ParseFailed(_) => "parse_failed",
        FeedError::FeedNotFound => "feed_not_found",
        FeedError::InvalidUid(_) => "invalid_uid",
        FeedError::InvalidField(_) => "invalid_field",
        FeedError::Io(_) => "io_error",
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Inserts `key => value` only when `value` is `Some` (output.md: omit absent).
fn insert_opt(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn bool_text(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

/// Collapses newlines and tabs to spaces so a value fits one text cell.
fn collapse(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if matches!(c, '\n' | '\r' | '\t') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn print_json(value: &Value, pretty: bool) {
    println!("{}", render_json(value, pretty));
}

fn render_json(value: &Value, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .expect("serializing a serde_json::Value never fails")
}

#[cfg(test)]
mod tests {
    use trakktor_core::feed::Field;

    use super::*;

    #[test]
    fn text_line_collapses_control_chars_in_every_field() {
        // A publication whose title and author name carry newlines/tabs — as
        // untrusted feed data might.
        let publication = Publication {
            uid: "u".into(),
            is_read: false,
            title: Some("multi\nline\ttitle".into()),
            link: Some("https://example.com/a".into()),
            published: None,
            updated: None,
            summary: None,
            content: Vec::new(),
            authors: vec![Author {
                name: Some("Jane\nDoe".into()),
                email: None,
                uri: None,
            }],
        };
        let line = text_line(
            &publication,
            &[Field::Uid, Field::Title, Field::Link, Field::Authors],
        );

        // Exactly one record, with one tab per field boundary (3 separators).
        assert!(!line.contains('\n'));
        assert_eq!(line.matches('\t').count(), 3);
        assert_eq!(
            line,
            "u\tmulti line title\thttps://example.com/a\tJane Doe"
        );
    }

    #[test]
    fn json_omits_absent_and_unselected_fields() {
        let publication = Publication {
            uid: "u".into(),
            is_read: true,
            title: Some("T".into()),
            link: None,
            published: None,
            updated: None,
            summary: None,
            content: Vec::new(),
            authors: Vec::new(),
        };
        // minimal = uid,title,link → link absent omitted, is_read not selected.
        let value = publication_to_json(&publication, Field::minimal());
        let object = value.as_object().unwrap();
        assert_eq!(object.get("uid").unwrap(), "u");
        assert_eq!(object.get("title").unwrap(), "T");
        assert!(object.get("link").is_none());
        assert!(object.get("is_read").is_none());
    }
}
