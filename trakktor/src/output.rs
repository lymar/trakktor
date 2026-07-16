//! Output formatting: JSON (default) and human-readable text (`--text`).
//!
//! Data goes to stdout; errors go to stderr. In JSON, lists are arrays and a
//! single result is an object; absent values are omitted (never `null`). In
//! text, lists are one record per line with tab-separated fields and composite
//! values collapsed onto one line.

use std::path::Path;

use serde_json::{Map, Value, json};
use trakktor_core::{
    feed::{
        Author, ContentBlock, DiscoveredFeed, Field, MarkReadSummary,
        Publication,
    },
    skill::WriteOutcome,
};

use crate::error::CliError;

// ---------------------------------------------------------------------------
// feed discover
// ---------------------------------------------------------------------------

/// Prints discovered feeds.
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

/// Prints publications, projected to the selected fields.
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
    if !publications.is_empty() {
        // The `uid` (first column) is each record's primary key; show how to
        // act on it. Printed to stderr so stdout stays a clean data
        // stream.
        eprintln!(
            "hint: each publication's \"uid\" (the first column) is its \
             primary key; mark an item read with: trakktor feed mark-read \
             <uid>"
        );
    }
}

/// Renders one publication as a single text line: the primary-key `uid` first,
/// then the selected fields in order, tab-separated. Every cell is collapsed so
/// newlines/tabs in any field — author names included — cannot break the
/// one-record-per-line layout.
fn text_line(publication: &Publication, selection: &[Field]) -> String {
    let mut cells = vec![publication.uid.clone()];
    cells.extend(
        selection
            .iter()
            .map(|field| collapse(&cell(publication, *field))),
    );
    cells.join("\t")
}

/// Builds the JSON object for one publication: the primary-key `uid` is always
/// present, followed by the selected fields that are present (absent values are
/// omitted).
fn publication_to_json(
    publication: &Publication,
    selection: &[Field],
) -> Value {
    let mut object = Map::new();
    object.insert("uid".into(), Value::String(publication.uid.clone()));
    for field in selection {
        let key = field.key();
        match field {
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

/// `content` as an array of `{ type?, value }`.
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

/// `authors` as an array of `{ name?, email?, uri? }`.
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
/// newlines/tabs to spaces uniformly, so this only shapes composite values:
/// content blocks join with spaces, authors join with commas.
fn cell(publication: &Publication, field: Field) -> String {
    match field {
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

/// Prints the mark-read summary (single-object form).
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
// skill show / install
// ---------------------------------------------------------------------------

/// Prints the generated skill guide (`skill show`). The guide is a prose
/// document; by default it is wrapped as `{ "content": "…" }`, and `--text`
/// prints the Markdown verbatim.
pub fn print_skill_show(content: &str, json: bool, pretty: bool) {
    if json {
        print_json(&json!({ "content": content }), pretty);
    } else {
        println!("{content}");
    }
}

/// Prints the install result (`skill install`). One destination per run, so the
/// JSON form is a single object
/// `{ "path": "<path>", "status": "written" | "skipped" }`: `written` when the
/// stub was created or overwritten, `skipped` when an existing file was left in
/// place without `--force`.
pub fn print_skill_install(
    path: &Path,
    outcome: WriteOutcome,
    json: bool,
    pretty: bool,
) {
    let status = match outcome {
        WriteOutcome::Written => "written",
        WriteOutcome::Skipped => "skipped",
    };
    if json {
        print_json(
            &json!({ "path": path.display().to_string(), "status": status }),
            pretty,
        );
        return;
    }
    match outcome {
        WriteOutcome::Written => println!("installed: {}", path.display()),
        WriteOutcome::Skipped => {
            println!("skipped (exists, use --force): {}", path.display());
        },
    }
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Emits an error to stderr. JSON form:
/// `{ "error": { "code", "message" } }`; text form: a plain message. The stable
/// `code` is resolved by [`CliError::code`] at the bin boundary.
pub fn emit_error(err: &CliError, json: bool, pretty: bool) {
    if json {
        let value = json!({
            "error": { "code": err.code(), "message": err.to_string() },
        });
        let rendered = render_json(&value, pretty);
        eprintln!("{rendered}");
    } else {
        eprintln!("error: {err}");
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Inserts `key => value` only when `value` is `Some` (omit absent).
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
mod tests;
