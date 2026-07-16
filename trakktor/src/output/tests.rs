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
    // `uid` is auto-prepended; the selection holds only display fields.
    let line =
        text_line(&publication, &[Field::Title, Field::Link, Field::Authors]);

    // Exactly one record, with one tab per field boundary (3 separators).
    assert!(!line.contains('\n'));
    assert_eq!(line.matches('\t').count(), 3);
    assert_eq!(line, "u\tmulti line title\thttps://example.com/a\tJane Doe");
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
    // uid is always present; minimal = title,link → link absent omitted,
    // is_read not selected.
    let value = publication_to_json(&publication, Field::minimal());
    let object = value.as_object().unwrap();
    assert_eq!(object.get("uid").unwrap(), "u");
    assert_eq!(object.get("title").unwrap(), "T");
    assert!(object.get("link").is_none());
    assert!(object.get("is_read").is_none());
}
