use super::*;

#[test]
fn minimal_is_default_set() {
    // `uid` is always emitted separately, so it is not in the set.
    assert_eq!(
        parse_fields("minimal").unwrap(),
        vec![Field::Title, Field::Link]
    );
}

#[test]
fn all_covers_every_field() {
    assert_eq!(parse_fields("all").unwrap(), Field::all().to_vec());
}

#[test]
fn explicit_list_preserves_order() {
    assert_eq!(
        parse_fields("link,title,summary").unwrap(),
        vec![Field::Link, Field::Title, Field::Summary]
    );
}

#[test]
fn list_tolerates_surrounding_spaces() {
    assert_eq!(
        parse_fields("title, summary").unwrap(),
        vec![Field::Title, Field::Summary]
    );
}

#[test]
fn uid_token_is_ignored() {
    // `uid` is not a selectable field; listing it is accepted but dropped.
    assert_eq!(parse_fields("uid,title").unwrap(), vec![Field::Title]);
    assert!(parse_fields("uid").unwrap().is_empty());
}

#[test]
fn unknown_field_is_rejected() {
    // `uid` is tolerated; the genuinely unknown name still errors.
    let err = parse_fields("uid,bogus").unwrap_err();
    assert!(matches!(err, FeedError::InvalidField(name) if name == "bogus"));
}
