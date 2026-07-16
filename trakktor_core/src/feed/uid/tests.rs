use super::*;

#[test]
fn lowercases_scheme_and_host() {
    assert_eq!(
        canonicalize_feed_key("HTTP://Example.COM/Feed.xml").unwrap(),
        "http://example.com/Feed.xml"
    );
}

#[test]
fn strips_default_ports() {
    assert_eq!(
        canonicalize_feed_key("http://example.com:80/feed").unwrap(),
        "http://example.com/feed"
    );
    assert_eq!(
        canonicalize_feed_key("https://example.com:443/feed").unwrap(),
        "https://example.com/feed"
    );
}

#[test]
fn keeps_non_default_port() {
    assert_eq!(
        canonicalize_feed_key("http://example.com:8080/feed").unwrap(),
        "http://example.com:8080/feed"
    );
}

#[test]
fn drops_fragment() {
    assert_eq!(
        canonicalize_feed_key("https://example.com/feed.xml#top").unwrap(),
        "https://example.com/feed.xml"
    );
}

#[test]
fn drops_single_trailing_slash() {
    assert_eq!(
        canonicalize_feed_key("https://example.com/").unwrap(),
        "https://example.com"
    );
    assert_eq!(
        canonicalize_feed_key("https://example.com/feed/").unwrap(),
        "https://example.com/feed"
    );
}

#[test]
fn keeps_query_verbatim() {
    assert_eq!(
        canonicalize_feed_key("https://example.com/f?b=2&a=1").unwrap(),
        "https://example.com/f?b=2&a=1"
    );
}

#[test]
fn path_follows_url_standard_normalization() {
    // Path and query come from the `url` parser (WHATWG/RFC 3986):
    // dot-segments resolve and disallowed characters are percent-encoded.
    // This is deterministic, which is the property uids require.
    assert_eq!(
        canonicalize_feed_key("https://example.com/a/../b").unwrap(),
        "https://example.com/b"
    );
}

#[test]
fn uid_is_64_lowercase_hex() {
    let uid = compute_uid(
        "https://example.com/feed.xml",
        ItemKeyTag::Id,
        "https://example.com/posts/42",
    );
    assert_eq!(uid.len(), 64);
    assert!(
        uid.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    );
}

#[test]
fn uid_is_deterministic_and_tag_sensitive() {
    let by_id = compute_uid("fk", ItemKeyTag::Id, "x");
    let by_id_again = compute_uid("fk", ItemKeyTag::Id, "x");
    let by_link = compute_uid("fk", ItemKeyTag::Link, "x");
    assert_eq!(by_id, by_id_again);
    // Same item_key but a different tag must yield a different uid.
    assert_ne!(by_id, by_link);
}

#[test]
fn uid_matches_documented_construction() {
    // The hash must equal BLAKE3 over the exact documented byte layout.
    let feed_key = "https://example.com/feed.xml";
    let item_key = "https://example.com/posts/42";
    let mut expected = blake3::Hasher::new();
    expected.update(feed_key.as_bytes());
    expected.update(&[0]);
    expected.update(b"id");
    expected.update(&[0]);
    expected.update(item_key.as_bytes());
    assert_eq!(
        compute_uid(feed_key, ItemKeyTag::Id, item_key),
        expected.finalize().to_hex().to_string()
    );
}
