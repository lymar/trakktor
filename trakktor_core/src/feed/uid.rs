//! Stable publication identifiers (`uid`) and feed-key canonicalization.
//!
//! Implements design.md §5 and ADR-0001:
//!
//! ```text
//! uid = hex( BLAKE3( feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key ) )
//! ```
//!
//! Determinism is critical: the same feed + entry must always hash to the same
//! uid, or read-state would be lost between runs.

/// Discriminates the source of `item_key` so that, e.g., one entry's `link`
/// accidentally equal to another's `id` cannot collide (design.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKeyTag {
    /// `item_key` came from `Entry.id`.
    Id,
    /// `item_key` came from `Entry.link`.
    Link,
    /// `item_key` was derived from `title` + date.
    Derived,
}

impl ItemKeyTag {
    fn as_str(self) -> &'static str {
        match self {
            ItemKeyTag::Id => "id",
            ItemKeyTag::Link => "link",
            ItemKeyTag::Derived => "derived",
        }
    }
}

/// Canonicalizes a feed URL into the `feed_key` used for uid computation
/// (design.md §5).
///
/// Normalization (determinism matters — see ADR-0001):
/// - scheme and host lowercased (the URL parser already does this);
/// - default port removed (`:80` for http, `:443` for https — also handled by
///   the parser);
/// - fragment (`#…`) removed;
/// - a single trailing `/` removed from the path;
/// - path and query are otherwise left as-is.
///
/// Returns `None` if `url` does not parse or has no host.
#[must_use]
pub fn canonicalize_feed_key(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    let host = parsed.host_str()?;

    let mut key = String::new();
    key.push_str(parsed.scheme());
    key.push_str("://");
    key.push_str(host);
    // `Url::port()` is `None` when the port equals the scheme default, so
    // default-port stripping is automatic.
    if let Some(port) = parsed.port() {
        key.push(':');
        key.push_str(&port.to_string());
    }
    // Drop a single trailing slash: "/" → "", "/feed/" → "/feed".
    let path = parsed.path();
    key.push_str(path.strip_suffix('/').unwrap_or(path));
    if let Some(query) = parsed.query() {
        key.push('?');
        key.push_str(query);
    }
    Some(key)
}

/// Computes a uid from its three parts (design.md §5).
///
/// `tag` is rendered between the feed key and the item key, each separated by a
/// `0x00` byte, then BLAKE3-hashed; the result is lowercase hex (64 chars).
#[must_use]
pub fn compute_uid(feed_key: &str, tag: ItemKeyTag, item_key: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(feed_key.as_bytes());
    hasher.update(&[0]);
    hasher.update(tag.as_str().as_bytes());
    hasher.update(&[0]);
    hasher.update(item_key.as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
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
        // Per §5, path/query come from the `url` parser (WHATWG/RFC 3986):
        // dot-segments resolve and disallowed characters are percent-encoded.
        // This is deterministic, which is the property §5 requires.
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
        // The hash must equal BLAKE3 over the exact byte layout from §5.
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
}
