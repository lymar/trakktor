//! Stable publication identifiers (`uid`) and feed-key canonicalization.
//!
//! Implements the uid construction and URL-normalization rules:
//!
//! ```text
//! uid = hex( BLAKE3( feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key ) )
//! ```
//!
//! Determinism is critical: the same feed + entry must always hash to the same
//! uid, or read-state would be lost between runs.

/// Discriminates the source of `item_key` so that, e.g., one entry's `link`
/// accidentally equal to another's `id` cannot collide.
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

/// Canonicalizes a feed URL into the `feed_key` used for uid computation.
///
/// Normalization (determinism matters):
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

/// Computes a uid from its three parts.
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
mod tests;
