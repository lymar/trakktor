//! Feed autodiscovery (design.md §3).
//!
//! Two strategies, in priority order:
//! 1. `<link rel="alternate" type="…">` elements in the page;
//! 2. only if none are found, a set of typical paths (`/feed`, `/rss`, …), each
//!    treated as a feed when it actually parses.
//!
//! Results are ordered: `<link>` feeds in document order first, then
//! typical-path feeds in the listed order. `feed read` uses the first of these.

use feedparser_rs::FeedVersion;
use scraper::{Html, Selector};
use url::Url;

use crate::{
    feed::{error::FeedError, model::DiscoveredFeed},
    http::HttpClient,
};

/// MIME types that mark a `<link>` as a feed (design.md §3).
const FEED_LINK_TYPES: &[&str] = &[
    "application/rss+xml",
    "application/atom+xml",
    "application/json",
    "application/feed+json",
];

/// Typical feed paths probed when no `<link>` is present (design.md §3),
/// in the order they are tried.
pub const TYPICAL_PATHS: &[&str] = &[
    "/feed",
    "/feed.xml",
    "/rss",
    "/rss.xml",
    "/atom.xml",
    "/feed.json",
];

/// Discovers feeds from an already-downloaded page body (design.md §3).
///
/// `base` is the page URL, used to resolve relative `href`s and typical paths.
/// `<link>` feeds take precedence; typical paths are probed (with network
/// requests) only when no `<link>` feed is declared.
///
/// # Errors
///
/// Currently infallible in practice (per-path probe failures are ignored), but
/// returns [`FeedError`] for forward compatibility with the operation API.
pub fn discover_from_page(
    client: &HttpClient,
    base: &Url,
    body: &[u8],
) -> Result<Vec<DiscoveredFeed>, FeedError> {
    let html = String::from_utf8_lossy(body);
    let link_feeds = extract_link_feeds(base, &html);
    if !link_feeds.is_empty() {
        return Ok(link_feeds);
    }
    Ok(probe_typical_paths(client, base))
}

/// Extracts feeds declared via `<head>` `<link rel="alternate" type="…">`
/// (design.md §3 — the search is scoped to `<head>`). Pure: no network.
/// Relative `href`s are resolved against `base`; results are in document order.
#[must_use]
pub fn extract_link_feeds(base: &Url, html: &str) -> Vec<DiscoveredFeed> {
    let document = Html::parse_document(html);
    // §3 restricts discovery to `<head>`; HTML5 parsing keeps a `<link>` placed
    // in flow content inside `<body>`, so a `<head>`-scoped selector excludes
    // stray feed-typed links from the page body.
    let selector =
        Selector::parse("head link").expect("static `head link` selector");

    let mut feeds = Vec::new();
    for element in document.select(&selector) {
        let el = element.value();

        // `rel` must contain the `alternate` token (case-insensitive).
        let rel = el.attr("rel").unwrap_or_default();
        if !rel
            .split_whitespace()
            .any(|token| token.eq_ignore_ascii_case("alternate"))
        {
            continue;
        }

        // `type` must be one of the feed MIME types (ignoring parameters and
        // case).
        let Some(type_attr) = el.attr("type") else {
            continue;
        };
        let mime = type_attr
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if !FEED_LINK_TYPES.contains(&mime.as_str()) {
            continue;
        }

        let Some(href) = el.attr("href") else {
            continue;
        };
        let Ok(absolute) = base.join(href.trim()) else {
            continue;
        };

        feeds.push(DiscoveredFeed {
            url: absolute.to_string(),
            mime: Some(type_attr.trim().to_string()),
            title: el
                .attr("title")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        });
    }
    feeds
}

/// Probes the typical feed paths against `base`, returning those that parse as
/// a feed (design.md §3), in [`TYPICAL_PATHS`] order. Per-path transport and
/// parse failures are expected for guesses and silently skipped.
fn probe_typical_paths(client: &HttpClient, base: &Url) -> Vec<DiscoveredFeed> {
    let mut feeds = Vec::new();
    for path in TYPICAL_PATHS {
        let Ok(candidate) = base.join(path) else {
            continue;
        };
        let url = candidate.to_string();
        let Ok(body) = client.get_bytes(&url) else {
            continue;
        };
        let Ok(parsed) = feedparser_rs::parse(&body) else {
            continue;
        };
        let Some(mime) = mime_for_version(parsed.version) else {
            continue; // Not a recognized feed.
        };
        feeds.push(DiscoveredFeed {
            url,
            mime: Some(mime.to_string()),
            title: parsed
                .feed
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        });
    }
    feeds
}

/// Maps a recognized feed format to the MIME type reported by `discover`
/// (design.md §3). Returns `None` for [`FeedVersion::Unknown`].
#[must_use]
pub fn mime_for_version(version: FeedVersion) -> Option<&'static str> {
    match version {
        FeedVersion::Rss090 |
        FeedVersion::Rss091Netscape |
        FeedVersion::Rss091Userland |
        FeedVersion::Rss092 |
        FeedVersion::Rss20 |
        FeedVersion::Rss10 => Some("application/rss+xml"),
        FeedVersion::Atom10 | FeedVersion::Atom03 => {
            Some("application/atom+xml")
        },
        FeedVersion::JsonFeed10 | FeedVersion::JsonFeed11 => {
            Some("application/feed+json")
        },
        FeedVersion::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Url { Url::parse("https://example.com/blog/").unwrap() }

    #[test]
    fn extracts_feeds_in_document_order() {
        let html = r#"
            <html><head>
              <link rel="alternate" type="application/rss+xml"
                    href="/feed.xml" title="RSS">
              <link rel="alternate" type="application/atom+xml"
                    href="https://other.example/atom">
              <link rel="alternate" type="application/feed+json"
                    href="feed.json">
            </head></html>
        "#;
        let feeds = extract_link_feeds(&base(), html);
        assert_eq!(feeds.len(), 3);

        // Relative href resolved against the page origin.
        assert_eq!(feeds[0].url, "https://example.com/feed.xml");
        assert_eq!(feeds[0].mime.as_deref(), Some("application/rss+xml"));
        assert_eq!(feeds[0].title.as_deref(), Some("RSS"));

        // Absolute href kept; no title attribute → None.
        assert_eq!(feeds[1].url, "https://other.example/atom");
        assert_eq!(feeds[1].title, None);

        // Relative href resolved against the page directory.
        assert_eq!(feeds[2].url, "https://example.com/blog/feed.json");
    }

    #[test]
    fn ignores_non_feed_links() {
        let html = r#"
            <head>
              <link rel="stylesheet" type="text/css" href="/style.css">
              <link rel="alternate" type="text/html" href="/amp">
              <link rel="icon" href="/favicon.ico">
              <link type="application/rss+xml" href="/no-rel.xml">
            </head>
        "#;
        // Only feed-typed `rel=alternate` links count; the last lacks `rel`.
        assert!(extract_link_feeds(&base(), html).is_empty());
    }

    #[test]
    fn ignores_feed_links_outside_head() {
        // §3 scopes discovery to `<head>`; a feed-typed link in `<body>` must
        // not be reported.
        let html = r#"
            <html>
              <head>
                <link rel="alternate" type="application/rss+xml"
                      href="/head-feed.xml">
              </head>
              <body>
                <link rel="alternate" type="application/rss+xml"
                      href="/body-feed.xml">
              </body>
            </html>
        "#;
        let feeds = extract_link_feeds(&base(), html);
        assert_eq!(feeds.len(), 1);
        assert_eq!(feeds[0].url, "https://example.com/head-feed.xml");
    }

    #[test]
    fn accepts_type_with_parameters_and_mixed_case() {
        let html = r#"
            <head>
              <link rel="ALTERNATE" type="Application/RSS+XML; charset=utf-8"
                    href="/feed.xml">
            </head>
        "#;
        let feeds = extract_link_feeds(&base(), html);
        assert_eq!(feeds.len(), 1);
        assert_eq!(feeds[0].url, "https://example.com/feed.xml");
    }

    #[test]
    fn maps_versions_to_mime() {
        assert_eq!(
            mime_for_version(FeedVersion::Rss20),
            Some("application/rss+xml")
        );
        assert_eq!(
            mime_for_version(FeedVersion::Atom10),
            Some("application/atom+xml")
        );
        assert_eq!(
            mime_for_version(FeedVersion::JsonFeed11),
            Some("application/feed+json")
        );
        assert_eq!(mime_for_version(FeedVersion::Unknown), None);
    }
}
