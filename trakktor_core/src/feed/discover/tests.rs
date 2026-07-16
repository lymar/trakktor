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
    // Discovery is scoped to `<head>`; a feed-typed link in `<body>` must
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
