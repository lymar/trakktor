use super::*;

const FEED_KEY: &str = "https://example.com/feed.xml";

fn rss(items: &str) -> Vec<u8> {
    format!(
        "<?xml version=\"1.0\"?><rss \
         version=\"2.0\"><channel><title>Example</title>{items}</channel></\
         rss>"
    )
    .into_bytes()
}

fn read_all(body: &[u8]) -> Vec<Publication> {
    let parsed = feedparser_rs::parse(body).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());
    build_publications(FEED_KEY, &parsed, &store, true).unwrap()
}

#[test]
fn maps_core_fields_and_uses_guid_for_uid() {
    let body = rss(
            "<item><title>First</title>\
             <link>https://example.com/posts/42</link>\
             <guid>https://example.com/posts/42</guid>\
             <pubDate>Sat, 30 May 2026 10:00:00 GMT</pubDate>\
             <description>Hello</description></item>",
        );
    let pubs = read_all(&body);
    assert_eq!(pubs.len(), 1);
    let p = &pubs[0];
    assert_eq!(p.title.as_deref(), Some("First"));
    assert_eq!(p.link.as_deref(), Some("https://example.com/posts/42"));
    assert_eq!(p.summary.as_deref(), Some("Hello"));
    assert_eq!(p.published.as_deref(), Some("2026-05-30T10:00:00Z"));
    assert!(!p.is_read);

    // uid must use the `id` (guid) branch.
    let expected =
        compute_uid(FEED_KEY, ItemKeyTag::Id, "https://example.com/posts/42");
    assert_eq!(p.uid, expected);
}

#[test]
fn falls_back_to_link_then_derived() {
    // No guid → link branch.
    let link_only = read_all(&rss(
        "<item><title>T</title><link>https://example.com/a</link></item>",
    ));
    assert_eq!(
        link_only[0].uid,
        compute_uid(FEED_KEY, ItemKeyTag::Link, "https://example.com/a")
    );

    // No guid, no link, but title + date → derived branch.
    let derived = read_all(&rss("<item><title>Only \
                                 Title</title><pubDate>Sat, 30 May 2026 \
                                 10:00:00 GMT</pubDate></item>"));
    assert_eq!(
        derived[0].uid,
        compute_uid(
            FEED_KEY,
            ItemKeyTag::Derived,
            "Only Title\u{0}2026-05-30T10:00:00Z"
        )
    );
}

#[test]
fn skips_unidentifiable_entries() {
    // An item with neither id/link nor title+date is dropped.
    let pubs = read_all(&rss("<item><description>orphan</description></item>"));
    assert!(pubs.is_empty());
}

#[test]
fn unread_filter_hides_marked_entries() {
    let body = rss(
            "<item><guid>https://example.com/posts/1</guid>\
             <title>One</title></item>",
        );
    let parsed = feedparser_rs::parse(&body).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());

    let uid =
        compute_uid(FEED_KEY, ItemKeyTag::Id, "https://example.com/posts/1");
    store.mark_read(std::slice::from_ref(&uid)).unwrap();

    // Default (unread only) hides it; --all shows it with is_read=true.
    assert!(
        build_publications(FEED_KEY, &parsed, &store, false)
            .unwrap()
            .is_empty()
    );
    let all = build_publications(FEED_KEY, &parsed, &store, true).unwrap();
    assert_eq!(all.len(), 1);
    assert!(all[0].is_read);
}
