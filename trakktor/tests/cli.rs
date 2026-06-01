//! End-to-end CLI tests against a local HTTP server (no external network).
//!
//! Exercises the full feed pipeline through the built binary: fetching,
//! feed-vs-page detection, autodiscovery, uid + read-state store, unread
//! filtering, and the text/JSON output contract.

use std::{
    io::{Read, Write},
    net::TcpListener,
    path::Path,
    process::{Command, Output},
    thread,
    time::Duration,
};

use serde_json::Value;

const RSS: &str = r#"<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Example</title>
  <item>
    <title>Post A</title>
    <link>http://example.com/a</link>
    <guid>http://example.com/a</guid>
    <pubDate>Sat, 30 May 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Post B</title>
    <link>http://example.com/b</link>
    <guid>http://example.com/b</guid>
  </item>
</channel></rss>"#;

const PAGE: &str = r#"<!DOCTYPE html><html><head>
  <link rel="alternate" type="application/rss+xml" href="/feed.xml">
</head><body>hello</body></html>"#;

/// Starts a throwaway HTTP/1.1 server on an ephemeral port and returns it.
fn server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();

            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request = String::from_utf8_lossy(&buf[..n]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");

            let (status, content_type, body) = if path.starts_with("/feed.xml")
            {
                ("200 OK", "application/rss+xml", RSS)
            } else if path == "/" || path.starts_with("/page") {
                ("200 OK", "text/html", PAGE)
            } else {
                ("404 Not Found", "text/plain", "")
            };

            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: \
                 {content_type}\r\nContent-Length: {}\r\nConnection: \
                 close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).ok();
            stream.flush().ok();
        }
    });
    port
}

/// Runs the built `trakktor` binary with the given arguments.
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_trakktor"))
        .args(args)
        .output()
        .expect("spawn trakktor")
}

fn stdout_json(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "expected success, got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout is valid JSON")
}

fn stderr_error_code(output: &Output) -> String {
    assert_eq!(output.status.code(), Some(1), "expected exit code 1");
    let value: Value =
        serde_json::from_slice(&output.stderr).expect("stderr JSON error");
    value["error"]["code"].as_str().unwrap().to_string()
}

fn work_dir(dir: &Path) -> String { dir.to_str().unwrap().to_string() }

#[test]
fn read_feed_returns_publications_in_order() {
    let port = server();
    let url = format!("http://127.0.0.1:{port}/feed.xml");
    let dir = tempfile::tempdir().unwrap();

    let out = run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "read",
        &url,
        "--json",
    ]);
    let value = stdout_json(&out);
    let items = value.as_array().expect("array");

    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["title"], "Post A");
    assert_eq!(items[1]["title"], "Post B");
    // minimal fields = uid,title,link → uid present, is_read absent.
    assert!(items[0]["uid"].is_string());
    assert!(items[0].get("is_read").is_none());
    assert_eq!(items[0]["link"], "http://example.com/a");
    assert_eq!(items[0]["published"], Value::Null); // not selected by minimal
}

#[test]
fn mark_read_hides_publication_until_all() {
    let port = server();
    let url = format!("http://127.0.0.1:{port}/feed.xml");
    let dir = tempfile::tempdir().unwrap();
    let wd = work_dir(dir.path());

    // Discover the uid of the first publication.
    let first =
        stdout_json(&run(&["--work-dir", &wd, "feed", "read", &url, "--json"]));
    let uid = first[0]["uid"].as_str().unwrap().to_string();

    // Mark it read.
    let marked = stdout_json(&run(&[
        "--work-dir",
        &wd,
        "feed",
        "mark-read",
        &uid,
        "--json",
    ]));
    assert_eq!(marked["marked"], 1);
    assert_eq!(marked["already_read"], 0);

    // Re-marking is idempotent.
    let again = stdout_json(&run(&[
        "--work-dir",
        &wd,
        "feed",
        "mark-read",
        &uid,
        "--json",
    ]));
    assert_eq!(again["marked"], 0);
    assert_eq!(again["already_read"], 1);

    // Default read now hides the marked publication.
    let unread =
        stdout_json(&run(&["--work-dir", &wd, "feed", "read", &url, "--json"]));
    let unread = unread.as_array().unwrap();
    assert_eq!(unread.len(), 1);
    assert_eq!(unread[0]["title"], "Post B");

    // --all returns everything; is_read is visible with --fields all.
    let all = stdout_json(&run(&[
        "--work-dir",
        &wd,
        "feed",
        "read",
        &url,
        "--all",
        "--fields",
        "all",
        "--json",
    ]));
    let all = all.as_array().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0]["is_read"], true);
    assert_eq!(all[1]["is_read"], false);
}

#[test]
fn page_url_autodiscovers_feed_with_stable_uid() {
    let port = server();
    let feed_url = format!("http://127.0.0.1:{port}/feed.xml");
    let page_url = format!("http://127.0.0.1:{port}/");
    let dir = tempfile::tempdir().unwrap();
    let wd = work_dir(dir.path());

    let via_feed = stdout_json(&run(&[
        "--work-dir",
        &wd,
        "feed",
        "read",
        &feed_url,
        "--json",
    ]));
    let via_page = stdout_json(&run(&[
        "--work-dir",
        &wd,
        "feed",
        "read",
        &page_url,
        "--json",
    ]));

    // §2 feed_key rule: reading the feed directly and via its page must yield
    // identical uids.
    assert_eq!(via_feed, via_page);
    assert_eq!(via_page.as_array().unwrap().len(), 2);
}

#[test]
fn discover_lists_declared_feed() {
    let port = server();
    let page_url = format!("http://127.0.0.1:{port}/");

    let out = run(&["feed", "discover", &page_url, "--json"]);
    let value = stdout_json(&out);
    let feeds = value.as_array().unwrap();

    assert_eq!(feeds.len(), 1);
    assert_eq!(feeds[0]["url"], format!("http://127.0.0.1:{port}/feed.xml"));
    assert_eq!(feeds[0]["type"], "application/rss+xml");
}

#[test]
fn mark_read_text_output() {
    let dir = tempfile::tempdir().unwrap();
    let uid =
        "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
    let out = run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "mark-read",
        uid,
    ]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("marked: 1"), "got: {text}");
    assert!(text.contains("already_read: 0"), "got: {text}");
}

#[test]
fn invalid_uid_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "mark-read",
        "not-a-valid-uid",
        "--json",
    ]);
    assert_eq!(stderr_error_code(&out), "invalid_uid");
}

#[test]
fn invalid_field_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    // Fields are validated before any network access.
    let out = run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "read",
        "http://127.0.0.1:1/feed.xml",
        "--fields",
        "bogus",
        "--json",
    ]);
    assert_eq!(stderr_error_code(&out), "invalid_field");
}

#[test]
fn invalid_url_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "read",
        "ftp://example.com",
        "--json",
    ]);
    assert_eq!(stderr_error_code(&out), "invalid_url");
}

#[test]
fn missing_required_argument_exits_two() {
    // Structural argument error → clap → exit code 2 (cli.md).
    let out = run(&["feed", "mark-read"]);
    assert_eq!(out.status.code(), Some(2));
}
