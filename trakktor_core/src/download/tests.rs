//! Tests for the shared downloader.
//!
//! The interesting behaviour is all in what happens when a transfer does *not*
//! go smoothly, so most of these run against a throwaway HTTP server that can
//! be told to cut a connection, refuse a range, or fail — the situations a
//! four-gigabyte checkpoint over a home connection eventually meets, and that
//! are far too expensive to reproduce against a real one.

use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
};

use super::*;

/// Does nothing with the progress a download reports.
fn quiet() -> impl FnMut(&str, u64, Option<u64>) { |_, _, _| {} }

/// A body of `len` bytes that differs at every offset, so a file assembled out
/// of order is not mistaken for a correct one.
fn body(len: usize) -> Vec<u8> { (0..len).map(|i| (i % 251) as u8).collect() }

/// What the server does with one request.
#[derive(Clone, Copy, Debug)]
enum Reply {
    /// Serve what was asked for, closing the connection after this many bytes
    /// when `Some` — a transfer that dies mid-file.
    Serve(Option<usize>),
    /// Ignore the range and serve the whole file with `200`, the way a server
    /// answers when `If-Range` no longer matches.
    Whole,
    /// Refuse with this status.
    Status(u16),
}

/// One request the server saw.
#[derive(Clone, Debug)]
struct Seen {
    range: Option<String>,
    if_range: Option<String>,
}

/// A throwaway HTTP server serving one body over ranges.
struct Server {
    address: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Server {
    /// A server that always serves what it is asked for.
    fn new(body: Vec<u8>) -> Self { Self::scripted(body, Vec::new()) }

    /// A server that answers the first requests from `script` and serves
    /// normally after that.
    fn scripted(body: Vec<u8>, script: Vec<Reply>) -> Self {
        Self::start(body, script, Reply::Serve(None))
    }

    /// A server that answers every request the same way.
    fn always(body: Vec<u8>, reply: Reply) -> Self {
        Self::start(body, Vec::new(), reply)
    }

    fn start(body: Vec<u8>, script: Vec<Reply>, default: Reply) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address =
            format!("http://{}", listener.local_addr().expect("address"));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(VecDeque::from(script)));
        let body = Arc::new(body);
        let served = Arc::clone(&seen);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (seen, queue, body) = (
                    Arc::clone(&served),
                    Arc::clone(&queue),
                    Arc::clone(&body),
                );
                thread::spawn(move || {
                    let reply = queue
                        .lock()
                        .expect("queue")
                        .pop_front()
                        .unwrap_or(default);
                    handle(stream, &body, reply, &seen);
                });
            }
        });
        Self { address, seen }
    }

    fn url(&self) -> String { format!("{}/model.bin", self.address) }

    /// The requests seen so far, oldest first.
    fn seen(&self) -> Vec<Seen> { self.seen.lock().expect("seen").clone() }

    /// The `Range` header of every request that carried one.
    fn ranges(&self) -> Vec<String> {
        self.seen()
            .into_iter()
            .filter_map(|request| request.range)
            .collect()
    }
}

/// Serves one request.
fn handle(
    mut stream: TcpStream,
    body: &[u8],
    reply: Reply,
    seen: &Mutex<Vec<Seen>>,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request = Seen {
        range: None,
        if_range: None,
    };
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let line = line.trim_end().to_owned();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "range" => request.range = Some(value.trim().to_owned()),
                "if-range" => request.if_range = Some(value.trim().to_owned()),
                _ => {},
            }
        }
    }
    let range = request.range.as_deref().and_then(parse_range);
    seen.lock().expect("seen").push(request);

    let len = body.len();
    let head = |status: &str, extra: String| {
        format!(
            "HTTP/1.1 {status}\r\nETag: \"v1\"\r\nAccept-Ranges: \
             bytes\r\nConnection: close\r\n{extra}\r\n"
        )
    };
    let empty = || "Content-Length: 0\r\n".to_owned();
    match (reply, range) {
        (Reply::Status(code), _) => {
            let _ = stream
                .write_all(head(&format!("{code} No"), empty()).as_bytes());
        },
        (Reply::Whole, _) | (Reply::Serve(_), None) => {
            let _ = stream.write_all(
                head("200 OK", format!("Content-Length: {len}\r\n")).as_bytes(),
            );
            let _ = stream.write_all(body);
        },
        (Reply::Serve(cut), Some((start, end))) => {
            let end = end.unwrap_or(len - 1).min(len - 1);
            if start > end {
                let _ = stream.write_all(head("416 No", empty()).as_bytes());
                return;
            }
            let piece = &body[start..=end];
            let _ = stream.write_all(
                head(
                    "206 Partial Content",
                    format!(
                        "Content-Length: {}\r\nContent-Range: bytes \
                         {start}-{end}/{len}\r\n",
                        piece.len()
                    ),
                )
                .as_bytes(),
            );
            // Writing less than was promised and hanging up is what a transfer
            // dying mid-file looks like from the other side.
            let sent = cut.map_or(piece.len(), |cut| cut.min(piece.len()));
            let _ = stream.write_all(&piece[..sent]);
        },
    }
}

/// Parses `bytes=<first>-[<last>]`.
fn parse_range(value: &str) -> Option<(usize, Option<usize>)> {
    let (first, last) = value.strip_prefix("bytes=")?.split_once('-')?;
    Some((
        first.parse().ok()?,
        (!last.is_empty()).then(|| last.parse().ok()).flatten(),
    ))
}

/// Leaves behind what an interrupted run would have: the first `done` bytes of
/// the file, and the state describing them.
fn interrupted(target: &Path, url: &str, content: &[u8], done: usize) {
    fs::write(partial_path(target), &content[..done]).expect("partial");
    fs::write(
        state_path(target),
        format!(
            "url={url}\ntotal={0}\netag=\"v1\"\nslice=0,{0},{done}\n",
            content.len()
        ),
    )
    .expect("state");
}

// --- what a download leaves behind ----------------------------------------

#[test]
fn fetches_a_file_and_leaves_nothing_beside_it() {
    let server = Server::new(body(3000));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), body(3000));
    assert_eq!(
        fs::read_dir(dir.path()).expect("read_dir").count(),
        1,
        "no partial and no state should survive a finished download"
    );
}

#[test]
fn creates_the_directory_it_writes_into() {
    let server = Server::new(body(64));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("deep").join("nested").join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), body(64));
}

// --- splitting across connections -----------------------------------------

#[test]
fn splits_a_large_file_across_connections() {
    // `MIN_SLICE_BYTES` is 4 KiB under test, so 64 KiB is large enough to be
    // split as far as the connection count allows.
    let content = body(64 << 10);
    let server = Server::new(content.clone());
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), content);
    let each = (64 << 10) / CONNECTIONS;
    let mut wanted: Vec<String> = (0..CONNECTIONS)
        .map(|slice| {
            format!("bytes={}-{}", slice * each, (slice + 1) * each - 1)
        })
        .collect();
    // Plus the one-byte request that asked how long the file is.
    wanted.push("bytes=0-0".to_owned());
    wanted.sort();
    let mut seen = server.ranges();
    seen.sort();
    assert_eq!(seen, wanted);
}

#[test]
fn keeps_a_small_file_on_one_connection() {
    let server = Server::new(body(3000));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(server.ranges(), vec!["bytes=0-0", "bytes=0-2999"]);
}

// --- resuming -------------------------------------------------------------

#[test]
fn continues_where_the_last_run_stopped() {
    let content = body(3000);
    let server = Server::new(content.clone());
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");
    interrupted(&target, &server.url(), &content, 1000);

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), content);
    assert_eq!(
        server.ranges(),
        vec!["bytes=1000-2999"],
        "only the missing bytes should be asked for, and without the request \
         that asks how long the file is — the state already says"
    );
    assert_eq!(
        server.seen()[0].if_range.as_deref(),
        Some("\"v1\""),
        "a resumed range must be conditional on the file not having changed"
    );
}

#[test]
fn resumes_after_the_connection_drops() {
    let content = body(3000);
    // The probe, then a slice request that dies 500 bytes in.
    let server = Server::scripted(
        content.clone(),
        vec![Reply::Serve(None), Reply::Serve(Some(500))],
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), content);
    assert_eq!(
        server.ranges(),
        vec!["bytes=0-0", "bytes=0-2999", "bytes=500-2999"],
        "the second attempt should ask for the rest, not for the file again"
    );
}

#[test]
fn ignores_a_partial_left_by_another_download() {
    let content = body(3000);
    let server = Server::new(content.clone());
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");
    interrupted(
        &target,
        "http://elsewhere.invalid/model.bin",
        &content,
        1000,
    );

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), content);
    assert_eq!(
        server.ranges(),
        vec!["bytes=0-0", "bytes=0-2999"],
        "bytes of unknown origin are worth less than the time to re-fetch them"
    );
}

#[test]
fn starts_over_when_the_file_changed_on_the_server() {
    let content = body(3000);
    // The server answers the resumed range with the whole file, which is what
    // it does when `If-Range` no longer matches.
    let server = Server::scripted(content.clone(), vec![Reply::Whole]);
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");
    interrupted(&target, &server.url(), &content, 1000);

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(
        fs::read(&target).expect("read"),
        content,
        "the stale bytes must not be spliced into the new file"
    );
    assert_eq!(
        server.ranges(),
        vec!["bytes=1000-2999", "bytes=0-0", "bytes=0-2999"],
        "the retry should re-plan from scratch"
    );
}

// --- failures -------------------------------------------------------------

#[test]
fn retries_a_server_error() {
    let server = Server::scripted(body(3000), vec![Reply::Status(503)]);
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), body(3000));
}

#[test]
fn keeps_going_while_the_drops_keep_delivering() {
    // A server that hangs up 300 bytes into every response: ten drops to get
    // through the file, twice what a fixed budget of attempts would have
    // allowed. Each of them still brings the end closer, which is the whole
    // difference between a flaky link and a source that is gone.
    let content = body(3000);
    let server = Server::always(content.clone(), Reply::Serve(Some(300)));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect("fetch");

    assert_eq!(fs::read(&target).expect("read"), content);
    // The one-byte request that asks how long the file is, then one attempt per
    // drop, each picking up where the last was cut off.
    let mut wanted = vec!["bytes=0-0".to_owned()];
    wanted.extend((0..10).map(|drop| format!("bytes={}-2999", drop * 300)));
    assert!(
        wanted.len() - 1 > MAX_FRUITLESS_ATTEMPTS,
        "a test that stayed inside the budget would prove nothing"
    );
    assert_eq!(server.ranges(), wanted);
}

#[test]
fn gives_up_when_the_attempts_bring_nothing() {
    // Headers and then silence, every time: the budget is for exactly this, and
    // a replenishable one must still run out when there is nothing to replenish
    // it with.
    let server = Server::always(body(3000), Reply::Serve(Some(0)));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    let error = Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect_err("nothing ever arrives");

    assert!(matches!(error, DownloadError::Transfer { .. }), "{error}");
    assert_eq!(
        server.seen().len(),
        MAX_FRUITLESS_ATTEMPTS + 1,
        "five attempts that get nowhere, plus the one that asked how long the \
         file is"
    );
    assert!(!target.exists(), "nothing was downloaded");
}

#[test]
fn does_not_retry_a_missing_file() {
    let server = Server::always(body(3000), Reply::Status(404));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    let error = Download::new(&server.url(), &target)
        .fetch(&mut quiet())
        .expect_err("404 is not a hiccup");

    assert!(error.to_string().contains("404"), "{error}");
    assert_eq!(server.seen().len(), 1, "a missing file stays missing");
}

// --- digests --------------------------------------------------------------

#[test]
fn accepts_a_matching_digest() {
    let content = body(3000);
    let server = Server::new(content.clone());
    let dir = tempfile::tempdir().expect("tempdir");
    let blake3 = blake3::hash(&content).to_hex().to_string();
    let target = dir.path().join("model.bin");

    Download::new(&server.url(), &target)
        .blake3(&blake3)
        .fetch(&mut quiet())
        .expect("fetch");
    assert_eq!(fs::read(&target).expect("read"), content);

    let md5: String = Md5::digest(&content)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let other = dir.path().join("again.bin");
    Download::new(&server.url(), &other)
        .md5(&md5)
        .fetch(&mut quiet())
        .expect("fetch");
    assert_eq!(fs::read(&other).expect("read"), content);
}

#[test]
fn rejects_a_mismatched_digest_without_retrying() {
    let server = Server::new(body(3000));
    let dir = tempfile::tempdir().expect("tempdir");
    let target = dir.path().join("model.bin");

    let error = Download::new(&server.url(), &target)
        .blake3(&"0".repeat(64))
        .fetch(&mut quiet())
        .expect_err("the digest does not match");

    assert!(error.to_string().contains("checksum mismatch"), "{error}");
    assert!(
        !target.exists(),
        "a file that failed its digest is not a model"
    );
    assert!(
        !partial_path(&target).exists(),
        "and the bytes behind it are not worth keeping"
    );
    assert_eq!(server.seen().len(), 2, "one probe, one download, no retry");
}

// --- pieces ---------------------------------------------------------------

#[test]
fn splits_only_into_slices_worth_their_own_connection() {
    let bounds = |total: u64| -> Vec<(u64, Option<u64>)> {
        split(Some(total))
            .iter()
            .map(|slice| (slice.start, slice.end))
            .collect()
    };
    assert_eq!(bounds(100), vec![(0, Some(100))]);
    assert_eq!(
        bounds(2 * MIN_SLICE_BYTES),
        vec![
            (0, Some(MIN_SLICE_BYTES)),
            (MIN_SLICE_BYTES, Some(2 * MIN_SLICE_BYTES)),
        ]
    );

    let big = 100 * MIN_SLICE_BYTES;
    let slices = split(Some(big));
    assert_eq!(
        slices.len() as u64,
        CONNECTIONS,
        "a large file is split no further than the connection count"
    );
    assert_eq!(slices[0].start, 0);
    assert_eq!(slices.last().expect("last").end, Some(big));
    for pair in slices.windows(2) {
        assert_eq!(
            pair[0].end,
            Some(pair[1].start),
            "slices must meet without gaps or overlap"
        );
    }

    // A length the server would not state is a file that cannot be split.
    let unknown = split(None);
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].end, None);
}

#[test]
fn progress_hands_the_budget_back() {
    let mut budget = Budget::new();
    for attempt in 1..MAX_FRUITLESS_ATTEMPTS {
        assert!(
            budget.spend(false).is_some(),
            "attempt {attempt} of {MAX_FRUITLESS_ATTEMPTS} is not the last one"
        );
    }
    assert!(
        budget.spend(false).is_none(),
        "attempts that get nowhere still run out"
    );

    // …but only while they follow one another. An attempt that delivered starts
    // the count — and the waiting — over, however many drops came before it.
    let mut budget = Budget::new();
    for _ in 0..3 * MAX_FRUITLESS_ATTEMPTS {
        assert!(budget.spend(false).is_some());
        assert_eq!(
            budget.spend(true),
            Some(FIRST_BACKOFF),
            "progress hands back the pause as well as the count"
        );
    }
}

#[test]
fn the_pause_grows_while_nothing_arrives() {
    let mut budget = Budget::new();
    assert_eq!(budget.spend(false), Some(FIRST_BACKOFF));
    assert_eq!(budget.spend(false), Some(FIRST_BACKOFF * 2));
    assert_eq!(budget.spend(false), Some(FIRST_BACKOFF * 4));
}

#[test]
fn a_trickle_is_not_progress_but_the_last_kilobyte_is() {
    // A plan of `total` bytes that had `before` of them when the attempt
    // started and `gained` more when it failed.
    let advanced = |total: Option<u64>, before: u64, gained: u64| {
        let plan = Plan {
            url: "https://example.invalid/model.bin".to_owned(),
            total,
            etag: None,
            slices: vec![Slice::new(0, total)],
        };
        plan.slices[0]
            .done
            .store(before + gained, Ordering::Relaxed);
        plan.advanced(before)
    };

    let big = 1 << 30;
    assert!(
        !advanced(Some(big), 0, 0),
        "an attempt that brought nothing"
    );
    assert!(advanced(Some(big), 0, PROGRESS_BYTES), "a megabyte of it");
    assert!(!advanced(Some(big), 0, 64 << 10), "a trickle of it");

    // Near the end there can be less left to fetch than the bar itself, so the
    // bar comes down with the remainder: a kilobyte of the last ten counts,
    // while nothing still does not.
    assert!(advanced(Some(big), big - (10 << 10), 1 << 10));
    assert!(!advanced(Some(big), big - (10 << 10), 0));

    // A length the server would not state leaves only the flat bar.
    assert!(!advanced(None, 0, 64 << 10));
    assert!(advanced(None, 0, PROGRESS_BYTES));
}

#[test]
fn state_survives_a_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.bin.partial.state");
    let plan = Plan {
        url: "https://example.invalid/model.bin".to_owned(),
        total: Some(4096),
        etag: Some("\"abc\"".to_owned()),
        slices: vec![Slice::new(0, Some(2048)), Slice::new(2048, Some(4096))],
    };
    plan.slices[0].done.store(1024, Ordering::Relaxed);
    plan.write(&path);

    let read = Plan::read(&path).expect("state");
    assert_eq!(read.url, plan.url);
    assert_eq!(read.total, plan.total);
    assert_eq!(read.etag, plan.etag);
    assert_eq!(read.done(), 1024);
    assert_eq!(read.slices.len(), 2);
    assert_eq!(read.slices[1].start, 2048);
    assert_eq!(read.slices[1].end, Some(4096));
}

#[test]
fn state_that_cannot_be_trusted_is_no_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("state");
    assert!(Plan::read(&path).is_none(), "nothing is there at all");

    fs::write(&path, "url=https://example.invalid/x\n").expect("write");
    assert!(
        Plan::read(&path).is_none(),
        "a plan with no slices is no plan"
    );

    fs::write(&path, "total=10\nslice=0,10,0\n").expect("write");
    assert!(
        Plan::read(&path).is_none(),
        "slices of nothing in particular"
    );

    fs::write(&path, "url=u\nslice=0,10,seven\n").expect("write");
    assert!(Plan::read(&path).is_none(), "a line that does not parse");

    fs::write(&path, "url=u\nslice=0,10,99\n").expect("write");
    assert!(
        Plan::read(&path).is_none(),
        "more bytes than the slice covers"
    );
}

#[test]
fn partial_and_state_hang_off_the_whole_name() {
    let target = Path::new("/models/model.safetensors");
    assert_eq!(
        partial_path(target),
        Path::new("/models/model.safetensors.partial")
    );
    assert_eq!(
        state_path(target),
        Path::new("/models/model.safetensors.partial.state")
    );
    // Appended rather than substituted, so files that differ only in extension
    // cannot end up sharing one partial.
    assert_ne!(
        partial_path(target),
        partial_path(Path::new("/models/model.json"))
    );
}

#[test]
fn only_a_strong_etag_is_worth_keeping() {
    let etag = |value: &str| {
        let mut headers = header::HeaderMap::new();
        headers.insert(header::ETAG, value.parse().expect("header"));
        strong_etag(&headers)
    };
    assert_eq!(etag("\"abc\""), Some("\"abc\"".to_owned()));
    assert_eq!(
        etag("W/\"abc\""),
        None,
        "a weak validator cannot gate a range"
    );
    assert_eq!(etag(""), None);
    assert_eq!(strong_etag(&header::HeaderMap::new()), None);
}

#[test]
fn reads_the_length_out_of_a_content_range() {
    let total = |value: &str| {
        let mut headers = header::HeaderMap::new();
        headers.insert(header::CONTENT_RANGE, value.parse().expect("header"));
        content_range_total(&headers)
    };
    assert_eq!(total("bytes 0-0/1983"), Some(1983));
    assert_eq!(total("bytes 100-199/4096"), Some(4096));
    assert_eq!(
        total("bytes 0-0/*"),
        None,
        "a length the server will not state"
    );
    assert_eq!(content_range_total(&header::HeaderMap::new()), None);
}

#[test]
fn builds_a_hugging_face_url() {
    assert_eq!(
        hugging_face_url("openai/whisper-tiny", "config.json"),
        "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json"
    );
}
