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

/// Runs the binary in a specific working directory with extra env vars set —
/// used by skill-install tests so project/`--global` writes land in a tempdir.
fn run_in(dir: &Path, env: &[(&str, &str)], args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_trakktor"));
    command.current_dir(dir).args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("spawn trakktor")
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

    let out = run(&["--work-dir", &work_dir(dir.path()), "feed", "read", &url]);
    let value = stdout_json(&out);
    let items = value.as_array().expect("array");

    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["title"], "Post A");
    assert_eq!(items[1]["title"], "Post B");
    // uid is always present; minimal display = title,link, so is_read absent.
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
    let first = stdout_json(&run(&["--work-dir", &wd, "feed", "read", &url]));
    let uid = first[0]["uid"].as_str().unwrap().to_string();

    // Mark it read.
    let marked =
        stdout_json(&run(&["--work-dir", &wd, "feed", "mark-read", &uid]));
    assert_eq!(marked["marked"], 1);
    assert_eq!(marked["already_read"], 0);

    // Re-marking is idempotent.
    let again =
        stdout_json(&run(&["--work-dir", &wd, "feed", "mark-read", &uid]));
    assert_eq!(again["marked"], 0);
    assert_eq!(again["already_read"], 1);

    // Default read now hides the marked publication.
    let unread = stdout_json(&run(&["--work-dir", &wd, "feed", "read", &url]));
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

    let via_feed =
        stdout_json(&run(&["--work-dir", &wd, "feed", "read", &feed_url]));
    let via_page =
        stdout_json(&run(&["--work-dir", &wd, "feed", "read", &page_url]));

    // feed_key rule: reading the feed directly and via its page must yield
    // identical uids.
    assert_eq!(via_feed, via_page);
    assert_eq!(via_page.as_array().unwrap().len(), 2);
}

#[test]
fn discover_lists_declared_feed() {
    let port = server();
    let page_url = format!("http://127.0.0.1:{port}/");

    let out = run(&["feed", "discover", &page_url]);
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
        "--text",
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
    ]);
    assert_eq!(stderr_error_code(&out), "invalid_url");
}

#[test]
fn missing_required_argument_exits_two() {
    // Structural argument error → clap → exit code 2.
    let out = run(&["feed", "mark-read"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn feed_read_keeps_uid_when_fields_narrowed() {
    let port = server();
    let url = format!("http://127.0.0.1:{port}/feed.xml");
    let dir = tempfile::tempdir().unwrap();
    // Narrowing the display fields must not drop the primary key.
    let value = stdout_json(&run(&[
        "--work-dir",
        &work_dir(dir.path()),
        "feed",
        "read",
        &url,
        "--fields",
        "title",
    ]));
    for item in value.as_array().unwrap() {
        assert!(item["uid"].is_string(), "uid dropped: {item}");
        assert!(item["title"].is_string());
    }
}

#[test]
fn feed_read_text_mode_hints_at_mark_read() {
    let port = server();
    let url = format!("http://127.0.0.1:{port}/feed.xml");
    let dir = tempfile::tempdir().unwrap();
    let wd = work_dir(dir.path());

    // Text mode: a mark-read hint naming uid as the key goes to stderr.
    let text = run(&["--work-dir", &wd, "feed", "read", &url, "--text"]);
    let stderr = String::from_utf8_lossy(&text.stderr);
    assert!(stderr.contains("mark-read"), "stderr: {stderr}");
    assert!(stderr.contains("uid"), "stderr: {stderr}");

    // Default (JSON) mode: stdout is a clean array and stderr has no hint.
    let json = run(&["--work-dir", &wd, "feed", "read", &url]);
    assert!(serde_json::from_slice::<Value>(&json.stdout).is_ok());
    assert!(String::from_utf8_lossy(&json.stderr).is_empty());
}

// ---------------------------------------------------------------------------
// skill
// ---------------------------------------------------------------------------

#[test]
fn skill_show_prints_narrative_guide() {
    let out = run(&["skill", "show", "--text"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.starts_with("# trakktor"), "got: {text}");
    assert!(text.contains("## Commands"));
    assert!(text.contains("### trakktor feed"));
    // Narrative only — no reference without --full.
    assert!(!text.contains("## Reference"));
}

#[test]
fn skill_show_full_includes_generated_reference() {
    let out = run(&["skill", "show", "--full", "--text"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("## Reference"));
    assert!(text.contains("### trakktor feed read"));
    // Flags, fixed values, and defaults are generated from clap.
    assert!(text.contains("`--fields <list>`"));
    assert!(text.contains("default: minimal"));
    assert!(text.contains("values: claude, agents"));
}

#[test]
fn skill_show_json_wraps_markdown() {
    let out = run(&["skill", "show"]);
    let value = stdout_json(&out);
    let content = value["content"].as_str().expect("content string");
    assert!(content.starts_with("# trakktor"));
    // The object carries only the prose document.
    assert_eq!(value.as_object().unwrap().len(), 1);
}

#[test]
fn skill_install_writes_stub_to_project() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_in(dir.path(), &[], &["skill", "install", "claude"]);
    let value = stdout_json(&out);
    assert_eq!(value["path"], "./.claude/skills/trakktor/SKILL.md");
    assert_eq!(value["status"], "written");

    let path = dir.path().join(".claude/skills/trakktor/SKILL.md");
    let stub = std::fs::read_to_string(&path).expect("stub written");
    assert!(stub.contains("name: trakktor"));
    assert!(stub.contains("allowed-tools: Bash(trakktor:*)"));
    // The stub points back at the binary; it must not embed the reference.
    assert!(stub.contains("trakktor skill show"));
    assert!(!stub.contains("--fields"));
}

#[test]
fn skill_install_skips_existing_until_forced() {
    let dir = tempfile::tempdir().unwrap();
    let args = ["skill", "install", "agents"];

    let first = stdout_json(&run_in(dir.path(), &[], &args));
    assert_eq!(first["path"], "./.agents/skills/trakktor/SKILL.md");
    assert_eq!(first["status"], "written");

    // Re-running without --force leaves the file and reports it skipped.
    let second = stdout_json(&run_in(dir.path(), &[], &args));
    assert_eq!(second["path"], "./.agents/skills/trakktor/SKILL.md");
    assert_eq!(second["status"], "skipped");

    // --force overwrites.
    let forced_args = ["skill", "install", "agents", "--force"];
    let forced = stdout_json(&run_in(dir.path(), &[], &forced_args));
    assert_eq!(forced["status"], "written");
}

#[test]
fn skill_install_global_writes_when_claude_dir_exists() {
    let project = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let home_str = home.path().to_str().unwrap();
    // The Claude home directory must already exist; we never create it.
    std::fs::create_dir(home.path().join(".claude")).unwrap();

    let out = run_in(
        project.path(),
        &[("HOME", home_str)],
        &["skill", "install", "claude", "--global"],
    );
    let value = stdout_json(&out);
    let installed = value["path"].as_str().unwrap();
    assert!(installed.starts_with(home_str), "got: {installed}");
    assert_eq!(value["status"], "written");

    let path = home.path().join(".claude/skills/trakktor/SKILL.md");
    assert!(path.exists(), "stub should be under HOME");
    // Nothing leaked into the project directory.
    assert!(!project.path().join(".claude").exists());
}

#[test]
fn skill_install_global_errors_when_claude_dir_missing() {
    let project = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap(); // deliberately has no `.claude`
    let home_str = home.path().to_str().unwrap();

    let out = run_in(
        project.path(),
        &[("HOME", home_str)],
        &["skill", "install", "claude", "--global"],
    );
    // We refuse to create the agent's home directory.
    assert_eq!(stderr_error_code(&out), "agent_dir_missing");
    // Nothing was written anywhere.
    assert!(!home.path().join(".claude").exists());
    assert!(!project.path().join(".claude").exists());
}

#[test]
fn skill_install_agents_global_is_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let out =
        run_in(dir.path(), &[], &["skill", "install", "agents", "--global"]);
    // `--global` is Claude-only; clap reports a usage error (exit 2).
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("global"), "stderr: {stderr}");
    // Nothing was written.
    assert!(!dir.path().join(".agents").exists());
}

#[test]
fn skill_install_requires_a_target() {
    let out = run(&["skill", "install"]);
    // The destination must be explicit; omitting it is a usage error (exit 2).
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("target"), "stderr: {stderr}");
}

#[test]
fn no_internal_doc_references_leak_into_user_facing_text() {
    // `skill show` is reproduced verbatim for coding agents and `--help` is
    // read by users; neither may expose internal design-doc identifiers (these
    // live in the docs repo and in `//` code comments, never in clap prose).
    let surfaces: &[&[&str]] = &[
        &["skill", "show", "--full"],
        &["--help"],
        &["skill", "show", "--help"],
        &["skill", "install", "--help"],
        &["text", "--help"],
        &["text", "stress", "--help"],
        &["tts", "espeech", "--help"],
        &["convert", "--help"],
        &["convert", "pdf", "--help"],
        &["enhance", "--help"],
        &["enhance", "mpsenet", "--help"],
        &["enhance", "resemble-denoise", "--help"],
        &["enhance", "resemble-enhance", "--help"],
    ];
    let needles = [
        "ADR-",
        "design.md",
        "output.md",
        "cli.md",
        "error-handling",
        "working-directory",
        "research.md",
        "§",
    ];
    for args in surfaces {
        let out = run(args);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        for needle in needles {
            assert!(
                !text.contains(needle),
                "internal reference {needle:?} leaked into `trakktor {}`",
                args.join(" ")
            );
        }
    }
}

#[test]
fn asr_start_conflicts_with_clip_timestamps() {
    // `--start`/`--end` and `--clip-timestamps` set the same clips; clap
    // reports the conflict as a usage error (exit 2) before any work.
    let out = run(&[
        "asr",
        "whisper",
        "audio.mp3",
        "--start",
        "1",
        "--clip-timestamps",
        "5",
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn asr_rejects_malformed_timecode() {
    // A `--start` that is neither seconds nor a clock is a usage error.
    let out = run(&["asr", "whisper", "audio.mp3", "--start", "1:2:3:4"]);
    assert_eq!(out.status.code(), Some(2));
}

// ---------------------------------------------------------------------------
// text stress
// ---------------------------------------------------------------------------

/// Writes `text` to a file in `dir` and returns its path as a string.
fn text_file(dir: &Path, name: &str, text: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, text).expect("write");
    path.to_str().expect("utf-8 path").to_string()
}

/// A sentence with a homograph (`замки`), a hidden `ё` (`Королев`), and words
/// of one syllable — everything the operation has to decide about.
const SAMPLE: &str =
    "Меня зовут Лева Королев. Я уже готов открыть все ваши замки.";

/// The marked form of [`SAMPLE`], as both runtimes produce it.
const MARKED: &str =
    "Мен+я зов+ут Л+ёва Корол+ёв. +Я уж+е гот+ов откр+ыть вс+е в+аши замк+и.";

#[test]
#[ignore = "needs the model in ~/.trakktor/text/stress"]
fn text_stress_marks_the_text_on_both_runtimes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);

    // The choice of runtime is a choice of arithmetic, not of answers.
    let runtimes: &[&str] = if cfg!(feature = "burn") {
        &["candle", "burn"]
    } else {
        &["candle"]
    };
    for runtime in runtimes {
        let out = run(&["text", "stress", &input, "--runtime", runtime]);
        let json = stdout_json(&out);
        assert_eq!(json["text"].as_str().unwrap(), MARKED, "runtime {runtime}");
        assert_eq!(json["model"].as_str().unwrap(), "silero-ru");
        assert_eq!(json["marker"].as_str().unwrap(), "plus");
        assert_eq!(json["stats"]["homographs"].as_u64().unwrap(), 6);
        assert_eq!(json["stats"]["yo_restored"].as_u64().unwrap(), 2);
        assert!(json["unstressed"].is_array());
    }
}

#[test]
#[ignore = "needs the model in ~/.trakktor/text/stress"]
fn text_stress_prints_the_marked_text_alone_with_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let out = run(&["text", "stress", &input, "--text"]);
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), MARKED);
}

#[test]
#[ignore = "needs the model in ~/.trakktor/text/stress"]
fn text_stress_reads_back_its_own_acute_output() {
    // The two mark forms say the same thing, so the operation accepts either.
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let acute = run(&["text", "stress", &input, "--text", "--marker", "acute"]);
    assert!(acute.status.success());
    let acute = String::from_utf8_lossy(&acute.stdout).to_string();
    assert!(acute.contains('\u{301}'), "no acute in {acute:?}");
    assert!(!acute.contains('+'), "a plus survived into {acute:?}");

    let back = text_file(dir.path(), "acute.txt", &acute);
    let out = run(&["text", "stress", &back, "--text"]);
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), MARKED);
}

#[test]
#[ignore = "needs the model in ~/.trakktor/text/stress"]
fn text_stress_with_yo_off_only_adds_marks() {
    // The flag's whole promise: your letters come back as you wrote them.
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let out = run(&["text", "stress", &input, "--text", "--yo", "off"]);
    let marked = String::from_utf8_lossy(&out.stdout).trim_end().to_string();
    let stripped: String = marked.chars().filter(|c| *c != '+').collect();
    assert_eq!(stripped, SAMPLE);
}

#[test]
#[ignore = "needs the model in ~/.trakktor/text/stress"]
fn text_stress_takes_the_users_spelling_over_the_models() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let dictionary = text_file(dir.path(), "dict.txt", "# mine\nз+амки\n");
    let out = run(&["text", "stress", &input, "--dict", &dictionary]);
    let json = stdout_json(&out);
    let marked = json["text"].as_str().unwrap();
    assert!(marked.contains("з+амки"), "{marked}");
    assert_eq!(json["stats"]["from_dictionary"].as_u64().unwrap(), 1);
    // The word is spoken for, so the homograph solver never sees it.
    assert_eq!(json["stats"]["homographs"].as_u64().unwrap(), 5);
}

#[test]
fn text_stress_rejects_a_malformed_dictionary_before_touching_the_model() {
    // The dictionary is the caller's own file, so a typo in it is reported as
    // theirs — and before anything is downloaded.
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let dictionary = text_file(dir.path(), "dict.txt", "hello\n");
    let out = run(&["text", "stress", &input, "--dict", &dictionary]);
    assert_eq!(stderr_error_code(&out), "invalid_options");
}

#[test]
fn text_stress_reports_an_unknown_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "in.txt", SAMPLE);
    let out = run(&["text", "stress", &input, "--model", "nonesuch"]);
    assert_eq!(stderr_error_code(&out), "model_unavailable");
}

#[test]
fn text_stress_reports_a_missing_input_file() {
    let out = run(&["text", "stress", "no-such-file.txt"]);
    assert_eq!(stderr_error_code(&out), "io_error");
}

// ---------------------------------------------------------------------------
// convert pdf
// ---------------------------------------------------------------------------

/// Writes a PDF of `pages` into `dir` and returns its path.
///
/// Each page is a list of lines set in Helvetica; an empty list makes a page
/// with no content stream at all, which is what a page with nothing to read
/// looks like from the outside.
fn pdf_file(dir: &Path, name: &str, pages: &[&[&str]]) -> String {
    let path = dir.join(name);
    std::fs::write(&path, build_pdf(pages)).expect("write the test PDF");
    path.to_str().expect("utf-8 path").to_string()
}

/// Assembles a PDF by hand: the objects in order, then a cross-reference table
/// of where each of them landed.
fn build_pdf(pages: &[&[&str]]) -> Vec<u8> {
    // 1 catalog, 2 page tree, 3 font, then two objects per page (the page and
    // its content stream, whether or not the stream is used).
    let page_id = |at: usize| 4 + at * 2;
    let kids: Vec<String> = (0..pages.len())
        .map(|at| format!("{} 0 R", page_id(at)))
        .collect();

    let mut objects: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            kids.join(" "),
            pages.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    for (at, lines) in pages.iter().enumerate() {
        let contents = if lines.is_empty() {
            String::new()
        } else {
            format!(" /Contents {} 0 R", page_id(at) + 1)
        };
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources \
             << /Font << /F1 3 0 R >> >>{contents} >>"
        ));
        let stream: String = lines
            .iter()
            .enumerate()
            .map(|(line, text)| {
                format!(
                    "BT /F1 12 Tf 72 {} Td ({text}) Tj ET\n",
                    720 - 20 * line
                )
            })
            .collect();
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}endstream",
            stream.len()
        ));
    }

    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (at, object) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(
            format!("{} 0 obj\n{object}\nendobj\n", at + 1).as_bytes(),
        );
    }
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1)
            .as_bytes(),
    );
    for offset in &offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    out
}

/// Four lines of prose, which is what it takes for a page to count as carrying
/// text rather than a stray mark.
fn prose(which: &str) -> Vec<String> {
    (1..=4)
        .map(|line| format!("This is the {which} page, line {line} of prose."))
        .collect()
}

#[test]
fn convert_pdf_returns_pages_and_the_assembled_markdown() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = prose("first");
    let second = prose("second");
    let first: Vec<&str> = first.iter().map(String::as_str).collect();
    let second: Vec<&str> = second.iter().map(String::as_str).collect();
    let input = pdf_file(dir.path(), "doc.pdf", &[&first, &second]);

    let out = run(&["convert", "pdf", &input]);
    let value = stdout_json(&out);
    assert_eq!(value["source"], "doc.pdf");
    assert_eq!(value["kind"], "text");
    assert_eq!(value["page_count"], 2);
    assert_eq!(value["pages"].as_array().unwrap().len(), 2);
    assert_eq!(value["pages"][0]["number"], 1);
    assert!(
        value["pages"][0]["markdown"]
            .as_str()
            .unwrap()
            .contains("first page"),
        "{value}"
    );
    // Nothing went wrong, so none of the trouble fields are there at all.
    assert!(value.get("pages_needing_ocr").is_none(), "{value}");
    assert!(value.get("issues").is_none(), "{value}");
    assert!(value.get("fonts_repaired").is_none(), "{value}");
    assert!(
        value["markdown"]
            .as_str()
            .unwrap()
            .contains("<!-- page 2 -->"),
        "{value}"
    );
}

#[test]
fn convert_pdf_text_prints_the_markdown_and_out_writes_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let lines = prose("only");
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    let input = pdf_file(dir.path(), "doc.pdf", &[&lines]);
    let output = dir.path().join("doc.md");

    let out = run(&[
        "convert",
        "pdf",
        &input,
        "--text",
        "--out",
        output.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{:?}", out.status.code());
    let printed = String::from_utf8_lossy(&out.stdout);
    assert!(printed.contains("only page"), "{printed}");
    let written = std::fs::read_to_string(&output).expect("the --out file");
    assert_eq!(written, printed);
}

#[test]
fn convert_pdf_selects_pages_by_the_documents_own_numbers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pages: Vec<Vec<String>> = ["first", "second", "third"]
        .iter()
        .map(|w| prose(w))
        .collect();
    let borrowed: Vec<Vec<&str>> = pages
        .iter()
        .map(|lines| lines.iter().map(String::as_str).collect())
        .collect();
    let refs: Vec<&[&str]> = borrowed.iter().map(Vec::as_slice).collect();
    let input = pdf_file(dir.path(), "doc.pdf", &refs);

    let value = stdout_json(&run(&["convert", "pdf", &input, "--pages", "2"]));
    assert_eq!(value["page_count"], 3, "the document is still three pages");
    let selected = value["pages"].as_array().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0]["number"], 2);
    assert!(
        selected[0]["markdown"]
            .as_str()
            .unwrap()
            .contains("second page"),
        "{value}"
    );
}

#[test]
fn convert_pdf_refuses_a_document_with_no_text_and_names_ocr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = pdf_file(dir.path(), "blank.pdf", &[&[], &[], &[]]);
    let out = run(&["convert", "pdf", &input]);
    assert_eq!(stderr_error_code(&out), "no_text_layer");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("trakktor ocr"), "{stderr}");
}

#[test]
fn convert_pdf_reports_a_file_that_is_not_a_pdf() {
    let dir = tempfile::tempdir().expect("tempdir");
    let input = text_file(dir.path(), "notes.txt", "just some notes\n");
    assert_eq!(
        stderr_error_code(&run(&["convert", "pdf", &input])),
        "invalid_input"
    );
}

#[test]
fn convert_pdf_reports_a_missing_file() {
    assert_eq!(
        stderr_error_code(&run(&["convert", "pdf", "no-such-file.pdf"])),
        "invalid_input"
    );
}

#[test]
fn convert_pdf_rejects_a_page_range_it_cannot_honour() {
    let dir = tempfile::tempdir().expect("tempdir");
    let lines = prose("only");
    let lines: Vec<&str> = lines.iter().map(String::as_str).collect();
    let input = pdf_file(dir.path(), "doc.pdf", &[&lines]);

    // Not a range at all…
    assert_eq!(
        stderr_error_code(&run(&["convert", "pdf", &input, "--pages", "abc"])),
        "invalid_options"
    );
    // …and a range the document is too short for.
    assert_eq!(
        stderr_error_code(&run(&["convert", "pdf", &input, "--pages", "7-9"])),
        "invalid_options"
    );
}

#[test]
fn convert_pdf_requires_a_file() {
    // The document is the whole argument list; omitting it is a usage error.
    assert_eq!(run(&["convert", "pdf"]).status.code(), Some(2));
}
