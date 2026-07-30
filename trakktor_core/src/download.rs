//! The one downloader for model weights and other large assets.
//!
//! Everything trakktor fetches that is not a feed comes through here:
//! checkpoints, tokenizers, vocabularies, archives. A gigabyte-sized file over
//! a home connection needs four properties that a plain `GET` into a file does
//! not have, and this is where they live once instead of once per engine.
//!
//! **Nothing half-finished ever wears the final name.** Bytes stream into a
//! sibling `<name>.partial`; the target appears only when the file is whole and
//! matches its digest, if the caller published one. An interrupted run leaves a
//! partial the next run knows how to continue, never a truncated checkpoint
//! that looks ready to load.
//!
//! **What was downloaded stays downloaded.** A partial is resumed with range
//! requests rather than restarted. Beside it sits a small state file recording
//! where those bytes came from, how far each connection got, and the server's
//! `ETag` for them — so the resume can be *verified*: `If-Range` makes the
//! server send the whole file instead of a range when its copy no longer
//! matches, and a partial belonging to another URL is discarded rather than
//! spliced in. Resuming a 4 GB checkpoint that broke at 90 % costs the missing
//! 10 %, not another hour.
//!
//! **A hiccup is not a failure.** A dropped connection, a stalled transfer, a
//! 5xx or a 429 is retried with exponential backoff, each attempt continuing
//! from what already arrived. A 404 or a digest mismatch on a fresh download is
//! not retried — repeating those only fails the same way. What the retries are
//! rationed by is attempts that get *nowhere*: an attempt that brought a real
//! piece of the file in hands the budget back, so a checkpoint crossing a link
//! that drops every half hour arrives instead of running out of tries while
//! plainly working.
//!
//! **A big file is fetched over several connections.** See below; this is the
//! one property that was measured before it was believed.
//!
//! Progress is reported through [`Progress`], the single interface every caller
//! takes from here rather than restating by convention.
//!
//! ## Why more than one connection
//!
//! Splitting a file across parallel range requests is the standard lever when a
//! single connection is the limit rather than the link, and it is just as
//! standard a way to add complexity for nothing when the link is the limit. So
//! it was measured first — the same 48 MiB span of a Hugging Face checkpoint
//! every time, runs alternating between counts across three sessions, speed
//! averaged over the whole transfer rather than sampled from a moment of it:
//!
//! | Connections | Average speed, MB/s |
//! |---|---|
//! | 1 | 0.32, 0.36, 0.37 |
//! | 2 | 0.97, 0.79, 0.77, 0.42 |
//! | 4 | 1.16, 1.05, 1.01, 0.87 |
//! | 8 | 1.17, 1.05, 0.86 |
//!
//! The link was not the limit: four connections move the same bytes about
//! **three times faster** than one, which turns an hour into twenty minutes.
//! Eight buy nothing beyond four, and two are erratic — so a file is split
//! [`CONNECTIONS`] ways, but only when the server offers ranges and the file is
//! large enough for the slices to be worth their own connection. Everything
//! else — a small file, an unknown length, a server without ranges — is one
//! connection, which is also the shape all of this degrades to.
//!
//! Measured on one home link against one host, where the spread between repeats
//! is wide enough that only the gap between one connection and four is solid.
//! On a link fast enough to saturate by itself the extra connections cost
//! nothing but their own setup.

#[cfg(test)]
mod tests;

use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use md5::{Digest as _, Md5};
use reqwest::{StatusCode, header};

use crate::http::USER_AGENT;

/// How a caller watches a download advance: the label of the file being
/// fetched, how many bytes are done, and the total when the server states it.
///
/// This is the whole progress contract between the core and whatever draws it.
pub type Progress<'a> = &'a mut (dyn FnMut(&str, u64, Option<u64>) + 'a);

/// How many connections a large file is split across. Measured, not guessed —
/// see the module documentation.
pub const CONNECTIONS: u64 = 4;

/// The smallest slice worth its own connection: a file is split only as far as
/// this allows, so small files keep to one.
///
/// Tests shrink it so the parallel path can be exercised without moving tens of
/// megabytes.
#[cfg(not(test))]
const MIN_SLICE_BYTES: u64 = 16 << 20;
#[cfg(test)]
const MIN_SLICE_BYTES: u64 = 4 << 10;

/// How many attempts in a row may get nowhere before the download gives up.
///
/// The budget counts attempts that bring nothing, not attempts: a
/// multi-gigabyte checkpoint over a link that drops every half hour needs a
/// dozen resumes to arrive, and each of them delivers hundreds of megabytes.
/// Spending the budget on those would fail a download that is plainly working,
/// while a source that is genuinely gone still runs out of tries just as fast.
const MAX_FRUITLESS_ATTEMPTS: usize = 5;

/// How much an attempt has to bring in to count as getting somewhere and hand
/// the budget back.
///
/// A megabyte is far more than a source that is stuck manages and far less than
/// a flaky link delivers between drops, so the two separate without having to
/// know anything about the link.
const PROGRESS_BYTES: u64 = 1 << 20;

/// The share of what is still missing that counts as getting somewhere too, one
/// part in this many.
///
/// It is what carries the rule into the tail of a file, where there can be less
/// left to fetch than [`PROGRESS_BYTES`] altogether and a bar that did not come
/// down with the remainder would call the last stretch a standstill.
const PROGRESS_SHARE: u64 = 100;

/// The pause after the first failed attempt; it doubles up to [`MAX_BACKOFF`]
/// and starts over whenever an attempt gets somewhere.
///
/// Tests shrink it: what they check is which attempts happen, not how long the
/// waiting between them lasts.
#[cfg(not(test))]
const FIRST_BACKOFF: Duration = Duration::from_secs(1);
#[cfg(test)]
const FIRST_BACKOFF: Duration = Duration::from_millis(1);

/// The longest pause between attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// How long to wait for a connection to be established.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a transfer may deliver nothing at all before it counts as stalled.
///
/// The blocking client applies its timeout to each read of the body rather than
/// to the response as a whole, so this bounds *silence*, not the size of a
/// download: an hour-long transfer is fine, a minute of nothing is not — and
/// since a stall is retryable, it ends in a resume instead of a hang.
const STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// How often the connections' progress is collected and reported.
const POLL: Duration = Duration::from_millis(20);

/// How often the state file is refreshed while a download runs. It only ever
/// costs a re-download of what it does not yet know about, so it is written
/// rarely rather than on every chunk.
const PERSIST_EVERY: Duration = Duration::from_secs(5);

/// The streaming buffer.
const BUFFER_BYTES: usize = 1 << 20;

/// The URL a file in a Hugging Face repository resolves to.
///
/// `resolve/main` follows the repository's main branch and redirects to
/// whichever host serves the bytes. That redirect is signed for the exact byte
/// range it was asked for, so every request — a resumed one, or one slice of a
/// split file — goes through this URL rather than reusing what it was
/// redirected to before.
#[must_use]
pub fn hugging_face_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Errors a download can fail with.
///
/// Callers map these into their own error type; the text is what the user
/// reads, so it names the stage and the file rather than the internals.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// The HTTP client could not be built.
    #[error("building the http client: {0}")]
    Client(String),

    /// The request could not be sent, or the server refused it.
    #[error("requesting {url}: {detail}")]
    Request {
        /// The URL that was asked for.
        url: String,
        /// What went wrong.
        detail: String,
    },

    /// The transfer broke down while the body was being read.
    #[error("downloading {url}: {detail}")]
    Transfer {
        /// The URL being read from.
        url: String,
        /// What went wrong.
        detail: String,
    },

    /// A local filesystem operation failed.
    #[error("{action} {path}: {detail}")]
    File {
        /// What was being done: `creating`, `writing`, `verifying` or
        /// `finalizing`.
        action: &'static str,
        /// The path it was being done to.
        path: String,
        /// What went wrong.
        detail: String,
    },

    /// The finished file does not match the digest the caller published.
    #[error("checksum mismatch for {path}: expected {expected}, got {got}")]
    Checksum {
        /// The file that was verified.
        path: String,
        /// The digest the caller expects.
        expected: String,
        /// The digest the bytes actually have.
        got: String,
    },
}

/// A file to fetch: where from, where to, and what proves it arrived intact.
///
/// ```no_run
/// # use std::path::Path;
/// # use trakktor_core::download::{self, Download, Progress};
/// # fn f(progress: Progress<'_>) -> Result<(), Box<dyn std::error::Error>> {
/// let url = download::hugging_face_url("openai/whisper-tiny", "config.json");
/// Download::new(&url, Path::new("/models/config.json")).fetch(progress)?;
/// # Ok(()) }
/// ```
pub struct Download<'a> {
    url: &'a str,
    target: &'a Path,
    label: Option<&'a str>,
    digest: Option<Digest<'a>>,
}

/// The digest a finished file must match, for callers that publish one.
#[derive(Debug, Clone, Copy)]
enum Digest<'a> {
    /// A BLAKE3 hex digest.
    Blake3(&'a str),
    /// An MD5 hex digest, for sources that publish only that.
    Md5(&'a str),
}

impl<'a> Download<'a> {
    /// A download of `url` into `target`.
    ///
    /// The target's parent directory is created if it does not exist.
    #[must_use]
    pub fn new(url: &'a str, target: &'a Path) -> Self {
        Self {
            url,
            target,
            label: None,
            digest: None,
        }
    }

    /// Names the file in progress reports.
    ///
    /// Defaults to the target's file name; set it when that would be ambiguous
    /// — a file inside a subdirectory of a checkpoint, or one whose name says
    /// nothing about which model it belongs to.
    #[must_use]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Requires the finished file to match this BLAKE3 hex digest.
    #[must_use]
    pub fn blake3(mut self, hex: &'a str) -> Self {
        self.digest = Some(Digest::Blake3(hex));
        self
    }

    /// Requires the finished file to match this MD5 hex digest.
    #[must_use]
    pub fn md5(mut self, hex: &'a str) -> Self {
        self.digest = Some(Digest::Md5(hex));
        self
    }

    /// Fetches the file: continuing a partial from an earlier run, retrying
    /// what is worth retrying — for as long as the retries keep bringing the
    /// file closer to done — and moving the result onto its final name only
    /// once it is whole.
    ///
    /// A target that already exists is **not** checked — deciding that a model
    /// is already there is the caller's job, which knows what makes it
    /// complete.
    ///
    /// # Errors
    ///
    /// Returns [`DownloadError`] when the file could not be fetched: the server
    /// refused the request, attempt after attempt brought nothing, the digest
    /// did not match, or a local write failed.
    pub fn fetch(self, progress: Progress<'_>) -> Result<(), DownloadError> {
        let label = self.label.unwrap_or_else(|| {
            self.target
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("file")
        });
        if let Some(parent) = self.target.parent() {
            create_dir(parent)?;
        }

        let mut budget = Budget::new();
        loop {
            match self.attempt(label, &mut *progress) {
                Ok(()) => return Ok(()),
                Err(setback) => {
                    if matches!(setback.failure.fault, Fault::Permanent) {
                        return Err(setback.failure.error);
                    }
                    let Some(pause) = budget.spend(setback.progressed) else {
                        return Err(setback.failure.error);
                    };
                    thread::sleep(pause);
                },
            }
        }
    }

    /// One attempt: work out what is still missing, fetch it, verify the
    /// result, and move it into place.
    fn attempt(
        &self,
        label: &str,
        progress: Progress<'_>,
    ) -> Result<(), Setback> {
        let partial = partial_path(self.target);
        let state = state_path(self.target);

        // A partial is only usable together with the state describing it: its
        // slices land out of order, so the file's length says nothing about how
        // much of it is real.
        let resumed = Plan::read(&state)
            .filter(|plan| plan.url == self.url && partial.is_file());
        let plan = match resumed {
            Some(plan) => plan,
            None => {
                discard(&partial, &state);
                self.plan()?
            },
        };
        // What is already on disk: the mark this attempt's own progress is
        // measured against.
        let before = plan.done();
        plan.write(&state);

        let outcome = self.run(&plan, &partial, &state, label, progress);
        plan.write(&state);
        if let Some(failure) = outcome {
            // A restart throws away everything that arrived, so however much
            // that was, the file is no closer to done than it was.
            if matches!(failure.fault, Fault::Restart) {
                discard(&partial, &state);
                return Err(failure.into());
            }
            return Err(Setback {
                progressed: plan.advanced(before),
                failure,
            });
        }

        // Every connection reported it was done, so a length that still
        // disagrees means the file on disk is not the one that was planned.
        if let Some(total) = plan.total {
            let size = partial.metadata().map(|meta| meta.len()).unwrap_or(0);
            if size != total {
                discard(&partial, &state);
                return Err(self
                    .fault(
                        Fault::Restart,
                        "reading",
                        &format!("got {size} bytes of {total}; starting over"),
                    )
                    .into());
            }
        }

        if let Some(digest) = self.digest &&
            let Err(error) = verify(&partial, digest)
        {
            discard(&partial, &state);
            // Bytes carried over from an earlier run can be stale, and starting
            // over fixes that. When there were none to carry over, the source
            // or the transfer is broken instead, and repeating would only fail
            // the same way.
            return Err(Failure {
                fault: if before == 0 {
                    Fault::Permanent
                } else {
                    Fault::Transient
                },
                error,
            }
            .into());
        }

        let _ = fs::remove_file(&state);
        fs::rename(&partial, self.target)
            .map_err(|e| local(self.target, "finalizing", &e))?;
        Ok(())
    }

    /// Asks the server the three questions a plan needs, in one small request:
    /// how long the file is, whether it serves ranges, and which validator it
    /// will recognise later.
    fn plan(&self) -> Result<Plan, Failure> {
        let response = client()
            .map_err(|error| Failure {
                fault: Fault::Permanent,
                error,
            })?
            .get(self.url)
            .header(header::RANGE, "bytes=0-0")
            .send()
            .map_err(|e| {
                self.fault(Fault::Transient, "requesting", &e.to_string())
            })?;
        let status = response.status();
        if !status.is_success() {
            return Err(self.status_fault(status));
        }
        let ranged = status == StatusCode::PARTIAL_CONTENT;
        let total = if ranged {
            content_range_total(response.headers())
        } else {
            response.content_length()
        };
        let etag = strong_etag(response.headers());
        // What a server that ignored the range is sending is the whole file;
        // dropping the response closes the connection instead of reading it.
        drop(response);

        Ok(Plan {
            url: self.url.to_owned(),
            total,
            etag,
            slices: split(total.filter(|_| ranged)),
        })
    }

    /// Runs the plan's unfinished slices, one connection each, reporting
    /// progress until they are all done. Returns the failure worth reporting,
    /// if any.
    fn run(
        &self,
        plan: &Plan,
        partial: &Path,
        state: &Path,
        label: &str,
        progress: Progress<'_>,
    ) -> Option<Failure> {
        // The workers seek into the partial rather than append to it, so it has
        // to exist before they start — and keep whatever an earlier run left in
        // it.
        if let Err(failure) = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(partial)
            .map_err(|e| local(partial, "writing", &e))
        {
            return Some(failure);
        }

        let mut persisted = Instant::now();
        let failures = thread::scope(|scope| {
            let workers: Vec<_> = plan
                .slices
                .iter()
                .filter(|slice| !slice.complete())
                .map(|slice| {
                    scope.spawn(move || {
                        self.stream(
                            slice,
                            partial,
                            plan.etag.as_deref(),
                            plan.total,
                        )
                    })
                })
                .collect();

            loop {
                progress(label, plan.done(), plan.total);
                if workers.iter().all(|worker| worker.is_finished()) {
                    break;
                }
                if persisted.elapsed() >= PERSIST_EVERY {
                    plan.write(state);
                    persisted = Instant::now();
                }
                thread::sleep(POLL);
            }

            workers
                .into_iter()
                .filter_map(|worker| match worker.join() {
                    Ok(result) => result.err(),
                    // A panicking worker is a bug, not a network condition, and
                    // says nothing about whether retrying would help.
                    Err(_) => Some(self.fault(
                        Fault::Permanent,
                        "reading",
                        "a download thread panicked",
                    )),
                })
                .collect::<Vec<_>>()
        });

        // The most serious failure is the one worth reporting: a permanent one
        // ends the download, a restart costs what has arrived, a transient one
        // is just another attempt.
        failures
            .into_iter()
            .max_by_key(|failure| match failure.fault {
                Fault::Transient => 0,
                Fault::Restart => 1,
                Fault::Permanent => 2,
            })
    }

    /// Streams one slice of the file into its place in the partial.
    fn stream(
        &self,
        slice: &Slice,
        partial: &Path,
        etag: Option<&str>,
        total: Option<u64>,
    ) -> Result<(), Failure> {
        if slice.complete() {
            return Ok(());
        }
        let start = slice.at();
        let mut request = client()
            .map_err(|error| Failure {
                fault: Fault::Permanent,
                error,
            })?
            .get(self.url);
        // Only a slice that is the whole file from its very start needs no
        // range — every other one does, and says which file it belongs to.
        let ranged = start > 0 || slice.end.is_some();
        if ranged {
            let range = match slice.end {
                Some(end) => format!("bytes={start}-{}", end - 1),
                None => format!("bytes={start}-"),
            };
            request = request.header(header::RANGE, range);
            if let Some(etag) = etag {
                request = request.header(header::IF_RANGE, etag);
            }
        }

        let mut response = request.send().map_err(|e| {
            self.fault(Fault::Transient, "requesting", &e.to_string())
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(self.status_fault(status));
        }
        if ranged {
            // Anything but `206` is the whole file from the start: the server
            // either does not serve ranges, or has just told us through
            // `If-Range` that its copy is no longer the one we planned against.
            if status != StatusCode::PARTIAL_CONTENT {
                return Err(self.fault(
                    Fault::Restart,
                    "requesting",
                    "the file changed on the server; starting over",
                ));
            }
            if let Some(known) = total &&
                let Some(seen) = content_range_total(response.headers()) &&
                seen != known
            {
                return Err(self.fault(
                    Fault::Restart,
                    "requesting",
                    &format!(
                        "the file is now {seen} bytes, not {known}; starting \
                         over"
                    ),
                ));
            }
        }

        let mut output = fs::OpenOptions::new()
            .write(true)
            .open(partial)
            .map_err(|e| local(partial, "writing", &e))?;
        output
            .seek(SeekFrom::Start(start))
            .map_err(|e| local(partial, "writing", &e))?;

        let mut buffer = vec![0u8; BUFFER_BYTES];
        let mut left = slice.remaining();
        loop {
            let read = response.read(&mut buffer).map_err(|e| {
                self.fault(Fault::Transient, "reading", &e.to_string())
            })?;
            if read == 0 {
                break;
            }
            // A server may send more than it was asked for; writing past the
            // slice would land on top of the next one.
            let take = left.map_or(read, |left| {
                read.min(usize::try_from(left).unwrap_or(usize::MAX))
            });
            output
                .write_all(&buffer[..take])
                .map_err(|e| local(partial, "writing", &e))?;
            slice.done.fetch_add(take as u64, Ordering::Relaxed);
            if let Some(left) = left.as_mut() {
                *left -= take as u64;
                if *left == 0 {
                    break;
                }
            }
        }
        output.flush().map_err(|e| local(partial, "writing", &e))?;

        // A body that stops early looks exactly like a finished one to a read
        // loop; only the length asked for tells them apart. What arrived stays
        // on disk, so the next attempt asks for the rest.
        if let Some(left) = slice.remaining() &&
            left > 0
        {
            return Err(self.fault(
                Fault::Transient,
                "reading",
                &format!("the transfer ended {left} bytes short"),
            ));
        }
        Ok(())
    }

    /// Classifies a refusal: a server that is overloaded, throttling or gone
    /// for a moment is worth asking again, one that says the file is not there
    /// is not, and one that rejects the range says the partial is not a piece
    /// of this file.
    fn status_fault(&self, status: StatusCode) -> Failure {
        let fault = if status == StatusCode::RANGE_NOT_SATISFIABLE {
            Fault::Restart
        } else if status.is_server_error() ||
            status == StatusCode::TOO_MANY_REQUESTS ||
            status == StatusCode::REQUEST_TIMEOUT
        {
            Fault::Transient
        } else {
            Fault::Permanent
        };
        self.fault(fault, "requesting", &format!("http status {status}"))
    }

    /// Builds a failure naming the stage it happened in.
    fn fault(&self, fault: Fault, stage: &str, detail: &str) -> Failure {
        let (url, detail) = (self.url.to_owned(), detail.to_owned());
        Failure {
            fault,
            error: if stage == "reading" {
                DownloadError::Transfer { url, detail }
            } else {
                DownloadError::Request { url, detail }
            },
        }
    }
}

/// Creates a directory and its parents, reporting failure as a download error.
///
/// Callers need the model directory before they can tell what is missing from
/// it; [`Download::fetch`] creates what it writes into on its own.
///
/// # Errors
///
/// Returns [`DownloadError::File`] when the directory cannot be created.
pub fn create_dir(dir: &Path) -> Result<(), DownloadError> {
    fs::create_dir_all(dir).map_err(|e| DownloadError::File {
        action: "creating",
        path: dir.display().to_string(),
        detail: e.to_string(),
    })
}

/// What a failed attempt says about trying again.
enum Fault {
    /// Repeating this cannot help.
    Permanent,
    /// Worth another attempt, continuing from what arrived.
    Transient,
    /// Worth another attempt, but only after dropping what arrived: it is not
    /// part of the file the server is serving now.
    Restart,
}

/// A failed attempt and what to do about it.
struct Failure {
    fault: Fault,
    error: DownloadError,
}

/// A failed attempt, and whether the file is any closer to done for it.
///
/// The two answer different questions: the failure says whether trying again
/// can help at all, the progress says whether trying again is free.
struct Setback {
    failure: Failure,
    progressed: bool,
}

impl From<Failure> for Setback {
    /// A failure with nothing to say about progress made none: it never got as
    /// far as the body, or what it did bring in has just been thrown away.
    fn from(failure: Failure) -> Self {
        Self {
            failure,
            progressed: false,
        }
    }
}

/// What is left of a download's patience: how many attempts in a row have
/// brought nothing, and how long to wait before the next one.
///
/// Progress hands both back, so the download is given up on only when attempt
/// after attempt gets nowhere — a source that is gone rather than a link that
/// keeps dropping.
struct Budget {
    fruitless: usize,
    backoff: Duration,
}

impl Budget {
    /// A full budget.
    fn new() -> Self {
        Self {
            fruitless: 0,
            backoff: FIRST_BACKOFF,
        }
    }

    /// Books a failed attempt and says how long to wait before the next one, or
    /// `None` when there is nothing left to try with.
    fn spend(&mut self, progressed: bool) -> Option<Duration> {
        if progressed {
            *self = Self::new();
        } else {
            self.fruitless += 1;
            if self.fruitless >= MAX_FRUITLESS_ATTEMPTS {
                return None;
            }
        }
        let pause = self.backoff;
        self.backoff = (self.backoff * 2).min(MAX_BACKOFF);
        Some(pause)
    }
}

/// A local filesystem failure: never worth retrying, since nothing about the
/// next attempt would be different.
fn local(path: &Path, action: &'static str, error: &std::io::Error) -> Failure {
    Failure {
        fault: Fault::Permanent,
        error: DownloadError::File {
            action,
            path: path.display().to_string(),
            detail: error.to_string(),
        },
    }
}

/// Removes a partial and its state, so the next attempt starts clean.
fn discard(partial: &Path, state: &Path) {
    let _ = fs::remove_file(partial);
    let _ = fs::remove_file(state);
}

/// The shared HTTP client.
///
/// One client for the whole process keeps connections to the host alive across
/// the files of a checkpoint, saving a handshake each time.
///
/// **HTTP/1.1 only, and that is the whole point of the slices.** Over HTTP/2 a
/// client multiplexes every request onto one TCP connection, so four range
/// requests would share one congestion window and one connection's worth of
/// whatever the far end is willing to give — measurably the same speed as
/// fetching the file in one piece, for all the machinery of splitting it. Only
/// separate connections gain anything, and only HTTP/1.1 forces them.
///
/// Content encodings are off on purpose too: model files are already
/// compressed, and a transparently inflated body would make the advertised
/// length disagree with the bytes written — the very number ranges and progress
/// are counted in.
fn client() -> Result<&'static reqwest::blocking::Client, DownloadError> {
    static CLIENT: OnceLock<Result<reqwest::blocking::Client, String>> =
        OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::blocking::Client::builder()
                .user_agent(USER_AGENT)
                .connect_timeout(CONNECT_TIMEOUT)
                .timeout(STALL_TIMEOUT)
                .http1_only()
                .no_gzip()
                .no_deflate()
                .no_brotli()
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|detail| DownloadError::Client(detail.clone()))
}

/// The file a download streams into: the target's name with `.partial`
/// appended.
///
/// Appended, not substituted — `model.safetensors` and `model.json` would
/// otherwise share one partial.
fn partial_path(target: &Path) -> PathBuf { suffixed(target, ".partial") }

/// Where the state of a partial lives.
fn state_path(target: &Path) -> PathBuf { suffixed(target, ".partial.state") }

/// `path` with `suffix` appended to its file name.
fn suffixed(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// How a file is being fetched, and how far that has got.
///
/// This is what a `.partial` needs to be continued rather than restarted: the
/// URL it came from, how long the whole file is, the validator that says the
/// remote copy is still the same one, and the byte ranges the connections are
/// filling in. It is stored beside the partial as `key=value` lines and removed
/// once the download completes.
#[derive(Debug)]
struct Plan {
    url: String,
    total: Option<u64>,
    etag: Option<String>,
    slices: Vec<Slice>,
}

/// One contiguous piece of the file, fetched over its own connection.
#[derive(Debug)]
struct Slice {
    /// The first byte of the file this slice covers.
    start: u64,
    /// One past its last byte, when the length is known at all.
    end: Option<u64>,
    /// How many of its bytes are on disk. Written by the connection filling
    /// the slice and read by the reporter and the state file, hence
    /// atomic.
    done: AtomicU64,
}

impl Slice {
    /// A slice of `[start, end)` with nothing fetched yet.
    fn new(start: u64, end: Option<u64>) -> Self {
        Self {
            start,
            end,
            done: AtomicU64::new(0),
        }
    }

    /// Where in the file the next byte of this slice goes.
    fn at(&self) -> u64 { self.start + self.done.load(Ordering::Relaxed) }

    /// How many bytes are still owed, when that is known.
    fn remaining(&self) -> Option<u64> {
        self.end.map(|end| end.saturating_sub(self.at()))
    }

    /// Whether there is nothing left to fetch. A slice of unknown length is
    /// never complete until its connection says so.
    fn complete(&self) -> bool { self.remaining() == Some(0) }
}

/// Splits a file of `total` bytes across connections.
///
/// One connection is the answer to everything a server will not let us split: a
/// length it does not state, ranges it does not serve (both arrive here as
/// `None`), and a file too small for the slices to be worth their own
/// connection.
fn split(total: Option<u64>) -> Vec<Slice> {
    let Some(total) = total else {
        return vec![Slice::new(0, None)];
    };
    let connections = (total / MIN_SLICE_BYTES).clamp(1, CONNECTIONS);
    let each = total.div_ceil(connections);
    (0..connections)
        .map(|index| {
            let start = index * each;
            Slice::new(start, Some((start + each).min(total)))
        })
        .collect()
}

impl Plan {
    /// Reads the state beside a partial, or `None` when there is none — or when
    /// what is there does not describe a download this code can continue.
    fn read(path: &Path) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let (mut url, mut total, mut etag) = (None, None, None);
        let mut slices = Vec::new();
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            match key {
                "url" => url = Some(value.to_owned()),
                "total" => total = value.parse().ok(),
                "etag" => etag = Some(value.to_owned()),
                // A line that does not parse invalidates the whole state: it
                // is better to fetch the file again than to guess which bytes
                // it was claiming.
                "slice" => slices.push(parse_slice(value)?),
                _ => {},
            }
        }
        if slices.is_empty() {
            return None;
        }
        Some(Self {
            url: url?,
            total,
            etag,
            slices,
        })
    }

    /// Writes the state through a temporary file: it is rewritten while the
    /// download runs, and a half-written one would claim bytes that are not
    /// there. Losing it costs a restart, never correctness, so a failure to
    /// write is not worth failing the download over.
    fn write(&self, path: &Path) {
        let mut text = format!("url={}\n", self.url);
        if let Some(total) = self.total {
            text.push_str(&format!("total={total}\n"));
        }
        if let Some(etag) = &self.etag {
            text.push_str(&format!("etag={etag}\n"));
        }
        for slice in &self.slices {
            let end =
                slice.end.map_or_else(|| "-".to_owned(), |e| e.to_string());
            text.push_str(&format!(
                "slice={},{end},{}\n",
                slice.start,
                slice.done.load(Ordering::Relaxed)
            ));
        }
        let temp = suffixed(path, ".new");
        if fs::write(&temp, text).is_ok() {
            let _ = fs::rename(&temp, path);
        }
    }

    /// How many bytes of the file are on disk.
    fn done(&self) -> u64 {
        self.slices
            .iter()
            .map(|slice| slice.done.load(Ordering::Relaxed))
            .sum()
    }

    /// Whether enough of the file has arrived since `before` for the attempt
    /// that brought it to count as getting somewhere rather than being stuck.
    fn advanced(&self, before: u64) -> bool {
        let gained = self.done().saturating_sub(before);
        let bar = self.total.map_or(PROGRESS_BYTES, |total| {
            PROGRESS_BYTES.min(total.saturating_sub(before) / PROGRESS_SHARE)
        });
        gained > 0 && gained >= bar
    }
}

/// Parses a `start,end,done` slice line; `-` is an unknown end.
fn parse_slice(value: &str) -> Option<Slice> {
    let mut fields = value.split(',');
    let start = fields.next()?.parse().ok()?;
    let end = match fields.next()? {
        "-" => None,
        end => Some(end.parse().ok()?),
    };
    let done: u64 = fields.next()?.parse().ok()?;
    // A slice claiming more than it covers describes bytes that cannot be
    // where it says they are.
    if end.is_some_and(|end| start + done > end) {
        return None;
    }
    let slice = Slice::new(start, end);
    slice.done.store(done, Ordering::Relaxed);
    Some(slice)
}

/// The response's `ETag`, when it is a strong one.
///
/// A weak validator (`W/"…"`) says two responses are equivalent, not identical,
/// and must not be used with `If-Range`: a server given one ignores the range
/// and resends the whole file — exactly what resuming is meant to avoid.
fn strong_etag(headers: &header::HeaderMap) -> Option<String> {
    let etag = headers.get(header::ETAG)?.to_str().ok()?.trim();
    (!etag.is_empty() && !etag.starts_with("W/")).then(|| etag.to_owned())
}

/// The full length stated by `Content-Range: bytes <first>-<last>/<total>`; an
/// unknown total (`*`) reads as none.
fn content_range_total(headers: &header::HeaderMap) -> Option<u64> {
    headers
        .get(header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .rsplit('/')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Hashes a finished file and compares it with what the caller expects.
fn verify(path: &Path, digest: Digest<'_>) -> Result<(), DownloadError> {
    let failed = |e: &std::io::Error| DownloadError::File {
        action: "verifying",
        path: path.display().to_string(),
        detail: e.to_string(),
    };
    let mut file = fs::File::open(path).map_err(|e| failed(&e))?;
    let mut hasher = Hasher::new(digest);
    let mut buffer = vec![0u8; BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(|e| failed(&e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let (Digest::Blake3(expected) | Digest::Md5(expected)) = digest;
    let got = hasher.finish();
    if !got.eq_ignore_ascii_case(expected) {
        return Err(DownloadError::Checksum {
            path: path.display().to_string(),
            expected: expected.to_owned(),
            got,
        });
    }
    Ok(())
}

/// The hash being computed over a downloaded file.
///
/// BLAKE3's state is two kilobytes against MD5's ninety-odd bytes, but only one
/// of these ever exists at a time and only while a file is being verified, so
/// the lopsided enum is cheaper than boxing it.
#[allow(clippy::large_enum_variant)]
enum Hasher {
    Blake3(blake3::Hasher),
    Md5(Md5),
}

impl Hasher {
    fn new(digest: Digest<'_>) -> Self {
        match digest {
            Digest::Blake3(_) => Self::Blake3(blake3::Hasher::new()),
            Digest::Md5(_) => Self::Md5(Md5::new()),
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        match self {
            Self::Blake3(hasher) => {
                hasher.update(bytes);
            },
            Self::Md5(hasher) => hasher.update(bytes),
        }
    }

    fn finish(self) -> String {
        match self {
            Self::Blake3(hasher) => hasher.finalize().to_hex().to_string(),
            Self::Md5(hasher) => hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        }
    }
}
