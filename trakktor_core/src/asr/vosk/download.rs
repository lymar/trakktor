//! Model resolution and download.
//!
//! A model is referred to either by a published name (see
//! [`catalog`](super::catalog)) or by a path to a local directory holding the
//! four bundle files. Named models are cached under the model directory as
//! `asr/vosk/<name>/`, fetched from the hosting on first use and verified
//! against the catalog's BLAKE3 hashes.
//!
//! **A file's final name only ever names complete, verified data.** Each file
//! streams into a sibling `<name>.partial`; only after its BLAKE3 hash checks
//! out is it moved onto the final name with an atomic same-directory rename.
//! So an interrupted or cancelled download leaves a `.partial` (never the real
//! name), which the skip check ignores — the next run cannot mistake it for a
//! finished file, and a resumed download continues from it via an HTTP range
//! request rather than starting over.
//!
//! Downloads are also **retried**: the hosting CDN is flaky, so a transient
//! failure (a request that fails to send or drops mid-stream, a 5xx or 429) is
//! retried with exponential backoff — resuming from the partial each time —
//! rather than aborting the whole model. Permanent failures (a 4xx, a checksum
//! mismatch) are not retried.

#[cfg(test)]
mod tests;

use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use super::{
    catalog::{self, ModelKind, ModelSpec},
    error::VoskError,
};

/// A resolved model: the local directory with the four bundle files, and the
/// catalog spec when the model came from the catalog.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// Directory holding `encoder.onnx`, `decoder.onnx`, `joiner.onnx`, and
    /// `tokens.txt`.
    pub dir: PathBuf,
    /// The catalog entry; `None` for a local directory.
    pub spec: Option<&'static ModelSpec>,
}

impl ResolvedModel {
    /// The model kind. A local directory's kind is decided later from its
    /// weights (causal models carry streaming metadata), so this is only
    /// available for catalog models.
    pub fn kind(&self) -> Option<ModelKind> { self.spec.map(|s| s.kind) }
}

/// The bundle file names every model directory must hold.
const BUNDLE: [&str; 4] = [
    catalog::ENCODER_FILE,
    catalog::DECODER_FILE,
    catalog::JOINER_FILE,
    catalog::TOKENS_FILE,
];

/// Resolves `model` to a local model directory, downloading a named model
/// into `<models_dir>/asr/vosk/<name>/` on first use.
///
/// A path to a local directory with the four bundle files is used directly.
///
/// `progress` is called as a download advances with
/// `(file name, bytes done, bytes total when known)`.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, VoskError> {
    // A local directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.is_dir() {
        for f in BUNDLE {
            if !as_path.join(f).is_file() {
                return Err(VoskError::InvalidModel(format!(
                    "model directory `{model}` has no `{f}` (expected the \
                     bundle {})",
                    BUNDLE.join(", ")
                )));
            }
        }
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            spec: None,
        });
    }

    let Some(spec) = catalog::spec_for(model) else {
        return Err(VoskError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a directory with the \
             model files)",
            catalog::known_names()
        )));
    };

    let dir = models_dir.join("asr").join("vosk").join(spec.name);
    fs::create_dir_all(&dir).map_err(|e| {
        VoskError::ModelDownload(format!("creating {}: {e}", dir.display()))
    })?;

    for f in &spec.files {
        let target = dir.join(f.local);
        if target.is_file() &&
            target.metadata().map(|m| m.len()).unwrap_or(0) == f.size
        {
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, f.remote
        );
        let label = format!("{}/{}", spec.name, f.local);
        download_resumable(&url, &target, f.blake3, &label, progress)?;
    }

    Ok(ResolvedModel {
        dir,
        spec: Some(spec),
    })
}

/// Attempts before giving up on a single file. The hosting CDN is flaky —
/// a request can fail to send or drop mid-stream — so a transient failure is
/// retried rather than aborting the whole model download.
const MAX_ATTEMPTS: usize = 5;

/// A failed download attempt, tagged with whether retrying could help.
struct AttemptError {
    /// A transient failure (network send/read, a 5xx or 429): retry from the
    /// partial file. A permanent one (4xx, checksum mismatch, local IO):
    /// give up.
    retryable: bool,
    error: VoskError,
}

/// Downloads `url` into `target`, resuming from a `.partial` file if one is
/// present, then verifies the BLAKE3 hash and moves it into place. Transient
/// failures are retried with exponential backoff; because the download
/// resumes from the partial, a retry continues rather than restarting.
fn download_resumable(
    url: &str,
    target: &Path,
    expected_blake3: &str,
    label: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<(), VoskError> {
    let mut backoff = Duration::from_secs(1);
    for attempt in 1..=MAX_ATTEMPTS {
        match download_attempt(url, target, expected_blake3, label, progress) {
            Ok(()) => return Ok(()),
            Err(AttemptError {
                retryable: false,
                error,
            }) => return Err(error),
            Err(AttemptError {
                retryable: true,
                error,
            }) => {
                if attempt == MAX_ATTEMPTS {
                    return Err(error);
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(30));
            },
        }
    }
    unreachable!("the loop returns on the last attempt")
}

/// One download attempt: request (resuming from the partial), stream to the
/// partial file, verify, and move into place.
fn download_attempt(
    url: &str,
    target: &Path,
    expected_blake3: &str,
    label: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<(), AttemptError> {
    // Local/permanent failures: no point retrying.
    let fatal = |stage: &str, detail: String| AttemptError {
        retryable: false,
        error: VoskError::ModelDownload(format!("{stage} {url}: {detail}")),
    };
    // Network/server hiccups: a retry from the partial may succeed.
    let transient = |stage: &str, detail: String| AttemptError {
        retryable: true,
        error: VoskError::ModelDownload(format!("{stage} {url}: {detail}")),
    };

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Model files are large; only the connection phase is bounded.
        .timeout(None)
        .build()
        .map_err(|e| fatal("building http client", e.to_string()))?;

    let temp = target.with_extension("partial");
    let mut have: u64 = temp.metadata().map(|m| m.len()).unwrap_or(0);

    let mut request = client.get(url);
    if have > 0 {
        request =
            request.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut response = request
        .send()
        .map_err(|e| transient("requesting", e.to_string()))?;

    // If the server ignores the range (200 instead of 206), restart at zero.
    let resuming = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !resuming {
        have = 0;
    }
    if !response.status().is_success() {
        let status = response.status();
        let detail = format!("http status {status}");
        // 5xx and 429 are worth retrying; other 4xx are not.
        return Err(
            if status.is_server_error() ||
                status == reqwest::StatusCode::TOO_MANY_REQUESTS
            {
                transient("requesting", detail)
            } else {
                fatal("requesting", detail)
            },
        );
    }
    let total = response.content_length().map(|len| len + have);

    let mut output = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!resuming)
        .open(&temp)
        .map_err(|e| fatal("writing", e.to_string()))?;
    if resuming {
        output
            .seek(SeekFrom::Start(have))
            .map_err(|e| fatal("writing", e.to_string()))?;
    }

    let mut buffer = vec![0u8; 1 << 20];
    let mut done = have;
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| transient("reading", e.to_string()))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|e| fatal("writing", e.to_string()))?;
        done += read as u64;
        progress(label, done, total);
    }
    output
        .flush()
        .map_err(|e| fatal("writing", e.to_string()))?;
    drop(output);

    verify_blake3(&temp, expected_blake3).map_err(|e| {
        // A corrupt partial is unrecoverable; drop it so a retry (or the next
        // run) restarts from zero rather than resuming the bad bytes.
        let _ = fs::remove_file(&temp);
        AttemptError {
            retryable: false,
            error: e,
        }
    })?;

    fs::rename(&temp, target)
        .map_err(|e| fatal("finalizing", e.to_string()))?;
    Ok(())
}

/// Verifies a file's BLAKE3 hash against the expected hex digest.
fn verify_blake3(path: &Path, expected: &str) -> Result<(), VoskError> {
    let err = |detail: String| {
        VoskError::ModelDownload(format!(
            "verifying {}: {detail}",
            path.display()
        ))
    };
    let mut file = fs::File::open(path).map_err(|e| err(e.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|e| err(e.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let got = hasher.finalize().to_hex().to_string();
    if got != expected {
        return Err(VoskError::ModelDownload(format!(
            "checksum mismatch for {}: expected {expected}, got {got}",
            path.display()
        )));
    }
    Ok(())
}
