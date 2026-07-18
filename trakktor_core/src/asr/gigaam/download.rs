//! Model resolution and download.
//!
//! A model is referred to either by a published name (`v3_ctc`,
//! `multilingual_ctc`, `multilingual_large_ctc`) or by a path to a local
//! `.ckpt` file. Named models are cached under the model directory as
//! `asr/gigaam/<name>.ckpt`, downloaded from the official CDN on first use and
//! verified against the published MD5.
//!
//! Downloads are **resumable**: the CDN is slow and rate-limits each
//! connection, so a large checkpoint may be interrupted; a retry continues from
//! the partial file via an HTTP range request rather than starting over.

#[cfg(test)]
mod tests;

use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use md5::{Digest, Md5};

use super::{config::ModelConfig, error::GigaamError};

/// A resolved model: its config and the local checkpoint path.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// Model geometry, vocabulary, and download coordinates.
    pub config: ModelConfig,
    /// The local `.ckpt` file.
    pub ckpt: PathBuf,
}

/// Resolves `model` to a checkpoint file, downloading a named model into
/// `<models_dir>/asr/gigaam/<name>.ckpt` on first use.
///
/// A path to a local `.ckpt` file is used directly (its config is inferred from
/// the file stem when that names a known model).
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, GigaamError> {
    // A path to a local checkpoint wins over the name table.
    let as_path = Path::new(model);
    if as_path.is_file() {
        let stem = as_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let config = super::config::config_for(stem).ok_or_else(|| {
            GigaamError::InvalidModel(format!(
                "checkpoint `{model}`: cannot infer geometry from the file \
                 name `{stem}` (expected one of: {})",
                super::config::KNOWN_MODELS.join(", ")
            ))
        })?;
        return Ok(ResolvedModel {
            config,
            ckpt: as_path.to_path_buf(),
        });
    }

    let Some(config) = super::config::config_for(model) else {
        return Err(GigaamError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint file)",
            super::config::KNOWN_MODELS.join(", ")
        )));
    };

    let dir = models_dir.join("asr").join("gigaam");
    let ckpt = dir.join(&config.download.ckpt);
    if ckpt.is_file() {
        return Ok(ResolvedModel { config, ckpt });
    }

    fs::create_dir_all(&dir).map_err(|e| {
        GigaamError::ModelDownload(format!("creating {}: {e}", dir.display()))
    })?;

    download_resumable(
        &config.download.url,
        &ckpt,
        &config.download.md5,
        &config.download.ckpt,
        progress,
    )?;

    Ok(ResolvedModel { config, ckpt })
}

/// Downloads `url` into `target`, resuming from a `.partial` file if one is
/// present, then verifies the MD5 and moves it into place.
fn download_resumable(
    url: &str,
    target: &Path,
    expected_md5: &str,
    label: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<(), GigaamError> {
    let failed = |stage: &str, detail: String| {
        GigaamError::ModelDownload(format!("{stage} {url}: {detail}"))
    };

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Checkpoints are large and the CDN is slow; only the connection phase
        // is bounded.
        .timeout(None)
        .build()
        .map_err(|e| failed("building http client", e.to_string()))?;

    let temp = target.with_extension("partial");
    // How many bytes we already have to resume from.
    let mut have: u64 = temp.metadata().map(|m| m.len()).unwrap_or(0);

    let mut request = client.get(url);
    if have > 0 {
        request =
            request.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut response = request
        .send()
        .map_err(|e| failed("requesting", e.to_string()))?;

    // If the server ignores the range (200 instead of 206), restart from zero.
    let resuming = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !resuming {
        have = 0;
    }
    if !response.status().is_success() {
        return Err(failed(
            "requesting",
            format!("http status {}", response.status()),
        ));
    }
    let total = response.content_length().map(|len| len + have);

    let mut output = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!resuming)
        .open(&temp)
        .map_err(|e| failed("writing", e.to_string()))?;
    if resuming {
        output
            .seek(SeekFrom::Start(have))
            .map_err(|e| failed("writing", e.to_string()))?;
    }

    let mut buffer = vec![0u8; 1 << 20];
    let mut done = have;
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| failed("reading", e.to_string()))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|e| failed("writing", e.to_string()))?;
        done += read as u64;
        progress(label, done, total);
    }
    output
        .flush()
        .map_err(|e| failed("writing", e.to_string()))?;
    drop(output);

    verify_md5(&temp, expected_md5).inspect_err(|_| {
        // A corrupt partial is unrecoverable; drop it so the next try restarts.
        let _ = fs::remove_file(&temp);
    })?;

    fs::rename(&temp, target)
        .map_err(|e| failed("finalizing", e.to_string()))?;
    Ok(())
}

/// Verifies a file's MD5 against the expected hex digest.
fn verify_md5(path: &Path, expected: &str) -> Result<(), GigaamError> {
    let mut file = fs::File::open(path).map_err(|e| {
        GigaamError::ModelDownload(format!("verifying {}: {e}", path.display()))
    })?;
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|e| {
            GigaamError::ModelDownload(format!(
                "verifying {}: {e}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let got = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if got != expected {
        return Err(GigaamError::ModelDownload(format!(
            "checksum mismatch for {}: expected {expected}, got {got}",
            path.display()
        )));
    }
    Ok(())
}
