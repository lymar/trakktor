//! Model resolution and download.
//!
//! A model is referred to either by a published name (`tiny`, `base.en`,
//! `large-v3`, `turbo`, ...) or by a path to a local checkpoint directory.
//! Named models are cached under the working directory as
//! `asr/whisper/<name>/` holding `config.json` and `model.safetensors`,
//! downloaded from the official repositories on first use.

#[cfg(test)]
mod tests;

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use super::error::WhisperError;

/// Published model names and their repository slugs.
pub const KNOWN_MODELS: &[(&str, &str)] = &[
    ("tiny", "whisper-tiny"),
    ("tiny.en", "whisper-tiny.en"),
    ("base", "whisper-base"),
    ("base.en", "whisper-base.en"),
    ("small", "whisper-small"),
    ("small.en", "whisper-small.en"),
    ("medium", "whisper-medium"),
    ("medium.en", "whisper-medium.en"),
    ("large-v1", "whisper-large"),
    ("large-v2", "whisper-large-v2"),
    ("large-v3", "whisper-large-v3"),
    ("large", "whisper-large-v3"),
    ("turbo", "whisper-large-v3-turbo"),
    ("large-v3-turbo", "whisper-large-v3-turbo"),
];

/// The files a checkpoint directory must hold.
const CHECKPOINT_FILES: &[&str] = &["config.json", "model.safetensors"];

/// A resolved model: where its checkpoint lives, and its canonical published
/// name when it was referred to by name (a plain directory has none).
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory (`config.json` + `model.safetensors`).
    pub dir: PathBuf,
    /// The canonical published name, when known — the key for the
    /// word-alignment head table.
    pub name: Option<&'static str>,
}

/// Resolves `model` to a checkpoint directory, downloading a named model
/// into `<models_dir>/asr/whisper/<name>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`WhisperError::InvalidModel`] for an unknown name and
/// [`WhisperError::ModelDownload`] when fetching the checkpoint fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, WhisperError> {
    // A path to a local checkpoint directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.join("config.json").is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(&(name, repo)) = KNOWN_MODELS
        .iter()
        .find(|(known_name, _)| *known_name == model)
    else {
        let known: Vec<&str> =
            KNOWN_MODELS.iter().map(|&(name, _)| name).collect();
        return Err(WhisperError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            known.join(", ")
        )));
    };

    let dir = models_dir.join("asr").join("whisper").join(name);
    if CHECKPOINT_FILES.iter().all(|file| dir.join(file).is_file()) {
        return Ok(ResolvedModel {
            dir,
            name: Some(name),
        });
    }

    fs::create_dir_all(&dir).map_err(|e| {
        WhisperError::ModelDownload(format!("creating {}: {e}", dir.display()))
    })?;

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Model files are large; only the connection phase is bounded.
        .timeout(None)
        .build()
        .map_err(|e| {
            WhisperError::ModelDownload(format!("building http client: {e}"))
        })?;

    for file in CHECKPOINT_FILES {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        let url =
            format!("https://huggingface.co/openai/{repo}/resolve/main/{file}");
        download_file(&client, &url, &target, file, progress)?;
    }

    Ok(ResolvedModel {
        dir,
        name: Some(name),
    })
}

/// Streams `url` into `target` via a temporary file, so an interrupted
/// download never leaves a half-written checkpoint behind.
fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    target: &Path,
    label: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<(), WhisperError> {
    let failed = |stage: &str, detail: String| {
        WhisperError::ModelDownload(format!("{stage} {url}: {detail}"))
    };

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| failed("requesting", e.to_string()))?;
    if !response.status().is_success() {
        return Err(failed(
            "requesting",
            format!("http status {}", response.status()),
        ));
    }
    let total = response.content_length();

    let temp = target.with_extension("partial");
    let mut output = fs::File::create(&temp)
        .map_err(|e| failed("writing", e.to_string()))?;

    let mut buffer = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
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

    fs::rename(&temp, target)
        .map_err(|e| failed("finalizing", e.to_string()))?;
    Ok(())
}
