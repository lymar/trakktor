//! Model resolution and download.
//!
//! A model is referred to either by a published name (`tiny`, `base.en`,
//! `large-v3`, `turbo`, ...) or by a path to a local checkpoint directory.
//! Named models are cached under the working directory as
//! `asr/whisper/<name>/` holding `config.json` and `model.safetensors`,
//! downloaded from the official repositories on first use.

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use super::error::WhisperError;
use crate::download::{self, Download, Progress};

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
    progress: Progress<'_>,
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

    for file in CHECKPOINT_FILES {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        let url = download::hugging_face_url(&format!("openai/{repo}"), file);
        Download::new(&url, &target).fetch(&mut *progress)?;
    }

    Ok(ResolvedModel {
        dir,
        name: Some(name),
    })
}
