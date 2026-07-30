//! Model resolution and download.
//!
//! A model is referred to either by a published name (`tiny`, `base.en`,
//! `large-v3`, `turbo`, ...) or by a path to a local checkpoint directory.
//! Named models are cached under the working directory as
//! `asr/whisper/<name>/` holding `config.json` and the safetensors weights —
//! a single `model.safetensors`, or a shard index with its shards —
//! downloaded from their repositories on first use.

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use super::error::WhisperError;
use crate::download::{self, Download, Progress};

/// The checkpoint files of a model whose weights ship as one file.
const SINGLE_FILE_CHECKPOINT: &[&str] = &["config.json", "model.safetensors"];

/// A published model: its name, the Hugging Face repository holding the
/// checkpoint, and the files the checkpoint directory needs.
#[derive(Debug, Clone, Copy)]
pub struct KnownModel {
    /// The published name the CLI accepts (`tiny`, `turbo`, `podlodka`, ...).
    pub name: &'static str,
    /// The Hugging Face repository (`owner/name`).
    pub repo: &'static str,
    /// The files to download: `config.json` plus the weights — a single
    /// `model.safetensors`, or a shard index and the shards it lists.
    pub files: &'static [&'static str],
}

/// Published model names and their checkpoints.
///
/// The `openai/*` entries are the official Whisper checkpoints. `podlodka`
/// and `podlodka-turbo` are Russian fine-tunes of `large-v3` and
/// `large-v3-turbo` (Apache-2.0); they keep the parent geometry, vocabulary,
/// and alignment heads, so the pipeline treats them like their parents.
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        name: "tiny",
        repo: "openai/whisper-tiny",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "tiny.en",
        repo: "openai/whisper-tiny.en",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "base",
        repo: "openai/whisper-base",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "base.en",
        repo: "openai/whisper-base.en",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "small",
        repo: "openai/whisper-small",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "small.en",
        repo: "openai/whisper-small.en",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "medium",
        repo: "openai/whisper-medium",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "medium.en",
        repo: "openai/whisper-medium.en",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "large-v1",
        repo: "openai/whisper-large",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "large-v2",
        repo: "openai/whisper-large-v2",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "large-v3",
        repo: "openai/whisper-large-v3",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "large",
        repo: "openai/whisper-large-v3",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "turbo",
        repo: "openai/whisper-large-v3-turbo",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "large-v3-turbo",
        repo: "openai/whisper-large-v3-turbo",
        files: SINGLE_FILE_CHECKPOINT,
    },
    KnownModel {
        name: "podlodka",
        repo: "bond005/whisper-large-v3-ru-podlodka",
        files: &[
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    },
    KnownModel {
        name: "podlodka-turbo",
        repo: "bond005/whisper-podlodka-turbo",
        files: SINGLE_FILE_CHECKPOINT,
    },
];

/// A resolved model: where its checkpoint lives, and its canonical published
/// name when it was referred to by name (a plain directory has none).
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory (`config.json` + the safetensors weights).
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

    let Some(known) = KNOWN_MODELS.iter().find(|entry| entry.name == model)
    else {
        let names: Vec<&str> =
            KNOWN_MODELS.iter().map(|entry| entry.name).collect();
        return Err(WhisperError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            names.join(", ")
        )));
    };

    let dir = models_dir.join("asr").join("whisper").join(known.name);
    if known.files.iter().all(|file| dir.join(file).is_file()) {
        return Ok(ResolvedModel {
            dir,
            name: Some(known.name),
        });
    }

    for file in known.files {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        let url = download::hugging_face_url(known.repo, file);
        Download::new(&url, &target).fetch(&mut *progress)?;
    }

    Ok(ResolvedModel {
        dir,
        name: Some(known.name),
    })
}
