//! Model resolution and download.
//!
//! A model is referred to either by a published variant name or by a path to a
//! local checkpoint directory. Named variants are cached under the model
//! directory as `tts/qwen3-tts/<variant>/`, fetched on first use.
//!
//! Each checkpoint carries its own copy of the codec under `speech_tokenizer/`,
//! so there is nothing else to fetch alongside it.

use std::path::{Path, PathBuf};

use super::{config::ModelType, error::Qwen3TtsError};
use crate::download::{self, Download, Progress};

/// A published checkpoint: the name `--model` accepts, the repository it comes
/// from, and the conditioning path it was trained for.
#[derive(Debug, Clone, Copy)]
pub struct KnownModel {
    /// The name `--model` accepts.
    pub name: &'static str,
    /// The Hugging Face repository holding the checkpoint.
    pub repo: &'static str,
    /// How the voice is chosen for this checkpoint.
    pub model_type: ModelType,
}

/// The checkpoints this engine can serve.
///
/// The family also publishes VoiceDesign and Base checkpoints; they condition
/// the voice in ways this engine does not implement yet, so they are not
/// offered — listing them would only let a caller spend gigabytes on a
/// checkpoint that cannot then be used.
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        name: "0.6b-customvoice",
        repo: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        model_type: ModelType::CustomVoice,
    },
    KnownModel {
        name: "1.7b-customvoice",
        repo: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_type: ModelType::CustomVoice,
    },
];

/// Files a checkpoint directory must hold for the engine to load it.
///
/// The text tokenizer ships as a raw byte-level BPE (`vocab.json` plus
/// `merges.txt`) rather than a packaged `tokenizer.json`, so both come along,
/// with `tokenizer_config.json` for the control tokens.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
];

/// A resolved checkpoint: where it lives and, when referred to by name, which
/// published variant it is.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory.
    pub dir: PathBuf,
    /// The published variant, when the model was named rather than pathed.
    pub known: Option<KnownModel>,
}

impl ResolvedModel {
    /// The name to report in the output: the published variant, or the
    /// directory that was given.
    #[must_use]
    pub fn label(&self) -> String {
        self.known
            .map(|model| model.name.to_owned())
            .unwrap_or_else(|| self.dir.display().to_string())
    }
}

/// The subdirectory of the model directory that holds this engine's state.
fn engine_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("tts").join("qwen3-tts")
}

/// Resolves `model` to a checkpoint directory, downloading a named variant into
/// `<models_dir>/tts/qwen3-tts/<variant>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidModel`] for an unknown name and
/// [`Qwen3TtsError::ModelDownload`] when fetching the checkpoint fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, Qwen3TtsError> {
    // A path to a local checkpoint directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.join("config.json").is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            known: None,
        });
    }

    let Some(&known) = KNOWN_MODELS.iter().find(|entry| entry.name == model)
    else {
        let names: Vec<&str> =
            KNOWN_MODELS.iter().map(|entry| entry.name).collect();
        return Err(Qwen3TtsError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            names.join(", ")
        )));
    };

    let dir = engine_dir(models_dir).join(known.name);
    if REQUIRED_FILES.iter().all(|file| dir.join(file).is_file()) {
        return Ok(ResolvedModel {
            dir,
            known: Some(known),
        });
    }

    for file in REQUIRED_FILES {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        let url = download::hugging_face_url(known.repo, file);
        // `speech_tokenizer/…` nests, so the label carries the path inside the
        // checkpoint rather than the bare file name two of them share.
        Download::new(&url, &target)
            .label(file)
            .fetch(&mut *progress)?;
    }

    Ok(ResolvedModel {
        dir,
        known: Some(known),
    })
}

#[cfg(test)]
mod tests;
