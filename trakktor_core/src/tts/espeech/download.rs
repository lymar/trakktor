//! Model resolution, download, and the one conversion this engine needs.
//!
//! ESpeech publishes training checkpoints: a pickle archive holding the network
//! **twice**, once as the live weights and once as their moving average. Only
//! the average is ever used for inference — the reference loads exactly that —
//! so a download does not keep the archive. It extracts the average, writes it
//! as safetensors, and deletes the archive: half the bytes on disk, one format
//! for both runtimes to read, and the pickle reader confined to this module.
//!
//! The vocoder is fetched once and shared by every variant.

use std::{
    fs,
    path::{Path, PathBuf},
};

use super::error::EspeechError;
use crate::download::{self, Download, Progress};

/// The converted weights inside a variant's directory.
pub const MODEL_WEIGHTS: &str = "model.safetensors";

/// The character table inside a variant's directory.
pub const VOCAB_FILE: &str = "vocab.txt";

/// The converted vocoder weights inside the vocoder's directory.
pub const VOCODER_WEIGHTS: &str = "model.safetensors";

/// Where the shared vocoder lives, under the engine's directory.
pub const VOCODER_DIR: &str = "vocos-mel-24khz";

/// The repository the vocoder comes from.
const VOCODER_REPO: &str = "charactr/vocos-mel-24khz";

/// The file the vocoder's weights come as.
const VOCODER_SOURCE: &str = "pytorch_model.bin";

/// The key of the moving average inside a published checkpoint, and the prefix
/// its tensor names carry.
const EMA_KEY: &str = "ema_model_state_dict";
const EMA_PREFIX: &str = "ema_model.";

/// Bookkeeping entries of the moving average, which are not weights.
const EMA_STATE: &[&str] = &["initted", "step"];

/// A published checkpoint: the name `--model` accepts, its repository, and the
/// file it publishes the weights as.
#[derive(Debug, Clone, Copy)]
pub struct KnownModel {
    /// The name `--model` accepts.
    pub name: &'static str,
    /// The Hugging Face repository holding the checkpoint.
    pub repo: &'static str,
    /// The checkpoint file inside it. Not derivable from the name: every
    /// variant spells it differently.
    pub file: &'static str,
}

/// The checkpoints this engine can serve.
///
/// All five are the same network trained differently — same geometry, same
/// size, same character table — so there is no large-versus-small choice to
/// make here, only a voice-and-manner one.
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        name: "rl-v2",
        repo: "ESpeech/ESpeech-TTS-1_RL-V2",
        file: "espeech_tts_rlv2.pt",
    },
    KnownModel {
        name: "rl-v1",
        repo: "ESpeech/ESpeech-TTS-1_RL-V1",
        file: "espeech_tts_rlv1.pt",
    },
    KnownModel {
        name: "sft-256k",
        repo: "ESpeech/ESpeech-TTS-1_SFT-256K",
        file: "espeech_tts_256k.pt",
    },
    KnownModel {
        name: "sft-95k",
        repo: "ESpeech/ESpeech-TTS-1_SFT-95K",
        file: "espeech_tts_95k.pt",
    },
    KnownModel {
        name: "podcaster",
        repo: "ESpeech/ESpeech-TTS-1_podcaster",
        file: "espeech_tts_podcaster.pt",
    },
];

/// A resolved checkpoint: where it lives, where the vocoder lives, and which
/// published variant it is.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory.
    pub dir: PathBuf,
    /// The vocoder directory, shared by every variant.
    pub vocoder_dir: PathBuf,
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
    models_dir.join("tts").join("espeech")
}

/// Resolves `model` to a checkpoint directory, downloading and converting a
/// named variant into `<models_dir>/tts/espeech/<variant>/` on first use, along
/// with the shared vocoder.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`EspeechError::InvalidModel`] for an unknown name and
/// [`EspeechError::ModelDownload`] when fetching or converting fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, EspeechError> {
    let vocoder_dir = engine_dir(models_dir).join(VOCODER_DIR);

    // A path to a local checkpoint directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.join(MODEL_WEIGHTS).is_file() {
        ensure_vocoder(&vocoder_dir, progress)?;
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            vocoder_dir,
            known: None,
        });
    }

    let Some(&known) = KNOWN_MODELS.iter().find(|entry| entry.name == model)
    else {
        let names: Vec<&str> =
            KNOWN_MODELS.iter().map(|entry| entry.name).collect();
        return Err(EspeechError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            names.join(", ")
        )));
    };
    let dir = engine_dir(models_dir).join(known.name);
    ensure_variant(&dir, known, progress)?;
    ensure_vocoder(&vocoder_dir, progress)?;
    Ok(ResolvedModel {
        dir,
        vocoder_dir,
        known: Some(known),
    })
}

/// Makes sure a variant's directory holds converted weights and its character
/// table.
fn ensure_variant(
    dir: &Path,
    known: KnownModel,
    progress: Progress<'_>,
) -> Result<(), EspeechError> {
    let weights = dir.join(MODEL_WEIGHTS);
    let vocab = dir.join(VOCAB_FILE);
    if weights.is_file() && vocab.is_file() {
        return Ok(());
    }
    if !vocab.is_file() {
        let url = download::hugging_face_url(known.repo, VOCAB_FILE);
        Download::new(&url, &vocab).fetch(&mut *progress)?;
    }
    if !weights.is_file() {
        // The archive lands next to the weights it becomes and is removed once
        // the conversion succeeds — so a conversion that fails does not cost
        // the download a second time.
        let archive = dir.join(known.file);
        if !archive.is_file() {
            let url = download::hugging_face_url(known.repo, known.file);
            Download::new(&url, &archive).fetch(progress)?;
        }
        convert_checkpoint(&archive, &weights)?;
        let _ = fs::remove_file(&archive);
    }
    Ok(())
}

/// Makes sure the shared vocoder is present and converted.
fn ensure_vocoder(
    dir: &Path,
    progress: Progress<'_>,
) -> Result<(), EspeechError> {
    let weights = dir.join(VOCODER_WEIGHTS);
    if weights.is_file() {
        return Ok(());
    }
    let archive = dir.join(VOCODER_SOURCE);
    if !archive.is_file() {
        let url = download::hugging_face_url(VOCODER_REPO, VOCODER_SOURCE);
        Download::new(&url, &archive).fetch(progress)?;
    }
    convert_vocoder(&archive, &weights)?;
    let _ = fs::remove_file(&archive);
    Ok(())
}

/// Extracts the moving average from a published checkpoint and writes it as
/// safetensors.
///
/// This is the one place that reads the reference's own save format, and it
/// checks what it found rather than trusting it: a checkpoint saved without the
/// average, or in half precision, would otherwise convert into something that
/// loads and sounds wrong.
fn convert_checkpoint(archive: &Path, out: &Path) -> Result<(), EspeechError> {
    let failed = |what: &str| {
        EspeechError::ModelDownload(format!("{}: {what}", archive.display()))
    };
    let tensors =
        candle_core::pickle::read_all_with_key(archive, Some(EMA_KEY))
            .map_err(|e| {
                failed(&format!(
                    "cannot read the `{EMA_KEY}` half of the checkpoint \
                     ({e}); a checkpoint saved without the moving average is \
                     not the one the reference runs"
                ))
            })?;

    let mut kept = std::collections::HashMap::new();
    for (name, tensor) in tensors {
        let name = name.strip_prefix(EMA_PREFIX).unwrap_or(&name).to_owned();
        if EMA_STATE.contains(&name.as_str()) {
            continue;
        }
        if tensor.dtype() != candle_core::DType::F32 {
            return Err(failed(&format!(
                "`{name}` is stored as {:?}, expected f32",
                tensor.dtype()
            )));
        }
        kept.insert(name, tensor);
    }
    if !kept.keys().any(|name| name.starts_with("transformer.")) {
        return Err(failed(
            "the moving average holds no `transformer.` tensors",
        ));
    }
    write_safetensors(kept, out)
}

/// Converts the vocoder's checkpoint, which is a plain state dictionary.
fn convert_vocoder(archive: &Path, out: &Path) -> Result<(), EspeechError> {
    let tensors = candle_core::pickle::read_all(archive).map_err(|e| {
        EspeechError::ModelDownload(format!(
            "{}: cannot read the vocoder ({e})",
            archive.display()
        ))
    })?;
    let kept: std::collections::HashMap<String, candle_core::Tensor> =
        tensors.into_iter().collect();
    if !kept.contains_key("feature_extractor.mel_spec.mel_scale.fb") {
        return Err(EspeechError::ModelDownload(format!(
            "{}: the vocoder carries no mel filterbank, which is where the \
             frontend takes it from",
            archive.display()
        )));
    }
    write_safetensors(kept, out)
}

/// Writes tensors to `out` through a temporary file, so an interrupted write
/// never leaves a half-converted checkpoint behind.
fn write_safetensors(
    tensors: std::collections::HashMap<String, candle_core::Tensor>,
    out: &Path,
) -> Result<(), EspeechError> {
    let temp = out.with_extension("partial");
    candle_core::safetensors::save(&tensors, &temp).map_err(|e| {
        EspeechError::ModelDownload(format!("writing {}: {e}", temp.display()))
    })?;
    fs::rename(&temp, out).map_err(|e| {
        EspeechError::ModelDownload(format!(
            "finalizing {}: {e}",
            out.display()
        ))
    })
}

#[cfg(test)]
mod tests;
