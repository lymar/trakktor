//! Model resolution and download.
//!
//! A model is referred to either by a published name (see
//! [`catalog`](super::catalog)) or by a path to a local directory holding the
//! four bundle files. Named models are cached under the model directory as
//! `asr/vosk/<name>/`, fetched from the hosting on first use and verified
//! against the catalog's BLAKE3 hashes.
//!
//! The catalog publishes each file's size as well as its hash, so a file is
//! skipped only when both its name and its length match — a file left over from
//! a different version of a model is fetched again rather than loaded.

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use super::{
    catalog::{self, ModelKind, ModelSpec},
    error::VoskError,
};
use crate::download::{self, Download, Progress};

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
    progress: Progress<'_>,
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
    for file in &spec.files {
        let target = dir.join(file.local);
        if target.is_file() &&
            target.metadata().map(|meta| meta.len()).unwrap_or(0) ==
                file.size
        {
            continue;
        }
        let url = download::hugging_face_url(spec.repo, file.remote);
        // The bundle names are the same in every model, so the label says which
        // model they belong to.
        let label = format!("{}/{}", spec.name, file.local);
        Download::new(&url, &target)
            .label(&label)
            .blake3(file.blake3)
            .fetch(&mut *progress)?;
    }

    Ok(ResolvedModel {
        dir,
        spec: Some(spec),
    })
}
