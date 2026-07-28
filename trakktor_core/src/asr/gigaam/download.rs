//! Model resolution and download.
//!
//! A model is referred to either by a published name (`v3_ctc`,
//! `multilingual_ctc`, `multilingual_large_ctc`) or by a path to a local
//! `.ckpt` file. Named models are cached under the model directory as
//! `asr/gigaam/<name>.ckpt`, downloaded from the official CDN on first use and
//! verified against the published MD5.

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use super::{config::ModelConfig, error::GigaamError};
use crate::download::{Download, Progress};

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
    progress: Progress<'_>,
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

    Download::new(&config.download.url, &ckpt)
        .md5(&config.download.md5)
        .fetch(progress)?;

    Ok(ResolvedModel { config, ckpt })
}
