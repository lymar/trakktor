//! Model resolution and download.
//!
//! A model is referred to either by a published name or by a path to a local
//! directory that already holds the extracted files. Named models are cached
//! under the model directory as `text/punctuate/<name>/`, downloaded on first
//! use: the SentencePiece model (`sp.model`) is fetched directly, and the
//! PyTorch weights (`model_weights.ckpt`) are extracted from the NeMo archive
//! (`*.nemo`, a tar), which is then removed.

use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use super::{error::PunctuateError, runtime::WEIGHTS_FILE};
use crate::download::{self, Download, Progress};

/// The SentencePiece model file name, in the repo and in the cache.
pub(super) const SPE_FILE: &str = "sp.model";

/// A published model: its cache name, HF repo, and the two files fetched from
/// the repo root — the SentencePiece model and the NeMo archive to extract the
/// weights from.
struct ModelSpec {
    name: &'static str,
    repo: &'static str,
    nemo: &'static str,
}

/// The known models. For now, the multilingual XLM-R punctuator (47 languages,
/// Apache-2.0).
const KNOWN_MODELS: &[ModelSpec] = &[ModelSpec {
    name: "xlmr-47lang",
    repo: "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    nemo: "pcs47_1jun.nemo",
}];

/// The published model names, for help text and error messages.
#[must_use]
pub fn known_model_names() -> Vec<&'static str> {
    KNOWN_MODELS.iter().map(|spec| spec.name).collect()
}

/// A resolved model: the directory holding `sp.model` and `model_weights.ckpt`,
/// and its canonical published name when referred to by name.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The model directory.
    pub dir: PathBuf,
    /// The canonical published name, when known.
    pub name: Option<&'static str>,
}

impl ResolvedModel {
    /// The `sp.model` path.
    #[must_use]
    pub fn spe_path(&self) -> PathBuf { self.dir.join(SPE_FILE) }
}

/// The subdirectory of the model directory that holds punctuate state.
fn feature_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("text").join("punctuate")
}

/// Resolves `model` to a directory holding `sp.model` and
/// `model_weights.ckpt`, downloading a named model into
/// `<models_dir>/text/punctuate/<name>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`PunctuateError::InvalidModel`] for an unknown name and
/// [`PunctuateError::ModelDownload`] when fetching or extracting fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, PunctuateError> {
    // A local directory that already holds the weights wins over the name
    // table.
    let as_path = Path::new(model);
    if as_path.join(WEIGHTS_FILE).is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(spec) = KNOWN_MODELS.iter().find(|spec| spec.name == model) else {
        return Err(PunctuateError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a model directory)",
            known_model_names().join(", ")
        )));
    };

    let dir = feature_dir(models_dir).join(spec.name);
    let spe = dir.join(SPE_FILE);
    let weights = dir.join(WEIGHTS_FILE);
    if spe.is_file() && weights.is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(spec.name),
        });
    }

    if !spe.is_file() {
        let url = download::hugging_face_url(spec.repo, SPE_FILE);
        Download::new(&url, &spe).fetch(&mut *progress)?;
    }
    if !weights.is_file() {
        // Fetch the NeMo archive, extract just the weights, then drop it. The
        // archive is only removed once the extraction has succeeded, so a
        // failure there does not cost the download again.
        let nemo = dir.join(spec.nemo);
        if !nemo.is_file() {
            let url = download::hugging_face_url(spec.repo, spec.nemo);
            Download::new(&url, &nemo).fetch(progress)?;
        }
        extract_weights(&nemo, &weights)?;
        let _ = fs::remove_file(&nemo);
    }

    Ok(ResolvedModel {
        dir,
        name: Some(spec.name),
    })
}

/// Extracts `model_weights.ckpt` from a NeMo archive (`*.nemo`, a plain or gzip
/// tar) into `target`.
fn extract_weights(nemo: &Path, target: &Path) -> Result<(), PunctuateError> {
    let failed = |detail: String| {
        PunctuateError::ModelDownload(format!(
            "extracting {WEIGHTS_FILE} from {}: {detail}",
            nemo.display()
        ))
    };
    // NeMo archives are usually a plain tar; support gzip too (magic 1f 8b),
    // probed on the same handle and rewound.
    let mut file = fs::File::open(nemo).map_err(|e| failed(e.to_string()))?;
    let mut magic = [0u8; 2];
    let gzip = file.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| failed(e.to_string()))?;
    let reader: Box<dyn Read> = if gzip {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);
    let entries = archive.entries().map_err(|e| failed(e.to_string()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| failed(e.to_string()))?;
        let is_weights = entry
            .path()
            .ok()
            .and_then(|p| {
                p.file_name().map(|n| n.to_string_lossy().into_owned())
            })
            .is_some_and(|name| name == WEIGHTS_FILE);
        if is_weights {
            let temp = target.with_extension("partial");
            let mut out =
                fs::File::create(&temp).map_err(|e| failed(e.to_string()))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| failed(e.to_string()))?;
            out.flush().map_err(|e| failed(e.to_string()))?;
            drop(out);
            fs::rename(&temp, target).map_err(|e| failed(e.to_string()))?;
            return Ok(());
        }
    }
    Err(failed(format!("no {WEIGHTS_FILE} member")))
}
