//! Provisioning: getting the checkpoint onto disk.
//!
//! The weights are cached under `<models_dir>/ocr/vl/<model name>/` as the
//! four files the port reads, each fetched at the revision the
//! [catalog](super::model) pins and verified against the digest recorded there.
//!
//! A model may also be named by a path to a directory that already holds the
//! files; that path wins over the catalog, so a locally converted or patched
//! checkpoint can be run without touching the code.

use std::path::{Path, PathBuf};

use super::{
    config::{CONFIG_FILE, PREPROCESSOR_FILE, TOKENIZER_FILE, WEIGHTS_FILE},
    model::{self, Model},
};
use crate::{
    download::{self, Download, Progress},
    ocr::error::OcrError,
};

/// The subdirectory of the model directory this engine keeps its weights in.
fn engine_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("ocr").join("vl")
}

/// A checkpoint that is on disk and ready to load.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub dir: PathBuf,
    /// The catalogued name, when it came from the catalog.
    pub name: Option<&'static str>,
}

/// Whether a directory looks like an unpacked checkpoint.
fn is_model_dir(dir: &Path) -> bool {
    [CONFIG_FILE, PREPROCESSOR_FILE, TOKENIZER_FILE, WEIGHTS_FILE]
        .iter()
        .all(|name| dir.join(name).is_file())
}

/// Resolves a model name — or a path to a directory of weights — to a local
/// directory, downloading it on first use.
pub fn resolve(
    models_dir: &Path,
    name: &str,
    progress: Progress<'_>,
) -> Result<Resolved, OcrError> {
    let as_path = Path::new(name);
    if is_model_dir(as_path) {
        return Ok(Resolved {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let spec = model::model(name)?;
    let dir = engine_dir(models_dir).join(spec.name);
    fetch(&dir, spec, progress)?;
    Ok(Resolved {
        dir,
        name: Some(spec.name),
    })
}

/// Downloads whatever of `spec` is not already in `dir`.
fn fetch(
    dir: &Path,
    spec: &Model,
    progress: Progress<'_>,
) -> Result<(), OcrError> {
    for file in spec.files {
        let target = dir.join(file.name);
        if target.is_file() {
            continue;
        }
        let url = download::hugging_face_url_at(
            &format!("PaddlePaddle/{}", spec.name),
            spec.revision,
            file.name,
        );
        Download::new(&url, &target)
            .label(&format!("{}/{}", spec.name, file.name))
            .blake3(file.blake3)
            .fetch(&mut *progress)?;
    }
    Ok(())
}

/// Total bytes still to fetch, for a caller that wants to say so before
/// starting. Two gigabytes is not "the first run takes a moment", and the
/// number should be on screen before the wait, not after it.
pub fn pending_bytes(models_dir: &Path, spec: &Model) -> u64 {
    let dir = engine_dir(models_dir).join(spec.name);
    spec.files
        .iter()
        .filter(|file| !dir.join(file.name).is_file())
        .map(|file| file.size)
        .sum()
}
