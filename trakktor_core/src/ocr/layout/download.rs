//! Provisioning: getting a catalogued layout model onto disk.
//!
//! Artifacts are cached under `<models_dir>/ocr/layout/<model name>/`, one
//! directory per model, holding the three files upstream publishes. Each is
//! fetched at the revision the [catalog](super::model) pins and verified
//! against the digest recorded there.
//!
//! A model may also be named by a path to a directory that already holds the
//! files; that path wins over the catalog.

use std::path::{Path, PathBuf};

use super::{
    config::CONFIG_FILE,
    model::{self, Model},
};
use crate::{
    download::{self, Download, Progress},
    ocr::{
        error::OcrError,
        paddle::artifact::{GRAPH_FILE, WEIGHTS_FILE},
    },
};

/// The subdirectory of the model directory this stage keeps its models in.
fn stage_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("ocr").join("layout")
}

/// A model that is on disk and ready to load.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub dir: PathBuf,
    /// The catalogued name, when the model came from the catalog.
    pub name: Option<&'static str>,
    /// The catalogued entry, when the model came from the catalog.
    pub spec: Option<&'static Model>,
}

/// Whether a directory looks like an unpacked model.
fn is_model_dir(dir: &Path) -> bool {
    dir.join(GRAPH_FILE).is_file() &&
        dir.join(WEIGHTS_FILE).is_file() &&
        dir.join(CONFIG_FILE).is_file()
}

/// Resolves a model name — or a path to a directory of artifacts — to a local
/// directory, downloading it on first use.
pub fn resolve(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<Resolved, OcrError> {
    let as_path = Path::new(model);
    if is_model_dir(as_path) {
        return Ok(Resolved {
            dir: as_path.to_path_buf(),
            name: None,
            spec: None,
        });
    }

    let spec = model::model(model)?;
    let dir = stage_dir(models_dir).join(spec.name);
    fetch(&dir, spec, progress)?;
    Ok(Resolved {
        dir,
        name: Some(spec.name),
        spec: Some(spec),
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

/// Total bytes still to fetch for a model, for a caller that wants to say so
/// before starting. A hundred and twenty-nine megabytes is not "the first run
/// takes a moment longer", so the size is named before the download begins.
pub fn pending_bytes(models_dir: &Path, spec: &Model) -> u64 {
    let dir = stage_dir(models_dir).join(spec.name);
    spec.files
        .iter()
        .filter(|file| !dir.join(file.name).is_file())
        .map(|file| file.size)
        .sum()
}
