//! Provisioning: getting a catalogued model onto disk.
//!
//! Artifacts are cached under `<models_dir>/ocr/paddle/<model name>/`, one
//! directory per model, holding the three files PaddleOCR publishes. Each is
//! fetched at the revision the [catalog](super::model) pins and verified
//! against the digest recorded there — a moving branch upstream is exactly the
//! situation the digest exists for.
//!
//! A model may also be named by a path to a directory that already holds the
//! files; that path wins over the catalog, so a locally built or patched model
//! can be run without touching the code.

use std::path::{Path, PathBuf};

use super::{
    artifact::{GRAPH_FILE, WEIGHTS_FILE},
    config::CONFIG_FILE,
    model::{self, Model},
    pipeline::Options,
};
use crate::{
    download::{self, Download, Progress},
    ocr::error::OcrError,
};

/// The subdirectory of the model directory this engine keeps its models in.
fn engine_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("ocr").join("paddle")
}

/// A model that is on disk and ready to load.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// The directory holding the three artifact files.
    pub dir: PathBuf,
    /// The catalogued name, when the model came from the catalog.
    pub name: Option<&'static str>,
}

/// Whether a directory looks like an unpacked model.
fn is_model_dir(dir: &Path) -> bool {
    dir.join(GRAPH_FILE).is_file() &&
        dir.join(WEIGHTS_FILE).is_file() &&
        dir.join(CONFIG_FILE).is_file()
}

/// Resolves a model name — or a path to a directory of artifacts — to a local
/// directory, downloading it on first use.
///
/// `progress` is called as the download advances with `(file name, bytes done,
/// bytes total when known)`.
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
        });
    }

    let spec = model::model(model)?;
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
        // Revision-pinned: the published branch moves, and a re-publish must
        // not change what this build runs.
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
/// before starting.
pub fn pending_bytes(models_dir: &Path, spec: &Model) -> u64 {
    let dir = engine_dir(models_dir).join(spec.name);
    spec.files
        .iter()
        .filter(|file| !dir.join(file.name).is_file())
        .map(|file| file.size)
        .sum()
}

/// What a run has yet to fetch, model by model, so that the caller can say it
/// before the first byte moves.
///
/// The engine's defaults are the strongest models the catalog has rather than
/// the cheapest, which makes a first run a download of tens of megabytes where
/// it used to be a few. That is worth announcing rather than discovering: a
/// caller with no models on disk gets the whole bill up front, and one with
/// them all gets an empty list.
///
/// A model named by a path is not counted — there is nothing to fetch for it —
/// and neither is a name outside the catalog, which fails later, where the
/// error can say so properly.
pub fn pending(
    models_dir: &Path,
    options: &Options,
) -> Vec<(&'static str, u64)> {
    let orientation = options.orientation.then_some(model::DEFAULT_ORIENTATION);
    [
        Some(options.detector()),
        options.recognizer().ok(),
        orientation,
    ]
    .into_iter()
    .flatten()
    .flat_map(|name| pending_named(models_dir, name))
    .collect()
}

/// What one catalogued model still costs to fetch. Empty when it is on disk,
/// or when the name is a path or outside the catalog.
///
/// The generative engine borrows this engine's detector, so it asks about it
/// here rather than keeping a second answer.
pub fn pending_named(
    models_dir: &Path,
    model: &str,
) -> Vec<(&'static str, u64)> {
    let Ok(spec) = model::model(model) else {
        return Vec::new();
    };
    match pending_bytes(models_dir, spec) {
        0 => Vec::new(),
        bytes => vec![(spec.name, bytes)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_of_artifacts_resolves_to_itself() {
        let dir = tempfile::tempdir().unwrap();
        for name in [GRAPH_FILE, WEIGHTS_FILE, CONFIG_FILE] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        let mut noop = |_: &str, _: u64, _: Option<u64>| {};
        let resolved = resolve(
            Path::new("/nonexistent"),
            dir.path().to_str().unwrap(),
            &mut noop,
        )
        .unwrap();
        assert_eq!(resolved.dir, dir.path());
        assert!(resolved.name.is_none());
    }

    #[test]
    fn an_unknown_name_is_reported_with_the_catalog() {
        let mut noop = |_: &str, _: u64, _: Option<u64>| {};
        let error =
            resolve(Path::new("/nonexistent"), "no-such-model", &mut noop)
                .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("no-such-model"));
        assert!(message.contains(model::MOBILE_DETECTION));
    }

    #[test]
    fn a_first_run_is_told_every_model_it_will_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let options = Options {
            orientation: true,
            ..Options::default()
        };
        let pending = pending(dir.path(), &options);
        let names: Vec<&str> = pending.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            vec![
                model::SERVER_DETECTION,
                "en_PP-OCRv5_mobile_rec",
                model::DEFAULT_ORIENTATION,
            ]
        );
        // The bill is the published size of the artifacts, not a guess.
        let total: u64 = pending.iter().map(|(_, bytes)| bytes).sum();
        assert_eq!(
            total,
            [
                model::SERVER_DETECTION,
                "en_PP-OCRv5_mobile_rec",
                model::DEFAULT_ORIENTATION
            ]
            .iter()
            .map(|name| model::model(name).unwrap().size())
            .sum::<u64>()
        );
    }

    #[test]
    fn a_model_already_on_disk_is_not_announced() {
        let dir = tempfile::tempdir().unwrap();
        let detector = model::model(model::SERVER_DETECTION).unwrap();
        let on_disk = engine_dir(dir.path()).join(detector.name);
        std::fs::create_dir_all(&on_disk).unwrap();
        for file in detector.files {
            std::fs::write(on_disk.join(file.name), b"x").unwrap();
        }
        let pending = pending(dir.path(), &Options::default());
        assert!(
            !pending.iter().any(|(name, _)| *name == detector.name),
            "{pending:?}"
        );
    }
}
