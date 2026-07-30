//! Tests for the model registry and the licence gate.

use super::{
    DEFAULT_MODEL, KNOWN_MODELS, License, permissive_model_names, resolve_model,
};
use crate::tts::silero::error::SileroError;

#[test]
fn every_published_model_is_named_once_and_pinned() {
    let mut names: Vec<&str> =
        KNOWN_MODELS.iter().map(|model| model.name).collect();
    names.sort_unstable();
    let unique = names.len();
    names.dedup();
    assert_eq!(names.len(), unique, "duplicate model names");
    for model in KNOWN_MODELS {
        assert_eq!(model.blake3.len(), 64, "{}: not a BLAKE3", model.name);
        assert!(
            model.blake3.chars().all(|c| c.is_ascii_hexdigit()),
            "{}: not hexadecimal",
            model.name
        );
        assert!(
            model.url.ends_with(model.file),
            "{}: the url and the file name disagree",
            model.name
        );
        assert!(!model.summary.is_empty(), "{}: no summary", model.name);
    }
}

#[test]
fn the_default_model_is_a_permissive_one() {
    let default = KNOWN_MODELS
        .iter()
        .find(|model| model.name == DEFAULT_MODEL)
        .expect("the default is in the registry");
    assert!(default.license.is_permissive());
}

#[test]
fn a_restricted_model_needs_the_flag_and_says_so_before_any_download() {
    let restricted = KNOWN_MODELS
        .iter()
        .find(|model| !model.license.is_permissive())
        .expect("the registry has a restricted model");
    let dir = tempfile::tempdir().expect("a temporary directory");
    let error =
        resolve_model(dir.path(), restricted.name, false, &mut |_, _, _| {
            panic!("nothing should be fetched")
        })
        .expect_err("should refuse");
    let SileroError::LicenseRestricted { model, license, .. } = &error else {
        panic!("{error}");
    };
    assert_eq!(model, restricted.name);
    assert_eq!(*license, License::NonCommercial.id());
    // The message points at what *is* available.
    assert!(error.to_string().contains(DEFAULT_MODEL), "{error}");
}

#[test]
fn an_unknown_name_lists_the_known_ones() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let error = resolve_model(dir.path(), "nope", true, &mut |_, _, _| {
        panic!("nothing should be fetched")
    })
    .expect_err("should refuse");
    assert!(matches!(error, SileroError::InvalidModel(_)), "{error}");
    assert!(error.to_string().contains(DEFAULT_MODEL), "{error}");
}

/// Converts a staged archive into the layout every later run reads.
///
/// `#[ignore]` — it needs the published `.pt` beside the directory it becomes,
/// which is where [`resolve_model`] leaves it after a download. Running this is
/// also how a model is installed from a copy already on disk, without fetching
/// ninety megabytes again.
#[test]
#[ignore = "needs a downloaded model archive"]
fn a_staged_archive_converts_into_what_the_runtime_reads() {
    let home = std::env::var("HOME").expect("HOME");
    let engine = std::path::PathBuf::from(home).join(".trakktor/tts/silero");
    let mut converted = 0;
    for model in KNOWN_MODELS {
        let dir = engine.join(model.name);
        let archive = dir.join(model.file);
        if !archive.is_file() {
            continue;
        }
        super::convert(&archive, &dir).expect("should convert");
        assert!(dir.join(super::WEIGHTS_FILE).is_file());
        let tables = crate::tts::silero::tables::Tables::load(&dir)
            .expect("should read the tables back");
        assert!(!tables.speakers.is_empty());
        // Every symbol of the alphabet string is one the model can be asked
        // for, which is what the frontend relies on.
        let index = tables.index();
        for symbol in tables.alphabet.chars() {
            assert!(index.contains_key(&symbol), "{symbol} has no id");
        }
        converted += 1;
    }
    assert!(
        converted > 0,
        "no archive staged under {}",
        engine.display()
    );
}

#[test]
fn the_permissive_names_are_the_ones_available_by_default() {
    let listed = permissive_model_names();
    for model in KNOWN_MODELS {
        assert_eq!(
            listed.contains(model.name),
            model.license.is_permissive(),
            "{}",
            model.name
        );
    }
}
