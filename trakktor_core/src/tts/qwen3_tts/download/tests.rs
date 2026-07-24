//! Resolution tests. Nothing here touches the network: a name that is not
//! already cached would download, so the tests cover only the paths that
//! resolve or fail before any request.

use super::*;

/// A progress sink for calls that must not reach the network.
fn no_progress() -> impl FnMut(&str, u64, Option<u64>) {
    |_: &str, _: u64, _: Option<u64>| {
        panic!("no download expected in this test");
    }
}

#[test]
fn a_local_checkpoint_directory_wins_over_the_name_table() {
    let temp = tempfile::tempdir().expect("temp dir");
    let checkpoint = temp.path().join("my-checkpoint");
    fs::create_dir_all(&checkpoint).expect("create");
    fs::write(checkpoint.join("config.json"), "{}").expect("write");

    let resolved = resolve_model(
        temp.path(),
        checkpoint.to_str().expect("utf-8"),
        &mut no_progress(),
    )
    .expect("resolves");

    assert_eq!(resolved.dir, checkpoint);
    assert!(resolved.known.is_none());
    // An unnamed checkpoint reports the directory it came from.
    assert_eq!(resolved.label(), checkpoint.display().to_string());
}

#[test]
fn an_unknown_name_fails_before_any_download() {
    let temp = tempfile::tempdir().expect("temp dir");

    let err = resolve_model(temp.path(), "no-such-model", &mut no_progress())
        .unwrap_err();

    let Qwen3TtsError::InvalidModel(message) = err else {
        panic!("expected InvalidModel, got {err:?}");
    };
    // The message names what is on offer, so a typo is self-correcting.
    assert!(message.contains("0.6b-customvoice"), "{message}");
}

#[test]
fn a_cached_checkpoint_resolves_without_downloading() {
    let temp = tempfile::tempdir().expect("temp dir");
    let dir = engine_dir(temp.path()).join("0.6b-customvoice");
    for file in REQUIRED_FILES {
        let target = dir.join(file);
        fs::create_dir_all(target.parent().expect("parent")).expect("create");
        fs::write(&target, "{}").expect("write");
    }

    let resolved =
        resolve_model(temp.path(), "0.6b-customvoice", &mut no_progress())
            .expect("resolves");

    assert_eq!(resolved.dir, dir);
    assert_eq!(resolved.label(), "0.6b-customvoice");
    assert_eq!(
        resolved.known.expect("known").repo,
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    );
}

#[test]
fn every_known_model_has_a_distinct_name_and_repo() {
    for (index, model) in KNOWN_MODELS.iter().enumerate() {
        for other in &KNOWN_MODELS[index + 1..] {
            assert_ne!(model.name, other.name, "duplicate name");
            assert_ne!(model.repo, other.repo, "duplicate repo");
        }
    }
}
