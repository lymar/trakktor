use super::*;

fn no_progress() -> impl FnMut(&str, u64, Option<u64>) {
    |_: &str, _: u64, _: Option<u64>| {}
}

#[test]
fn a_checkpoint_directory_resolves_as_is() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), "{}").unwrap();

    let resolved = resolve_model(
        Path::new("unused-work-dir"),
        dir.path().to_str().unwrap(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(resolved.dir, dir.path());
    assert_eq!(resolved.name, None);
}

#[test]
fn unknown_names_fail_with_the_known_list() {
    let err = resolve_model(
        Path::new("unused-work-dir"),
        "no-such-model",
        &mut no_progress(),
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(matches!(err, WhisperError::InvalidModel(_)));
    assert!(message.contains("tiny"));
    assert!(message.contains("turbo"));
}

#[test]
fn cached_models_resolve_without_downloading() {
    let work_dir = tempfile::tempdir().unwrap();
    let model_dir = work_dir.path().join("asr").join("whisper").join("tiny");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    std::fs::write(model_dir.join("model.safetensors"), "x").unwrap();

    let resolved =
        resolve_model(work_dir.path(), "tiny", &mut no_progress()).unwrap();
    assert_eq!(resolved.dir, model_dir);
    assert_eq!(resolved.name, Some("tiny"));
}

#[test]
fn cached_sharded_models_resolve_without_downloading() {
    let work_dir = tempfile::tempdir().unwrap();
    let model_dir =
        work_dir.path().join("asr").join("whisper").join("podlodka");
    std::fs::create_dir_all(&model_dir).unwrap();
    let entry = KNOWN_MODELS
        .iter()
        .find(|entry| entry.name == "podlodka")
        .unwrap();
    assert!(entry.files.len() > 2, "podlodka ships sharded weights");
    for file in entry.files {
        std::fs::write(model_dir.join(file), "x").unwrap();
    }

    let resolved =
        resolve_model(work_dir.path(), "podlodka", &mut no_progress()).unwrap();
    assert_eq!(resolved.dir, model_dir);
    assert_eq!(resolved.name, Some("podlodka"));
}

#[test]
fn aliases_share_the_canonical_repository() {
    let repo_of = |name: &str| {
        KNOWN_MODELS
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.repo)
            .unwrap()
    };
    assert_eq!(repo_of("turbo"), repo_of("large-v3-turbo"));
    assert_eq!(repo_of("large"), repo_of("large-v3"));
}

#[test]
fn every_model_downloads_a_config_and_weights() {
    for entry in KNOWN_MODELS {
        assert!(
            entry.files.contains(&"config.json"),
            "{}: checkpoint must include config.json",
            entry.name
        );
        assert!(
            entry.files.iter().any(|f| f.ends_with(".safetensors")),
            "{}: checkpoint must include safetensors weights",
            entry.name
        );
    }
}
