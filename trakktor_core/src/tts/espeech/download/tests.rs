use super::{KNOWN_MODELS, convert_checkpoint, resolve_model};

/// Where the miniature published-layout checkpoints live.
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/espeech/fixture")
        .join(name)
}

#[test]
#[ignore = "needs tmp/espeech/fixture (a miniature of the published layout)"]
fn conversion_keeps_the_moving_average_and_drops_its_bookkeeping() {
    let out =
        std::env::temp_dir().join("trakktor-espeech-converted.safetensors");
    let _ = std::fs::remove_file(&out);
    convert_checkpoint(&fixture("mini.pt"), &out).expect("convert");

    let shapes = crate::tts::espeech::runtime::shapes(&out).expect("shapes");
    let names: Vec<&str> = shapes.keys().map(String::as_str).collect();
    // The `ema_model.` prefix is stripped, and `initted`/`step` are not
    // weights.
    assert_eq!(
        names,
        vec!["transformer.proj_out.bias", "transformer.proj_out.weight"]
    );
    assert_eq!(
        shapes["transformer.proj_out.weight"],
        vec![2, 2usize.pow(1) + 1]
    );

    // The values are the average, not the live weights: the fixture's average
    // is twice the model's.
    // SAFETY: read-only mapping of a file nothing mutates.
    let mapped = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&out).expect("open")
    };
    let values: Vec<f32> = mapped
        .load("transformer.proj_out.weight", &candle_core::Device::Cpu)
        .and_then(|t| t.flatten_all()?.to_vec1())
        .expect("values");
    assert_eq!(values, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    let _ = std::fs::remove_file(&out);
}

#[test]
#[ignore = "needs tmp/espeech/fixture (a miniature of the published layout)"]
fn conversion_refuses_a_checkpoint_it_does_not_recognize() {
    let out =
        std::env::temp_dir().join("trakktor-espeech-rejected.safetensors");
    // Half precision would convert into something that loads and sounds wrong,
    // so it is refused rather than accepted.
    let error =
        convert_checkpoint(&fixture("mini_f16.pt"), &out).expect_err("f16");
    assert!(error.to_string().contains("expected f32"), "{error}");

    // A checkpoint saved without the average is not the one the reference runs.
    let error = convert_checkpoint(&fixture("mini_noema.pt"), &out)
        .expect_err("no ema");
    assert!(
        error.to_string().contains("ema_model_state_dict"),
        "{error}"
    );
    assert!(!out.exists(), "nothing should have been written");
}

#[test]
fn every_variant_names_its_own_checkpoint_file() {
    for model in KNOWN_MODELS {
        assert!(
            model.file.ends_with(".pt"),
            "{} points at {}",
            model.name,
            model.file
        );
        assert!(model.repo.starts_with("ESpeech/"), "{}", model.repo);
    }
    let names: Vec<&str> = KNOWN_MODELS.iter().map(|m| m.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), names.len(), "duplicate names in {names:?}");
}

#[test]
fn the_default_is_the_first_entry() {
    assert_eq!(KNOWN_MODELS[0].name, "rl-v2");
}

#[test]
fn an_unknown_name_lists_the_known_ones_and_downloads_nothing() {
    let dir = std::env::temp_dir().join("trakktor-espeech-unknown");
    let mut calls = 0;
    let error = resolve_model(&dir, "no-such-model", &mut |_, _, _| calls += 1)
        .expect_err("should fail");
    let message = error.to_string();
    for model in KNOWN_MODELS {
        assert!(message.contains(model.name), "{message}");
    }
    assert_eq!(calls, 0, "an unknown name must not start a download");
    assert!(!dir.exists(), "an unknown name must not create directories");
}
