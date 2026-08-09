use super::*;

#[test]
fn an_unknown_name_is_rejected_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let error = resolve_model(dir.path(), "no-such-model", &mut |_, _, _| {})
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("no-such-model"), "{message}");
    assert!(message.contains(DEFAULT_MODEL), "{message}");
}

#[test]
fn a_directory_with_weights_is_used_as_it_is() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(WEIGHTS_FILE), b"not really").unwrap();
    let resolved = resolve_model(
        dir.path(),
        dir.path().to_str().unwrap(),
        &mut |_, _, _| {},
    )
    .unwrap();
    assert_eq!(resolved.dir, dir.path());
    assert!(resolved.name.is_none());
}

#[test]
fn the_download_is_one_small_file() {
    // The whole point of this engine: the weights arrive before a progress
    // line can draw.
    assert!(download_size() < 1_000_000, "{}", download_size());
}

#[test]
fn every_batch_norm_knows_its_convolution() {
    for (norm, conv) in [
        ("encoder.en_convs.0.bn", "encoder.en_convs.0.conv"),
        (
            "encoder.en_convs.2.point_bn1",
            "encoder.en_convs.2.point_conv1",
        ),
        (
            "encoder.en_convs.2.point_bn2",
            "encoder.en_convs.2.point_conv2",
        ),
        (
            "decoder.de_convs.0.depth_bn",
            "decoder.de_convs.0.depth_conv",
        ),
    ] {
        assert_eq!(conv_for(norm).as_deref(), Some(conv));
    }
    assert!(conv_for("dpgrnn1.intra_ln").is_none());
}

/// Converts the published checkpoint into the file the runtimes load.
/// `#[ignore]` — it fetches 580 KB unless the archive is already in place.
#[test]
#[ignore = "downloads and converts the checkpoint"]
fn the_published_checkpoint_converts_with_its_batch_norms_folded() {
    let home = std::env::var("HOME").expect("HOME");
    let models = std::path::PathBuf::from(home).join(".trakktor");
    let resolved =
        resolve_model(&models, DEFAULT_MODEL, &mut |_, _, _| {}).unwrap();
    let weights = resolved.dir.join(WEIGHTS_FILE);
    let tensors =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights) }
            .unwrap();
    let names: Vec<String> = tensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    // Nothing of the batch norms survives: no statistics, no counters, and the
    // convolutions they belonged to are still there.
    assert!(
        !names.iter().any(|n| n.contains("running_")),
        "statistics left"
    );
    assert!(
        !names.iter().any(|n| n.contains("num_batches")),
        "counters left"
    );
    assert!(!names.iter().any(|n| n.contains(".bn.")), "norms left");
    assert!(names.iter().any(|n| n == "encoder.en_convs.0.conv.weight"));
    assert!(names.iter().any(|n| n == "erb.erb_fc.weight"));
}
