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
fn a_directory_without_weights_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let error = resolve_model(
        dir.path(),
        dir.path().to_str().unwrap(),
        &mut |_, _, _| {},
    )
    .unwrap_err();
    assert!(error.to_string().contains(WEIGHTS_FILE));
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
fn the_download_is_the_three_networks_that_run() {
    // The evaluation vocoder and the bandwidth extender are not fetched.
    assert_eq!(PARTS.len(), 3);
    assert_eq!(download_size(), 2_175_119_135);
    assert!(!PARTS.iter().any(|part| part.file.contains("PostNet")));
    assert!(!PARTS.iter().any(|part| part.file.contains("WavLM-L24")));
}

/// Converts the published checkpoints into the file the runtimes load.
/// `#[ignore]` — it needs the four archives (2.19 GB), either already in the
/// model directory or fetched here.
#[test]
#[ignore = "downloads and converts 2.19 GB"]
fn the_published_checkpoints_convert_into_one_file() {
    let home = std::env::var("HOME").expect("HOME");
    let models = std::path::PathBuf::from(home).join(".trakktor");
    let resolved =
        resolve_model(&models, DEFAULT_MODEL, &mut |name, done, total| {
            if let Some(total) = total {
                eprint!("\r{name}: {done}/{total}");
            }
        })
        .expect("resolving the model");
    let weights = resolved.dir.join(WEIGHTS_FILE);
    assert!(weights.is_file(), "no {}", weights.display());
    let tensors =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&weights) }
            .unwrap();
    let names: Vec<String> = tensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    // 488 + 162 + 161 tensors, less the vocoder's window, less the two halves
    // of the folded positional kernel plus the one kernel they became.
    assert_eq!(names.len(), 488 + 162 + 161 - 1 - 1);
    assert!(names.iter().any(|name| name == POS_CONV_WEIGHT));
    assert!(!names.iter().any(|name| name.ends_with("weight_g")));
}
