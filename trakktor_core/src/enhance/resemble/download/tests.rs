use candle_core::Device;

use super::*;

#[test]
fn both_engines_share_one_directory() {
    let dir = model_dir(Path::new("/models"));
    assert!(dir.ends_with("enhance/resemble"));
    // One download, two converted files — the denoiser is inside the
    // enhancer's checkpoint, so there is nothing separate to fetch for it.
    assert_ne!(DENOISER_FILE, ENHANCER_FILE);
}

#[test]
fn an_unknown_name_is_refused_before_anything_is_fetched() {
    let temp = std::env::temp_dir().join("trakktor-resemble-unknown");
    let error =
        resolve_model(&temp, "not-a-model", DENOISER_FILE, &mut |_, _, _| {})
            .expect_err("an unknown name is refused");
    assert!(matches!(error, EnhanceError::InvalidModel(_)));
    assert!(
        !temp.exists(),
        "nothing was created for a name that is wrong"
    );
}

#[test]
fn a_directory_without_the_file_the_caller_wants_is_refused() {
    let temp = std::env::temp_dir().join("trakktor-resemble-empty");
    std::fs::create_dir_all(&temp).expect("a scratch directory");
    let error = resolve_model(
        Path::new("/models"),
        temp.to_str().expect("a utf-8 path"),
        ENHANCER_FILE,
        &mut |_, _, _| {},
    )
    .expect_err("an empty directory is refused");
    assert!(matches!(error, EnhanceError::InvalidModel(_)));
    let _ = std::fs::remove_dir(&temp);
}

#[test]
fn folding_a_normalized_kernel_gives_it_the_length_it_stores() {
    // Two output channels, two inputs, three taps. Whatever the direction, the
    // folded kernel's norm over everything but the output axis is the stored
    // length.
    let direction = Tensor::from_vec(
        (0..12).map(|v| (v as f32) - 5.0).collect::<Vec<_>>(),
        (2, 2, 3),
        &Device::Cpu,
    )
    .expect("a direction");
    let length = Tensor::from_vec(vec![2.0f32, 0.5], (2, 1, 1), &Device::Cpu)
        .expect("a length");
    let folded = fold(&direction, &length).expect("the fold");
    let values = folded
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("the folded values");
    for (channel, &want) in [2.0f32, 0.5].iter().enumerate() {
        let norm: f32 = values[channel * 6..(channel + 1) * 6]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - want).abs() < 1e-5,
            "channel {channel}: {norm} is not {want}"
        );
    }
}

#[test]
fn the_pair_the_runtimes_read_is_a_mean_and_a_standard_deviation() {
    let mut raw = HashMap::new();
    raw.insert(
        "normalizer.running_mean_unsafe".to_owned(),
        Tensor::from_vec(vec![0.25f32], 1, &Device::Cpu).expect("a mean"),
    );
    raw.insert(
        "normalizer.running_var_unsafe".to_owned(),
        Tensor::from_vec(vec![4.0f32], 1, &Device::Cpu).expect("a variance"),
    );
    let pair = centre(&raw)
        .and_then(|t| {
            t.to_vec1::<f32>().map_err(|e| {
                EnhanceError::Checkpoint(format!("reading it back: {e}"))
            })
        })
        .expect("the centring");
    assert!((pair[0] - 0.25).abs() < 1e-6);
    assert!((pair[1] - 2.0).abs() < 1e-5, "the variance is not rooted");
}

#[test]
fn a_checkpoint_that_never_estimated_its_centring_is_refused() {
    // The reference starts those two buffers at NaN and fills them in during
    // training; a checkpoint that still holds NaN would silently poison every
    // mel it centres.
    let mut raw = HashMap::new();
    raw.insert(
        "normalizer.running_mean_unsafe".to_owned(),
        Tensor::from_vec(vec![f32::NAN], 1, &Device::Cpu).expect("a mean"),
    );
    raw.insert(
        "normalizer.running_var_unsafe".to_owned(),
        Tensor::from_vec(vec![f32::NAN], 1, &Device::Cpu).expect("a variance"),
    );
    assert!(matches!(centre(&raw), Err(EnhanceError::Checkpoint(_))));
}
