use super::*;

/// A filterbank that passes exactly one analysis bin into one band, so that
/// what the transform did can be read off the result.
fn single_bin(bin: usize, band: usize) -> Vec<f32> {
    let mut filterbank = vec![0f32; MEL_BINS * MELS];
    filterbank[bin * MELS + band] = 1.0;
    filterbank
}

/// A periodic Hann window, which is what torchaudio computes before the
/// checkpoint overwrites it. These tests are about the transform, not about
/// which window the reference ended up with.
fn hann() -> Vec<f32> {
    (0..MEL_N_FFT)
        .map(|index| {
            let phase =
                2.0 * std::f64::consts::PI * index as f64 / MEL_N_FFT as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

#[test]
fn the_first_difference_filter_leaves_the_first_sample_alone() {
    let samples = [1.0f32, 2.0, 3.0];
    let out = preemphasize(&samples);
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - (2.0 - 0.97)).abs() < 1e-6);
    assert!((out[2] - (3.0 - 0.97 * 2.0)).abs() < 1e-6);
}

#[test]
fn there_is_one_frame_per_hop() {
    let samples = vec![0f32; 12 * HOP + 5];
    let mel = analyze(&samples, &single_bin(0, 0), &hann());
    assert_eq!(mel.len(), MELS * 12);
}

#[test]
fn silence_sits_at_the_bottom_of_the_scale() {
    // Nothing in the band clamps to the floor, whose decibels are the bottom of
    // the normalized range — which is zero, not a negative number.
    let mel = analyze(&vec![0f32; 6 * HOP], &single_bin(3, 7), &hann());
    assert!(
        mel.iter().all(|&value| value.abs() < 1e-6),
        "silence is not at zero"
    );
}

#[test]
fn a_tone_lights_the_band_its_bin_is_wired_to() {
    // A sinusoid at the centre of an analysis bin, read through a filterbank
    // that passes only that bin into band 5 and nothing into any other.
    let bin = 64;
    let frequency = bin as f32 / MEL_N_FFT as f32;
    let samples: Vec<f32> = (0..12 * HOP)
        .map(|index| {
            (2.0 * std::f32::consts::PI * frequency * index as f32).sin()
        })
        .collect();
    let mel = analyze(&samples, &single_bin(bin, 5), &hann());
    let frames = mel.len() / MELS;
    // Away from the ends, where the zero padding cuts the tone short.
    let middle = frames / 2;
    let lit = mel[5 * frames + middle];
    assert!(lit > 0.5, "the wired band reads {lit}");
    for band in 0..MELS {
        if band == 5 {
            continue;
        }
        let dark = mel[band * frames + middle];
        assert!(dark.abs() < 1e-6, "band {band} reads {dark}");
    }
}
