use super::{MelBasis, reflect_pad};
use crate::tts::espeech::config::{HOP, N_FFT};

/// A basis with a flat window and a single pass-through band: enough to check
/// the framing and the round trip without a checkpoint.
fn flat_basis() -> MelBasis {
    let bins = N_FFT / 2 + 1;
    MelBasis::new(vec![1.0; N_FFT], vec![1.0; bins], 1).expect("basis")
}

#[test]
fn framing_matches_the_centred_transform() {
    assert_eq!(MelBasis::frames_for(216_480), 846);
    assert_eq!(MelBasis::frames_for(HOP), 2);
    assert_eq!(MelBasis::frames_for(0), 1);
}

#[test]
fn reflection_mirrors_without_repeating_the_edge() {
    let padded = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2);
    assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn a_short_signal_is_refused_rather_than_padded_out_of_shape() {
    let error = flat_basis()
        .log_mel(&vec![0.0; N_FFT / 2])
        .expect_err("short");
    assert!(error.to_string().contains("too short"), "{error}");
}

#[test]
fn the_spectrogram_has_a_row_per_frame() {
    let wave: Vec<f32> = (0..4096)
        .map(|index| (index as f32 * 0.05).sin() * 0.5)
        .collect();
    let mel = flat_basis().log_mel(&wave).expect("mel");
    assert_eq!(mel.len(), MelBasis::frames_for(wave.len()));
    // Silence would sit at the floor; a real signal has to be above it.
    let floor = (1e-5f64).ln() as f32;
    assert!(mel.iter().any(|value| *value > floor + 1.0));
}

#[test]
fn the_inverse_transform_reconstructs_a_windowed_signal() {
    // With a Hann window on both sides, analysis followed by overlap-add
    // synthesis returns the signal itself — that is the property the vocoder's
    // inverse relies on.
    let window: Vec<f32> = (0..N_FFT)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / N_FFT as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect();
    let bins = N_FFT / 2 + 1;
    let basis =
        MelBasis::new(window.clone(), vec![1.0; bins], 1).expect("basis");

    let samples = HOP * 40;
    let wave: Vec<f32> = (0..samples)
        .map(|index| (index as f32 * 0.031).sin() * 0.3)
        .collect();

    // Analyse by hand: the forward transform of each frame, kept as magnitude
    // and phase, which is what the vocoder's head predicts.
    let frames = MelBasis::frames_for(samples);
    let padded = reflect_pad(&wave, N_FFT / 2);
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut magnitude = Vec::with_capacity(frames * bins);
    let mut phase = Vec::with_capacity(frames * bins);
    for frame in 0..frames {
        for (index, slot) in input.iter_mut().enumerate() {
            *slot = padded[frame * HOP + index] * window[index];
        }
        fft.process(&mut input, &mut spectrum).expect("fft");
        for value in &spectrum {
            magnitude.push(value.norm());
            phase.push(value.im.atan2(value.re));
        }
    }

    let back = basis.istft(&magnitude, &phase, frames);
    assert_eq!(back.len(), HOP * (frames - 1));
    // The reconstruction covers the original minus the last partial hop.
    let compared = back.len().min(wave.len());
    let worst = (0..compared)
        .map(|index| (back[index] - wave[index]).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "worst deviation {worst}");
}
