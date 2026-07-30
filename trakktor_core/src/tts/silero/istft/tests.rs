//! Tests for the inverse transform and the filterbank.

use super::{Pqmf, Window, split_spectrum};

/// A Hann window of `n` points, the shape the checkpoint carries.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / n as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

#[test]
fn the_result_is_exactly_one_hop_per_frame() {
    let window = Window {
        samples: hann(24),
        hop: 6,
    };
    let bins = 24 / 2 + 1;
    let frames = 10;
    let wave = window.inverse(
        &vec![0.0; frames * bins],
        &vec![0.0; frames * bins],
        frames,
    );
    assert_eq!(wave.len(), frames * window.hop);
}

#[test]
fn a_constant_spectrum_comes_back_as_a_steady_signal() {
    let window = Window {
        samples: hann(24),
        hop: 6,
    };
    let bins = 24 / 2 + 1;
    let frames = 12;
    // Only the DC bin is set, so every frame is the same constant. Each frame
    // is weighted by the window and the sum is divided by the window's squared
    // envelope, so what comes back is that constant times `Σw / Σw²` — the
    // point of the test is that it is *steady* wherever the envelope is
    // complete, with no ripple at the frame rate.
    let mut magnitude = vec![0.0f32; frames * bins];
    for frame in 0..frames {
        magnitude[frame * bins] = 24.0;
    }
    let wave = window.inverse(&magnitude, &vec![0.0; frames * bins], frames);
    let interior = &wave[window.n_fft()..wave.len() - window.n_fft()];
    let level = interior[0];
    assert!(level > 0.5, "the signal vanished: {level}");
    for value in interior {
        assert!((value - level).abs() < 1e-4, "{value} in {interior:?}");
    }
}

#[test]
fn the_head_splits_into_a_bounded_magnitude_and_a_phase() {
    // Two frames, three bins: the first half is a log magnitude.
    let head =
        vec![0.0, 1.0, 500.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5, 0.6];
    let (magnitude, phase) = split_spectrum(&head, 3);
    assert_eq!(magnitude.len(), 6);
    assert!((magnitude[0] - 1.0).abs() < 1e-6);
    // The ceiling keeps a runaway exponent from becoming a click.
    assert!((magnitude[2] - 100.0).abs() < 1e-6);
    assert_eq!(&phase[..3], &[0.1, 0.2, 0.3]);
}

#[test]
fn the_filterbank_decimates_by_its_band_count() {
    // A trivial one-tap filterbank: band zero passes the signal through.
    let bank = Pqmf {
        filters: vec![1.0, 0.0],
        bands: 2,
        taps: 1,
    };
    let wave: Vec<f32> = (0..10).map(|index| index as f32 / 10.0).collect();
    let low = bank.low_band(&wave);
    assert_eq!(low.len(), 10 / 2 + (10 % 2));
    assert!((low[0] - 0.0).abs() < 1e-6);
    assert!((low[1] - 0.2).abs() < 1e-6);
}
