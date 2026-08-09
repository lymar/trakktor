//! The transforms at each end of the network, checked on their own.

use super::*;
use crate::enhance::mpsenet::config::{HOP, SAMPLE_RATE, samples};

/// A deterministic pseudo-speech signal.
fn signal(len: usize) -> Vec<f32> {
    let mut state = 0x2026_0809u32;
    (0..len)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / f32::from(u16::MAX) / 256.0 - 0.5;
            let t = index as f32 / SAMPLE_RATE as f32;
            0.6 * (2.0 * std::f32::consts::PI * 190.0 * t).sin() +
                0.25 * (2.0 * std::f32::consts::PI * 730.0 * t).sin() +
                0.1 * noise
        })
        .collect()
}

#[test]
fn the_window_is_a_periodic_hann() {
    let win = window();
    assert_eq!(win.len(), N_FFT);
    assert!(win[0].abs() < 1e-7, "a periodic Hann starts at zero");
    // Periodic, not symmetric: the peak is at the midpoint and the last
    // sample is not a mirror of the first.
    assert!((win[N_FFT / 2] - 1.0).abs() < 1e-6);
    assert!(win[N_FFT - 1] > 0.0);
}

#[test]
fn analysis_and_synthesis_come_back_to_the_signal() {
    let wave = signal(8_000);
    let spectrum = analyze(&wave);
    assert_eq!(spectrum.frames, wave.len() / HOP + 1);
    assert_eq!(spectrum.magnitude.len(), BINS * spectrum.frames);

    let back = synthesize(&spectrum);
    assert_eq!(back.len(), samples(spectrum.frames));
    // Everything but the first and last half-window, where the reflection
    // padding is what was analysed.
    let edge = N_FFT / 2;
    let worst = wave[edge..back.len() - edge]
        .iter()
        .zip(&back[edge..back.len() - edge])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 2e-4, "round trip is off by {worst}");
}

#[test]
fn the_magnitude_is_compressed_and_the_phase_is_an_angle() {
    let spectrum = analyze(&signal(4_000));
    assert!(spectrum.magnitude.iter().all(|&v| v >= 0.0));
    assert!(
        spectrum
            .phase
            .iter()
            .all(|&v| v >= -std::f32::consts::PI && v <= std::f32::consts::PI)
    );
    // A compressed magnitude is much flatter than a raw one: raising a
    // spectrum to 0.3 pulls three decades into one.
    let peak = spectrum.magnitude.iter().fold(0f32, |a, &b| a.max(b));
    let floor = spectrum.magnitude.iter().fold(f32::MAX, |a, &b| a.min(b));
    assert!(peak / floor < 1e3, "compressed range {floor}..{peak}");
}

#[test]
fn a_unit_mask_and_the_analysed_phase_reproduce_the_analysis() {
    let spectrum = analyze(&signal(4_000));
    let prediction = Prediction {
        mask: vec![1.0; spectrum.magnitude.len()],
        phase_real: spectrum.phase.iter().map(|p| p.cos()).collect(),
        phase_imag: spectrum.phase.iter().map(|p| p.sin()).collect(),
    };
    let applied = apply(&spectrum, &prediction);
    assert_eq!(applied.magnitude, spectrum.magnitude);
    // Angles, so a difference is only meaningful once it is wrapped: a bin at
    // exactly −π comes back as +π and is the same direction.
    let worst = applied
        .phase
        .iter()
        .zip(&spectrum.phase)
        .map(|(a, b)| {
            let delta = (a - b).abs() % (2.0 * std::f32::consts::PI);
            delta.min(2.0 * std::f32::consts::PI - delta)
        })
        .fold(0f32, f32::max);
    assert!(worst < 1e-6, "phase is off by {worst}");
}
