use super::*;

/// A smooth, non-trivial signal to analyse.
fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let phase = index as f32;
            0.5 * (phase * 0.013).sin() +
                0.25 * (phase * 0.101).cos() +
                0.1 * (phase * 0.7).sin()
        })
        .collect()
}

#[test]
fn the_window_is_a_periodic_hann() {
    let win = window();
    assert_eq!(win.len(), N_FFT);
    assert!(win[0].abs() < 1e-7, "it starts at zero");
    // Periodic rather than symmetric: the peak is at the exact centre and the
    // last sample is not zero.
    assert!((win[N_FFT / 2] - 1.0).abs() < 1e-6);
    assert!(win[N_FFT - 1] > 0.0);
}

#[test]
fn the_analysis_keeps_one_frame_per_hop() {
    let samples = signal(20 * HOP + 13);
    let spectrum = analyze(&samples);
    assert_eq!(spectrum.frames, 20);
    assert_eq!(spectrum.magnitude.len(), BINS * 20);
    assert_eq!(spectrum.cos.len(), BINS * 20);
    assert_eq!(spectrum.sin.len(), BINS * 20);
}

#[test]
fn the_phase_components_are_a_unit_vector() {
    let spectrum = analyze(&signal(8 * HOP));
    for (&cos, &sin) in spectrum.cos.iter().zip(&spectrum.sin) {
        let size = cos.hypot(sin);
        assert!((size - 1.0).abs() < 1e-5, "{cos},{sin} is not a direction");
    }
}

#[test]
fn doing_nothing_to_the_spectrum_gives_the_recording_back() {
    // The transform pair on its own, with the network replaced by an identity:
    // a gain of one and a rotation of zero degrees.
    let samples = signal(24 * HOP);
    let spectrum = analyze(&samples);
    let identity = Prediction {
        mask: vec![1.0; spectrum.magnitude.len()],
        cos: vec![1.0; spectrum.magnitude.len()],
        sin: vec![0.0; spectrum.magnitude.len()],
    };
    let out = synthesize(&apply(&spectrum, &identity));
    assert_eq!(out.len(), 24 * HOP);
    // The two ends are not reconstructible: the analysis reflects the signal at
    // the front, and the reference's synthesis repeats the last frame it kept
    // rather than analysing the one it dropped.
    let interior = N_FFT..out.len() - N_FFT;
    let worst = interior
        .clone()
        .map(|index| (out[index] - samples[index]).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "worst difference inside the recording: {worst}"
    );
}

#[test]
fn a_rotation_moves_the_phase_and_leaves_the_magnitude_alone() {
    // What a real mask cannot do, and the reason this network has two extra
    // planes: a quarter turn changes the waveform without touching the
    // spectrum's magnitude.
    let spectrum = analyze(&signal(16 * HOP));
    let count = spectrum.magnitude.len();
    let quarter = Prediction {
        mask: vec![1.0; count],
        cos: vec![0.0; count],
        sin: vec![1.0; count],
    };
    let turned = apply(&spectrum, &quarter);
    let worst = turned
        .magnitude
        .iter()
        .zip(&spectrum.magnitude)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-6, "the magnitude moved by {worst}");
    // sin(φ + π/2) = cos φ, and cos(φ + π/2) = −sin φ.
    for index in 0..count {
        assert!((turned.cos[index] + spectrum.sin[index]).abs() < 1e-6);
        assert!((turned.sin[index] - spectrum.cos[index]).abs() < 1e-6);
    }
}

#[test]
fn a_recording_of_nothing_analyses_and_synthesises_to_nothing() {
    let spectrum = analyze(&vec![0f32; 8 * HOP]);
    assert!(spectrum.magnitude.iter().all(|&m| m == 0.0));
    // `atan2(0, 0)` is zero, whose cosine is one.
    assert!(spectrum.cos.iter().all(|&c| c == 1.0));
    assert!(spectrum.sin.iter().all(|&s| s == 0.0));
    let out = synthesize(&spectrum);
    assert!(out.iter().all(|&sample| sample == 0.0));
}

#[test]
fn a_recording_too_short_for_a_frame_yields_none() {
    let spectrum = analyze(&signal(HOP - 1));
    assert_eq!(spectrum.frames, 0);
    assert!(synthesize(&spectrum).is_empty());
}
