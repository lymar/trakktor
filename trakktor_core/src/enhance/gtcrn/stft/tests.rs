use super::*;

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f32 / 16_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 240.0 * t).sin() +
                0.2 * (2.0 * std::f32::consts::PI * 1730.0 * t).cos()
        })
        .collect()
}

#[test]
fn the_window_is_the_square_root_of_a_periodic_hann() {
    let win = window();
    // Squared, it must be the periodic Hann itself.
    for (index, value) in win.iter().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64;
        let hann = (0.5 - 0.5 * phase.cos()) as f32;
        assert!((value * value - hann).abs() < 1e-6, "tap {index}");
    }
}

#[test]
fn a_round_trip_recovers_the_interior() {
    let original = signal(16_000);
    let spectrum = analyze(&original);
    assert_eq!(spectrum.frames, frames(original.len()));
    let recovered = synthesize(&spectrum);
    // The synthesis returns `(frames - 1) * hop`, which for a centred analysis
    // is the whole signal less its last partial frame.
    let compared = recovered.len().min(original.len());
    let worst = original[..compared]
        .iter()
        .zip(&recovered[..compared])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-4, "worst sample differs by {worst}");
}

#[test]
fn a_recording_shorter_than_the_transform_still_analyses() {
    // Reflecting a signal shorter than the padding is where torch gives up;
    // here it is an ordinary short file.
    let spectrum = analyze(&signal(100));
    assert_eq!(spectrum.frames, frames(100));
    assert_eq!(spectrum.real.len(), BINS * spectrum.frames);
}
