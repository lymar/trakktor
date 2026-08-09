use realfft::RealFftPlanner;

use super::*;

/// The forward half of the same transform, for a round trip: `same` padding,
/// one frame per hop, the window this module inverts with.
fn forward(samples: &[f32], frames: usize) -> (Vec<f32>, Vec<f32>) {
    let pad = (N_FFT - HOP) / 2;
    let window = hann(N_FFT);
    let mut padded = vec![0f32; pad];
    padded.extend_from_slice(samples);
    padded.resize(pad + samples.len() + N_FFT, 0.0);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let bins = bins();
    let mut magnitude = vec![0f32; frames * bins];
    let mut phase = vec![0f32; frames * bins];
    for frame in 0..frames {
        let start = frame * HOP;
        for index in 0..N_FFT {
            input[index] = padded[start + index] * window[index];
        }
        fft.process(&mut input, &mut output).unwrap();
        for bin in 0..bins {
            magnitude[frame * bins + bin] = output[bin].norm();
            phase[frame * bins + bin] = output[bin].arg();
        }
    }
    (magnitude, phase)
}

fn signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f32 / 16_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin() +
                0.2 * (2.0 * std::f32::consts::PI * 1310.0 * t).cos()
        })
        .collect()
}

#[test]
fn the_inverse_returns_one_hop_per_frame() {
    let frames = 7;
    let wave = inverse(
        &vec![0.0; frames * bins()],
        &vec![0.0; frames * bins()],
        frames,
    );
    assert_eq!(wave.len(), frames * HOP);
}

#[test]
fn a_round_trip_recovers_the_signal() {
    let frames = 12;
    let original = signal(frames * HOP);
    let (magnitude, phase) = forward(&original, frames);
    let recovered = inverse(&magnitude, &phase, frames);
    assert_eq!(recovered.len(), original.len());
    let worst = original
        .iter()
        .zip(&recovered)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-4, "worst sample differs by {worst}");
}

#[test]
fn the_head_split_clamps_the_log_magnitudes() {
    let frames = 3;
    let bins = bins();
    // Log-magnitudes far above the ceiling, phases all zero.
    let mut head = vec![0f32; 2 * bins * frames];
    for value in head.iter_mut().take(bins * frames) {
        *value = 40.0;
    }
    let wave = spectrum_to_wave(&head, frames);
    assert_eq!(wave.len(), frames * HOP);
    let peak = wave.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    // exp(5) per bin, not exp(40): the clamp is what keeps this finite.
    assert!(peak.is_finite() && peak < 1e5, "peak was {peak}");
}

#[test]
fn the_window_is_periodic_not_symmetric() {
    let window = hann(8);
    assert!(window[0].abs() < 1e-7);
    // A symmetric Hann would return to zero at the last tap; a periodic one
    // does not.
    assert!(window[7] > 0.1);
}
