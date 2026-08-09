//! The way back from the vocoder's spectrum to a waveform.
//!
//! It lives on the host and is shared by the runtimes — neither tensor backend
//! has an FFT — which has the pleasant side effect that the last stage of
//! enhancement is literally the same code whichever runtime produced the
//! spectrum.
//!
//! The padding is `same`, not `center`: the transform trims half of what the
//! window overhangs the hop by from each end, which makes the result exactly
//! `frames × hop` samples long — one vocoder frame per encoder frame, so the
//! enhanced window is as long as the window that went in.
//!
//! Ported from Vocos by way of WavTokenizer (MIT).

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::config::{HOP, LOG_MAGNITUDE_RANGE, N_FFT, bins};

/// Below this the window envelope is treated as zero rather than divided by.
/// The reference asserts the envelope never gets there; the guard only keeps a
/// degenerate frame count from dividing by zero.
const ENVELOPE_FLOOR: f32 = 1e-11;

/// The periodic Hann window the head is trained with, `n_fft` long.
///
/// Read from the checkpoint rather than recomputed would be the tree's habit,
/// but this one is a plain `torch.hann_window` buffer and the conversion drops
/// it: a window that disagreed with the reference's would show up in the parity
/// check immediately, and there is nothing model-specific to preserve.
#[must_use]
pub fn hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase =
                2.0 * std::f64::consts::PI * index as f64 / length as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// Turns the head's raw output into a waveform.
///
/// `head` is the head's `[2 · bins, frames]` output in row-major order — the
/// log-magnitudes first, then the phases, the way the reference splits it.
/// The result is `frames × 320` samples long.
#[must_use]
pub fn spectrum_to_wave(head: &[f32], frames: usize) -> Vec<f32> {
    let bins = bins();
    let (low, high) = LOG_MAGNITUDE_RANGE;
    let mut magnitude = vec![0f32; bins * frames];
    let mut phase = vec![0f32; bins * frames];
    for bin in 0..bins {
        for frame in 0..frames {
            let at = bin * frames + frame;
            let clamped = f64::from(head[at]).clamp(low, high).exp() as f32;
            magnitude[frame * bins + bin] = clamped;
            phase[frame * bins + bin] = head[(bins + bin) * frames + frame];
        }
    }
    inverse(&magnitude, &phase, frames)
}

/// Overlap-adds a magnitude/phase pair back into a waveform.
///
/// Both are `[frames, bins]` in row-major order. The result is `frames × 320`
/// samples long.
#[must_use]
pub fn inverse(magnitude: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    let bins = bins();
    let pad = (N_FFT - HOP) / 2;
    let span = N_FFT + HOP * frames.saturating_sub(1);
    let window = hann(N_FFT);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(N_FFT);
    let mut spectrum = fft.make_input_vec();
    let mut samples = fft.make_output_vec();
    let mut sum = vec![0f32; span];
    let mut envelope = vec![0f32; span];
    // The inverse transform is unnormalized here, unlike torch's, so the 1/N
    // rides along with the window.
    let scale = 1.0 / N_FFT as f32;

    for frame in 0..frames {
        let row = frame * bins;
        for bin in 0..bins {
            let (sin, cos) = phase[row + bin].sin_cos();
            spectrum[bin] = Complex32::new(
                magnitude[row + bin] * cos,
                magnitude[row + bin] * sin,
            );
        }
        // A real signal has no imaginary part at DC or at Nyquist, and the
        // network is free to predict a phase there anyway. torch's inverse
        // ignores those two; this one refuses to run with them, so they are
        // dropped explicitly.
        spectrum[0].im = 0.0;
        if let Some(last) = spectrum.last_mut() {
            last.im = 0.0;
        }
        fft.process(&mut spectrum, &mut samples)
            .expect("the plan and the buffers are the same length");
        let start = frame * HOP;
        for index in 0..N_FFT {
            let weight = window[index];
            sum[start + index] += samples[index] * scale * weight;
            envelope[start + index] += weight * weight;
        }
    }

    let end = span.saturating_sub(pad);
    (pad..end)
        .map(|index| {
            let weight = envelope[index];
            if weight > ENVELOPE_FLOOR {
                sum[index] / weight
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
