//! The mel spectrogram the enhancer conditions on — a second analysis of the
//! same signal, and not the denoiser's ([`stft`](super::stft)).
//!
//! Four things about it differ from the transform next door, and every one of
//! them changes the numbers:
//!
//! - the window is [`MEL_N_FFT`] points rather than 1680, at the same hop;
//! - the signal is padded with **zeros** at both ends, not reflected;
//! - a first-difference filter runs over the waveform first;
//! - and what comes out is a logarithm, shifted and scaled into roughly `[0,
//!   1]`.
//!
//! # Both the filterbank and the window are read from the checkpoint
//!
//! The filterbank is torchaudio's Slaney-scaled matrix, and it is a persistent
//! buffer, so the published weights carry the exact matrix — less code, and one
//! fewer thing that can be subtly wrong.
//!
//! **The window is read from the checkpoint too, and that one is not an
//! economy but a correctness fix.** torchaudio computes a Hann window at full
//! precision when the module is built — and then `load_state_dict` overwrites
//! it, because the window is a persistent buffer as well and this checkpoint
//! was saved in **half** precision. So the reference does not analyse through
//! the window it computed; it analyses through that window rounded to fp16, and
//! a port that recomputes the exact one is a third of a decibel out across the
//! whole spectrum. Reading it back is the only way to reproduce what the
//! network was actually trained and run against.
//!
//! (The denoiser's own transform ([`stft`](super::stft)) is not affected: it
//! builds its window inline on every call rather than holding it as a buffer,
//! so nothing overwrites it and the exact one is right there.)

use realfft::RealFftPlanner;

use super::config::{
    HOP, MEL_HEADROOM_DB, MEL_MAGNITUDE_MIN, MEL_N_FFT, MELS, PREEMPHASIS,
    frames,
};

/// Frequency bins the mel transform's own analysis has, and the height of the
/// filterbank matrix.
pub const MEL_BINS: usize = MEL_N_FFT / 2 + 1;

/// The mel spectrogram of `samples`, `[MELS, frames]` in row-major order —
/// band-major, matching the reference's `(d, t)`.
///
/// `filterbank` is the checkpoint's own matrix, `[MEL_BINS, MELS]` row-major,
/// and `window` its own analysis window, [`MEL_N_FFT`] long. The frame a
/// centred analysis adds at the end is dropped, as the reference drops it.
#[must_use]
pub fn analyze(
    samples: &[f32],
    filterbank: &[f32],
    window: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(filterbank.len(), MEL_BINS * MELS);
    debug_assert_eq!(window.len(), MEL_N_FFT);
    let count = frames(samples.len());
    let emphasized = preemphasize(samples);
    let win = window;
    let pad = MEL_N_FFT / 2;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(MEL_N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    // The floor and the shift, from the reference: 20·log10 of the floor is the
    // bottom of the scale, and the range is that plus the headroom.
    let floor_db = 20.0 * f64::from(MEL_MAGNITUDE_MIN).log10();
    let range = (-floor_db + f64::from(MEL_HEADROOM_DB)) as f32;

    let mut mel = vec![0f32; MELS * count];
    let mut spectrum = vec![0f32; MEL_BINS];
    for frame in 0..count {
        // Zero padding, not reflection: the reference asks torchaudio for
        // `pad_mode="constant"` to follow librosa's default.
        let start = (frame * HOP) as i64 - pad as i64;
        for index in 0..MEL_N_FFT {
            let at = start + index as i64;
            let sample = if at < 0 {
                0.0
            } else {
                emphasized.get(at as usize).copied().unwrap_or(0.0)
            };
            input[index] = sample * win[index];
        }
        fft.process(&mut input, &mut output)
            .expect("the plan and the buffers are the same length");
        for (bin, value) in spectrum.iter_mut().enumerate() {
            *value = output[bin].norm();
        }
        for band in 0..MELS {
            let mut sum = 0f32;
            for (bin, &magnitude) in spectrum.iter().enumerate() {
                sum += magnitude * filterbank[bin * MELS + band];
            }
            let clamped = sum.max(MEL_MAGNITUDE_MIN);
            let decibels = 20.0 * clamped.log10();
            mel[band * count + frame] = (decibels - floor_db as f32) / range;
        }
    }
    mel
}

/// The reference's first-difference filter: `x[n] − 0.97·x[n−1]`, with a zero
/// before the first sample.
#[must_use]
fn preemphasize(samples: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut previous = 0f32;
    for &sample in samples {
        out.push(sample - PREEMPHASIS * previous);
        previous = sample;
    }
    out
}

#[cfg(test)]
mod tests;
