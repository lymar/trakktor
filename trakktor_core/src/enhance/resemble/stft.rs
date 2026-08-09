//! The denoiser's analysis and synthesis, on the host and shared by the
//! runtimes — neither tensor backend has an FFT.
//!
//! This reproduces `torch.stft`/`torch.istft` in the configuration the
//! reference builds from its hop: a [`N_FFT`]-point transform every [`HOP`]
//! samples through a periodic Hann window, `center = true` (the signal is
//! reflected by half a transform at each end rather than zero-filled), and no
//! normalization.
//!
//! Two details of the reference's own handling are reproduced rather than
//! tidied, because the network was trained through them:
//!
//! - **the last frame is dropped** after the analysis, so the network sees
//!   `samples / hop` frames rather than the `1 + samples / hop` a centred
//!   analysis produces;
//! - **and put back by repeating the one before it** before the synthesis, so
//!   the transform has the frame count it needs. The tail of the result is
//!   therefore a copy, which is why the driver appends a tenth of a second of
//!   silence to every chunk before running it.

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::config::{BINS, HOP, N_FFT, frames};

/// Below this the synthesis envelope is treated as zero rather than divided by.
const ENVELOPE_FLOOR: f32 = 1e-11;

/// The analysis window: a periodic Hann.
#[must_use]
pub fn window() -> Vec<f32> {
    (0..N_FFT)
        .map(|index| {
            let phase =
                2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

/// One analysed chunk in the form the network takes it: a magnitude and the
/// two components of its phase, each `[bins, frames]` in row-major order — the
/// layout of the reference's `(f, t)` tensors.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Magnitudes, bin-major.
    pub magnitude: Vec<f32>,
    /// Cosines of the phase, bin-major.
    pub cos: Vec<f32>,
    /// Sines of it.
    pub sin: Vec<f32>,
    /// Frames analysed.
    pub frames: usize,
}

/// What the network predicts for one chunk, in the same layout: a gain per
/// point, and a rotation as the two components of a unit vector.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The magnitude mask, in `[0, 1]`.
    pub mask: Vec<f32>,
    /// The cosine of the phase correction.
    pub cos: Vec<f32>,
    /// Its sine.
    pub sin: Vec<f32>,
}

/// Analyses `samples`.
///
/// The signal is reflected by half a transform at each end, as `center = true`
/// does, so the first frame is centred on the first sample. The frame a centred
/// analysis adds at the end is dropped, as the reference drops it.
#[must_use]
pub fn analyze(samples: &[f32]) -> Spectrum {
    let count = frames(samples.len());
    let padded = reflect(samples, count);
    let win = window();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let mut magnitude = vec![0f32; BINS * count];
    let mut cos = vec![0f32; BINS * count];
    let mut sin = vec![0f32; BINS * count];
    for frame in 0..count {
        let start = frame * HOP;
        for index in 0..N_FFT {
            input[index] = padded[start + index] * win[index];
        }
        fft.process(&mut input, &mut output)
            .expect("the plan and the buffers are the same length");
        for (bin, value) in output.iter().enumerate().take(BINS) {
            let (real, imag) = (f64::from(value.re), f64::from(value.im));
            // The reference takes `abs()` and `angle()` of the complex value
            // and then the cosine and sine of that angle. Going straight to the
            // components is the same number without the round trip through an
            // arctangent — which candle does not have in the first place.
            let size = (real * real + imag * imag).sqrt();
            let at = bin * count + frame;
            magnitude[at] = size as f32;
            let (unit_cos, unit_sin) = if size > 0.0 {
                (real / size, imag / size)
            } else {
                // `atan2(0, 0)` is zero, and its cosine and sine are 1 and 0.
                (1.0, 0.0)
            };
            cos[at] = unit_cos as f32;
            sin[at] = unit_sin as f32;
        }
    }
    Spectrum {
        magnitude,
        cos,
        sin,
        frames: count,
    }
}

/// Turns what the network predicted into the spectrum the synthesis takes.
///
/// The magnitude is gated, and the phase is **rotated**: the two components the
/// network produces are a unit complex number, and multiplying by it turns the
/// phase without touching the magnitude. That is the one thing a real mask
/// cannot do.
#[must_use]
pub fn apply(noisy: &Spectrum, prediction: &Prediction) -> Spectrum {
    let magnitude = noisy
        .magnitude
        .iter()
        .zip(&prediction.mask)
        .map(|(&magnitude, &mask)| (magnitude * mask).max(0.0))
        .collect();
    let cos = noisy
        .cos
        .iter()
        .zip(&noisy.sin)
        .zip(prediction.cos.iter().zip(&prediction.sin))
        .map(|((&cos, &sin), (&cos_res, &sin_res))| {
            cos * cos_res - sin * sin_res
        })
        .collect();
    let sin = noisy
        .cos
        .iter()
        .zip(&noisy.sin)
        .zip(prediction.cos.iter().zip(&prediction.sin))
        .map(|((&cos, &sin), (&cos_res, &sin_res))| {
            sin * cos_res + cos * sin_res
        })
        .collect();
    Spectrum {
        magnitude,
        cos,
        sin,
        frames: noisy.frames,
    }
}

/// Synthesises a waveform from a magnitude and the two components of a phase.
///
/// The frame the analysis dropped is put back by repeating the last one, as the
/// reference does, so the result is `frames × hop` samples long.
#[must_use]
pub fn synthesize(spectrum: &Spectrum) -> Vec<f32> {
    let kept = spectrum.frames;
    if kept == 0 {
        return Vec::new();
    }
    // The reference pads the spectrum by one frame with `replicate` before the
    // inverse transform; the extra frame is what makes the length come out at
    // `kept × hop`.
    let count = kept + 1;
    let pad = N_FFT / 2;
    let span = N_FFT + HOP * (count - 1);
    let win = window();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(N_FFT);
    let mut input = fft.make_input_vec();
    let mut samples = fft.make_output_vec();
    let mut sum = vec![0f32; span];
    let mut envelope = vec![0f32; span];
    // The inverse transform here is unnormalized, unlike torch's.
    let scale = 1.0 / N_FFT as f32;

    for frame in 0..count {
        let source = frame.min(kept - 1);
        for (bin, value) in input.iter_mut().enumerate().take(BINS) {
            let at = bin * kept + source;
            let magnitude = spectrum.magnitude[at];
            *value = Complex32::new(
                magnitude * spectrum.cos[at],
                magnitude * spectrum.sin[at],
            );
        }
        // A real signal has no imaginary part at DC or at Nyquist, and this
        // transform refuses to run with one there.
        input[0].im = 0.0;
        if let Some(last) = input.last_mut() {
            last.im = 0.0;
        }
        fft.process(&mut input, &mut samples)
            .expect("the plan and the buffers are the same length");
        let start = frame * HOP;
        for index in 0..N_FFT {
            let weight = win[index];
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

/// `samples` with half a transform of mirrored signal at each end, the way
/// `center = true` with `pad_mode = "reflect"` does it, long enough for `count`
/// frames.
fn reflect(samples: &[f32], count: usize) -> Vec<f32> {
    let pad = N_FFT / 2;
    let span = N_FFT + HOP * count.saturating_sub(1);
    let last = samples.len().saturating_sub(1);
    let at = |index: i64| -> f32 {
        let index = if index < 0 {
            (-index) as usize
        } else if index as usize > last {
            last.saturating_sub(index as usize - last)
        } else {
            index as usize
        };
        samples.get(index).copied().unwrap_or(0.0)
    };
    (0..span)
        .map(|offset| at(offset as i64 - pad as i64))
        .collect()
}

#[cfg(test)]
mod tests;
