//! The analysis and synthesis transforms, on the host and shared by the
//! runtimes — neither tensor backend has an FFT.
//!
//! These reproduce `torch.stft`/`torch.istft` in the configuration the
//! reference uses: a 400-point transform every 100 samples through a periodic
//! Hann window, `center = true` (the signal is padded by half a transform at
//! each end and reflected there rather than zero-filled), and no
//! normalization.
//!
//! What the network is handed is not the complex spectrum but a magnitude and
//! a phase, with the magnitude raised to [`COMPRESS`](super::config::COMPRESS)
//! — and with the reference's three guard constants, which are neither
//! symmetric nor negligible: the phase's `atan2` is taken of the imaginary part
//! plus 1e-10 over the real part plus **1e-5**, and that second one is large
//! enough to rotate a quiet bin visibly. Reproduced exactly rather than
//! tidied, because every frame of the network's input goes through it.

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::config::{
    BINS, COMPRESS, HOP, MAG_EPS, N_FFT, PHASE_EPS_IMAG, PHASE_EPS_REAL, frames,
};

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

/// One analysed window, in the form the network takes it: a compressed
/// magnitude and a phase, each `[bins, frames]` in row-major order — the layout
/// of the reference's `(F, T)` tensors.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Magnitudes, bin-major, already raised to
    /// [`COMPRESS`](super::config::COMPRESS).
    pub magnitude: Vec<f32>,
    /// Phases, bin-major, in radians.
    pub phase: Vec<f32>,
    /// Frames analysed.
    pub frames: usize,
}

/// Analyses `samples`.
///
/// The signal is reflected by half a transform at each end, as `center = true`
/// does, so the first frame is centred on the first sample.
#[must_use]
pub fn analyze(samples: &[f32]) -> Spectrum {
    let count = frames(samples.len());
    let padded = reflect(samples);
    let win = window();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let mut magnitude = vec![0f32; BINS * count];
    let mut phase = vec![0f32; BINS * count];
    for frame in 0..count {
        let start = frame * HOP;
        for index in 0..N_FFT {
            input[index] = padded[start + index] * win[index];
        }
        fft.process(&mut input, &mut output)
            .expect("the plan and the buffers are the same length");
        for bin in 0..BINS {
            let (real, imag) =
                (f64::from(output[bin].re), f64::from(output[bin].im));
            let at = bin * count + frame;
            magnitude[at] = (real * real + imag * imag + MAG_EPS)
                .sqrt()
                .powf(COMPRESS) as f32;
            phase[at] =
                (imag + PHASE_EPS_IMAG).atan2(real + PHASE_EPS_REAL) as f32;
        }
    }
    Spectrum {
        magnitude,
        phase,
        frames: count,
    }
}

/// What the network predicts for one window, in the same `[bins, frames]`
/// layout: a gain per bin, and a phase as the two components an arctangent
/// takes.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The magnitude mask, in `[0, β]`.
    pub mask: Vec<f32>,
    /// The real component of the predicted phase.
    pub phase_real: Vec<f32>,
    /// Its imaginary component.
    pub phase_imag: Vec<f32>,
}

/// Turns what the network predicted into the spectrum the synthesis takes.
///
/// Two operations, and they are on the host in both runtimes on purpose. The
/// arctangent is the reason: candle has none at all, so one runtime would have
/// to approximate what the other computes exactly. Doing it here costs a pass
/// over a quarter of a million floats and makes the last step of the network
/// the same code whichever backend produced the step before it.
#[must_use]
pub fn apply(noisy: &Spectrum, prediction: &Prediction) -> Spectrum {
    Spectrum {
        magnitude: noisy
            .magnitude
            .iter()
            .zip(&prediction.mask)
            .map(|(&magnitude, &mask)| magnitude * mask)
            .collect(),
        phase: prediction
            .phase_imag
            .iter()
            .zip(&prediction.phase_real)
            .map(|(&imag, &real)| imag.atan2(real))
            .collect(),
        frames: noisy.frames,
    }
}

/// Synthesises a waveform from a magnitude and a phase in the same layout.
///
/// The magnitude is raised by the reciprocal of the compression first. The
/// result is `(frames − 1) × hop` samples long — what `torch.istft` returns for
/// a centred analysis — which is not in general the length that went in.
#[must_use]
pub fn synthesize(spectrum: &Spectrum) -> Vec<f32> {
    let count = spectrum.frames;
    let pad = N_FFT / 2;
    let span = N_FFT + HOP * count.saturating_sub(1);
    let win = window();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(N_FFT);
    let mut input = fft.make_input_vec();
    let mut samples = fft.make_output_vec();
    let mut sum = vec![0f32; span];
    let mut envelope = vec![0f32; span];
    // The inverse transform here is unnormalized, unlike torch's.
    let scale = 1.0 / N_FFT as f32;
    let expand = 1.0 / COMPRESS;

    for frame in 0..count {
        for bin in 0..BINS {
            let at = bin * count + frame;
            let magnitude =
                f64::from(spectrum.magnitude[at]).abs().powf(expand);
            let phase = f64::from(spectrum.phase[at]);
            input[bin] = Complex32::new(
                (magnitude * phase.cos()) as f32,
                (magnitude * phase.sin()) as f32,
            );
        }
        // A real signal has no imaginary part at DC or at Nyquist; the network
        // is free to predict a phase there, and this transform refuses to run
        // with the imaginary part it implies.
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
/// `center = true` with `pad_mode = "reflect"` does it.
fn reflect(samples: &[f32]) -> Vec<f32> {
    let pad = N_FFT / 2;
    let count = frames(samples.len());
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
