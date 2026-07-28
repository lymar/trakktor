//! The mel spectrogram, and its way back to a waveform.
//!
//! Both transforms live on the host and are shared by the runtimes, for two
//! different reasons. The forward one is cheap — one short-time transform over
//! at most twelve seconds of reference audio — and keeping it in one place
//! means the two runtimes cannot disagree about the model's input. The inverse
//! one has no choice: neither tensor backend has an FFT, so the vocoder's head
//! hands over magnitudes and phases and the overlap-add happens here.
//!
//! The triangular filterbank and the analysis window are **not** rebuilt from
//! their definitions: they ship inside the vocoder checkpoint, which the
//! reference builds with the very same parameters. Reading them removes a whole
//! class of near-miss (a mel scale that is off by a rounding rule sounds fine
//! and matches nothing).

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::{
    config::{HOP, MEL_FLOOR, N_FFT},
    error::EspeechError,
};

/// The analysis basis of one loaded vocoder: the window and the filterbank.
#[derive(Debug, Clone)]
pub struct MelBasis {
    /// The analysis window, `N_FFT` long.
    pub window: Vec<f32>,
    /// The triangular filterbank, `[n_freqs, n_mels]` in row-major order.
    pub filters: Vec<f32>,
    /// Mel bands the filterbank produces.
    pub n_mels: usize,
}

impl MelBasis {
    /// Wraps the two arrays a vocoder checkpoint carries, checking their shapes
    /// against the transform they are for.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] when the shapes do not line up with
    /// `N_FFT`.
    pub fn new(
        window: Vec<f32>,
        filters: Vec<f32>,
        n_mels: usize,
    ) -> Result<Self, EspeechError> {
        let n_freqs = N_FFT / 2 + 1;
        if window.len() != N_FFT {
            return Err(EspeechError::Checkpoint(format!(
                "the vocoder's analysis window is {} long, expected {N_FFT}",
                window.len()
            )));
        }
        if filters.len() != n_freqs * n_mels {
            return Err(EspeechError::Checkpoint(format!(
                "the vocoder's filterbank is {} values, expected {}",
                filters.len(),
                n_freqs * n_mels
            )));
        }
        Ok(Self {
            window,
            filters,
            n_mels,
        })
    }

    /// Frames a signal of `samples` samples produces, centred padding included.
    #[must_use]
    pub fn frames_for(samples: usize) -> usize { 1 + samples / HOP }

    /// The log-mel spectrogram of `wave`, as `[frames, n_mels]` in row-major
    /// order.
    ///
    /// The layout is frames-major because that is how the model wants it: the
    /// mel is a sequence of frames, and both the conditioning and the generated
    /// output are indexed by frame.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::RefAudioTooShort`] when the signal is shorter
    /// than the padding the centred transform needs.
    pub fn log_mel(&self, wave: &[f32]) -> Result<Vec<f32>, EspeechError> {
        let pad = N_FFT / 2;
        if wave.len() <= pad {
            return Err(EspeechError::RefAudioTooShort(
                wave.len() as f64 / f64::from(super::config::SAMPLE_RATE),
            ));
        }
        let padded = reflect_pad(wave, pad);
        let frames = Self::frames_for(wave.len());
        let n_freqs = N_FFT / 2 + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut input = fft.make_input_vec();
        let mut spectrum = fft.make_output_vec();
        let mut magnitude = vec![0f32; n_freqs];
        let mut out = vec![0f32; frames * self.n_mels];

        for frame in 0..frames {
            let start = frame * HOP;
            for (index, slot) in input.iter_mut().enumerate() {
                *slot = padded[start + index] * self.window[index];
            }
            fft.process(&mut input, &mut spectrum)
                .expect("the plan and the buffers are the same length");
            for (bin, value) in spectrum.iter().enumerate() {
                magnitude[bin] = value.norm();
            }
            // mel[m] = Σ_f magnitude[f] · filters[f][m]
            let row = &mut out[frame * self.n_mels..(frame + 1) * self.n_mels];
            for (bin, &value) in magnitude.iter().enumerate() {
                let filter = &self.filters[bin * self.n_mels..][..self.n_mels];
                for (band, &weight) in filter.iter().enumerate() {
                    row[band] += value * weight;
                }
            }
            for band in row.iter_mut() {
                *band = f64::from(*band).max(MEL_FLOOR).ln() as f32;
            }
        }
        Ok(out)
    }

    /// Overlap-adds the vocoder head's output back into a waveform.
    ///
    /// `magnitude` and `phase` are both `[frames, n_freqs]` in row-major order.
    /// The result is `HOP * (frames - 1)` samples long: the centred transform's
    /// padding is trimmed from both ends, exactly as the reference's inverse
    /// does.
    #[must_use]
    pub fn istft(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
    ) -> Vec<f32> {
        let n_freqs = N_FFT / 2 + 1;
        let pad = N_FFT / 2;
        let span = N_FFT + HOP * frames.saturating_sub(1);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(N_FFT);
        let mut spectrum = fft.make_input_vec();
        let mut samples = fft.make_output_vec();
        let mut sum = vec![0f32; span];
        let mut envelope = vec![0f32; span];
        let scale = 1.0 / N_FFT as f32;

        for frame in 0..frames {
            let row = frame * n_freqs;
            for bin in 0..n_freqs {
                let (sin, cos) = phase[row + bin].sin_cos();
                spectrum[bin] = Complex32::new(
                    magnitude[row + bin] * cos,
                    magnitude[row + bin] * sin,
                );
            }
            // A real signal has no imaginary part at DC or at Nyquist, and the
            // network is free to predict a phase there anyway. The reference's
            // inverse transform ignores those two imaginary parts; this one
            // refuses to run with them, so they are dropped explicitly.
            spectrum[0].im = 0.0;
            if let Some(last) = spectrum.last_mut() {
                last.im = 0.0;
            }
            fft.process(&mut spectrum, &mut samples)
                .expect("the plan and the buffers are the same length");
            let start = frame * HOP;
            for index in 0..N_FFT {
                let window = self.window[index];
                // The inverse transform is unnormalized here, unlike the
                // reference's, hence the explicit 1/N.
                sum[start + index] += samples[index] * scale * window;
                envelope[start + index] += window * window;
            }
        }

        let end = span.saturating_sub(pad);
        (pad..end)
            .map(|index| {
                let weight = envelope[index];
                // The window never sums to zero inside the trimmed span; the
                // guard only keeps a degenerate frame count from dividing by 0.
                if weight > 1e-11 {
                    sum[index] / weight
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Pads `wave` by `pad` samples on each side by reflecting it, the way a
/// centred short-time transform does.
fn reflect_pad(wave: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(wave.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|offset| wave[offset]));
    padded.extend_from_slice(wave);
    padded.extend((1..=pad).map(|offset| wave[wave.len() - 1 - offset]));
    padded
}

#[cfg(test)]
mod tests;
