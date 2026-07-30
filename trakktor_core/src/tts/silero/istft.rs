//! The way back from a spectrum to a waveform, and the model's own way down to
//! a lower rate.
//!
//! Both live on the host and are shared by the runtimes. The inverse transform
//! has no choice — neither tensor backend has an FFT — and keeping it in one
//! place has the pleasant side effect that the last stage of synthesis is
//! literally the same code whichever runtime produced the spectrum.
//!
//! The second half of this module is the part that was a guess until the
//! artifact was read: the model's 24 kHz and 8 kHz outputs are **not**
//! resampled. The vocoder always synthesizes 48 kHz, and a lower rate is the
//! zeroth band of a PQMF analysis filterbank applied to it — a different
//! signal, not the same one interpolated.
//!
//! Ported from Vocos (MIT) and the Silero vocoder.

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::config::MAX_MAGNITUDE;

/// Below this the window envelope is treated as zero rather than divided by.
/// The reference asserts the envelope never gets there; the guard only keeps a
/// degenerate frame count from dividing by zero.
const ENVELOPE_FLOOR: f32 = 1e-11;

/// The analysis window and the geometry that goes with it.
#[derive(Debug, Clone)]
pub struct Window {
    /// The window, `n_fft` long, as the checkpoint carries it.
    pub samples: Vec<f32>,
    /// Hop between frames.
    pub hop: usize,
}

impl Window {
    /// Length of the transform, which is also the window's length.
    #[must_use]
    pub fn n_fft(&self) -> usize { self.samples.len() }

    /// Samples the transform trims from each end. With `same` padding that is
    /// half of what the window overhangs the hop by — which makes the result
    /// exactly `frames × hop` samples long.
    #[must_use]
    pub fn pad(&self) -> usize { (self.n_fft() - self.hop) / 2 }

    /// Overlap-adds the vocoder head's output back into a waveform.
    ///
    /// `magnitude` and `phase` are `[frames, bins]` in row-major order, where
    /// `bins = n_fft / 2 + 1`. The result is `frames × hop` samples long.
    #[must_use]
    pub fn inverse(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
    ) -> Vec<f32> {
        let n_fft = self.n_fft();
        let bins = n_fft / 2 + 1;
        let pad = self.pad();
        let span = n_fft + self.hop * frames.saturating_sub(1);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(n_fft);
        let mut spectrum = fft.make_input_vec();
        let mut samples = fft.make_output_vec();
        let mut sum = vec![0f32; span];
        let mut envelope = vec![0f32; span];
        // The inverse transform is unnormalized here, unlike torch's, so the
        // 1/N rides along with the window.
        let scaled: Vec<f32> = self
            .samples
            .iter()
            .map(|value| value / n_fft as f32)
            .collect();

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
            let start = frame * self.hop;
            let target = &mut sum[start..start + n_fft];
            let weights = &mut envelope[start..start + n_fft];
            for index in 0..n_fft {
                let window = self.samples[index];
                target[index] += samples[index] * scaled[index];
                weights[index] += window * window;
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
}

/// Turns the vocoder head's raw output into the magnitudes and phases the
/// inverse transform takes: the first half is a log magnitude, the second a
/// phase in radians.
///
/// Shared by the runtimes so the ceiling on the magnitudes — the reference's
/// own guard against a spectrum that would explode into a click — is applied
/// once.
#[must_use]
pub fn split_spectrum(head: &[f32], bins: usize) -> (Vec<f32>, Vec<f32>) {
    let mut magnitude = Vec::with_capacity(head.len() / 2);
    let mut phase = Vec::with_capacity(head.len() / 2);
    for row in head.chunks_exact(2 * bins) {
        magnitude.extend(
            row[..bins]
                .iter()
                .map(|value| value.exp().min(MAX_MAGNITUDE as f32)),
        );
        phase.extend_from_slice(&row[bins..]);
    }
    (magnitude, phase)
}

/// A pseudo-QMF analysis filterbank, as the vocoder carries it.
#[derive(Debug, Clone)]
pub struct Pqmf {
    /// The analysis filters, `[bands, taps]` in row-major order.
    pub filters: Vec<f32>,
    /// Bands, which is also the decimation factor.
    pub bands: usize,
    /// Taps per filter.
    pub taps: usize,
}

impl Pqmf {
    /// The lowest band of `wave`, decimated by [`Self::bands`] — the model's
    /// own way of producing a lower sample rate.
    ///
    /// Every band is computed, not just the one returned: the reference's
    /// soft clip looks at all of them, and skipping the rest would change when
    /// it fires.
    #[must_use]
    pub fn low_band(&self, wave: &[f32]) -> Vec<f32> {
        // torch pads the convolution by half the declared tap count, which is
        // one less than the filters actually hold.
        let pad = (self.taps.saturating_sub(1)) / 2;
        let length = if wave.is_empty() {
            0
        } else {
            (wave.len() + 2 * pad - self.taps) / self.bands + 1
        };
        let mut bands = vec![0f32; self.bands * length];
        let mut peak = 0f32;
        for band in 0..self.bands {
            let filter = &self.filters[band * self.taps..][..self.taps];
            let row = &mut bands[band * length..][..length];
            for (index, slot) in row.iter_mut().enumerate() {
                let start = index * self.bands;
                let mut total = 0f32;
                for (tap, weight) in filter.iter().enumerate() {
                    let at = start + tap;
                    if at >= pad && at - pad < wave.len() {
                        total += weight * wave[at - pad];
                    }
                }
                *slot = total;
                peak = peak.max(total.abs());
            }
        }
        let low = &bands[..length];
        if peak <= 1.0 {
            low.to_vec()
        } else {
            low.iter().map(|value| value.tanh()).collect()
        }
    }
}

#[cfg(test)]
mod tests;
