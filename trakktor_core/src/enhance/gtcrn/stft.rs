//! The analysis and synthesis transforms, on the host and shared by the
//! runtimes — neither tensor backend has an FFT.
//!
//! These reproduce `torch.stft`/`torch.istft` in the configuration the
//! reference uses: 512-point transform, 256 hop, the **square root** of a
//! periodic Hann window, and `center = true`, which pads the signal by half a
//! transform at each end and reflects it rather than zero-filling. The square
//! root matters: analysis and synthesis each apply it, so together they apply
//! one Hann — which is what makes the overlap-add sum to unity at this hop.

use realfft::{RealFftPlanner, num_complex::Complex32};

use super::config::{BINS, HOP, N_FFT, frames};

/// Below this the synthesis envelope is treated as zero rather than divided by.
const ENVELOPE_FLOOR: f32 = 1e-11;

/// The analysis window: the square root of a periodic Hann.
#[must_use]
pub fn window() -> Vec<f32> {
    (0..N_FFT)
        .map(|index| {
            let phase =
                2.0 * std::f64::consts::PI * index as f64 / N_FFT as f64;
            ((0.5 - 0.5 * phase.cos()).sqrt()) as f32
        })
        .collect()
}

/// One analysed spectrum: real and imaginary parts, `[bins, frames]` in
/// row-major order — the layout the reference's `(F, T, 2)` tensor has.
pub struct Spectrum {
    /// Real parts, bin-major.
    pub real: Vec<f32>,
    /// Imaginary parts, bin-major.
    pub imag: Vec<f32>,
    /// Frames analysed.
    pub frames: usize,
}

/// Analyses `samples` into a spectrum.
///
/// The signal is reflected by half a transform at each end, as `center = true`
/// does, so the first frame is centred on the first sample.
#[must_use]
pub fn analyze(samples: &[f32]) -> Spectrum {
    analyze_range(samples, 0, frames(samples.len()))
}

/// Analyses `count` frames of `samples` starting at frame `first`.
///
/// The frames come out **identical to what analysing the whole recording would
/// give**, because the context each frame needs is taken from the recording
/// itself; only at the true ends is anything reflected. That is what lets a
/// long recording be processed in pieces without a seam at every join.
#[must_use]
pub fn analyze_range(samples: &[f32], first: usize, count: usize) -> Spectrum {
    let pad = N_FFT / 2;
    let padded = reflect_window(samples, first * HOP, count);
    let win = window();

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let mut real = vec![0f32; BINS * count];
    let mut imag = vec![0f32; BINS * count];
    let _ = pad;
    for frame in 0..count {
        let start = frame * HOP;
        for index in 0..N_FFT {
            input[index] = padded[start + index] * win[index];
        }
        fft.process(&mut input, &mut output)
            .expect("the plan and the buffers are the same length");
        for bin in 0..BINS {
            real[bin * count + frame] = output[bin].re;
            imag[bin * count + frame] = output[bin].im;
        }
    }
    Spectrum {
        real,
        imag,
        frames: count,
    }
}

/// Synthesises a waveform from a spectrum in the same layout.
///
/// The result is `(frames - 1) × hop` samples long — what `torch.istft` returns
/// for a centred analysis — which is not in general the length that went in.
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

    for frame in 0..count {
        for bin in 0..BINS {
            input[bin] = Complex32::new(
                spectrum.real[bin * count + frame],
                spectrum.imag[bin * count + frame],
            );
        }
        // A real signal has no imaginary part at DC or at Nyquist; the network
        // is free to predict one, and this transform refuses to run with it.
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

/// The samples `count` frames starting at `at` need, with the recording's own
/// neighbours for context and a mirror only where the recording ends.
///
/// `torch.stft` mirrors both ends of what it is given; mirroring the *middle*
/// of a recording just because a chunk happens to start there would put a seam
/// in every join, so real audio is used wherever there is any.
fn reflect_window(samples: &[f32], at: usize, count: usize) -> Vec<f32> {
    let pad = N_FFT / 2;
    let span = N_FFT + HOP * count.saturating_sub(1);
    let last = samples.len().saturating_sub(1);
    // Index into the recording as `torch.stft` would into its padded copy:
    // position zero of the padded copy is sample `pad` mirrored back.
    let at_index = |index: i64| -> f32 {
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
        .map(|offset| at_index(at as i64 + offset as i64 - pad as i64))
        .collect()
}

/// Overlap-add across chunk boundaries: the last `n_fft - hop` samples of a
/// chunk are still being summed when the next one starts, so they are carried
/// rather than emitted.
pub struct Synthesizer {
    sum: Vec<f32>,
    envelope: Vec<f32>,
    emitted: usize,
}

impl Default for Synthesizer {
    fn default() -> Self { Self::new() }
}

impl Synthesizer {
    /// A synthesizer at the start of a recording.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sum: vec![0f32; N_FFT],
            envelope: vec![0f32; N_FFT],
            emitted: 0,
        }
    }

    /// Adds one chunk's frames and returns the samples that are now complete.
    #[must_use]
    pub fn push(&mut self, spectrum: &Spectrum) -> Vec<f32> {
        let count = spectrum.frames;
        let win = window();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(N_FFT);
        let mut input = fft.make_input_vec();
        let mut samples = fft.make_output_vec();

        let need = N_FFT + HOP * count.saturating_sub(1);
        self.sum.resize(need.max(self.sum.len()), 0.0);
        self.envelope.resize(need.max(self.envelope.len()), 0.0);
        let scale = 1.0 / N_FFT as f32;

        for frame in 0..count {
            for bin in 0..BINS {
                input[bin] = Complex32::new(
                    spectrum.real[bin * count + frame],
                    spectrum.imag[bin * count + frame],
                );
            }
            input[0].im = 0.0;
            if let Some(last) = input.last_mut() {
                last.im = 0.0;
            }
            fft.process(&mut input, &mut samples)
                .expect("the plan and the buffers are the same length");
            let start = frame * HOP;
            for index in 0..N_FFT {
                let weight = win[index];
                self.sum[start + index] += samples[index] * scale * weight;
                self.envelope[start + index] += weight * weight;
            }
        }

        // Everything before the last frame's own span is finished.
        let complete = count * HOP;
        let out = self.drain(complete);
        self.sum.drain(..complete);
        self.envelope.drain(..complete);
        out
    }

    /// The samples still held once the recording ends.
    #[must_use]
    pub fn finish(&mut self) -> Vec<f32> {
        let len = self.sum.len();
        let out = self.drain(len);
        self.sum.clear();
        self.envelope.clear();
        out
    }

    /// Normalizes and emits `count` samples, dropping the leading half-window
    /// the centred analysis put there.
    fn drain(&mut self, count: usize) -> Vec<f32> {
        let pad = N_FFT / 2;
        let skip = pad.saturating_sub(self.emitted);
        self.emitted += count;
        (skip..count)
            .map(|index| {
                let weight = self.envelope[index];
                if weight > ENVELOPE_FLOOR {
                    self.sum[index] / weight
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
