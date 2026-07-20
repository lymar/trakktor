//! Kaldi-compatible log-mel fbank extraction.
//!
//! A port of the fbank front end the reference runtime uses for transducer
//! models (the `kaldi-native-fbank` library with sherpa-onnx's configuration):
//! 16 kHz input, 80 mel bins, 25 ms povey-windowed frames every 10 ms, a
//! 512-point FFT, and an HTK-scale triangular mel bank from 20 Hz to 7600 Hz.
//! Unlike the torchaudio-style features of the GigaAM engine, frames are
//! processed Kaldi-style: `snip_edges=false` centering with edge reflection,
//! per-frame DC-offset removal and pre-emphasis before windowing, and a
//! `ln(max(x, ε))` floor instead of a clamp. Dithering is disabled for
//! determinism (the reference default; its decode scripts add noise at the
//! 3·10⁻⁵ level, which is below any quality-relevant threshold).
//!
//! Samples are consumed as `f32` in `[-1, 1]` exactly as decoded — the
//! reference applies no scaling or normalization for these models.

#[cfg(test)]
mod tests;

use realfft::RealFftPlanner;

use super::constants::SAMPLE_RATE;

/// Number of mel bins every model of the line consumes.
pub const N_MELS: usize = 80;

/// Samples per frame hop (10 ms).
pub const FRAME_SHIFT: usize = SAMPLE_RATE / 100;

/// Samples per analysis window (25 ms).
pub const FRAME_LENGTH: usize = SAMPLE_RATE * 25 / 1000;

/// FFT size: the window length rounded up to a power of two.
const N_FFT: usize = 512;

/// One-sided FFT bins the mel bank projects (the Nyquist bin is excluded,
/// matching the reference).
const N_FFT_BINS: usize = N_FFT / 2;

/// Pre-emphasis coefficient.
const PREEMPH: f32 = 0.97;

/// Mel-bank edge frequencies (Hz); the high edge is `nyquist - 400`.
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 8000.0 - 400.0;

/// How far a centered frame's window starts before its nominal position:
/// `start(f) = f·FRAME_SHIFT − OFFSET` (the `snip_edges=false` centering).
pub(crate) const OFFSET: usize = FRAME_LENGTH / 2 - FRAME_SHIFT / 2;

/// Log-mel features stored frame-major: `n_frames` rows of [`N_MELS`]
/// columns, i.e. `data[frame * N_MELS + mel]` — exactly the encoder's
/// `[n_frames, n_mels]` input layout.
#[derive(Debug, Clone)]
pub struct Features {
    n_frames: usize,
    data: Vec<f32>,
}

impl Features {
    /// Number of feature frames (rows).
    pub fn n_frames(&self) -> usize { self.n_frames }

    /// The features as a flat frame-major slice of `n_frames * N_MELS`
    /// values.
    pub fn data(&self) -> &[f32] { &self.data }

    /// The value at time frame `frame` and mel bin `mel`.
    pub fn get(&self, frame: usize, mel: usize) -> f32 {
        self.data[frame * N_MELS + mel]
    }
}

/// Extracts Kaldi-style log-mel features. Holds the precomputed povey window
/// and mel filterbank.
pub struct FbankExtractor {
    /// Analysis window of length [`FRAME_LENGTH`].
    window: Vec<f32>,
    /// Mel filterbank, mel-major `[N_MELS * N_FFT_BINS]`.
    fbank: Vec<f32>,
}

impl Default for FbankExtractor {
    fn default() -> Self { Self::new() }
}

impl FbankExtractor {
    /// Builds the extractor (window and mel bank are fixed by the line's
    /// feature geometry).
    pub fn new() -> Self {
        Self {
            window: povey_window(FRAME_LENGTH),
            fbank: mel_filterbank(),
        }
    }

    /// Number of frames extracted from `n_samples` samples. With `flush` the
    /// tail is reflected and every centered frame is produced
    /// (`(n + shift/2) / shift`); without it only frames fully covered by the
    /// signal are counted, so a frame never changes once more samples arrive.
    pub fn n_frames(&self, n_samples: usize, flush: bool) -> usize {
        let full = (n_samples + FRAME_SHIFT / 2) / FRAME_SHIFT;
        if flush {
            return full;
        }
        // Frame f covers samples [start(f), start(f) + 400) with
        // start(f) = 160·f − 120; keep only frames whose window ends within
        // the signal (written with +120 on both sides to stay unsigned).
        let mut n = full;
        while n > 0 && (n - 1) * FRAME_SHIFT + FRAME_LENGTH > n_samples + OFFSET
        {
            n -= 1;
        }
        n
    }

    /// Computes log-mel features for frames
    /// `[first_frame, first_frame + n_frames)` of the signal's frame grid.
    /// `samples` is the retained window of the signal starting at absolute
    /// sample `buffer_start` (0 for a whole signal). Frame windows reaching
    /// past either end of `samples` read the reflected buffer — correct
    /// exactly at the true signal edges, so a caller holding a partial buffer
    /// must keep every sample its frames' windows cover (and only flush the
    /// final frames once the signal has ended, per
    /// [`n_frames`](Self::n_frames)).
    pub fn compute_range(
        &self,
        samples: &[f32],
        buffer_start: usize,
        first_frame: usize,
        n_frames: usize,
    ) -> Features {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(N_FFT);
        let mut frame_buf = r2c.make_input_vec();
        let mut spectrum = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        let n = samples.len() as isize;
        // Reflection needs at least one sample to bounce off.
        debug_assert!(n_frames == 0 || n > 0, "frames requested of no signal");
        let mut data = vec![0.0f32; n_frames * N_MELS];
        let mut power = [0.0f32; N_FFT_BINS];
        for (row, f) in (first_frame..first_frame + n_frames).enumerate() {
            // Window start relative to the buffer (may be negative).
            let s0 = (f * FRAME_SHIFT) as isize -
                OFFSET as isize -
                buffer_start as isize;
            for (i, slot) in frame_buf[..FRAME_LENGTH].iter_mut().enumerate() {
                let mut idx = s0 + i as isize;
                // Reflect (without repeating the edge sample) until inside.
                while idx < 0 || idx >= n {
                    idx = if idx < 0 { -idx - 1 } else { 2 * n - 1 - idx };
                }
                *slot = samples[idx as usize];
            }
            let frame = &mut frame_buf[..FRAME_LENGTH];

            // DC-offset removal over the raw frame.
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
            // Pre-emphasis, in place, backwards.
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            // Analysis window, then zero-pad to the FFT size.
            for (s, w) in frame.iter_mut().zip(&self.window) {
                *s *= w;
            }
            frame_buf[FRAME_LENGTH..].fill(0.0);

            r2c.process_with_scratch(
                &mut frame_buf,
                &mut spectrum,
                &mut scratch,
            )
            .expect("real FFT of a correctly sized frame cannot fail");
            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }

            for m in 0..N_MELS {
                let bank = &self.fbank[m * N_FFT_BINS..(m + 1) * N_FFT_BINS];
                let mut acc = 0.0f32;
                for (w, p) in bank.iter().zip(power.iter()) {
                    acc += w * p;
                }
                data[row * N_MELS + m] = acc.max(f32::EPSILON).ln();
            }
        }
        Features { n_frames, data }
    }

    /// Computes the whole signal's features in one call (offline chunks).
    pub fn compute(&self, samples: &[f32]) -> Features {
        self.compute_range(samples, 0, 0, self.n_frames(samples.len(), true))
    }
}

/// The povey window: `(0.5 − 0.5·cos(2πk/(n−1)))^0.85`.
fn povey_window(n: usize) -> Vec<f32> {
    let a = 2.0 * std::f64::consts::PI / (n - 1) as f64;
    (0..n)
        .map(|k| (0.5 - 0.5 * (a * k as f64).cos()).powf(0.85) as f32)
        .collect()
}

/// HTK mel scale.
fn mel_scale(freq: f32) -> f32 { 1127.0 * (1.0 + freq / 700.0).ln() }

/// The Kaldi/HTK triangular mel bank over the one-sided FFT bins, mel-major
/// `[N_MELS * N_FFT_BINS]`: peak 1.0, no area normalization.
fn mel_filterbank() -> Vec<f32> {
    let fft_bin_width = SAMPLE_RATE as f32 / N_FFT as f32;
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(HIGH_FREQ);
    let mel_delta = (mel_high - mel_low) / (N_MELS + 1) as f32;

    let mut fbank = vec![0.0f32; N_MELS * N_FFT_BINS];
    for m in 0..N_MELS {
        let left = mel_low + m as f32 * mel_delta;
        let center = mel_low + (m + 1) as f32 * mel_delta;
        let right = mel_low + (m + 2) as f32 * mel_delta;
        for i in 0..N_FFT_BINS {
            let mel = mel_scale(fft_bin_width * i as f32);
            if mel > left && mel < right {
                fbank[m * N_FFT_BINS + i] = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
            }
        }
    }
    fbank
}
