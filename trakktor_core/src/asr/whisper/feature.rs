//! Log-mel spectrogram extraction.
//!
//! Turns 16 kHz mono PCM into the log-mel features Whisper's encoder consumes,
//! reproducing the reference pipeline: an STFT with a periodic Hann window over
//! centered, reflect-padded frames; the power spectrum with its trailing frame
//! dropped; projection through a mel filterbank; and a log10 rescale with an
//! 80 dB dynamic-range floor.

use realfft::RealFftPlanner;

use super::{
    assets,
    constants::{HOP_LENGTH, N_FFT, N_FRAMES, N_FREQS},
};

/// The mel-band count: the two filterbanks Whisper ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelBands {
    /// 80 bands — every model except large-v3 and its turbo variant.
    Mel80,
    /// 128 bands — large-v3 and large-v3-turbo.
    Mel128,
}

impl MelBands {
    /// The number of mel bands.
    pub const fn count(self) -> usize {
        match self {
            MelBands::Mel80 => 80,
            MelBands::Mel128 => 128,
        }
    }
}

/// A log-mel spectrogram stored row-major: `n_mels` rows of `n_frames` columns.
#[derive(Debug, Clone)]
pub struct Mel {
    n_mels: usize,
    n_frames: usize,
    data: Vec<f32>,
}

impl Mel {
    /// Number of mel bands (rows).
    pub fn n_mels(&self) -> usize { self.n_mels }

    /// Number of time frames (columns).
    pub fn n_frames(&self) -> usize { self.n_frames }

    /// The spectrogram as a flat row-major slice of `n_mels * n_frames` values.
    pub fn data(&self) -> &[f32] { &self.data }

    /// The value at mel band `mel` and time frame `frame`.
    pub fn get(&self, mel: usize, frame: usize) -> f32 {
        self.data[mel * self.n_frames + frame]
    }

    /// Cuts one encoder window: `len` frames starting at `start`, zero-padded
    /// or trimmed on the right to exactly [`N_FRAMES`] columns.
    ///
    /// Ranges reaching past the end are clamped (Python slice semantics), so
    /// callers can slice right up to the trailing silence padding.
    pub fn window(&self, start: usize, len: usize) -> MelWindow {
        let start = start.min(self.n_frames);
        let take = len.min(self.n_frames - start).min(N_FRAMES);

        let mut data = vec![0.0f32; self.n_mels * N_FRAMES];
        for m in 0..self.n_mels {
            let src_offset = m * self.n_frames + start;
            data[m * N_FRAMES..m * N_FRAMES + take]
                .copy_from_slice(&self.data[src_offset..src_offset + take]);
        }
        MelWindow {
            n_mels: self.n_mels,
            data,
        }
    }
}

/// One 30 s encoder window of a log-mel spectrogram: `n_mels` rows by
/// [`N_FRAMES`] columns, row-major. This is exactly the encoder's input.
#[derive(Debug, Clone)]
pub struct MelWindow {
    n_mels: usize,
    data: Vec<f32>,
}

impl MelWindow {
    /// Number of mel bands (rows).
    pub fn n_mels(&self) -> usize { self.n_mels }

    /// The window as a flat row-major slice of `n_mels * N_FRAMES` values.
    pub fn data(&self) -> &[f32] { &self.data }
}

/// Computes the log-mel spectrogram of `audio`.
///
/// When `padding` is non-zero, that many zero samples are appended before
/// framing — the reference appends one 30 s window of silence so the final
/// chunk of a file can still be decoded. Normalization uses the global maximum
/// over the whole spectrogram, so the caller computes the mel once for the
/// entire signal and slices windows out of the result.
pub fn log_mel_spectrogram(
    audio: &[f32],
    bands: MelBands,
    padding: usize,
) -> Mel {
    let padded_signal: Vec<f32>;
    let signal: &[f32] = if padding > 0 {
        let mut s = Vec::with_capacity(audio.len() + padding);
        s.extend_from_slice(audio);
        s.resize(audio.len() + padding, 0.0);
        padded_signal = s;
        &padded_signal
    } else {
        audio
    };

    let pad = N_FFT / 2;
    let framed = reflect_pad(signal, pad);

    // Centered framing yields `1 + len/hop` frames; the reference drops the
    // trailing frame, leaving exactly `len/hop`.
    let n_frames = signal.len() / HOP_LENGTH;

    let window = hann_window();

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let mut frame_buf = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    // Power spectrum, stored frame-major: `[frame][freq]`.
    let mut power = vec![0.0f32; n_frames * N_FREQS];
    for f in 0..n_frames {
        let start = f * HOP_LENGTH;
        for (i, slot) in frame_buf.iter_mut().enumerate() {
            *slot = framed[start + i] * window[i];
        }
        r2c.process_with_scratch(&mut frame_buf, &mut spectrum, &mut scratch)
            .expect("real FFT of a correctly sized frame cannot fail");
        let row = &mut power[f * N_FREQS..(f + 1) * N_FREQS];
        for (p, c) in row.iter_mut().zip(spectrum.iter()) {
            *p = c.re * c.re + c.im * c.im;
        }
    }

    // Mel projection: filters (n_mels x N_FREQS) times power (N_FREQS x
    // frames).
    let filters = assets::mel_filters(bands);
    let n_mels = bands.count();
    let mut data = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        let fbank = &filters[m * N_FREQS..(m + 1) * N_FREQS];
        for f in 0..n_frames {
            let frame_power = &power[f * N_FREQS..(f + 1) * N_FREQS];
            let mut acc = 0.0f32;
            for (w, p) in fbank.iter().zip(frame_power.iter()) {
                acc += w * p;
            }
            data[m * n_frames + f] = acc;
        }
    }

    log_rescale(&mut data);

    Mel {
        n_mels,
        n_frames,
        data,
    }
}

/// Applies `log10` with a `1e-10` floor, clamps to an 80 dB dynamic range below
/// the global maximum, then rescales into roughly `[-1, 1]`.
fn log_rescale(data: &mut [f32]) {
    let mut max_log = f32::NEG_INFINITY;
    for v in data.iter_mut() {
        let l = v.max(1e-10).log10();
        *v = l;
        if l > max_log {
            max_log = l;
        }
    }
    let floor = max_log - 8.0;
    for v in data.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }
}

/// A periodic Hann window of length [`N_FFT`], matching a periodic-mode window.
fn hann_window() -> [f32; N_FFT] {
    let mut w = [0.0f32; N_FFT];
    for (n, wn) in w.iter_mut().enumerate() {
        let phase = 2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32;
        *wn = 0.5 - 0.5 * phase.cos();
    }
    w
}

/// Reflect-pads `x` by `pad` samples on each end without repeating the edge
/// sample, matching numpy/torch reflect mode. Requires `x.len() > pad`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for k in 0..pad {
        out.push(x[pad - k]);
    }
    out.extend_from_slice(x);
    for k in 0..pad {
        out.push(x[n - 2 - k]);
    }
    out
}

#[cfg(test)]
mod tests;
