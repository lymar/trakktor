//! Log-mel spectrogram extraction.
//!
//! Turns 16 kHz mono PCM into the log-mel features GigaAM's Conformer encoder
//! consumes, reproducing the reference `FeatureExtractor`
//! (`torchaudio.transforms.MelSpectrogram` + a natural-log `SpecScaler`): an
//! STFT with a periodic Hann window; the power spectrum; projection through an
//! HTK mel filterbank; and `log` with a `[1e-9, 1e9]` clamp. Unlike Whisper's
//! features there is no global-max normalization, so a chunk's features can be
//! computed independently.
//!
//! Feature geometry varies per checkpoint (`n_fft`, window length, whether the
//! STFT is centered, mel-band count), so it is passed in as a [`MelConfig`]
//! rather than fixed.

mod fbank;
#[cfg(test)]
mod tests;

use realfft::RealFftPlanner;

use super::constants::{SAMPLE_RATE, SPEC_CLAMP_MAX, SPEC_CLAMP_MIN};

/// Feature-extraction geometry, read from a checkpoint's preprocessor config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelConfig {
    /// FFT size. The window length equals this in every published checkpoint.
    pub n_fft: usize,
    /// STFT hop in samples.
    pub hop_length: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Whether the STFT is centered (reflect-pad by `n_fft / 2`, keeping the
    /// leading/trailing frames) or not (frames start at sample 0).
    pub center: bool,
}

impl MelConfig {
    /// Number of one-sided FFT bins (`n_fft / 2 + 1`).
    pub fn n_freqs(&self) -> usize { self.n_fft / 2 + 1 }

    /// Frames produced for `n_samples` input samples — the reference `out_len`.
    pub fn out_len(&self, n_samples: usize) -> usize {
        if self.center {
            n_samples / self.hop_length + 1
        } else {
            // Requires n_samples >= n_fft; guarded by the caller.
            (n_samples - self.n_fft) / self.hop_length + 1
        }
    }
}

/// A log-mel spectrogram stored mel-major: `n_mels` rows of `n_frames`
/// columns, i.e. `data[mel * n_frames + frame]`. This is exactly the encoder's
/// `[n_mels, n_frames]` input.
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

    /// The spectrogram as a flat mel-major slice of `n_mels * n_frames` values.
    pub fn data(&self) -> &[f32] { &self.data }

    /// The value at mel band `mel` and time frame `frame`.
    pub fn get(&self, mel: usize, frame: usize) -> f32 {
        self.data[mel * self.n_frames + frame]
    }
}

/// Extracts log-mel features. Holds the mel filterbank and analysis window;
/// these come from the checkpoint (see [`new`](FeatureExtractor::new)) so a
/// chunk's features reproduce the reference exactly.
pub struct FeatureExtractor {
    config: MelConfig,
    /// Mel filterbank, mel-major `[n_mels * n_freqs]`.
    fbank: Vec<f32>,
    /// Analysis window of length `n_fft`.
    window: Vec<f32>,
}

impl FeatureExtractor {
    /// Builds an extractor from the checkpoint's saved buffers: the mel
    /// filterbank `fbank_freq_major` in torchaudio's `[n_freqs, n_mels]`
    /// row-major layout, and the STFT `window` of length `n_fft`.
    ///
    /// The buffers must be used verbatim, not recomputed: they are stored in
    /// half precision, and that quantization — negligible in the filterbank
    /// itself — is amplified by the log at quiet mel bins into differences of
    /// whole log units.
    pub fn new(
        config: MelConfig,
        fbank_freq_major: &[f32],
        window: &[f32],
    ) -> Self {
        let n_freqs = config.n_freqs();
        let n_mels = config.n_mels;
        assert_eq!(
            fbank_freq_major.len(),
            n_freqs * n_mels,
            "filterbank shape"
        );
        assert_eq!(window.len(), config.n_fft, "window length");
        // Transpose to mel-major so a frame's mel value is a contiguous dot
        // product of one row with the power spectrum.
        let mut fbank = vec![0.0f32; n_mels * n_freqs];
        for fk in 0..n_freqs {
            for m in 0..n_mels {
                fbank[m * n_freqs + fk] = fbank_freq_major[fk * n_mels + m];
            }
        }
        Self {
            config,
            fbank,
            window: window.to_vec(),
        }
    }

    /// Builds an extractor with a freshly computed HTK filterbank and periodic
    /// Hann window. Only for contexts without a checkpoint (tests); the runtime
    /// uses [`new`](FeatureExtractor::new) with the checkpoint buffers, which
    /// reproduce the reference exactly.
    pub fn computed(config: MelConfig) -> Self {
        Self {
            fbank: fbank::mel_filterbank(
                config.n_freqs(),
                config.n_mels,
                SAMPLE_RATE,
            ),
            window: hann_window(config.n_fft),
            config,
        }
    }

    /// The geometry this extractor was built for.
    pub fn config(&self) -> MelConfig { self.config }

    /// Number of mel bands this extractor produces.
    pub fn n_mels(&self) -> usize { self.config.n_mels }

    /// Computes the log-mel spectrogram of `audio` (16 kHz mono, roughly
    /// `[-1, 1]`). Frame count is [`MelConfig::out_len`].
    pub fn log_mel(&self, audio: &[f32]) -> Mel {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_freqs = self.config.n_freqs();
        let n_mels = self.config.n_mels;

        // Framing source: reflect-padded when centered, otherwise the raw
        // signal. `frame_start(f)` indexes this buffer.
        let source: std::borrow::Cow<[f32]> = if self.config.center {
            std::borrow::Cow::Owned(reflect_pad(audio, n_fft / 2))
        } else {
            std::borrow::Cow::Borrowed(audio)
        };
        let n_frames = self.config.out_len(audio.len());

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(n_fft);
        let mut frame_buf = r2c.make_input_vec();
        let mut spectrum = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        let mut data = vec![0.0f32; n_mels * n_frames];
        let mut power = vec![0.0f32; n_freqs];
        for f in 0..n_frames {
            let start = f * hop;
            for (i, slot) in frame_buf.iter_mut().enumerate() {
                *slot = source[start + i] * self.window[i];
            }
            r2c.process_with_scratch(
                &mut frame_buf,
                &mut spectrum,
                &mut scratch,
            )
            .expect("real FFT of a correctly sized frame cannot fail");
            for (p, c) in power.iter_mut().zip(spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }
            for m in 0..n_mels {
                let row = &self.fbank[m * n_freqs..(m + 1) * n_freqs];
                let mut acc = 0.0f32;
                for (w, p) in row.iter().zip(power.iter()) {
                    acc += w * p;
                }
                // SpecScaler: log(clamp(x, 1e-9, 1e9)), natural log.
                data[m * n_frames + f] =
                    acc.clamp(SPEC_CLAMP_MIN, SPEC_CLAMP_MAX).ln();
            }
        }

        Mel {
            n_mels,
            n_frames,
            data,
        }
    }
}

/// A periodic Hann window of length `n`, matching `torch.hann_window(n)`
/// (periodic by default).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
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
