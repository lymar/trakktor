//! The HTK mel filterbank, reproducing `torchaudio.functional.melscale_fbanks`.
//!
//! GigaAM's feature extractor uses `torchaudio.transforms.MelSpectrogram` with
//! its default mel settings — `norm=None`, `mel_scale="htk"`, `f_min=0`,
//! `f_max=sample_rate/2` — so the filterbank is fully determined by
//! `(n_freqs, n_mels, sample_rate)` and computed here rather than embedded. The
//! result is validated bit-close against the filterbank extracted from the
//! reference module.

/// HTK hz → mel.
fn hz_to_mel(freq: f64) -> f64 { 2595.0 * (1.0 + freq / 700.0).log10() }

/// HTK mel → hz.
fn mel_to_hz(mel: f64) -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) }

/// `n` evenly spaced points from `start` to `stop` inclusive.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Builds the mel filterbank, stored **mel-major**: row `m` holds the
/// `n_freqs` triangular weights of band `m`, so a frame's mel value is the dot
/// product of one row with the frame's power spectrum. Length is
/// `n_mels * n_freqs`.
///
/// Matches torchaudio's `melscale_fbanks(n_freqs, f_min=0, f_max=sample_rate/2,
/// n_mels, sample_rate, norm=None, mel_scale="htk")`, which stores the
/// transpose (`[n_freqs, n_mels]`).
pub fn mel_filterbank(
    n_freqs: usize,
    n_mels: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let f_min = 0.0;
    // torchaudio uses integer `sample_rate // 2` for the top FFT frequency.
    let f_max = (sample_rate / 2) as f64;

    // FFT-bin center frequencies, 0..=f_max.
    let all_freqs = linspace(f_min, f_max, n_freqs);

    // `n_mels + 2` band edges, evenly spaced on the mel scale.
    let m_min = hz_to_mel(f_min);
    let m_max = hz_to_mel(f_max);
    let f_pts: Vec<f64> = linspace(m_min, m_max, n_mels + 2)
        .into_iter()
        .map(mel_to_hz)
        .collect();

    let mut fb = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let lower = f_pts[m];
        let center = f_pts[m + 1];
        let upper = f_pts[m + 2];
        let down_den = center - lower;
        let up_den = upper - center;
        for (fk, &freq) in all_freqs.iter().enumerate() {
            let down = (freq - lower) / down_den; // rising edge
            let up = (upper - freq) / up_den; // falling edge
            let w = down.min(up).max(0.0);
            fb[m * n_freqs + fk] = w as f32;
        }
    }
    fb
}
