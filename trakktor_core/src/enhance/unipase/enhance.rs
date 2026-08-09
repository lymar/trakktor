//! The driver: a file in, an enhanced file out.
//!
//! Everything here is host-side and runtime-independent — the windowing, the
//! packet-loss detection, the stitching, the loudness, the output rate. A
//! runtime is asked for exactly one thing at a time: enhance this window.
//!
//! # Why windows at all
//!
//! The encoder is quadratic in the window and the whole pipeline holds its
//! activations, so a recording is cut into eight-second windows that start
//! every four seconds. Consecutive windows therefore overlap by four seconds,
//! and each contributes only its middle: the first four seconds of a window are
//! the second half of its predecessor's overlap and are thrown away. A window's
//! edges are where the encoder has the least context, so the arrangement means
//! no sample is ever taken from an edge except at the very start and the very
//! end of the recording.

use std::path::Path;

use super::{
    config::{
        HOP, HOP_SECONDS, PAD_REMAINDER, SAMPLE_RATE, WINDOW_SECONDS,
        aligned_len,
    },
    error::UnipaseError,
    model::EnhanceModel,
    plc,
};
use crate::audio::{
    AudioError, DecodedAudio, MonoS16Stream,
    encode::{self, Format},
    pipeline::NativeBuf,
    resample::Resampler,
};

/// Everything one enhancement run needs beyond the recording itself.
#[derive(Debug, Clone)]
pub struct EnhanceOptions {
    /// The rate to write at. `None` keeps the recording's own rate.
    ///
    /// The pipeline works at 16 kHz whatever this is, and the result is
    /// resampled to what is asked for. A rate above 16 kHz therefore buys a
    /// container that matches the source, not bandwidth that is not there.
    pub sample_rate: Option<u32>,
    /// Whether to conceal lost packets.
    pub plc: bool,
}

impl Default for EnhanceOptions {
    fn default() -> Self {
        Self {
            sample_rate: None,
            plc: true,
        }
    }
}

/// How far along a run is, reported as it goes.
#[derive(Debug, Clone, Copy)]
pub struct EnhanceProgress {
    /// 1-based index of the window just finished.
    pub window: usize,
    /// Windows in the recording.
    pub windows: usize,
    /// Seconds of audio the finished windows cover.
    pub done_seconds: f64,
    /// Seconds of audio in the recording.
    pub total_seconds: f64,
}

/// A callback invoked after each window.
pub type Progress<'a> = &'a mut dyn FnMut(EnhanceProgress);

/// The enhanced recording.
#[derive(Debug, Clone)]
pub struct Enhanced {
    /// Mono samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Their rate.
    pub sample_rate: u32,
    /// The recording's own rate, before anything was done to it.
    pub source_sample_rate: u32,
    /// Windows the pipeline ran.
    pub windows: usize,
    /// Frames the packet-loss detector concealed.
    pub concealed_frames: usize,
}

impl Enhanced {
    /// Length of the result, in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Seconds of audio the packet-loss detector filled in.
    #[must_use]
    pub fn concealed_seconds(&self) -> f64 {
        self.concealed_frames as f64 * HOP as f64 / f64::from(SAMPLE_RATE)
    }

    /// Writes the audio to `path`.
    ///
    /// WAV keeps the samples exactly as enhanced — 32-bit float. FLAC is
    /// integer-only, so the samples are quantized to 24 bits first.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when the encoder or the file write fails.
    pub fn write(&self, path: &Path, format: Format) -> Result<(), AudioError> {
        let audio = match format {
            Format::Wav => DecodedAudio::from_parts(
                self.sample_rate,
                None,
                None,
                NativeBuf::F32(vec![self.samples.clone()]),
            ),
            Format::Flac => {
                let peak = f64::from((1u32 << 23) - 1);
                let quantized: Vec<i32> = self
                    .samples
                    .iter()
                    .map(|&sample| {
                        let scaled = (f64::from(sample) * peak).round();
                        (scaled.clamp(-peak, peak) as i32) << 8
                    })
                    .collect();
                DecodedAudio::from_parts(
                    self.sample_rate,
                    None,
                    Some(24),
                    NativeBuf::S32(vec![quantized]),
                )
            },
        };
        encode::write(path, &audio, format)
    }

    /// Writes the audio through an external `ffmpeg`, which picks the container
    /// and codec from `path`'s extension.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when ffmpeg is missing, fails, or the file cannot
    /// be written.
    pub fn write_ffmpeg(
        &self,
        path: &Path,
        bitrate: Option<&str>,
    ) -> Result<(), AudioError> {
        let audio = DecodedAudio::from_parts(
            self.sample_rate,
            None,
            None,
            NativeBuf::F32(vec![self.samples.clone()]),
        );
        encode::write_ffmpeg(path, &audio, bitrate)
    }
}

/// Enhances a recording.
///
/// # Errors
///
/// Returns [`UnipaseError::Decode`] when the file cannot be read,
/// [`UnipaseError::Empty`] when it holds no audio, and
/// [`UnipaseError::Compute`] when the network fails.
pub fn enhance_file(
    path: &Path,
    model: &mut dyn EnhanceModel,
    options: &EnhanceOptions,
    progress: Progress<'_>,
) -> Result<Enhanced, UnipaseError> {
    let mut stream = MonoS16Stream::open(path, SAMPLE_RATE)?;
    let source_rate = stream.source_sample_rate();
    let mut samples = Vec::new();
    while let Some(block) = stream.next_block()? {
        samples.extend(
            block.into_iter().map(|sample| f32::from(sample) / 32768.0),
        );
    }
    if samples.is_empty() {
        return Err(UnipaseError::Empty);
    }
    let out_rate = options.sample_rate.unwrap_or(source_rate);
    let enhanced =
        enhance_samples(&samples, model, options, out_rate, progress)?;
    Ok(Enhanced {
        source_sample_rate: source_rate,
        ..enhanced
    })
}

/// The same, over samples already at 16 kHz — the entry point the tests and the
/// ASR preprocessing stage use.
///
/// # Errors
///
/// Returns [`UnipaseError::Compute`] when the network fails.
pub fn enhance_samples(
    samples: &[f32],
    model: &mut dyn EnhanceModel,
    options: &EnhanceOptions,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, UnipaseError> {
    if samples.is_empty() {
        return Err(UnipaseError::Empty);
    }
    let peak = samples.iter().fold(0f32, |acc, &v| acc.max(v.abs()));

    let spans = windows(samples.len());
    let total_seconds = samples.len() as f64 / f64::from(SAMPLE_RATE);
    let mut pieces = Vec::with_capacity(spans.len());
    let mut concealed = 0usize;
    for (index, &(start, end)) in spans.iter().enumerate() {
        let window = &samples[start..end];
        let aligned = align(window);
        let lost = if options.plc {
            plc::lost_frames(&aligned)
        } else {
            Vec::new()
        };
        concealed += lost.iter().filter(|&&flag| flag).count();
        let mut piece = model.enhance_window(&aligned, &lost)?;
        // The reference brings every window back to the length that went in.
        piece.resize(window.len(), 0.0);
        pieces.push(piece);
        progress(EnhanceProgress {
            window: index + 1,
            windows: spans.len(),
            done_seconds: (end as f64 / f64::from(SAMPLE_RATE))
                .min(total_seconds),
            total_seconds,
        });
    }

    let mut out = stitch(&pieces, SAMPLE_RATE);
    let level = out.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
    let gain = peak / (level + 1e-8);
    for sample in &mut out {
        *sample *= gain;
    }

    if out_rate != SAMPLE_RATE {
        out = resample_mono(&out, SAMPLE_RATE, out_rate)?;
    }
    Ok(Enhanced {
        samples: out,
        sample_rate: out_rate,
        source_sample_rate: SAMPLE_RATE,
        windows: spans.len(),
        concealed_frames: concealed,
    })
}

/// The window boundaries of a recording, as `[start, end)` sample ranges.
///
/// The reference's own arithmetic: eight-second windows every four seconds,
/// and a tail shorter than six seconds is glued onto the previous window
/// instead of becoming one of its own — a two-second window has too little
/// context on either side to be worth running.
fn windows(len: usize) -> Vec<(usize, usize)> {
    let span = WINDOW_SECONDS * SAMPLE_RATE as usize;
    let hop = HOP_SECONDS * SAMPLE_RATE as usize;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < len {
        let end = start + span;
        if end > len {
            if spans.is_empty() || end - len < 2 * SAMPLE_RATE as usize {
                spans.push((start, len));
            } else {
                // Extend the last full window to the end of the recording.
                let last = spans.len() - 1;
                spans[last].1 = len;
            }
            break;
        }
        spans.push((start, end));
        start += hop;
    }
    spans
}

/// Brings a window to the length the extractor wants, padding with silence or
/// trimming as [`aligned_len`] says.
fn align(window: &[f32]) -> Vec<f32> {
    let wanted = aligned_len(window.len());
    let mut aligned = window.to_vec();
    aligned.resize(wanted, 0.0);
    debug_assert_eq!(aligned.len() % HOP, PAD_REMAINDER);
    aligned
}

/// Joins the enhanced windows, taking each one's middle.
fn stitch(pieces: &[Vec<f32>], rate: u32) -> Vec<f32> {
    if pieces.len() == 1 {
        return pieces[0].clone();
    }
    let overlap = (WINDOW_SECONDS - HOP_SECONDS) * rate as usize;
    let head = (WINDOW_SECONDS * rate as usize) - overlap / 2;
    let mut out = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        if index == 0 {
            out.extend_from_slice(&piece[..head.min(piece.len())]);
        } else if index + 1 == pieces.len() {
            let from = (overlap / 2).min(piece.len());
            out.extend_from_slice(&piece[from..]);
        } else {
            let from = (overlap / 2).min(piece.len());
            let to = head.min(piece.len());
            if to > from {
                out.extend_from_slice(&piece[from..to]);
            }
        }
    }
    out
}

/// Resamples one mono track between two rates.
fn resample_mono(
    samples: &[f32],
    from: u32,
    to: u32,
) -> Result<Vec<f32>, UnipaseError> {
    if from == to {
        return Ok(samples.to_vec());
    }
    let mut resampler =
        Resampler::<f32>::new(to, from, 1).map_err(UnipaseError::from)?;
    let mut out = resampler.feed(&[samples]).remove(0);
    out.extend(resampler.finish().remove(0));
    Ok(out)
}

#[cfg(test)]
mod tests;
