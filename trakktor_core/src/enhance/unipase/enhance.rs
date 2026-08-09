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
    plc,
};
use crate::{
    audio::{MonoS16Stream, resample::Resampler},
    enhance::{
        EnhanceError, EnhanceModel, EnhanceOptions, EnhanceProgress, Enhanced,
        Progress,
    },
};

/// Enhances a recording.
///
/// # Errors
///
/// Returns [`EnhanceError::Decode`] when the file cannot be read,
/// [`EnhanceError::Empty`] when it holds no audio, and
/// [`EnhanceError::Compute`] when the network fails.
pub fn enhance_file(
    path: &Path,
    model: &mut dyn EnhanceModel,
    options: &EnhanceOptions,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    let mut stream = MonoS16Stream::open(path, SAMPLE_RATE)?;
    let source_rate = stream.source_sample_rate();
    let mut samples = Vec::new();
    while let Some(block) = stream.next_block()? {
        samples.extend(
            block.into_iter().map(|sample| f32::from(sample) / 32768.0),
        );
    }
    if samples.is_empty() {
        return Err(EnhanceError::Empty);
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
/// Returns [`EnhanceError::Compute`] when the network fails.
pub fn enhance_samples(
    samples: &[f32],
    model: &mut dyn EnhanceModel,
    options: &EnhanceOptions,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    if samples.is_empty() {
        return Err(EnhanceError::Empty);
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
        concealed_seconds: concealed as f64 * HOP as f64 /
            f64::from(SAMPLE_RATE),
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
) -> Result<Vec<f32>, EnhanceError> {
    if from == to {
        return Ok(samples.to_vec());
    }
    let mut resampler =
        Resampler::<f32>::new(to, from, 1).map_err(EnhanceError::from)?;
    let mut out = resampler.feed(&[samples]).remove(0);
    out.extend(resampler.finish().remove(0));
    Ok(out)
}

#[cfg(test)]
mod tests;
