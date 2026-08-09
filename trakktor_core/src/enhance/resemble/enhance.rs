//! The driver: a file in, a repaired one out, for both of this project's
//! networks.
//!
//! Everything here is host-side and runtime-independent — decoding, the chunks,
//! the per-chunk normalization, the cross-fade, the output rate. A runtime is
//! asked for exactly one thing at a time: run this chunk.
//!
//! # The chunks are upstream's own
//!
//! Unlike [`mpsenet`](super::super::mpsenet), this project **has** a long-form
//! path and it is the one reproduced here: thirty-second chunks that share one
//! second, each normalized to its own peak, cross-faded linearly over the
//! shared second. Nothing had to be designed by measurement, because the
//! reference already answered it.
//!
//! Two details of that path are worth naming, because neither is obvious and
//! both change the samples:
//!
//! - **each chunk is scaled to its own peak** and scaled back afterwards, so a
//!   quiet passage is presented to the network as loud as a shouted one. That
//!   is the reference's decision, not this port's, and it is the opposite of
//!   what [`mpsenet`](super::super::mpsenet) does;
//! - **a tenth of a second of silence is appended** to every chunk before it is
//!   run and dropped afterwards, because the transform's last frame is a copy
//!   of the one before it and this keeps the copy off the end of real audio.
//!
//! # What is not reproduced, and why
//!
//! Upstream re-aligns consecutive chunks before adding them, by cross-
//! correlating the mel spectrograms of the second they share and shifting by
//! the argmax. Nothing in either network can move the waveform in time — the
//! analysis is centred, the vocoder's upsampling is fixed, and the conditioning
//! is frame-aligned — so the shift it computes should always be zero, and on
//! real recordings it measures zero. It is left out, and the driver adds the
//! chunks where they belong.

use std::path::Path;

use super::config::{SAMPLE_RATE, TAIL_PAD, chunk_samples, overlap_samples};
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
    let mut decoded = Vec::new();
    while let Some(block) = stream.next_block()? {
        decoded.extend(
            block.into_iter().map(|sample| f32::from(sample) / 32768.0),
        );
    }
    if decoded.is_empty() {
        return Err(EnhanceError::Empty);
    }
    let out_rate = options.sample_rate.unwrap_or(source_rate);
    let enhanced = enhance_owned(decoded, model, out_rate, progress)?;
    Ok(Enhanced {
        source_sample_rate: source_rate,
        ..enhanced
    })
}

/// The same, over samples already at [`SAMPLE_RATE`].
///
/// # Errors
///
/// Returns [`EnhanceError::Empty`] for no audio and
/// [`EnhanceError::Compute`] when the network fails.
pub fn enhance_samples(
    input: &[f32],
    model: &mut dyn EnhanceModel,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    enhance_owned(input.to_vec(), model, out_rate, progress)
}

/// The same, taking the track rather than borrowing it, so that the result can
/// be written back over the input as it is finished. A three-hour recording is
/// 1.9 GB per full-length buffer at this rate, and there is no reason to hold
/// two.
fn enhance_owned(
    mut track: Vec<f32>,
    model: &mut dyn EnhanceModel,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    if track.is_empty() {
        return Err(EnhanceError::Empty);
    }
    let span = chunk_samples();
    let overlap = overlap_samples();
    let hop = span - overlap;
    let length = track.len();
    let total_seconds = length as f64 / f64::from(SAMPLE_RATE);
    let starts = chunk_starts(length);
    let count = starts.len();

    // The faded tail of the chunk before, waiting for the chunk it is shared
    // with. Everything earlier than this has already been written back over the
    // input and will not be read again.
    let mut carry = vec![0f32; overlap];

    for (index, &start) in starts.iter().enumerate() {
        let end = (start + span).min(length);
        let mut piece = run_chunk(model, &track[start..end])?;

        let first = index == 0;
        let last = index + 1 == count;
        if !first {
            fade_in(&mut piece, overlap);
        }
        // The reference fades the tail of every chunk but the last — including
        // a lone chunk's, whose faded second lies past the end of the recording
        // and is trimmed away before anyone sees it. Skipping it there is the
        // same result.
        if !last {
            fade_out(&mut piece, overlap);
        }
        for (sample, &carried) in piece.iter_mut().zip(&carry) {
            *sample += carried;
        }

        // What is settled: everything up to where the next chunk begins.
        let settled = if last { piece.len() } else { hop };
        track[start..start + settled].copy_from_slice(&piece[..settled]);
        if !last {
            carry.clear();
            carry.extend_from_slice(&piece[settled..]);
            carry.resize(overlap, 0.0);
        }

        progress(EnhanceProgress {
            window: index + 1,
            windows: count,
            done_seconds: (end as f64 / f64::from(SAMPLE_RATE))
                .min(total_seconds),
            total_seconds,
        });
    }

    let mut out = track;
    if out_rate != SAMPLE_RATE {
        out = resample_mono(&out, SAMPLE_RATE, out_rate)?;
    }
    Ok(Enhanced {
        samples: out,
        sample_rate: out_rate,
        source_sample_rate: SAMPLE_RATE,
        windows: count,
        concealed_frames: 0,
        concealed_seconds: 0.0,
    })
}

/// Where each chunk starts, as the reference steps them: every hop from zero
/// while there is any recording left, so the last chunk may be very short —
/// down to a single sample — and is zero-padded by whoever needs it whole.
fn chunk_starts(length: usize) -> Vec<usize> {
    let hop = chunk_samples() - overlap_samples();
    (0..length).step_by(hop).collect()
}

/// One chunk through the network: scale to its own peak, append the silence the
/// transform's copied last frame wants, run, trim, scale back.
fn run_chunk(
    model: &mut dyn EnhanceModel,
    chunk: &[f32],
) -> Result<Vec<f32>, EnhanceError> {
    let peak = chunk
        .iter()
        .fold(0f32, |peak, &sample| peak.max(sample.abs()))
        .max(super::config::PEAK_EPS);
    let mut padded = Vec::with_capacity(chunk.len() + TAIL_PAD);
    padded.extend(chunk.iter().map(|&sample| sample / peak));
    padded.resize(chunk.len() + TAIL_PAD, 0.0);

    let mut out = model.enhance_window(&padded, &[])?;
    out.truncate(chunk.len());
    out.resize(chunk.len(), 0.0);
    for sample in &mut out {
        *sample *= peak;
    }
    Ok(out)
}

/// Ramps the first `overlap` samples in from zero, the way
/// `linspace(0, 1, overlap)` does — both endpoints included.
fn fade_in(piece: &mut [f32], overlap: usize) {
    let span = overlap.min(piece.len());
    if span < 2 {
        return;
    }
    let last = (span - 1) as f32;
    for (index, sample) in piece[..span].iter_mut().enumerate() {
        *sample *= index as f32 / last;
    }
}

/// Ramps the last `overlap` samples out to zero.
fn fade_out(piece: &mut [f32], overlap: usize) {
    let span = overlap.min(piece.len());
    if span < 2 {
        return;
    }
    let last = (span - 1) as f32;
    let start = piece.len() - span;
    for (index, sample) in piece[start..].iter_mut().enumerate() {
        *sample *= 1.0 - index as f32 / last;
    }
}

/// Resamples one mono track between two rates.
fn resample_mono(
    input: &[f32],
    from: u32,
    to: u32,
) -> Result<Vec<f32>, EnhanceError> {
    if from == to {
        return Ok(input.to_vec());
    }
    let mut resampler =
        Resampler::<f32>::new(to, from, 1).map_err(EnhanceError::from)?;
    let mut out = resampler.feed(&[input]).remove(0);
    out.extend(resampler.finish().remove(0));
    Ok(out)
}

#[cfg(test)]
mod tests;
