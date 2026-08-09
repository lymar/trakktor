//! The driver: a file in, a repaired one out.
//!
//! Much simpler than the generative engine's. That one has to cut a recording
//! into eight-second windows because its encoder holds the whole window's
//! activations at once and its edges are where it knows least. This network is
//! causal and carries its state in its convolutions and recurrences, so the
//! only reason to cut at all is memory: the dual-path block holds the entire
//! time-by-frequency grid, and an hour of audio is 225 000 frames of it.
//!
//! The chunks therefore **continue** one another rather than overlap: the
//! recurrences, the causal convolutions' history and the overlap-add all carry
//! across, and each chunk's analysis takes its context from the recording
//! rather than mirroring at the join. The result is identical to one long pass,
//! sample for sample.
//!
//! Overlapping was tried first and is not good enough, which is worth writing
//! down: this network's memory is far longer than any overlap worth paying for.
//! Measured against the reference on two minutes of audio, thirty-second chunks
//! overlapping by one second gave a cosine of 0.9971, and *fifteen* seconds of
//! overlap only reached 0.9980 — while carrying the state costs a few hundred
//! floats and is exact.

use std::path::Path;

use super::{
    config::{CHUNK_SECONDS, HOP, SAMPLE_RATE, frames},
    runtime::CandleModel,
};
use crate::{
    audio::{MonoS16Stream, resample::Resampler},
    enhance::{
        EnhanceError, EnhanceOptions, EnhanceProgress, Enhanced, Progress,
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
    model: &mut CandleModel,
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
    let enhanced = enhance_samples(&samples, model, out_rate, progress)?;
    Ok(Enhanced {
        source_sample_rate: source_rate,
        ..enhanced
    })
}

/// The same, over samples already at 16 kHz.
///
/// # Errors
///
/// Returns [`EnhanceError::Empty`] for no audio and
/// [`EnhanceError::Compute`] when the network fails.
pub fn enhance_samples(
    samples: &[f32],
    model: &mut CandleModel,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    if samples.is_empty() {
        return Err(EnhanceError::Empty);
    }
    model.reset();
    let spans = chunks(frames(samples.len()));
    let total_seconds = samples.len() as f64 / f64::from(SAMPLE_RATE);

    let mut out: Vec<f32> = Vec::with_capacity(samples.len());
    for (index, &count) in spans.iter().enumerate() {
        out.extend(model.enhance_frames(samples, count)?);
        let done = spans[..=index].iter().sum::<usize>() * HOP;
        progress(EnhanceProgress {
            window: index + 1,
            windows: spans.len(),
            done_seconds: (done as f64 / f64::from(SAMPLE_RATE))
                .min(total_seconds),
            total_seconds,
        });
    }
    out.extend(model.finish());
    out.resize(samples.len(), 0.0);

    if out_rate != SAMPLE_RATE {
        out = resample_mono(&out, SAMPLE_RATE, out_rate)?;
    }
    Ok(Enhanced {
        samples: out,
        sample_rate: out_rate,
        source_sample_rate: SAMPLE_RATE,
        windows: spans.len(),
        concealed_frames: 0,
        concealed_seconds: 0.0,
    })
}

/// How many frames each chunk takes. They simply partition the recording —
/// there is no overlap to arrange, because the state carries.
fn chunks(total: usize) -> Vec<usize> {
    let span = CHUNK_SECONDS * SAMPLE_RATE as usize / HOP;
    let mut spans = Vec::new();
    let mut left = total;
    while left > 0 {
        let take = left.min(span);
        spans.push(take);
        left -= take;
    }
    spans
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
