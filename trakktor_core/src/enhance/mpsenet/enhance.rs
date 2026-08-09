//! The driver: a file in, a repaired one out.
//!
//! Everything here is host-side and runtime-independent — decoding, the one
//! gain the reference applies, the windows, the stitching, the output rate. A
//! runtime is asked for exactly one thing at a time: enhance this window.
//!
//! # The gain, and why it is taken over the whole recording
//!
//! The reference scales its input to unit mean square before the transform and
//! divides the result back afterwards, which is how the network was trained to
//! be fed. It does that per file, and its files are utterances of a few
//! seconds. Here the same factor is taken over the whole recording rather than
//! per window: the network then sees a quiet passage as quiet, which is what a
//! per-file factor does to a quiet passage of a short file too — where a
//! per-window factor would lift a pause full of nothing to the level of speech
//! and ask the network to clean *that*.
//!
//! # Why windows at all
//!
//! The reference has no long-form inference: it runs an utterance in one pass.
//! It cannot be followed here, because the attention inside each of the four
//! two-stage blocks spans the recording along **both** axes — one pass over
//! frequency and one over time — and the time one is quadratic. An hour of
//! audio is 576 000 frames: the score matrix of **one** head at **one**
//! frequency would be 1.3 TB, and the pass wants a hundred and one frequencies
//! times four heads of it.
//!
//! So the recording is cut into [`WINDOW_SECONDS`]-second windows that share
//! [`OVERLAP_SECONDS`] second, and the shared second is cross-faded. The
//! overlap is short because the measurement says a long one buys nothing:
//! a window's output depends on *which* window a moment lands in — the
//! instance norms take their statistics over the whole window — but hardly at
//! all on how far from its edge it lands, past the first quarter-second.

use std::path::Path;

use super::config::{
    OVERLAP_SECONDS, SAMPLE_RATE, WINDOW_SECONDS, frames, samples,
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
    // The decoded track is handed over rather than borrowed: it is scaled in
    // place, so a three-hour recording costs one 636 MB buffer here instead of
    // two.
    let enhanced = enhance_owned(decoded, model, out_rate, progress)?;
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
    input: &[f32],
    model: &mut dyn EnhanceModel,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    enhance_owned(input.to_vec(), model, out_rate, progress)
}

/// The same, taking the track rather than borrowing it, so that the scaling
/// happens in place. Every full-length buffer matters on a recording of hours.
fn enhance_owned(
    mut scaled: Vec<f32>,
    model: &mut dyn EnhanceModel,
    out_rate: u32,
    progress: Progress<'_>,
) -> Result<Enhanced, EnhanceError> {
    if scaled.is_empty() {
        return Err(EnhanceError::Empty);
    }
    let gain = level_gain(&scaled);
    for value in &mut scaled {
        *value *= gain;
    }

    let spans = windows(scaled.len());
    let total_seconds = scaled.len() as f64 / f64::from(SAMPLE_RATE);
    let mut sum = vec![0f32; scaled.len()];
    let mut weight = vec![0f32; scaled.len()];

    for (index, &(start, end)) in spans.iter().enumerate() {
        let mut piece = model.enhance_window(&scaled[start..end], &[])?;
        // The transform returns whole frames, which is at most the length that
        // went in; the reference writes exactly that and lets the file end
        // there, so the last fraction of a hop is silence rather than a guess.
        piece.resize(end - start, 0.0);
        let rising = if index == 0 {
            0
        } else {
            spans[index - 1].1.saturating_sub(start)
        };
        let falling = spans
            .get(index + 1)
            .map_or(0, |next| end.saturating_sub(next.0));
        for (offset, value) in piece.iter().enumerate() {
            let ramp = fade(offset, end - start, rising, falling);
            sum[start + offset] += value * ramp;
            weight[start + offset] += ramp;
        }
        progress(EnhanceProgress {
            window: index + 1,
            windows: spans.len(),
            done_seconds: (end as f64 / f64::from(SAMPLE_RATE))
                .min(total_seconds),
            total_seconds,
        });
    }

    // In place, and the fade weights are released before anything else is
    // allocated: three hours of audio is 636 MB per full-length buffer, and
    // there is no reason to hold a fourth.
    for (value, &weight) in sum.iter_mut().zip(&weight) {
        *value = if weight > 1e-6 {
            *value / (weight * gain)
        } else {
            0.0
        };
    }
    drop(weight);
    let mut out = sum;

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

/// Enhances one window: analyse, hand the spectrum to the network, synthesise.
///
/// Shared by the runtimes, so neither of them owns a transform and the two
/// cannot disagree about what the network was given or about the last step of
/// what it produced.
///
/// # Errors
///
/// Whatever `denoise` returns.
pub(super) fn enhance_window<F>(
    window: &[f32],
    denoise: F,
) -> Result<Vec<f32>, EnhanceError>
where
    F: FnOnce(
        &super::stft::Spectrum,
    ) -> Result<super::stft::Spectrum, EnhanceError>,
{
    let spectrum = super::stft::analyze(window);
    let denoised = denoise(&spectrum)?;
    debug_assert_eq!(denoised.frames, spectrum.frames);
    let mut wave = super::stft::synthesize(&denoised);
    wave.resize(samples(frames(window.len())), 0.0);
    Ok(wave)
}

/// The factor that brings a recording to unit mean square — the reference's
/// `sqrt(n / Σx²)`. Silence has no level to match, and keeps its own; so does
/// anything so faint that the factor would not fit in an `f32`, which cannot
/// arrive through the decoder (its quietest non-zero sample is 1/32768) but can
/// through [`enhance_samples`].
fn level_gain(input: &[f32]) -> f32 {
    let energy: f64 = input.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    if energy <= 0.0 {
        return 1.0;
    }
    let gain = (input.len() as f64 / energy).sqrt() as f32;
    if gain.is_finite() { gain } else { 1.0 }
}

/// The cross-fade weight of a sample at `offset` in a window of `len`, given
/// how much of it the previous and the next window overlap.
///
/// Linear, and complementary by construction: what one window gives up over
/// the shared second the next takes on, so the two weights sum to one.
fn fade(offset: usize, len: usize, rising: usize, falling: usize) -> f32 {
    let up = if rising > 0 && offset < rising {
        offset as f32 / rising as f32
    } else {
        1.0
    };
    let down = if falling > 0 && offset >= len - falling {
        (len - offset) as f32 / falling as f32
    } else {
        1.0
    };
    up.min(down)
}

/// The window boundaries of a recording, as `[start, end)` sample ranges.
///
/// Consecutive windows share [`OVERLAP_SECONDS`], and the last one runs to the
/// end of the recording however short that leaves it.
///
/// **It cannot leave a sliver**, and that is arithmetic rather than a guard:
/// the loop advances only while a whole window still fits, so on the step that
/// ended it the previous start satisfied `start − hop + span < len` — that is,
/// `len − start > span − hop`, the overlap. The tail is therefore always longer
/// than the second it shares with its predecessor, and there is nothing to
/// special-case. (An earlier version special-cased it anyway, with a branch no
/// input could reach.)
fn windows(len: usize) -> Vec<(usize, usize)> {
    let span = WINDOW_SECONDS * SAMPLE_RATE as usize;
    let hop = span - OVERLAP_SECONDS * SAMPLE_RATE as usize;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start + span < len {
        spans.push((start, start + span));
        start += hop;
    }
    spans.push((start, len));
    spans
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
