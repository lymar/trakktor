//! Preparing the reference recording — the only thing that decides the voice.
//!
//! The reference implementation does this with an external audio library: split
//! the file on long silences, take up to twelve seconds of it, trim the edges,
//! append a little silence, then normalize the loudness. This does the same
//! work with the built-in decoder, with one deliberate difference: it does not
//! reassemble the reference out of pieces. It trims the edges and, if what is
//! left is still longer than the model's window, cuts at the last pause that
//! fits — a predictable "the first so many seconds of it" rather than a
//! collage.

use std::path::Path;

use super::{
    config::{
        HOP, REF_MAX_SECONDS, REF_TAIL_SILENCE, SAMPLE_RATE, SILENCE_DBFS,
        TARGET_RMS,
    },
    error::EspeechError,
};

/// Window the silence detector works in, in seconds — the granularity the
/// reference's own detector uses.
const DETECT_WINDOW: f64 = 0.01;

/// Shortest reference that can be used at all, in seconds. Below this there is
/// no voice to take and no speech rate to measure.
const MIN_SECONDS: f64 = 0.5;

/// Earliest a long reference may be cut at a pause, in seconds. Cutting at the
/// first pause of a twelve-second recording would throw away most of it.
const MIN_KEEP_SECONDS: f64 = 6.0;

/// A reference recording, ready to condition on.
#[derive(Debug, Clone)]
pub struct Reference {
    /// Mono samples at [`SAMPLE_RATE`], loudness-normalized.
    pub wave: Vec<f32>,
    /// The loudness it had before normalization; the synthesized waveform is
    /// scaled back by it so a quiet reference does not force a quiet result.
    pub rms: f32,
    /// Whether it had to be shortened to the model's window.
    pub clipped: bool,
}

impl Reference {
    /// Seconds of audio.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        self.wave.len() as f64 / f64::from(SAMPLE_RATE)
    }

    /// Frames the duration estimate counts it as.
    ///
    /// This is the plain sample count over the hop — one less than the frames
    /// its mel spectrogram actually has, because the centred transform adds a
    /// frame. The reference cuts the generated audio at this frame too, so the
    /// last reference frame ends up as the first frame of the result.
    #[must_use]
    pub fn cut_frames(&self) -> usize { self.wave.len() / HOP }

    /// The gain the synthesized waveform is scaled by, undoing the
    /// normalization applied here.
    #[must_use]
    pub fn output_gain(&self) -> f32 {
        if self.rms < TARGET_RMS && self.rms > 0.0 {
            self.rms / TARGET_RMS
        } else {
            1.0
        }
    }
}

/// Reads and prepares the reference at `path`.
///
/// # Errors
///
/// Returns [`EspeechError::RefAudioDecode`] when the file cannot be decoded and
/// [`EspeechError::RefAudioTooShort`] when it holds too little audio to
/// condition on.
#[cfg(feature = "audio")]
pub fn prepare(path: &Path) -> Result<Reference, EspeechError> {
    let wave = crate::audio::decode_to_mono_f32(path, SAMPLE_RATE)
        .map_err(|e| EspeechError::RefAudioDecode(e.to_string()))?;
    let reference = prepare_wave(wave);
    if reference.seconds() < MIN_SECONDS {
        return Err(EspeechError::RefAudioTooShort(reference.seconds()));
    }
    Ok(reference)
}

/// Trims, caps, and normalizes an already-decoded reference.
#[must_use]
pub fn prepare_wave(wave: Vec<f32>) -> Reference {
    let window = (DETECT_WINDOW * f64::from(SAMPLE_RATE)) as usize;
    let threshold = 10f32.powf(SILENCE_DBFS / 20.0);
    let trimmed = trim_silence(&wave, window, threshold);

    let cap = (REF_MAX_SECONDS * f64::from(SAMPLE_RATE)) as usize;
    let clipped = trimmed.len() > cap;
    let kept = if clipped {
        let floor = (MIN_KEEP_SECONDS * f64::from(SAMPLE_RATE)) as usize;
        let cut = last_pause(&trimmed[..cap], window, threshold, floor)
            .unwrap_or(cap);
        &trimmed[..cut]
    } else {
        trimmed
    };

    let mut prepared = kept.to_vec();
    prepared.extend(std::iter::repeat_n(
        0.0,
        (REF_TAIL_SILENCE * f64::from(SAMPLE_RATE)) as usize,
    ));

    let rms = rms(&prepared);
    if rms < TARGET_RMS && rms > 0.0 {
        let gain = TARGET_RMS / rms;
        for sample in &mut prepared {
            *sample *= gain;
        }
    }
    Reference {
        wave: prepared,
        rms,
        clipped,
    }
}

/// Drops the leading and trailing windows whose loudness is under `threshold`.
fn trim_silence(wave: &[f32], window: usize, threshold: f32) -> &[f32] {
    let quiet = |start: usize| {
        rms(&wave[start..(start + window).min(wave.len())]) < threshold
    };
    let mut head = 0;
    while head + window <= wave.len() && quiet(head) {
        head += window;
    }
    if head >= wave.len() {
        return &[];
    }
    let mut tail = wave.len();
    while tail >= head + window && rms(&wave[tail - window..tail]) < threshold {
        tail -= window;
    }
    &wave[head..tail]
}

/// The end of the last silent window at or after `floor`, if there is one — the
/// point a too-long reference is cut at.
fn last_pause(
    wave: &[f32],
    window: usize,
    threshold: f32,
    floor: usize,
) -> Option<usize> {
    let mut found = None;
    let mut start = floor;
    while start + window <= wave.len() {
        if rms(&wave[start..start + window]) < threshold {
            found = Some(start + window);
        }
        start += window;
    }
    found
}

/// Root mean square of `samples`, on the full-scale-is-one scale the silence
/// threshold and the loudness target are both stated in.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum();
    (sum / samples.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests;
