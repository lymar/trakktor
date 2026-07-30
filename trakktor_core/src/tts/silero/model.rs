//! The contract a runtime fulfils, and the rules that sit above it.
//!
//! Everything the reference does to the numbers *between* its networks — the
//! exponential the duration head is read through, the clamps on the first and
//! last symbols, the threshold under which a predicted pitch counts as none —
//! lives here rather than in either backend. Those are decisions, not
//! arithmetic, and a runtime that restated them could disagree with the other
//! one about how long a word is.
//!
//! The seam itself is a single call. The pipeline is one pass: there is no
//! solver to step and no cache to keep, so there is nothing to gain from
//! handing the caller anything but the finished spectrum.

use super::{
    config::{Config, HEAD_DURATION, PITCH_FLOOR, TAIL_DURATIONS},
    error::SileroError,
    istft::{Pqmf, Window},
};

/// One utterance, as a runtime receives it.
#[derive(Debug, Clone)]
pub struct Utterance {
    /// Symbol ids, opened and closed by the model's own symbols.
    pub ids: Vec<u32>,
    /// Which speaker to use, as a row of the speaker table.
    pub speaker: usize,
    /// Speech-rate multiplier per symbol.
    pub rate: Vec<f32>,
    /// Pitch multiplier per symbol.
    pub pitch: Vec<f32>,
    /// Utterance-type id per symbol, for a model with an intonation head.
    pub types: Option<Vec<u32>>,
}

/// What a runtime hands back: the vocoder head's output, ready for the inverse
/// transform, and the durations that shaped it.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Magnitudes, `[frames, bins]` in row-major order.
    pub magnitude: Vec<f32>,
    /// Phases, same shape.
    pub phase: Vec<f32>,
    /// Mel frames the utterance came to.
    pub frames: usize,
    /// Frames each symbol was given.
    pub durations: Vec<u32>,
}

/// The loaded networks, behind which candle and burn are interchangeable.
pub trait SpeechModel: Send {
    /// The geometry the checkpoint was loaded with.
    fn config(&self) -> &Config;

    /// The analysis window of the inverse transform, as the checkpoint carries
    /// it.
    fn window(&self) -> &Window;

    /// The filterbank that produces a rate `bands` times lower than the
    /// native one.
    fn pqmf(&self, bands: usize) -> Option<&Pqmf>;

    /// Runs the whole pipeline for one utterance.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::Checkpoint`] on a backend failure and
    /// [`SileroError::TextTooLong`] when the predicted durations do not fit the
    /// model's window.
    fn synthesize(
        &self,
        utterance: &Utterance,
    ) -> Result<Spectrum, SileroError>;
}

/// Turns the duration head's output into frame counts.
///
/// The head predicts `log(1 + frames)`; what follows is the reference's own
/// post-processing, and every step of it is audible. The rounding is
/// ties-to-even because that is what torch does, and a half-frame difference
/// at a word boundary is a different reading of the durations that follow.
#[must_use]
pub fn durations(log_durations: &[f32], rate: &[f32]) -> Vec<u32> {
    let mut frames: Vec<f32> = log_durations
        .iter()
        .map(|value| (value.exp() - 1.0).max(0.0).round_ties_even())
        .collect();
    // The reference clamps the opening symbol before dividing by the rate as
    // well as after, so a slow rate cannot stretch it back out.
    clamp_first(&mut frames);
    for (slot, rate) in frames.iter_mut().zip(rate) {
        *slot = (*slot / rate).round_ties_even();
    }
    clamp_first(&mut frames);
    let length = frames.len();
    for (from_end, value, force) in TAIL_DURATIONS {
        let Some(index) = length.checked_sub(from_end) else {
            continue;
        };
        let value = value as f32;
        if force || frames[index] > value {
            frames[index] = value;
        }
    }
    frames.iter().map(|value| *value as u32).collect()
}

/// Holds the first symbol — the start-of-sequence marker — to the length the
/// reference allows it.
fn clamp_first(frames: &mut [f32]) {
    if let Some(first) = frames.first_mut() {
        *first = first.min(HEAD_DURATION as f32);
    }
}

/// Applies the pitch coefficients to the pitch head's output, in place.
///
/// `mean_std` is the speaker's own scale: raising the pitch shifts the whole
/// contour as well as stretching it, and by how much depends on how wide that
/// speaker's range is. A symbol the head gave no pitch at all keeps none.
pub fn shape_pitch(pitch: &mut [f32], coefficients: &[f32], mean_std: f32) {
    for value in pitch.iter_mut() {
        if value.abs() < PITCH_FLOOR {
            *value = 0.0;
        }
    }
    for (value, coefficient) in pitch.iter_mut().zip(coefficients) {
        let scaled = *value * coefficient;
        // A coefficient of zero is the reference's "flatten this" — it takes
        // the contour away but leaves the speaker's own level alone.
        let shift = if *coefficient == 0.0 || scaled == 0.0 {
            0.0
        } else {
            (coefficient - 1.0) * mean_std
        };
        *value = scaled + shift;
    }
}

/// Checks that an utterance fits the model's position table before the decoder
/// is asked to build it.
///
/// # Errors
///
/// Returns [`SileroError::TextTooLong`] with the numbers the caller needs to
/// explain itself.
pub fn check_frames(frames: usize, config: &Config) -> Result<(), SileroError> {
    if frames > config.max_frames() {
        return Err(SileroError::TextTooLong {
            frames,
            limit: config.max_frames(),
            seconds: config.max_seconds(),
        });
    }
    Ok(())
}

/// Guards the two indices a backend would otherwise take on faith.
///
/// candle turns an out-of-range narrow into an error, but burn panics — and a
/// panic past loading is a bug by this engine's own contract. The synthesis
/// driver cannot produce either overrun; this is for a library caller building
/// an [`Utterance`] by hand.
pub(super) fn check_input(
    utterance: &Utterance,
    config: &Config,
) -> Result<(), SileroError> {
    if utterance.ids.len() > config.positions {
        return Err(SileroError::InvalidOptions(format!(
            "the utterance holds {} symbols but the model's position table \
             holds {}; split the text",
            utterance.ids.len(),
            config.positions
        )));
    }
    if utterance.speaker >= config.speaker_slots {
        return Err(SileroError::InvalidOptions(format!(
            "speaker {} is outside the model's {} rows",
            utterance.speaker, config.speaker_slots
        )));
    }
    Ok(())
}

/// The index vector that expands one value per symbol into one per frame.
///
/// This is the length regulator: `repeat_interleave` has no direct equivalent
/// in either backend, and both build it the same way, from here.
#[must_use]
pub fn expansion(durations: &[u32]) -> Vec<u32> {
    let total: usize = durations.iter().map(|d| *d as usize).sum();
    let mut index = Vec::with_capacity(total);
    for (symbol, count) in durations.iter().enumerate() {
        index.extend(std::iter::repeat_n(symbol as u32, *count as usize));
    }
    index
}

#[cfg(test)]
mod tests;
