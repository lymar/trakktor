//! What the solver is told before it starts: how long the utterance will be,
//! which timesteps to visit, and what noise to begin from.
//!
//! All three are arithmetic over host values, shared by the runtimes. That the
//! noise is drawn here and handed down is what makes a seeded run reproduce on
//! either of them.

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::config::{SHORT_TEXT_BYTES, SHORT_TEXT_SPEED, UTTERANCE_WINDOW};

/// Ceiling on a single utterance, in mel frames — the reference's own guard
/// against a duration estimate that has gone wrong.
const MAX_FRAMES: usize = 65_536;

/// Timesteps the solver visits for these step counts are not the plain division
/// of the interval: the reference ships a pruned schedule for low step counts,
/// found empirically, and uses it whenever one is tabulated.
///
/// The values are thirty-seconds of the interval.
const PRUNED: &[(usize, &[u32])] = &[
    (5, &[0, 2, 4, 8, 16, 32]),
    (6, &[0, 2, 4, 6, 8, 16, 32]),
    (7, &[0, 2, 4, 6, 8, 16, 24, 32]),
    (10, &[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]),
    (12, &[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]),
    (
        16,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    ),
];

/// The timesteps of a run, `steps + 1` of them, from 0 to 1.
///
/// The schedule is then bent by `sway`: negative values crowd the steps toward
/// the start, where the trajectory is least certain and detail is decided.
#[must_use]
pub fn timesteps(steps: usize, sway: f32) -> Vec<f32> {
    let base: Vec<f32> = PRUNED
        .iter()
        .find(|(count, _)| *count == steps)
        .map_or_else(
            || {
                (0..=steps)
                    .map(|index| index as f32 / steps as f32)
                    .collect()
            },
            |(_, points)| {
                points
                    .iter()
                    .map(|point| f32::from(*point as u16) / 32.0)
                    .collect()
            },
        );
    base.into_iter()
        .map(|t| t + sway * ((std::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t))
        .collect()
}

/// How many mel frames the utterance will be.
///
/// The estimate is the reference's: the reference recording's own speech rate,
/// in frames per byte of its transcript, applied to the text to speak. Nothing
/// about it is learned — it is a proportion, which is why `--speed` is simply a
/// divisor on it.
///
/// `cut_frames` is the reference's length in whole hops, `cond_frames` the
/// frames its mel actually has, and `text_len` the characters handed to the
/// model; the last two only matter as a floor, so that something is always
/// generated.
#[must_use]
pub fn duration_frames(
    cut_frames: usize,
    ref_bytes: usize,
    gen_bytes: usize,
    cond_frames: usize,
    text_len: usize,
    speed: f32,
) -> usize {
    let speed = if gen_bytes < SHORT_TEXT_BYTES {
        SHORT_TEXT_SPEED
    } else {
        speed
    };
    let estimate = if ref_bytes == 0 || speed <= 0.0 {
        cut_frames
    } else {
        let rate = cut_frames as f64 / ref_bytes as f64;
        cut_frames + (rate * gen_bytes as f64 / f64::from(speed)) as usize
    };
    estimate.max(text_len.max(cond_frames) + 1).min(MAX_FRAMES)
}

/// How much text one piece may hold, in bytes of UTF-8.
///
/// The reference's rule, and it depends on the reference recording rather than
/// on the model: what is left of the safe window after the reference takes its
/// seconds, at the speech rate the reference itself shows. A longer reference
/// therefore means shorter pieces.
#[must_use]
pub fn chunk_budget(ref_bytes: usize, ref_seconds: f64, speed: f32) -> usize {
    if ref_seconds <= 0.0 {
        return SHORT_TEXT_BYTES;
    }
    let room = UTTERANCE_WINDOW - ref_seconds;
    if room <= 0.0 {
        return SHORT_TEXT_BYTES;
    }
    let budget =
        ref_bytes as f64 / ref_seconds * room * f64::from(speed.max(0.0));
    (budget as usize).max(SHORT_TEXT_BYTES)
}

/// The noise the solver starts from: `frames × channels` standard normal
/// values, drawn from `seed`.
///
/// Drawn on the host, from a seeded generator, so both runtimes solve from the
/// same starting point and a run reproduces exactly. It does not reproduce the
/// reference implementation's noise for the same seed — different generator —
/// which is why parity against it is checked by handing the noise over, not by
/// sharing a seed.
#[must_use]
pub fn noise(seed: u64, frames: usize, channels: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(frames * channels);
    while out.len() < frames * channels {
        // Box-Muller: two uniforms in, two independent normals out.
        let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2: f64 = rng.random::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push((radius * angle.cos()) as f32);
        if out.len() < frames * channels {
            out.push((radius * angle.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests;
