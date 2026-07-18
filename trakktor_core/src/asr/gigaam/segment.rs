//! Long-form segmentation.
//!
//! The models transcribe audio only up to a few tens of seconds, so a longer
//! recording is split into chunks along speech boundaries. Chunk boundaries are
//! chosen by dynamic programming over the pauses between speech intervals: each
//! candidate cut is a pause, and the optimal set of cuts trades the chunk-size
//! penalty against a bonus that grows with the pause length, so cuts land on
//! the longest available pauses (likely sentence/thought boundaries) while
//! chunks stay near the target duration.
//!
//! A chunk is a *contiguous* span of the original audio (silence between its
//! speech intervals is kept), so the encoder sees continuous context; only the
//! chunk boundaries are chosen from the speech layout, and the pause on a
//! boundary is dropped (it belongs to neither chunk).
//!
//! The DP transition is windowed by [`MAX_DURATION_S`]: a chunk can only span
//! pauses within that many seconds, so the cost is `O(n·W)` — linear in the
//! number of pauses `n`, with `W` the (small, bounded) count of pauses inside
//! one max-length window. This stays cheap even for multi-hour files.

#[cfg(test)]
mod tests;

/// Preferred chunk length in seconds; the size penalty is minimized here.
pub const TARGET_DURATION_S: f64 = 18.0;

/// Hard upper bound on a chunk: no chunk may exceed this, and the DP transition
/// window spans exactly this duration.
pub const MAX_DURATION_S: f64 = 30.0;

/// A speech interval shorter than this (seconds) is dropped as noise before
/// segmentation.
pub const MIN_INTERVAL_S: f64 = 0.05;

/// Weight of the pause-length bonus relative to the size penalty. Larger favors
/// cutting on long pauses at the cost of size uniformity.
const GAP_BONUS: f64 = 12.0;

/// Pause length (seconds) past which the bonus saturates — any pause longer
/// than this is treated as an equally good boundary.
const GAP_CAP_S: f64 = 2.5;

/// A speech interval on the original timeline, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub start: f64,
    pub end: f64,
}

impl Interval {
    fn duration(&self) -> f64 { self.end - self.start }
}

/// Groups speech intervals into chunk boundaries `[start, end)` in seconds,
/// choosing cuts by dynamic programming so they fall on the longest pauses
/// while chunks stay near [`TARGET_DURATION_S`].
///
/// `content_end` is the audio duration; the last interval's end is clamped to
/// it. Returns an empty vector when there is no speech.
pub fn chunk_boundaries(
    speech: &[Interval],
    content_end: f64,
) -> Vec<(f64, f64)> {
    let units = build_units(speech, content_end);
    if units.is_empty() {
        return Vec::new();
    }
    let n = units.len();

    // Pause length after unit k (0 for the boundary introduced by presplitting
    // an over-long interval, and unused for the last unit).
    let gap_after = |k: usize| -> f64 {
        if k + 1 < n {
            (units[k + 1].start - units[k].end).max(0.0)
        } else {
            0.0
        }
    };

    // dp[k] = minimum cost to cover units[0..=k] with a cut right after unit k
    // (unit k is the last of its chunk). start_of[k] = the first unit of that
    // final chunk, for reconstruction.
    let mut dp = vec![f64::INFINITY; n];
    let mut start_of = vec![0usize; n];

    for k in 0..n {
        // Walk candidate chunk starts i from k downward while the chunk
        // [units[i].start, units[k].end] fits the window — this is the
        // O(W) windowed transition.
        let mut i = k;
        loop {
            let chunk_len = units[k].end - units[i].start;
            if chunk_len > MAX_DURATION_S && i != k {
                // Past the window; longer chunks are inadmissible. (i == k is
                // always admissible: presplitting keeps every unit within
                // MAX_DURATION_S.)
                break;
            }
            let base = if i == 0 { 0.0 } else { dp[i - 1] };
            if base.is_finite() {
                let cost = base +
                    size_cost(chunk_len) +
                    boundary_cost(gap_after(k), k, n);
                if cost < dp[k] {
                    dp[k] = cost;
                    start_of[k] = i;
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    // Reconstruct: walk cuts backward from the last unit.
    let mut boundaries = Vec::new();
    let mut k = n - 1;
    loop {
        let i = start_of[k];
        boundaries.push((units[i].start, units[k].end));
        if i == 0 {
            break;
        }
        k = i - 1;
    }
    boundaries.reverse();
    boundaries
}

/// Size penalty: squared deviation from the target duration.
fn size_cost(len: f64) -> f64 {
    let d = len - TARGET_DURATION_S;
    d * d
}

/// Boundary cost: a bonus (negative) that grows with the pause length, capped.
/// The final unit ends the file — a mandatory boundary with no pause, so no
/// bonus.
fn boundary_cost(gap: f64, k: usize, n: usize) -> f64 {
    if k + 1 == n {
        0.0
    } else {
        -GAP_BONUS * gap.min(GAP_CAP_S)
    }
}

/// Turns raw speech intervals into DP units: clamps to the content, drops
/// too-short intervals, and presplits any interval longer than
/// [`MAX_DURATION_S`] into equal parts (so every unit fits the window; the
/// introduced boundaries have a zero-length pause).
fn build_units(speech: &[Interval], content_end: f64) -> Vec<Interval> {
    let mut units = Vec::new();
    for seg in speech {
        let start = seg.start.max(0.0);
        let end = seg.end.min(content_end);
        if end - start < MIN_INTERVAL_S {
            continue;
        }
        let interval = Interval { start, end };
        if interval.duration() > MAX_DURATION_S {
            let parts = (interval.duration() / MAX_DURATION_S).ceil() as usize;
            let step = interval.duration() / parts as f64;
            for p in 0..parts {
                let s = start + step * p as f64;
                let e = if p + 1 == parts { end } else { s + step };
                units.push(Interval { start: s, end: e });
            }
        } else {
            units.push(interval);
        }
    }
    units
}
