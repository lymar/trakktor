//! Long-form segmentation, shared by the chunking ASR engines.
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
//! chunk boundaries are chosen from the speech layout.
//!
//! Each chunk carries two spans, which [`Chunk`] keeps apart: the **speech
//! span** it reports as its timestamps, and the wider **audio window** actually
//! fed to the model. The window reaches into the pauses on both boundaries, so
//! the pause is split between the neighbours instead of being dropped — see
//! [`BOUNDARY_CONTEXT_S`].
//!
//! The DP transition is windowed by [`MAX_DURATION_S`]: a chunk can only span
//! pauses within that many seconds, so the cost is `O(n·W)` — linear in the
//! number of pauses `n`, with `W` the (small, bounded) count of pauses inside
//! one max-length window. This stays cheap even for multi-hour files.
//!
//! A streamed run drives the same optimization through [`ChunkPlanner`]: it
//! holds a rolling window of finalized speech and commits chunks once they are
//! far enough behind the processed frontier that no future speech can change
//! them, so a stream and a whole-file run produce identical chunks.

#[cfg(test)]
mod tests;

/// Preferred chunk length in seconds; the size penalty is minimized here.
pub const TARGET_DURATION_S: f64 = 18.0;

/// Hard upper bound on a chunk: no chunk may exceed this, and the DP transition
/// window spans exactly this duration. It also bounds the audio window, which
/// is trimmed rather than allowed to push a chunk past the limit.
pub const MAX_DURATION_S: f64 = 30.0;

/// A speech interval shorter than this (seconds) is dropped as noise before
/// segmentation.
pub const MIN_INTERVAL_S: f64 = 0.05;

/// How far, in seconds, a chunk's audio window may reach past its speech span
/// into the pause on each boundary.
///
/// Speech boundaries come from the VAD, which puts them where the energy
/// crosses its threshold plus a 30 ms pad — too tight for an encoder, which
/// then starts and ends mid-breath and clips the quiet run-in of a phrase. The
/// pause on a chunk boundary is real audio and belongs to both neighbours, so
/// each takes up to this much of it as run-in and run-out context.
///
/// Never more than *half* the pause goes to either side, so the two windows
/// meet inside the pause and never overlap: no audio is transcribed twice, and
/// none is dropped. A boundary introduced by presplitting continuous speech has
/// no pause at all and so gets no context — widening there would overlap.
pub const BOUNDARY_CONTEXT_S: f64 = 1.0;

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

/// One chunk of a long recording: the speech it covers, and the audio window
/// handed to the model.
///
/// The two differ only at the boundaries: `window_start`/`window_end` extend
/// into the pauses by up to [`BOUNDARY_CONTEXT_S`], while `start`/`end` stay on
/// the speech and are what the engine reports as the segment's timestamps.
/// Local times inside a decoded chunk are therefore relative to `window_start`,
/// not to `start`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    /// Start of the speech span, seconds — the segment's reported start.
    pub start: f64,
    /// End of the speech span, seconds — the segment's reported end.
    pub end: f64,
    /// Start of the audio window fed to the model, at or before `start`.
    pub window_start: f64,
    /// End of the audio window fed to the model, at or after `end`.
    pub window_end: f64,
}

impl Chunk {
    /// Length of the audio window in seconds — what the model actually sees.
    #[must_use]
    pub fn window_len(&self) -> f64 { self.window_end - self.window_start }
}

/// Groups speech intervals into chunks, choosing cuts by dynamic programming so
/// they fall on the longest pauses while chunks stay near
/// [`TARGET_DURATION_S`], then widening each chunk's audio window into the
/// pauses on its boundaries.
///
/// `content_start` and `content_end` bound the audio available to the windows:
/// `content_end` is also the clamp for the last interval's end. In a streaming
/// run `content_start` is the right edge of the previously committed window, so
/// windows stay disjoint across commits. Returns an empty vector when there is
/// no speech.
pub fn chunk_boundaries(
    speech: &[Interval],
    content_start: f64,
    content_end: f64,
) -> Vec<Chunk> {
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
    let mut spans = Vec::new();
    let mut k = n - 1;
    loop {
        let i = start_of[k];
        spans.push((units[i].start, units[k].end));
        if i == 0 {
            break;
        }
        k = i - 1;
    }
    spans.reverse();
    widen(&spans, content_start, content_end)
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

/// Turns the chosen speech spans into chunks, widening each one's audio window
/// into the pauses around it.
///
/// An inner boundary yields at most half its pause to either side, so adjacent
/// windows meet inside the pause without overlapping. The outer edges have no
/// neighbour to share with and take up to the full context from the audio left
/// between them and `content_start`/`content_end`. Finally the window is
/// trimmed so it never exceeds [`MAX_DURATION_S`].
fn widen(
    spans: &[(f64, f64)],
    content_start: f64,
    content_end: f64,
) -> Vec<Chunk> {
    let n = spans.len();
    spans
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let mut left = if i == 0 {
                start - content_start
            } else {
                (start - spans[i - 1].1) / 2.0
            }
            .clamp(0.0, BOUNDARY_CONTEXT_S);
            let mut right = if i + 1 == n {
                content_end - end
            } else {
                (spans[i + 1].0 - end) / 2.0
            }
            .clamp(0.0, BOUNDARY_CONTEXT_S);

            // Keep the window within the hard bound; the speech span itself
            // already fits, so the budget is what is left of it.
            let budget = (MAX_DURATION_S - (end - start)).max(0.0);
            if left + right > budget {
                let scale = budget / (left + right);
                left *= scale;
                right *= scale;
            }

            Chunk {
                start,
                end,
                window_start: start - left,
                window_end: end + right,
            }
        })
        .collect()
}

/// Once the pending (uncommitted) speech spans this many seconds, the planner
/// runs the cut optimization and commits the stable prefix.
pub const COMMIT_HORIZON_S: f64 = 180.0;

/// Only chunks ending at least this many seconds before the pending frontier
/// are committed; the tail is re-optimized when more speech arrives. Far
/// larger than the DP's transition window, so committed cuts match what a
/// whole-file optimization would have chosen.
pub const COMMIT_MARGIN_S: f64 = 120.0;

/// Plans chunk boundaries over a rolling window of finalized speech intervals,
/// committing cuts once they are far enough behind the frontier to be stable.
///
/// Feed finalized speech with [`push`](Self::push), advance over silence with
/// [`poll`](Self::poll), and close with [`finish`](Self::finish); each returns
/// the chunks that became final, in order. Commits are keyed to the speech
/// timeline only, so the same audio always yields the same chunks no matter
/// how it was fed.
#[derive(Debug, Clone, Default)]
pub struct ChunkPlanner {
    /// Finalized speech not yet covered by a committed chunk, ordered.
    pending: Vec<Interval>,
    /// Right edge of the last committed chunk's audio window. No later window
    /// may reach below it, so windows stay disjoint across commits just as
    /// they do within one DP run.
    floor: f64,
}

impl ChunkPlanner {
    /// A fresh planner.
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Adds a finalized speech interval; returns chunks that became final.
    pub fn push(&mut self, interval: Interval) -> Vec<Chunk> {
        self.pending.push(interval);
        let start = self.pending[0].start;
        let frontier = interval.end;
        if frontier - start < COMMIT_HORIZON_S {
            return Vec::new();
        }
        self.commit(frontier)
    }

    /// Re-evaluates without new speech, `frontier` being the fully processed
    /// position — long silences also push pending speech far enough behind to
    /// commit (and release its PCM).
    pub fn poll(&mut self, frontier: f64) -> Vec<Chunk> {
        match self.pending.first() {
            Some(first) if frontier - first.start >= COMMIT_HORIZON_S => {
                self.commit(frontier)
            },
            _ => Vec::new(),
        }
    }

    /// Runs the DP over the pending intervals and commits every chunk ending
    /// at least [`COMMIT_MARGIN_S`] before `frontier`.
    ///
    /// `frontier` is the fully processed position, so it also bounds the audio
    /// available to the windows: the final pending chunk, committed by a
    /// [`poll`](Self::poll) across a long silence, takes its run-out context
    /// from that silence — the same context [`finish`](Self::finish) would
    /// have given it. (On a push-driven commit the frontier is the newest
    /// interval's end, where the bound is inert: the last span is never far
    /// enough behind it to commit.)
    fn commit(&mut self, frontier: f64) -> Vec<Chunk> {
        let cuts = chunk_boundaries(&self.pending, self.floor, frontier);
        let ready: Vec<Chunk> = cuts
            .into_iter()
            .filter(|chunk| chunk.end <= frontier - COMMIT_MARGIN_S)
            .collect();
        if let Some(last) = ready.last() {
            let boundary = last.end;
            self.floor = last.window_end;
            // Keep only speech past the committed boundary. A cut can fall
            // inside an interval (the DP presplits over-long continuous
            // speech), in which case its tail half stays pending.
            self.pending.retain_mut(|unit| {
                if unit.end <= boundary {
                    false
                } else {
                    if unit.start < boundary {
                        unit.start = boundary;
                    }
                    true
                }
            });
        }
        ready
    }

    /// The earliest position PCM may still be needed from — the start of the
    /// earliest pending speech, less the context its window may reach back
    /// into, and never below the last committed window.
    #[must_use]
    pub fn earliest_pending_start(&self) -> Option<f64> {
        self.pending
            .first()
            .map(|i| (i.start - BOUNDARY_CONTEXT_S).max(self.floor))
    }

    /// Final cuts over whatever is still pending; `total` clamps the ends.
    pub fn finish(&mut self, total: f64) -> Vec<Chunk> {
        let cuts = chunk_boundaries(&self.pending, self.floor, total);
        self.pending.clear();
        if let Some(last) = cuts.last() {
            self.floor = last.window_end;
        }
        cuts
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
