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
//! Each chunk carries three nested spans, which [`Chunk`] keeps apart: the
//! **speech** it reports as its timestamps, the span it **owns** (the speech
//! plus its share of the boundary pauses — see [`BOUNDARY_CONTEXT_S`]), and the
//! wider audio window it **hears** (the owned span plus
//! [`BOUNDARY_OVERLAP_S`] of the neighbour on each side). Owned spans tile the
//! recording without overlapping; the windows deliberately overlap, and what a
//! chunk transcribes outside its owned span is context that the caller drops.
//!
//! The DP transition is windowed by [`MAX_SPEECH_S`]: a chunk can only span
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

/// Hard upper bound on the audio window handed to the model, and the span of
/// the DP transition window.
pub const MAX_DURATION_S: f64 = 30.0;

/// A speech interval shorter than this (seconds) is dropped as noise before
/// segmentation.
pub const MIN_INTERVAL_S: f64 = 0.05;

/// How much of the pause on each boundary, in seconds, is worth *hearing*.
///
/// Speech boundaries come from the VAD, which puts them where the energy
/// crosses its threshold plus a 30 ms pad — too tight for an encoder, which
/// then starts and ends mid-breath and clips the quiet run-in of a phrase. The
/// pause is real audio, so the window takes some of it; but a long pause is
/// silence, and feeding more of it than this buys nothing while costing
/// compute.
///
/// This bounds only the audio. Which side a word found in the pause belongs to
/// is a separate question, settled by ownership — see [`Chunk`] — and there the
/// pause is always split down the middle, however long it is.
pub const BOUNDARY_CONTEXT_S: f64 = 1.0;

/// How far, in seconds, the audio window reaches past the owned span into the
/// neighbour — heard for context only, never transcribed into the output.
///
/// [`BOUNDARY_CONTEXT_S`] hands a chunk the silence around its boundary, but
/// silence is not what the encoder is missing there: past the edge of the
/// window it sees nothing at all, so the last word before a cut is decoded
/// without the speech that follows it. Widening the *heard* audio past the
/// owned span — and dropping whatever the model transcribes out there, because
/// the neighbour owns it — gives that word its right (and the first word after
/// the cut its left) context back.
///
/// Measured on spontaneous Russian speech: the word next to a cut is decoded
/// wrong about twice as often as the same word next to an in-chunk pause, and
/// three seconds of overlap closes that gap — beyond it the curve is flat,
/// matching what the word gets with unlimited context. The cost is the extra
/// audio: at the target chunk length, about a third more compute.
pub const BOUNDARY_OVERLAP_S: f64 = 3.0;

/// Hard upper bound on a chunk's *speech* span: what is left of
/// [`MAX_DURATION_S`] once both boundaries take their context and overlap. Also
/// the presplit threshold for continuous speech, so that every window fits the
/// model's bound.
pub const MAX_SPEECH_S: f64 =
    MAX_DURATION_S - 2.0 * (BOUNDARY_CONTEXT_S + BOUNDARY_OVERLAP_S);

/// Decode-time disagreement allowed at an ownership boundary, seconds.
///
/// The two neighbours decode the seam audio independently, so the same word
/// comes back with different timings from each — slightly different for clear
/// speech, but a quiet word inside the boundary pause, or ill-fitting audio
/// the model segments arbitrarily, can move by seconds. A hard midpoint test
/// against the same line would then sometimes keep such a word twice — both
/// sides placing it on their own side — and sometimes drop it twice, both
/// placing it on the other's. A word this close to the line is instead
/// reconciled against what the previous chunk actually kept there: one whose
/// audio is already covered is a duplicate and is dropped, an uncovered one
/// is kept even from past the line.
///
/// The rule is asymmetric, because the two mistakes cost differently. On the
/// chunk's own side of the line dropping a *covered* word merely removes a
/// second transcription of the same audio. On the neighbour's side a word is
/// adopted only when its audio went completely untranscribed: a partial
/// overlap means the neighbour heard the stretch and segmented it its own way,
/// and adopting a differently-cut variant next to the neighbour's words would
/// transcribe the audio twice.
///
/// The zone is as wide as the overlap itself: a word farther from the line
/// than that was heard by one side only, so there is no second decode to
/// disagree with — plain ownership is exact there.
pub const SEAM_JITTER_S: f64 = BOUNDARY_OVERLAP_S;

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

/// One chunk of a long recording, as three nested spans.
///
/// - `start`/`end` — the **speech**, and what the engine reports as the
///   segment's timestamps.
/// - `keep_start`/`keep_end` — what the chunk **owns**: out to the middle of
///   the pause on each side, however long that pause is. Owned spans **tile**
///   the recording — they neither overlap nor leave a gap — so every word,
///   wherever it is decoded, belongs to exactly one chunk. Splitting only up to
///   [`BOUNDARY_CONTEXT_S`] would leave the middle of a long pause unowned, and
///   a word the overlap lets a neighbour hear there would be dropped by both.
/// - `window_start`/`window_end` — what the model **hears**: the owned span
///   plus [`BOUNDARY_OVERLAP_S`] of the neighbour on each side. Adjacent
///   windows *do* overlap; whatever the model transcribes outside the owned
///   span is context, and the caller drops it.
///
/// Local times inside a decoded chunk are relative to `window_start`, not to
/// `start`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    /// Start of the speech span, seconds — the segment's reported start.
    pub start: f64,
    /// End of the speech span, seconds — the segment's reported end.
    pub end: f64,
    /// Start of the owned span: words decoded before this belong to the
    /// previous chunk.
    pub keep_start: f64,
    /// End of the owned span: words decoded after this belong to the next one.
    pub keep_end: f64,
    /// Start of the audio window fed to the model, at or before `keep_start`.
    pub window_start: f64,
    /// End of the audio window fed to the model, at or after `keep_end`.
    pub window_end: f64,
}

impl Chunk {
    /// Length of the audio window in seconds — what the model actually sees.
    #[must_use]
    pub fn window_len(&self) -> f64 { self.window_end - self.window_start }

    /// Whether a word centred at `time` (seconds, original timeline) belongs to
    /// this chunk rather than to a neighbour that also heard it.
    #[must_use]
    pub fn owns(&self, time: f64) -> bool {
        self.keep_start <= time && time < self.keep_end
    }
}

/// Groups speech intervals into chunks, choosing cuts by dynamic programming so
/// they fall on the longest pauses while chunks stay near
/// [`TARGET_DURATION_S`], then widening each chunk's audio window into the
/// pauses on its boundaries.
///
/// `content_start` and `content_end` bound the audio: `content_end` is also the
/// clamp for the last interval's end and for the last window. In a streaming
/// run `content_start` is the right edge of the previously *owned* span, so
/// ownership tiles across commits; a window may still reach below it, which is
/// the overlap. Returns an empty vector when there is no speech.
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
            if chunk_len > MAX_SPEECH_S && i != k {
                // Past the window; longer chunks are inadmissible. (i == k is
                // always admissible: presplitting keeps every unit within
                // MAX_SPEECH_S.)
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
            // Half the pause on each side — the whole of it between the two
            // neighbours, so ownership tiles without gaps. An outer edge has no
            // neighbour to halve with and claims only what it can hear:
            // otherwise a chunk committed across a long silence would own it
            // all the way to the frontier, and speech still open there would be
            // left inside someone else's span.
            let reach = BOUNDARY_CONTEXT_S + BOUNDARY_OVERLAP_S;
            let half_left = (if i == 0 {
                (start - content_start).min(reach)
            } else {
                (start - spans[i - 1].1) / 2.0
            })
            .max(0.0);
            let half_right = (if i + 1 == n {
                (content_end - end).min(reach)
            } else {
                (spans[i + 1].0 - end) / 2.0
            })
            .max(0.0);

            // Of that pause, only so much is worth hearing; past it the window
            // reaches into the neighbour for context.
            let mut left = half_left.min(BOUNDARY_CONTEXT_S);
            let mut right = half_right.min(BOUNDARY_CONTEXT_S);
            // Keep the window within the hard bound. Presplitting caps the
            // speech span at MAX_SPEECH_S, which leaves room for both sides;
            // the trim only guards a span that reaches the cap exactly.
            let budget =
                (MAX_DURATION_S - 2.0 * BOUNDARY_OVERLAP_S - (end - start))
                    .max(0.0);
            if left + right > budget {
                let scale = budget / (left + right);
                left *= scale;
                right *= scale;
            }

            // The overlap is worth paying for only when it can reach the
            // neighbour's *speech*: across a pause of `reach` or longer it
            // would hear nothing but silence, which measurably buys no
            // accuracy while costing encoder time on ~a third of the seams.
            // An outer edge knows only the distance to the content bound —
            // half the pause, in a streamed run — so its test doubles it.
            let l_pause = if i == 0 {
                2.0 * (start - content_start)
            } else {
                start - spans[i - 1].1
            };
            let r_pause = if i + 1 == n {
                2.0 * (content_end - end)
            } else {
                spans[i + 1].0 - end
            };
            let l_over = if l_pause < reach {
                BOUNDARY_OVERLAP_S
            } else {
                0.0
            };
            let r_over = if r_pause < reach {
                BOUNDARY_OVERLAP_S
            } else {
                0.0
            };

            Chunk {
                start,
                end,
                keep_start: start - half_left,
                keep_end: end + half_right,
                // The overlap is heard, not owned: it may reach past
                // `content_start` into what a previous commit already
                // transcribed, but never past the audio that exists.
                window_start: (start - left - l_over).max(0.0),
                window_end: (end + right + r_over).min(content_end),
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
    /// Right edge of the last committed chunk's *owned* span. No later chunk
    /// may own audio below it, so ownership tiles across commits just as it
    /// does within one DP run. Windows may still reach below it — that is the
    /// overlap, and it is heard, not owned.
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
            self.floor = last.keep_end;
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
    /// earliest pending speech, less everything its window may reach back
    /// into: the pause it owns and the overlap past that. The overlap crosses
    /// the last committed boundary by design, so `floor` does not bound it.
    #[must_use]
    pub fn earliest_pending_start(&self) -> Option<f64> {
        self.pending.first().map(|i| {
            (i.start - BOUNDARY_CONTEXT_S - BOUNDARY_OVERLAP_S).max(0.0)
        })
    }

    /// Final cuts over whatever is still pending; `total` clamps the ends.
    pub fn finish(&mut self, total: f64) -> Vec<Chunk> {
        let cuts = chunk_boundaries(&self.pending, self.floor, total);
        self.pending.clear();
        if let Some(last) = cuts.last() {
            self.floor = last.keep_end;
        }
        cuts
    }
}

/// A decoded word, as ownership filtering sees it: a text with a time span.
/// Implemented by every engine's word type so the seam logic lives once.
pub trait DecodedWord {
    /// The word's `(start, end)` in seconds.
    fn span(&self) -> (f64, f64);
    /// Moves the word by `offset` seconds (chunk-local → original timeline).
    fn shift(&mut self, offset: f64);
    /// The word's text.
    fn text(&self) -> &str;
}

/// Shifts a chunk's decoded words onto the original timeline (by
/// `window_start` expressed in whole samples), keeps the words the chunk owns,
/// and renders the kept text.
///
/// Words outside the owned span were heard across a boundary and belong to the
/// neighbour, which decodes them with its own full context. Right at the
/// ownership line the two decodes may disagree by [`SEAM_JITTER_S`] about
/// which side a word's midpoint falls on, so there the test defers to
/// `prev_kept` — the `(start, end)` spans the previous chunk actually kept:
/// audio already covered is dropped as a duplicate, uncovered audio is kept
/// even from just past the line, so a word is never emitted twice and never
/// lost to a disagreement. (The far end needs no such rule here — the *next*
/// chunk applies it against this chunk's kept words.)
///
/// A chunk that keeps everything it decoded keeps `whole` verbatim: rebuilding
/// would be a no-op in principle, but it would also quietly normalize whatever
/// spacing the decoder produced. Only a chunk that actually discards words has
/// its text rebuilt, and there the words are the decoded text split on its own
/// word boundaries, so joining them is faithful.
pub fn own_words<W: DecodedWord>(
    words: Vec<W>,
    origin: f64,
    chunk: &Chunk,
    prev_kept: &[(f64, f64)],
    whole: String,
) -> (String, Vec<W>) {
    let heard = words.len();
    let kept: Vec<W> = words
        .into_iter()
        .map(|mut w| {
            w.shift(origin);
            w
        })
        .filter(|w| {
            let (start, end) = w.span();
            let mid = (start + end) / 2.0;
            let line = mid - chunk.keep_start;
            if (0.0..SEAM_JITTER_S).contains(&line) {
                // Our side of the line: drop a word whose audio the previous
                // chunk clearly already transcribed — its timing merely put
                // the midpoint over here.
                !covered(prev_kept, start, end)
            } else if (-SEAM_JITTER_S..0.0).contains(&line) {
                // The neighbour's side: adopt only audio nobody transcribed
                // at all — anything of the neighbour's anywhere near means it
                // heard this stretch and rendered it with its own timing or
                // its own word boundaries, and adopting our variant next to
                // that would transcribe the audio twice. Punctuation-only
                // pieces are never worth adopting.
                w.text().chars().any(char::is_alphanumeric) &&
                    untouched(prev_kept, start, end)
            } else {
                chunk.owns(mid)
            }
        })
        .collect();
    let text = if kept.len() == heard {
        whole
    } else {
        kept.iter()
            .map(DecodedWord::text)
            .collect::<Vec<_>>()
            .join(" ")
    };
    (text, kept)
}

/// Whether `[start, end]` overlaps one of the kept spans by more than half of
/// the shorter of the two — the same audio, not merely adjacent words touching.
fn covered(kept: &[(f64, f64)], start: f64, end: f64) -> bool {
    kept.iter().any(|&(ks, ke)| {
        let overlap = end.min(ke) - start.max(ks);
        overlap > 0.0 && overlap > 0.5 * (end - start).min(ke - ks)
    })
}

/// Clearance required around an adopted word, seconds. Word spans carry only
/// the frames that emitted tokens — a CTC word often spans a tenth of its
/// acoustic duration — so two decodes of the same word can produce disjoint
/// spans a few dozen milliseconds apart, and an intersection test would call
/// the audio free. Re-decodes of the same audio start within ~50 ms of each
/// other, while a genuinely untranscribed word sits in a pause, several
/// hundred milliseconds clear of the neighbour's last one; this threshold
/// separates the two with margin on both sides.
const ADOPT_CLEARANCE_S: f64 = 0.2;

/// Whether nothing kept comes within [`ADOPT_CLEARANCE_S`] of `[start, end]` —
/// audio that genuinely went untranscribed, as opposed to audio the neighbour
/// transcribed with its own timing or its own word boundaries.
fn untouched(kept: &[(f64, f64)], start: f64, end: f64) -> bool {
    kept.iter().all(|&(ks, ke)| {
        ke < start - ADOPT_CLEARANCE_S || ks > end + ADOPT_CLEARANCE_S
    })
}

/// Turns raw speech intervals into DP units: clamps to the content, drops
/// too-short intervals, and presplits any interval longer than
/// [`MAX_SPEECH_S`] into equal parts (so every unit fits the window; the
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
        if interval.duration() > MAX_SPEECH_S {
            let parts = (interval.duration() / MAX_SPEECH_S).ceil() as usize;
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
