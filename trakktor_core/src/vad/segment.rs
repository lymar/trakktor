//! The speech-timestamp state machine.
//!
//! A faithful port of the original Silero-VAD `get_speech_timestamps`: it turns
//! the per-window speech probabilities into a list of speech segments, with
//! hysteresis (a separate lower threshold to leave speech), a minimum silence
//! before a segment closes, a minimum speech length, an optional maximum speech
//! length, and symmetric padding that splits the gap between close neighbours.
//!
//! The machine is **incremental** ([`SpeechDetector`]): probabilities are fed
//! one window at a time and segments are emitted as soon as they are final —
//! a segment's trailing pad depends on the gap to its successor, so emission
//! trails by one raw segment (or less, once enough silence has passed that the
//! gap rule is already decided). The batch [`speech_timestamps`] drives the
//! detector over a whole probability vector and returns everything at once.
//!
//! Everything works in samples at 16 kHz; positions are quantized to the
//! 512-sample window grid (`cur = 512 * window_index`). Segments are returned
//! in seconds on the original timeline.

#[cfg(test)]
mod tests;

use super::{SAMPLE_RATE, WINDOW_SIZE};

/// A detected speech segment, in seconds on the original timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSegment {
    /// Start time, seconds.
    pub start: f64,
    /// End time, seconds.
    pub end: f64,
}

/// Tuning of the speech-timestamp state machine. The defaults are Silero's
/// canonical values (`min_silence` 100 ms matches whisper.cpp too).
#[derive(Debug, Clone, Copy)]
pub struct VadOptions {
    /// Probability at or above which a window is speech. Leaving speech uses a
    /// lower threshold, `max(threshold - 0.15, 0.01)`.
    pub threshold: f32,
    /// Segments shorter than this are dropped as non-speech.
    pub min_speech_duration_ms: u32,
    /// A silence shorter than this does not end a segment (brief pauses are
    /// bridged).
    pub min_silence_duration_ms: u32,
    /// Padding added on each side of a segment.
    pub speech_pad_ms: u32,
    /// Force-split speech longer than this many seconds; `None` never splits.
    pub max_speech_duration_s: Option<f32>,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            max_speech_duration_s: None,
        }
    }
}

/// The fixed 98 ms "minimum silence at maximum speech" of the reference, in
/// samples at 16 kHz.
const MIN_SILENCE_AT_MAX_MS: i64 = 98;

/// Milliseconds → samples at 16 kHz (`16000 / 1000 = 16`, always integer).
fn ms_to_samples(ms: i64) -> i64 { (SAMPLE_RATE as i64 / 1000) * ms }

/// Turns per-window speech probabilities (one per 512-sample window) into
/// speech segments in seconds. `n_samples` is the original audio length.
#[must_use]
pub fn speech_timestamps(
    probs: &[f32],
    n_samples: usize,
    options: &VadOptions,
) -> Vec<SpeechSegment> {
    let mut detector = SpeechDetector::new(options);
    let mut out = Vec::new();
    for &prob in probs {
        detector.push(prob, &mut out);
    }
    detector.finish(n_samples, &mut out);
    out
}

/// The incremental speech-timestamp state machine.
///
/// Feed one probability per 512-sample window with [`push`](Self::push);
/// finalized segments are appended to the caller's vector. Call
/// [`finish`](Self::finish) once at end of audio to flush the tail. The
/// concatenated emissions equal the batch [`speech_timestamps`] exactly.
#[derive(Debug, Clone)]
pub struct SpeechDetector {
    // Derived thresholds (samples / probabilities).
    threshold: f32,
    neg_threshold: f32,
    min_speech: i64,
    speech_pad: i64,
    min_silence: i64,
    min_silence_at_max: i64,
    max_speech: f64,
    // Raw state machine.
    window_index: i64,
    triggered: bool,
    cur_start: i64,
    temp_end: i64,
    prev_end: i64,
    next_start: i64,
    possible_ends: Vec<(i64, i64)>,
    // Emission: the last closed raw segment (start pad already applied),
    // whose trailing pad is not yet decided.
    pending: Option<[i64; 2]>,
}

impl SpeechDetector {
    /// A fresh detector for the given options.
    #[must_use]
    pub fn new(options: &VadOptions) -> Self {
        let threshold = options.threshold;
        let speech_pad = ms_to_samples(i64::from(options.speech_pad_ms));
        // `max_speech` in samples, or +∞ when unset. Reference:
        // sr * max_speech_s - window - 2 * speech_pad.
        let max_speech: f64 = match options.max_speech_duration_s {
            Some(s) if s.is_finite() && s > 0.0 => {
                f64::from(SAMPLE_RATE as u32) * f64::from(s) -
                    WINDOW_SIZE as f64 -
                    2.0 * speech_pad as f64
            },
            _ => f64::INFINITY,
        };
        Self {
            threshold,
            neg_threshold: (threshold - 0.15).max(0.01),
            min_speech: ms_to_samples(i64::from(
                options.min_speech_duration_ms,
            )),
            speech_pad,
            min_silence: ms_to_samples(i64::from(
                options.min_silence_duration_ms,
            )),
            min_silence_at_max: ms_to_samples(MIN_SILENCE_AT_MAX_MS),
            max_speech,
            window_index: 0,
            triggered: false,
            cur_start: 0,
            temp_end: 0,
            prev_end: 0,
            next_start: 0,
            possible_ends: Vec::new(),
            pending: None,
        }
    }

    /// Feeds the probability of the next window, appending any segments this
    /// window finalizes to `out`.
    pub fn push(&mut self, prob: f32, out: &mut Vec<SpeechSegment>) {
        let cur = WINDOW_SIZE as i64 * self.window_index;
        self.window_index += 1;
        self.step(prob, cur, out);

        // Once enough silence has passed after the pending segment that no
        // future neighbour can trigger the gap-splitting rule (a future raw
        // segment starts at `cur` at the earliest), its trailing pad is
        // decided; emit it without waiting for the next segment.
        if !self.triggered &&
            let Some(seg) = self.pending &&
            cur - seg[1] >= 2 * self.speech_pad
        {
            self.pending = None;
            self.emit([seg[0], seg[1] + self.speech_pad], None, out);
        }
    }

    /// Flushes the machine at end of audio (`n_samples` is the total length),
    /// appending the remaining segments to `out`.
    pub fn finish(mut self, n_samples: usize, out: &mut Vec<SpeechSegment>) {
        let audio_len = n_samples as i64;
        // A segment still open at the end.
        if self.triggered && (audio_len - self.cur_start) > self.min_speech {
            self.raw_closed([self.cur_start, audio_len], out);
        }
        if let Some(seg) = self.pending.take() {
            self.emit(
                [seg[0], (seg[1] + self.speech_pad).min(audio_len)],
                Some(audio_len),
                out,
            );
        }
    }

    /// The earliest second any *future* output could start at, or `None` when
    /// nothing is pending — everything strictly before it is settled. Callers
    /// holding raw audio (to cut the detected speech out of) may drop samples
    /// below this point.
    #[must_use]
    pub fn earliest_pending_second(&self) -> Option<f64> {
        let pending = self.pending.map(|seg| seg[0]);
        let open = self
            .triggered
            .then(|| (self.cur_start - self.speech_pad).max(0));
        match (pending, open) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
        .map(|samples| samples as f64 / SAMPLE_RATE as f64)
    }

    /// One step of the reference state machine at position `cur`. Any raw
    /// segments closed by this window go through
    /// [`raw_closed`](Self::raw_closed).
    fn step(&mut self, prob: f32, cur: i64, out: &mut Vec<SpeechSegment>) {
        // Speech returned after a provisional end: record a candidate silence
        // and clear the pending end.
        if prob >= self.threshold && self.temp_end != 0 {
            let silence = cur - self.temp_end;
            if silence > self.min_silence_at_max {
                self.possible_ends.push((self.temp_end, silence));
            }
            self.temp_end = 0;
            if self.next_start < self.prev_end {
                self.next_start = cur;
            }
        }

        // Start of speech.
        if prob >= self.threshold && !self.triggered {
            self.triggered = true;
            self.cur_start = cur;
            return;
        }

        // Maximum speech length reached: cut.
        if self.triggered && (cur - self.cur_start) as f64 > self.max_speech {
            if let Some(&(pe, dur)) =
                self.possible_ends.iter().max_by_key(|&&(_, d)| d)
            {
                // Cut at the longest internal silence and possibly reopen.
                let start = self.cur_start;
                self.raw_closed([start, pe], out);
                self.next_start = pe + dur;
                if self.next_start < pe + cur {
                    self.cur_start = self.next_start;
                } else {
                    self.triggered = false;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.possible_ends.clear();
            } else if self.prev_end != 0 {
                // Legacy cut at the last valid silence.
                let seg = [self.cur_start, self.prev_end];
                self.raw_closed(seg, out);
                if self.next_start < self.prev_end {
                    self.triggered = false;
                } else {
                    self.cur_start = self.next_start;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.possible_ends.clear();
            } else {
                // No candidate silence: hard cut at the current sample.
                let seg = [self.cur_start, cur];
                self.raw_closed(seg, out);
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
                self.possible_ends.clear();
                return;
            }
        }

        // Silence while in speech.
        if prob < self.neg_threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = cur;
            }
            if cur - self.temp_end < self.min_silence {
                // Too short to be a real gap: bridge it.
                return;
            }
            // Long enough: close the segment at the silence start, keeping it
            // only if it is long enough.
            if self.temp_end - self.cur_start > self.min_speech {
                let seg = [self.cur_start, self.temp_end];
                self.raw_closed(seg, out);
            }
            self.triggered = false;
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.possible_ends.clear();
        }
    }

    /// A raw segment has closed: apply the pairwise padding rule against the
    /// pending predecessor (emitting it), and hold this one as pending.
    ///
    /// Mirrors the reference `pad_and_merge` forward pass, which reads only
    /// *raw* neighbour values: the first segment's start moves back by the pad;
    /// a gap of at least `2 * pad` gives both sides the full pad, a smaller gap
    /// is split evenly.
    fn raw_closed(&mut self, mut seg: [i64; 2], out: &mut Vec<SpeechSegment>) {
        match self.pending.take() {
            Some(mut prev) => {
                let silence = seg[0] - prev[1];
                if silence < 2 * self.speech_pad {
                    prev[1] += silence / 2;
                    seg[0] = (seg[0] - silence / 2).max(0);
                } else {
                    prev[1] += self.speech_pad;
                    seg[0] = (seg[0] - self.speech_pad).max(0);
                }
                self.emit(prev, None, out);
            },
            None => {
                // Either the very first segment (plain leading pad) or one
                // whose predecessor was emitted early — which only happens
                // once the gap is at least `2 * pad`, so the full pad applies
                // on this side too. The same arithmetic either way.
                seg[0] = (seg[0] - self.speech_pad).max(0);
            },
        }
        self.pending = Some(seg);
    }

    /// Converts a finalized segment to seconds and appends it. `audio_len`
    /// clamps the end at end-of-audio (only the last segment can need it).
    fn emit(
        &self,
        seg: [i64; 2],
        audio_len: Option<i64>,
        out: &mut Vec<SpeechSegment>,
    ) {
        let sr = SAMPLE_RATE as f64;
        let end = seg[1] as f64 / sr;
        out.push(SpeechSegment {
            start: (seg[0] as f64 / sr).max(0.0),
            end: match audio_len {
                Some(len) => end.min(len as f64 / sr),
                None => end,
            },
        });
    }
}
