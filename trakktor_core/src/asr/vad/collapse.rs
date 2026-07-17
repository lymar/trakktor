//! Building a dense speech-only buffer and the map back to the original time.
//!
//! This is how detected speech reaches the engine: the speech segments are
//! concatenated into one short buffer (with a short silence between them, as in
//! whisper.cpp), the engine transcribes that, and the resulting timestamps are
//! mapped back to the original timeline. Because the port builds the buffer
//! itself, every region maps 1:1 (slope 1), so the map is a single
//! piecewise-linear function — simpler than whisper.cpp's separate segment- and
//! token-level schemes.
//!
//! This module is engine-independent: it returns the buffer and a
//! [`TimeMapping`]; applying the map to a specific engine's result is left to
//! the caller.

#[cfg(test)]
mod tests;

use super::{SAMPLE_RATE, segment::SpeechSegment};

/// Silence inserted between glued speech regions, milliseconds. Matches
/// whisper.cpp: a pause cue so adjacent speech spans do not run together.
const SILENCE_MS: usize = 100;

/// One collapsed region: a span in the processed buffer that maps 1:1 back to
/// `[orig_start, orig_start + (proc_end - proc_start))` in the original audio.
/// All fields are seconds.
#[derive(Debug, Clone, Copy)]
struct Region {
    proc_start: f64,
    proc_end: f64,
    orig_start: f64,
}

impl Region {
    fn orig_end(&self) -> f64 {
        self.orig_start + (self.proc_end - self.proc_start)
    }
}

/// A piecewise-linear (slope-1) map from processed (collapsed) time back to the
/// original timeline.
#[derive(Debug, Clone)]
pub struct TimeMapping {
    regions: Vec<Region>,
}

impl TimeMapping {
    /// Maps a time in the processed buffer (seconds) back to the original
    /// timeline (seconds).
    ///
    /// Inside a region the map is a slope-1 shift. A time in the silence
    /// inserted between two regions snaps to the nearer region boundary — so a
    /// token never lands inside a removed gap. Times before the first or after
    /// the last region clamp to those ends.
    #[must_use]
    pub fn map(&self, t: f64) -> f64 {
        let (Some(first), Some(last)) =
            (self.regions.first(), self.regions.last())
        else {
            return t.max(0.0);
        };
        if t <= first.proc_start {
            return first.orig_start;
        }
        if t >= last.proc_end {
            return last.orig_end();
        }
        for (k, region) in self.regions.iter().enumerate() {
            if t <= region.proc_end {
                if t >= region.proc_start {
                    return region.orig_start + (t - region.proc_start);
                }
                // In the silence before region k (k ≥ 1 here): snap to the
                // nearer of the two boundaries.
                let prev = &self.regions[k - 1];
                let midpoint = (prev.proc_end + region.proc_start) / 2.0;
                return if t <= midpoint {
                    prev.orig_end()
                } else {
                    region.orig_start
                };
            }
        }
        last.orig_end()
    }
}

/// The collapsed audio and its time map.
pub struct Collapsed {
    /// The dense speech-only buffer (16 kHz mono f32) to transcribe.
    pub buffer: Vec<f32>,
    /// The map from the buffer's timeline back to the original.
    pub mapping: TimeMapping,
    /// The original audio duration, seconds (for reporting).
    pub duration: f64,
}

/// Concatenates the speech segments (seconds) of `audio` into a dense buffer
/// with a short silence between them, returning the buffer and its time map.
#[must_use]
pub fn collapse(audio: &[f32], speech: &[SpeechSegment]) -> Collapsed {
    let sr = SAMPLE_RATE;
    let silence = SILENCE_MS * sr / 1000;
    let n = audio.len();
    let to_sample = |seconds: f64| -> usize {
        (seconds * sr as f64).round().clamp(0.0, n as f64) as usize
    };

    let mut buffer: Vec<f32> = Vec::new();
    let mut regions: Vec<Region> = Vec::new();
    for segment in speech {
        let start = to_sample(segment.start);
        let end = to_sample(segment.end);
        if end <= start {
            continue;
        }
        if !regions.is_empty() {
            buffer.resize(buffer.len() + silence, 0.0);
        }
        let proc_start = buffer.len();
        buffer.extend_from_slice(&audio[start..end]);
        regions.push(Region {
            proc_start: proc_start as f64 / sr as f64,
            proc_end: buffer.len() as f64 / sr as f64,
            orig_start: start as f64 / sr as f64,
        });
    }

    Collapsed {
        buffer,
        mapping: TimeMapping { regions },
        duration: n as f64 / sr as f64,
    }
}
