//! The speech-timestamp state machine.
//!
//! A faithful port of the original Silero-VAD `get_speech_timestamps`: it turns
//! the per-window speech probabilities into a list of speech segments, with
//! hysteresis (a separate lower threshold to leave speech), a minimum silence
//! before a segment closes, a minimum speech length, an optional maximum speech
//! length, and symmetric padding that splits the gap between close neighbours.
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
    let window = WINDOW_SIZE as i64;
    let threshold = options.threshold;
    let neg_threshold = (threshold - 0.15).max(0.01);
    let min_speech = ms_to_samples(i64::from(options.min_speech_duration_ms));
    let speech_pad = ms_to_samples(i64::from(options.speech_pad_ms));
    let min_silence = ms_to_samples(i64::from(options.min_silence_duration_ms));
    let min_silence_at_max = ms_to_samples(MIN_SILENCE_AT_MAX_MS);
    let audio_len = n_samples as i64;
    // `max_speech` in samples, or +∞ when unset. Reference:
    // sr * max_speech_s - window - 2 * speech_pad.
    let max_speech: f64 = match options.max_speech_duration_s {
        Some(s) if s.is_finite() && s > 0.0 => {
            f64::from(SAMPLE_RATE as u32) * f64::from(s) -
                window as f64 -
                2.0 * speech_pad as f64
        },
        _ => f64::INFINITY,
    };

    // `use_max_poss_sil_at_max_speech` is the v5 default (true); under it
    // `triggered` holds exactly while a segment is open (`cur_start` valid).
    let mut speeches: Vec<[i64; 2]> = Vec::new();
    let mut triggered = false;
    let mut cur_start: i64 = 0;
    let mut temp_end: i64 = 0;
    let mut prev_end: i64 = 0;
    let mut next_start: i64 = 0;
    let mut possible_ends: Vec<(i64, i64)> = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
        let cur = window * i as i64;

        // Speech returned after a provisional end: record a candidate silence
        // and clear the pending end.
        if prob >= threshold && temp_end != 0 {
            let silence = cur - temp_end;
            if silence > min_silence_at_max {
                possible_ends.push((temp_end, silence));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur;
            }
        }

        // Start of speech.
        if prob >= threshold && !triggered {
            triggered = true;
            cur_start = cur;
            continue;
        }

        // Maximum speech length reached: cut.
        if triggered && (cur - cur_start) as f64 > max_speech {
            if let Some(&(pe, dur)) =
                possible_ends.iter().max_by_key(|&&(_, d)| d)
            {
                // Cut at the longest internal silence and possibly reopen.
                speeches.push([cur_start, pe]);
                next_start = pe + dur;
                if next_start < pe + cur {
                    cur_start = next_start;
                } else {
                    triggered = false;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else if prev_end != 0 {
                // Legacy cut at the last valid silence.
                speeches.push([cur_start, prev_end]);
                if next_start < prev_end {
                    triggered = false;
                } else {
                    cur_start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else {
                // No candidate silence: hard cut at the current sample.
                speeches.push([cur_start, cur]);
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        // Silence while in speech.
        if prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = cur;
            }
            if cur - temp_end < min_silence {
                // Too short to be a real gap: bridge it.
                continue;
            }
            // Long enough: close the segment at the silence start, keeping it
            // only if it is long enough.
            if temp_end - cur_start > min_speech {
                speeches.push([cur_start, temp_end]);
            }
            triggered = false;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            possible_ends.clear();
            continue;
        }
    }

    // A segment still open at the end.
    if triggered && (audio_len - cur_start) > min_speech {
        speeches.push([cur_start, audio_len]);
    }

    pad_and_merge(&mut speeches, speech_pad, audio_len);

    let sr = SAMPLE_RATE as f64;
    let audio_seconds = audio_len as f64 / sr;
    speeches
        .into_iter()
        .map(|[start, end]| SpeechSegment {
            start: (start as f64 / sr).max(0.0),
            end: (end as f64 / sr).min(audio_seconds),
        })
        .collect()
}

/// Pads each segment by `speech_pad` outward, splitting the gap evenly between
/// close neighbours so padded boundaries meet but never overlap. Clamps to
/// `[0, audio_len]`.
fn pad_and_merge(speeches: &mut [[i64; 2]], speech_pad: i64, audio_len: i64) {
    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[0][0] = (speeches[0][0] - speech_pad).max(0);
        }
        if i != n - 1 {
            let silence = speeches[i + 1][0] - speeches[i][1];
            if silence < 2 * speech_pad {
                speeches[i][1] += silence / 2;
                speeches[i + 1][0] = (speeches[i + 1][0] - silence / 2).max(0);
            } else {
                speeches[i][1] = (speeches[i][1] + speech_pad).min(audio_len);
                speeches[i + 1][0] = (speeches[i + 1][0] - speech_pad).max(0);
            }
        } else {
            speeches[i][1] = (speeches[i][1] + speech_pad).min(audio_len);
        }
    }
}
