//! Turning detected speech into the ranges to keep when editing audio.
//!
//! Pure interval arithmetic on top of [`detect_speech`](super::detect_speech):
//! it takes the speech segments (seconds) and the editing options and returns
//! the time ranges to copy from the original audio. Keeping speech and dropping
//! it (inversion) are the two directions; on top of that it can widen each
//! range by a margin, merge ranges separated by a tiny gap, collapse long
//! pauses to a fixed length instead of removing them, and drop ranges that end
//! up too short. It knows nothing about samples or encoding — the audio editor
//! ([`crate::audio::edit`]) turns these ranges into PCM.

#[cfg(test)]
mod tests;

use super::segment::SpeechSegment;

/// A time range to copy, in seconds on the original timeline: `(start, end)`.
pub type Range = (f64, f64);

/// Which part of the audio to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// Keep speech, drop non-speech — the silence-removal direction.
    Speech,
    /// Keep non-speech, drop speech — the inversion.
    NonSpeech,
}

/// How to shape the kept ranges around the detected speech. All durations are
/// seconds.
#[derive(Clone, Copy, Debug)]
pub struct EditOptions {
    /// Which side of the speech/non-speech split to keep.
    pub keep: Keep,
    /// Extra audio kept on each side of a range (added to the detector's own
    /// padding, which is already baked into the speech segments).
    pub margin: f64,
    /// Ranges whose gap is smaller than this are merged into one.
    pub merge_gap: f64,
    /// With [`Keep::Speech`]: shorten any pause between kept speech to at most
    /// this long instead of removing it. `None` removes pauses entirely.
    pub max_silence: Option<f64>,
    /// Drop any kept range shorter than this.
    pub min_duration: f64,
}

/// Computes the ranges to keep from the detected `speech` (seconds) over the
/// working `window` (`(start, end)` seconds — the whole file, or a
/// `--start`/`--end` sub-range). Ranges are sorted, non-overlapping, and within
/// the window; the complement for inversion is taken over the window too, so a
/// windowed inversion stays inside it.
#[must_use]
pub fn plan(
    speech: &[SpeechSegment],
    window: (f64, f64),
    opts: &EditOptions,
) -> Vec<Range> {
    // Base ranges: the speech spans, or their complement for inversion.
    let base: Vec<Range> = match opts.keep {
        Keep::Speech => speech.iter().map(|s| (s.start, s.end)).collect(),
        Keep::NonSpeech => complement(speech, window),
    };
    // Widen by the margin, then coalesce anything that now overlaps or sits
    // within `merge_gap`.
    let widened = expand(&base, opts.margin, window);
    let mut ranges = merge(&widened, opts.merge_gap);
    // Collapse long pauses (speech direction only), then coalesce the pauses
    // that ended up touching their neighbours back into single ranges.
    if opts.keep == Keep::Speech &&
        let Some(max_silence) = opts.max_silence
    {
        collapse(&mut ranges, max_silence);
        ranges = merge(&ranges, 0.0);
    }
    ranges.retain(|&(start, end)| end - start >= opts.min_duration);
    ranges
}

/// The gaps between speech segments over the window — the non-speech. Segments
/// are clamped into the window, so speech that spills past its edges does not
/// leak into the complement.
fn complement(speech: &[SpeechSegment], window: (f64, f64)) -> Vec<Range> {
    let (start, end) = window;
    let mut out = Vec::new();
    let mut cursor = start;
    for segment in speech {
        let seg_start = segment.start.clamp(start, end);
        let seg_end = segment.end.clamp(start, end);
        if seg_start > cursor {
            out.push((cursor, seg_start));
        }
        cursor = cursor.max(seg_end);
    }
    if cursor < end {
        out.push((cursor, end));
    }
    out
}

/// Widens each range by `margin` on both sides, clamped to the window.
fn expand(ranges: &[Range], margin: f64, window: (f64, f64)) -> Vec<Range> {
    let (start, end) = window;
    ranges
        .iter()
        .map(|&(a, b)| ((a - margin).max(start), (b + margin).min(end)))
        .collect()
}

/// Coalesces sorted ranges whose gap is below `gap` (overlaps count as a
/// negative gap, so they merge too).
fn merge(ranges: &[Range], gap: f64) -> Vec<Range> {
    let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        match out.last_mut() {
            Some(last) if start - last.1 <= gap => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

/// Extends each range's end into the following pause by up to `max_silence`,
/// so a long pause is kept but shortened rather than dropped.
fn collapse(ranges: &mut [Range], max_silence: f64) {
    for i in 0..ranges.len() {
        let Some(&(next_start, _)) = ranges.get(i + 1) else {
            break;
        };
        let pause = next_start - ranges[i].1;
        if pause > 0.0 {
            ranges[i].1 += pause.min(max_silence);
        }
    }
}
