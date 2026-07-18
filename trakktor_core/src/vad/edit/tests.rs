use super::{EditOptions, Keep, plan};
use crate::vad::segment::SpeechSegment;

fn seg(start: f64, end: f64) -> SpeechSegment { SpeechSegment { start, end } }

/// Options with every shaping knob off, so a test enables just the one it
/// exercises.
fn opts(keep: Keep) -> EditOptions {
    EditOptions {
        keep,
        margin: 0.0,
        merge_gap: 0.0,
        max_silence: None,
        min_duration: 0.0,
    }
}

#[test]
fn keep_speech_returns_the_spans() {
    let speech = [seg(1.0, 2.0), seg(5.0, 6.0)];
    assert_eq!(
        plan(&speech, (0.0, 10.0), &opts(Keep::Speech)),
        vec![(1.0, 2.0), (5.0, 6.0)]
    );
}

#[test]
fn keep_non_speech_returns_the_complement() {
    let speech = [seg(1.0, 2.0), seg(5.0, 6.0)];
    assert_eq!(
        plan(&speech, (0.0, 10.0), &opts(Keep::NonSpeech)),
        vec![(0.0, 1.0), (2.0, 5.0), (6.0, 10.0)]
    );
}

#[test]
fn complement_of_no_speech_is_the_whole_file() {
    assert_eq!(
        plan(&[], (0.0, 10.0), &opts(Keep::NonSpeech)),
        vec![(0.0, 10.0)]
    );
}

#[test]
fn margin_widens_and_clamps_to_the_audio() {
    let mut options = opts(Keep::Speech);
    options.margin = 0.5;
    assert_eq!(
        plan(&[seg(1.0, 2.0)], (0.0, 10.0), &options),
        vec![(0.5, 2.5)]
    );
    // A margin past the start clamps to 0.
    assert_eq!(
        plan(&[seg(0.25, 2.0)], (0.0, 10.0), &options),
        vec![(0.0, 2.5)]
    );
}

#[test]
fn merge_gap_coalesces_close_ranges() {
    let mut options = opts(Keep::Speech);
    options.merge_gap = 0.5;
    let speech = [seg(1.0, 2.0), seg(2.25, 3.0)];
    assert_eq!(plan(&speech, (0.0, 10.0), &options), vec![(1.0, 3.0)]);
}

#[test]
fn collapse_keeps_a_short_pause_whole() {
    let mut options = opts(Keep::Speech);
    options.max_silence = Some(0.5);
    // A 0.25 s pause is below the budget, so it is kept and the ranges join.
    let speech = [seg(1.0, 2.0), seg(2.25, 3.0)];
    assert_eq!(plan(&speech, (0.0, 10.0), &options), vec![(1.0, 3.0)]);
}

#[test]
fn collapse_shortens_a_long_pause_to_the_budget() {
    let mut options = opts(Keep::Speech);
    options.max_silence = Some(0.5);
    // Both pauses (1.0 s and 4.0 s) exceed 0.5 s, so each keeps 0.5 s trailing.
    let speech = [seg(1.0, 2.0), seg(3.0, 4.0), seg(8.0, 9.0)];
    assert_eq!(
        plan(&speech, (0.0, 10.0), &options),
        vec![(1.0, 2.5), (3.0, 4.5), (8.0, 9.0)]
    );
}

#[test]
fn inversion_is_bounded_by_the_window() {
    // Speech at 3..5 inside a 2..8 window: non-speech is 2..3 and 5..8, never
    // the whole file.
    let speech = [seg(3.0, 5.0)];
    assert_eq!(
        plan(&speech, (2.0, 8.0), &opts(Keep::NonSpeech)),
        vec![(2.0, 3.0), (5.0, 8.0)]
    );
}

#[test]
fn kept_speech_is_clamped_to_the_window() {
    // Speech wider than the window is clamped to the window bounds.
    let speech = [seg(1.0, 9.0)];
    assert_eq!(
        plan(&speech, (2.0, 8.0), &opts(Keep::Speech)),
        vec![(2.0, 8.0)]
    );
}

#[test]
fn min_duration_drops_short_ranges() {
    let mut options = opts(Keep::Speech);
    options.min_duration = 0.2;
    let speech = [seg(1.0, 1.1), seg(5.0, 6.0)];
    assert_eq!(plan(&speech, (0.0, 10.0), &options), vec![(5.0, 6.0)]);
}
