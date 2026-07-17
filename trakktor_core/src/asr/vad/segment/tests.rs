//! Tests for the speech-timestamp state machine on synthetic probability
//! vectors. One window is 512 samples = 0.032 s at 16 kHz.

use super::*;

/// Builds a probability vector from `(count, value)` runs.
fn probs(runs: &[(usize, f32)]) -> Vec<f32> {
    let mut out = Vec::new();
    for &(count, value) in runs {
        out.extend(std::iter::repeat_n(value, count));
    }
    out
}

fn n_samples(probs: &[f32]) -> usize { probs.len() * WINDOW_SIZE }

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-6, "{a} vs {b}");
}

#[test]
fn all_silence_has_no_speech() {
    let p = probs(&[(100, 0.0)]);
    assert!(
        speech_timestamps(&p, n_samples(&p), &VadOptions::default()).is_empty()
    );
}

#[test]
fn a_single_speech_block_is_one_padded_segment() {
    // 10 windows silence, 30 speech, 10 silence.
    let p = probs(&[(10, 0.0), (30, 0.9), (10, 0.0)]);
    let segs = speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    assert_eq!(segs.len(), 1);
    // Speech runs [window 10, window 40); the gap closes 4 windows into the
    // silence (2048 ≥ 1600 samples), then ±480-sample padding applies.
    // start = (5120 - 480) / 16000, end = (20480 + 480) / 16000.
    approx(segs[0].start, 4640.0 / 16_000.0);
    approx(segs[0].end, 20_960.0 / 16_000.0);
}

#[test]
fn a_short_silence_does_not_split_speech() {
    // A 2-window dip (1024 samples < 1600) is bridged: one segment.
    let p = probs(&[(10, 0.0), (20, 0.9), (2, 0.0), (20, 0.9), (10, 0.0)]);
    let segs = speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    assert_eq!(segs.len(), 1);
}

#[test]
fn a_long_silence_splits_speech() {
    // A 5-window gap (2560 ≥ 1600) closes the first segment; two remain.
    let p = probs(&[(10, 0.0), (20, 0.9), (5, 0.0), (20, 0.9), (10, 0.0)]);
    let segs = speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    assert_eq!(segs.len(), 2);
}

#[test]
fn a_too_short_burst_is_dropped() {
    // 3 windows of speech (1536 < 4000 min-speech samples) is discarded.
    let p = probs(&[(5, 0.0), (3, 0.9), (10, 0.0)]);
    assert!(
        speech_timestamps(&p, n_samples(&p), &VadOptions::default()).is_empty()
    );
}

#[test]
fn hysteresis_keeps_speech_through_the_dead_zone() {
    // Values in [neg_threshold, threshold) neither open nor close a segment;
    // the surrounding 0.9 run stays one segment.
    let p = probs(&[(10, 0.0), (10, 0.9), (5, 0.4), (10, 0.9), (10, 0.0)]);
    let segs = speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    assert_eq!(segs.len(), 1);
}

#[test]
fn a_finite_max_speech_splits_long_speech() {
    let p = probs(&[(200, 0.9)]);
    let unlimited =
        speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    let limited = speech_timestamps(
        &p,
        n_samples(&p),
        &VadOptions {
            max_speech_duration_s: Some(1.0),
            ..VadOptions::default()
        },
    );
    assert_eq!(unlimited.len(), 1);
    assert!(limited.len() > 1);
}

#[test]
fn segments_stay_within_the_audio() {
    let p = probs(&[(30, 0.9)]);
    let total = n_samples(&p) as f64 / 16_000.0;
    let segs = speech_timestamps(&p, n_samples(&p), &VadOptions::default());
    for s in &segs {
        assert!(s.start >= 0.0 && s.end <= total && s.start < s.end);
    }
}
