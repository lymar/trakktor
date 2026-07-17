//! Tests for the collapse buffer and its time map.

use super::*;

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-6, "{a} vs {b}");
}

fn seg(start: f64, end: f64) -> SpeechSegment { SpeechSegment { start, end } }

#[test]
fn concatenates_speech_with_silence_between() {
    let audio = vec![0.5f32; 10 * SAMPLE_RATE]; // 10 s
    let c = collapse(&audio, &[seg(1.0, 3.0), seg(5.0, 6.0)]);
    // 2 s + 0.1 s silence + 1 s = 3.1 s.
    assert_eq!(
        c.buffer.len(),
        2 * SAMPLE_RATE + SAMPLE_RATE / 10 + SAMPLE_RATE
    );
    approx(c.duration, 10.0);
}

#[test]
fn maps_processed_time_back_with_slope_one() {
    let audio = vec![0.5f32; 10 * SAMPLE_RATE];
    let c = collapse(&audio, &[seg(1.0, 3.0), seg(5.0, 6.0)]);
    let m = &c.mapping;
    approx(m.map(0.0), 1.0); // before the first region → its start
    approx(m.map(1.0), 2.0); // inside region 0 (orig 1 + 1)
    approx(m.map(2.0), 3.0); // region 0 end
    approx(m.map(2.1), 5.0); // region 1 start (orig 5)
    approx(m.map(2.6), 5.5); // inside region 1
    approx(m.map(3.1), 6.0); // after the last region → its end
    approx(m.map(9.0), 6.0); // well past the end → clamps
}

#[test]
fn gap_time_snaps_to_the_nearer_boundary() {
    let audio = vec![0.5f32; 10 * SAMPLE_RATE];
    let c = collapse(&audio, &[seg(1.0, 3.0), seg(5.0, 6.0)]);
    // The silence spans processed [2.0, 2.1]; midpoint 2.05.
    approx(c.mapping.map(2.02), 3.0); // nearer region 0 end
    approx(c.mapping.map(2.08), 5.0); // nearer region 1 start
}

#[test]
fn empty_speech_is_identity() {
    let c = collapse(&[0.0f32; 1000], &[]);
    assert!(c.buffer.is_empty());
    approx(c.mapping.map(4.2), 4.2);
}

#[test]
fn zero_length_segments_are_skipped() {
    let audio = vec![0.5f32; 10 * SAMPLE_RATE];
    let c = collapse(&audio, &[seg(2.0, 2.0), seg(4.0, 5.0)]);
    // Only the second segment contributes: 1 s, no leading silence.
    assert_eq!(c.buffer.len(), SAMPLE_RATE);
    approx(c.mapping.map(0.5), 4.5);
}
