//! Tests for the rules that sit between the networks.

use super::{durations, expansion, shape_pitch};
use crate::tts::silero::config::{Config, HEAD_DURATION};

/// The duration head's output for a frame count: it predicts `log(1 + n)`.
fn logs(frames: &[f32]) -> Vec<f32> {
    frames.iter().map(|value| (value + 1.0).ln()).collect()
}

#[test]
fn a_predicted_length_comes_back_as_its_frame_count() {
    // Five symbols, so the three tail rules land on the last three and leave
    // the middle alone.
    let log = logs(&[3.0, 9.0, 4.0, 6.0, 2.0]);
    let rate = vec![1.0; 5];
    assert_eq!(durations(&log, &rate)[1], 9);
}

#[test]
fn the_opening_symbol_is_held_short() {
    let log = logs(&[40.0, 9.0, 4.0, 6.0, 2.0]);
    let frames = durations(&log, &vec![1.0; 5]);
    assert_eq!(frames[0], HEAD_DURATION);
}

#[test]
fn the_closing_symbols_follow_the_references_own_rules() {
    let log = logs(&[3.0, 9.0, 40.0, 40.0, 40.0]);
    let frames = durations(&log, &vec![1.0; 5]);
    // Third from the end is clamped, second from the end is set outright, and
    // the end-of-speech symbol is clamped shorter still.
    assert_eq!(&frames[2..], &[13, 13, 7]);
    // And the one that is *set* is set even when the head predicted less.
    let short = durations(&logs(&[3.0, 9.0, 4.0, 1.0, 2.0]), &vec![1.0; 5]);
    assert_eq!(short[3], 13);
}

#[test]
fn a_faster_rate_shortens_the_middle() {
    let log = logs(&[3.0, 20.0, 8.0, 6.0, 2.0]);
    let slow = durations(&log, &vec![1.0; 5]);
    let fast = durations(&log, &vec![2.0; 5]);
    assert!(fast[1] < slow[1], "{fast:?} vs {slow:?}");
    assert_eq!(fast[1], 10);
}

#[test]
fn rounding_goes_to_even_the_way_torch_does() {
    // 2.5 and 3.5 both round to even, so they land on the same side as torch.
    let log = logs(&[2.5, 3.5, 4.0, 6.0, 2.0]);
    let frames = durations(&log, &vec![1.0; 5]);
    assert_eq!(&frames[..2], &[2, 4]);
}

#[test]
fn a_pitch_at_the_floor_is_taken_as_none() {
    let mut pitch = vec![0.0005, 0.5, -0.4];
    shape_pitch(&mut pitch, &[1.0, 1.0, 1.0], 4.0);
    assert_eq!(pitch[0], 0.0);
    // Unit coefficients leave the rest exactly as predicted.
    assert!((pitch[1] - 0.5).abs() < f32::EPSILON);
    assert!((pitch[2] + 0.4).abs() < f32::EPSILON);
}

#[test]
fn raising_the_pitch_scales_and_shifts_by_the_speakers_range() {
    let mut pitch = vec![0.5, 0.0];
    shape_pitch(&mut pitch, &[1.2, 1.2], 4.0);
    // 0.5 · 1.2 + (1.2 − 1) · 4.0
    assert!((pitch[0] - 1.4).abs() < 1e-6, "{pitch:?}");
    // A symbol with no pitch keeps none rather than picking up the shift.
    assert_eq!(pitch[1], 0.0);
}

#[test]
fn the_expansion_repeats_each_symbol_for_its_frames() {
    assert_eq!(expansion(&[2, 0, 3]), vec![0, 0, 2, 2, 2]);
}

#[test]
fn an_utterance_past_the_position_table_is_refused() {
    let config = Config {
        symbols: 85,
        speaker_slots: 60,
        dim: 128,
        ff_inner: 512,
        encoder_layers: 4,
        predictor_dim: 64,
        predictor_ff_inner: 64,
        predictor_layers: 4,
        utterance_types: 0,
        mel_channels: 192,
        positions: 5000,
        vocoder_dim: 512,
        vocoder_ff_inner: 1536,
        vocoder_layers: 8,
        n_fft: 2400,
    };
    super::check_frames(5000, &config).expect("the ceiling itself fits");
    let error =
        super::check_frames(5001, &config).expect_err("one past it does not");
    assert!(error.to_string().contains("62 s"), "{error}");
}
