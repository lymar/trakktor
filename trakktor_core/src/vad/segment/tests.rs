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

/// The pre-incremental batch implementation, kept verbatim as the reference
/// for the property test below: the detector-driven [`speech_timestamps`]
/// must reproduce it exactly, early emission and all.
mod reference {
    use super::{
        MIN_SILENCE_AT_MAX_MS, SAMPLE_RATE, SpeechSegment, VadOptions,
        WINDOW_SIZE, ms_to_samples,
    };

    pub fn speech_timestamps(
        probs: &[f32],
        n_samples: usize,
        options: &VadOptions,
    ) -> Vec<SpeechSegment> {
        let window = WINDOW_SIZE as i64;
        let threshold = options.threshold;
        let neg_threshold = (threshold - 0.15).max(0.01);
        let min_speech =
            ms_to_samples(i64::from(options.min_speech_duration_ms));
        let speech_pad = ms_to_samples(i64::from(options.speech_pad_ms));
        let min_silence =
            ms_to_samples(i64::from(options.min_silence_duration_ms));
        let min_silence_at_max = ms_to_samples(MIN_SILENCE_AT_MAX_MS);
        let audio_len = n_samples as i64;
        let max_speech: f64 = match options.max_speech_duration_s {
            Some(s) if s.is_finite() && s > 0.0 => {
                f64::from(SAMPLE_RATE as u32) * f64::from(s) -
                    window as f64 -
                    2.0 * speech_pad as f64
            },
            _ => f64::INFINITY,
        };

        let mut speeches: Vec<[i64; 2]> = Vec::new();
        let mut triggered = false;
        let mut cur_start: i64 = 0;
        let mut temp_end: i64 = 0;
        let mut prev_end: i64 = 0;
        let mut next_start: i64 = 0;
        let mut possible_ends: Vec<(i64, i64)> = Vec::new();

        for (i, &prob) in probs.iter().enumerate() {
            let cur = window * i as i64;

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

            if prob >= threshold && !triggered {
                triggered = true;
                cur_start = cur;
                continue;
            }

            if triggered && (cur - cur_start) as f64 > max_speech {
                if let Some(&(pe, dur)) =
                    possible_ends.iter().max_by_key(|&&(_, d)| d)
                {
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
                    speeches.push([cur_start, cur]);
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                    possible_ends.clear();
                    continue;
                }
            }

            if prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = cur;
                }
                if cur - temp_end < min_silence {
                    continue;
                }
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

    fn pad_and_merge(
        speeches: &mut [[i64; 2]],
        speech_pad: i64,
        audio_len: i64,
    ) {
        let n = speeches.len();
        for i in 0..n {
            if i == 0 {
                speeches[0][0] = (speeches[0][0] - speech_pad).max(0);
            }
            if i != n - 1 {
                let silence = speeches[i + 1][0] - speeches[i][1];
                if silence < 2 * speech_pad {
                    speeches[i][1] += silence / 2;
                    speeches[i + 1][0] =
                        (speeches[i + 1][0] - silence / 2).max(0);
                } else {
                    speeches[i][1] =
                        (speeches[i][1] + speech_pad).min(audio_len);
                    speeches[i + 1][0] =
                        (speeches[i + 1][0] - speech_pad).max(0);
                }
            } else {
                speeches[i][1] = (speeches[i][1] + speech_pad).min(audio_len);
            }
        }
    }
}

/// A tiny deterministic generator (xorshift) for the property test.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn detector_matches_the_reference_on_random_sequences() {
    let option_sets = [
        VadOptions::default(),
        VadOptions {
            max_speech_duration_s: Some(1.0),
            ..VadOptions::default()
        },
        VadOptions {
            threshold: 0.3,
            min_silence_duration_ms: 20,
            speech_pad_ms: 30,
            ..VadOptions::default()
        },
        VadOptions {
            speech_pad_ms: 0,
            min_speech_duration_ms: 0,
            ..VadOptions::default()
        },
        VadOptions {
            min_silence_duration_ms: 0,
            max_speech_duration_s: Some(0.5),
            ..VadOptions::default()
        },
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for options in &option_sets {
        for round in 0..40 {
            // Runs of speech-ish and silence-ish probabilities with varying
            // lengths — exercises bridging, closing, max-speech cuts, pads.
            let mut p = Vec::new();
            let runs = 3 + (xorshift(&mut state) % 12) as usize;
            for _ in 0..runs {
                let len = 1 + (xorshift(&mut state) % 90) as usize;
                let value = match xorshift(&mut state) % 4 {
                    0 => 0.0,
                    1 => 0.9,
                    2 => 0.45, // hysteresis dead zone
                    _ => 0.2,
                };
                p.extend(std::iter::repeat_n(value, len));
            }
            // Sometimes end mid-window to exercise the tail clamp.
            let n = p.len() * WINDOW_SIZE -
                (xorshift(&mut state) % WINDOW_SIZE as u64) as usize;
            let expected = reference::speech_timestamps(&p, n, options);
            let got = speech_timestamps(&p, n, options);
            assert_eq!(
                got, expected,
                "round {round} options {options:?} diverged"
            );
        }
    }
}

#[test]
fn early_emission_frees_the_pending_segment() {
    // One speech burst followed by long silence: the segment must be emitted
    // long before finish(), and earliest_pending_second() must clear.
    let options = VadOptions::default();
    let mut detector = SpeechDetector::new(&options);
    let mut out = Vec::new();
    for _ in 0..10 {
        detector.push(0.9, &mut out);
    }
    // 2*pad = 960 samples < 2 windows; min_silence closes after 4 windows.
    for _ in 0..10 {
        detector.push(0.0, &mut out);
    }
    assert_eq!(
        out.len(),
        1,
        "segment emitted during silence, not at finish"
    );
    assert!(detector.earliest_pending_second().is_none());
    let mut tail = Vec::new();
    detector.finish(20 * WINDOW_SIZE, &mut tail);
    assert!(tail.is_empty());
    // And the emitted value matches the batch run of the same probabilities.
    let p = probs(&[(10, 0.9), (10, 0.0)]);
    let batch = speech_timestamps(&p, 20 * WINDOW_SIZE, &options);
    assert_eq!(out, batch);
}
