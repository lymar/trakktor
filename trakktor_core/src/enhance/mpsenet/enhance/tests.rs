//! The driver's host-side arithmetic: where the windows fall, how they are
//! faded together, and the one gain the reference applies.

use super::*;

const RATE: usize = SAMPLE_RATE as usize;

#[test]
fn a_short_recording_is_one_window() {
    for len in [1, RATE, WINDOW_SECONDS * RATE] {
        assert_eq!(windows(len), vec![(0, len)], "at {len} samples");
    }
}

#[test]
fn windows_share_exactly_the_overlap() {
    let spans = windows(30 * RATE);
    assert!(spans.len() > 1);
    for pair in spans.windows(2) {
        let (previous, next) = (pair[0], pair[1]);
        assert_eq!(
            previous.1 - next.0,
            OVERLAP_SECONDS * RATE,
            "{previous:?} then {next:?}"
        );
    }
    assert_eq!(spans[0].0, 0);
    assert_eq!(spans.last().expect("a window").1, 30 * RATE);
}

#[test]
fn a_tail_is_never_shorter_than_the_overlap_it_shares() {
    // The claim `windows` makes, over every length that could produce an
    // awkward tail: a hop's worth either side of each window boundary, out to
    // five windows. A sliver would be faded in over more of itself than it
    // has.
    let span = WINDOW_SECONDS * RATE;
    let hop = span - OVERLAP_SECONDS * RATE;
    let lengths = (1..=span + 2)
        .chain((1..=5).flat_map(|k| {
            let base = k * span;
            (base.saturating_sub(3)..base + hop + 3).collect::<Vec<_>>()
        }))
        .collect::<Vec<_>>();
    for len in lengths {
        let spans = windows(len);
        assert_eq!(spans.last().expect("a window").1, len, "at {len}");
        if spans.len() > 1 {
            for pair in spans.windows(2) {
                assert_eq!(
                    pair[0].1 - pair[1].0,
                    OVERLAP_SECONDS * RATE,
                    "at {len}: {:?} then {:?}",
                    pair[0],
                    pair[1]
                );
            }
            let (start, end) = *spans.last().expect("a window");
            assert!(
                end - start > OVERLAP_SECONDS * RATE,
                "at {len}: the tail {start}..{end} is no longer than the \
                 overlap"
            );
        }
    }
}

#[test]
fn the_fade_weights_of_two_windows_sum_to_one() {
    let overlap = OVERLAP_SECONDS * RATE;
    let len = WINDOW_SECONDS * RATE;
    for offset in 0..overlap {
        let leaving = fade(len - overlap + offset, len, 0, overlap);
        let arriving = fade(offset, len, overlap, 0);
        assert!(
            (leaving + arriving - 1.0).abs() < 1e-6,
            "at {offset}: {leaving} + {arriving}"
        );
    }
}

#[test]
fn the_fade_is_one_away_from_the_seams() {
    let overlap = OVERLAP_SECONDS * RATE;
    let len = WINDOW_SECONDS * RATE;
    assert_eq!(fade(overlap, len, overlap, overlap), 1.0);
    assert_eq!(fade(len / 2, len, overlap, overlap), 1.0);
    assert_eq!(fade(0, len, 0, overlap), 1.0);
}

#[test]
fn the_gain_brings_a_recording_to_unit_mean_square() {
    let wave: Vec<f32> = (0..8_000)
        .map(|index| 0.03 * (index as f32 / 40.0).sin())
        .collect();
    let gain = level_gain(&wave);
    let energy: f64 = wave
        .iter()
        .map(|&v| f64::from(v * gain) * f64::from(v * gain))
        .sum();
    let mean_square = energy / wave.len() as f64;
    assert!((mean_square - 1.0).abs() < 1e-5, "{mean_square}");
}

#[test]
fn silence_keeps_its_own_level() {
    assert_eq!(level_gain(&[0.0; 128]), 1.0);
}
