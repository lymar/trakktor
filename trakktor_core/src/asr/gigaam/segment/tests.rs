use super::{Interval, MAX_DURATION_S, chunk_boundaries};

fn iv(start: f64, end: f64) -> Interval { Interval { start, end } }

/// Every chunk must respect the hard duration bound and tile the covered span
/// without overlap.
fn assert_valid(chunks: &[(f64, f64)]) {
    for &(s, e) in chunks {
        assert!(e > s, "empty chunk {s}..{e}");
        assert!(
            e - s <= MAX_DURATION_S + 1e-9,
            "chunk {s}..{e} exceeds MAX_DURATION_S"
        );
    }
    for pair in chunks.windows(2) {
        assert!(pair[1].0 >= pair[0].1 - 1e-9, "chunks overlap: {pair:?}");
    }
}

#[test]
fn empty_speech_yields_no_chunks() {
    assert!(chunk_boundaries(&[], 30.0).is_empty());
}

#[test]
fn short_audio_is_one_chunk() {
    let speech = [iv(0.0, 3.0), iv(5.0, 8.0), iv(10.0, 14.0)];
    let chunks = chunk_boundaries(&speech, 15.0);
    assert_eq!(chunks, vec![(0.0, 14.0)]);
}

#[test]
fn clamps_last_interval_to_content_end() {
    let speech = [iv(0.0, 5.0), iv(6.0, 100.0)];
    let chunks = chunk_boundaries(&speech, 12.0);
    assert_eq!(chunks.last().unwrap().1, 12.0);
}

#[test]
fn presplits_interval_over_max_duration() {
    // A single 70 s speech span must split into parts each <= MAX_DURATION_S.
    let speech = [iv(0.0, 70.0)];
    let chunks = chunk_boundaries(&speech, 70.0);
    assert!(
        chunks.len() >= 3,
        "expected >=3 parts, got {}",
        chunks.len()
    );
    assert_valid(&chunks);
    // Parts tile the whole span.
    assert_eq!(chunks.first().unwrap().0, 0.0);
    assert_eq!(chunks.last().unwrap().1, 70.0);
}

#[test]
fn cut_lands_on_the_long_pause() {
    // Six ~6 s speech intervals (~36 s total) must split into two chunks. One
    // pause (at t≈18) is long (2 s); the rest are short (0.1 s). The optimal
    // cut is the long pause, giving an 18 s and a 16 s chunk.
    let speech = [
        iv(0.0, 6.0),
        iv(6.1, 12.0),
        iv(12.1, 18.0),
        iv(20.0, 26.0), // 2 s pause before this one
        iv(26.1, 32.0),
        iv(32.1, 36.0),
    ];
    let chunks = chunk_boundaries(&speech, 36.0);
    assert_valid(&chunks);
    assert_eq!(chunks, vec![(0.0, 18.0), (20.0, 36.0)]);
}

#[test]
fn does_not_over_fragment_on_short_pauses() {
    // Uniform short pauses: the DP should not chop into many tiny chunks just
    // to collect pause bonuses — chunks stay near the target, not near the
    // pause spacing.
    let mut speech = Vec::new();
    let mut t = 0.0;
    for _ in 0..20 {
        speech.push(iv(t, t + 2.0));
        t += 2.1; // 0.1 s pauses
    }
    let content_end = t;
    let chunks = chunk_boundaries(&speech, content_end);
    assert_valid(&chunks);
    // ~42 s total → 2-3 chunks near the 18 s target, not ~20 tiny ones.
    assert!(
        chunks.len() <= 3,
        "over-fragmented into {} chunks",
        chunks.len()
    );
}

#[test]
fn prefers_longer_pause_between_two_candidates() {
    // Two plausible cut points around the target; the longer pause wins.
    // Intervals: [0,16](0.3 pause)[16.3,20](1.5 pause)[21.5,38].
    // Cutting at t=16 → 16 s + ~21.7 s; at t=20 → 20 s + ~16.5 s. The 1.5 s
    // pause at t=20 should be chosen over the 0.3 s pause at t=16.
    let speech = [iv(0.0, 16.0), iv(16.3, 20.0), iv(21.5, 38.0)];
    let chunks = chunk_boundaries(&speech, 38.0);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 2);
    // First chunk ends at 20.0 (the long-pause cut), not 16.0.
    assert_eq!(chunks[0].1, 20.0);
    assert_eq!(chunks[1].0, 21.5);
}
