use super::{
    BOUNDARY_CONTEXT_S, COMMIT_HORIZON_S, COMMIT_MARGIN_S, Chunk, ChunkPlanner,
    Interval, MAX_DURATION_S, chunk_boundaries,
};

fn iv(start: f64, end: f64) -> Interval { Interval { start, end } }

/// The speech spans, for the assertions that only care about the cuts.
fn spans(chunks: &[Chunk]) -> Vec<(f64, f64)> {
    chunks.iter().map(|c| (c.start, c.end)).collect()
}

/// Every chunk must respect the hard duration bound and tile the covered span
/// without overlap — both as speech spans and as audio windows, which must
/// enclose their speech and stay disjoint from their neighbours'.
fn assert_valid(chunks: &[Chunk]) {
    for c in chunks {
        assert!(c.end > c.start, "empty chunk {c:?}");
        assert!(
            c.end - c.start <= MAX_DURATION_S + 1e-9,
            "chunk {c:?} exceeds MAX_DURATION_S"
        );
        assert!(
            c.window_len() <= MAX_DURATION_S + 1e-9,
            "window of {c:?} exceeds MAX_DURATION_S"
        );
        assert!(
            c.window_start <= c.start + 1e-9 && c.window_end >= c.end - 1e-9,
            "window of {c:?} does not enclose its speech"
        );
    }
    for pair in chunks.windows(2) {
        assert!(
            pair[1].start >= pair[0].end - 1e-9,
            "chunks overlap: {pair:?}"
        );
        assert!(
            pair[1].window_start >= pair[0].window_end - 1e-9,
            "windows overlap: {pair:?}"
        );
    }
}

#[test]
fn empty_speech_yields_no_chunks() {
    assert!(chunk_boundaries(&[], 0.0, 30.0).is_empty());
}

#[test]
fn short_audio_is_one_chunk() {
    let speech = [iv(0.0, 3.0), iv(5.0, 8.0), iv(10.0, 14.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 15.0);
    assert_eq!(spans(&chunks), vec![(0.0, 14.0)]);
}

#[test]
fn clamps_last_interval_to_content_end() {
    let speech = [iv(0.0, 5.0), iv(6.0, 100.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 12.0);
    assert_eq!(chunks.last().unwrap().end, 12.0);
    assert_eq!(chunks.last().unwrap().window_end, 12.0);
}

#[test]
fn presplits_interval_over_max_duration() {
    // A single 70 s speech span must split into parts each <= MAX_DURATION_S.
    let speech = [iv(0.0, 70.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 70.0);
    assert!(
        chunks.len() >= 3,
        "expected >=3 parts, got {}",
        chunks.len()
    );
    assert_valid(&chunks);
    // Parts tile the whole span.
    assert_eq!(chunks.first().unwrap().start, 0.0);
    assert_eq!(chunks.last().unwrap().end, 70.0);
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
    let chunks = chunk_boundaries(&speech, 0.0, 36.0);
    assert_valid(&chunks);
    assert_eq!(spans(&chunks), vec![(0.0, 18.0), (20.0, 36.0)]);
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
    let chunks = chunk_boundaries(&speech, 0.0, t);
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
    let chunks = chunk_boundaries(&speech, 0.0, 38.0);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 2);
    // First chunk ends at 20.0 (the long-pause cut), not 16.0.
    assert_eq!(chunks[0].end, 20.0);
    assert_eq!(chunks[1].start, 21.5);
}

#[test]
fn a_short_boundary_pause_is_split_evenly_and_fully() {
    // The 1.5 s pause at the cut is below 2 * BOUNDARY_CONTEXT_S, so both
    // windows take half of it and meet in its middle: no audio is dropped.
    let speech = [iv(0.0, 16.0), iv(16.3, 20.0), iv(21.5, 38.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 38.0);
    assert_valid(&chunks);
    let meet = (chunks[0].end + chunks[1].start) / 2.0;
    assert!((chunks[0].window_end - meet).abs() < 1e-9);
    assert!((chunks[1].window_start - meet).abs() < 1e-9);
}

#[test]
fn a_long_boundary_pause_is_capped_at_the_context() {
    // The 2 s pause exceeds 2 * BOUNDARY_CONTEXT_S (2 * 1 s), so each side
    // takes only the context and the middle of the pause is left out.
    let speech = [
        iv(0.0, 6.0),
        iv(6.1, 12.0),
        iv(12.1, 18.0),
        iv(20.0, 26.0),
        iv(26.1, 32.0),
        iv(32.1, 36.0),
    ];
    let chunks = chunk_boundaries(&speech, 0.0, 36.0);
    assert_valid(&chunks);
    assert!((chunks[0].window_end - (18.0 + BOUNDARY_CONTEXT_S)).abs() < 1e-9);
    assert!(
        (chunks[1].window_start - (20.0 - BOUNDARY_CONTEXT_S)).abs() < 1e-9
    );
}

#[test]
fn presplit_boundaries_get_no_context() {
    // Boundaries introduced inside continuous speech have no pause to share;
    // widening there would transcribe the same audio twice.
    let speech = [iv(0.0, 70.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 70.0);
    assert_valid(&chunks);
    for pair in chunks.windows(2) {
        assert!(
            (pair[0].window_end - pair[1].window_start).abs() < 1e-9,
            "presplit windows must abut exactly: {pair:?}"
        );
        assert!((pair[0].window_end - pair[0].end).abs() < 1e-9);
    }
}

#[test]
fn content_bounds_limit_the_outer_windows() {
    // The outer edges never reach past the audio the caller declared: in a
    // streaming run `content_start` is the previous commit's window end.
    let speech = [iv(10.2, 16.0), iv(17.0, 24.0)];
    let chunks = chunk_boundaries(&speech, 10.0, 24.3);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 1);
    assert!((chunks[0].window_start - 10.0).abs() < 1e-9);
    assert!((chunks[0].window_end - 24.3).abs() < 1e-9);
}

#[test]
fn window_is_trimmed_to_the_hard_bound() {
    // A 29.5 s chunk leaves only 0.5 s of budget: the two sides share it
    // instead of pushing the window past MAX_DURATION_S.
    let speech = [iv(10.0, 39.5)];
    let chunks = chunk_boundaries(&speech, 0.0, 50.0);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 1);
    assert!((chunks[0].window_len() - MAX_DURATION_S).abs() < 1e-9);
    // Symmetric: the trim scales both sides by the same factor.
    let left = chunks[0].start - chunks[0].window_start;
    let right = chunks[0].window_end - chunks[0].end;
    assert!((left - right).abs() < 1e-9);
}

#[test]
fn windows_never_overlap_across_many_pauses() {
    // Mixed pause lengths, including some far below and above the context.
    let mut speech = Vec::new();
    let mut t = 0.0;
    for i in 0..40 {
        let len = 3.0 + f64::from(i % 5);
        speech.push(iv(t, t + len));
        t += len + 0.05 + f64::from(i % 7) * 0.5;
    }
    let chunks = chunk_boundaries(&speech, 0.0, t);
    assert_valid(&chunks);
    assert!(chunks.len() > 5);
}

// ---- ChunkPlanner: the rolling-commit logic over the same DP --------------

#[test]
fn no_commit_before_the_horizon() {
    let mut planner = ChunkPlanner::new();
    let mut t = 0.0;
    while t + 10.0 < COMMIT_HORIZON_S - 20.0 {
        assert!(planner.push(iv(t, t + 10.0)).is_empty());
        t += 12.0;
    }
    assert!(planner.earliest_pending_start().is_some());
}

#[test]
fn crossing_the_horizon_commits_a_stable_prefix() {
    let mut planner = ChunkPlanner::new();
    let mut committed = Vec::new();
    let mut t = 0.0;
    while committed.is_empty() {
        committed.extend(planner.push(iv(t, t + 10.0)));
        t += 12.0;
    }
    let frontier = t - 12.0 + 10.0; // end of the interval that triggered
    for c in &committed {
        assert!(c.end - c.start <= MAX_DURATION_S + 1e-9);
        assert!(c.window_len() <= MAX_DURATION_S + 1e-9);
        assert!(
            c.end <= frontier - COMMIT_MARGIN_S + 1e-9,
            "committed chunk {}..{} inside the margin",
            c.start,
            c.end
        );
    }
    // The uncommitted tail is retained for the next round, and no future
    // window may reach below the last committed one.
    let pending_start = planner.earliest_pending_start().unwrap();
    let last = committed.last().unwrap();
    assert!(pending_start >= last.window_end - 1e-9);
}

#[test]
fn poll_commits_pending_speech_after_long_silence() {
    let mut planner = ChunkPlanner::new();
    assert!(planner.push(iv(3.0, 9.0)).is_empty());
    // Silence rolls the frontier far past the speech: commit without new
    // intervals so the PCM behind it can be released.
    assert!(planner.poll(9.0 + COMMIT_MARGIN_S - 1.0).is_empty());
    let committed = planner.poll(3.0 + COMMIT_HORIZON_S + 1.0);
    assert_eq!(spans(&committed), vec![(3.0, 9.0)]);
    // The window takes its run-in from the leading audio and its run-out from
    // the silence the frontier has crossed — the same context a whole-file
    // finish would have given it.
    assert!(
        (committed[0].window_start - (3.0 - BOUNDARY_CONTEXT_S)).abs() < 1e-9
    );
    assert!(
        (committed[0].window_end - (9.0 + BOUNDARY_CONTEXT_S)).abs() < 1e-9
    );
    assert!(planner.earliest_pending_start().is_none());
}

#[test]
fn a_cut_inside_a_presplit_interval_keeps_the_tail() {
    let mut planner = ChunkPlanner::new();
    // One continuous 200 s speech interval: the DP presplits it; the commit
    // boundary falls inside the interval and the tail must stay pending.
    let committed = planner.push(iv(0.0, 200.0));
    assert!(!committed.is_empty());
    let boundary = committed.last().unwrap().end;
    assert!(boundary <= 200.0 - COMMIT_MARGIN_S + 1e-9);
    // Continuous speech has no boundary pause, so the window ends on the cut.
    assert!((committed.last().unwrap().window_end - boundary).abs() < 1e-9);
    let pending = planner.earliest_pending_start().unwrap();
    assert!(
        (pending - boundary).abs() < 1e-9,
        "tail starts at the boundary"
    );
    // Finishing yields the rest, tiling up to 200 s.
    let rest = planner.finish(200.0);
    assert!((rest.last().unwrap().end - 200.0).abs() < 1e-9);
    let mut prev = boundary;
    for c in &rest {
        assert!((c.start - prev).abs() < 1e-9, "gap in presplit tiling");
        assert!(c.end - c.start <= MAX_DURATION_S + 1e-9);
        prev = c.end;
    }
}

#[test]
fn commits_then_finish_cover_all_speech_in_order() {
    let mut planner = ChunkPlanner::new();
    let mut chunks = Vec::new();
    let mut t = 0.0;
    for _ in 0..60 {
        chunks.extend(planner.push(iv(t, t + 8.0)));
        t += 9.5;
    }
    chunks.extend(planner.finish(t));
    // Ordered, non-overlapping, hard cap respected, all speech covered — and
    // the audio windows tile the same way, across commit rounds included.
    for pair in chunks.windows(2) {
        assert!(pair[1].start >= pair[0].end - 1e-9);
        assert!(
            pair[1].window_start >= pair[0].window_end - 1e-9,
            "windows overlap across a commit: {pair:?}"
        );
    }
    for c in &chunks {
        assert!(c.end - c.start <= MAX_DURATION_S + 1e-9);
        assert!(c.window_len() <= MAX_DURATION_S + 1e-9);
    }
    assert!((chunks.first().unwrap().start - 0.0).abs() < 1e-9);
    assert!((chunks.last().unwrap().end - (t - 9.5 + 8.0)).abs() < 1e-9);
}
