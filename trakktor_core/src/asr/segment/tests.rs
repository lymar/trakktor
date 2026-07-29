use super::{
    BOUNDARY_CONTEXT_S, BOUNDARY_OVERLAP_S, COMMIT_HORIZON_S, COMMIT_MARGIN_S,
    Chunk, ChunkPlanner, DecodedWord, Interval, MAX_DURATION_S, MAX_SPEECH_S,
    chunk_boundaries, own_words,
};

fn iv(start: f64, end: f64) -> Interval { Interval { start, end } }

/// The speech spans, for the assertions that only care about the cuts.
fn spans(chunks: &[Chunk]) -> Vec<(f64, f64)> {
    chunks.iter().map(|c| (c.start, c.end)).collect()
}

/// Per-chunk and neighbour-wise invariants: the speech fits the bound and sits
/// inside both the owned span and the window, and owned spans never overlap.
/// Windows are free to overlap — that is the point of [`BOUNDARY_OVERLAP_S`] —
/// and an owned span may reach past its own window into silence nobody hears,
/// so neither is forbidden here. The property that actually makes the overlap
/// safe is checked by [`assert_heard_is_owned`].
fn assert_valid(chunks: &[Chunk]) {
    for c in chunks {
        assert!(c.end > c.start, "empty chunk {c:?}");
        assert!(
            c.end - c.start <= MAX_SPEECH_S + 1e-9,
            "speech of {c:?} exceeds MAX_SPEECH_S"
        );
        assert!(
            c.window_len() <= MAX_DURATION_S + 1e-9,
            "window of {c:?} exceeds MAX_DURATION_S"
        );
        assert!(
            c.keep_start <= c.start + 1e-9 && c.keep_end >= c.end - 1e-9,
            "owned span of {c:?} does not enclose its speech"
        );
        assert!(
            c.window_start <= c.start + 1e-9 && c.window_end >= c.end - 1e-9,
            "window of {c:?} does not enclose its speech"
        );
        assert!(
            c.window_start >= -1e-9,
            "window of {c:?} starts before zero"
        );
    }
    for pair in chunks.windows(2) {
        assert!(
            pair[1].start >= pair[0].end - 1e-9,
            "speech spans overlap: {pair:?}"
        );
        assert!(
            pair[1].keep_start >= pair[0].keep_end - 1e-9,
            "owned spans overlap: {pair:?}"
        );
    }
}

/// The property the whole design rests on: **every instant any chunk hears
/// belongs to exactly one chunk**. Fewer would drop a word both neighbours
/// decoded; more would emit it twice. Needs the complete set of chunks — on a
/// prefix the last window legitimately reaches into what the next chunk owns.
fn assert_heard_is_owned(chunks: &[Chunk]) {
    let lo = chunks
        .iter()
        .map(|c| c.window_start)
        .fold(f64::MAX, f64::min);
    let hi = chunks.iter().map(|c| c.window_end).fold(f64::MIN, f64::max);
    let mut t = lo + 1e-6;
    while t < hi {
        if chunks
            .iter()
            .any(|c| c.window_start <= t && t < c.window_end)
        {
            let owners = chunks.iter().filter(|c| c.owns(t)).count();
            assert_eq!(owners, 1, "instant {t} is heard but owned {owners}×");
        }
        t += 0.01;
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
    assert_eq!(chunks.last().unwrap().keep_end, 12.0);
    assert_eq!(chunks.last().unwrap().window_end, 12.0);
}

#[test]
fn presplits_interval_over_the_speech_bound() {
    // Continuous speech is cut into parts that each leave room for both
    // boundaries inside the model's window bound.
    let speech = [iv(0.0, 70.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 70.0);
    assert!(
        chunks.len() >= 3,
        "expected >=3 parts, got {}",
        chunks.len()
    );
    assert_valid(&chunks);
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
fn a_short_boundary_pause_is_owned_evenly_and_fully() {
    // The 1.5 s pause at the cut is below 2 * BOUNDARY_CONTEXT_S, so the two
    // owned spans meet in its middle: every sample of it belongs to someone.
    let speech = [iv(0.0, 16.0), iv(16.3, 20.0), iv(21.5, 38.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 38.0);
    assert_valid(&chunks);
    let meet = (chunks[0].end + chunks[1].start) / 2.0;
    assert!((chunks[0].keep_end - meet).abs() < 1e-9);
    assert!((chunks[1].keep_start - meet).abs() < 1e-9);
}

#[test]
fn a_long_pause_is_owned_in_half_but_heard_only_in_part() {
    // A 6 s pause: ownership still splits it down the middle, so no stretch of
    // it is unowned — but only BOUNDARY_CONTEXT_S of it is worth hearing, and
    // the window stops there plus the overlap.
    let speech = [iv(0.0, 10.0), iv(16.0, 26.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 30.0);
    assert_valid(&chunks);
    assert_heard_is_owned(&chunks);
    assert!((chunks[0].keep_end - 13.0).abs() < 1e-9);
    assert!((chunks[1].keep_start - 13.0).abs() < 1e-9);
    // The overlap cannot reach the neighbour's speech across a pause this
    // long, and extending into silence alone measurably buys nothing: the
    // windows stop at the context.
    assert!((chunks[0].window_end - (10.0 + BOUNDARY_CONTEXT_S)).abs() < 1e-9);
    assert!(
        (chunks[1].window_start - (16.0 - BOUNDARY_CONTEXT_S)).abs() < 1e-9
    );
}

#[test]
fn windows_overlap_by_twice_the_declared_amount() {
    // Neighbours hear across the boundary into each other's speech; the
    // stretch heard twice is exactly 2 * BOUNDARY_OVERLAP_S wide.
    let speech = [iv(0.0, 16.0), iv(16.3, 20.0), iv(21.5, 38.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 38.0);
    assert_valid(&chunks);
    assert_heard_is_owned(&chunks);
    let heard_twice = chunks[0].window_end - chunks[1].window_start;
    assert!((heard_twice - 2.0 * BOUNDARY_OVERLAP_S).abs() < 1e-9);
}

#[test]
fn every_heard_instant_is_owned_exactly_once() {
    // A layout mixing pauses far below and far above the overlap width: what
    // `owns` decides must be a partition of everything heard, not a matter of
    // rounding.
    let speech = [
        iv(0.0, 6.0),
        iv(7.2, 14.0),
        iv(15.0, 30.0),
        iv(33.0, 44.0),
        iv(44.1, 52.0),
    ];
    let chunks = chunk_boundaries(&speech, 0.0, 55.0);
    assert_valid(&chunks);
    assert_heard_is_owned(&chunks);
}

#[test]
fn presplit_boundaries_are_owned_back_to_back() {
    // Boundaries introduced inside continuous speech have no pause to share:
    // ownership abuts exactly, so no word is claimed twice — while the windows
    // still overlap, which is how the split word keeps its context.
    let speech = [iv(0.0, 70.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 70.0);
    assert_valid(&chunks);
    for pair in chunks.windows(2) {
        assert!(
            (pair[0].keep_end - pair[1].keep_start).abs() < 1e-9,
            "presplit ownership must abut exactly: {pair:?}"
        );
        assert!((pair[0].keep_end - pair[0].end).abs() < 1e-9);
        assert!(
            pair[1].window_start < pair[0].window_end - 1e-9,
            "presplit windows must still overlap: {pair:?}"
        );
    }
}

#[test]
fn content_bounds_limit_ownership_and_the_outer_window() {
    // `content_start` bounds what may be *owned* — in a streaming run it is the
    // previous commit's owned edge. The window still reaches past it, which is
    // the overlap, and stops at zero.
    let speech = [iv(10.2, 16.0), iv(17.0, 24.0)];
    let chunks = chunk_boundaries(&speech, 10.0, 24.3);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 1);
    assert!((chunks[0].keep_start - 10.0).abs() < 1e-9);
    assert!((chunks[0].window_start - 7.0).abs() < 1e-9);
    assert!((chunks[0].keep_end - 24.3).abs() < 1e-9);
    assert!((chunks[0].window_end - 24.3).abs() < 1e-9);
}

#[test]
fn the_window_reaches_the_hard_bound_but_never_passes_it() {
    // The speech bound is set so that a maximal chunk with speech close on
    // both sides — contexts and overlaps in full — lands exactly on
    // MAX_DURATION_S.
    let speech = [
        iv(0.0, 8.0),
        iv(10.0, 10.0 + MAX_SPEECH_S),
        iv(12.0 + MAX_SPEECH_S, 18.0 + MAX_SPEECH_S),
    ];
    let chunks = chunk_boundaries(&speech, 0.0, 20.0 + MAX_SPEECH_S);
    assert_valid(&chunks);
    assert_eq!(chunks.len(), 3, "the middle interval cannot merge");
    assert!((chunks[1].window_len() - MAX_DURATION_S).abs() < 1e-9);
}

#[test]
fn overlap_is_paid_only_where_it_reaches_speech() {
    // Boundary pauses shorter than the reach get the overlap on both sides;
    // at a pause past it the windows stop at the context and hear no silence
    // beyond — while ownership still splits the whole pause.
    let speech = [iv(0.0, 18.0), iv(20.0, 38.0), iv(44.0, 60.0)];
    let chunks = chunk_boundaries(&speech, 0.0, 60.0);
    assert_valid(&chunks);
    assert_heard_is_owned(&chunks);
    assert_eq!(chunks.len(), 3);
    // 2 s pause: the overlap crosses it into the neighbour.
    assert!(
        (chunks[0].window_end -
            (18.0 + BOUNDARY_CONTEXT_S + BOUNDARY_OVERLAP_S))
            .abs() <
            1e-9
    );
    assert!(
        (chunks[1].window_start -
            (20.0 - BOUNDARY_CONTEXT_S - BOUNDARY_OVERLAP_S))
            .abs() <
            1e-9
    );
    // 6 s pause: nothing but silence to reach — no overlap on either side.
    assert!((chunks[1].window_end - (38.0 + BOUNDARY_CONTEXT_S)).abs() < 1e-9);
    assert!(
        (chunks[2].window_start - (44.0 - BOUNDARY_CONTEXT_S)).abs() < 1e-9
    );
}

#[test]
fn ownership_tiles_across_many_pauses() {
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
    assert_heard_is_owned(&chunks);
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
    assert_valid(&committed);
    for c in &committed {
        assert!(
            c.end <= frontier - COMMIT_MARGIN_S + 1e-9,
            "committed chunk {}..{} inside the margin",
            c.start,
            c.end
        );
    }
    // Ownership carries over: nothing committed later may own audio below the
    // last committed owned edge.
    let last = committed.last().unwrap();
    let rest = planner.finish(t);
    assert!(rest[0].keep_start >= last.keep_end - 1e-9);
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
    // The chunk takes its context and overlap from the leading audio and from
    // the silence the frontier has crossed — the same a whole-file finish
    // would have given it.
    let c = committed[0];
    // An outer edge claims only what it can hear, not everything up to the
    // frontier: were it otherwise, speech still open near the frontier would
    // end up inside this chunk's span.
    assert!((c.keep_start - 0.0).abs() < 1e-9);
    assert!(
        (c.keep_end - (9.0 + BOUNDARY_CONTEXT_S + BOUNDARY_OVERLAP_S)).abs() <
            1e-9
    );
    // On both sides the overlap would hear only silence — the leading hush
    // and the crossed frontier silence — so it is not paid for: the window
    // stops at the context.
    assert!((c.window_start - (3.0 - BOUNDARY_CONTEXT_S)).abs() < 1e-9);
    assert!((c.window_end - (9.0 + BOUNDARY_CONTEXT_S)).abs() < 1e-9);
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
    // Continuous speech has no boundary pause, so ownership ends on the cut.
    assert!((committed.last().unwrap().keep_end - boundary).abs() < 1e-9);
    let pending = planner.earliest_pending_start().unwrap();
    assert!(
        pending <= boundary + 1e-9,
        "PCM the next window needs must not be released"
    );
    // Finishing yields the rest, tiling up to 200 s.
    let rest = planner.finish(200.0);
    assert!((rest.last().unwrap().end - 200.0).abs() < 1e-9);
    let mut prev = boundary;
    for c in &rest {
        assert!((c.start - prev).abs() < 1e-9, "gap in presplit tiling");
        assert!(c.end - c.start <= MAX_SPEECH_S + 1e-9);
        prev = c.end;
    }
}

#[test]
fn commits_then_finish_cover_all_speech_in_order() {
    let mut planner = ChunkPlanner::new();
    let mut chunks = Vec::new();
    let mut retained: Vec<(usize, f64)> = Vec::new();
    let mut t = 0.0;
    for _ in 0..60 {
        chunks.extend(planner.push(iv(t, t + 8.0)));
        if let Some(p) = planner.earliest_pending_start() {
            retained.push((chunks.len(), p));
        }
        t += 9.5;
    }
    chunks.extend(planner.finish(t));
    assert_valid(&chunks);
    assert_heard_is_owned(&chunks);
    assert!((chunks.first().unwrap().start - 0.0).abs() < 1e-9);
    assert!((chunks.last().unwrap().end - (t - 9.5 + 8.0)).abs() < 1e-9);
    // Shedding safety: whatever PCM the planner declared releasable must lie
    // below every window still to come.
    for (produced, floor) in retained {
        for c in &chunks[produced..] {
            assert!(
                c.window_start >= floor - 1e-9,
                "window at {} needs PCM released at {floor}",
                c.window_start
            );
        }
    }
}

// ---- own_words: the seam filter over decoded words ------------------------

/// A minimal word for the filter tests.
#[derive(Debug, Clone, PartialEq)]
struct Tw {
    text: String,
    start: f64,
    end: f64,
}

impl DecodedWord for Tw {
    fn span(&self) -> (f64, f64) { (self.start, self.end) }

    fn shift(&mut self, offset: f64) {
        self.start += offset;
        self.end += offset;
    }

    fn text(&self) -> &str { &self.text }
}

fn tw(text: &str, start: f64, end: f64) -> Tw {
    Tw {
        text: text.to_string(),
        start,
        end,
    }
}

/// A chunk owning `[keep_start, keep_end)` with a window widened by the
/// overlap on both sides.
fn owner(keep_start: f64, keep_end: f64) -> Chunk {
    Chunk {
        start: keep_start,
        end: keep_end,
        keep_start,
        keep_end,
        window_start: keep_start - BOUNDARY_OVERLAP_S,
        window_end: keep_end + BOUNDARY_OVERLAP_S,
    }
}

#[test]
fn own_words_keeps_the_whole_text_verbatim_when_nothing_is_dropped() {
    let words = vec![tw("alpha", 1.0, 1.4), tw("beta", 2.0, 2.5)];
    let chunk = owner(0.0, 10.0);
    // Odd spacing must survive: the decoded text is not rebuilt.
    let (text, kept) =
        own_words(words, 0.0, &chunk, &[], "alpha  beta".to_string());
    assert_eq!(text, "alpha  beta");
    assert_eq!(kept.len(), 2);
}

#[test]
fn own_words_shifts_onto_the_original_timeline() {
    let words = vec![tw("alpha", 1.0, 1.4)];
    let chunk = owner(100.0, 110.0);
    let (_, kept) = own_words(words, 100.0, &chunk, &[], String::new());
    assert_eq!(kept[0].span(), (101.0, 101.4));
}

#[test]
fn own_words_drops_the_context_and_rebuilds_the_text() {
    // The third word lies past the owned span: heard for context only.
    let words = vec![
        tw("alpha", 1.0, 1.4),
        tw("beta", 2.0, 2.5),
        tw("gamma", 11.0, 11.5),
    ];
    let chunk = owner(0.0, 10.0);
    let (text, kept) =
        own_words(words, 0.0, &chunk, &[], "alpha beta gamma".to_string());
    assert_eq!(text, "alpha beta");
    assert_eq!(kept.len(), 2);
}

#[test]
fn own_words_drops_a_seam_duplicate_the_previous_chunk_kept() {
    // The first word sits just inside the owned span, but the previous chunk
    // already kept a word over the same audio — its timing merely put the
    // midpoint on its own side. Emitting both would double the word.
    let words = vec![tw("alpha", 9.8, 10.4), tw("beta", 11.0, 11.5)];
    let chunk = owner(10.0, 20.0);
    let prev = [(9.7, 10.3)];
    let (text, kept) =
        own_words(words, 0.0, &chunk, &prev, "alpha beta".to_string());
    assert_eq!(text, "beta");
    assert_eq!(kept.len(), 1);
}

#[test]
fn own_words_adopts_a_seam_orphan_no_one_kept() {
    // The first word's midpoint falls just short of the owned span, but the
    // previous chunk kept nothing over that audio — its own timing pushed the
    // word past its line. Dropping it here would lose it entirely.
    let words = vec![tw("alpha", 9.4, 9.9), tw("beta", 11.0, 11.5)];
    let chunk = owner(10.0, 20.0);
    let prev = [(8.0, 8.9)];
    let (_, kept) = own_words(words, 0.0, &chunk, &prev, String::new());
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].text(), "alpha");
}

#[test]
fn own_words_does_not_adopt_a_differently_cut_variant() {
    // The neighbour transcribed that stretch — merely with its own word
    // boundaries, so no span covers this variant, and with the thin spans of
    // emission-frame timing they may not even intersect it. Anything kept
    // within the clearance blocks adoption.
    let words = vec![tw("alphabet", 8.9, 9.9), tw("beta", 11.0, 11.5)];
    let chunk = owner(10.0, 20.0);
    let prev = [(8.6, 9.2), (9.3, 10.1)];
    let (text, kept) =
        own_words(words, 0.0, &chunk, &prev, "alphabet beta".to_string());
    assert_eq!(text, "beta");
    assert_eq!(kept.len(), 1);
}

#[test]
fn own_words_does_not_adopt_a_retimed_thin_span() {
    // The same short word decoded by both sides with emission-frame spans
    // that do not even touch: span intersection sees free audio, but the
    // neighbour's word sits well inside the clearance.
    let words = vec![tw("al", 9.70, 9.76), tw("beta", 11.0, 11.5)];
    let chunk = owner(10.0, 20.0);
    let prev = [(9.62, 9.68)];
    let (text, kept) =
        own_words(words, 0.0, &chunk, &prev, "al beta".to_string());
    assert_eq!(text, "beta");
    assert_eq!(kept.len(), 1);
}

#[test]
fn own_words_never_adopts_bare_punctuation() {
    // A pause transcribed as a lone dash is not a word worth recovering.
    let words = vec![tw("—", 9.4, 9.8), tw("beta", 11.0, 11.5)];
    let chunk = owner(10.0, 20.0);
    let (text, kept) = own_words(words, 0.0, &chunk, &[], "— beta".to_string());
    assert_eq!(text, "beta");
    assert_eq!(kept.len(), 1);
}

#[test]
fn own_words_leaves_far_words_to_plain_ownership() {
    // Beyond the reconciliation zone plain ownership applies: a word far
    // below the line is the neighbour's even if nothing covers it, and a word
    // deep inside is ours.
    let words = vec![
        tw("alpha", 5.0, 5.6),
        tw("beta", 15.0, 15.6),
        tw("gamma", 19.0, 19.4),
    ];
    let chunk = owner(10.0, 20.0);
    let (text, kept) =
        own_words(words, 0.0, &chunk, &[], "alpha beta gamma".to_string());
    assert_eq!(text, "beta gamma");
    assert_eq!(kept.len(), 2);
}

#[test]
fn own_words_reconciles_a_wildly_mistimed_duplicate() {
    // Ill-fitting audio can come back segmented completely differently by the
    // two decodes: the previous chunk kept a short word, this one heard the
    // same stretch as the head of one long pseudo-word seconds past the line.
    // Coverage, not the midpoint, decides — the long word goes.
    let words = vec![tw("alphabeta", 9.6, 12.4), tw("gamma", 13.0, 13.5)];
    let chunk = owner(10.0, 20.0);
    let prev = [(9.5, 10.6)];
    let (text, kept) =
        own_words(words, 0.0, &chunk, &prev, "alphabeta gamma".to_string());
    assert_eq!(text, "gamma");
    assert_eq!(kept.len(), 1);
}
