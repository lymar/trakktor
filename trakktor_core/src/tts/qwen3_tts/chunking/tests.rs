//! Tests for the decode layout. It has to match the reference exactly — a
//! chunk boundary that drifts changes the audio.

use super::*;

#[test]
fn the_attention_window_looks_back_and_never_forward() {
    let window = 3;
    // Frame 4 sees frames 2, 3 and 4.
    let row: Vec<bool> =
        (0..6).map(|key| window_visible(4, key, window)).collect();
    assert_eq!(row, vec![false, false, true, true, true, false]);
    // Frame 0 sees only itself.
    let row: Vec<bool> =
        (0..6).map(|key| window_visible(0, key, window)).collect();
    assert_eq!(row, vec![true, false, false, false, false, false]);
}

#[test]
fn a_short_decode_is_a_single_chunk_without_context() {
    let plan = chunk_plan(10);
    assert_eq!(
        plan,
        vec![Chunk {
            start: 0,
            end: 10,
            context: 0
        }]
    );
    assert_eq!(plan[0].context_start(), 0);
    assert_eq!(plan[0].span(), 10);
}

#[test]
fn a_long_decode_primes_every_chunk_after_the_first() {
    let total = CHUNK_FRAMES * 2 + 40;
    let plan = chunk_plan(total);

    assert_eq!(plan.len(), 3);
    // The first chunk starts cold.
    assert_eq!(plan[0].context, 0);
    assert_eq!(plan[0].context_start(), 0);
    // Later chunks re-decode the fixed context and then discard it.
    for chunk in &plan[1..] {
        assert_eq!(chunk.context, LEFT_CONTEXT_FRAMES);
        assert_eq!(chunk.context_start(), chunk.start - LEFT_CONTEXT_FRAMES);
        assert_eq!(chunk.span(), chunk.end - chunk.start + chunk.context);
    }

    // The kept spans tile the input exactly: no frame decoded twice, none lost.
    assert_eq!(plan[0].start, 0);
    assert_eq!(plan.last().expect("last").end, total);
    for pair in plan.windows(2) {
        assert_eq!(pair[0].end, pair[1].start);
    }
}

#[test]
fn an_empty_decode_has_no_chunks() {
    assert!(chunk_plan(0).is_empty());
}

#[cfg(feature = "tts-burn")]
#[test]
fn a_chunk_is_decoded_at_a_rounded_length() {
    // A full chunk plus its context already lands on the step.
    assert_eq!(aligned_span(DECODE_ALIGN), DECODE_ALIGN);
    // Anything shorter is rounded up, so a run's leftover frames never
    // introduce a shape of their own.
    assert_eq!(aligned_span(1), DECODE_ALIGN);
    assert_eq!(aligned_span(DECODE_ALIGN + 1), 2 * DECODE_ALIGN);
    // The rounding never loses frames.
    for span in [1, 25, 190, 300, 325] {
        assert!(aligned_span(span) >= span);
    }
}
