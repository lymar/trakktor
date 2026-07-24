//! Tests for the parts of the decoder that do not need weights: the attention
//! window and the way a long decode is split into chunks. Both have to match
//! the reference exactly — a chunk boundary that drifts changes the audio.

use candle_core::{Device, IndexOp};

use super::*;

#[test]
fn the_attention_window_looks_back_and_never_forward() {
    let frames = 6;
    let window = 3;
    let mask = sliding_window_mask(frames, window, &Device::Cpu)
        .expect("mask")
        .i((0, 0))
        .expect("squeeze")
        .to_vec2::<f32>()
        .expect("rows");

    for (query, row) in mask.iter().enumerate() {
        for (key, &value) in row.iter().enumerate() {
            let visible = value == 0.0;
            assert_eq!(
                visible,
                key <= query && key + window > query,
                "query {query} key {key}"
            );
        }
    }

    // Spot-check the shape of one row: frame 4 sees frames 2, 3 and 4.
    let row: Vec<bool> = mask[4].iter().map(|&v| v == 0.0).collect();
    assert_eq!(row, vec![false, false, true, true, true, false]);
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
