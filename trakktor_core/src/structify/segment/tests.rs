//! Unit tests for the pure segmentation logic.

use super::*;

#[test]
fn single_window_when_text_fits() {
    // Under the block cap, one window covers everything.
    assert_eq!(plan_windows(5, 256), vec![Window { start: 0, end: 5 }]);
    assert_eq!(block_size(5), 5);
    assert_eq!(
        plan_windows(MAX_BLOCK, 256),
        vec![Window {
            start: 0,
            end: MAX_BLOCK
        }]
    );
}

#[test]
fn windows_tile_with_stride_and_pin_last() {
    // block == MAX_BLOCK (510); the final window is pinned to the end.
    let w = plan_windows(1000, 256);
    assert_eq!(
        w,
        vec![
            Window { start: 0, end: 510 },
            Window {
                start: 256,
                end: 766
            },
            Window {
                start: 490,
                end: 1000
            },
        ]
    );
    // Every window is exactly `block` long, so no padding is ever needed.
    for window in &w {
        assert_eq!(window.end - window.start, block_size(1000));
    }
}

#[test]
fn windows_just_over_block() {
    assert_eq!(
        plan_windows(511, 256),
        vec![Window { start: 0, end: 510 }, Window { start: 1, end: 511 },]
    );
}

#[test]
fn no_windows_for_empty() {
    assert_eq!(plan_windows(0, 256), Vec::<Window>::new());
}

#[test]
fn uniform_weights_are_flat() {
    assert_eq!(weights(3, Weighting::Uniform), vec![1.0, 1.0, 1.0]);
}

#[test]
fn hat_weights_are_triangular() {
    let w = weights(4, Weighting::Hat);
    let expected = [0.25f32, 0.75, 0.75, 0.25];
    for (got, want) in w.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    // Degenerate block of 1 keeps a unit weight.
    assert_eq!(weights(1, Weighting::Hat), vec![1.0]);
}

#[test]
fn stitch_averages_overlaps_by_weight() {
    // Windows [0,2) and [1,3) over 3 tokens, uniform weights.
    let windows = [Window { start: 0, end: 2 }, Window { start: 1, end: 3 }];
    let per_window = [vec![1.0f32, 2.0], vec![4.0f32, 6.0]];
    let w = weights(2, Weighting::Uniform);
    let out = stitch(3, &windows, &per_window, &w);
    // token0: 1/1; token1: (2 + 4)/2 = 3; token2: 6/1.
    assert_eq!(out, vec![1.0, 3.0, 6.0]);
}

#[test]
fn char_probs_land_on_the_last_character_of_each_token() {
    // Two tokens spanning [0,2) and [2,5); logits 0 and +10.
    let offsets = [(0usize, 2usize), (2, 5)];
    let logits = [0.0f32, 10.0];
    let probs = char_probs(5, &offsets, &logits);
    assert!((probs[1] - 0.5).abs() < 1e-6); // sigmoid(0)
    assert!(probs[4] > 0.999); // sigmoid(10)
    // Non-final characters stay zero.
    assert_eq!(probs[0], 0.0);
    assert_eq!(probs[2], 0.0);
    assert_eq!(probs[3], 0.0);
}

#[test]
fn paragraph_spans_cut_after_boundary_and_swallow_whitespace() {
    let text: Vec<char> = "ab cd".chars().collect();
    // Boundary probability high on 'b' (index 1).
    let probs = [0.0f32, 0.6, 0.0, 0.0, 0.0];
    let spans = paragraph_spans(&text, &probs, 0.5);
    // Cut after index 1, then swallow the space into the first paragraph.
    assert_eq!(
        spans,
        vec![Span { start: 0, end: 3 }, Span { start: 3, end: 5 },]
    );
}

#[test]
fn whole_text_is_one_span_without_boundaries() {
    let text: Vec<char> = "abcde".chars().collect();
    let probs = [0.0f32; 5];
    assert_eq!(
        paragraph_spans(&text, &probs, 0.5),
        vec![Span { start: 0, end: 5 }]
    );
}

#[test]
fn trim_span_drops_surrounding_whitespace() {
    let text: Vec<char> = "ab  ".chars().collect();
    assert_eq!(
        trim_span(&text, Span { start: 0, end: 4 }),
        Span { start: 0, end: 2 }
    );
    // A boundary that swallowed a trailing space trims back to the word.
    let text2: Vec<char> = " x ".chars().collect();
    assert_eq!(
        trim_span(&text2, Span { start: 0, end: 3 }),
        Span { start: 1, end: 2 }
    );
}
