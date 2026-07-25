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

/// The driver's forward contract: every row is `[cls] + window + [sep]`, rows
/// arrive batched, and the stitched result equals running `stitch` directly
/// over the per-window logits.
#[test]
fn windowed_logits_frames_batches_and_stitches() {
    let ids: Vec<u32> = (100..100 + 1500).collect();
    let (cls, sep, stride, batch_size) = (7u32, 9u32, 256, 2);

    let n_tokens = ids.len();
    let block = block_size(n_tokens);
    let windows = plan_windows(n_tokens, stride);
    assert!(windows.len() > batch_size, "the test should span >1 batch");

    let got = windowed_logits::<()>(
        &ids,
        cls,
        sep,
        stride,
        batch_size,
        Weighting::Hat,
        |buffer, n_batch, seq_len| {
            assert_eq!(seq_len, block + 2);
            assert_eq!(buffer.len(), n_batch * seq_len);
            assert!(n_batch <= batch_size);
            Ok(buffer
                .chunks(seq_len)
                .map(|row| {
                    assert_eq!(row[0], cls);
                    assert_eq!(row[seq_len - 1], sep);
                    // The logit of each real token is the token id itself.
                    row[1..seq_len - 1].iter().map(|&t| t as f32).collect()
                })
                .collect())
        },
    )
    .unwrap();

    let per_window: Vec<Vec<f32>> = windows
        .iter()
        .map(|w| ids[w.start..w.end].iter().map(|&t| t as f32).collect())
        .collect();
    let want = stitch(
        n_tokens,
        &windows,
        &per_window,
        &weights(block, Weighting::Hat),
    );
    assert_eq!(got, want);
}

/// Batch size must not change the result — only how the windows are grouped.
#[test]
fn windowed_logits_is_batch_size_invariant() {
    let ids: Vec<u32> = (0..900).map(|i| i % 251).collect();
    let forward = |buffer: &[u32], _n: usize, seq_len: usize| {
        Ok::<_, ()>(
            buffer
                .chunks(seq_len)
                .map(|row| {
                    row[1..seq_len - 1]
                        .iter()
                        .map(|&t| (t as f32).sin())
                        .collect()
                })
                .collect(),
        )
    };
    let one = windowed_logits(&ids, 1, 2, 200, 1, Weighting::Uniform, forward)
        .unwrap();
    let many =
        windowed_logits(&ids, 1, 2, 200, 64, Weighting::Uniform, forward)
            .unwrap();
    assert_eq!(one, many);
}

/// Empty input short-circuits without calling the forward.
#[test]
fn windowed_logits_empty_input() {
    let got = windowed_logits::<()>(
        &[],
        1,
        2,
        256,
        32,
        Weighting::Uniform,
        |_, _, _| panic!("forward must not run on empty input"),
    )
    .unwrap();
    assert!(got.is_empty());
}

// ---------------------------------------------------------------------------
// splitting to a budget
// ---------------------------------------------------------------------------

/// A text with boundary probabilities placed by hand: a strong boundary after
/// every sentence, a weaker one after every comma.
fn sentences() -> (Vec<char>, Vec<f32>) {
    let text: Vec<char> = "One two, three four. Five six, seven eight. Nine \
                           ten, eleven twelve."
        .chars()
        .collect();
    let probs = text
        .iter()
        .map(|c| match c {
            '.' => 0.9,
            ',' => 0.4,
            _ => 0.0,
        })
        .collect();
    (text, probs)
}

/// Cost in characters — the unit the tests state their budgets in.
fn chars_cost(text: &str) -> usize { text.chars().count() }

#[test]
fn the_highest_fitting_threshold_wins() {
    let (text, probs) = sentences();

    // A budget that fits a sentence: cut on sentences only, three pieces.
    let pieces = split_to_budget(&text, &probs, 0.5, 25, &chars_cost);
    assert_eq!(
        pieces,
        [
            "One two, three four.",
            "Five six, seven eight.",
            "Nine ten, eleven twelve.",
        ]
    );

    // A tighter budget forces the search below the comma level, which is the
    // only way to fit — and it stops there rather than cutting finer.
    let pieces = split_to_budget(&text, &probs, 0.5, 15, &chars_cost);
    assert_eq!(
        pieces,
        [
            "One two,",
            "three four.",
            "Five six,",
            "seven eight.",
            "Nine ten,",
            "eleven twelve.",
        ]
    );
}

#[test]
fn a_text_within_budget_is_left_whole() {
    let (text, probs) = sentences();

    let pieces = split_to_budget(&text, &probs, 0.5, 1000, &chars_cost);

    assert_eq!(pieces.len(), 1);
    assert_eq!(pieces[0].chars().count(), text.len());
}

#[test]
fn a_piece_the_model_cannot_cut_is_split_anyway() {
    // No boundary anywhere: the guarantee has to come from the mechanical
    // split, which prefers the sentence end nearest the middle.
    let text: Vec<char> = "aaa bbb ccc. ddd eee fff ggg hhh".chars().collect();
    let probs = vec![0.0; text.len()];

    let pieces = split_to_budget(&text, &probs, 0.5, 20, &chars_cost);

    assert_eq!(pieces, ["aaa bbb ccc.", "ddd eee fff ggg hhh"]);
    for piece in &pieces {
        assert!(chars_cost(piece) <= 20, "{piece} is over budget");
    }
}

#[test]
fn every_piece_fits_even_without_punctuation_or_spaces() {
    let text: Vec<char> = "щ".repeat(50).chars().collect();
    let probs = vec![0.0; text.len()];

    let pieces = split_to_budget(&text, &probs, 0.5, 7, &chars_cost);

    assert_eq!(pieces.iter().map(|p| p.chars().count()).sum::<usize>(), 50);
    for piece in &pieces {
        assert!(chars_cost(piece) <= 7, "{piece} is over budget");
    }
}

#[test]
fn pieces_are_merged_back_when_the_threshold_had_to_go_low() {
    // Three short sentences and one long one. Fitting the long one forces the
    // search down to the comma level, which would chop the short ones apart
    // too — merging puts them back together.
    let text: Vec<char> = "Раз. Два. Три. Это предложение, увы, длиннее \
                           прочих вместе взятых."
        .chars()
        .collect();
    let probs: Vec<f32> = text
        .iter()
        .map(|c| match c {
            '.' => 0.9,
            ',' => 0.4,
            _ => 0.0,
        })
        .collect();

    let pieces = split_to_budget(&text, &probs, 0.5, 35, &chars_cost);

    assert_eq!(
        pieces,
        [
            "Раз. Два. Три. Это предложение,",
            "увы, длиннее прочих вместе взятых.",
        ]
    );
    for piece in &pieces {
        assert!(chars_cost(piece) <= 35, "{piece} is over budget");
    }
}
