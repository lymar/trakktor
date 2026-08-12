//! Unit tests for the pure windowing/stitching logic, on fixed predictions.

use super::*;

/// A prediction tagged with `post` (used to identify which window/token a
/// merged entry came from).
fn tagged(post: u8, pre: u8) -> TokenPred {
    TokenPred {
        post,
        pre,
        sbd: false,
        cap: [false; MAX_SUBWORD_LEN],
    }
}

#[test]
fn plan_windows_overlapping() {
    // n=9, max=4, overlap=2: each window after the first starts 2 earlier than
    // the previous window's unclamped end.
    let windows = plan_windows(9, 4, 2);
    assert_eq!(
        windows,
        vec![
            Window { start: 0, end: 4 },
            Window { start: 2, end: 6 },
            Window { start: 4, end: 8 },
            Window { start: 6, end: 9 }, /* last window is short (content
                                          * length 3) */
        ]
    );
}

#[test]
fn plan_windows_single_when_short() {
    assert_eq!(plan_windows(3, 254, 16), vec![Window { start: 0, end: 3 }]);
    assert_eq!(plan_windows(0, 254, 16), Vec::new());
}

#[test]
fn plan_windows_survives_an_overlap_as_wide_as_the_window() {
    // Un-clamped, an overlap >= max_content would keep `start` from ever
    // advancing — an infinite loop. The clamp caps it at max_content - 1,
    // the largest stride that still moves forward.
    for overlap in [4, 5, 300] {
        let windows = plan_windows(9, 4, overlap);
        assert_eq!(windows.last(), Some(&Window { start: 5, end: 9 }));
        assert!(windows.len() <= 9, "one token of progress per window");
    }
}

#[test]
fn stitch_tiles_without_gaps_or_dups() {
    // The overlap seam is split in half: each interior window keeps its half.
    let windows = plan_windows(9, 4, 2);
    let ids: Vec<u32> = (0..9).collect();
    // Tag each window's content preds with the window index (post) and the
    // global token index (pre), so we can see which window owns each token.
    let per_window: Vec<Vec<TokenPred>> = windows
        .iter()
        .enumerate()
        .map(|(w, win)| {
            (win.start..win.end)
                .map(|g| tagged(w as u8, g as u8))
                .collect()
        })
        .collect();

    let merged = stitch(&windows, &per_window, &ids, 2);

    // Every original token appears exactly once, in order.
    let got_ids: Vec<u32> = merged.iter().map(|&(id, _)| id).collect();
    assert_eq!(got_ids, ids);
    // Each merged token carries its own global index (pre) — no misalignment.
    for &(id, pred) in &merged {
        assert_eq!(u32::from(pred.pre), id);
    }
    // Ownership: [0,3)->w0, [3,5)->w1, [5,7)->w2, [7,9)->w3.
    let owners: Vec<u8> = merged.iter().map(|&(_, p)| p.post).collect();
    assert_eq!(owners, vec![0, 0, 0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn windowed_predictions_strips_frame_and_reconstructs() {
    let ids: Vec<u32> = (10..19).collect(); // 9 tokens
    let (bos, eos) = (0u32, 1u32);
    let mut seen_lengths = Vec::new();

    let merged = windowed_predictions(
        &ids,
        bos,
        eos,
        4,
        2,
        16,
        |batch: &[Vec<u32>]| -> Result<Vec<Vec<TokenPred>>, ()> {
            // Every window in a batch must be equal length (no padding).
            let len = batch[0].len();
            assert!(batch.iter().all(|w| w.len() == len));
            seen_lengths.push(len);
            // Tag each position's post with its token id, so stripping BOS/EOS
            // and stitching is verifiable.
            Ok(batch
                .iter()
                .map(|w| w.iter().map(|&id| tagged(id as u8, 0)).collect())
                .collect())
        },
    )
    .unwrap();

    // The merged stream reconstructs the original ids, and each prediction's
    // post equals its token id (so BOS/EOS were dropped correctly).
    let got_ids: Vec<u32> = merged.iter().map(|&(id, _)| id).collect();
    assert_eq!(got_ids, ids);
    for &(id, pred) in &merged {
        assert_eq!(u32::from(pred.post), id);
    }
    // Two length buckets: three full windows (len 6) and one short (len 5).
    seen_lengths.sort_unstable();
    assert_eq!(seen_lengths, vec![5, 6]);
}

#[test]
fn windowed_predictions_empty() {
    let merged = windowed_predictions(
        &[],
        0,
        1,
        4,
        2,
        16,
        |_: &[Vec<u32>]| -> Result<Vec<Vec<TokenPred>>, ()> {
            panic!("forward should not be called for empty input")
        },
    )
    .unwrap();
    assert!(merged.is_empty());
}

#[test]
fn odd_overlap_rounds_down_to_even() {
    // An odd overlap would leave one token covered by both windows at each
    // seam (the halves dropped sum to overlap − 1); the driver rounds it down,
    // so the merged stream still reconstructs every token exactly once.
    let ids: Vec<u32> = (10..19).collect();
    let merged = windowed_predictions(
        &ids,
        0,
        1,
        4,
        3, // odd: effectively 2
        16,
        |batch: &[Vec<u32>]| -> Result<Vec<Vec<TokenPred>>, ()> {
            Ok(batch
                .iter()
                .map(|w| w.iter().map(|&id| tagged(id as u8, 0)).collect())
                .collect())
        },
    )
    .unwrap();
    let got_ids: Vec<u32> = merged.iter().map(|&(id, _)| id).collect();
    assert_eq!(got_ids, ids);
}
