use super::*;

#[test]
fn the_banded_width_is_the_split_plus_the_bands() {
    assert_eq!(BANDED, ERB_LOW_BINS + ERB_BANDS);
    assert!(ERB_LOW_BINS + (BINS - ERB_LOW_BINS) == BINS);
}

#[test]
fn two_stride_two_convolutions_reach_the_core_width() {
    // 129 → 65 → 33, each `(n + 1) / 2` for a stride-2 kernel of 5 padded by 2.
    let after_first = BANDED.div_ceil(2);
    assert_eq!(after_first.div_ceil(2), CORE_WIDTH);
}

#[test]
fn the_analysis_frames_a_second_of_audio_as_the_reference_does() {
    assert_eq!(frames(SAMPLE_RATE as usize), 63);
}

#[test]
fn a_chunk_is_a_memory_budget_and_holds_whole_frames() {
    // Chunks partition the recording exactly — the state carries, so there is
    // no overlap to get wrong.
    assert_eq!(CHUNK_SECONDS * SAMPLE_RATE as usize % HOP, 0);
    let reach = 2 * GT_DILATIONS[GT_BLOCKS - 1];
    assert!(CHUNK_SECONDS * SAMPLE_RATE as usize / HOP > reach);
}
