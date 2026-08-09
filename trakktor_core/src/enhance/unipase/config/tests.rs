use super::*;

#[test]
fn alignment_lands_on_the_extractors_remainder() {
    for samples in [80usize, 400, 32_000, 32_080, 128_000, 129_999] {
        assert_eq!(aligned_len(samples) % HOP, PAD_REMAINDER);
    }
}

#[test]
fn alignment_leaves_an_already_aligned_window_alone() {
    assert_eq!(aligned_len(32_080), 32_080);
    assert_eq!(aligned_len(80), 80);
}

#[test]
fn alignment_trims_when_the_remainder_is_past_eighty() {
    // 500 = 320 + 180, and 180 > 80: the reference pads by -100.
    assert_eq!(aligned_len(500), 400);
}

#[test]
fn the_window_advances_by_half_of_itself() {
    assert_eq!(WINDOW_SECONDS, 2 * HOP_SECONDS);
}

#[test]
fn the_vocoder_hops_by_one_encoder_frame() {
    // The vocoder's hop is what makes one encoder frame one vocoder frame.
    assert_eq!(HOP, 320);
    assert_eq!(
        CONV_LAYERS.iter().map(|&(_, _, s)| s).product::<usize>(),
        HOP
    );
}
