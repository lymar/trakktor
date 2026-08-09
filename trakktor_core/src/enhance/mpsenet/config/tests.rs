//! The geometry, checked against the arithmetic that depends on it.

use super::*;

#[test]
fn the_spectrum_is_the_width_the_checkpoint_expects() {
    // The magnitude decoder's learned sigmoid has one slope per bin, and the
    // checkpoint stores 201 of them.
    assert_eq!(BINS, 201);
    // The encoder halves the frequency axis once, and the decoders double it
    // back with a sub-pixel convolution.
    assert_eq!(CORE_WIDTH, 101);
    assert_eq!(CORE_WIDTH * 2 - 1, BINS);
}

#[test]
fn frames_and_samples_are_inverse_where_they_can_be() {
    // A centred analysis produces one frame per hop plus one, and the
    // synthesis gives back all but the last partial hop.
    for len in [0, 1, 99, 100, 101, 16_000, 128_000] {
        let count = frames(len);
        assert_eq!(count, len / HOP + 1);
        assert!(samples(count) <= len);
        assert!(len - samples(count) < HOP);
    }
}

#[test]
fn a_window_is_longer_than_the_overlap_it_shares() {
    assert!(WINDOW_SECONDS > OVERLAP_SECONDS);
    // Consecutive windows advance by this much; the redundancy is the ratio.
    let hop = WINDOW_SECONDS - OVERLAP_SECONDS;
    assert_eq!(hop, 7);
    let redundancy = WINDOW_SECONDS as f64 / hop as f64;
    assert!((redundancy - 1.142_857).abs() < 1e-5);
}

#[test]
fn compression_and_its_inverse_come_back_to_the_start() {
    for value in [1e-6f64, 0.01, 1.0, 3.7, 1e3] {
        let there = value.powf(COMPRESS);
        let back = there.powf(1.0 / COMPRESS);
        assert!(
            (back - value).abs() <= value * 1e-9,
            "{value} → {there} → {back}"
        );
    }
}
