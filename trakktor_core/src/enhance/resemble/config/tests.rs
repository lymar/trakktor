use super::*;

#[test]
fn the_vocoder_upsamples_by_exactly_one_hop() {
    // Every conditioning frame must become exactly one hop of waveform, or the
    // vocoder's output and the mel it was conditioned on would drift apart.
    let product: usize = STRIDES.iter().product();
    assert_eq!(product, HOP);
}

#[test]
fn the_transform_is_four_hops_wide() {
    assert_eq!(N_FFT, 1680);
    assert_eq!(BINS, 841);
}

#[test]
fn the_vocoder_takes_the_mel_and_its_extra_channels() {
    assert_eq!(VOCODER_INPUT, MELS + VOCODER_EXTRA);
}

#[test]
fn a_chunk_is_thirty_seconds_and_shares_one() {
    assert_eq!(chunk_samples(), 1_323_000);
    assert_eq!(overlap_samples(), 44_100);
    assert!(overlap_samples() < chunk_samples());
}

#[test]
fn the_unet_can_halve_both_axes_as_many_times_as_it_has_blocks() {
    assert_eq!(UNET_ALIGN, 1 << UNET_BLOCKS);
    // Whatever the input, the padded height divides down without a remainder.
    for height in [1usize, 100, BINS, 4096] {
        let padded = height.div_ceil(UNET_ALIGN) * UNET_ALIGN;
        let mut size = padded;
        for _ in 0..UNET_BLOCKS {
            assert_eq!(size % 2, 0, "{height} → {padded} does not halve");
            size /= 2;
        }
    }
}

#[test]
fn frames_drop_the_one_a_centred_analysis_adds() {
    // A centred analysis of `n` samples yields `1 + n/hop` frames and the
    // reference keeps `n/hop` of them.
    assert_eq!(frames(HOP), 1);
    assert_eq!(frames(HOP - 1), 0);
    assert_eq!(frames(10 * HOP + 7), 10);
}
