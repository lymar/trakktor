use super::*;

#[test]
fn a_recording_shorter_than_one_hop_is_one_chunk() {
    let hop = chunk_samples() - overlap_samples();
    assert_eq!(chunk_starts(1), vec![0]);
    assert_eq!(chunk_starts(hop), vec![0]);
    assert_eq!(chunk_starts(hop + 1), vec![0, hop]);
}

#[test]
fn chunks_step_by_a_hop_and_cover_the_recording() {
    let span = chunk_samples();
    let hop = span - overlap_samples();
    for length in [1, hop, hop + 1, span, 3 * hop, 3 * hop + 12345] {
        let starts = chunk_starts(length);
        assert_eq!(starts[0], 0);
        // Every sample is inside some chunk.
        let last = *starts.last().expect("at least one chunk");
        assert!(
            last < length,
            "{length}: the last chunk starts past the end"
        );
        for pair in starts.windows(2) {
            assert_eq!(pair[1] - pair[0], hop);
        }
    }
}

#[test]
fn the_two_fades_sum_to_one_over_the_second_they_share() {
    let overlap = 1000;
    let len = 4000;
    let mut leaving = vec![1f32; len];
    let mut arriving = vec![1f32; len];
    fade_out(&mut leaving, overlap);
    fade_in(&mut arriving, overlap);
    // The tail of one window and the head of the next are the same stretch of
    // recording, so their weights have to add up to exactly one.
    for offset in 0..overlap {
        let sum = leaving[len - overlap + offset] + arriving[offset];
        assert!((sum - 1.0).abs() < 1e-6, "at {offset}: {sum}");
    }
    // And nothing outside the shared second is touched.
    assert!(leaving[..len - overlap].iter().all(|&w| w == 1.0));
    assert!(arriving[overlap..].iter().all(|&w| w == 1.0));
}

#[test]
fn a_fade_shorter_than_the_piece_it_is_asked_for_does_not_panic() {
    let mut piece = vec![1f32; 3];
    fade_in(&mut piece, 1000);
    fade_out(&mut piece, 1000);
    assert_eq!(piece.len(), 3);
    assert!(piece.iter().all(|w| w.is_finite()));
}

/// A model that hands the samples straight back, so the driver's own
/// arithmetic is what is being measured.
struct Echo;

impl EnhanceModel for Echo {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        Ok(samples.to_vec())
    }

    fn runtime(&self) -> crate::enhance::Runtime {
        crate::enhance::Runtime::Candle
    }
}

#[test]
fn a_transparent_model_returns_the_recording() {
    // Peak normalization, the tail padding, the fades and the overlap-add all
    // have to cancel exactly when the network does nothing.
    let span = chunk_samples();
    let hop = span - overlap_samples();
    let length = hop + span / 2;
    let input: Vec<f32> = (0..length)
        .map(|index| {
            let phase = index as f32 * 0.01;
            0.4 * phase.sin() + 0.1 * (phase * 7.3).cos()
        })
        .collect();
    let enhanced = enhance_samples(&input, &mut Echo, SAMPLE_RATE, &mut |_| {})
        .expect("the driver runs");
    assert_eq!(enhanced.samples.len(), length);
    assert_eq!(enhanced.windows, 2);
    let worst = enhanced
        .samples
        .iter()
        .zip(&input)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-5, "worst difference {worst}");
}

#[test]
fn silence_is_not_amplified() {
    let input = vec![0f32; 44_100];
    let enhanced = enhance_samples(&input, &mut Echo, SAMPLE_RATE, &mut |_| {})
        .expect("the driver runs");
    assert!(enhanced.samples.iter().all(|&sample| sample == 0.0));
}
