use super::*;

/// One packet of speech-like samples, then one of digital silence, and so on.
fn alternating(packets: usize) -> Vec<f32> {
    let packet = 320;
    (0..packets * packet)
        .map(|index| {
            if (index / packet) % 2 == 0 {
                0.3 * ((index as f32) * 0.05).sin()
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn silence_reads_as_loss_and_speech_does_not() {
    let flags = lost_frames(&alternating(6));
    assert_eq!(flags, vec![false, true, false, true, false, true]);
}

#[test]
fn a_partial_packet_at_the_end_is_not_judged() {
    let mut samples = alternating(2);
    samples.extend(std::iter::repeat_n(0.0, 100));
    assert_eq!(lost_frames(&samples).len(), 2);
}

#[test]
fn a_window_shorter_than_a_packet_has_no_frames() {
    assert!(lost_frames(&[0.0; 100]).is_empty());
}

#[test]
fn one_loud_sample_in_a_hundred_still_leaves_the_packet_lost() {
    // The rule is 99% of samples below the floor, not all of them.
    let mut samples = vec![0.0f32; 320];
    samples[7] = 0.9;
    assert_eq!(lost_frames(&samples), vec![true]);
}

#[test]
fn four_loud_samples_keep_the_packet() {
    let mut samples = vec![0.0f32; 320];
    for index in 0..4 {
        samples[index * 11] = 0.9;
    }
    assert_eq!(lost_frames(&samples), vec![false]);
}
