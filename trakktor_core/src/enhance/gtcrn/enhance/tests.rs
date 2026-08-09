use super::*;

const SECOND: usize = SAMPLE_RATE as usize;

#[test]
fn a_short_recording_is_one_chunk() {
    assert_eq!(chunks(frames(10 * SECOND)), vec![frames(10 * SECOND)]);
}

#[test]
fn chunks_partition_the_recording_exactly() {
    // No overlap to arrange: the state carries, so the pieces simply follow
    // one another.
    let total = frames(100 * SECOND);
    let spans = chunks(total);
    assert_eq!(spans.iter().sum::<usize>(), total);
    assert_eq!(spans.len(), 4);
    for &count in &spans[..spans.len() - 1] {
        assert_eq!(count, 30 * SECOND / HOP);
    }
}

#[test]
fn an_empty_recording_has_no_chunks() {
    assert!(chunks(0).is_empty());
}
