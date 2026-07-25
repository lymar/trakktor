//! Tests for joining synthesized pieces into one waveform.

use super::*;

/// A piece: `silence` samples of near-silence, then `speech` samples of full
/// level, then the same silence again — the shape the models actually return.
fn piece(silence: usize, speech: usize) -> Speech {
    let mut samples = vec![1e-5; silence];
    samples.extend(std::iter::repeat_n(0.5, speech));
    samples.extend(std::iter::repeat_n(1e-5, silence));
    Speech {
        samples,
        sample_rate: 100,
    }
}

/// No fade, no edge padding: the sample-level arithmetic is easier to state.
const BARE: Join = Join {
    fade: 0.0,
    edge: 0.0,
};

#[test]
fn the_pause_between_pieces_is_the_one_asked_for() {
    // Each piece carries 20 samples (0.2 s at 100 Hz) of its own silence per
    // side; the caller asked for 0.5 s between them, and that is what must
    // land there — not 0.5 s plus whatever the model left.
    let pieces = [piece(20, 50), piece(20, 30)];

    let joined = stitch(&pieces, &[0.5], BARE);

    assert_eq!(joined.sample_rate, 100);
    assert_eq!(joined.samples.len(), 50 + 50 + 30);
    assert!(joined.samples[50..100].iter().all(|s| *s == 0.0));
    assert_eq!(joined.samples[49], 0.5);
    assert_eq!(joined.samples[100], 0.5);
}

#[test]
fn a_piece_keeps_a_breath_of_silence_at_its_edges() {
    let pieces = [piece(20, 50)];

    let joined = stitch(
        &pieces,
        &[],
        Join {
            fade: 0.0,
            edge: 0.05,
        },
    );

    // Five samples of the original silence are kept on each side.
    assert_eq!(joined.samples.len(), 5 + 50 + 5);
    assert_eq!(joined.samples[0], 1e-5);
    assert_eq!(joined.samples[5], 0.5);
}

#[test]
fn the_joins_are_faded_so_they_cannot_click() {
    let pieces = [piece(20, 50)];

    let joined = stitch(
        &pieces,
        &[],
        Join {
            fade: 0.05,
            edge: 0.0,
        },
    );

    assert_eq!(joined.samples.len(), 50);
    // Both ends ramp from zero; the middle is untouched.
    assert_eq!(joined.samples[0], 0.0);
    assert_eq!(joined.samples[49], 0.0);
    assert!(joined.samples[1] < 0.5 && joined.samples[1] > 0.0);
    assert_eq!(joined.samples[25], 0.5);
}

#[test]
fn a_silent_piece_contributes_nothing_but_its_pause() {
    let pieces = [piece(20, 40), piece(30, 0)];

    let joined = stitch(&pieces, &[0.1], BARE);

    assert_eq!(joined.samples.len(), 40 + 10);
}

#[test]
fn nothing_to_join_is_no_audio() {
    let joined = stitch(&[], &[], Join::default());

    assert!(joined.samples.is_empty());
    assert_eq!(joined.duration(), 0.0);
}
