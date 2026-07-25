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
    match_levels: false,
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
            match_levels: false,
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
            match_levels: false,
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

/// A piece of `speech` samples at a fixed level, framed by silence.
fn piece_at(level: f32, speech: usize) -> Speech {
    let mut samples = vec![1e-5; 10];
    samples
        .extend((0..speech).map(|i| if i % 2 == 0 { level } else { -level }));
    samples.extend(std::iter::repeat_n(1e-5, 10));
    Speech {
        samples,
        sample_rate: 100,
    }
}

/// The loudness of a stitched piece, measured the way the matcher does.
fn loudness(samples: &[f32]) -> f32 {
    let speech: Vec<f32> =
        samples.iter().copied().filter(|s| s.abs() > 1e-3).collect();
    (speech.iter().map(|s| s * s).sum::<f32>() / speech.len() as f32).sqrt()
}

#[test]
fn pieces_are_brought_to_a_common_level() {
    // Three pieces 6 dB apart: 0.1 is the median and stays put, the others
    // move to meet it.
    let pieces = [piece_at(0.05, 200), piece_at(0.1, 200), piece_at(0.2, 200)];

    let joined = stitch(
        &pieces,
        &[0.0, 0.0],
        Join {
            fade: 0.0,
            edge: 0.0,
            match_levels: true,
        },
    );

    let levels: Vec<f32> = joined.samples.chunks(200).map(loudness).collect();
    for level in &levels {
        assert!(
            (level - 0.1).abs() < 0.005,
            "piece landed at {level}, expected the median 0.1"
        );
    }
}

#[test]
fn matching_never_pushes_a_piece_into_clipping() {
    // The quiet piece would need +20 dB to reach the loud one; the shift is
    // capped, and the loud piece cannot be pushed past full scale either.
    let pieces = [
        piece_at(0.95, 200),
        piece_at(0.95, 200),
        piece_at(0.01, 200),
    ];

    let joined = stitch(
        &pieces,
        &[0.0, 0.0],
        Join {
            fade: 0.0,
            edge: 0.0,
            match_levels: true,
        },
    );

    let peak = joined.samples.iter().fold(0.0f32, |p, s| p.max(s.abs()));
    assert!(peak <= 1.0, "peak {peak} clips");
    // The outlier is lifted by the 6 dB the cap allows, no further.
    let quiet = loudness(&joined.samples[400..600]);
    assert!(
        (quiet - 0.02).abs() < 0.002,
        "the quiet piece landed at {quiet}, expected +6 dB from 0.01"
    );
}

#[test]
fn keeping_the_levels_leaves_the_samples_alone() {
    let pieces = [piece_at(0.05, 200), piece_at(0.2, 200)];

    let joined = stitch(&pieces, &[0.0], BARE);

    assert!((loudness(&joined.samples[..200]) - 0.05).abs() < 1e-6);
    assert!((loudness(&joined.samples[200..]) - 0.2).abs() < 1e-6);
}

#[test]
fn silence_is_not_amplified() {
    // A piece with no speech has no level to match; it must not be scaled by
    // some accidental ratio.
    let pieces = [piece_at(0.1, 200), piece(30, 0)];

    let joined = stitch(
        &pieces,
        &[0.0],
        Join {
            fade: 0.0,
            edge: 0.0,
            match_levels: true,
        },
    );

    assert_eq!(joined.samples.len(), 200);
}
