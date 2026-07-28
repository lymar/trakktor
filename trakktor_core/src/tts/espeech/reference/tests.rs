use super::prepare_wave;
use crate::tts::espeech::config::{
    REF_MAX_SECONDS, REF_TAIL_SILENCE, SAMPLE_RATE, TARGET_RMS,
};

/// A tone of `seconds` at `level`, as a stand-in for speech.
fn tone(seconds: f64, level: f32) -> Vec<f32> {
    let samples = (seconds * f64::from(SAMPLE_RATE)) as usize;
    (0..samples)
        .map(|index| (index as f32 * 0.05).sin() * level)
        .collect()
}

fn silence(seconds: f64) -> Vec<f32> {
    vec![0.0; (seconds * f64::from(SAMPLE_RATE)) as usize]
}

#[test]
fn silence_at_the_edges_is_trimmed_and_a_tail_appended() {
    let mut wave = silence(0.5);
    wave.extend(tone(2.0, 0.3));
    wave.extend(silence(0.7));
    let prepared = prepare_wave(wave);
    let expected = 2.0 + REF_TAIL_SILENCE;
    assert!(
        (prepared.seconds() - expected).abs() < 0.05,
        "kept {:.3} s, expected about {expected:.3}",
        prepared.seconds()
    );
    assert!(!prepared.clipped);
}

#[test]
fn a_quiet_reference_is_brought_up_and_remembers_by_how_much() {
    // Quiet, but above the −42 dBFS the trimmer treats as silence: a signal
    // below that line is not quiet speech, it *is* silence, and is trimmed.
    let prepared = prepare_wave(tone(2.0, 0.04));
    assert!(prepared.rms < TARGET_RMS);
    let gain = prepared.output_gain();
    assert!(gain < 1.0, "gain {gain}");
    // The normalized waveform sits at the target, so undoing the gain returns
    // the original loudness.
    let rms: f32 = {
        let sum: f64 = prepared
            .wave
            .iter()
            .map(|s| f64::from(*s) * f64::from(*s))
            .sum();
        (sum / prepared.wave.len() as f64).sqrt() as f32
    };
    assert!((rms - TARGET_RMS).abs() < 0.02, "normalized to {rms}");
}

#[test]
fn a_loud_reference_is_left_alone() {
    let prepared = prepare_wave(tone(2.0, 0.9));
    assert!(prepared.rms > TARGET_RMS);
    assert_eq!(prepared.output_gain(), 1.0);
}

#[test]
fn a_long_reference_is_cut_at_a_pause_inside_the_window() {
    let mut wave = tone(8.0, 0.3);
    wave.extend(silence(0.3));
    wave.extend(tone(8.0, 0.3));
    let prepared = prepare_wave(wave);
    assert!(prepared.clipped);
    // The cut lands at the pause, not at the twelve-second mark.
    let spoken = prepared.seconds() - REF_TAIL_SILENCE;
    assert!((8.0..8.4).contains(&spoken), "kept {spoken:.3} s of speech");
}

#[test]
fn a_long_reference_without_a_pause_is_cut_at_the_window() {
    let prepared = prepare_wave(tone(20.0, 0.3));
    assert!(prepared.clipped);
    let spoken = prepared.seconds() - REF_TAIL_SILENCE;
    assert!(
        (spoken - REF_MAX_SECONDS).abs() < 0.05,
        "kept {spoken:.3} s of speech"
    );
}

#[test]
fn the_frame_count_the_duration_estimate_uses_is_one_short_of_the_spectrogram()
{
    // The centred transform adds a frame; the reference's cut does not know
    // about it, and the port has to keep that asymmetry.
    let prepared = prepare_wave(tone(2.0, 0.3));
    let frames =
        crate::tts::espeech::mel::MelBasis::frames_for(prepared.wave.len());
    assert_eq!(prepared.cut_frames() + 1, frames);
}
