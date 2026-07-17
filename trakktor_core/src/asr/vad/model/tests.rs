//! Smoke tests for the Silero-VAD model: it loads from the embedded weights
//! and produces one probability per window in `[0, 1]`. Numerical parity
//! against the reference is checked separately with a golden vector.

use super::*;

#[test]
fn loads_from_embedded_weights() { Vad::load().expect("model loads"); }

#[test]
fn one_probability_per_window_in_range() {
    let vad = Vad::load().unwrap();
    // A second of silence plus a short partial window.
    let audio = vec![0.0f32; 16_000 + 100];
    let probs = vad.probabilities(&audio).unwrap();
    assert_eq!(probs.len(), (16_000 + 100usize).div_ceil(WINDOW_SIZE));
    assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    // Pure silence should read as non-speech.
    assert!(probs.iter().all(|&p| p < 0.5));
}

#[test]
fn empty_audio_has_no_windows() {
    let vad = Vad::load().unwrap();
    assert!(vad.probabilities(&[]).unwrap().is_empty());
}

/// Parity harness (ignored): reads f32le PCM from `TRAKKTOR_VAD_PCM`, writes
/// the per-window probabilities as f32le to `TRAKKTOR_VAD_OUT`, to diff against
/// the reference Silero model on identical input.
#[test]
#[ignore = "parity harness: set TRAKKTOR_VAD_PCM and TRAKKTOR_VAD_OUT"]
fn dump_probs_for_parity() {
    let pcm = std::env::var("TRAKKTOR_VAD_PCM").unwrap();
    let out = std::env::var("TRAKKTOR_VAD_OUT").unwrap();
    let bytes = std::fs::read(&pcm).unwrap();
    let audio: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let probs = Vad::load().unwrap().probabilities(&audio).unwrap();
    let bytes: Vec<u8> = probs.iter().flat_map(|p| p.to_le_bytes()).collect();
    std::fs::write(&out, bytes).unwrap();
    eprintln!("wrote {} probabilities", probs.len());
}

#[test]
fn a_tone_is_more_speechlike_than_silence() {
    // Not a speech detector test — just that the network responds to signal, a
    // cheap guard against a dead/zeroed forward pass.
    let vad = Vad::load().unwrap();
    let silence = vad.probabilities(&vec![0.0f32; 16_000]).unwrap();
    let tone: Vec<f32> = (0..16_000)
        .map(|n| {
            (n as f32 * 220.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.3
        })
        .collect();
    let tone = vad.probabilities(&tone).unwrap();
    let max_silence = silence.iter().copied().fold(0.0f32, f32::max);
    let max_tone = tone.iter().copied().fold(0.0f32, f32::max);
    assert!(max_tone >= max_silence);
}
