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
fn streamed_pushes_match_batch_exactly() {
    // Feeding the same signal in odd-sized blocks must give bit-identical
    // probabilities: streaming buffers windows into the same encoder-chunk
    // batches as the whole-buffer path.
    let vad = Vad::load().unwrap();
    // ~3.2 s of a deterministic signal with structure (chirp + noise-ish).
    let audio: Vec<f32> = (0..51_000)
        .map(|n| {
            let t = n as f32 / 16_000.0;
            (t * (200.0 + 40.0 * t) * std::f32::consts::TAU).sin() * 0.25 +
                ((n as f32 * 12.9898).sin() * 43758.547).fract() * 0.02
        })
        .collect();
    let batch = vad.probabilities(&audio).unwrap();

    let mut state = VadStreamState::new();
    let mut streamed = Vec::new();
    // Deliberately awkward block sizes, none aligned to the 512 window.
    let mut pos = 0;
    for (i, step) in [1usize, 511, 513, 7, 4096, 100_000]
        .iter()
        .cycle()
        .enumerate()
    {
        if pos >= audio.len() {
            break;
        }
        let end = (pos + step + i % 3).min(audio.len());
        streamed.extend(vad.stream_push(&mut state, &audio[pos..end]).unwrap());
        pos = end;
    }
    streamed.extend(vad.stream_finish(&mut state).unwrap());

    assert_eq!(streamed.len(), batch.len());
    let identical = streamed
        .iter()
        .zip(&batch)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(identical, "streamed probabilities must be bit-identical");
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
