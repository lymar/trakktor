use super::{
    super::constants::{N_FRAMES, N_SAMPLES, SAMPLE_RATE},
    *,
};

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const PCM: &[u8] = include_bytes!("../testdata/sample_2s.pcm.bin");
const MEL80: &[u8] = include_bytes!("../testdata/sample_2s.mel80.bin");
const MEL128: &[u8] = include_bytes!("../testdata/sample_2s.mel128.bin");

#[test]
fn log_mel_matches_reference_golden_80() {
    let mel = log_mel_spectrogram(&le_f32(PCM), MelBands::Mel80, 0);
    assert_eq!(mel.n_mels(), 80);
    assert_eq!(mel.n_frames(), 200);
    let diff = max_abs_diff(mel.data(), &le_f32(MEL80));
    assert!(diff < 1e-3, "max abs diff {diff} exceeds tolerance");
}

#[test]
fn log_mel_matches_reference_golden_128() {
    let mel = log_mel_spectrogram(&le_f32(PCM), MelBands::Mel128, 0);
    assert_eq!(mel.n_mels(), 128);
    assert_eq!(mel.n_frames(), 200);
    let diff = max_abs_diff(mel.data(), &le_f32(MEL128));
    assert!(diff < 1e-3, "max abs diff {diff} exceeds tolerance");
}

#[test]
fn hann_window_is_periodic() {
    let w = hann_window();
    assert!(w[0].abs() < 1e-6);
    assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6);
    for k in 1..N_FFT / 2 {
        assert!((w[N_FFT / 2 - k] - w[N_FFT / 2 + k]).abs() < 1e-6);
    }
}

#[test]
fn frame_count_follows_hop() {
    let audio = vec![0.0f32; N_SAMPLES];
    let mel = log_mel_spectrogram(&audio, MelBands::Mel80, 0);
    assert_eq!(mel.n_frames(), N_FRAMES);
}

#[test]
fn reflect_pad_matches_numpy_semantics() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        reflect_pad(&x, 2),
        vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
    );
}

#[test]
fn window_slices_and_zero_pads_to_encoder_width() {
    let mel = log_mel_spectrogram(&le_f32(PCM), MelBands::Mel80, 0);
    // 150 real columns remain from frame 50 of 200; the rest is padding.
    let window = mel.window(50, N_FRAMES);
    assert_eq!(window.n_mels(), 80);
    assert_eq!(window.data().len(), 80 * N_FRAMES);
    for m in 0..mel.n_mels() {
        for f in 0..150 {
            assert_eq!(window.data()[m * N_FRAMES + f], mel.get(m, 50 + f));
        }
        for f in 150..N_FRAMES {
            assert_eq!(window.data()[m * N_FRAMES + f], 0.0);
        }
    }
}

#[test]
fn window_trims_to_encoder_width() {
    // 31 s of silence: more mel frames than one encoder window holds.
    let audio = vec![0.0f32; N_SAMPLES + SAMPLE_RATE];
    let mel = log_mel_spectrogram(&audio, MelBands::Mel80, 0);
    assert!(mel.n_frames() > N_FRAMES);
    let window = mel.window(0, mel.n_frames());
    assert_eq!(window.data().len(), 80 * N_FRAMES);
}
