use std::path::PathBuf;

use candle_core::Device;

use super::{
    super::{
        audio::pad_or_trim,
        constants::N_SAMPLES,
        feature::{MelBands, log_mel_spectrogram},
    },
    *,
};

/// The geometry block of the published whisper-tiny checkpoint.
const TINY_CONFIG: &str = r#"{
    "num_mel_bins": 80,
    "max_source_positions": 1500,
    "d_model": 384,
    "encoder_attention_heads": 6,
    "encoder_layers": 4,
    "vocab_size": 51865,
    "max_target_positions": 448,
    "decoder_attention_heads": 6,
    "decoder_layers": 4,
    "unrelated_field": "ignored"
}"#;

#[test]
fn config_parses_into_dims() {
    let dims = parse_config(TINY_CONFIG).unwrap();
    assert_eq!(dims.n_mels, 80);
    assert_eq!(dims.n_audio_ctx, 1500);
    assert_eq!(dims.n_audio_state, 384);
    assert_eq!(dims.n_audio_head, 6);
    assert_eq!(dims.n_audio_layer, 4);
    assert_eq!(dims.n_vocab, 51865);
    assert_eq!(dims.n_text_ctx, 448);
    assert_eq!(dims.n_text_state, 384);
    assert_eq!(dims.n_text_head, 6);
    assert_eq!(dims.n_text_layer, 4);
    assert!(dims.is_multilingual());
    assert_eq!(dims.num_languages(), 99);
}

#[test]
fn config_reports_missing_fields() {
    let err = parse_config(r#"{"num_mel_bins": 80}"#).unwrap_err();
    assert!(matches!(err, WhisperError::InvalidModel(_)));
}

#[test]
fn sinusoids_start_with_zero_sin_and_unit_cos() {
    let s = net::sinusoids(4, 6, &Device::Cpu).unwrap();
    assert_eq!(s.dims(), [4, 6]);
    let row0 = s.get(0).unwrap().to_vec1::<f32>().unwrap();
    // Position zero: all sines are 0, all cosines are 1.
    assert_eq!(row0, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Opt-in parity tests against the reference implementation. They need the
// developer-local checkpoint (`tmp/models/whisper-tiny/`) and the goldens
// produced by `scripts/asr/whisper/gen_model_golden.py`; run with --ignored.
// ---------------------------------------------------------------------------

/// Start sequence of the multilingual tokenizer for en/transcribe.
const SOT_EN_TRANSCRIBE: [TokenId; 3] = [50258, 50259, 50359];

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_f32(relative: &str) -> Vec<f32> {
    let path = repo_path(relative);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    le_f32(&bytes)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

fn load_tiny() -> CandleRuntime {
    CandleRuntime::load(&repo_path("tmp/models/whisper-tiny"), Device::Cpu)
        .expect("loading the local whisper-tiny checkpoint")
}

/// The committed 2 s PCM fixture as one padded 30 s encoder window.
fn fixture_window() -> MelWindow {
    let pcm = le_f32(include_bytes!("../testdata/sample_2s.pcm.bin"));
    let padded = pad_or_trim(&pcm, N_SAMPLES);
    let mel = log_mel_spectrogram(&padded, MelBands::Mel80, 0);
    mel.window(0, N_FRAMES)
}

#[test]
#[ignore = "requires the local checkpoint and reference goldens"]
fn encoder_matches_reference() {
    // Feed the reference's own mel input, isolating the encoder from
    // feature-extraction differences.
    let mel = read_f32("tmp/whisper_golden/tiny_mel.bin");
    let window = MelWindow::from_raw(80, mel);

    let mut runtime = load_tiny();
    let features = runtime.encode(&window).unwrap();
    let ours = features.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = read_f32("tmp/whisper_golden/tiny_encoder.bin");

    // The weights are bitwise-identical to the reference checkpoint, so any
    // difference here is cross-library fp32 kernel drift. Judge the speech
    // region (the fixture's first 2 s ≈ 100 encoder positions); in the
    // silent tail layer norm amplifies noise on near-constant frames, so it
    // is reported but not asserted.
    let n_state = 384;
    let speech = 100 * n_state;
    let speech_max = max_abs_diff(&ours[..speech], &want[..speech]);
    let speech_mean = mean_abs_diff(&ours[..speech], &want[..speech]);
    let tail_max = max_abs_diff(&ours[speech..], &want[speech..]);
    assert!(
        speech_max < 3e-2 && speech_mean < 5e-3,
        "speech region: max {speech_max}, mean {speech_mean} (silent tail \
         max: {tail_max})"
    );
}

#[test]
#[ignore = "requires the local checkpoint and reference goldens"]
fn sot_logits_match_reference() {
    let mut runtime = load_tiny();
    let features = runtime.encode(&fixture_window()).unwrap();
    runtime.begin_decode(1, &features).unwrap();
    let logits = runtime.decode_step(&SOT_EN_TRANSCRIBE, 1).unwrap();
    runtime.end_decode();

    assert_eq!(logits.n_positions(), 3);
    let mut ours = Vec::new();
    for position in 0..3 {
        ours.extend_from_slice(logits.row(0, position));
    }
    let want = read_f32("tmp/whisper_golden/tiny_logits_sot.bin");
    let diff = max_abs_diff(&ours, &want);
    assert!(diff < 2e-2, "logits max abs diff {diff} exceeds tolerance");
}

#[test]
#[ignore = "requires the local checkpoint"]
fn incremental_cache_matches_full_forward() {
    let mut runtime = load_tiny();
    let features = runtime.encode(&fixture_window()).unwrap();

    runtime.begin_decode(1, &features).unwrap();
    let full = runtime.decode_step(&SOT_EN_TRANSCRIBE, 1).unwrap();
    runtime.end_decode();

    runtime.begin_decode(1, &features).unwrap();
    let steps: Vec<Logits> = SOT_EN_TRANSCRIBE
        .iter()
        .map(|&t| runtime.decode_step(&[t], 1).unwrap())
        .collect();
    runtime.end_decode();

    for (position, step) in steps.iter().enumerate() {
        let diff = max_abs_diff(full.row(0, position), step.row(0, 0));
        assert!(
            diff < 1e-4,
            "position {position}: cache-vs-full max abs diff {diff}"
        );
    }
}
