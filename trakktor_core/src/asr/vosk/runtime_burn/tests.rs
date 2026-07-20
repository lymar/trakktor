//! burn encoder parity against the reference dumps and cross-parity with
//! candle, CPU f32.
//!
//! Model files and golden dumps live outside the repo (`tmp/vosk/`), so these
//! tests are `#[ignore]` and run on demand.

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use super::VoskBurnModel;
use crate::asr::vosk::{
    decode::{Decoding, TransducerHead},
    runtime::{EncoderSeam, Precision, VoskModel},
    stream::transcribe,
    tokenizer::Tokenizer,
    transcribe::TranscribeOptions,
    weights,
};

fn model_dir(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/vosk/models")
        .join(name)
}

fn golden_dir(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/vosk/golden")
        .join(name)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The burn offline encoder matches onnxruntime, and matches the candle
/// runtime bit-closely on CPU f32.
#[test]
#[ignore = "needs tmp/vosk/models/ru and tmp/vosk/golden/ru"]
fn ru_encoder_matches_reference_and_candle() {
    let w = weights::load_dir(&model_dir("ru")).unwrap();
    let reference = read_f32(&golden_dir("ru").join("clip.encoder.bin"));
    let pcm = read_f32(&golden_dir("ru").join("clip.pcm.bin"));

    let burn =
        VoskBurnModel::<NdArray<f32>>::load(&w, NdArrayDevice::Cpu).unwrap();
    let features = burn.feature().compute(&pcm);
    let (burn_out, _) = burn.encode(&features).unwrap();
    let diff = max_abs_diff(&burn_out, &reference);
    println!("burn vs onnxruntime max abs diff: {diff}");
    assert!(diff < 1e-3, "burn vs reference max abs diff {diff}");

    let candle = VoskModel::load_cpu(&w, Precision::F32).unwrap();
    let (candle_out, _) = candle.encode(&features).unwrap();
    let cross = max_abs_diff(&burn_out, &candle_out);
    println!("burn vs candle max abs diff: {cross}");
    assert!(cross < 1e-4, "burn vs candle max abs diff {cross}");
}

/// The burn streaming encoder matches onnxruntime, driven chunk by chunk.
#[test]
#[ignore = "needs tmp/vosk/models/small-streaming-ru and its golden"]
fn small_streaming_encoder_matches_reference() {
    let w = weights::load_dir(&model_dir("small-streaming-ru")).unwrap();
    let reference =
        read_f32(&golden_dir("small-streaming-ru").join("clip.encoder.bin"));
    let pcm = read_f32(&golden_dir("small-streaming-ru").join("clip.pcm.bin"));

    let mut padded = pcm.clone();
    padded.extend(std::iter::repeat_n(0.0f32, 16000 * 8 / 10));
    let streaming = w.config.streaming.clone().unwrap();
    let burn =
        VoskBurnModel::<NdArray<f32>>::load(&w, NdArrayDevice::Cpu).unwrap();
    let fbank = burn.feature();
    let n_frames = fbank.n_frames(padded.len(), true);
    let mut session = burn.start_stream().unwrap();
    let mut out: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + streaming.window_frames < n_frames {
        let window =
            fbank.compute_range(&padded, 0, start, streaming.window_frames);
        let (encoded, _) = session.accept(&window).unwrap();
        out.extend_from_slice(&encoded);
        start += streaming.shift_frames;
    }
    assert_eq!(out.len(), reference.len());
    let diff = max_abs_diff(&out, &reference);
    println!("burn streaming vs onnxruntime max abs diff: {diff}");
    assert!(diff < 1e-3, "burn streaming max abs diff {diff}");
}

/// On the CPU in f32 the candle and burn runtimes produce a byte-identical
/// transcription — the whole pipeline, not just the encoder output.
#[test]
#[ignore = "needs tmp/vosk/models/ru and tmp/vosk/golden/ru"]
fn candle_and_burn_transcription_byte_identical() {
    let w = weights::load_dir(&model_dir("ru")).unwrap();
    let tokens =
        std::fs::read_to_string(model_dir("ru").join("tokens.txt")).unwrap();
    let tokenizer = Tokenizer::parse(&tokens).unwrap();
    let head = TransducerHead::load(&w, tokenizer.unk_id()).unwrap();
    let pcm = read_f32(&golden_dir("ru").join("clip.pcm.bin"));

    let candle = VoskModel::load_cpu(&w, Precision::F32).unwrap();
    let burn =
        VoskBurnModel::<NdArray<f32>>::load(&w, NdArrayDevice::Cpu).unwrap();

    for decoding in [Decoding::Greedy, Decoding::Beam { max_active: 10 }] {
        let options = TranscribeOptions {
            word_timestamps: true,
            decoding,
        };
        let c = transcribe(&candle, &head, &tokenizer, &pcm, &options).unwrap();
        let b = transcribe(&burn, &head, &tokenizer, &pcm, &options).unwrap();
        assert_eq!(c.text, b.text, "text for {decoding:?}");
        assert_eq!(c.segments.len(), b.segments.len());
        for (cs, bs) in c.segments.iter().zip(&b.segments) {
            assert_eq!(cs.text, bs.text);
            assert_eq!(cs.words, bs.words);
        }
    }
}
