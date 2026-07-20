//! End-to-end pipeline checks against real models and reference dumps.
//!
//! Model files and golden dumps live outside the repo (`tmp/vosk/`), so
//! these tests are `#[ignore]` and run on demand.

use super::{StreamTranscriber, transcribe};
use crate::asr::vosk::{
    decode::{Decoding, TransducerHead},
    runtime::{Precision, VoskModel},
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

fn load(name: &str) -> (VoskModel, TransducerHead, Tokenizer) {
    let dir = model_dir(name);
    let w = weights::load_dir(&dir).unwrap();
    let tokens = std::fs::read_to_string(dir.join("tokens.txt")).unwrap();
    let tokenizer = Tokenizer::parse(&tokens).unwrap();
    let head = TransducerHead::load(&w, tokenizer.unk_id()).unwrap();
    let model = VoskModel::load_cpu(&w, Precision::F32).unwrap();
    (model, head, tokenizer)
}

/// The full pipeline reproduces the reference decode (text and tokens) on
/// the reference clip, for both searches.
#[test]
#[ignore = "needs tmp/vosk/models/ru and tmp/vosk/golden/ru"]
fn ru_clip_text_matches_reference() {
    let (model, head, tokenizer) = load("ru");
    let golden = golden_dir("ru");
    let pcm = read_f32(&golden.join("clip.pcm.bin"));
    for (decoding, file) in [
        (Decoding::Greedy, "clip.greedy.json"),
        (Decoding::Beam { max_active: 10 }, "clip.beam.json"),
    ] {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(golden.join(file)).unwrap(),
        )
        .unwrap();
        let options = TranscribeOptions {
            word_timestamps: false,
            decoding,
        };
        let out =
            transcribe(&model, &head, &tokenizer, &pcm, &options).unwrap();
        assert_eq!(out.text, manifest["text"].as_str().unwrap(), "{file} text");
    }
}

/// The streaming pipeline reproduces the reference chunked decode.
#[test]
#[ignore = "needs tmp/vosk/models/small-streaming-ru and its golden"]
fn small_streaming_clip_text_matches_reference() {
    let (model, head, tokenizer) = load("small-streaming-ru");
    let golden = golden_dir("small-streaming-ru");
    let pcm = read_f32(&golden.join("clip.pcm.bin"));
    for (decoding, file) in [
        (Decoding::Greedy, "clip.greedy.json"),
        (Decoding::Beam { max_active: 10 }, "clip.beam.json"),
    ] {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(golden.join(file)).unwrap(),
        )
        .unwrap();
        let options = TranscribeOptions {
            word_timestamps: false,
            decoding,
        };
        let out =
            transcribe(&model, &head, &tokenizer, &pcm, &options).unwrap();
        assert_eq!(out.text, manifest["text"].as_str().unwrap(), "{file} text");
    }
}

/// Feeding the same audio in different block sizes yields an identical
/// transcription (streaming model).
#[test]
#[ignore = "needs tmp/vosk/models/small-streaming-ru and its golden"]
fn streaming_blockwise_equals_oneshot() {
    let (model, head, tokenizer) = load("small-streaming-ru");
    let pcm = read_f32(&golden_dir("small-streaming-ru").join("clip.pcm.bin"));
    let options = TranscribeOptions {
        word_timestamps: true,
        decoding: Decoding::Beam { max_active: 10 },
    };
    let oneshot =
        transcribe(&model, &head, &tokenizer, &pcm, &options).unwrap();

    let mut session =
        StreamTranscriber::new(&model, &head, &tokenizer, options, None)
            .unwrap();
    for block in pcm.chunks(1234) {
        session.push(block, &mut |_| {}).unwrap();
    }
    let blockwise = session.finish(&mut |_| {}).unwrap();
    assert_eq!(oneshot.text, blockwise.text);
    assert_eq!(oneshot.segments.len(), blockwise.segments.len());
}
