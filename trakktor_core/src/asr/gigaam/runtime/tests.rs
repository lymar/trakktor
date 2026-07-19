//! Runtime parity and loading tests. `#[ignore]` — they need the reference
//! checkpoint cached at `~/.cache/gigaam/` and the dumps in `tmp/gigaam/`.

use candle_core::Device;

use super::{AsrModel, GigaamModel, Precision};
use crate::asr::gigaam::config::config_for;

fn ckpt(name: &str) -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home)
        .join(".cache/gigaam")
        .join(format!("{name}.ckpt"))
}

fn golden_dir(model: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/gigaam")
        .join(model)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i32(path: &std::path::Path) -> Vec<i32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads a model's checkpoint on the CPU in f32 (the parity target).
fn load_f32_cpu(name: &str) -> GigaamModel {
    let config = config_for(name).unwrap();
    GigaamModel::load(&ckpt(name), &config, Device::Cpu, Precision::F32)
        .unwrap()
}

fn read_trace(model: &str) -> serde_json::Value {
    let dir = golden_dir(model);
    serde_json::from_slice(
        &std::fs::read(dir.join("clip_40_12.trace.json")).unwrap(),
    )
    .unwrap()
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt"]
fn loads_state_dict() {
    let tensors =
        super::load_state_dict(&ckpt("v3_ctc"), &Device::Cpu).unwrap();
    assert_eq!(tensors.len(), 552);
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn encoder_and_ctc_match_reference() {
    let model = load_f32_cpu("v3_ctc");

    let dir = golden_dir("v3_ctc");
    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));

    // End-to-end: features (already validated) -> encoder -> CTC.
    let mel = model.feature().log_mel(&pcm);
    assert_eq!(mel.n_frames(), 1199);
    let encoded = model.encode(&mel).unwrap(); // [T'=300, D=768]
    assert_eq!(encoded.dims(), &[300, 768]);

    // Encoded output vs reference [D, T'].
    let golden_encoded = read_f32(&dir.join("clip_40_12.encoded.bin"));
    let ours_dt = encoded.transpose(0, 1).unwrap().flatten_all().unwrap();
    let ours_encoded = ours_dt.to_vec1::<f32>().unwrap();
    let enc_diff = ours_encoded
        .iter()
        .zip(&golden_encoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("encoder max abs diff: {enc_diff}");

    // CTC argmax labels must match exactly.
    let logits = model.ctc_logits(&encoded).unwrap(); // [300, 34]
    let labels: Vec<i32> = logits
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let golden_labels = read_i32(&dir.join("clip_40_12.labels.bin"));
    assert_eq!(labels.len(), golden_labels.len());
    let mismatches = labels
        .iter()
        .zip(&golden_labels)
        .filter(|(a, b)| a != b)
        .count();
    println!("label mismatches: {mismatches}/{}", labels.len());

    assert!(enc_diff < 0.2, "encoder max abs diff {enc_diff}");
    assert_eq!(mismatches, 0, "CTC labels must match the reference exactly");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_e2e_ctc.ckpt and its dump"]
fn e2e_ctc_decode_matches_reference_with_punctuation() {
    let model = load_f32_cpu("v3_e2e_ctc");

    // Tokenizer comes from the embedded config (SentencePiece).
    let tokenizer = config_for("v3_e2e_ctc").unwrap().tokenizer.build();

    let trace = read_trace("v3_e2e_ctc");
    let pcm = read_f32(&golden_dir("v3_e2e_ctc").join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let emitted = model.emissions(&mel).unwrap();
    let text = tokenizer.decode(&emitted.token_ids);
    println!("e2e decoded: {text:?}");
    let golden = trace["text"].as_str().unwrap();
    // The reference text carries punctuation and capitalization.
    assert!(
        golden.contains('.') || golden.contains(','),
        "golden has punctuation"
    );
    assert_eq!(text, golden, "e2e text with punctuation");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/multilingual_large_ctc.ckpt and its dump"]
fn large_ctc_decode_matches_reference() {
    let model = load_f32_cpu("multilingual_large_ctc");
    let tokenizer = config_for("multilingual_large_ctc")
        .unwrap()
        .tokenizer
        .build();

    let trace = read_trace("multilingual_large_ctc");
    let pcm = read_f32(
        &golden_dir("multilingual_large_ctc").join("clip_40_12.pcm.bin"),
    );
    let mel = model.feature().log_mel(&pcm);
    let encoded = model.encode(&mel).unwrap();
    assert_eq!(encoded.dims(), &[300, 1024]);
    let emitted = model.emissions(&mel).unwrap();
    let text = tokenizer.decode(&emitted.token_ids);
    println!("large decoded: {text:?}");
    assert_eq!(text, trace["text"].as_str().unwrap(), "large text");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn ctc_decode_matches_reference() {
    use crate::asr::gigaam::decode::frames_to_words;

    let model = load_f32_cpu("v3_ctc");
    let tokenizer = config_for("v3_ctc").unwrap().tokenizer.build();

    let trace = read_trace("v3_ctc");
    let pcm = read_f32(&golden_dir("v3_ctc").join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let emitted = model.emissions(&mel).unwrap();

    let text = tokenizer.decode(&emitted.token_ids);
    let golden_text = trace["text"].as_str().unwrap();
    assert_eq!(text, golden_text, "decoded text");

    let golden_ids: Vec<u32> = trace["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(emitted.token_ids, golden_ids, "token ids");

    // Word timestamps.
    let shift = pcm.len() as f64 / 16000.0 / emitted.enc_frames as f64;
    let words = frames_to_words(
        &tokenizer,
        &emitted.token_ids,
        &emitted.token_frames,
        shift,
    );
    let golden_words = trace["words"].as_array().unwrap();
    assert_eq!(words.len(), golden_words.len(), "word count");
    for (w, g) in words.iter().zip(golden_words) {
        assert_eq!(w.text, g["text"].as_str().unwrap());
        assert!((w.start - g["start"].as_f64().unwrap()).abs() < 1e-6);
        assert!((w.end - g["end"].as_f64().unwrap()).abs() < 1e-6);
    }
    println!("decoded {} words, text matches", words.len());
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_e2e_rnnt.ckpt and its dump"]
fn e2e_rnnt_decode_matches_reference_with_punctuation() {
    let model = load_f32_cpu("v3_e2e_rnnt");
    let tokenizer = config_for("v3_e2e_rnnt").unwrap().tokenizer.build();

    let trace = read_trace("v3_e2e_rnnt");
    let pcm = read_f32(&golden_dir("v3_e2e_rnnt").join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let emitted = model.emissions(&mel).unwrap();

    let golden_ids: Vec<u32> = trace["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(emitted.token_ids, golden_ids, "token ids");
    let golden_frames: Vec<usize> = trace["token_frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(emitted.token_frames, golden_frames, "token frames");

    let text = tokenizer.decode(&emitted.token_ids);
    println!("e2e rnnt decoded: {text:?}");
    let golden = trace["text"].as_str().unwrap();
    // The reference text is cased (this clip happens to carry no
    // punctuation marks).
    assert!(
        golden.chars().any(char::is_uppercase),
        "golden is capitalized"
    );
    assert_eq!(text, golden, "e2e rnnt text");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_rnnt.ckpt and tmp/gigaam/v3_rnnt dump"]
fn rnnt_decode_matches_reference() {
    use crate::asr::gigaam::decode::frames_to_words;

    let model = load_f32_cpu("v3_rnnt");
    let tokenizer = config_for("v3_rnnt").unwrap().tokenizer.build();

    let dir = golden_dir("v3_rnnt");
    let trace = read_trace("v3_rnnt");
    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);

    // The encoder is the v3 geometry; check it against the RNN-T dump too.
    let encoded = model.encode(&mel).unwrap();
    assert_eq!(encoded.dims(), &[300, 768]);
    let golden_encoded = read_f32(&dir.join("clip_40_12.encoded.bin"));
    let ours_dt = encoded.transpose(0, 1).unwrap().flatten_all().unwrap();
    let enc_diff = ours_dt
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .zip(&golden_encoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("rnnt encoder max abs diff: {enc_diff}");
    assert!(enc_diff < 0.2, "encoder max abs diff {enc_diff}");

    // Greedy transducer decode: tokens, frames, text, and words must match.
    let emitted = model.emissions(&mel).unwrap();
    let golden_ids: Vec<u32> = trace["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(emitted.token_ids, golden_ids, "token ids");
    let golden_frames: Vec<usize> = trace["token_frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(emitted.token_frames, golden_frames, "token frames");

    let text = tokenizer.decode(&emitted.token_ids);
    assert_eq!(text, trace["text"].as_str().unwrap(), "decoded text");

    let shift = pcm.len() as f64 / 16000.0 / emitted.enc_frames as f64;
    let words = frames_to_words(
        &tokenizer,
        &emitted.token_ids,
        &emitted.token_frames,
        shift,
    );
    let golden_words = trace["words"].as_array().unwrap();
    assert_eq!(words.len(), golden_words.len(), "word count");
    for (w, g) in words.iter().zip(golden_words) {
        assert_eq!(w.text, g["text"].as_str().unwrap());
        assert!((w.start - g["start"].as_f64().unwrap()).abs() < 1e-6);
        assert!((w.end - g["end"].as_f64().unwrap()).abs() < 1e-6);
    }
    println!("rnnt decoded {} words, text matches", words.len());
}
