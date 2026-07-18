//! Runtime parity and loading tests. `#[ignore]` — they need the reference
//! checkpoint cached at `~/.cache/gigaam/` and the dumps in `tmp/gigaam/`.

use candle_core::Device;

use super::{GigaamModel, Precision};
use crate::asr::gigaam::{
    config::{Attention, ConvNorm, EncoderConfig, Subsampling},
    feature::MelConfig,
};

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

/// v3_ctc geometry.
const V3_ENCODER: EncoderConfig = EncoderConfig {
    n_mels: 64,
    d_model: 768,
    n_layers: 16,
    n_heads: 16,
    subsampling: Subsampling::Conv1d,
    subs_kernel_size: 5,
    subsampling_factor: 4,
    conv_kernel_size: 5,
    conv_norm: ConvNorm::LayerNorm,
    attention: Attention::Rotary,
};

const V3_MEL: MelConfig = MelConfig {
    n_fft: 320,
    hop_length: 160,
    n_mels: 64,
    center: false,
};

/// multilingual_large_ctc geometry (the largest model: 1024-wide, 24 layers).
const LARGE_ENCODER: EncoderConfig = EncoderConfig {
    n_mels: 64,
    d_model: 1024,
    n_layers: 24,
    n_heads: 16,
    subsampling: Subsampling::Conv1d,
    subs_kernel_size: 5,
    subsampling_factor: 4,
    conv_kernel_size: 5,
    conv_norm: ConvNorm::LayerNorm,
    attention: Attention::Rotary,
};

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
    let device = Device::Cpu;
    let model = GigaamModel::load_ctc(
        &ckpt("v3_ctc"),
        V3_ENCODER,
        V3_MEL,
        34,
        device,
        Precision::F32,
    )
    .unwrap();

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
    use crate::asr::gigaam::{config::config_for, decode::decode_chunk};

    let device = Device::Cpu;
    // v3_e2e_ctc uses the v3 encoder geometry with a SentencePiece head.
    let model = GigaamModel::load_ctc(
        &ckpt("v3_e2e_ctc"),
        V3_ENCODER,
        V3_MEL,
        257, // 256 pieces + blank
        device,
        Precision::F32,
    )
    .unwrap();

    // Tokenizer comes from the embedded config (SentencePiece).
    let tokenizer = config_for("v3_e2e_ctc").unwrap().tokenizer.build();

    let dir = golden_dir("v3_e2e_ctc");
    let trace: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("clip_40_12.trace.json")).unwrap(),
    )
    .unwrap();

    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let encoded = model.encode(&mel).unwrap();
    let enc_len = encoded.dims()[0];
    let logits = model.ctc_logits(&encoded).unwrap();
    let labels: Vec<u32> = logits
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    let decoded = decode_chunk(&tokenizer, &labels, enc_len);
    println!("e2e decoded: {:?}", decoded.text);
    let golden = trace["text"].as_str().unwrap();
    // The reference text carries punctuation and capitalization.
    assert!(
        golden.contains('.') || golden.contains(','),
        "golden has punctuation"
    );
    assert_eq!(decoded.text, golden, "e2e text with punctuation");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/multilingual_large_ctc.ckpt and its dump"]
fn large_ctc_decode_matches_reference() {
    use crate::asr::gigaam::{decode::decode_chunk, tokenizer::Tokenizer};

    let device = Device::Cpu;
    let model = GigaamModel::load_ctc(
        &ckpt("multilingual_large_ctc"),
        LARGE_ENCODER,
        V3_MEL, // same mel geometry
        71,
        device,
        Precision::F32,
    )
    .unwrap();

    let dir = golden_dir("multilingual_large_ctc");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vocab: Vec<String> = config["vocab"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let tokenizer = Tokenizer::charwise(vocab);
    let trace: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("clip_40_12.trace.json")).unwrap(),
    )
    .unwrap();

    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let encoded = model.encode(&mel).unwrap();
    assert_eq!(encoded.dims(), &[300, 1024]);
    let enc_len = encoded.dims()[0];
    let logits = model.ctc_logits(&encoded).unwrap();
    let labels: Vec<u32> = logits
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    let decoded = decode_chunk(&tokenizer, &labels, enc_len);
    println!("large decoded: {:?}", decoded.text);
    assert_eq!(decoded.text, trace["text"].as_str().unwrap(), "large text");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn ctc_decode_matches_reference() {
    use crate::asr::gigaam::{
        decode::{decode_chunk, frames_to_words},
        tokenizer::Tokenizer,
    };

    let device = Device::Cpu;
    let model = GigaamModel::load_ctc(
        &ckpt("v3_ctc"),
        V3_ENCODER,
        V3_MEL,
        34,
        device,
        Precision::F32,
    )
    .unwrap();

    let dir = golden_dir("v3_ctc");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let vocab: Vec<String> = config["vocab"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let tokenizer = Tokenizer::charwise(vocab);

    let trace: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("clip_40_12.trace.json")).unwrap(),
    )
    .unwrap();

    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));
    let mel = model.feature().log_mel(&pcm);
    let encoded = model.encode(&mel).unwrap();
    let enc_len = encoded.dims()[0];
    let logits = model.ctc_logits(&encoded).unwrap();
    let labels: Vec<u32> = logits
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();

    let decoded = decode_chunk(&tokenizer, &labels, enc_len);
    let golden_text = trace["text"].as_str().unwrap();
    assert_eq!(decoded.text, golden_text, "decoded text");

    let golden_ids: Vec<u32> = trace["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(decoded.token_ids, golden_ids, "token ids");

    // Word timestamps.
    let shift = pcm.len() as f64 / 16000.0 / enc_len as f64;
    let words = frames_to_words(
        &tokenizer,
        &decoded.token_ids,
        &decoded.token_frames,
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
