//! burn-runtime tests: hermetic checks of the hand-rolled layers, and
//! `#[ignore]` parity tests that need the reference checkpoint cached at
//! `~/.cache/gigaam/` and the dumps in `tmp/gigaam/` (the same fixtures as the
//! candle runtime's tests).

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};

use super::{
    GigaamBurnModel,
    net::{DepthwiseConv1d, LayerNorm},
};
use crate::asr::gigaam::{
    config::{Attention, ConvNorm, EncoderConfig, Subsampling},
    feature::MelConfig,
};

type B = NdArray<f32>;

const DEV: NdArrayDevice = NdArrayDevice::Cpu;

fn t3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(values, shape), &DEV)
}

fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// A tiny deterministic value sequence for hermetic layer tests.
fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 23) as f32 / 23.0 * scale + offset)
        .collect()
}

#[test]
fn depthwise_matches_direct_convolution() {
    let (channels, kernel, time) = (3usize, 5usize, 7usize);
    let padding = (kernel - 1) / 2;
    let x = ramp(2 * channels * time, 2.0, -1.0);
    let w = ramp(channels * kernel, 1.0, -0.5);
    let b = ramp(channels, 0.5, 0.1);

    let layer = DepthwiseConv1d::<B> {
        weight: Tensor::from_data(
            TensorData::new(w.clone(), [channels, 1, kernel]),
            &DEV,
        ),
        bias: Tensor::from_data(TensorData::new(b.clone(), [channels]), &DEV),
        padding,
    };
    let got = to_vec(layer.forward(t3(x.clone(), [2, channels, time])));

    // Direct per-channel convolution over the zero-padded input.
    let mut want = vec![0.0f32; 2 * channels * time];
    for batch in 0..2 {
        for c in 0..channels {
            for t in 0..time {
                let mut acc = b[c];
                for k in 0..kernel {
                    let src = t as isize + k as isize - padding as isize;
                    if (0..time as isize).contains(&src) {
                        let x_idx =
                            (batch * channels + c) * time + src as usize;
                        acc += w[c * kernel + k] * x[x_idx];
                    }
                }
                want[(batch * channels + c) * time + t] = acc;
            }
        }
    }
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-5, "depthwise mismatch: {g} vs {w}");
    }
}

#[test]
fn layer_norm_matches_direct_computation() {
    let d = 4usize;
    let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, 2.25];
    let gamma = vec![1.5f32, 1.0, 0.5, 2.0];
    let beta = vec![0.1f32, -0.1, 0.0, 0.2];
    let eps = 1e-5f64;

    let layer = LayerNorm::<B> {
        gamma: Tensor::from_data(
            TensorData::new(gamma.clone(), [1, 1, d]),
            &DEV,
        ),
        beta: Tensor::from_data(TensorData::new(beta.clone(), [1, 1, d]), &DEV),
        eps,
    };
    let got = to_vec(layer.forward(t3(x.clone(), [1, 2, d])));

    for row in 0..2 {
        let vals = &x[row * d..(row + 1) * d];
        let mean = vals.iter().sum::<f32>() / d as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() /
            d as f32;
        let std = (var + eps as f32).sqrt();
        for i in 0..d {
            let want = (vals[i] - mean) / std * gamma[i] + beta[i];
            let g = got[row * d + i];
            assert!((g - want).abs() < 1e-5, "ln mismatch: {g} vs {want}");
        }
    }
}

// --- Reference-parity tests (`#[ignore]`, shared fixtures) ---

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

fn load_v3() -> GigaamBurnModel<B> {
    GigaamBurnModel::<B>::load_ctc(&ckpt("v3_ctc"), V3_ENCODER, V3_MEL, 34, DEV)
        .unwrap()
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn encoder_and_ctc_match_reference() {
    let model = load_v3();
    let dir = golden_dir("v3_ctc");
    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));

    let mel = model.feature.log_mel(&pcm);
    assert_eq!(mel.n_frames(), 1199);
    let encoded = model.encode(&mel); // [T'=300, D=768]
    assert_eq!(encoded.dims(), [300, 768]);

    // Encoded output vs reference [D, T'].
    let golden_encoded = read_f32(&dir.join("clip_40_12.encoded.bin"));
    let ours_encoded = to_vec(encoded.clone().swap_dims(0, 1));
    let enc_diff = ours_encoded
        .iter()
        .zip(&golden_encoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("burn encoder max abs diff: {enc_diff}");

    // CTC argmax labels must match exactly.
    let logits = model.ctc_logits(encoded); // [300, 34]
    let labels: Vec<i32> = logits
        .argmax(1)
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap();
    let golden_labels = read_i32(&dir.join("clip_40_12.labels.bin"));
    assert_eq!(labels.len(), golden_labels.len());
    let mismatches = labels
        .iter()
        .zip(&golden_labels)
        .filter(|(a, b)| a != b)
        .count();
    println!("burn label mismatches: {mismatches}/{}", labels.len());

    assert!(enc_diff < 0.2, "encoder max abs diff {enc_diff}");
    assert_eq!(mismatches, 0, "CTC labels must match the reference exactly");
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn ctc_decode_matches_reference() {
    use crate::asr::gigaam::{
        decode::{decode_chunk, frames_to_words},
        runtime::CtcModel,
        tokenizer::Tokenizer,
    };

    let model = load_v3();
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
    let mel = model.feature.log_mel(&pcm);
    let labels = model.ctc_labels(&mel).unwrap();
    let enc_len = labels.len();

    let decoded = decode_chunk(&tokenizer, &labels, enc_len);
    assert_eq!(
        decoded.text,
        trace["text"].as_str().unwrap(),
        "decoded text"
    );

    let golden_ids: Vec<u32> = trace["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(decoded.token_ids, golden_ids, "token ids");

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
    println!("burn decoded {} words, text matches", words.len());
}

#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/gigaam/v3_ctc dump"]
fn burn_matches_candle() {
    use crate::asr::gigaam::runtime::{CtcModel, GigaamModel, Precision};

    let burn_model = load_v3();
    let candle_model = GigaamModel::load_ctc_cpu(
        &ckpt("v3_ctc"),
        V3_ENCODER,
        V3_MEL,
        34,
        Precision::F32,
    )
    .unwrap();

    let dir = golden_dir("v3_ctc");
    let pcm = read_f32(&dir.join("clip_40_12.pcm.bin"));
    let mel = burn_model.feature.log_mel(&pcm);

    // Encoder outputs, both row-major [T', D].
    let burn_encoded = to_vec(burn_model.encode(&mel));
    let candle_encoded = candle_model
        .encode(&mel)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(burn_encoded.len(), candle_encoded.len());
    let enc_diff = burn_encoded
        .iter()
        .zip(&candle_encoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("burn-vs-candle encoder max abs diff: {enc_diff}");
    assert!(enc_diff < 1e-3, "encoder outputs diverge: {enc_diff}");

    // Per-frame labels must agree exactly.
    let burn_labels = CtcModel::ctc_labels(&burn_model, &mel).unwrap();
    let candle_labels = CtcModel::ctc_labels(&candle_model, &mel).unwrap();
    assert_eq!(burn_labels, candle_labels, "argmax labels");
}
