//! burn-runtime tests: hermetic equivalence checks of the burn-specific
//! shapes (fused q/k/v, the cat-grown cache, the beam fold), and `#[ignore]`
//! parity tests that need the developer-local checkpoint
//! (`tmp/models/whisper-tiny/`) and the goldens of
//! `scripts/asr/whisper/gen_model_golden.py` — the same fixtures as the
//! candle runtime's tests.

use std::path::PathBuf;

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};

use super::{
    BurnModel,
    net::{CrossAttention, LayerNorm, Linear, SelfAttention, sinusoids},
};
use crate::asr::whisper::{
    audio::pad_or_trim,
    constants::{N_FRAMES, N_SAMPLES},
    feature::{MelBands, MelWindow, log_mel_spectrogram},
    model::{ForwardProvider, Logits},
    tokenizer::TokenId,
};

type B = NdArray<f32>;

const DEV: NdArrayDevice = NdArrayDevice::Cpu;

/// A tiny deterministic value sequence for hermetic layer tests.
fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 23) as f32 / 23.0 * scale + offset)
        .collect()
}

fn t3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(values, shape), &DEV)
}

fn to_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn linear(in_dim: usize, out_dim: usize, salt: f32) -> Linear<B> {
    Linear {
        weight: Tensor::from_data(
            TensorData::new(
                ramp(in_dim * out_dim, 1.0, -0.5 + salt),
                [in_dim, out_dim],
            ),
            &DEV,
        ),
        bias: Some(Tensor::from_data(
            TensorData::new(ramp(out_dim, 0.3, salt), [out_dim]),
            &DEV,
        )),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn sinusoids_start_with_zero_sin_and_unit_cos() {
    let s = sinusoids(4, 6);
    assert_eq!(s.len(), 24);
    // Position zero: all sines are 0, all cosines are 1.
    assert_eq!(&s[..6], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    // Later positions must not repeat position zero.
    assert!(s[6..12].iter().zip(&s[..6]).any(|(a, b)| a != b));
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

/// Builds a [`SelfAttention`] whose fused q/k/v weight is assembled from
/// three separate matrices, plus those matrices as stand-alone linears.
fn self_attention_fixture(
    d: usize,
    n_head: usize,
) -> (SelfAttention<B>, [Linear<B>; 3]) {
    let (q, k, v) = (linear(d, d, 0.1), linear(d, d, 0.2), linear(d, d, 0.3));
    let k = Linear { bias: None, ..k }; // the key projection has no bias
    let parts: Vec<Vec<f32>> = [&q, &k, &v]
        .iter()
        .map(|l| to_vec(l.weight.clone()))
        .collect();
    let mut fused = Vec::with_capacity(3 * d * d);
    for row in 0..d {
        for part in &parts {
            fused.extend_from_slice(&part[row * d..(row + 1) * d]);
        }
    }
    let mut bias = to_vec(q.bias.clone().unwrap());
    bias.extend(std::iter::repeat_n(0.0f32, d));
    bias.extend(to_vec(v.bias.clone().unwrap()));
    let attn = SelfAttention {
        qkv: Linear {
            weight: Tensor::from_data(TensorData::new(fused, [d, 3 * d]), &DEV),
            bias: Some(Tensor::from_data(TensorData::new(bias, [3 * d]), &DEV)),
        },
        out: linear(d, d, 0.4),
        n_head,
    };
    (attn, [q, k, v])
}

/// Manual per-row reference attention over projected q/k/v, causal when
/// asked: softmax((q·kᵀ)/sqrt(d_head)) · v, all in plain host f32.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_k: usize,
    n_head: usize,
    d: usize,
    causal_offset: Option<usize>,
) -> Vec<f32> {
    let dh = d / n_head;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0f32; n_q * d];
    for head in 0..n_head {
        for row in 0..n_q {
            let limit = match causal_offset {
                Some(offset) => offset + row + 1,
                None => n_k,
            };
            let mut scores = vec![f32::NEG_INFINITY; n_k];
            for col in 0..limit {
                let mut dot = 0.0f32;
                for i in 0..dh {
                    dot +=
                        q[row * d + head * dh + i] * k[col * d + head * dh + i];
                }
                scores[col] = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exp: Vec<f32> =
                scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for i in 0..dh {
                let mut acc = 0.0f32;
                for col in 0..limit {
                    acc += exp[col] / sum * v[col * d + head * dh + i];
                }
                out[row * d + head * dh + i] = acc;
            }
        }
    }
    out
}

/// Applies a stand-alone [`Linear`] to a flat row-major `[t, d]` input.
fn apply_linear(l: &Linear<B>, x: &[f32], t: usize, d_in: usize) -> Vec<f32> {
    to_vec(l.forward(t3(x.to_vec(), [1, t, d_in])))
}

#[test]
fn fused_qkv_attention_matches_reference() {
    let (d, h, t) = (8usize, 2usize, 5usize);
    let (attn, [q, k, v]) = self_attention_fixture(d, h);
    let x = ramp(t * d, 2.0, -1.0);

    let got = to_vec(attn.forward_plain(t3(x.clone(), [1, t, d]), true));

    let (qp, kp, vp) = (
        apply_linear(&q, &x, t, d),
        apply_linear(&k, &x, t, d),
        apply_linear(&v, &x, t, d),
    );
    let ctx = reference_attention(&qp, &kp, &vp, t, t, h, d, Some(0));
    let want = apply_linear(&attn.out, &ctx, t, d);
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 1e-4, "fused attention diverges: {diff}");
}

#[test]
fn cached_steps_match_full_forward() {
    let (d, h) = (8usize, 2usize);
    let (attn, _) = self_attention_fixture(d, h);
    let (b, t_total) = (2usize, 5usize);
    let x = ramp(b * t_total * d, 2.0, -1.0);

    let full = to_vec(attn.forward_plain(t3(x.clone(), [b, t_total, d]), true));

    // The session shape: the first step feeds three positions, the rest one.
    let mut cache = None;
    let mut got = vec![0.0f32; b * t_total * d];
    let mut offset = 0usize;
    for n_step in [3usize, 1, 1] {
        // Rows of the step input: each batch row's positions
        // offset..offset+n_step.
        let mut step = Vec::with_capacity(b * n_step * d);
        for row in 0..b {
            let start = (row * t_total + offset) * d;
            step.extend_from_slice(&x[start..start + n_step * d]);
        }
        let out = to_vec(attn.forward_cached(
            t3(step, [b, n_step, d]),
            &mut cache,
            offset,
        ));
        for row in 0..b {
            let start = (row * t_total + offset) * d;
            got[start..start + n_step * d].copy_from_slice(
                &out[row * n_step * d..(row + 1) * n_step * d],
            );
        }
        offset += n_step;
    }

    let diff = max_abs_diff(&got, &full);
    assert!(diff < 1e-5, "cached forward diverges from full: {diff}");
}

/// Builds a [`CrossAttention`] from deterministic weights.
fn cross_attention_fixture(d: usize, n_head: usize) -> CrossAttention<B> {
    CrossAttention {
        q: linear(d, d, 0.15),
        k: Linear {
            bias: None,
            ..linear(d, d, 0.25)
        },
        v: linear(d, d, 0.35),
        out: linear(d, d, 0.45),
        n_head,
    }
}

#[test]
fn folded_cross_attention_matches_capture_path() {
    let (d, h, frames) = (8usize, 2usize, 4usize);
    let cross = cross_attention_fixture(d, h);
    let features = t3(ramp(frames * d, 1.5, -0.7), [1, frames, d]);
    let kv = cross.project_kv(&features);

    let (b, t) = (3usize, 2usize);
    let x = ramp(b * t * d, 2.0, -1.0);
    let folded = to_vec(cross.forward_folded(t3(x.clone(), [b, t, d]), &kv));

    // The capture path computes attention manually; row by row it must agree
    // with the folded fused call.
    for row in 0..b {
        let row_x = x[row * t * d..(row + 1) * t * d].to_vec();
        let (out, _) = cross.forward_capture(t3(row_x, [1, t, d]), &kv);
        let want = to_vec(out);
        let got = &folded[row * t * d..(row + 1) * t * d];
        let diff = max_abs_diff(got, &want);
        assert!(diff < 1e-4, "fold row {row} diverges: {diff}");
    }
}

// ---------------------------------------------------------------------------
// Opt-in parity tests against the local checkpoint and the candle runtime.
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

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

fn load_tiny() -> BurnModel<B> {
    BurnModel::<B>::load(&repo_path("tmp/models/whisper-tiny"), DEV)
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

    let mut model = load_tiny();
    let features = model.encode(&window).unwrap();
    let ours = to_vec(features);
    let want = read_f32("tmp/whisper_golden/tiny_encoder.bin");

    // Same judgement as the candle test: assert the speech region (the
    // fixture's first 2 s ≈ 100 encoder positions); in the silent tail layer
    // norm amplifies noise on near-constant frames, so it is reported but
    // not asserted.
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
    let mut model = load_tiny();
    let features = model.encode(&fixture_window()).unwrap();
    model.begin_decode(1, &features).unwrap();
    let logits = model.decode_step(&SOT_EN_TRANSCRIBE, 1).unwrap();
    model.end_decode();

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
    let mut model = load_tiny();
    let features = model.encode(&fixture_window()).unwrap();

    model.begin_decode(1, &features).unwrap();
    let full = model.decode_step(&SOT_EN_TRANSCRIBE, 1).unwrap();
    model.end_decode();

    model.begin_decode(1, &features).unwrap();
    let steps: Vec<Logits> = SOT_EN_TRANSCRIBE
        .iter()
        .map(|&t| model.decode_step(&[t], 1).unwrap())
        .collect();
    model.end_decode();

    for (position, step) in steps.iter().enumerate() {
        let diff = max_abs_diff(full.row(0, position), step.row(0, 0));
        assert!(
            diff < 1e-4,
            "position {position}: cache-vs-full max abs diff {diff}"
        );
    }
}

#[test]
#[ignore = "requires the local checkpoint"]
fn burn_matches_candle() {
    use crate::asr::whisper::runtime::{CandleRuntime, Precision};

    let mut burn_model = load_tiny();
    let mut candle_model = CandleRuntime::load(
        &repo_path("tmp/models/whisper-tiny"),
        candle_core::Device::Cpu,
        Precision::F32,
    )
    .expect("loading the local whisper-tiny checkpoint");

    let window = fixture_window();

    // Encoder outputs, both row-major [1500, d]; judged like the reference
    // parity test — tightly in the speech region, reported in the tail.
    let burn_features = burn_model.encode(&window).unwrap();
    let candle_features = candle_model.encode(&window).unwrap();
    let burn_encoded = to_vec(burn_features.clone());
    let candle_encoded = candle_features
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n_state = 384;
    let speech = 100 * n_state;
    let enc_speech =
        max_abs_diff(&burn_encoded[..speech], &candle_encoded[..speech]);
    let enc_tail =
        max_abs_diff(&burn_encoded[speech..], &candle_encoded[speech..]);
    println!(
        "burn-vs-candle encoder max abs diff: speech {enc_speech}, tail \
         {enc_tail}"
    );
    assert!(enc_speech < 5e-3, "encoder outputs diverge: {enc_speech}");

    // The timing forward: logits and raw cross-attention scores.
    let (burn_logits, burn_qk) = burn_model
        .forward_with_cross_qk(&SOT_EN_TRANSCRIBE, &burn_features)
        .unwrap();
    let (candle_logits, candle_qk) = candle_model
        .forward_with_cross_qk(&SOT_EN_TRANSCRIBE, &candle_features)
        .unwrap();
    let logits_diff = max_abs_diff(
        burn_logits.last_position(0),
        candle_logits.last_position(0),
    );
    println!("burn-vs-candle sot logits max abs diff: {logits_diff}");
    assert!(logits_diff < 2e-2, "logits diverge: {logits_diff}");

    assert_eq!(burn_qk.n_layers(), candle_qk.n_layers());
    assert_eq!(burn_qk.n_heads(), candle_qk.n_heads());
    let mut qk_diff = 0.0f32;
    for layer in 0..burn_qk.n_layers() {
        for head in 0..burn_qk.n_heads() {
            qk_diff = qk_diff.max(max_abs_diff(
                burn_qk.head(layer, head),
                candle_qk.head(layer, head),
            ));
        }
    }
    println!("burn-vs-candle cross-qk max abs diff: {qk_diff}");
    assert!(qk_diff < 2e-2, "cross-attention scores diverge: {qk_diff}");
}
