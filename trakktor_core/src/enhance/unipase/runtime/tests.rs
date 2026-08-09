//! Parity against the Python reference, stage by stage. `#[ignore]` — they need
//! the converted checkpoint in `~/.trakktor/enhance/unipase/` and the reference
//! dumps in `tmp/enhance/golden/`.
//!
//! The pipeline has no sampling in it at all: one waveform in, one waveform
//! out, no seed and no solver. So these are not "close enough on average"
//! tests — the reference's own input is fed in and every stage is compared
//! against what the reference produced from it. What is left over is the
//! arithmetic: a different order of summation in the convolutions, the
//! matmuls, and the transform.

use candle_core::{DType, Device, Tensor};

use super::CandleModel;
use crate::enhance::{
    EnhanceModel, Precision,
    unipase::{
        config::{HOP, SAMPLE_RATE},
        istft, plc,
    },
};

/// Where the reference dumps live.
fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/enhance/golden")
}

/// Where the converted checkpoint lives.
fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/enhance/unipase")
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = golden_dir().join(format!("{name}.f32"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_flags(name: &str) -> Vec<bool> {
    let path = golden_dir().join(format!("{name}.u8"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes.into_iter().map(|byte| byte != 0).collect()
}

/// The worst absolute difference, and the relative size of it.
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("{name}: worst {worst:.3e}, relative {:.3e}", worst / scale);
    (worst, worst / scale)
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    let (worst, relative) = compare(name, got, want);
    assert!(
        relative < tolerance,
        "{name}: worst {worst:.3e} is {relative:.3e} of the scale, over \
         {tolerance:.0e}"
    );
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("reading a tensor back")
}

fn load() -> CandleModel {
    CandleModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint")
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_encoder_matches_the_reference_stage_by_stage() {
    let model = load();
    let padded = read_f32("padded");
    let lost = read_flags("mask_indices");
    let input =
        Tensor::from_slice(&padded, (1, padded.len()), model.device()).unwrap();
    let stages = model.encoder.stages(&input, &lost).unwrap();

    assert_close(
        "conv_out",
        &values(&stages.conv_out),
        &read_f32("conv_out"),
        1e-5,
    );
    assert_close(
        "conv_norm",
        &values(&stages.conv_norm),
        &read_f32("conv_norm"),
        1e-5,
    );
    assert_close(
        "projected",
        &values(&stages.projected),
        &read_f32("projected"),
        1e-5,
    );
    assert_close("masked", &values(&stages.masked), &read_f32("masked"), 1e-5);
    assert_close(
        "pos_conv",
        &values(&stages.positioned),
        &read_f32("layer0"),
        1e-5,
    );
    assert_close(
        "layer1",
        &values(&stages.layers[0]),
        &read_f32("layer1"),
        1e-4,
    );
    assert_close(
        "layer24",
        &values(&stages.layers[23]),
        &read_f32("layer24"),
        1e-4,
    );
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_taps_match_the_reference() {
    let model = load();
    let padded = read_f32("padded");
    let lost = read_flags("mask_indices");
    let input =
        Tensor::from_slice(&padded, (1, padded.len()), model.device()).unwrap();
    let (acoustic, phonetic) = model.encoder.features(&input, &lost).unwrap();
    assert_close("feat_a", &values(&acoustic), &read_f32("feat_a"), 1e-4);
    assert_close("feat_p", &values(&phonetic), &read_f32("feat_p"), 1e-4);
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_adapter_and_the_vocoder_match_the_reference() {
    let model = load();
    let device = model.device();
    let frames = read_f32("feat_a").len() / 1024;
    let tap = |name: &str| {
        Tensor::from_slice(&read_f32(name), (1, frames, 1024), device).unwrap()
    };
    let acoustic = tap("feat_a");
    let phonetic = tap("feat_p");

    let projected = model.adapter.project(&phonetic).unwrap();
    assert_close(
        "adapter_proj",
        &values(&projected),
        &read_f32("adapter_proj"),
        1e-4,
    );

    // The backbone on its own, so a mismatch downstream can be told from one
    // inside the twelve ConvNeXt blocks.
    let summed = (projected + &acoustic).unwrap();
    let backbone = model
        .adapter
        .backbone()
        .forward(&summed.transpose(1, 2).unwrap().contiguous().unwrap())
        .unwrap();
    assert_close(
        "adapter_backbone",
        &values(&backbone),
        &read_f32("adapter_backbone"),
        1e-4,
    );

    let adapted = model.adapter.forward(&acoustic, &phonetic).unwrap();
    assert_close("adapted", &values(&adapted), &read_f32("adapted"), 1e-4);

    let vocoder_backbone = model
        .vocoder
        .backbone()
        .forward(&adapted.transpose(1, 2).unwrap().contiguous().unwrap())
        .unwrap();
    assert_close(
        "vocoder_backbone",
        &values(&vocoder_backbone),
        &read_f32("vocoder_backbone"),
        1e-4,
    );

    let head = model.vocoder.spectrum(&adapted).unwrap();
    assert_close(
        "vocoder_head_out",
        &head,
        &read_f32("vocoder_head_out"),
        1e-4,
    );

    let wave = istft::spectrum_to_wave(&head, frames);
    assert_close("enhanced_16k", &wave, &read_f32("enhanced_16k"), 1e-3);
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_whole_window_matches_the_reference() {
    let mut model = load();
    let padded = read_f32("padded");
    let lost = plc::lost_frames(&padded);
    assert_eq!(lost, read_flags("mask_indices"), "the detector disagrees");
    let wave = model.enhance_window(&padded, &lost).unwrap();
    assert_eq!(wave.len(), padded.len() / HOP * HOP);
    assert_close("pipeline", &wave, &read_f32("pipeline_16k"), 1e-3);
}

#[test]
#[ignore = "needs the converted checkpoint"]
fn a_window_comes_back_the_length_the_frames_say() {
    let mut model = load();
    let samples = vec![0.01f32; 4 * SAMPLE_RATE as usize + 80];
    let wave = model.enhance_window(&samples, &[]).unwrap();
    assert_eq!(wave.len(), (samples.len() / HOP) * HOP);
}
