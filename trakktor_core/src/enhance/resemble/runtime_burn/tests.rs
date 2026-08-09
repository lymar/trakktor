//! Parity of the burn runtime, against the reference and against candle.
//! `#[ignore]` — they need the converted checkpoint and the reference dumps.
//!
//! The denoiser is checked on the CPU backend, which needs nothing to be
//! present. **The enhancer is checked on Metal**, and that is a measured
//! decision rather than a preference: burn's `ndarray` backend convolves in one
//! thread, and one four-second chunk of this pipeline takes it tens of minutes.
//! The CPU path is left working — it is the same code — but nothing is verified
//! through it at that price.

use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    wgpu::{Metal, WgpuDevice},
};

use super::{BurnDenoiser, BurnEnhancer};
use crate::enhance::{
    EnhanceModel, Precision,
    resemble::{EnhancerSettings, noise::Source, runtime::DenoiserModel},
};

/// The backend the denoiser runs on: the CPU one, which needs no device.
type Cpu = NdArray<f32>;

fn golden_dir(mode: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/resemble")
        .join(format!("golden-{mode}"))
}

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/enhance/resemble")
}

fn read_f32(mode: &str, name: &str) -> Vec<f32> {
    let path = golden_dir(mode).join(format!("{name}.f32"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    worst / scale
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_denoiser_matches_the_reference() {
    let chunk = read_f32("denoise", "chunk_in");
    let mut model = BurnDenoiser::<Cpu>::load(&model_dir(), NdArrayDevice::Cpu)
        .expect("the converted checkpoint");
    let got = model
        .enhance_window(&chunk, &[])
        .expect("the chunk is enhanced");
    let relative =
        compare("chunk_out", &got, &read_f32("denoise", "chunk_out"));
    println!("the denoiser on burn: relative {relative:.3e}");
    assert!(relative < 1e-3, "the denoiser on burn: {relative}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_two_denoiser_runtimes_agree_with_each_other() {
    let chunk = read_f32("denoise", "chunk_in");
    let mut burn = BurnDenoiser::<Cpu>::load(&model_dir(), NdArrayDevice::Cpu)
        .expect("the converted checkpoint");
    let mut candle = DenoiserModel::load(
        &model_dir(),
        crate::enhance::resemble::runtime::Device::Cpu,
        Precision::F32,
    )
    .expect("the converted checkpoint");
    let from_burn = burn.enhance_window(&chunk, &[]).expect("burn runs");
    let from_candle = candle.enhance_window(&chunk, &[]).expect("candle runs");
    let relative = compare("chunk_out", &from_burn, &from_candle);
    println!("the two denoiser runtimes: relative {relative:.3e}");
    assert!(relative < 1e-4, "the two runtimes: {relative}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_enhancer_matches_the_reference() {
    // Fed the reference's own two draws, as the candle parity test is.
    let chunk = read_f32("enhance", "chunk_in");
    let mut model = BurnEnhancer::<Metal<f32>>::load(
        &model_dir(),
        WgpuDevice::default(),
        EnhancerSettings::default(),
    )
    .expect("the converted checkpoint");
    model.with_noise(Source::Scripted(vec![
        read_f32("enhance", "noise_0"),
        read_f32("enhance", "noise_1"),
    ]));
    let got = model
        .enhance_window(&chunk, &[])
        .expect("the chunk is enhanced");
    let relative =
        compare("chunk_out", &got, &read_f32("enhance", "chunk_out"));
    // The same bound the candle runtime is held to, and for the same reason:
    // the vocoder amplifies what it is conditioned on.
    println!("the enhancer on burn: relative {relative:.3e}");
    assert!(relative < 1e-2, "the enhancer on burn: {relative}");
}
