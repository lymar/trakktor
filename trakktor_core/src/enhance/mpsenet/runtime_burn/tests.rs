//! Parity of the burn runtime against the same reference dumps the candle one
//! is checked against. `#[ignore]` — they need the converted checkpoint in
//! `~/.trakktor/enhance/mpsenet/dns/` and the dumps in `tmp/mpsenet/golden/`.

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use super::BurnModel;
use crate::enhance::mpsenet::{config::BINS, stft};

fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/mpsenet/golden")
}

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/enhance/mpsenet/dns")
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

/// The worst absolute difference, and the relative size of it.
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (worst, worst / scale)
}

fn reference_spectrum() -> stft::Spectrum {
    let magnitude = read_f32("stft_mag");
    let phase = read_f32("stft_pha");
    let frames = magnitude.len() / BINS;
    stft::Spectrum {
        magnitude,
        phase,
        frames,
    }
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_heads_match_the_reference() {
    let spectrum = reference_spectrum();
    let model =
        BurnModel::<NdArray<f32>>::load(&model_dir(), NdArrayDevice::Cpu)
            .expect("the converted checkpoint");
    let prediction = model.predict(&spectrum);

    // The reference dumps the phase heads before it transposes them.
    let transpose = |values: &[f32]| -> Vec<f32> {
        let frames = spectrum.frames;
        (0..frames)
            .flat_map(|frame| {
                (0..BINS).map(move |bin| values[bin * frames + frame])
            })
            .collect()
    };
    for (name, got) in [
        ("mask", prediction.mask.clone()),
        ("phase_r", transpose(&prediction.phase_real)),
        ("phase_i", transpose(&prediction.phase_imag)),
    ] {
        let (worst, relative) = compare(name, &got, &read_f32(name));
        println!("{name}: worst {worst:.3e}, relative {relative:.3e}");
        assert!(relative < 1e-3, "{name}: relative error {relative}");
    }
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_waveform_matches_the_reference() {
    let spectrum = reference_spectrum();
    let model =
        BurnModel::<NdArray<f32>>::load(&model_dir(), NdArrayDevice::Cpu)
            .expect("the converted checkpoint");
    let got =
        stft::synthesize(&stft::apply(&spectrum, &model.predict(&spectrum)));
    let (worst, relative) =
        compare("denoised", &got, &read_f32("wave_normalized"));
    println!("the waveform: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-3, "the waveform: relative error {relative}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_two_runtimes_agree_with_each_other() {
    use candle_core::Device;

    use crate::enhance::{Precision, mpsenet::runtime::CandleModel};

    let spectrum = reference_spectrum();
    let burn =
        BurnModel::<NdArray<f32>>::load(&model_dir(), NdArrayDevice::Cpu)
            .expect("the converted checkpoint");
    let candle = CandleModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");

    let theirs = burn.predict(&spectrum);
    let ours = candle.predict(&spectrum).expect("the network runs");
    for (name, a, b) in [
        ("mask", &ours.mask, &theirs.mask),
        ("phase_r", &ours.phase_real, &theirs.phase_real),
        ("phase_i", &ours.phase_imag, &theirs.phase_imag),
    ] {
        let (worst, relative) = compare(name, a, b);
        println!("{name}: worst {worst:.3e}, relative {relative:.3e}");
        assert!(relative < 1e-3, "{name}: the runtimes differ by {relative}");
    }
}
