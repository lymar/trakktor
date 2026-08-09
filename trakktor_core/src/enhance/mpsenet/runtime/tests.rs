//! Parity against the Python reference, stage by stage. `#[ignore]` — they
//! need the converted checkpoint in `~/.trakktor/enhance/mpsenet/dns/` and the
//! reference dumps in `tmp/mpsenet/golden/`.
//!
//! There is no sampling anywhere in this network: one waveform in, one
//! waveform out, no seed and no solver. So these are not "close enough on
//! average" tests — the reference's own input is fed in and every stage is
//! compared against what the reference produced from it. What is left over is
//! the arithmetic: a different order of summation in the convolutions, the
//! matmuls and the transform.

use candle_core::{DType, Device, Tensor};

use super::CandleModel;
use crate::enhance::{
    EnhanceModel, Precision,
    mpsenet::{config::BINS, stft},
};

/// Where the reference dumps live.
fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/mpsenet/golden")
}

/// Where the converted checkpoint lives.
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

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("a stage as f32 values")
}

/// The reference's own input, and the spectrum it analysed from it.
fn reference_input() -> (Vec<f32>, stft::Spectrum) {
    let normalized = read_f32("normalized");
    let magnitude = read_f32("stft_mag");
    let phase = read_f32("stft_pha");
    let frames = magnitude.len() / BINS;
    (
        normalized,
        stft::Spectrum {
            magnitude,
            phase,
            frames,
        },
    )
}

/// The difference between two angles, wrapped into `[0, π]`.
fn angle_gap(a: f32, b: f32) -> f32 {
    let full = 2.0 * std::f32::consts::PI;
    let delta = (a - b).abs() % full;
    delta.min(full - delta)
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_analysis_matches_the_reference() {
    let (normalized, want) = reference_input();
    let got = stft::analyze(&normalized);
    assert_eq!(got.frames, want.frames);
    let (_, magnitude) = compare("stft_mag", &got.magnitude, &want.magnitude);
    println!("the magnitude: relative {magnitude:.3e}");
    assert!(magnitude < 1e-5, "magnitude: {magnitude}");

    // The phase is an angle, so a difference only counts once it is wrapped —
    // and at two frames of every window it can only be compared that way. A
    // centred analysis mirrors the signal at each end, which makes the first
    // and last frames symmetric about their own centre; the transform of a
    // symmetric frame is real, its imaginary part is rounding noise, and the
    // reference's `atan2` of that noise lands on +π or −π by luck of which FFT
    // computed it. That is a property of the reference's own formulation, not
    // a difference between it and this port.
    let worst = got
        .phase
        .iter()
        .zip(&want.phase)
        .map(|(&a, &b)| angle_gap(a, b))
        .fold(0f32, f32::max);
    // A tenth of a milliradian, and it comes from the same guard: adding 1e-5
    // to a real part that is itself near zero turns a rounding difference in
    // the transform into a visible rotation of that one bin.
    println!("the phase, wrapped: worst {worst:.3e} rad");
    assert!(worst < 1e-3, "phase, wrapped: {worst}");

    let unwrapped = got
        .phase
        .iter()
        .zip(&want.phase)
        .enumerate()
        .filter(|&(_, (&a, &b))| (a - b).abs() > 1e-3)
        .map(|(index, _)| index % got.frames)
        .collect::<std::collections::BTreeSet<_>>();
    println!("frames where the sign of ±π differs: {unwrapped:?}");
    assert!(
        unwrapped
            .iter()
            .all(|&frame| frame == 0 || frame + 1 == got.frames),
        "only the two mirrored frames may flip"
    );
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn every_stage_matches_the_reference() {
    let (_, spectrum) = reference_input();
    let model = CandleModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");
    let stages = model
        .net()
        .stages(&spectrum.magnitude, &spectrum.phase, spectrum.frames)
        .expect("the network runs");

    let mut worst_overall = 0f32;
    for (name, tensor) in &stages {
        let want = read_f32(name);
        let (worst, relative) = compare(name, &host(tensor), &want);
        println!("{name}: worst {worst:.3e}, relative {relative:.3e}");
        assert!(relative < 1e-3, "{name}: relative error {relative}");
        worst_overall = worst_overall.max(relative);
    }
    println!("worst relative error over every stage: {worst_overall:.3e}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_waveform_matches_the_reference_from_its_own_spectrum() {
    // The network and the synthesis, on exactly what the reference analysed —
    // so what is measured here is the port and nothing else.
    let (_, spectrum) = reference_input();
    let model = CandleModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");
    let prediction = model.predict(&spectrum).expect("the network runs");
    let got = stft::synthesize(&stft::apply(&spectrum, &prediction));

    let (worst, relative) =
        compare("denoised", &got, &read_f32("wave_normalized"));
    println!("the waveform: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-3, "the waveform: relative error {relative}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_whole_window_matches_the_reference() {
    // The same thing end to end, this time analysing the samples here rather
    // than borrowing the reference's spectrum. The bound is looser on purpose:
    // this run carries the ±π coin-flip of the two mirrored frames — see
    // `the_analysis_matches_the_reference` — into the network's input, and the
    // network has no idea that a phase is circular.
    let (normalized, _) = reference_input();
    let mut model =
        CandleModel::load(&model_dir(), Device::Cpu, Precision::F32)
            .expect("the converted checkpoint");
    let got = model
        .enhance_window(&normalized, &[])
        .expect("the window is enhanced");
    let want = read_f32("wave_normalized");
    let (worst, relative) = compare("wave", &got, &want);
    println!("the waveform: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-2, "the waveform: relative error {relative}");
}
