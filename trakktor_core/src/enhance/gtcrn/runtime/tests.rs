//! Parity against the Python reference. `#[ignore]` — they need the converted
//! checkpoint in `~/.trakktor/enhance/gtcrn/` and the reference dumps in
//! `tmp/gtcrn/golden/`.
//!
//! The network is a plain function of its input — no sampling, no state carried
//! between calls — so the reference's own signal is fed in and what comes back
//! is compared with what the reference produced from it.

use super::CandleModel;
use crate::enhance::{EnhanceModel, gtcrn::stft};

fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/gtcrn/golden")
}

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/enhance/gtcrn")
}

fn read(name: &str) -> Vec<f32> {
    let path = golden_dir().join(format!("{name}.f32"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("{name}: worst {worst:.3e}, relative {:.3e}", worst / scale);
    assert!(
        worst / scale < tolerance,
        "{name}: worst {worst:.3e} is {:.3e} of the scale",
        worst / scale
    );
}

/// The dump stores a spectrum as `(bins, frames, 2)` interleaved.
fn split(name: &str) -> (Vec<f32>, Vec<f32>) {
    let want = read(name);
    let mut real = Vec::with_capacity(want.len() / 2);
    let mut imag = Vec::with_capacity(want.len() / 2);
    for pair in want.chunks_exact(2) {
        real.push(pair[0]);
        imag.push(pair[1]);
    }
    (real, imag)
}

#[test]
#[ignore = "needs the reference dumps"]
fn the_analysis_matches_the_reference() {
    let spectrum = stft::analyze(&read("input"));
    let (real, imag) = split("spec");
    assert_close("stft real", &spectrum.real, &real, 1e-5);
    assert_close("stft imag", &spectrum.imag, &imag, 1e-5);
}

#[test]
#[ignore = "needs the reference dumps"]
fn the_synthesis_matches_the_reference() {
    let (real, imag) = split("enhanced_spec");
    let frames = real.len() / crate::enhance::gtcrn::config::BINS;
    let wave = stft::synthesize(&stft::Spectrum { real, imag, frames });
    let want = read("enhanced");
    assert_close("istft", &wave, &want[..wave.len()], 1e-4);
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_whole_network_matches_the_reference() {
    let mut model = CandleModel::load(&model_dir()).expect("the checkpoint");
    let input = read("input");
    let wave = model.enhance_window(&input, &[]).unwrap();
    let want = read("enhanced");
    // The reference returns `(frames - 1) * hop` samples; the driver pads back
    // to the input length, so compare over what the reference produced.
    assert_close("enhanced", &wave[..want.len()], &want, 2e-3);
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_network_matches_the_reference_stage_by_stage() {
    let model = CandleModel::load(&model_dir()).expect("the checkpoint");
    let input = read("input");
    let spectrum = stft::analyze(&input);
    let stages = model
        .net()
        .stages(&spectrum.real, &spectrum.imag, spectrum.frames)
        .unwrap();
    for (name, tensor) in stages {
        let got: Vec<f32> = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap();
        assert_close(name, &got, &read(name), 1e-3);
    }
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn cutting_a_recording_into_chunks_changes_nothing() {
    // The whole point of carrying the state: where the chunk boundaries fall
    // must not be visible in the result. Overlapping chunks could not manage
    // this — this network's recurrences remember far longer than any overlap
    // worth paying for.
    let input = read("input");
    let total = crate::enhance::gtcrn::config::frames(input.len());

    let mut whole = CandleModel::load(&model_dir()).expect("the checkpoint");
    whole.reset();
    let mut one = whole.enhance_frames(&input, total).unwrap();
    one.extend(whole.finish());

    let mut split = CandleModel::load(&model_dir()).expect("the checkpoint");
    split.reset();
    let mut many = split.enhance_frames(&input, total / 3).unwrap();
    many.extend(split.enhance_frames(&input, total / 3).unwrap());
    many.extend(
        split
            .enhance_frames(&input, total - 2 * (total / 3))
            .unwrap(),
    );
    many.extend(split.finish());

    assert_eq!(one.len(), many.len());
    let worst = one
        .iter()
        .zip(&many)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst == 0.0, "chunking moved samples by {worst}");
}
