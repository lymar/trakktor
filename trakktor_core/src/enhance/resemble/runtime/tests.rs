//! Parity against the Python reference, stage by stage. `#[ignore]` — they need
//! the converted checkpoint in `~/.trakktor/enhance/resemble/` and the
//! reference dumps in `tmp/resemble/`.
//!
//! # What can be compared, and what cannot
//!
//! The **denoiser** is deterministic: one waveform in, one waveform out, no
//! seed and no solver. Its stages are compared against what the reference
//! produced from the same input, and what is left over is arithmetic — a
//! different order of summation in the convolutions and the transform.
//!
//! The **enhancer** is not. It draws Gaussian noise twice per chunk from a
//! generator this port cannot reproduce, so a run of it here and a run of it
//! there are different files by construction. The reference stand therefore
//! records its own draws and these tests feed them in, which measures the
//! network and the solver rather than the generator.

use candle_core::{DType, Device, Tensor};

use super::{DenoiserModel, EnhancerModel, normalize_peak};
use crate::enhance::{
    Precision,
    resemble::{EnhancerSettings, noise::Source, stft},
};

/// Where the reference dumps live.
fn golden_dir(mode: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/resemble")
        .join(format!("golden-{mode}"))
}

/// Where the converted checkpoint lives.
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

/// The chunk the reference ran, and the spectrum it analysed from it.
fn reference_input() -> (Vec<f32>, stft::Spectrum) {
    let chunk = read_f32("denoise", "chunk_in");
    let magnitude = read_f32("denoise", "stft_mag");
    let frames = magnitude.len() / crate::enhance::resemble::config::BINS;
    (
        chunk,
        stft::Spectrum {
            magnitude,
            cos: read_f32("denoise", "stft_cos"),
            sin: read_f32("denoise", "stft_sin"),
            frames,
        },
    )
}

/// The two components scaled back by their magnitude — that is, the real and
/// imaginary parts the transform actually produced.
fn as_complex(spectrum: &stft::Spectrum) -> (Vec<f32>, Vec<f32>) {
    (
        spectrum
            .magnitude
            .iter()
            .zip(&spectrum.cos)
            .map(|(&m, &c)| m * c)
            .collect(),
        spectrum
            .magnitude
            .iter()
            .zip(&spectrum.sin)
            .map(|(&m, &s)| m * s)
            .collect(),
    )
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_analysis_matches_the_reference() {
    let (chunk, want) = reference_input();
    // The network normalizes its input once more, on top of what the driver
    // already did, and the analysis is of that.
    let got = stft::analyze(&normalize_peak(&chunk));
    assert_eq!(got.frames, want.frames);

    let (worst, relative) =
        compare("magnitude", &got.magnitude, &want.magnitude);
    println!("the magnitude: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-5, "the magnitude: relative error {relative}");

    // The complex value itself, which is what the transform computed and what
    // the split into a size and a direction is only a rewriting of.
    let (got_real, got_imag) = as_complex(&got);
    let (want_real, want_imag) = as_complex(&want);
    for (name, got, want) in [
        ("real", &got_real, &want_real),
        ("imaginary", &got_imag, &want_imag),
    ] {
        let (worst, relative) = compare(name, got, want);
        println!("the {name} part: worst {worst:.3e}, relative {relative:.3e}");
        assert!(
            relative < 1e-5,
            "the {name} part: relative error {relative}"
        );
    }

    // The **direction** on its own is a different matter, and the difference is
    // in the question rather than in the answer: where the magnitude is
    // nothing, dividing by it turns a rounding difference in the last bit
    // of the transform into a visible rotation. The reference reaches the
    // same direction by a different route — an arctangent and then its
    // cosine — so the two disagree exactly there and nowhere else. It does
    // not reach the output: what leaves the network is the direction times
    // the magnitude, and that is the row above.
    let scale = want
        .magnitude
        .iter()
        .fold(0f32, |acc, &v| acc.max(v.abs()))
        .max(1e-12);
    // Stated without a threshold to argue about: the rotation the two
    // formulations disagree by, **weighted by the magnitude it applies to**,
    // is the disagreement that can reach anything downstream. It is bounded by
    // the difference in the complex value itself, so it stays at the level of
    // the transform's own rounding however wild the bare angle looks.
    let mut worst_weighted = 0f32;
    let mut worst_bare = 0f32;
    let mut wildest_magnitude = 0f32;
    for index in 0..want.magnitude.len() {
        let turn = (got.cos[index] - want.cos[index])
            .abs()
            .max((got.sin[index] - want.sin[index]).abs());
        if turn > worst_bare {
            worst_bare = turn;
            wildest_magnitude = want.magnitude[index];
        }
        worst_weighted = worst_weighted.max(turn * want.magnitude[index]);
    }
    let weighted = worst_weighted / scale;
    println!(
        "the direction: worst {worst_bare:.3e} bare (at a magnitude of \
         {wildest_magnitude:.3e} out of {scale:.3e}), {weighted:.3e} weighted"
    );
    assert!(weighted < 1e-6, "the weighted direction: {weighted}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn every_stage_matches_the_reference() {
    let (_, spectrum) = reference_input();
    let model = DenoiserModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");
    let stages = model.net().stages(&spectrum).expect("the network runs");

    let mut worst_overall = 0f32;
    for (name, tensor) in &stages {
        let want = read_f32("denoise", name);
        let (worst, relative) = compare(name, &host(tensor), &want);
        println!("{name}: worst {worst:.3e}, relative {relative:.3e}");
        // The two residual planes are the components of a **unit vector**, so
        // the difference to judge them by is the absolute one: it is the angle
        // they disagree by, and a thousandth of it is a twentieth of a degree.
        // Their relative-to-the-largest figure says the same thing and reads
        // worse only because the largest is one.
        let bound = if name.ends_with("_res") { 5e-3 } else { 1e-3 };
        assert!(relative < bound, "{name}: relative error {relative}");
        worst_overall = worst_overall.max(relative);
    }
    println!("worst relative error over every stage: {worst_overall:.3e}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_rotated_spectrum_matches_the_reference() {
    // What the two residual planes are *for*: the spectrum they turn. This is
    // the tight check on the phase path — the rotation applied and scaled by
    // the magnitude it belongs to, which is exactly what the transform reads.
    let (_, spectrum) = reference_input();
    let model = DenoiserModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");
    let prediction = model.net().predict(&spectrum).expect("the network runs");
    let separated = stft::apply(&spectrum, &prediction);

    let want_mag = read_f32("denoise", "sep_mag");
    let want_cos = read_f32("denoise", "sep_cos");
    let want_sin = read_f32("denoise", "sep_sin");
    let want_real: Vec<f32> = want_mag
        .iter()
        .zip(&want_cos)
        .map(|(&m, &c)| m * c)
        .collect();
    let want_imag: Vec<f32> = want_mag
        .iter()
        .zip(&want_sin)
        .map(|(&m, &s)| m * s)
        .collect();
    let (got_real, got_imag) = as_complex(&separated);

    for (name, got, want) in [
        ("magnitude", &separated.magnitude, &want_mag),
        ("real", &got_real, &want_real),
        ("imaginary", &got_imag, &want_imag),
    ] {
        let (worst, relative) = compare(name, got, want);
        println!(
            "the separated {name}: worst {worst:.3e}, relative {relative:.3e}"
        );
        assert!(relative < 1e-4, "the separated {name}: {relative}");
    }
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_waveform_matches_the_reference_from_its_own_spectrum() {
    // The network and the synthesis, on exactly what the reference analysed, so
    // what is measured is the port and nothing else.
    let (chunk, spectrum) = reference_input();
    let model = DenoiserModel::load(&model_dir(), Device::Cpu, Precision::F32)
        .expect("the converted checkpoint");
    let prediction = model.net().predict(&spectrum).expect("the network runs");
    let mut got = stft::synthesize(&stft::apply(&spectrum, &prediction));
    got.resize(chunk.len(), 0.0);

    let (worst, relative) =
        compare("chunk_out", &got, &read_f32("denoise", "chunk_out"));
    println!("the waveform: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-3, "the waveform: relative error {relative}");
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn every_stage_of_the_enhancer_matches_the_reference() {
    // Fed the reference's own two draws, the generative pipeline becomes
    // comparable: everything downstream of them is a function of the recording
    // and the weights.
    let chunk = read_f32("enhance", "chunk_in");
    let mut model = EnhancerModel::load(
        &model_dir(),
        Device::Cpu,
        Precision::F32,
        EnhancerSettings::default(),
    )
    .expect("the converted checkpoint");
    model.with_noise(Source::Scripted(vec![
        read_f32("enhance", "noise_0"),
        read_f32("enhance", "noise_1"),
    ]));

    let stages = model.stages(&chunk).expect("the pipeline runs");
    let mut worst_overall = 0f32;
    for (name, got) in &stages {
        let want = read_f32("enhance", name);
        let (worst, relative) = compare(name, got, &want);
        println!(
            "{name}: worst {worst:.3e}, relative {relative:.3e}{}",
            where_worst(name, got, &want)
        );
        assert!(relative < bound(name), "{name}: relative error {relative}");
        worst_overall = worst_overall.max(relative);
    }
    println!("worst relative error over every stage: {worst_overall:.3e}");
}

/// What each stage is allowed to differ by, and why they are not all the same.
///
/// The pipeline has one amplifier in it. Everything the recording passes
/// through on the way in agrees at the level of arithmetic; the **vocoder**
/// then turns four hundred and twenty samples out of every conditioning frame
/// through four gated stages, and a difference in what it is conditioned on
/// comes out an order of magnitude larger. That is a property of the network,
/// not of this port —
/// [`the_vocoder_matches_the_reference_from_its_own_conditioning`] shows the
/// vocoder reproducing the reference to two parts in a million when it is
/// handed the reference's own conditioning.
///
/// The stages that carry the denoiser's output are looser still, for a reason
/// of their own: a mel is a logarithm, and a difference of two parts in a
/// million in a waveform is a much larger fraction of a band that has almost
/// nothing in it.
fn bound(name: &str) -> f32 {
    match name {
        "mel_1" | "mel_norm_1" | "ae_decoder" => 1e-3,
        "chunk_out" | "voc_post" => 1e-2,
        _ if name.starts_with("voc") => 1e-1,
        _ => 1e-5,
    }
}

/// Where in a mel the worst point is, which is the difference between a
/// padding mistake, a floor mistake and a window mistake.
fn where_worst(name: &str, got: &[f32], want: &[f32]) -> String {
    if !name.starts_with("mel") {
        return String::new();
    }
    let frames = got.len() / crate::enhance::resemble::config::MELS;
    let at = got
        .iter()
        .zip(want)
        .enumerate()
        .max_by(|a, b| (a.1.0 - a.1.1).abs().total_cmp(&(b.1.0 - b.1.1).abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    // A mel is a logarithm with a floor under it, so a band with nothing in it
    // sits a hair above that floor and a difference of one part in a hundred
    // thousand of the spectrum becomes a visible difference in decibels. What
    // matters is whether the bands that carry something also disagree.
    let loud = got
        .iter()
        .zip(want)
        .filter(|&(_, &want)| want > 0.3)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0f32, f32::max);
    format!(
        " (worst at band {} of {}, frame {} of {frames}: {} against {}; worst \
         above a fifth of the scale: {loud:.3e})",
        at / frames,
        crate::enhance::resemble::config::MELS,
        at % frames,
        got[at],
        want[at]
    )
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_vocoder_matches_the_reference_from_its_own_conditioning() {
    // Handed the reference's own conditioning and its own noise, the vocoder
    // has nothing left of its own to be wrong about — so this separates what
    // the vocoder computes from what it *amplifies*. It amplifies a great
    // deal: it upsamples four hundred and twenty to one through four gated
    // stages, and a difference of two parts in ten thousand in what it is
    // conditioned on comes out twenty times larger.
    let model = EnhancerModel::load(
        &model_dir(),
        Device::Cpu,
        Precision::F32,
        EnhancerSettings::default(),
    )
    .expect("the converted checkpoint");

    let decoded = read_f32("enhance", "ae_decoder");
    let frames =
        decoded.len() / crate::enhance::resemble::config::VOCODER_INPUT;
    let cond = Tensor::from_vec(
        decoded,
        (1, crate::enhance::resemble::config::VOCODER_INPUT, frames),
        model.device(),
    )
    .expect("the conditioning");
    let got = model
        .vocoder()
        .forward(&cond, &read_f32("enhance", "noise_1"))
        .expect("the vocoder runs");

    let (worst, relative) =
        compare("chunk_out", &got, &read_f32("enhance", "chunk_out"));
    println!("the vocoder alone: worst {worst:.3e}, relative {relative:.3e}");
    assert!(
        relative < 1e-4,
        "the vocoder alone: relative error {relative}"
    );
}

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_whole_chunk_matches_the_reference() {
    // The same thing end to end, this time analysing the samples here rather
    // than borrowing the reference's spectrum.
    use crate::enhance::EnhanceModel;
    let (chunk, _) = reference_input();
    let mut model =
        DenoiserModel::load(&model_dir(), Device::Cpu, Precision::F32)
            .expect("the converted checkpoint");
    let got = model
        .enhance_window(&chunk, &[])
        .expect("the chunk is enhanced");
    let (worst, relative) =
        compare("chunk_out", &got, &read_f32("denoise", "chunk_out"));
    println!("the waveform: worst {worst:.3e}, relative {relative:.3e}");
    assert!(relative < 1e-3, "the waveform: relative error {relative}");
}
