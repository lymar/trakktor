//! The burn runtime against the same golden trace, and against candle.
//! `#[ignore]` — they need the converted model in
//! `~/.trakktor/tts/silero/cis-base/` and the reference dumps in
//! `tmp/silero/golden/`.

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use super::BurnSpeech;
use crate::tts::silero::{
    model::{SpeechModel, Utterance},
    runtime,
    tables::Tables,
};

/// Where the reference dumps live.
fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/silero/golden")
}

/// Where the converted model lives.
fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/tts/silero/cis-base")
}

fn read_i32(name: &str) -> Vec<i32> {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(ours: &[f32], theirs: &[f32]) -> f64 {
    let mut dot = 0f64;
    let (mut a2, mut b2) = (0f64, 0f64);
    for (a, b) in ours.iter().zip(theirs) {
        dot += f64::from(*a) * f64::from(*b);
        a2 += f64::from(*a) * f64::from(*a);
        b2 += f64::from(*b) * f64::from(*b);
    }
    dot / (a2.sqrt() * b2.sqrt()).max(1e-30)
}

fn worst(ours: &[f32], theirs: &[f32]) -> f32 {
    ours.iter()
        .zip(theirs)
        .fold(0f32, |worst, (a, b)| worst.max((a - b).abs()))
}

/// The dump's own symbol ids, and the speaker it was made with.
fn utterance() -> Utterance {
    let dir = model_dir();
    let tables = Tables::load(&dir).expect("the converted tables");
    let ids: Vec<u32> =
        read_i32("ids.bin").iter().map(|id| *id as u32).collect();
    Utterance {
        rate: vec![1.0; ids.len()],
        pitch: vec![1.0; ids.len()],
        types: None,
        speaker: tables.speaker("ru_zhadyra").expect("the dump's speaker"),
        ids,
    }
}

#[test]
#[ignore = "needs the converted model and the reference dumps"]
fn the_burn_runtime_matches_the_reference() {
    let model = BurnSpeech::<NdArray>::load(&model_dir(), NdArrayDevice::Cpu)
        .expect("the converted weights");
    let spectrum = model.synthesize(&utterance()).expect("should synthesize");

    let expected: Vec<u32> = read_i32("dur.bin")
        .iter()
        .map(|value| *value as u32)
        .collect();
    assert_eq!(spectrum.durations, expected);

    let wave = model.window().inverse(
        &spectrum.magnitude,
        &spectrum.phase,
        spectrum.frames,
    );
    let theirs = read_f32("wave_48000.bin");
    let cos = cosine(&wave, &theirs);
    println!(
        "burn vs reference: cosine {cos:.12}, worst {:e}",
        worst(&wave, &theirs)
    );
    assert!(cos > 0.999_99, "waveform cosine {cos}");
}

#[test]
#[ignore = "needs the converted model"]
fn the_two_runtimes_agree() {
    let dir = model_dir();
    let utterance = utterance();
    let candle = runtime::load(&dir, runtime::Device::Cpu).expect("candle");
    let burn =
        BurnSpeech::<NdArray>::load(&dir, NdArrayDevice::Cpu).expect("burn");
    let ours = candle.synthesize(&utterance).expect("candle run");
    let theirs = burn.synthesize(&utterance).expect("burn run");

    // The durations are decided by shared host-side code from the head's
    // output, so a difference here would mean the heads themselves disagree by
    // more than a rounding step.
    assert_eq!(ours.durations, theirs.durations);
    assert_eq!(ours.frames, theirs.frames);

    let one =
        candle
            .window()
            .inverse(&ours.magnitude, &ours.phase, ours.frames);
    let two =
        burn.window()
            .inverse(&theirs.magnitude, &theirs.phase, theirs.frames);
    let cos = cosine(&one, &two);
    let identical = one == two;
    println!(
        "candle vs burn: cosine {cos:.12}, worst {:e}, identical {identical}",
        worst(&one, &two)
    );
    assert!(cos > 0.999_999, "the runtimes disagree: cosine {cos}");
}
