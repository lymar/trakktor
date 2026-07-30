//! Parity against the Python reference, stage by stage. `#[ignore]` — they need
//! the converted model in `~/.trakktor/tts/silero/cis-base/` and the reference
//! dumps in `tmp/silero/golden/`, which
//! `scripts/tts/silero/dump_reference.py` writes.
//!
//! The pipeline samples nothing, so these are not "close enough in spirit"
//! tests: the same symbol ids go in, and every stage is compared against what
//! the reference made of them. The durations have to match **exactly** — they
//! are integers, and one frame more anywhere shifts every frame after it — and
//! what is left is arithmetic, which is what the tolerances are sized for.

use super::{Device, load};
use crate::tts::silero::{
    model::{SpeechModel, Utterance},
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

fn read_f32(name: &str) -> Vec<f32> {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

/// The speaker the dump was made with.
const SPEAKER: &str = "ru_zhadyra";

/// Cosine similarity, the measure a waveform is judged by: it is insensitive to
/// nothing that matters and to a uniform scale, which no stage here applies.
fn cosine(ours: &[f32], theirs: &[f32]) -> f64 {
    let mut dot = 0f64;
    let (mut ours_norm, mut theirs_norm) = (0f64, 0f64);
    for (a, b) in ours.iter().zip(theirs) {
        dot += f64::from(*a) * f64::from(*b);
        ours_norm += f64::from(*a) * f64::from(*a);
        theirs_norm += f64::from(*b) * f64::from(*b);
    }
    dot / (ours_norm.sqrt() * theirs_norm.sqrt()).max(1e-30)
}

/// Largest absolute difference.
fn worst(ours: &[f32], theirs: &[f32]) -> f32 {
    ours.iter()
        .zip(theirs)
        .fold(0f32, |worst, (a, b)| worst.max((a - b).abs()))
}

/// Runs the whole pipeline on the dump's own symbol ids.
fn run() -> (
    crate::tts::silero::model::Spectrum,
    Box<dyn SpeechModel>,
    Vec<f32>,
) {
    let dir = model_dir();
    let tables = Tables::load(&dir).expect("the converted tables");
    let speaker = tables.speaker(SPEAKER).expect("the dump's speaker");
    let model = load(&dir, Device::Cpu).expect("the converted weights");
    let ids: Vec<u32> =
        read_i32("ids.bin").iter().map(|id| *id as u32).collect();
    let spectrum = model
        .synthesize(&Utterance {
            rate: vec![1.0; ids.len()],
            pitch: vec![1.0; ids.len()],
            types: None,
            ids,
            speaker,
        })
        .expect("should synthesize");
    let wave = model.window().inverse(
        &spectrum.magnitude,
        &spectrum.phase,
        spectrum.frames,
    );
    (spectrum, Box::new(model), wave)
}

#[test]
#[ignore = "needs the converted model and the reference dumps"]
fn the_durations_match_the_reference_exactly() {
    let (spectrum, _, _) = run();
    let theirs: Vec<u32> = read_i32("dur.bin")
        .iter()
        .map(|value| *value as u32)
        .collect();
    assert_eq!(spectrum.durations, theirs);
    assert_eq!(
        spectrum.frames,
        theirs.iter().map(|value| *value as usize).sum::<usize>()
    );
}

#[test]
#[ignore = "needs the converted model and the reference dumps"]
fn the_spectrum_matches_the_reference() {
    let (spectrum, _, _) = run();
    let magnitude = read_f32("magnitude.bin");
    let phase = read_f32("phase.bin");
    assert_eq!(spectrum.magnitude.len(), magnitude.len());
    let cos = cosine(&spectrum.magnitude, &magnitude);
    println!(
        "magnitude: cosine {cos:.12}, worst {:e}",
        worst(&spectrum.magnitude, &magnitude)
    );
    assert!(cos > 0.999_999, "magnitude cosine {cos}");
    let cos = cosine(&spectrum.phase, &phase);
    println!(
        "phase:     cosine {cos:.12}, worst {:e}",
        worst(&spectrum.phase, &phase)
    );
    assert!(cos > 0.999_99, "phase cosine {cos}");
}

#[test]
#[ignore = "needs the converted model and the reference dumps"]
fn the_waveform_matches_the_reference() {
    let (_, _, wave) = run();
    let theirs = read_f32("wave_48000.bin");
    assert_eq!(wave.len(), theirs.len());
    let cos = cosine(&wave, &theirs);
    println!(
        "wave 48 kHz: cosine {cos:.12}, worst {:e}",
        worst(&wave, &theirs)
    );
    assert!(cos > 0.999_99, "waveform cosine {cos}");
}

#[test]
#[ignore = "needs the converted model and the reference dumps"]
fn the_lower_rates_come_from_the_models_own_filterbank() {
    let (_, model, wave) = run();
    for (bands, rate) in [(2usize, 24_000), (6, 8_000)] {
        let bank = model.pqmf(bands).expect("the filterbank");
        let ours = bank.low_band(&wave);
        let theirs = read_f32(&format!("wave_{rate}.bin"));
        assert_eq!(ours.len(), theirs.len(), "at {rate} Hz");
        let cos = cosine(&ours, &theirs);
        println!(
            "wave {rate} Hz: cosine {cos:.12}, worst {:e}",
            worst(&ours, &theirs)
        );
        assert!(cos > 0.999_99, "at {rate} Hz: cosine {cos}");
    }
}
