//! Parity of the burn runtime, against the same reference dumps the candle one
//! is checked against. `#[ignore]` — they need the converted checkpoint in
//! `~/.trakktor/tts/espeech/rl-v2/` and the dumps in `tmp/espeech/golden4/`.
//!
//! The starting noise comes from the dump rather than from a seed, so this is a
//! comparison of arithmetic, not of random draws: both runtimes solve the same
//! trajectory, and the reference solved it too.

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use super::{BurnSpeech, load_cpu, load_metal};
use crate::tts::espeech::{Precision, config::SAMPLE_RATE, model::SpeechModel};

#[test]
fn half_precision_is_refused_before_anything_is_loaded() {
    // The refusal is a property of the runtime, not of the checkpoint, so it
    // comes back for a path that does not even exist — and it says what to do
    // instead.
    let nowhere = std::path::Path::new("/nonexistent/espeech");
    for loader in [load_cpu, load_metal] {
        let error = loader(nowhere, nowhere, Precision::F16)
            .err()
            .expect("f16 should be refused");
        let message = error.to_string();
        assert!(message.contains("f32 only"), "{message}");
        assert!(message.contains("--runtime candle"), "{message}");
    }
}

fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/espeech/golden4")
}

fn model_dirs() -> (std::path::PathBuf, std::path::PathBuf) {
    let home = std::env::var("HOME").expect("HOME");
    let engine = std::path::PathBuf::from(home).join(".trakktor/tts/espeech");
    (
        engine.join("rl-v2"),
        engine.join(crate::tts::espeech::download::VOCODER_DIR),
    )
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

fn dims() -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(golden_dir().join("dims.json")).unwrap(),
    )
    .unwrap()
}

fn usize_field(dims: &serde_json::Value, name: &str) -> usize {
    dims[name].as_u64().expect(name) as usize
}

fn agreement(ours: &[f32], theirs: &[f32]) -> (f32, f64) {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    let worst = ours
        .iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (mut dot, mut ours_norm, mut theirs_norm) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in ours.iter().zip(theirs) {
        dot += f64::from(*a) * f64::from(*b);
        ours_norm += f64::from(*a) * f64::from(*a);
        theirs_norm += f64::from(*b) * f64::from(*b);
    }
    (worst, dot / (ours_norm.sqrt() * theirs_norm.sqrt()))
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_vocoder_reproduces_the_reference_waveform() {
    let (model_dir, vocoder_dir) = model_dirs();
    let model = BurnSpeech::<NdArray<f32>>::load(
        &model_dir,
        &vocoder_dir,
        NdArrayDevice::Cpu,
    )
    .expect("load");
    let frames = usize_field(&dims(), "cond_frames");
    let channels = model.config().mel_channels;
    let mel = burn::tensor::Tensor::<NdArray<f32>, 3>::from_data(
        burn::tensor::TensorData::new(
            read_f32("ref_mel.bin"),
            [1, frames, channels],
        ),
        &NdArrayDevice::Cpu,
    );
    let (magnitude, phase) = model.vocoder.spectrum(mel).expect("vocoder");
    let wave = model.basis.istft(&magnitude, &phase, frames);
    let (worst, cosine) = agreement(&wave, &read_f32("vocos_roundtrip.bin"));
    println!("burn vocoder: worst {worst:e}, cosine {cosine:.12}");
    assert!(worst < 1e-3, "worst {worst:e}");
    assert!(cosine > 0.999_999_9, "cosine {cosine}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_whole_run_reproduces_the_reference_audio() {
    let (model_dir, vocoder_dir) = model_dirs();
    let mut model = BurnSpeech::<NdArray<f32>>::load(
        &model_dir,
        &vocoder_dir,
        NdArrayDevice::Cpu,
    )
    .expect("load");
    let dims = dims();
    let frames = usize_field(&dims, "duration");
    let cond_frames = usize_field(&dims, "cond_frames");
    let cut_frames = usize_field(&dims, "ref_frames");
    let ids: Vec<u32> = read_i32("text_ids.bin")
        .into_iter()
        .map(|id| id as u32)
        .collect();
    model
        .prepare(
            &read_f32("ref_mel.bin"),
            cond_frames,
            &ids,
            frames,
            &read_f32("y0.bin"),
        )
        .expect("prepare");

    let times = read_f32("t.bin");
    for step in 0..times.len() - 1 {
        model
            .step(times[step], times[step + 1] - times[step], 2.0)
            .expect("step");
    }

    let state = model.solving.as_ref().expect("prepared");
    let mel = super::net::into_values(model.generated_mel(state, cut_frames))
        .expect("mel");
    let (worst, cosine) = agreement(&mel, &read_f32("mel_gen.bin"));
    println!("burn solved mel: worst {worst:e}, cosine {cosine:.12}");
    assert!(worst < 5e-3, "worst {worst:e}");
    assert!(cosine > 0.999_99, "cosine {cosine}");

    let wave = model.finish(cut_frames).expect("finish");
    let golden = read_f32("wave.bin");
    let (worst, cosine) = agreement(&wave, &golden);
    println!(
        "burn waveform: worst {worst:e}, cosine {cosine:.12}, {:.3} s",
        wave.len() as f64 / f64::from(SAMPLE_RATE)
    );
    assert_eq!(wave.len(), golden.len());
    assert!(worst < 2e-2, "worst {worst:e}");
    assert!(cosine > 0.999_9, "cosine {cosine}");
}
