//! Parity against the Python reference, stage by stage. `#[ignore]` — they need
//! the converted checkpoint in `~/.trakktor/tts/espeech/rl-v2/`, the vocoder
//! next to it, and the reference dumps in `tmp/espeech/golden4/`.
//!
//! Because a flow-matching run is deterministic once the noise is fixed, these
//! are not "close enough" tests: the reference's own noise is fed in, and every
//! stage is compared against what the reference produced from it. What is left
//! is the arithmetic — a different order of summation in the convolutions,
//! matmuls, and transforms — which is what the tolerances are sized for.

use candle_core::{Device, Tensor};

use super::{load, model_err};
use crate::tts::espeech::{
    Precision,
    config::SAMPLE_RATE,
    mel::MelBasis,
    model::SpeechModel,
    reference,
    tokenizer::{CharTokenizer, close_reference_text},
};

/// Where the reference dumps live.
fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/espeech/golden4")
}

/// Where the converted checkpoint and the vocoder live.
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

fn read_text(name: &str) -> String {
    let path = golden_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
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

/// How far apart two runs of the same arithmetic are: the largest absolute
/// deviation, and the cosine of the angle between them.
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
    let cosine = if ours_norm > 0.0 && theirs_norm > 0.0 {
        dot / (ours_norm.sqrt() * theirs_norm.sqrt())
    } else {
        0.0
    };
    (worst, cosine)
}

/// The reference recording, prepared as a run would prepare it.
fn prepared_reference() -> reference::Reference {
    let wave = read_f32("ref_wave.bin");
    // The dump is already the prepared waveform, so it only has to be wrapped;
    // preparing it again would trim the tail silence the reference appended.
    reference::Reference {
        rms: dims()["ref_rms"].as_f64().unwrap() as f32,
        clipped: false,
        wave,
    }
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_decoder_hands_back_the_same_samples_as_the_reference() {
    // The dumped waveform is the file as the reference read it — 16-bit PCM at
    // 24 kHz, so the built-in decoder should give the identical samples, not
    // merely close ones. (What the port then does to it — trim the edges, cap
    // the length, append silence — is its own, documented, divergence, so the
    // comparison is against the decode alone.)
    let decoded = crate::audio::decode_to_mono_f32(
        &golden_dir().join("ref.wav"),
        SAMPLE_RATE,
    )
    .expect("decode");
    let dumped = read_f32("ref_wave.bin");
    let (worst, cosine) = agreement(&decoded, &dumped);
    println!("decoded wave: worst {worst:e}, cosine {cosine:.12}");
    assert_eq!(decoded.len(), dumped.len());
    assert!(worst == 0.0, "worst {worst:e}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn preparation_only_trims_the_silent_edges_of_this_reference() {
    // The recording is nine seconds of speech with a little silence around it,
    // so preparation should keep nearly all of it and hit neither the cap nor
    // the loudness floor.
    let prepared =
        reference::prepare(&golden_dir().join("ref.wav")).expect("prepare");
    let dumped = read_f32("ref_wave.bin");
    let trimmed = dumped.len() as f64 / f64::from(SAMPLE_RATE) -
        prepared.seconds() +
        crate::tts::espeech::config::REF_TAIL_SILENCE;
    println!(
        "prepared {:.3} s of {:.3} s ({trimmed:.3} s of silence trimmed)",
        prepared.seconds(),
        dumped.len() as f64 / f64::from(SAMPLE_RATE)
    );
    assert!(!prepared.clipped);
    assert!((0.0..0.5).contains(&trimmed), "trimmed {trimmed:.3} s");
    assert_eq!(prepared.output_gain(), 1.0);
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_mel_spectrogram_matches_the_reference() {
    let (model_dir, vocoder_dir) = model_dirs();
    let model = load(&model_dir, &vocoder_dir, Device::Cpu, Precision::F32)
        .expect("load");
    let mel = model
        .mel_basis()
        .log_mel(&read_f32("ref_wave.bin"))
        .expect("mel");
    let golden = read_f32("ref_mel.bin");
    let (worst, cosine) = agreement(&mel, &golden);
    let deviation = (mel
        .iter()
        .zip(&golden)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>() /
        mel.len() as f64)
        .sqrt();
    println!("mel: worst {worst:e}, rms {deviation:e}, cosine {cosine:.12}");
    // The worst case lives in the near-silent bands, and it has to: the mel is
    // a logarithm clamped at 1e-5, so where a magnitude is that small the log
    // magnifies the transform's own f32 rounding into something visible. The
    // deviation that matters is the typical one, and the cosine says the two
    // spectrograms are the same picture.
    assert!(worst < 1e-2, "worst {worst:e}");
    assert!(deviation < 1e-4, "rms deviation {deviation:e}");
    assert!(cosine > 0.999_999_99, "cosine {cosine}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_text_becomes_the_same_tokens() {
    let (model_dir, _) = model_dirs();
    let tokenizer =
        CharTokenizer::load(&model_dir.join("vocab.txt")).expect("vocab");
    let ids = tokenizer.encode(&format!(
        "{}{}",
        close_reference_text(read_text("ref.txt").trim()),
        read_text("test.txt")
    ));
    let golden: Vec<u32> = read_i32("text_ids.bin")
        .into_iter()
        .map(|id| id as u32)
        .collect();
    assert_eq!(ids, golden);
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_vocoder_reproduces_the_reference_waveform() {
    // The most isolated stage there is: a fixed mel in, a waveform out, no
    // solver involved.
    let (model_dir, vocoder_dir) = model_dirs();
    let model = load(&model_dir, &vocoder_dir, Device::Cpu, Precision::F32)
        .expect("load");
    let mel = read_f32("ref_mel.bin");
    let frames = usize_field(&dims(), "cond_frames");
    let channels = model.config().mel_channels;
    let tensor = Tensor::from_slice(&mel, (1, frames, channels), &Device::Cpu)
        .expect("mel tensor");
    let (magnitude, phase) = model.vocoder.spectrum(&tensor).expect("vocoder");
    let wave = model.basis.istft(&magnitude, &phase, frames);
    let golden = read_f32("vocos_roundtrip.bin");
    let (worst, cosine) = agreement(&wave, &golden);
    println!("vocoder: worst {worst:e}, cosine {cosine:.12}");
    assert!(worst < 1e-3, "worst {worst:e}");
    assert!(cosine > 0.999_999_9, "cosine {cosine}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_first_flow_prediction_matches_the_reference() {
    let (model_dir, vocoder_dir) = model_dirs();
    let mut model = load(&model_dir, &vocoder_dir, Device::Cpu, Precision::F32)
        .expect("load");
    let dims = dims();
    let frames = usize_field(&dims, "duration");
    let cond_frames = usize_field(&dims, "cond_frames");
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
    let state = model.solving.as_ref().expect("prepared");
    let flow = model
        .dit
        .forward(&state.x, &state.context, times[0])
        .expect("flow");
    let guided = super::combine(&flow, 2.0).expect("combine");
    let ours: Vec<f32> = guided.flatten_all().unwrap().to_vec1().unwrap();
    let (worst, cosine) = agreement(&ours, &read_f32("v0.bin"));
    println!("first flow: worst {worst:e}, cosine {cosine:.12}");
    assert!(worst < 2e-3, "worst {worst:e}");
    assert!(cosine > 0.999_999, "cosine {cosine}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn the_whole_run_reproduces_the_reference_audio() {
    let (model_dir, vocoder_dir) = model_dirs();
    let mut model = load(&model_dir, &vocoder_dir, Device::Cpu, Precision::F32)
        .expect("load");
    let dims = dims();
    let frames = usize_field(&dims, "duration");
    let cond_frames = usize_field(&dims, "cond_frames");
    let cut_frames = usize_field(&dims, "ref_frames");
    let ids: Vec<u32> = read_i32("text_ids.bin")
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let cond = read_f32("ref_mel.bin");
    model
        .prepare(&cond, cond_frames, &ids, frames, &read_f32("y0.bin"))
        .expect("prepare");

    let times = read_f32("t.bin");
    for step in 0..times.len() - 1 {
        model
            .step(times[step], times[step + 1] - times[step], 2.0)
            .expect("step");
    }

    // The mel before it goes to the vocoder, which is where the solver's own
    // error stops accumulating.
    let state = model.solving.as_ref().expect("prepared");
    let mel = model
        .generated_mel(state, cut_frames)
        .and_then(|mel| mel.flatten_all()?.to_vec1::<f32>())
        .map_err(|e| model_err("mel", e))
        .expect("mel");
    let (worst, cosine) = agreement(&mel, &read_f32("mel_gen.bin"));
    println!("solved mel: worst {worst:e}, cosine {cosine:.12}");
    assert!(worst < 5e-3, "worst {worst:e}");
    assert!(cosine > 0.999_99, "cosine {cosine}");

    let wave = model.finish(cut_frames).expect("finish");
    let golden = read_f32("wave.bin");
    let (worst, cosine) = agreement(&wave, &golden);
    println!(
        "waveform: worst {worst:e}, cosine {cosine:.12}, {:.3} s",
        wave.len() as f64 / f64::from(SAMPLE_RATE)
    );
    assert_eq!(wave.len(), golden.len());
    assert!(worst < 2e-2, "worst {worst:e}");
    assert!(cosine > 0.999_9, "cosine {cosine}");
}

#[test]
#[ignore = "needs tmp/espeech/golden4 and a converted checkpoint"]
fn framing_of_the_dumped_reference_is_what_the_port_assumes() {
    let dims = dims();
    let wave = read_f32("ref_wave.bin");
    assert_eq!(
        MelBasis::frames_for(wave.len()),
        usize_field(&dims, "cond_frames")
    );
    assert_eq!(wave.len() / 256, usize_field(&dims, "ref_frames"));
    assert_eq!(prepared_reference().cut_frames(), wave.len() / 256);
}
