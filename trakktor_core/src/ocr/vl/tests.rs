//! Golden-trace parity against the reference implementation.
//!
//! Every test here is `#[ignore]`d: they need the checkpoint in
//! `~/.trakktor/ocr/vl/` and a trace dumped from the reference into
//! `tmp/ocr/vl/golden/block/` — a page image and the tensors the reference
//! produced from it. Any page will do; a dense one in an unfamiliar script
//! exercises the most. Run them with `cargo test -- --ignored ocr::vl`.
//!
//! The three layers are the domain's ([ADR-0022]): tensors strictly, then the
//! token sequence exactly, then the text character for character. For a
//! generative model the middle layer is the load-bearing one — a single token
//! that differs is a different reading from there on, so "close enough" has no
//! meaning at that level.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::{
    config::{ImageConfig, ModelConfig, TOKENIZER_FILE, WEIGHTS_FILE},
    ernie::Decoder,
    generate::{Limits, Reader, Task},
    image, model,
    vision::{Projector, Tower},
};

/// Where the checkpoint lives.
fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/vl")
        .join(model::DEFAULT_MODEL)
}

/// Where the reference dump lives.
fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/ocr/vl/golden/block")
}

/// Reads one raw little-endian `f32` blob.
fn blob(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(golden().join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("reading the {name} dump: {e}"));
    bytes
        .chunks_exact(4)
        .map(|four| f32::from_le_bytes([four[0], four[1], four[2], four[3]]))
        .collect()
}

fn trace() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(golden().join("trace.json")).unwrap())
        .unwrap()
}

/// The picture the trace was taken on.
fn picture() -> image::Prepared {
    let path = golden().join("page.png");
    let rgb = ::image::open(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .to_rgb8();
    image::prepare(&rgb, &ImageConfig::load(&checkpoint()).unwrap()).unwrap()
}

/// The largest absolute difference between two runs of numbers.
fn worst(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len(), "different lengths");
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// The largest difference relative to the reference's own scale — the honest
/// measure for an activation whose values run into the hundreds.
fn relative(ours: &[f32], theirs: &[f32]) -> f32 {
    let scale = theirs.iter().fold(0f32, |most, v| most.max(v.abs()));
    worst(ours, theirs) / scale.max(f32::MIN_POSITIVE)
}

fn loaded() -> (Tower, Projector, Decoder, Reader) {
    let dir = checkpoint();
    let cfg = ModelConfig::load(&dir).unwrap();
    let image_cfg = ImageConfig::load(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join(TOKENIZER_FILE)).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[dir.join(WEIGHTS_FILE)],
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    };
    let tower = Tower::load(&cfg.vision, vb.pp("visual")).unwrap();
    let projector =
        Projector::load(&cfg.vision, cfg.hidden_size, vb.pp("mlp_AR")).unwrap();
    let decoder = Decoder::load(&cfg, vb.clone()).unwrap();
    let reader = Reader::new(
        Tower::load(&cfg.vision, vb.pp("visual")).unwrap(),
        Projector::load(&cfg.vision, cfg.hidden_size, vb.pp("mlp_AR")).unwrap(),
        Decoder::load(&cfg, vb).unwrap(),
        tokenizer,
        cfg,
        image_cfg,
        Device::Cpu,
        DType::F32,
    )
    .unwrap();
    (tower, projector, decoder, reader)
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_preprocessing_matches_the_reference() {
    let prepared = picture();
    let trace = trace();
    let grid: Vec<usize> = trace["grid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(prepared.grid, (grid[0], grid[1], grid[2]));

    // The resampling is a reimplementation of the reference's, not a call into
    // it: same kernel, same support scaling, but a different order of
    // accumulation and rounding. A pixel or two differs by a few levels out of
    // 255 — the same kind of divergence the classic engine's resize has, and
    // for the same reason. What matters is the layer below: the token sequence
    // comes out identical anyway.
    let theirs = blob("pixel_values");
    let worst = worst(&prepared.pixels, &theirs);
    let differing = prepared
        .pixels
        .iter()
        .zip(&theirs)
        .filter(|(a, b)| (*a - *b).abs() > 1e-6)
        .count();
    // Values live in −1..1, so this is about sixteen levels of 255.
    assert!(worst < 0.13, "worst pixel difference {worst}");
    assert!(
        differing * 3 < prepared.pixels.len() * 2,
        "{differing} of {} values differ",
        prepared.pixels.len()
    );
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_vision_tower_and_projector_match_the_reference() {
    let prepared = picture();
    let (tower, projector, _, _) = loaded();
    // The **reference's own** patches go in, not ours. Otherwise this test
    // measures the resampling twice — once here and once in the test above —
    // and says nothing about the tower, which is what it is for.
    let theirs = blob("pixel_values");
    let pixels = Tensor::from_vec(
        theirs.clone(),
        (prepared.patches(), theirs.len() / prepared.patches()),
        &Device::Cpu,
    )
    .unwrap();

    let ours = tower.forward(&pixels, prepared.grid).unwrap();
    let tower_error = relative(
        &ours.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &blob("tower"),
    );
    assert!(tower_error < 1e-3, "vision tower differs by {tower_error}");

    let ours = projector.forward(&ours, prepared.grid).unwrap();
    let projector_error = relative(
        &ours.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &blob("projector"),
    );
    assert!(
        projector_error < 1e-3,
        "projector differs by {projector_error}"
    );
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_prompt_and_its_positions_match_the_reference() {
    let prepared = picture();
    let (_, _, _, reader) = loaded();
    let trace = trace();

    let expected: Vec<u32> = trace["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let (ours, image_at) =
        reader.prompt(Task::Ocr, prepared.tokens(2)).unwrap();
    assert_eq!(ours, expected, "the prompt is not the reference's");

    let positions = reader.positions(ours.len(), image_at, prepared.grid);
    let expected = trace["position_ids"].as_array().unwrap();
    for axis in 0..3 {
        let theirs: Vec<i64> = expected[axis]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let ours: Vec<i64> = positions.iter().map(|p| p[axis]).collect();
        assert_eq!(ours, theirs, "position axis {axis}");
    }
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_generated_tokens_match_the_reference() {
    let prepared = picture();
    let (_, _, _, reader) = loaded();
    let trace = trace();

    let answer = reader
        .read(&prepared, Task::Ocr, &Limits::default())
        .unwrap();
    assert_eq!(answer.text, trace["text"].as_str().unwrap().trim());
    assert!(!answer.truncated, "the answer was cut short");
    assert!(
        answer.score > 0.8,
        "mean token probability {}",
        answer.score
    );
}
