//! Parity of the burn runtime against the same reference dump the candle one
//! is checked against. `#[ignore]` — they need the converted checkpoint in
//! `~/.trakktor/enhance/unipase/` and the dumps in `tmp/enhance/golden/`.

use burn::backend::ndarray::{NdArray, NdArrayDevice};

use super::BurnModel;
use crate::enhance::{
    EnhanceModel,
    unipase::{config::HOP, plc},
};

fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/enhance/golden")
}

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home).join(".trakktor/enhance/unipase")
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

#[test]
#[ignore = "needs the converted checkpoint and the reference dumps"]
fn the_whole_window_matches_the_reference() {
    let mut model =
        BurnModel::<NdArray<f32>>::load(&model_dir(), NdArrayDevice::Cpu)
            .expect("the converted checkpoint");
    let padded = read_f32("padded");
    let lost = plc::lost_frames(&padded);
    let wave = model.enhance_window(&padded, &lost).unwrap();
    assert_eq!(wave.len(), padded.len() / HOP * HOP);

    let want = read_f32("pipeline_16k");
    let scale = want.iter().fold(0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
    let worst = wave
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "burn pipeline: worst {worst:.3e}, relative {:.3e}",
        worst / scale
    );
    assert!(
        worst / scale < 1e-3,
        "worst {worst:.3e} of scale {scale:.3e}"
    );
}
