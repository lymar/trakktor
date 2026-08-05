//! Parity of the burn runtime, against the same reference dump as candle.
//!
//! `#[ignore]`d for the same reasons as the candle tests next door: they need
//! the artifacts in `~/.trakktor/ocr/layout/` and a trace in
//! `tmp/ocr/golden/layout/`.

use std::path::PathBuf;

use super::*;
use crate::ocr::layout::{config::LayoutConfig, model, post};

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/layout")
        .join(model::DEFAULT_MODEL)
}

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/ocr/golden/layout/page")
}

fn blob(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(golden().join(format!("{name}.bin")))
        .unwrap_or_else(|e| panic!("reading the {name} dump: {e}"));
    bytes
        .chunks_exact(4)
        .map(|four| f32::from_le_bytes([four[0], four[1], four[2], four[3]]))
        .collect()
}

fn trace() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(golden().join("input.json")).unwrap())
        .unwrap()
}

fn page_size(trace: &serde_json::Value) -> (f32, f32) {
    let page = trace["page"].as_array().expect("the page size");
    (
        page[0].as_f64().expect("a width") as f32,
        page[1].as_f64().expect("a height") as f32,
    )
}

fn decode(prediction: &Prediction, page: (f32, f32)) -> Vec<post::Raw> {
    let classes = prediction.classes;
    let mut ranked: Vec<(usize, f32)> = prediction
        .logits
        .iter()
        .enumerate()
        .map(|(at, logit)| (at, 1.0 / (1.0 + (-logit).exp())))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(300);
    ranked
        .into_iter()
        .map(|(at, score)| {
            let query = at / classes;
            let class = at % classes;
            let box_ = &prediction.boxes[query * 4..query * 4 + 4];
            let (cx, cy, w, h) = (box_[0], box_[1], box_[2], box_[3]);
            post::Raw {
                class,
                score,
                bounds: (
                    (cx - 0.5 * w) * page.0,
                    (cy - 0.5 * h) * page.1,
                    (cx + 0.5 * w) * page.0,
                    (cy + 0.5 * h) * page.1,
                ),
            }
        })
        .collect()
}

fn overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let width = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let height = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let shared = width * height;
    let area = |r: (f32, f32, f32, f32)| (r.2 - r.0) * (r.3 - r.1);
    let union = area(a) + area(b) - shared;
    if union <= 0.0 { 0.0 } else { shared / union }
}

/// Boxes below this score are noise whose ranking is decided by the last bits
/// of arithmetic; see the candle test for the reasoning.
const COMPARABLE: f32 = 0.05;

/// The burn network against the reference, on the reference's own input.
#[test]
#[ignore = "needs the model and a reference dump"]
fn the_burn_network_matches_the_reference() {
    let dir = checkpoint();
    let config = LayoutConfig::load(&dir).expect("the model description");
    let side = config.target_size[0];
    let net = Net::load(&dir, Device::Cpu, side, config.labels.len())
        .expect("the network");

    let prediction =
        net.forward(&blob("input"), side).expect("the forward pass");
    let expected = blob("boxes");
    let got = decode(&prediction, page_size(&trace()));

    let mut worst_corner = 0f32;
    let mut worst_score = 0f32;
    let mut compared = 0usize;
    for theirs in expected.chunks_exact(6) {
        if theirs[1] < COMPARABLE {
            continue;
        }
        compared += 1;
        let class = theirs[0] as usize;
        let bounds = (theirs[2], theirs[3], theirs[4], theirs[5]);
        let ours = got
            .iter()
            .filter(|box_| box_.class == class)
            .max_by(|a, b| {
                overlap(a.bounds, bounds).total_cmp(&overlap(b.bounds, bounds))
            })
            .unwrap_or_else(|| panic!("no box of class {class} at all"));
        assert!(
            overlap(ours.bounds, bounds) > 0.99,
            "the closest box of class {class} to {bounds:?} is {:?}",
            ours.bounds
        );
        worst_score = worst_score.max((ours.score - theirs[1]).abs());
        for (a, b) in [
            (ours.bounds.0, bounds.0),
            (ours.bounds.1, bounds.1),
            (ours.bounds.2, bounds.2),
            (ours.bounds.3, bounds.3),
        ] {
            worst_corner = worst_corner.max((a - b).abs());
        }
    }
    assert!(compared >= 8, "only {compared} boxes were worth comparing");
    println!(
        "burn: worst score {worst_score:.5}, worst corner {worst_corner:.3} px"
    );
    assert!(worst_score < 2e-3, "the worst score is {worst_score} out");
    assert!(
        worst_corner < 2.0,
        "the worst corner is {worst_corner} pixels out"
    );
}
