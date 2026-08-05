//! Golden-trace parity against the reference implementation.
//!
//! The tests are `#[ignore]`d: they need the artifacts in
//! `~/.trakktor/ocr/layout/` and a trace dumped from the reference into
//! `tmp/ocr/golden/layout/page/` — the input tensor the reference built and
//! the boxes it got back. Run them with
//! `cargo test -p trakktor_core --features ocr-runtime -- --ignored
//! ocr::layout`.
//!
//! The first layer of the domain's parity discipline is the network on the
//! reference's *own* input: feeding it our own pre-processing instead would
//! measure the resampler twice and say nothing about the port of the net.

use std::path::PathBuf;

use candle_core::Device;

use super::*;
use crate::ocr::{
    layout::{config::LayoutConfig, model, post},
    paddle::artifact::Artifact,
};

/// Where the artifacts live.
fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/layout")
        .join(model::DEFAULT_MODEL)
}

/// Where the reference dump lives.
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

fn load() -> (Net, LayoutConfig) {
    let dir = checkpoint();
    let config = LayoutConfig::load(&dir).expect("the model description");
    config.check_labels().expect("the published labels");
    let artifact = Artifact::load(&dir).expect("the artifacts");
    let device = Device::Cpu;
    let loader = Loader::new(&artifact, &device);
    let net = Net::load(&loader, config.target_size[0], config.labels.len())
        .expect("the network");
    (net, config)
}

/// The graph's tail, in the shape the tests need it.
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

fn page_size(trace: &serde_json::Value) -> (f32, f32) {
    let page = trace["page"].as_array().expect("the page size");
    (
        page[0].as_f64().expect("a width") as f32,
        page[1].as_f64().expect("a height") as f32,
    )
}

/// Boxes below this score are compared for nothing.
///
/// The network answers three hundred `(query, class)` pairs whatever the page
/// holds, and most of them are noise: on a plain page everything past the
/// twentieth scores under a hundredth. Down there the ranking is a sort of
/// near-equal numbers, so the *order* of the tail is decided by the last bits
/// of arithmetic and differs between any two implementations — while the
/// lowest floor the post-processing applies is 0.3. This bar sits far below
/// that and far above the noise.
const COMPARABLE: f32 = 0.05;

/// The whole network against the reference, on the reference's own input.
///
/// The comparison is on the *decoded* boxes rather than on raw tensors,
/// because that is what the reference dumps: its graph decodes internally. The
/// class of every box that matters must match exactly — that is a decision,
/// not arithmetic — while the coordinates are compared with a tolerance in
/// page pixels.
#[test]
#[ignore = "needs the model and a reference dump"]
fn the_network_matches_the_reference() {
    let (net, config) = load();
    let trace = trace();
    let side = config.target_size[0];
    let data = blob("input");
    assert_eq!(data.len(), 3 * side * side, "the dump is a {side}² input");

    let tensor = input(&data, side, &Device::Cpu).expect("the input");
    let prediction = net.forward(&tensor).expect("the forward pass");

    let expected = blob("boxes");
    let rows = expected.len() / 6;
    let got = decode(&prediction, page_size(&trace));
    assert_eq!(got.len(), rows, "the reference kept {rows} boxes");

    // Compared as a *set*, not position by position. Two boxes of nearly equal
    // score can trade places in the ranking on the last bit of a logit, and
    // that says nothing about the port; what has to hold is that the same
    // (class, box, score) triples are there.
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
            .unwrap_or_else(|| {
                panic!("no box of class {class} at all, for one at {bounds:?}")
            });
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
    assert!(worst_score < 2e-3, "the worst score is {worst_score} out");
    assert!(
        worst_corner < 2.0,
        "the worst corner is {worst_corner} pixels out"
    );
}

/// Intersection over union of two boxes.
fn overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let width = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let height = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let shared = width * height;
    let area = |r: (f32, f32, f32, f32)| (r.2 - r.0) * (r.3 - r.1);
    let union = area(a) + area(b) - shared;
    if union <= 0.0 { 0.0 } else { shared / union }
}

/// The page marks up into something once the post-processing has run.
#[test]
#[ignore = "needs the model and a reference dump"]
fn the_page_marks_up() {
    let (net, config) = load();
    let trace = trace();
    let side = config.target_size[0];
    let tensor = input(&blob("input"), side, &Device::Cpu).expect("the input");
    let prediction = net.forward(&tensor).expect("the forward pass");
    let (width, height) = page_size(&trace);
    let raw = decode(&prediction, (width, height));
    let regions = post::regions(
        &raw,
        (width as u32, height as u32),
        &post::Settings::default(),
    );
    assert!(
        !regions.is_empty(),
        "the page should mark up into something"
    );
}
