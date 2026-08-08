//! Pipeline tests. The end-to-end one is `#[ignore]`d: it needs the models in
//! `~/.trakktor/ocr/paddle/` and the page in `tmp/ocr/golden/`.

use std::path::PathBuf;

use super::{Device, Engine, Options, sort_boxes, straddling};
use crate::ocr::{
    layout::region::{Label, Region},
    page::Quad,
};

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> Quad {
    Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

#[test]
fn boxes_sort_into_rows_then_columns() {
    let mut boxes = vec![
        (quad(500.0, 100.0, 600.0, 130.0), 1.0),
        (quad(100.0, 300.0, 200.0, 330.0), 1.0),
        // Four pixels above the first one: the same row, and to its left.
        (quad(100.0, 96.0, 200.0, 126.0), 1.0),
    ];
    sort_boxes(&mut boxes);
    let xs: Vec<f32> = boxes.iter().map(|b| b.0.points[0].0).collect();
    let ys: Vec<f32> = boxes.iter().map(|b| b.0.points[0].1).collect();
    assert_eq!(ys, vec![96.0, 100.0, 300.0]);
    assert_eq!(xs, vec![100.0, 500.0, 100.0]);
}

#[test]
fn a_row_that_is_far_below_never_swaps_up() {
    let mut boxes = vec![
        (quad(500.0, 100.0, 600.0, 130.0), 1.0),
        (quad(100.0, 130.0, 200.0, 160.0), 1.0),
    ];
    sort_boxes(&mut boxes);
    assert_eq!(boxes[0].0.points[0].0, 500.0);
}

/// The glued line stays in the batch beside its pieces: which of the two is
/// kept is decided on what the recognizer returns, so both have to be read.
#[test]
fn a_straddling_box_is_read_whole_as_well_as_in_pieces() {
    let region = |x0: f32, x1: f32| Region {
        label: Label::Text,
        score: 0.9,
        quad: quad(x0, 500.0, x1, 900.0),
    };
    let regions = vec![region(100.0, 700.0), region(760.0, 1400.0)];
    let mut boxes = vec![
        (quad(100.0, 600.0, 1400.0, 650.0), 1.0),
        (quad(100.0, 700.0, 690.0, 750.0), 1.0),
    ];
    let cuts = straddling(&mut boxes, &regions);

    assert_eq!(cuts.len(), 1);
    assert_eq!(cuts[0].whole, 0);
    assert_eq!(cuts[0].pieces, 2..4);
    assert_eq!(boxes.len(), 4);
    assert_eq!(boxes[0].0, quad(100.0, 600.0, 1400.0, 650.0));
    assert_eq!(boxes[2].0, quad(100.0, 600.0, 730.0, 650.0));
    assert_eq!(boxes[3].0, quad(730.0, 600.0, 1400.0, 650.0));

    // Without a layout there is nothing to cut against.
    let mut alone = boxes[..1].to_vec();
    assert!(straddling(&mut alone, &[]).is_empty());
    assert_eq!(alone.len(), 1);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle and tmp/ocr/golden"]
fn reads_the_reference_page() {
    let models =
        PathBuf::from(std::env::var("HOME").expect("HOME")).join(".trakktor");
    let page = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/ocr/golden/page.png");

    let mut noop = |_: &str, _: u64, _: Option<u64>| {};
    let engine = Engine::load(
        &models,
        Device::Cpu,
        Options {
            drop_score: 0.0,
            ..Options::default()
        },
        &mut noop,
    )
    .unwrap();
    let read = engine.read_file(&page, 1).unwrap();

    // The whole page, line for line, against what the reference read — this
    // is the layer of parity that matters to a user.
    let trace: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../tmp/ocr/golden/mobile_eslav/trace.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let expected: Vec<&str> = trace["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|line| line["text"].as_str().unwrap())
        .collect();
    let got: Vec<&str> =
        read.page.lines.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(got, expected);
    assert_eq!(read.crops.len(), read.page.lines.len());
}
