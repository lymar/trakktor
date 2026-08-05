//! The stage end to end against the reference, on real pages.
//!
//! This is the second layer of the domain's parity discipline: the network
//! test next door feeds the reference's *own* input tensor, so it says nothing
//! about the resampler; this one starts from the page file and compares the
//! regions that come out.
//!
//! `#[ignore]`d: it needs the model in `~/.trakktor/ocr/layout/` and a dump
//! from the reference in `tmp/ocr/golden/layout/regions.json`, taken with the
//! *pipeline's* thresholds rather than the model's own flat one.

use std::path::PathBuf;

use super::*;
use crate::ocr::layout::region::Label;

fn golden() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/ocr/golden/layout")
}

fn pages() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/ocr/pages")
}

fn detector() -> Detector {
    let models =
        PathBuf::from(std::env::var("HOME").expect("HOME")).join(".trakktor");
    let mut noop = |_: &str, _: u64, _: Option<u64>| {};
    Detector::load(
        &models,
        Device::Cpu,
        Runtime::Candle,
        Options::default(),
        &mut noop,
    )
    .expect("the stage")
}

fn overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let width = (a.2.min(b.2) - a.0.max(b.0)).max(0.0);
    let height = (a.3.min(b.3) - a.1.max(b.1)).max(0.0);
    let shared = width * height;
    let area = |r: (f32, f32, f32, f32)| (r.2 - r.0) * (r.3 - r.1);
    let union = area(a) + area(b) - shared;
    if union <= 0.0 { 0.0 } else { shared / union }
}

/// How far above its own floor a region has to score before the comparison
/// insists on it.
///
/// The port's resize is the same bicubic kernel as OpenCV's evaluated in
/// floating point instead of eleven-bit fixed point, so a pixel here and there
/// differs by a level of brightness and every score moves in its last
/// thousandths. A region the reference kept at 0.4004 against a floor of 0.400
/// is therefore free to fall the other way, and insisting on it would be
/// testing the resampler, not the port. Regions this far clear of their floor
/// have to be there.
const CLEAR: f32 = 0.02;

/// Every page of the set marks up the way the reference marks it up.
///
/// Matched as a set with an overlap bar rather than position by position: two
/// boxes of nearly equal score can trade places in the ranking without either
/// being wrong.
#[test]
#[ignore = "needs the model and a reference dump"]
fn the_pages_mark_up_the_way_the_reference_marks_them_up() {
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(golden().join("regions.json")).expect("the dump"),
    )
    .expect("the dump parses");
    let detector = detector();
    let settings = detector.options().post;

    let mut worst_overlap = 1f32;
    let mut at_the_bar = 0usize;
    for (page, boxes) in expected.as_object().expect("a map of pages") {
        let path = pages().join(format!("{page}.png"));
        let ours = detector.detect_file(&path).expect("the stage runs");
        let theirs = boxes.as_array().expect("a list of regions");

        let mut wanted = 0usize;
        for one in theirs {
            let name = one["label"].as_str().expect("a label");
            let label = Label::from_index(
                one["cls_id"].as_u64().expect("a class") as usize,
            )
            .expect("a known class");
            assert_eq!(label.name(), name, "the class list has moved");
            let score = one["score"].as_f64().expect("a score") as f32;
            let bounds = one["box"].as_array().expect("a box");
            let bounds = (
                bounds[0].as_f64().unwrap() as f32,
                bounds[1].as_f64().unwrap() as f32,
                bounds[2].as_f64().unwrap() as f32,
                bounds[3].as_f64().unwrap() as f32,
            );
            if score < settings.floor_for(label) + CLEAR {
                at_the_bar += 1;
                continue;
            }
            wanted += 1;
            let best = ours
                .iter()
                .filter(|region| region.label == label)
                .map(|region| overlap(region.bounds(), bounds))
                .fold(0f32, f32::max);
            assert!(
                best > 0.98,
                "{page}: the closest `{name}` to {bounds:?} overlaps by {best}"
            );
            worst_overlap = worst_overlap.min(best);
        }

        // And nothing invented: every region of ours that is clear of its own
        // floor has to be one of theirs.
        let invented = ours
            .iter()
            .filter(|region| {
                region.score >= settings.floor_for(region.label) + CLEAR
            })
            .count();
        assert_eq!(
            invented, wanted,
            "{page}: {invented} regions clear of the bar against {wanted}"
        );
    }
    println!(
        "worst overlap over the set: {worst_overlap:.4}; {at_the_bar} regions \
         sat within {CLEAR} of their floor and were not compared"
    );
}

/// The stage reports what it ran, which is what the result envelope prints.
#[test]
#[ignore = "needs the model"]
fn the_stage_names_its_model() {
    assert_eq!(detector().model(), super::model::DEFAULT_MODEL);
}

/// A page with no layout at all comes back empty rather than failing: the
/// stage is an improvement, not a precondition.
#[test]
#[ignore = "needs the model"]
fn a_blank_page_marks_up_into_nothing() {
    let detector = detector();
    let blank =
        ::image::RgbImage::from_pixel(600, 800, ::image::Rgb([255u8; 3]));
    let page = crate::ocr::paddle::image::Page::from_rgb8(&blank);
    let regions = detector.detect(&page).expect("the stage runs");
    assert!(regions.is_empty(), "a blank page found {regions:?}");
}

/// Labels are what the caller keys off, so the round trip through the class
/// index has to hold for every one of them.
#[test]
fn every_label_survives_its_index() {
    for at in 0..Label::COUNT {
        let label = Label::from_index(at).expect("a label");
        assert_eq!(label.index(), at);
    }
    assert!(Label::from_index(Label::COUNT).is_none());
}

/// A line belongs to the innermost region that holds it.
///
/// The page this is drawn from is real: the model returned nine paragraph boxes
/// **and** a box around the whole column, and the lines of a stacking script
/// poke a few pixels out of every box drawn around them. Picking the region a
/// line overlaps most therefore hands every one of those lines to the column,
/// which then becomes one page-sized block and swallows the reading order.
#[test]
fn a_line_goes_to_the_smallest_region_that_holds_it() {
    use crate::ocr::{layout::region, page::Quad};

    let rect = |x0: f32, y0: f32, x1: f32, y1: f32| {
        Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    };
    let regions = vec![
        Region {
            label: Label::Text,
            score: 0.5,
            quad: rect(245.0, 781.0, 1465.0, 936.0),
        },
        // The column: it covers every line completely.
        Region {
            label: Label::Text,
            score: 0.4,
            quad: rect(236.0, 233.0, 2316.0, 3057.0),
        },
    ];
    // A line that sticks a few pixels out of the paragraph around it.
    let line = (240.0, 772.0, 1476.0, 943.0);
    assert_eq!(region::owner(line, &regions), Some(0));

    // A line only the column holds still goes to the column.
    let stray = (250.0, 2000.0, 1400.0, 2100.0);
    assert_eq!(region::owner(stray, &regions), Some(1));

    // A line nowhere near either belongs to neither, and is read by geometry.
    let outside = (250.0, 3200.0, 400.0, 3260.0);
    assert_eq!(region::owner(outside, &regions), None);
}

/// A picture is never a line's container: text over one is a caption printed on
/// it, not part of it.
#[test]
fn a_picture_never_owns_a_line() {
    use crate::ocr::{layout::region, page::Quad};

    let regions = vec![Region {
        label: Label::Image,
        score: 0.9,
        quad: Quad::new([
            (100.0, 100.0),
            (900.0, 100.0),
            (900.0, 900.0),
            (100.0, 900.0),
        ]),
    }];
    assert_eq!(region::owner((200.0, 200.0, 800.0, 260.0), &regions), None);
}
