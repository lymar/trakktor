//! Detector parity against a reference run.
//!
//! These need the golden dumps in `tmp/ocr/golden/<detector>_<recognizer>/`
//! and the artifacts in `~/.trakktor/ocr/paddle/`, so they are `#[ignore]`d.
//! The dumps are a page rendered for this purpose — mixed Russian and English,
//! a small type size, a short line and a line set at an angle — pushed through
//! the reference pipeline stage by stage.
//!
//! The three tests are the three layers the port is held to: the input tensor
//! and the probability map are compared numerically, the boxes by overlap.
//!
//! One test compares nothing against the reference and is here for a different
//! reason: the large detector convolves with a kernel big enough that the Metal
//! backend used to return a wrong answer without saying so. That failure has no
//! symptom other than the page reading differently, so it is checked directly
//! — the same input on both devices, and the two maps must agree.

use std::path::PathBuf;

use candle_core::{Device, Tensor};

use super::Detector;
use crate::ocr::paddle::{
    artifact::Artifact,
    db::{self, Params},
    image::{self, LimitType, Page},
    net::Loader,
};

/// The dumps of one reference run.
fn golden(run: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/ocr/golden")
        .join(run)
}

fn golden_dir() -> PathBuf { golden("mobile_eslav") }

fn page_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/ocr/golden/page.png")
}

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle")
        .join(name)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn trace_of(run: &str) -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(golden(run).join("trace.json")).unwrap(),
    )
    .unwrap()
}

/// Runs one detector over the input tensor a reference run was given and
/// compares the probability map it returns with the one that run recorded.
///
/// The map is a sigmoid, so most of it sits in the flat tail where large logit
/// differences vanish; what the port is held to is the part that decides
/// anything — no pixel may fall on the other side of the binarization
/// threshold.
fn probability_map_matches(run: &str, model: &str) {
    let trace = trace_of(run);
    let shape = trace["det_input_shape"].as_array().unwrap();
    let (height, width) = (
        shape[2].as_u64().unwrap() as usize,
        shape[3].as_u64().unwrap() as usize,
    );

    let device = Device::Cpu;
    let artifact = Artifact::load(&model_dir(model)).unwrap();
    let detector = Detector::load(&Loader::new(&artifact, &device)).unwrap();

    let golden_input = read_f32(&golden_dir().join("det_input.bin"));
    let input =
        Tensor::from_vec(golden_input, (1, 3, height, width), &device).unwrap();
    let map = detector.forward(&input).unwrap();
    assert_eq!(map.dims(), &[1, 1, height, width]);

    let ours = map.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let reference = read_f32(&golden(run).join("det_prob.bin"));
    assert_eq!(ours.len(), reference.len());

    let worst = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let big = ours
        .iter()
        .zip(&reference)
        .filter(|(a, b)| (**a - **b).abs() > 1e-5)
        .count();
    eprintln!(
        "{model}: worst {worst:e}, over 1e-5: {big} of {}",
        ours.len()
    );
    assert!(worst < 1e-3, "worst probability differs by {worst}");

    let threshold = trace["det_thresh"].as_f64().unwrap() as f32;
    // A pixel may only land on the other side of the threshold if it was
    // sitting on the line to begin with — inside the arithmetic noise the two
    // implementations differ by. Further out than that is a real disagreement,
    // however small the count.
    let mut on_the_line = 0usize;
    for (ours, reference) in ours.iter().zip(&reference) {
        if (*ours >= threshold) == (*reference >= threshold) {
            continue;
        }
        assert!(
            (reference - threshold).abs() <= worst,
            "a pixel {} from the threshold reads {ours} against {reference}",
            (reference - threshold).abs()
        );
        on_the_line += 1;
    }
    let above = reference.iter().filter(|p| **p >= threshold).count();
    eprintln!(
        "{model}: {on_the_line} of {above} pixels above the threshold sit on \
         the line"
    );
    assert!(
        on_the_line * 1000 <= above,
        "{on_the_line} of {above} pixels are on the line, which is not a \
         handful"
    );
}

/// The side-length limit the golden run used.
const LIMIT_SIDE_LEN: usize = 960;

#[test]
#[ignore = "needs tmp/ocr/golden"]
fn preprocessing_matches_the_reference_tensor() {
    let page = Page::load(&page_path()).unwrap();
    let input = image::detector_input(
        &page,
        LIMIT_SIDE_LEN,
        LimitType::Max,
        &image::DETECTOR_NORMALIZE,
    );
    let golden = read_f32(&golden_dir().join("det_input.bin"));

    assert_eq!(input.data.len(), golden.len(), "tensor length");
    let worst = input
        .data
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The resize is the reference's fixed-point bilinear and the
    // normalization is the same three operations in the same order, so the
    // tensors agree to within the resampler's last unit: a couple of the 256
    // brightness levels on about one sample in a hundred, which is
    // `2 / 255 / 0.225` in normalized units. Chasing the last unit is not
    // worth it — the library the reference resizes with dispatches to a
    // different vectorized kernel per platform and per build, so there is no
    // single answer to be bit-exact against.
    let tolerance = 2.5 / 255.0 / 0.224;
    assert!(worst < tolerance, "worst element differs by {worst}");
}

#[test]
#[ignore = "needs tmp/ocr/golden and ~/.trakktor/ocr/paddle"]
fn the_probability_map_matches_the_reference() {
    probability_map_matches("mobile_eslav", "PP-OCRv5_mobile_det");
}

#[test]
#[ignore = "needs tmp/ocr/golden and ~/.trakktor/ocr/paddle"]
fn the_server_probability_map_matches_the_reference() {
    probability_map_matches("server_eslav", "PP-OCRv5_server_det");
}

#[test]
#[ignore = "needs tmp/ocr/golden and ~/.trakktor/ocr/paddle"]
fn the_medium_probability_map_matches_the_reference() {
    probability_map_matches("medium_v6", "PP-OCRv6_medium_det");
}

/// The large detector on a page-sized input, on both devices.
///
/// The size is the point: it is a scanned A4 page at `--limit-side-len 1920`,
/// which is where the convolutions grow past what the GPU backend computes
/// correctly. A smaller input proves nothing here.
#[cfg(feature = "ocr-metal")]
#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle and a Metal device"]
fn the_server_detector_agrees_between_devices() {
    let (height, width) = (1920usize, 1376usize);
    let page: Vec<f32> = (0..3 * height * width)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect();

    let map = |device: Device| {
        let artifact =
            Artifact::load(&model_dir("PP-OCRv5_server_det")).unwrap();
        let detector =
            Detector::load(&Loader::new(&artifact, &device)).unwrap();
        let input =
            Tensor::from_vec(page.clone(), (1, 3, height, width), &device)
                .unwrap();
        detector
            .forward(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };

    let cpu = map(Device::Cpu);
    let metal = map(Device::new_metal(0).expect("a Metal device"));
    let worst = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cpu vs metal: worst {worst:e} over {} pixels", cpu.len());
    assert!(worst < 1e-3, "the two devices disagree by {worst}");
}

/// Intersection over union of two axis-aligned bounds.
fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let overlap = (x1 - x0) * (y1 - y0);
    let area = |r: (f32, f32, f32, f32)| (r.2 - r.0) * (r.3 - r.1);
    overlap / (area(a) + area(b) - overlap)
}

#[test]
#[ignore = "needs tmp/ocr/golden"]
fn the_boxes_match_the_reference_by_overlap() { boxes_match("mobile_eslav"); }

#[test]
#[ignore = "needs tmp/ocr/golden"]
fn the_server_boxes_match_the_reference_by_overlap() {
    boxes_match("server_eslav");
}

#[test]
#[ignore = "needs tmp/ocr/golden"]
fn the_medium_boxes_match_the_reference_by_overlap() {
    boxes_match("medium_v6");
}

/// Turns one reference run's probability map into boxes and compares them with
/// the boxes that run recorded.
fn boxes_match(run: &str) {
    let trace = trace_of(run);
    let shape = trace["det_prob_shape"].as_array().unwrap();
    let (height, width) = (
        shape[2].as_u64().unwrap() as usize,
        shape[3].as_u64().unwrap() as usize,
    );
    let source = (
        trace["page_size"][0].as_u64().unwrap() as u32,
        trace["page_size"][1].as_u64().unwrap() as u32,
    );

    // The thresholds the run was taken at, which are the model's own: the
    // newest detector's map is calibrated differently from the older ones'.
    let number = |key: &str, fallback: f32| {
        trace[key].as_f64().map_or(fallback, |v| v as f32)
    };
    let params = Params {
        thresh: number("det_thresh", 0.3),
        box_thresh: number("det_box_thresh", 0.6),
        unclip_ratio: number("det_unclip_ratio", 1.5),
        ..Params::default()
    };

    let prob = read_f32(&golden(run).join("det_prob.bin"));
    let boxes = db::boxes_from_bitmap(
        &prob,
        width,
        height,
        source,
        (1.0, 1.0),
        &params,
    );

    let golden: Vec<(f32, f32, f32, f32)> = trace["boxes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|quad| {
            let points: Vec<(f32, f32)> = quad
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        p[0].as_f64().unwrap() as f32,
                        p[1].as_f64().unwrap() as f32,
                    )
                })
                .collect();
            let xs: Vec<f32> = points.iter().map(|p| p.0).collect();
            let ys: Vec<f32> = points.iter().map(|p| p.1).collect();
            (
                xs.iter().copied().fold(f32::INFINITY, f32::min),
                ys.iter().copied().fold(f32::INFINITY, f32::min),
                xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        })
        .collect();

    assert_eq!(boxes.len(), golden.len(), "number of boxes");

    // Geometry is compared with a tolerance: the reference's contour tracing,
    // minimum-area rectangle and polygon offset each carry their own rounding,
    // and a box that sits a pixel over still crops the same line.
    let mut worst = 1.0f32;
    for reference in &golden {
        let best = boxes
            .iter()
            .map(|(quad, _)| iou(quad.bounds(), *reference))
            .fold(0.0f32, f32::max);
        let ours = boxes
            .iter()
            .max_by(|a, b| {
                iou(a.0.bounds(), *reference)
                    .total_cmp(&iou(b.0.bounds(), *reference))
            })
            .map(|(quad, _)| quad.bounds());
        eprintln!("ref {reference:?} ours {ours:?} iou {best:.3}");
        worst = worst.min(best);
    }
    assert!(worst > 0.9, "worst overlap {worst:.3}");
}

#[test]
#[ignore = "needs tmp/ocr/golden"]
fn the_resize_matches_opencv() {
    let page = Page::load(&page_path()).unwrap();
    let ours = image::resize_bgr(&page, 960, 704);
    let golden = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tmp/ocr/golden/resized_960x704.bgr"),
    )
    .unwrap();
    assert_eq!(ours.len(), golden.len());

    let mut worst = 0i32;
    let mut differing = 0usize;
    let mut first = None;
    for (i, (a, b)) in ours.iter().zip(&golden).enumerate() {
        let d = i32::from(*a) - i32::from(*b);
        if d != 0 {
            differing += 1;
            if first.is_none() {
                let pixel = i / 3;
                first = Some((pixel % 960, pixel / 960, i % 3, *a, *b));
            }
        }
        worst = worst.max(d.abs());
    }
    eprintln!(
        "resize: worst {worst}, differing {differing} of {}, first {first:?}",
        golden.len()
    );
    // Not an exact match, and deliberately not asserted as one: the reference
    // resizes through a vectorized kernel chosen per platform and per build,
    // so "the" reference output is not a single number. What must hold is
    // that the deviation stays at the resampler's last unit — a couple of
    // brightness levels on a small minority of samples.
    assert!(worst <= 2, "resize differs by {worst} levels");
    assert!(
        differing * 50 < golden.len(),
        "{differing} of {} samples differ",
        golden.len()
    );
}

#[test]
fn a_solid_rectangle_keeps_its_pixel_bounds() {
    // A filled rectangle from (58, 431) to (138, 441) inclusive: the contour
    // and its minimum-area rectangle must land on those pixels, not on the
    // cracks around them.
    let (w, h) = (200usize, 500usize);
    let mut mask = vec![false; w * h];
    for y in 431..=441 {
        for x in 58..=138 {
            mask[y * w + x] = true;
        }
    }
    let contours = crate::ocr::paddle::db::contour::find_contours(&mask, w, h);
    assert_eq!(contours.len(), 1);
    let (corners, sside) =
        crate::ocr::paddle::db::contour::mini_box(&contours[0].points).unwrap();
    let xs: Vec<f32> = corners.iter().map(|p| p.0).collect();
    let ys: Vec<f32> = corners.iter().map(|p| p.1).collect();
    eprintln!("corners {corners:?} sside {sside}");
    assert_eq!(xs.iter().cloned().fold(f32::INFINITY, f32::min), 58.0);
    assert_eq!(xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max), 138.0);
    assert_eq!(ys.iter().cloned().fold(f32::INFINITY, f32::min), 431.0);
    assert_eq!(ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max), 441.0);
    assert_eq!(sside, 10.0);
}
