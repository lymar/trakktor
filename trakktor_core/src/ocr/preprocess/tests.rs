//! Parity against the reference implementation, on a photograph.
//!
//! Both models are checked against what PaddleOCR's own pipeline produces for
//! the same file: the classifier on all four right angles of it, and the
//! unwarper on the straightened page pixel by pixel. Four readings rather than
//! one because a port that has the classes in the wrong order still agrees
//! with the reference on an upright page.
//!
//! The fixtures live under `tmp/ocr/preprocess/` and are not in the
//! repository; the tests skip themselves when they are absent.

use std::path::{Path, PathBuf};

use super::*;
use crate::ocr::paddle::{artifact::Artifact, image::Page, net::Loader};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tmp/ocr/preprocess")
        .join(name)
}

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".trakktor/ocr/preprocess")
        .join(name)
}

fn load_orientation() -> Option<orientation::Classifier> {
    let dir = model_dir(model::ORIENTATION);
    if !dir.is_dir() {
        return None;
    }
    let artifact = Artifact::load(&dir).unwrap();
    let loader = Loader::new(&artifact, &candle_core::Device::Cpu);
    Some(orientation::Classifier::load(&loader).unwrap())
}

fn load_unwarper() -> Option<unwarp::Unwarper> {
    let dir = model_dir(model::UNWARP);
    if !dir.is_dir() {
        return None;
    }
    let artifact = Artifact::load(&dir).unwrap();
    let loader = Loader::new(&artifact, &candle_core::Device::Cpu);
    Some(unwarp::Unwarper::load(&loader).unwrap())
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/preprocess and tmp/ocr/preprocess"]
fn the_classifier_answers_the_reference_at_every_right_angle() {
    let photo = fixture("photo.png");
    let Some(classifier) = load_orientation() else {
        return;
    };
    if !photo.is_file() {
        return;
    }
    let page = Page::load(&photo).unwrap();

    // The photograph is upright, so each turn of it must come back as the
    // turn that undoes it: a page turned a quarter counter-clockwise needs
    // three quarters more.
    for (applied, expected) in [(0u16, 0u16), (90, 270), (180, 180), (270, 90)]
    {
        let turned = orientation::turn(&page, applied);
        let reading = classifier.read(&turned).unwrap().expect("a reading");
        assert_eq!(
            reading.turn, expected,
            "a page turned {applied} degrees answered {}",
            reading.turn
        );
        // The reference is sure of this page to within a hundredth of the
        // same number at every angle; anything much lower means the port is
        // reading a differently prepared square.
        assert!(
            reading.score > 0.85,
            "unsure at {applied}: {}",
            reading.score
        );
    }
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/preprocess and tmp/ocr/preprocess"]
fn the_unwarper_reproduces_the_reference_page() {
    let (photo, expected) = (fixture("photo.png"), fixture("unwarped.png"));
    let Some(unwarper) = load_unwarper() else {
        return;
    };
    if !photo.is_file() || !expected.is_file() {
        return;
    }

    let page = Page::load(&photo).unwrap();
    let field = unwarper.field(&page).unwrap();
    let backmap =
        Backmap::new(field, page.width as usize, page.height as usize);
    let straightened = backmap.apply(&page);
    let reference = Page::load(&expected).unwrap();

    assert_eq!(straightened.width, reference.width);
    assert_eq!(straightened.height, reference.height);

    let mut worst = 0i32;
    let mut sum = 0f64;
    for (ours, theirs) in straightened.bgr.iter().zip(reference.bgr.iter()) {
        let difference = i32::from(*ours) - i32::from(*theirs);
        worst = worst.max(difference.abs());
        sum += f64::from(difference.abs());
    }
    let mean = sum / straightened.bgr.len() as f64;
    // Not byte for byte, and it cannot be: the reference truncates the
    // straightened page to bytes where this port rounds, so a level is the
    // whole of the disagreement.
    assert!(worst <= 1, "worst difference {worst}");
    assert!(mean < 0.5, "mean difference {mean}");
}
