//! The box bookkeeping, on made-up boxes.
//!
//! Each test states one rule of the reference's post-processing; together they
//! pin down the order the rules run in, which is the part that is easy to get
//! wrong and impossible to see in the output.

use super::*;

fn raw(class: Label, score: f32, bounds: (f32, f32, f32, f32)) -> Raw {
    Raw {
        class: class.index(),
        score,
        bounds,
    }
}

const PAGE: (u32, u32) = (1000, 1400);

#[test]
fn a_box_under_its_own_floor_is_dropped() {
    let settings = Settings::default();
    // Running text clears 0.4 and a picture does not clear 0.5 at the same
    // score: the floors are per class.
    let boxes = [
        raw(Label::Text, 0.45, (10.0, 10.0, 200.0, 60.0)),
        raw(Label::Image, 0.45, (10.0, 200.0, 200.0, 400.0)),
    ];
    let regions = regions(&boxes, PAGE, &settings);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].label, Label::Text);
}

#[test]
fn a_flat_threshold_replaces_every_floor() {
    let settings = Settings {
        threshold: Some(0.9),
        ..Settings::default()
    };
    let boxes = [raw(Label::ParagraphTitle, 0.5, (10.0, 10.0, 200.0, 60.0))];
    assert!(regions(&boxes, PAGE, &settings).is_empty());
}

#[test]
fn duplicates_of_one_class_suppress_each_other() {
    let settings = Settings::default();
    let boxes = [
        raw(Label::Text, 0.9, (10.0, 10.0, 210.0, 60.0)),
        raw(Label::Text, 0.8, (12.0, 11.0, 208.0, 61.0)),
    ];
    assert_eq!(regions(&boxes, PAGE, &settings).len(), 1);
}

#[test]
fn two_classes_on_the_same_spot_both_survive() {
    // The bar for a different class is 0.98, which two genuinely different
    // blocks never reach — a formula inside a paragraph must not delete it.
    let settings = Settings::default();
    let boxes = [
        raw(Label::Text, 0.9, (10.0, 10.0, 210.0, 60.0)),
        raw(Label::Formula, 0.8, (14.0, 12.0, 206.0, 58.0)),
    ];
    assert_eq!(regions(&boxes, PAGE, &settings).len(), 2);
}

#[test]
fn a_picture_covering_the_page_is_the_scan_itself() {
    let settings = Settings::default();
    let boxes = [
        raw(Label::Image, 0.99, (0.0, 0.0, 1000.0, 1400.0)),
        raw(Label::Text, 0.9, (10.0, 10.0, 210.0, 60.0)),
    ];
    let regions = regions(&boxes, PAGE, &settings);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].label, Label::Text);
}

#[test]
fn a_page_that_really_is_one_picture_keeps_it() {
    let settings = Settings::default();
    let boxes = [raw(Label::Image, 0.99, (0.0, 0.0, 1000.0, 1400.0))];
    assert_eq!(regions(&boxes, PAGE, &settings).len(), 1);
}

#[test]
fn a_swallowing_class_eats_what_sits_inside_it() {
    let settings = Settings::default();
    let boxes = [
        raw(Label::Image, 0.9, (10.0, 10.0, 400.0, 400.0)),
        raw(Label::Text, 0.9, (20.0, 20.0, 380.0, 380.0)),
    ];
    let regions = regions(&boxes, PAGE, &settings);
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].label, Label::Image);
}

#[test]
fn a_formula_is_never_eaten_by_anything_else() {
    let settings = Settings::default();
    let boxes = [
        raw(Label::Image, 0.9, (10.0, 10.0, 400.0, 400.0)),
        raw(Label::Formula, 0.9, (20.0, 20.0, 380.0, 380.0)),
    ];
    assert_eq!(regions(&boxes, PAGE, &settings).len(), 2);
}

#[test]
fn a_text_block_does_not_swallow_what_sits_inside_it() {
    let settings = Settings::default();
    let boxes = [
        raw(Label::Text, 0.9, (10.0, 10.0, 400.0, 400.0)),
        raw(Label::FigureTitle, 0.9, (20.0, 20.0, 380.0, 380.0)),
    ];
    assert_eq!(regions(&boxes, PAGE, &settings).len(), 2);
}

#[test]
fn widening_scales_about_the_centre() {
    let settings = Settings {
        unclip: (2.0, 1.0),
        ..Settings::default()
    };
    let boxes = [raw(Label::Text, 0.9, (100.0, 100.0, 200.0, 200.0))];
    let regions = regions(&boxes, PAGE, &settings);
    let (x0, y0, x1, y1) = regions[0].bounds();
    assert_eq!((x0, x1), (50.0, 250.0));
    assert_eq!((y0, y1), (100.0, 200.0));
}

#[test]
fn a_box_is_clipped_to_the_page() {
    let settings = Settings::default();
    let boxes = [raw(Label::Text, 0.9, (-50.0, -20.0, 1200.0, 500.0))];
    let regions = regions(&boxes, PAGE, &settings);
    let (x0, y0, x1, _) = regions[0].bounds();
    assert_eq!((x0, y0, x1), (0.0, 0.0, PAGE.0 as f32));
}

#[test]
fn a_box_entirely_off_the_page_is_dropped() {
    let settings = Settings::default();
    let boxes = [raw(Label::Text, 0.9, (1100.0, 10.0, 1200.0, 60.0))];
    assert!(regions(&boxes, PAGE, &settings).is_empty());
}
