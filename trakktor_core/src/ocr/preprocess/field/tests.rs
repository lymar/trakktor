use super::*;
use crate::ocr::preprocess::unwarp::{FIELD_HEIGHT, FIELD_WIDTH};

/// A field that asks for the page exactly as it is: every cell points at its
/// own place, so the straightened page must come back byte for byte.
fn identity(width: usize, height: usize) -> Backmap {
    let mut field = vec![0f32; 2 * FIELD_HEIGHT * FIELD_WIDTH];
    let plane = FIELD_HEIGHT * FIELD_WIDTH;
    for y in 0..FIELD_HEIGHT {
        for x in 0..FIELD_WIDTH {
            let at = y * FIELD_WIDTH + x;
            field[at] = x as f32 / (FIELD_WIDTH - 1) as f32 * 2.0 - 1.0;
            field[plane + at] =
                y as f32 / (FIELD_HEIGHT - 1) as f32 * 2.0 - 1.0;
        }
    }
    Backmap::new(field, width, height)
}

fn ramp(width: usize, height: usize) -> Page {
    let mut bgr = vec![0u8; width * height * CHANNELS];
    for y in 0..height {
        for x in 0..width {
            let at = (y * width + x) * CHANNELS;
            bgr[at] = (x * 7 % 256) as u8;
            bgr[at + 1] = (y * 5 % 256) as u8;
            bgr[at + 2] = ((x + y) % 256) as u8;
        }
    }
    Page {
        width: width as u32,
        height: height as u32,
        bgr,
    }
}

#[test]
fn an_identity_field_returns_the_page_it_was_given() {
    let page = ramp(40, 60);
    let map = identity(40, 60);
    let same = map.apply(&page);
    assert_eq!(same.width, page.width);
    assert_eq!(same.height, page.height);
    assert_eq!(same.bgr, page.bgr);
}

#[test]
fn an_identity_field_sends_every_point_to_itself() {
    let map = identity(100, 200);
    for (x, y) in [(0.0, 0.0), (99.0, 199.0), (50.0, 100.0), (12.5, 33.25)] {
        let (sx, sy) = map.source(x, y);
        assert!((sx - x).abs() < 1e-3, "{sx} != {x}");
        assert!((sy - y).abs() < 1e-3, "{sy} != {y}");
    }
}

#[test]
fn a_field_pointing_outside_the_photograph_gives_black() {
    let page = ramp(20, 20);
    // Every cell asks for a point well past the right edge.
    let field = vec![9.0f32; 2 * FIELD_HEIGHT * FIELD_WIDTH];
    let map = Backmap::new(field, 20, 20);
    assert!(map.apply(&page).bgr.iter().all(|byte| *byte == 0));
}

#[test]
fn a_quadrangle_comes_back_on_the_photograph() {
    let map = identity(100, 100);
    let quad =
        Quad::new([(10.0, 10.0), (90.0, 10.0), (90.0, 40.0), (10.0, 40.0)]);
    let back = map.unmap(&quad);
    for (a, b) in quad.points.iter().zip(back.points.iter()) {
        assert!((a.0 - b.0).abs() < 1e-2 && (a.1 - b.1).abs() < 1e-2);
    }
}
