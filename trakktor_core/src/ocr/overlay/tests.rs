use image::RgbImage;

use super::{
    CAPTION_INK, CHANNELS, CONFIDENT, DOUBTFUL, Rgb, Shape, Weight, draw_png,
    font, score_color,
};
use crate::ocr::page::Quad;

/// The page every test draws on: a flat colour that is neither grey nor
/// symmetric, so a channel swapped anywhere shows up as a wrong pixel.
const PAPER_BGR: Rgb = [10, 20, 30];

/// The same colour the way it must come back out.
const PAPER_RGB: Rgb = [30, 20, 10];

/// An axis-aligned box, clockwise from the top left.
fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Quad {
    Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

/// Draws over a blank page and reads the result back as pixels.
fn drawn(width: usize, height: usize, shapes: &[Shape]) -> RgbImage {
    let page: Vec<u8> = PAPER_BGR
        .iter()
        .copied()
        .cycle()
        .take(width * height * CHANNELS)
        .collect();
    let png = draw_png(&page, width, height, shapes).expect("a drawable page");
    image::load_from_memory(&png)
        .expect("the overlay encodes as PNG")
        .to_rgb8()
}

/// How many pixels of this colour a rectangle of the result holds.
fn count(page: &RgbImage, area: (u32, u32, u32, u32), color: Rgb) -> usize {
    let (x0, y0, x1, y1) = area;
    (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .filter(|(x, y)| page.get_pixel(*x, *y).0 == color)
        .count()
}

#[test]
fn a_page_with_nothing_on_it_comes_back_unchanged() {
    let page = drawn(16, 9, &[]);
    assert_eq!((page.width(), page.height()), (16, 9));
    assert_eq!(count(&page, (0, 0, 16, 9), PAPER_RGB), 16 * 9);
}

#[test]
fn an_outline_lands_on_the_edges_and_leaves_the_inside_alone() {
    let page = drawn(
        200,
        200,
        &[Shape {
            quad: rect(50.0, 60.0, 150.0, 120.0),
            color: CONFIDENT,
            weight: Weight::Thick,
            caption: None,
        }],
    );
    for edge in [(100, 60), (100, 120), (50, 90), (150, 90)] {
        assert_eq!(page.get_pixel(edge.0, edge.1).0, CONFIDENT, "at {edge:?}");
    }
    // The text the box is drawn around must stay readable underneath it.
    assert_eq!(count(&page, (55, 65, 145, 115), PAPER_RGB), 90 * 50);
}

#[test]
fn a_caption_hangs_off_the_first_corner_on_a_plate_of_its_own_colour() {
    // A 200-pixel page draws one-pixel cells, so the plate is 7 by 9 for a
    // one-character caption: five cells of glyph and one of clearance. It
    // hangs up and to the left of the corner, where the margin is.
    let page = drawn(
        200,
        200,
        &[Shape {
            quad: rect(50.0, 60.0, 150.0, 120.0),
            color: DOUBTFUL,
            weight: Weight::Thick,
            caption: Some("7".into()),
        }],
    );
    let plate = (43, 51, 50, 60);
    assert_eq!(count(&page, plate, PAPER_RGB), 0, "the plate is opaque");
    assert!(count(&page, plate, CAPTION_INK) > 0, "something is written");
    assert!(
        count(&page, plate, DOUBTFUL) > 0,
        "on the shape's own colour"
    );
    // …and nothing at all above it.
    assert_eq!(count(&page, (0, 0, 200, 51), PAPER_RGB), 200 * 51);
}

#[test]
fn a_caption_with_no_room_above_drops_into_the_box() {
    let page = drawn(
        200,
        200,
        &[Shape {
            quad: rect(50.0, 2.0, 150.0, 60.0),
            color: CONFIDENT,
            weight: Weight::Thick,
            caption: Some("7".into()),
        }],
    );
    assert_eq!(count(&page, (43, 2, 50, 11), PAPER_RGB), 0);
    assert!(count(&page, (43, 2, 50, 11), CAPTION_INK) > 0);
}

#[test]
fn a_caption_shrinks_to_the_box_it_belongs_to() {
    // On this page a cell is four pixels, so a two-character caption wants 52
    // of them; the narrow box is 20 wide and gets the smallest cell allowed,
    // which is half the page's own.
    let narrow = Shape {
        quad: rect(1000.0, 1000.0, 1020.0, 1040.0),
        color: CONFIDENT,
        weight: Weight::Thick,
        caption: Some("12".into()),
    };
    let roomy = Shape {
        quad: rect(100.0, 1000.0, 600.0, 1040.0),
        color: CONFIDENT,
        weight: Weight::Thick,
        caption: Some("12".into()),
    };
    let page = drawn(2100, 2100, &[narrow, roomy]);

    // A band above the boxes, clear of the outlines themselves. Both plates
    // hang to the left of their box, so both end where the box starts.
    let band = |x0, x1| (x0, 991, x1, 997);
    assert_eq!(count(&page, band(974, 1000), PAPER_RGB), 0);
    assert_eq!(
        count(&page, band(1000, 2100), PAPER_RGB),
        6 * (2100 - 1000),
        "the caption is no wider than twice the box it belongs to"
    );
    // The roomy box keeps the page's own size: 13 cells of four pixels.
    assert_eq!(count(&page, band(48, 100), PAPER_RGB), 0);
    assert_eq!(count(&page, band(100, 600), PAPER_RGB), 6 * 500);
}

#[test]
fn a_shape_hanging_off_the_page_is_clipped_rather_than_refused() {
    let page = drawn(
        64,
        64,
        &[Shape {
            quad: rect(-40.0, -40.0, 30.0, 30.0),
            color: CONFIDENT,
            weight: Weight::Thin,
            caption: Some("0".into()),
        }],
    );
    assert_eq!((page.width(), page.height()), (64, 64));
    assert_eq!(
        page.get_pixel(30, 10).0,
        CONFIDENT,
        "the edge that is on it"
    );
}

#[test]
fn a_degenerate_quadrangle_draws_nothing_and_panics_at_nothing() {
    let page = drawn(
        32,
        32,
        &[
            Shape {
                quad: rect(10.0, 10.0, 10.0, 10.0),
                color: CONFIDENT,
                weight: Weight::Thick,
                caption: Some(String::new()),
            },
            Shape {
                quad: rect(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
                color: CONFIDENT,
                weight: Weight::Thin,
                caption: Some("1".into()),
            },
        ],
    );
    // All that is left of a box with no area is the stamp under its corner.
    assert_eq!(count(&page, (9, 9, 11, 11), CONFIDENT), 4);
    assert_eq!(count(&page, (0, 0, 32, 32), PAPER_RGB), 32 * 32 - 4);
}

#[test]
fn a_raster_too_short_for_the_page_it_claims_is_refused() {
    let page = vec![0u8; 10 * 10 * CHANNELS - 1];
    assert!(draw_png(&page, 10, 10, &[]).is_err());
    assert!(draw_png(&[], 0, 0, &[]).is_err());
}

#[test]
fn confidence_picks_the_colour_at_a_round_half() {
    assert_eq!(score_color(0.5), CONFIDENT);
    assert_eq!(score_color(0.499), DOUBTFUL);
    assert_eq!(score_color(1.0), CONFIDENT);
    assert_eq!(score_color(0.0), DOUBTFUL);
}

#[test]
fn the_alphabet_spells_what_it_can_and_says_so_when_it_cannot() {
    assert_eq!(font::glyph('a'), font::glyph('A'));
    assert_eq!(font::glyph('Ж'), font::glyph('?'));
    assert_ne!(font::glyph('0'), font::glyph('?'));
    assert_eq!(font::glyph(' '), [0; font::HEIGHT]);
    // Every glyph fits the five cells the drawing gives it.
    for c in ('A'..='Z').chain('0'..='9').chain(" .,-_:/%?".chars()) {
        let rows = font::glyph(c);
        assert!(rows.iter().all(|row| *row >> font::WIDTH == 0), "{c}");
    }
}

#[test]
fn a_caption_is_as_wide_as_its_characters_and_the_gaps_between_them() {
    assert_eq!(font::width(0), 0);
    assert_eq!(font::width(1), font::WIDTH);
    assert_eq!(font::width(3), font::WIDTH * 3 + font::GAP * 2);
}
