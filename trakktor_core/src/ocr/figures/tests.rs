use super::*;

/// The page every fixture is drawn on. Large enough that one mask cell is
/// small against it, so the fractions in [`Settings`] have room to mean
/// something.
const WIDTH: usize = 400;
const HEIGHT: usize = 400;

const BLACK: [u8; CHANNELS] = [0, 0, 0];
const WHITE: [u8; CHANNELS] = [255, 255, 255];

/// A blank page, white to the edges.
fn blank() -> Vec<u8> { vec![255u8; WIDTH * HEIGHT * CHANNELS] }

/// Every filter but the border rule turned off, so that a test can pin a
/// rejection on that rule alone.
fn permissive() -> Settings {
    Settings {
        min_area_fraction: 0.0,
        aspect_range: (0.0, f32::INFINITY),
        ..Settings::default()
    }
}

/// Paints a solid rectangle, `x0..x1` by `y0..y1`.
fn fill(
    page: &mut [u8],
    rect: (usize, usize, usize, usize),
    colour: [u8; CHANNELS],
) {
    let (x0, y0, x1, y1) = rect;
    for y in y0..y1 {
        for x in x0..x1 {
            let at = (y * WIDTH + x) * CHANNELS;
            page[at..at + CHANNELS].copy_from_slice(&colour);
        }
    }
}

/// Paints the outline of a rectangle, `stroke` pixels thick, inwards.
fn outline(
    page: &mut [u8],
    rect: (usize, usize, usize, usize),
    stroke: usize,
    colour: [u8; CHANNELS],
) {
    let (x0, y0, x1, y1) = rect;
    fill(page, (x0, y0, x1, y0 + stroke), colour);
    fill(page, (x0, y1 - stroke, x1, y1), colour);
    fill(page, (x0, y0, x0 + stroke, y1), colour);
    fill(page, (x1 - stroke, y0, x1, y1), colour);
}

/// A rectangle as the detector would hand it over.
fn quad(rect: (usize, usize, usize, usize)) -> Quad {
    let (x0, y0, x1, y1) = rect;
    let (x0, y0) = (x0 as f32, y0 as f32);
    let (x1, y1) = (x1 as f32, y1 as f32);
    Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

/// Three bars of "type", painted onto the page and declared as text lines.
fn body(page: &mut [u8]) -> Vec<Quad> {
    [(40, 20, 360, 38), (40, 300, 360, 318), (40, 330, 360, 348)]
        .iter()
        .map(|rect| {
            fill(page, *rect, BLACK);
            quad(*rect)
        })
        .collect()
}

/// A figure box is a whole number of mask cells, so it may sit a few pixels
/// outside the ink it was measured from — but never inside it.
fn assert_covers(figure: &Figure, rect: (usize, usize, usize, usize)) {
    let slack = SCALE as u32 - 1;
    let (x0, y0, x1, y1) =
        (rect.0 as u32, rect.1 as u32, rect.2 as u32, rect.3 as u32);
    let (right, bottom) = (figure.x + figure.width, figure.y + figure.height);
    assert!(
        figure.x <= x0 && x0 - figure.x <= slack,
        "left edge {} is not within {slack} px inside {x0}: {figure:?}",
        figure.x
    );
    assert!(
        figure.y <= y0 && y0 - figure.y <= slack,
        "top edge {} is not within {slack} px above {y0}: {figure:?}",
        figure.y
    );
    assert!(
        right >= x1 && right - x1 <= slack,
        "right edge {right} is not within {slack} px past {x1}: {figure:?}"
    );
    assert!(
        bottom >= y1 && bottom - y1 <= slack,
        "bottom edge {bottom} is not within {slack} px past {y1}: {figure:?}"
    );
}

#[test]
fn a_block_of_ink_beside_text_is_a_figure() {
    let mut page = blank();
    let text = body(&mut page);
    fill(&mut page, (100, 150, 220, 240), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &text, &Settings::default());
    assert_eq!(figures.len(), 1, "{figures:?}");
    assert_covers(&figures[0], (100, 150, 220, 240));
}

#[test]
fn a_page_whose_only_ink_is_text_has_no_figures() {
    let mut page = blank();
    let text = body(&mut page);

    let figures = find(&page, WIDTH, HEIGHT, &text, &Settings::default());
    assert!(figures.is_empty(), "{figures:?}");
}

#[test]
fn ink_just_outside_a_text_box_is_painted_out_with_it() {
    let line = (40, 200, 360, 218);
    let mut page = blank();
    fill(&mut page, line, BLACK);
    // An accent five pixels above the box: inside the margin, outside the
    // quadrangle.
    fill(&mut page, (100, 192, 130, 196), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[quad(line)], &permissive());
    assert!(figures.is_empty(), "{figures:?}");
}

#[test]
fn nearly_touching_blobs_come_back_as_one_figure() {
    let mut page = blank();
    fill(&mut page, (100, 100, 160, 160), BLACK);
    fill(&mut page, (168, 100, 228, 160), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert_eq!(figures.len(), 1, "{figures:?}");
    assert_covers(&figures[0], (100, 100, 228, 160));
}

#[test]
fn blobs_a_long_way_apart_stay_apart() {
    let mut page = blank();
    fill(&mut page, (100, 100, 160, 160), BLACK);
    fill(&mut page, (280, 100, 340, 160), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert_eq!(figures.len(), 2, "{figures:?}");
    assert_covers(&figures[0], (100, 100, 160, 160));
    assert_covers(&figures[1], (280, 100, 340, 160));
}

#[test]
fn a_rule_along_the_edge_of_the_page_is_not_a_figure() {
    let mut page = blank();
    fill(&mut page, (0, 4, WIDTH, 12), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &permissive());
    assert!(figures.is_empty(), "{figures:?}");
}

#[test]
fn a_printed_frame_around_the_page_is_not_a_figure() {
    let mut page = blank();
    outline(&mut page, (8, 8, 392, 392), 3, BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &permissive());
    assert!(figures.is_empty(), "{figures:?}");
}

#[test]
fn a_solid_picture_at_the_edge_of_the_page_survives() {
    let mut page = blank();
    fill(&mut page, (0, 0, 120, 120), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert_eq!(figures.len(), 1, "{figures:?}");
    assert_covers(&figures[0], (0, 0, 120, 120));
}

#[test]
fn a_blob_below_the_area_floor_is_dropped() {
    let mut page = blank();
    fill(&mut page, (100, 100, 120, 120), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert!(figures.is_empty(), "{figures:?}");

    let lower = Settings {
        min_area_fraction: 0.001,
        ..Settings::default()
    };
    let figures = find(&page, WIDTH, HEIGHT, &[], &lower);
    assert_eq!(figures.len(), 1, "{figures:?}");
}

#[test]
fn a_streak_outside_the_aspect_band_is_dropped() {
    let mut page = blank();
    fill(&mut page, (50, 200, 350, 208), BLACK);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert!(figures.is_empty(), "{figures:?}");

    let wider = Settings {
        aspect_range: (0.1, 100.0),
        ..Settings::default()
    };
    let figures = find(&page, WIDTH, HEIGHT, &[], &wider);
    assert_eq!(figures.len(), 1, "{figures:?}");
}

#[test]
fn a_saturated_colour_is_ink_even_where_its_luminance_is_not() {
    // Yellow: dark in the blue channel, bright everywhere a luminance would
    // look.
    let mut page = blank();
    fill(&mut page, (100, 150, 220, 240), [0, 255, 255]);

    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert_eq!(figures.len(), 1, "{figures:?}");
    assert_covers(&figures[0], (100, 150, 220, 240));
}

#[test]
fn a_raster_shorter_than_the_page_it_declares_finds_nothing() {
    let page = vec![0u8; WIDTH * HEIGHT * CHANNELS - 1];
    let figures = find(&page, WIDTH, HEIGHT, &[], &Settings::default());
    assert!(figures.is_empty(), "{figures:?}");
}

#[test]
fn a_page_smaller_than_one_mask_cell_is_handled() {
    // Every span and every band on this page rounds to nothing, which is where
    // an off-by-one in the mask arithmetic would show up as a panic.
    let page = vec![0u8; 3 * 2 * CHANNELS];
    let line = Quad::new([(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]);
    assert!(find(&page, 3, 2, &[line], &Settings::default()).is_empty());
    assert!(find(&page, 3, 2, &[], &permissive()).is_empty());

    let whole = Figure {
        x: 0,
        y: 0,
        width: 3,
        height: 2,
    };
    let png = crop_png(&page, 3, 2, &whole).expect("a crop");
    let crop = image::load_from_memory(&png).expect("a PNG").to_rgb8();
    assert_eq!(crop.dimensions(), (3, 2));
}

#[test]
fn a_crop_comes_back_as_a_png_in_rgb() {
    let mut page = blank();
    fill(&mut page, (40, 60, 100, 120), [16, 32, 200]);

    let figure = Figure {
        x: 40,
        y: 60,
        width: 60,
        height: 60,
    };
    let png = crop_png(&page, WIDTH, HEIGHT, &figure).expect("a crop");
    let crop = image::load_from_memory(&png).expect("a PNG").to_rgb8();
    assert_eq!(crop.dimensions(), (60, 60));
    assert_eq!(crop.get_pixel(0, 0).0, [200, 32, 16]);
    assert_eq!(crop.get_pixel(59, 59).0, [200, 32, 16]);

    // Two pixels up and to the left of the block, so the crop's own origin has
    // to be right for the corner to land where it does.
    let corner = Figure {
        x: 38,
        y: 58,
        width: 4,
        height: 4,
    };
    let png = crop_png(&page, WIDTH, HEIGHT, &corner).expect("a crop");
    let crop = image::load_from_memory(&png).expect("a PNG").to_rgb8();
    assert_eq!(crop.get_pixel(0, 0).0, WHITE);
    assert_eq!(crop.get_pixel(3, 3).0, [200, 32, 16]);
}

#[test]
fn a_crop_is_clamped_to_the_page_unless_it_misses_it() {
    let page = blank();
    let over = Figure {
        x: (WIDTH - 10) as u32,
        y: (HEIGHT - 10) as u32,
        width: 40,
        height: 40,
    };
    let png = crop_png(&page, WIDTH, HEIGHT, &over).expect("a clamped crop");
    let crop = image::load_from_memory(&png).expect("a PNG").to_rgb8();
    assert_eq!(crop.dimensions(), (10, 10));

    let missed = Figure {
        x: WIDTH as u32,
        y: 0,
        width: 10,
        height: 10,
    };
    assert!(crop_png(&page, WIDTH, HEIGHT, &missed).is_err());
}
