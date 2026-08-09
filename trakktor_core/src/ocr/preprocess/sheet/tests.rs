use super::*;

/// A frame with a bright rectangle on a dark ground, at a known place.
fn frame(
    width: usize,
    height: usize,
    sheet: (usize, usize, usize, usize),
) -> Page {
    let (x0, y0, x1, y1) = sheet;
    let mut bgr = vec![28u8; width * height * CHANNELS];
    for y in y0..y1 {
        for x in x0..x1 {
            let at = (y * width + x) * CHANNELS;
            bgr[at..at + CHANNELS].copy_from_slice(&[236, 238, 240]);
        }
    }
    Page {
        width: width as u32,
        height: height as u32,
        bgr,
    }
}

#[test]
fn a_sheet_on_a_desk_is_found_where_it_is() {
    let page = frame(800, 1000, (200, 250, 600, 800));
    let sheet = find(&page).expect("a sheet");
    let (x0, y0, x1, y1) = sheet.quad.bounds();
    // Within the margin the cut deliberately adds, and a pixel of the scale
    // the search runs at.
    assert!((x0 - 200.0).abs() < 12.0, "left {x0}");
    assert!((y0 - 250.0).abs() < 12.0, "top {y0}");
    assert!((x1 - 600.0).abs() < 12.0, "right {x1}");
    assert!((y1 - 800.0).abs() < 12.0, "bottom {y1}");
}

#[test]
fn the_cut_page_is_the_sheet_and_nothing_else() {
    let page = frame(800, 1000, (200, 250, 600, 800));
    let sheet = find(&page).expect("a sheet");
    let cut = sheet.apply(&page);
    assert!(cut.width > 380 && cut.width < 430, "{}", cut.width);
    assert!(cut.height > 530 && cut.height < 580, "{}", cut.height);
    // The middle of the cut is paper, not desk.
    let middle = ((cut.height as usize / 2) * cut.width as usize +
        cut.width as usize / 2) *
        CHANNELS;
    assert!(cut.bgr[middle] > 200);
}

#[test]
fn a_point_of_the_cut_maps_back_into_the_sheet() {
    let page = frame(800, 1000, (200, 250, 600, 800));
    let sheet = find(&page).expect("a sheet");
    let (x, y) = sheet.source(0.0, 0.0);
    assert!((x - 200.0).abs() < 14.0, "{x}");
    assert!((y - 250.0).abs() < 14.0, "{y}");
    let (x, y) = sheet.source(sheet.width as f32, sheet.height as f32);
    assert!((x - 600.0).abs() < 14.0, "{x}");
    assert!((y - 800.0).abs() < 14.0, "{y}");
}

/// A frame the sheet runs out of: the quadrangle is given as four corners,
/// one of which sits outside the picture, and only the part inside is drawn.
fn cropped_frame(width: usize, height: usize, quad: [(f32, f32); 4]) -> Page {
    let mut bgr = vec![28u8; width * height * CHANNELS];
    // Inside a convex polygon every edge turns the same way, whichever way
    // round the corners were given.
    let inside = |x: f32, y: f32| {
        let side = |at: usize| {
            let a = quad[at];
            let b = quad[(at + 1) % 4];
            (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)
        };
        (0..4).all(|at| side(at) >= 0.0) || (0..4).all(|at| side(at) <= 0.0)
    };
    for y in 0..height {
        for x in 0..width {
            if inside(x as f32 + 0.5, y as f32 + 0.5) {
                let at = (y * width + x) * CHANNELS;
                bgr[at..at + CHANNELS].copy_from_slice(&[236, 238, 240]);
            }
        }
    }
    Page {
        width: width as u32,
        height: height as u32,
        bgr,
    }
}

#[test]
fn a_corner_outside_the_picture_is_reconstructed_from_the_sides() {
    // The bottom-left corner is two hundred pixels off the left edge, so the
    // brightest region's own extreme point is nowhere near it. Cutting to
    // that extreme point shears the sheet; fitting its sides does not.
    let quad = [
        (240.0, 120.0),
        (760.0, 60.0),
        (900.0, 900.0),
        (-200.0, 940.0),
    ];
    let page = cropped_frame(800, 1000, quad);
    let sheet = find(&page).expect("a sheet");

    let bottom_left = sheet.quad.points[3];
    assert!(
        bottom_left.0 < -100.0,
        "the corner was pinned inside the frame at {bottom_left:?}"
    );
    // Every corner within a few pixels of where the sheet really has one.
    for (found, wanted) in sheet.quad.points.iter().zip(quad.iter()) {
        assert!(
            (found.0 - wanted.0).abs() < 25.0 &&
                (found.1 - wanted.1).abs() < 25.0,
            "{found:?} is not {wanted:?}"
        );
    }
}

#[test]
fn a_page_that_fills_its_frame_has_no_sheet_to_cut_out() {
    // A scan: paper edge to edge, nothing around it.
    let page = frame(400, 500, (0, 0, 400, 500));
    assert!(find(&page).is_none());
}

#[test]
fn a_sliver_of_brightness_is_not_a_sheet() {
    let page = frame(800, 1000, (10, 10, 60, 60));
    assert!(find(&page).is_none());
}
