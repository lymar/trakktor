use super::*;

/// Page size the fixtures below are cut out of.
const PAGE_WIDTH: usize = 20;
const PAGE_HEIGHT: usize = 24;

/// A synthetic page with structure in all three channels — a gradient, a
/// quadratic and a diagonal, each wrapping at a different modulus, so that
/// every crop pixel depends on where it was read from and a warp that is off
/// by a fraction of a pixel cannot come out looking right.
fn page() -> Vec<u8> {
    let mut bgr = vec![0u8; PAGE_WIDTH * PAGE_HEIGHT * CHANNELS];
    for y in 0..PAGE_HEIGHT {
        for x in 0..PAGE_WIDTH {
            let at = (y * PAGE_WIDTH + x) * CHANNELS;
            bgr[at] = ((x * 11 + y * 5) % 251) as u8;
            bgr[at + 1] = ((x * x + 3 * y * y) % 253) as u8;
            bgr[at + 2] = (((x + y) * 17) % 249) as u8;
        }
    }
    bgr
}

/// A flat page of one color, for the tests that only care about geometry.
fn flat(width: usize, height: usize) -> Vec<u8> {
    vec![64u8; width * height * CHANNELS]
}

fn quad(points: [(f32, f32); 4]) -> Quad { Quad::new(points) }

/// The pixel at `(x, y)` of the synthetic page.
fn pixel(bgr: &[u8], x: usize, y: usize) -> [u8; CHANNELS] {
    let at = (y * PAGE_WIDTH + x) * CHANNELS;
    [bgr[at], bgr[at + 1], bgr[at + 2]]
}

/// The pixel at `(x, y)` of a crop.
fn crop_pixel(crop: &Crop, x: usize, y: usize) -> [u8; CHANNELS] {
    let at = (y * crop.width + x) * CHANNELS;
    [crop.bgr[at], crop.bgr[at + 1], crop.bgr[at + 2]]
}

/// Compares a crop against a reference warp, which a float kernel matches to
/// within one level rather than exactly. Reports the worst and the mean
/// difference so a real regression is not mistaken for that one level.
fn assert_matches(crop: &Crop, width: usize, height: usize, golden: &[u8]) {
    assert_eq!((crop.width, crop.height), (width, height));
    assert_eq!(crop.bgr.len(), golden.len());
    let mut worst = 0i32;
    let mut total = 0i32;
    for (got, want) in crop.bgr.iter().zip(golden) {
        let delta = i32::from(*got) - i32::from(*want);
        worst = worst.max(delta.abs());
        total += delta.abs();
    }
    let mean = f64::from(total) / golden.len() as f64;
    assert!(worst <= 1, "worst difference {worst} levels, mean {mean}");
    assert!(mean <= 0.2, "mean difference {mean} levels");
}

#[test]
fn an_axis_aligned_quadrangle_comes_back_untouched() {
    let bgr = page();
    let crop = rotate_crop(
        &bgr,
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(3.0, 2.0), (15.0, 2.0), (15.0, 9.0), (3.0, 9.0)]),
    );

    assert_eq!((crop.width, crop.height), (12, 7));
    for y in 0..7 {
        for x in 0..12 {
            assert_eq!(
                crop_pixel(&crop, x, y),
                pixel(&bgr, 3 + x, 2 + y),
                "at ({x}, {y})"
            );
        }
    }
}

#[test]
fn a_rotated_quadrangle_matches_the_reference_warp() {
    let crop = rotate_crop(
        &page(),
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(2.25, 3.5), (15.75, 6.25), (14.5, 11.0), (1.0, 8.25)]),
    );
    assert_matches(&crop, 13, 4, &GOLDEN_SLANTED);
}

#[test]
fn a_tall_quadrangle_is_turned_and_matches_the_reference_warp() {
    let crop = rotate_crop(
        &page(),
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(12.0, 2.5), (16.25, 3.25), (13.5, 15.0), (9.25, 14.25)]),
    );
    // 4 x 12 before the turn, 12 x 4 after it.
    assert_matches(&crop, 12, 4, &GOLDEN_TURNED);
}

#[test]
fn the_turn_carries_the_right_hand_column_to_the_top_row() {
    let bgr = page();
    // An upright rectangle, tall enough to turn: the crop is a plain
    // sub-rectangle of the page, so the turn is the only thing under test.
    let crop = rotate_crop(
        &bgr,
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(4.0, 3.0), (10.0, 3.0), (10.0, 21.0), (4.0, 21.0)]),
    );

    assert_eq!((crop.width, crop.height), (18, 6));
    for y in 0..6 {
        for x in 0..18 {
            // Row y of the turned crop is column 5 - y of the upright one.
            assert_eq!(
                crop_pixel(&crop, x, y),
                pixel(&bgr, 4 + (5 - y), 3 + x),
                "at ({x}, {y})"
            );
        }
    }
}

#[test]
fn the_turn_threshold_is_inclusive() {
    let (width, height) = (64usize, 64usize);
    let bgr = flat(width, height);

    let exactly = rotate_crop(
        &bgr,
        width,
        height,
        &quad([(2.0, 2.0), (42.0, 2.0), (42.0, 62.0), (2.0, 62.0)]),
    );
    // 40 x 60 is exactly 1.5 times taller than wide, and does turn.
    assert_eq!((exactly.width, exactly.height), (60, 40));

    let just_under = rotate_crop(
        &bgr,
        width,
        height,
        &quad([(2.0, 2.0), (42.0, 2.0), (42.0, 61.0), (2.0, 61.0)]),
    );
    assert_eq!((just_under.width, just_under.height), (40, 59));
}

#[test]
fn the_crop_size_truncates() {
    let crop = rotate_crop(
        &page(),
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(1.0, 1.0), (13.7, 1.0), (13.7, 11.9), (1.0, 11.9)]),
    );
    // 12.7 x 10.9, with the fraction dropped rather than rounded.
    assert_eq!((crop.width, crop.height), (12, 10));
}

#[test]
fn pixels_outside_the_page_repeat_its_border() {
    let bgr = page();
    // Entirely off the left edge: every sample clamps to column 0.
    let crop = rotate_crop(
        &bgr,
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([(-9.0, 2.0), (-3.0, 2.0), (-3.0, 6.0), (-9.0, 6.0)]),
    );

    assert_eq!((crop.width, crop.height), (6, 4));
    for y in 0..4 {
        for x in 0..6 {
            assert_eq!(
                crop_pixel(&crop, x, y),
                pixel(&bgr, 0, 2 + y),
                "at ({x}, {y})"
            );
        }
    }
}

#[test]
fn a_quadrangle_one_past_the_last_column_still_crops() {
    // The detector clamps corners to the page size inclusive, so a corner may
    // legitimately sit one past the last column or row.
    let bgr = page();
    let crop = rotate_crop(
        &bgr,
        PAGE_WIDTH,
        PAGE_HEIGHT,
        &quad([
            (16.0, 20.0),
            (PAGE_WIDTH as f32, 20.0),
            (PAGE_WIDTH as f32, PAGE_HEIGHT as f32),
            (16.0, PAGE_HEIGHT as f32),
        ]),
    );

    assert_eq!((crop.width, crop.height), (4, 4));
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                crop_pixel(&crop, x, y),
                pixel(&bgr, 16 + x, 20 + y),
                "at ({x}, {y})"
            );
        }
    }
}

#[test]
fn degenerate_input_yields_an_empty_crop() {
    let bgr = page();
    let upright = quad([(3.0, 2.0), (15.0, 2.0), (15.0, 9.0), (3.0, 9.0)]);

    let flat_quad = quad([(3.0, 2.0), (15.0, 2.0), (15.0, 2.4), (3.0, 2.4)]);
    let collinear = quad([(1.0, 1.0), (5.0, 1.0), (9.0, 1.0), (13.0, 1.0)]);
    let far = quad([(0.0, 0.0), (1e9, 0.0), (1e9, 10.0), (0.0, 10.0)]);

    for (name, degenerate) in [
        // Sides that truncate to nothing.
        ("flat", flat_quad),
        // Four corners on one line: no transform carries them onto a
        // rectangle.
        ("collinear", collinear),
        // Wider than the page could possibly hold.
        ("far", far),
    ] {
        let crop = rotate_crop(&bgr, PAGE_WIDTH, PAGE_HEIGHT, &degenerate);
        assert!(crop.is_empty(), "{name} quadrangle produced a crop");
        assert_eq!((crop.width, crop.height), (0, 0), "{name}");
    }

    // A page that does not hold the pixels it claims to.
    let short = rotate_crop(&bgr[..90], PAGE_WIDTH, PAGE_HEIGHT, &upright);
    assert!(short.is_empty());
    // A page with no pixels at all.
    assert!(rotate_crop(&[], 0, 0, &upright).is_empty());
}

#[test]
fn the_bicubic_weights_sum_to_one_and_reduce_to_a_copy() {
    for step in 0..=16 {
        let t = f64::from(step) / 16.0;
        let weights = cubic_weights(t);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights at {t} sum to {sum}");
    }
    // On a tap, the kernel picks that tap alone.
    assert_eq!(cubic_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
}

/// The reference warp of the slanted quadrangle over the synthetic page:
/// OpenCV's bicubic with a replicated border, 13 x 4 pixels, blue first. A
/// float kernel differs from OpenCV's fixed-point weights by at most one
/// level, which is what [`assert_matches`] allows.
const GOLDEN_SLANTED: [u8; 156] = [
    43, 42, 98, 55, 51, 119, 68, 64, 140, 80, 81, 163, 93, 98, 184, 105, 117,
    202, 117, 139, 255, 130, 164, 121, 142, 196, 0, 154, 254, 42, 167, 171, 60,
    179, 28, 81, 191, 72, 103, 44, 68, 112, 57, 80, 133, 70, 96, 156, 83, 112,
    178, 95, 129, 198, 107, 149, 224, 120, 173, 213, 133, 207, 0, 145, 221, 38,
    158, 91, 55, 170, 65, 76, 182, 74, 98, 195, 114, 120, 47, 104, 126, 60,
    119, 149, 72, 134, 170, 84, 149, 190, 96, 166, 208, 109, 188, 255, 122,
    189, 13, 134, 155, 23, 147, 129, 48, 159, 40, 69, 172, 83, 91, 185, 122,
    113, 197, 161, 134, 50, 152, 143, 63, 166, 165, 75, 188, 185, 87, 226, 204,
    100, 230, 255, 112, 240, 98, 125, 76, 0, 137, 6, 43, 149, 57, 62, 162, 100,
    84, 174, 141, 106, 187, 192, 127, 199, 255, 147,
];

/// The reference warp of the tall quadrangle, after its quarter turn: 12 x 4
/// pixels.
const GOLDEN_TURNED: [u8; 144] = [
    183, 208, 62, 184, 20, 74, 186, 27, 86, 190, 65, 100, 192, 98, 113, 194,
    134, 125, 196, 174, 136, 199, 255, 150, 202, 50, 163, 204, 85, 176, 206,
    150, 187, 208, 255, 200, 170, 229, 40, 172, 255, 51, 174, 118, 64, 177, 24,
    78, 180, 48, 91, 181, 94, 103, 183, 135, 115, 186, 214, 128, 189, 32, 142,
    191, 31, 155, 193, 109, 166, 195, 212, 179, 157, 192, 3, 159, 200, 27, 162,
    255, 44, 164, 214, 57, 167, 108, 70, 168, 39, 82, 171, 102, 94, 174, 162,
    108, 176, 198, 121, 178, 138, 134, 180, 49, 146, 183, 172, 159, 144, 162,
    122, 146, 174, 10, 149, 190, 0, 152, 246, 40, 154, 139, 49, 156, 24, 61,
    159, 61, 74, 162, 118, 88, 164, 189, 101, 166, 157, 112, 168, 6, 125, 171,
    123, 139,
];
