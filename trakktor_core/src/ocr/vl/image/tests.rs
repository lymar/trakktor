use super::*;

/// The factor every published size is a multiple of.
const FACTOR: u32 = 28;
const MIN: u32 = 112_896;
const MAX: u32 = 1_003_520;

#[test]
fn a_size_inside_the_bounds_is_only_rounded() {
    // 700×980 = 686 000 pixels, comfortably between the bounds, and both
    // sides already multiples of 28.
    let (height, width) = smart_resize(700, 980, FACTOR, MIN, MAX).unwrap();
    assert_eq!((height, width), (700, 980));
}

#[test]
fn rounding_goes_to_the_nearest_multiple() {
    let (height, width) = smart_resize(701, 981, FACTOR, MIN, MAX).unwrap();
    assert_eq!(height % FACTOR, 0);
    assert_eq!(width % FACTOR, 0);
    assert_eq!((height, width), (700, 980));
}

#[test]
fn a_large_picture_is_scaled_under_the_ceiling() {
    let (height, width) = smart_resize(2000, 3000, FACTOR, MIN, MAX).unwrap();
    assert!(height * width <= MAX, "{height}×{width}");
    assert_eq!(height % FACTOR, 0);
    assert_eq!(width % FACTOR, 0);
    // The proportions survive the rounding to within one band.
    let before = 3000f32 / 2000f32;
    let after = width as f32 / height as f32;
    assert!((before - after).abs() < 0.05, "{before} vs {after}");
}

#[test]
fn a_small_picture_is_scaled_over_the_floor() {
    let (height, width) = smart_resize(80, 400, FACTOR, MIN, MAX).unwrap();
    assert!(height * width >= MIN, "{height}×{width}");
    assert_eq!(height % FACTOR, 0);
    assert_eq!(width % FACTOR, 0);
}

#[test]
fn a_sliver_narrower_than_one_patch_is_grown_first() {
    // Ten pixels tall is less than one patch row; it must come back at least
    // one factor tall rather than as a zero-row grid.
    let (height, _) = smart_resize(10, 900, FACTOR, MIN, MAX).unwrap();
    assert!(height >= FACTOR, "{height}");
}

#[test]
fn an_impossible_aspect_ratio_is_refused() {
    assert!(smart_resize(4, 4000, FACTOR, MIN, MAX).is_err());
}

#[test]
fn the_same_refusal_is_available_as_a_question() {
    // The engine asks before it hands a block over, because there the refusal
    // would cost a page of blocks rather than the one that is a hairline.
    let cfg = ImageConfig {
        min_pixels: MIN,
        max_pixels: MAX,
        patch_size: 14,
        merge_size: 2,
        rescale_factor: 1.0 / 255.0,
        image_mean: [0.5, 0.5, 0.5],
        image_std: [0.5, 0.5, 0.5],
    };
    assert!(!fits(&RgbImage::new(4000, 4), &cfg));
    assert!(fits(&RgbImage::new(980, 700), &cfg));
}

#[test]
fn patches_carry_the_pixels_channel_by_channel() {
    let cfg = ImageConfig {
        min_pixels: MIN,
        max_pixels: MAX,
        patch_size: 14,
        merge_size: 2,
        rescale_factor: 1.0 / 255.0,
        image_mean: [0.5, 0.5, 0.5],
        image_std: [0.5, 0.5, 0.5],
    };
    // One patch grid of exactly 28×28 keeps its size (it is a multiple of the
    // factor) but is below the floor, so it will be scaled up; ask for a size
    // that is already inside the bounds instead.
    let mut picture = RgbImage::new(980, 700);
    for (x, y, pixel) in picture.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, 255]);
    }
    let prepared = prepare(&picture, &cfg).unwrap();
    assert_eq!(prepared.grid, (1, 50, 70));
    assert_eq!(prepared.patches(), 3500);
    assert_eq!(prepared.tokens(2), 875);
    assert_eq!(prepared.pixels.len(), 3500 * 3 * 14 * 14);

    // The first patch's third channel is the constant 255, normalized to 1.
    let blue = &prepared.pixels[2 * 14 * 14..3 * 14 * 14];
    assert!(blue.iter().all(|v| (*v - 1.0).abs() < 1e-6));
}
