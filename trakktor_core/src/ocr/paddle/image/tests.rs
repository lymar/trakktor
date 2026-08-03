use image::{DynamicImage, ImageBuffer};

use super::*;

/// A deterministic test page: each channel of each pixel is its own slope, so
/// no two neighbours agree and a wrong tap or a swapped channel shows up.
fn ramp(width: u32, height: u32) -> Page {
    let mut bgr =
        Vec::with_capacity(width as usize * height as usize * CHANNELS);
    for y in 0..height as usize {
        for x in 0..width as usize {
            for c in 0..CHANNELS {
                bgr.push(((37 * x + 91 * y + 53 * c) % 256) as u8);
            }
        }
    }
    Page { width, height, bgr }
}

/// The whole sizing chain: pad first, then apply the side-length rule, and
/// report the ratios against the size the resize actually saw.
fn target(
    height: usize,
    width: usize,
    limit: usize,
    kind: LimitType,
) -> (usize, usize, f64, f64) {
    let (width, height) = padded_size(width, height);
    let (target_height, target_width) =
        resize_target(height, width, limit, kind);
    (
        target_height,
        target_width,
        target_height as f64 / height as f64,
        target_width as f64 / width as f64,
    )
}

#[test]
fn rounding_a_side_breaks_ties_toward_the_even_band() {
    // Every side that is exactly halfway between two bands, plus the clamp.
    let expected = [
        (0, 32),
        (16, 32),
        (31, 32),
        (32, 32),
        (48, 64),
        (80, 64),
        (112, 128),
        (144, 128),
        (176, 192),
        (208, 192),
        (240, 256),
        (1000, 992),
        (1080, 1088),
    ];
    for (side, want) in expected {
        assert_eq!(round_to_multiple(side), want, "side {side}");
    }
}

#[test]
fn the_longest_side_rule_scales_pages_down_only() {
    // (source height, source width, height, width, ratio height, ratio width)
    let expected = [
        (1080, 1920, 544, 960, 0.503_703_703_703_703_7, 0.5),
        (720, 1280, 544, 960, 0.755_555_555_555_555_5, 0.75),
        (2160, 3840, 544, 960, 0.251_851_851_851_851_8, 0.25),
        (3000, 2000, 960, 640, 0.32, 0.32),
        (5000, 6000, 800, 960, 0.16, 0.16),
        (960, 960, 960, 960, 1.0, 1.0),
        (100, 100, 96, 96, 0.96, 0.96),
        (
            49,
            49,
            64,
            64,
            1.306_122_448_979_591_7,
            1.306_122_448_979_591_7,
        ),
        (
            47,
            47,
            32,
            32,
            0.680_851_063_829_787_2,
            0.680_851_063_829_787_2,
        ),
        (
            33,
            33,
            32,
            32,
            0.969_696_969_696_969_7,
            0.969_696_969_696_969_7,
        ),
        // The floor on the short side blows its ratio up; harmless, because
        // the detector's boxes are rescaled from the map, not by the ratios.
        (1000, 1, 960, 32, 0.96, 32.0),
    ];
    for (height, width, want_h, want_w, ratio_h, ratio_w) in expected {
        let got = target(height, width, 960, LimitType::Max);
        assert_eq!((got.0, got.1), (want_h, want_w), "{height}x{width}");
        assert!(
            (got.2 - ratio_h).abs() < 1e-12 && (got.3 - ratio_w).abs() < 1e-12,
            "{height}x{width}: ratios {:?} want ({ratio_h}, {ratio_w})",
            (got.2, got.3),
        );
    }
}

#[test]
fn the_shortest_side_rule_scales_pages_up_only() {
    let expected = [
        (1080, 1920, 1088, 1920, 1.007_407_407_407_407_3, 1.0),
        (100, 100, 96, 96, 0.96, 0.96),
        (
            49,
            49,
            64,
            64,
            1.306_122_448_979_591_7,
            1.306_122_448_979_591_7,
        ),
        (
            33,
            33,
            64,
            64,
            1.939_393_939_393_939_4,
            1.939_393_939_393_939_4,
        ),
        // Both of these are padded to at least 32 a side first, so their
        // ratios are against the padded size.
        (20, 20, 64, 64, 2.0, 2.0),
        (10, 50, 64, 96, 2.0, 1.92),
        (64, 64, 64, 64, 1.0, 1.0),
        // 300 scales to 304, which is a tie: it rounds down to 320's
        // neighbour only if the tie goes the other way.
        (
            63,
            300,
            64,
            320,
            1.015_873_015_873_015_8,
            1.066_666_666_666_666_7,
        ),
    ];
    for (height, width, want_h, want_w, ratio_h, ratio_w) in expected {
        let got = target(height, width, 64, LimitType::Min);
        assert_eq!((got.0, got.1), (want_h, want_w), "{height}x{width}");
        assert!(
            (got.2 - ratio_h).abs() < 1e-12 && (got.3 - ratio_w).abs() < 1e-12,
            "{height}x{width}: ratios {:?} want ({ratio_h}, {ratio_w})",
            (got.2, got.3),
        );
    }
}

#[test]
fn the_long_side_rule_scales_in_both_directions() {
    assert_eq!(target(100, 200, 64, LimitType::Long).0, 32);
    assert_eq!(target(100, 200, 64, LimitType::Long).1, 64);
    assert_eq!(target(100, 200, 400, LimitType::Long).0, 192);
    assert_eq!(target(100, 200, 400, LimitType::Long).1, 384);
}

#[test]
fn the_side_cap_bounds_a_page_the_rule_would_leave_huge() {
    // The short side wants 40 000 px; the cap pulls the pair back under
    // 4000 before the bands are rounded.
    let (height, width, _, _) = target(10_000, 100, 400, LimitType::Min);
    assert_eq!((height, width), (4000, 32));
}

/// Output pixels of the reference resize, taken from OpenCV's fixed-point
/// bilinear on the same `ramp` pages. The cases are 1-D in each direction, a
/// general 2-D downscale, and an exact-2x upscale.
const RESIZE: &[(u32, u32, usize, usize, &[u8])] = &[
    (
        8,
        1,
        4,
        1,
        &[19, 72, 125, 93, 146, 199, 167, 220, 145, 113, 38, 91],
    ),
    (
        1,
        8,
        1,
        4,
        &[46, 99, 152, 100, 153, 78, 154, 207, 132, 80, 133, 186],
    ),
    (
        7,
        5,
        4,
        3,
        &[
            44, 97, 150, 109, 162, 130, 174, 152, 194, 153, 99, 88, //
            196, 153, 46, 5, 58, 111, 69, 122, 175, 134, 187, 240, //
            91, 144, 197, 156, 188, 91, 71, 103, 82, 115, 115, 136,
        ],
    ),
    (
        5,
        3,
        3,
        2,
        &[
            35, 88, 141, 97, 150, 139, 158, 169, 200, //
            172, 160, 86, 41, 94, 83, 103, 113, 145,
        ],
    ),
    (
        3,
        2,
        6,
        4,
        &[
            0, 53, 106, 9, 62, 115, 28, 81, 134, 46, 99, 152, 65, 118, 171, 74,
            127, 180, //
            23, 76, 129, 32, 85, 138, 50, 103, 156, 69, 122, 159, 87, 140, 145,
            97, 150, 139, //
            68, 121, 174, 77, 130, 183, 96, 149, 202, 114, 167, 172, 133, 186,
            95, 142, 195, 56, //
            91, 144, 197, 100, 153, 206, 119, 172, 225, 137, 190, 179, 156,
            209, 70, 165, 218, 15,
        ],
    ),
];

#[test]
fn resizing_reproduces_the_fixed_point_bilinear() {
    for &(source_width, source_height, width, height, expected) in RESIZE {
        let page = ramp(source_width, source_height);
        let got = resize_bgr(&page, width, height);
        assert_eq!(
            got, expected,
            "{source_width}x{source_height} -> {width}x{height}"
        );
    }
}

#[test]
fn resizing_to_the_same_size_returns_the_page_unchanged() {
    let page = ramp(9, 7);
    assert_eq!(resize_bgr(&page, 9, 7), page.bgr);
}

#[test]
fn a_tiny_page_is_padded_black_and_normalized_into_bgr_planes() {
    // 5 + 4 is well under the padding threshold, so the page lands in the top
    // left of a black 32x32 one and is not scaled at all.
    let page = ramp(5, 4);
    let input = detector_input(&page, 960, LimitType::Max, &DETECTOR_NORMALIZE);
    assert_eq!((input.width, input.height), (32, 32));
    assert_eq!((input.ratio_width, input.ratio_height), (1.0, 1.0));
    assert_eq!(input.data.len(), CHANNELS * 32 * 32);

    let plane = 32 * 32;
    let normalized = |value: u8, channel: usize| {
        (f32::from(value) * DETECTOR_NORMALIZE.scale -
            DETECTOR_NORMALIZE.mean[channel]) /
            DETECTOR_NORMALIZE.std[channel]
    };
    for channel in 0..CHANNELS {
        // The page's own top-left pixel, in the plane of its own channel.
        assert_eq!(
            input.data[channel * plane],
            normalized(page.bgr[channel], channel)
        );
        // A pixel of the margin: black, which is a long way from zero once
        // the mean is subtracted.
        assert_eq!(
            input.data[channel * plane + 20 * 32 + 20],
            normalized(0, channel)
        );
    }
    // The planes are distinguishable, i.e. the layout really is per channel.
    assert_ne!(input.data[0], input.data[plane]);
}

#[test]
fn a_page_over_the_short_side_limit_is_only_pulled_onto_the_band() {
    let page = ramp(200, 120);
    let input = detector_input(&page, 64, LimitType::Min, &DETECTOR_NORMALIZE);
    assert_eq!((input.width, input.height), (192, 128));
    assert_eq!(input.data.len(), CHANNELS * 192 * 128);
    assert_eq!(input.ratio_width, 192.0 / 200.0);
    assert_eq!(input.ratio_height, 128.0 / 120.0);
}

#[test]
fn decoding_matches_what_the_reference_loader_hands_the_pipeline() {
    // Alpha is dropped, not composited: a fully transparent pixel keeps its
    // colour.
    let rgba: Vec<u8> = vec![10, 120, 250, 0, 10, 120, 250, 255];
    let rgba = ImageBuffer::from_raw(2, 1, rgba).expect("a 2x1 RGBA page");
    let page = Page::from_dynamic(&DynamicImage::ImageRgba8(rgba));
    assert_eq!(page.bgr, [250, 120, 10, 250, 120, 10]);

    // Grayscale becomes three identical channels.
    let luma = ImageBuffer::from_raw(3, 1, vec![0u8, 17, 200])
        .expect("a 3x1 gray page");
    let page = Page::from_dynamic(&DynamicImage::ImageLuma8(luma));
    assert_eq!(page.bgr, [0, 0, 0, 17, 17, 17, 200, 200, 200]);

    // Gray plus alpha: same, the alpha is gone.
    let luma_alpha = ImageBuffer::from_raw(2, 1, vec![5u16, 0, 60_000, 65_535])
        .expect("a 2x1 gray page with alpha");
    let page = Page::from_dynamic(&DynamicImage::ImageLumaA16(luma_alpha));
    assert_eq!(page.bgr, [0, 0, 0, 234, 234, 234]);

    // 16-bit narrows by keeping the high byte, so 0xFF00 stays saturated
    // rather than rounding down a level.
    let deep = ImageBuffer::from_raw(1, 1, vec![0xFF00u16, 0x0100, 0x00FF])
        .expect("a 1x1 16-bit page");
    let page = Page::from_dynamic(&DynamicImage::ImageRgb16(deep));
    assert_eq!(page.bgr, [0, 1, 255]);

    // And an RGB page is simply swapped.
    let rgb = ImageBuffer::from_raw(1, 1, vec![1u8, 2, 3]).expect("a 1x1 page");
    assert_eq!(Page::from_rgb8(&rgb).bgr, [3, 2, 1]);
}
