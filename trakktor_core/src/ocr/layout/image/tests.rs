//! Preparing a page for the layout network.

use super::*;

/// A page of one colour, so every sample of the resize is that colour and any
/// slip in the channel order or the scaling shows up as a number.
fn page(width: u32, height: u32, bgr: [u8; 3]) -> Page {
    Page {
        bgr: (0..width as usize * height as usize)
            .flat_map(|_| bgr)
            .collect(),
        width,
        height,
    }
}

#[test]
fn the_page_is_squashed_into_the_square() {
    let prepared = prepare(&page(1000, 400, [0, 0, 0]), 800);
    assert_eq!(prepared.data.len(), 3 * 800 * 800);
    assert_eq!(prepared.side, 800);
    assert_eq!(prepared.page, (1000, 400));
}

#[test]
fn the_scale_factor_is_height_first() {
    let prepared = prepare(&page(1000, 400, [0, 0, 0]), 800);
    let (height, width) = prepared.scale_factor();
    assert_eq!(height, 2.0);
    assert_eq!(width, 0.8);
}

#[test]
fn the_channels_come_out_rgb_and_scaled() {
    // A page that is pure blue in the BGR the domain reads.
    let prepared = prepare(&page(64, 64, [255, 0, 0]), 32);
    let plane = 32 * 32;
    assert_eq!(prepared.data[0], 0.0, "red plane");
    assert_eq!(prepared.data[plane], 0.0, "green plane");
    assert_eq!(prepared.data[2 * plane], 1.0, "blue plane, scaled by 255");
}

#[test]
fn a_flat_page_survives_the_resize_flat() {
    // Bicubic overshoots at an edge; over a flat field it must not move at
    // all, which is the cheapest check that the weights sum to one.
    let prepared = prepare(&page(500, 300, [40, 90, 200]), 128);
    let plane = 128 * 128;
    for (at, expected) in [(0usize, 200.0), (plane, 90.0), (2 * plane, 40.0)] {
        for value in &prepared.data[at..at + plane] {
            assert!(
                (value - expected / 255.0).abs() < 1e-6,
                "got {value} instead of {}",
                expected / 255.0
            );
        }
    }
}

#[test]
fn a_resize_to_the_same_size_is_the_identity() {
    let source: Vec<u8> = (0..3 * 16 * 16).map(|at| (at % 251) as u8).collect();
    let out = resize_cubic(&source, 16, 16, 16, 16);
    assert_eq!(out, source);
}

#[test]
fn a_gradient_keeps_its_direction() {
    // A horizontal ramp, downscaled: the result must still rise left to right.
    let width = 200usize;
    let source: Vec<u8> = (0..width * 4)
        .flat_map(|at| {
            let value = ((at % width) * 255 / (width - 1)) as u8;
            [value, value, value]
        })
        .collect();
    let out = resize_cubic(&source, width, 4, 50, 4);
    let row: Vec<u8> = out[..50 * 3].iter().step_by(3).copied().collect();
    assert!(row.windows(2).all(|pair| pair[0] <= pair[1]), "{row:?}");
    assert!(
        row[0] < 10 && row[49] > 245,
        "{:?} .. {:?}",
        row[0],
        row[49]
    );
}
