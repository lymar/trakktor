//! Tests for the parts of the decoder that do not need weights. The decode
//! layout they rest on is tested next to itself.

use candle_core::{Device, IndexOp};

use super::*;

#[test]
fn the_attention_window_looks_back_and_never_forward() {
    let frames = 6;
    let window = 3;
    let mask = sliding_window_mask(frames, window, &Device::Cpu)
        .expect("mask")
        .i((0, 0))
        .expect("squeeze")
        .to_vec2::<f32>()
        .expect("rows");

    for (query, row) in mask.iter().enumerate() {
        for (key, &value) in row.iter().enumerate() {
            let visible = value == 0.0;
            assert_eq!(
                visible,
                key <= query && key + window > query,
                "query {query} key {key}"
            );
        }
    }

    // Spot-check the shape of one row: frame 4 sees frames 2, 3 and 4.
    let row: Vec<bool> = mask[4].iter().map(|&v| v == 0.0).collect();
    assert_eq!(row, vec![false, false, true, true, true, false]);
}
