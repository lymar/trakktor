//! Tests for the parts of the runtime that need no weights: the weight
//! rearrangements done at load, the rotary tables, and the key/value cache.

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};

use super::{
    net::{KvCache, glue_columns, rope_values},
    transpose_2d,
};

type B = NdArray<f32>;

#[test]
fn transposing_moves_rows_into_columns() {
    // [[1, 2, 3], [4, 5, 6]] transposed is [[1, 4], [2, 5], [3, 6]].
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        transpose_2d(&values, 2, 3),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    // Past one tile in both directions, transposing twice is the identity.
    let big: Vec<f32> = (0..70 * 130).map(|i| i as f32).collect();
    let once = transpose_2d(&big, 70, 130);
    assert_eq!(transpose_2d(&once, 130, 70), big);
}

#[test]
fn gluing_lays_the_blocks_out_as_columns() {
    // Two [out, in] matrices over the same two inputs.
    let first = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let second = vec![5.0f32, 6.0]; // [1, 2]
    let glued = glue_columns(&[(first.clone(), 2), (second.clone(), 1)], 2);

    // Row `i` of the result is row `i` of each transposed block, in order.
    assert_eq!(
        glued,
        vec![
            1.0, 3.0, 5.0, // input 0: first's column 0, then second's
            2.0, 4.0, 6.0, // input 1
        ]
    );
    // Which is exactly the two blocks transposed and concatenated column-wise.
    let a = transpose_2d(&first, 2, 2);
    let b = transpose_2d(&second, 1, 2);
    assert_eq!(glued[0..2], a[0..2]);
    assert_eq!(glued[3..5], a[2..4]);
    assert_eq!([glued[2], glued[5]], [b[0], b[1]]);
}

#[test]
fn the_rotary_table_starts_at_the_identity_rotation() {
    let dim = 8;
    let (cos, sin) = rope_values(dim, 3, 10000.0);
    assert_eq!(cos.len(), 3 * dim);

    // Position 0 rotates by nothing: cos = 1, sin = 0.
    assert!(cos[..dim].iter().all(|v| (v - 1.0).abs() < 1e-6));
    assert!(sin[..dim].iter().all(|v| v.abs() < 1e-6));

    // Every row repeats its half across both halves of the head, which is what
    // pairs it with the rotation of the halves.
    for position in 0..3 {
        let row = &cos[position * dim..(position + 1) * dim];
        assert_eq!(row[..dim / 2], row[dim / 2..]);
    }
}

/// Fills `[1, heads, seq, dim]` with a value per position, so a cached prefix
/// can be read back by eye.
fn step(heads: usize, dim: usize, marker: f32) -> Tensor<B, 4> {
    Tensor::from_data(
        TensorData::new(vec![marker; heads * dim], [1, heads, 1, dim]),
        &NdArrayDevice::Cpu,
    )
}

#[test]
fn the_cache_returns_every_position_fed_to_it() {
    let (heads, dim) = (2, 3);
    let mut cache = KvCache::<B>::new(heads, dim);

    for position in 0..4 {
        let marker = (position + 1) as f32;
        let (keys, values) =
            cache.append(step(heads, dim, marker), step(heads, dim, -marker));
        assert_eq!(keys.dims(), [1, heads, position + 1, dim]);

        let keys = keys.into_data().to_vec::<f32>().expect("keys");
        let values = values.into_data().to_vec::<f32>().expect("values");
        // Position `p` of every head holds the marker it was fed.
        for head in 0..heads {
            for (p, want) in (0..=position).map(|p| (p, (p + 1) as f32)) {
                let at = (head * (position + 1) + p) * dim;
                assert_eq!(keys[at], want, "head {head} position {p}");
                assert_eq!(values[at], -want, "head {head} position {p}");
            }
        }
    }
}

#[test]
fn the_cache_survives_growing_past_its_first_allocation() {
    let (heads, dim) = (1, 2);
    let mut cache = KvCache::<B>::new(heads, dim);
    // More positions than the initial capacity, so the buffer is grown.
    let total = super::net::CACHE_MIN_CAPACITY + 5;
    let mut keys = None;
    for position in 0..total {
        let marker = position as f32;
        keys = Some(
            cache
                .append(step(heads, dim, marker), step(heads, dim, marker))
                .0,
        );
    }

    let keys = keys
        .expect("fed")
        .into_data()
        .to_vec::<f32>()
        .expect("keys");
    assert_eq!(keys.len(), total * dim);
    for position in 0..total {
        assert_eq!(
            keys[position * dim],
            position as f32,
            "position {position}"
        );
    }
}

#[test]
fn clearing_the_cache_starts_a_fresh_run() {
    let (heads, dim) = (1, 2);
    let mut cache = KvCache::<B>::new(heads, dim);
    cache.append(step(heads, dim, 1.0), step(heads, dim, 1.0));
    cache.append(step(heads, dim, 2.0), step(heads, dim, 2.0));
    cache.clear();

    let (keys, _) = cache.append(step(heads, dim, 9.0), step(heads, dim, 9.0));
    assert_eq!(keys.dims(), [1, heads, 1, dim]);
    assert_eq!(
        keys.into_data().to_vec::<f32>().expect("keys"),
        vec![9.0, 9.0]
    );
}
