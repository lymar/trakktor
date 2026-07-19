//! burn-runtime tests: hermetic checks of the load-time weight
//! transformations the burn net applies (q/k/v glue, the position+type
//! table). Cross-runtime parity against the candle path runs at the CLI
//! level on real checkpoints.

use burn::tensor::{Tensor, TensorData, module::linear};

use super::net::{glue_qkv, position_type_table};

type B = burn::backend::ndarray::NdArray<f32>;

const DEV: burn::backend::ndarray::NdArrayDevice =
    burn::backend::ndarray::NdArrayDevice::Cpu;

/// A tiny deterministic value sequence for hermetic tests.
fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 23) as f32 / 23.0 * scale + offset)
        .collect()
}

/// `y = x·Wᵀ + b` computed directly on the host for one `[out, in]` weight.
fn project(x: &[f32], w: &[f32], b: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * d];
    for row in 0..n {
        for o in 0..d {
            let mut acc = b[o];
            for i in 0..d {
                acc += x[row * d + i] * w[o * d + i];
            }
            y[row * d + o] = acc;
        }
    }
    y
}

#[test]
fn glued_qkv_matches_separate_projections() {
    let (n, d) = (5usize, 8usize);
    let x = ramp(n * d, 2.0, -1.0);
    let q = ramp(d * d, 1.0, -0.5);
    let k = ramp(d * d, 0.7, 0.2);
    let v = ramp(d * d, 1.3, -0.1);
    let qb = ramp(d, 0.5, 0.1);
    let kb = ramp(d, 0.4, -0.2);
    let vb = ramp(d, 0.3, 0.3);

    let glued = glue_qkv(&q, &k, &v, d);
    let mut bias = qb.clone();
    bias.extend_from_slice(&kb);
    bias.extend_from_slice(&vb);

    let out = linear(
        Tensor::<B, 3>::from_data(TensorData::new(x.clone(), [1, n, d]), &DEV),
        Tensor::from_data(TensorData::new(glued, [d, 3 * d]), &DEV),
        Some(Tensor::from_data(TensorData::new(bias, [3 * d]), &DEV)),
    );

    let got = out.into_data().to_vec::<f32>().unwrap();
    for (block, (w, b)) in [(&q, &qb), (&k, &kb), (&v, &vb)].iter().enumerate()
    {
        let want = project(&x, w, b, n, d);
        for row in 0..n {
            for o in 0..d {
                let value = got[row * 3 * d + block * d + o];
                assert!(
                    (value - want[row * d + o]).abs() < 1e-5,
                    "block {block}, row {row}, col {o}: {value} vs {}",
                    want[row * d + o]
                );
            }
        }
    }
}

#[test]
fn position_type_table_sums_shifted_rows() {
    let (max_pos, d, offset) = (7usize, 3usize, 2usize);
    let pos = ramp(max_pos * d, 1.0, 0.0);
    let type_row = ramp(d, 0.5, -0.2);

    let table = position_type_table(&pos, &type_row, offset, d);
    assert_eq!(table.len(), (max_pos - offset) * d);
    for t in 0..max_pos - offset {
        for j in 0..d {
            let want = pos[(t + offset) * d + j] + type_row[j];
            assert_eq!(table[t * d + j], want, "row {t}, col {j}");
        }
    }
}
