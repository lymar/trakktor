//! Unit tests for the primitives, checked against values computed by hand.

use candle_core::{Device, IndexOp, Tensor};

use super::*;

/// Builds a `[batch, channels, time]` tensor from a row-major slice.
fn tensor3(values: &[f32], shape: (usize, usize, usize)) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape, &Device::Cpu).expect("tensor")
}

#[test]
fn rms_norm_scales_by_the_root_mean_square() {
    let vb = candle_nn::VarBuilder::from_tensors(
        [(
            "weight".to_string(),
            Tensor::from_vec(vec![1f32, 1.0, 1.0, 1.0], 4, &Device::Cpu)
                .expect("weight"),
        )]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let norm = RmsNorm::load(4, 0.0, vb).expect("load");

    let xs = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu)
        .expect("input");
    let out = norm.forward(&xs).expect("forward");
    let out = out
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");

    // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    let rms = 7.5f32.sqrt();
    for (got, want) in out.iter().zip([1.0, 2.0, 3.0, 4.0].map(|v| v / rms)) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

#[test]
fn causal_conv_never_reads_the_future() {
    // A width-3 kernel picking only the newest sample must reproduce its
    // input; picking only the oldest must shift it right by two.
    let newest =
        Tensor::from_vec(vec![0f32, 0.0, 1.0], (1, 1, 3), &Device::Cpu)
            .expect("kernel");
    let oldest =
        Tensor::from_vec(vec![1f32, 0.0, 0.0], (1, 1, 3), &Device::Cpu)
            .expect("kernel");
    let bias =
        Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu).expect("bias");
    let xs = tensor3(&[1.0, 2.0, 3.0, 4.0], (1, 1, 4));

    for (weight, want) in [
        (newest, [1.0, 2.0, 3.0, 4.0]),
        (oldest, [0.0, 0.0, 1.0, 2.0]),
    ] {
        let vb = candle_nn::VarBuilder::from_tensors(
            [
                ("conv.weight".to_string(), weight),
                ("conv.bias".to_string(), bias.clone()),
            ]
            .into_iter()
            .collect(),
            candle_core::DType::F32,
            &Device::Cpu,
        );
        let conv = CausalConv1d::load(1, 1, 3, 1, 1, vb).expect("load");
        let out = conv.forward(&xs).expect("forward");
        // Causality also means the length is preserved.
        assert_eq!(out.dims(), &[1, 1, 4]);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want.to_vec());
    }
}

#[test]
fn snake_beta_is_identity_when_the_sine_vanishes() {
    let zeros =
        Tensor::zeros(2, candle_core::DType::F32, &Device::Cpu).expect("zeros");
    let vb = candle_nn::VarBuilder::from_tensors(
        [
            ("alpha".to_string(), zeros.clone()),
            ("beta".to_string(), zeros),
        ]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let snake = SnakeBeta::load(2, vb).expect("load");

    // With alpha = 0 → e⁰ = 1, the activation is x + sin²(x); at x = 0 and
    // x = π the sine term vanishes, leaving the input untouched.
    let xs = tensor3(
        &[0.0, std::f32::consts::PI, 0.0, std::f32::consts::PI],
        (1, 2, 2),
    );
    let out = snake.forward(&xs).expect("forward");
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (got, want) in
        got.iter()
            .zip([0.0, std::f32::consts::PI, 0.0, std::f32::consts::PI])
    {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}

#[test]
fn rotate_half_swaps_and_negates() {
    let xs = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu)
        .expect("input");
    let out = rotate_half(&xs).expect("rotate");
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0]);
}

#[test]
fn rope_tables_start_at_the_identity_rotation() {
    let (cos, sin) = rope_tables(4, 3, 10000.0, &Device::Cpu).expect("tables");
    assert_eq!(cos.dims(), &[3, 2]);

    // Position 0 rotates by nothing: cos = 1, sin = 0.
    let first_cos = cos.i(0).unwrap().to_vec1::<f32>().unwrap();
    let first_sin = sin.i(0).unwrap().to_vec1::<f32>().unwrap();
    for value in first_cos {
        assert!((value - 1.0).abs() < 1e-6);
    }
    for value in first_sin {
        assert!(value.abs() < 1e-6);
    }
}
