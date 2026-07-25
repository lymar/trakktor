//! Cross-runtime equivalence of the codec's primitives.
//!
//! The codec is the stage whose output has to reproduce, and it is built from
//! four pieces the two runtimes implement independently: the causal
//! convolutions, the transposed one, and the Snake activation (whose
//! parameters this runtime exponentiates at load rather than per call). Each is
//! checked here against the candle implementation on the same synthetic
//! weights, so a divergence surfaces as a failing unit test rather than as
//! audio that sounds slightly wrong.

use std::collections::HashMap;

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};
use candle_core::{Device, Tensor as CandleTensor};

use super::*;
use crate::tts::qwen3_tts::runtime::layers as candle_layers;

type B = NdArray<f32>;

/// The name both loaders hang the layer's tensors under.
const PREFIX: &str = "layer";

/// Deterministic pseudo-random values, so both runtimes see the same weights.
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 - 0.5
        })
        .collect()
}

/// Writes named tensors to a safetensors file and opens both runtimes' readers
/// on it.
struct Fixture {
    _dir: tempfile::TempDir,
    weights: Weights,
    tensors: HashMap<String, CandleTensor>,
}

impl Fixture {
    fn new(tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) -> Self {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("weights.safetensors");
        let tensors: HashMap<String, CandleTensor> = tensors
            .into_iter()
            .map(|(name, values, shape)| {
                let tensor =
                    CandleTensor::from_vec(values, shape, &Device::Cpu)
                        .expect("tensor");
                (name.to_string(), tensor)
            })
            .collect();
        candle_core::safetensors::save(&tensors, &path).expect("save");
        Self {
            weights: Weights::open(&path).expect("open"),
            tensors,
            _dir: dir,
        }
    }

    /// A candle var-builder rooted at the same prefix the burn loader takes.
    fn var_builder(&self) -> candle_nn::VarBuilder<'_> {
        candle_nn::VarBuilder::from_tensors(
            self.tensors.clone(),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .pp(PREFIX)
    }
}

/// Runs the same `[1, channels, time]` input through both implementations.
fn compare(
    input: &[f32],
    channels: usize,
    time: usize,
    burn: impl FnOnce(Tensor<B, 3>) -> Tensor<B, 3>,
    candle: impl FnOnce(&CandleTensor) -> candle_core::Result<CandleTensor>,
) {
    let mine = burn(Tensor::from_data(
        TensorData::new(input.to_vec(), [1, channels, time]),
        &NdArrayDevice::Cpu,
    ))
    .into_data()
    .to_vec::<f32>()
    .expect("burn output");

    let theirs = candle(
        &CandleTensor::from_vec(
            input.to_vec(),
            (1, channels, time),
            &Device::Cpu,
        )
        .expect("input"),
    )
    .expect("candle output")
    .flatten_all()
    .and_then(|t| t.to_vec1::<f32>())
    .expect("values");

    assert_eq!(mine.len(), theirs.len(), "output lengths differ");
    for (index, (got, want)) in mine.iter().zip(&theirs).enumerate() {
        assert!((got - want).abs() < 1e-5, "sample {index}: {got} vs {want}");
    }
}

#[test]
fn the_causal_convolution_matches_candle() {
    let (in_channels, out_channels, kernel, dilation, time) = (3, 4, 7, 3, 11);
    let fixture = Fixture::new(vec![
        (
            "layer.conv.weight",
            values(out_channels * in_channels * kernel, 7),
            vec![out_channels, in_channels, kernel],
        ),
        (
            "layer.conv.bias",
            values(out_channels, 13),
            vec![out_channels],
        ),
    ]);
    let mine = CausalConv1d::<B>::load(
        &fixture.weights,
        &NdArrayDevice::Cpu,
        PREFIX,
        in_channels,
        out_channels,
        kernel,
        dilation,
        1,
    )
    .expect("burn load");
    let theirs = candle_layers::CausalConv1d::load(
        in_channels,
        out_channels,
        kernel,
        dilation,
        1,
        fixture.var_builder(),
    )
    .expect("candle load");

    compare(
        &values(in_channels * time, 21),
        in_channels,
        time,
        |x| mine.forward(x),
        |x| theirs.forward(x),
    );
}

#[test]
fn the_depthwise_causal_convolution_matches_candle() {
    let (channels, kernel, time) = (5, 7, 9);
    let fixture = Fixture::new(vec![
        (
            "layer.conv.weight",
            values(channels * kernel, 31),
            vec![channels, 1, kernel],
        ),
        ("layer.conv.bias", values(channels, 37), vec![channels]),
    ]);
    let mine = CausalConv1d::<B>::load(
        &fixture.weights,
        &NdArrayDevice::Cpu,
        PREFIX,
        channels,
        channels,
        kernel,
        1,
        channels,
    )
    .expect("burn load");
    let theirs = candle_layers::CausalConv1d::load(
        channels,
        channels,
        kernel,
        1,
        channels,
        fixture.var_builder(),
    )
    .expect("candle load");

    compare(
        &values(channels * time, 41),
        channels,
        time,
        |x| mine.forward(x),
        |x| theirs.forward(x),
    );
}

#[test]
fn the_transposed_convolution_matches_candle_including_the_trim() {
    let (in_channels, out_channels, stride, time) = (4, 2, 3, 6);
    let kernel = 2 * stride;
    let fixture = Fixture::new(vec![
        (
            "layer.conv.weight",
            values(in_channels * out_channels * kernel, 51),
            vec![in_channels, out_channels, kernel],
        ),
        (
            "layer.conv.bias",
            values(out_channels, 57),
            vec![out_channels],
        ),
    ]);
    let mine = CausalConvTranspose1d::<B>::load(
        &fixture.weights,
        &NdArrayDevice::Cpu,
        PREFIX,
        in_channels,
        out_channels,
        kernel,
        stride,
    )
    .expect("burn load");
    let theirs = candle_layers::CausalConvTranspose1d::load(
        in_channels,
        out_channels,
        kernel,
        stride,
        fixture.var_builder(),
    )
    .expect("candle load");

    compare(
        &values(in_channels * time, 61),
        in_channels,
        time,
        |x| {
            let out = mine.forward(x);
            // Causality also means the upsample is exact: stride samples out
            // per sample in.
            assert_eq!(out.dims(), [1, out_channels, time * stride]);
            out
        },
        |x| theirs.forward(x),
    );
}

#[test]
fn the_snake_activation_matches_candle_despite_folding_the_exponentials() {
    let (channels, time) = (4, 8);
    let fixture = Fixture::new(vec![
        ("layer.alpha", values(channels, 71), vec![channels]),
        ("layer.beta", values(channels, 73), vec![channels]),
    ]);
    let mine = SnakeBeta::<B>::load(
        &fixture.weights,
        &NdArrayDevice::Cpu,
        PREFIX,
        channels,
    )
    .expect("burn load");
    let theirs =
        candle_layers::SnakeBeta::load(channels, fixture.var_builder())
            .expect("candle load");

    compare(
        &values(channels * time, 79),
        channels,
        time,
        |x| mine.forward(x),
        |x| theirs.forward(x),
    );
}
