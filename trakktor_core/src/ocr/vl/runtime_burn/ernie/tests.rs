use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};

use super::Cache;
use crate::ocr::vl::config::{ModelConfig, VisionConfig};

type B = NdArray<f32>;

/// A toy geometry: two layers, two key/value heads of six, room for four
/// positions. Only the fields the cache reads matter; the rest are filler.
fn config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 4,
        num_hidden_layers: 2,
        num_key_value_heads: 2,
        head_dim: 6,
        vocab_size: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        mrope_section: vec![1, 1, 1],
        image_token_id: 0,
        vision: VisionConfig {
            image_size: 28,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            num_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            layer_norm_eps: 1e-6,
        },
    }
}

fn filled(from: f32, shape: [usize; 4]) -> Tensor<B, 4> {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| from + i as f32).collect();
    Tensor::from_data(TensorData::new(values, shape), &NdArrayDevice::Cpu)
}

fn host(tensor: Tensor<B, 4>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn the_cache_returns_everything_pushed_in_order() {
    let cfg = config();
    let mut cache = Cache::<B>::new(&cfg, 4, &NdArrayDevice::Cpu);
    assert!(cache.is_empty());

    // A three-position prefill, then one decode step.
    let key = filled(0.0, [1, 2, 3, 6]);
    let value = filled(100.0, [1, 2, 3, 6]);
    let (keys, _) = cache.push(0, key.clone(), value.clone());
    assert_eq!(keys.dims(), [1, 2, 3, 6]);
    assert_eq!(host(keys), host(key.clone()));
    cache.push(1, key, value);
    cache.advance(3);
    assert_eq!(cache.len(), 3);

    let step_key = filled(1000.0, [1, 2, 1, 6]);
    let step_value = filled(2000.0, [1, 2, 1, 6]);
    let (keys, values) = cache.push(0, step_key, step_value);
    assert_eq!(keys.dims(), [1, 2, 4, 6]);
    // Per head: the prefill's three positions, then the step's one.
    let expected: Vec<f32> = (0..2)
        .flat_map(|head| {
            let mut rows: Vec<f32> =
                (0..18).map(|i| (head * 18 + i) as f32).collect();
            rows.extend((0..6).map(|i| 1000.0 + (head * 6 + i) as f32));
            rows
        })
        .collect();
    assert_eq!(host(keys), expected);
    assert_eq!(
        values.clone().narrow(2, 3, 1).dims(),
        [1, 2, 1, 6],
        "the step's values are the fourth position"
    );
    cache.advance(1);
    assert_eq!(cache.len(), 4);
}

#[test]
fn each_layer_keeps_its_own_positions() {
    let cfg = config();
    let mut cache = Cache::<B>::new(&cfg, 4, &NdArrayDevice::Cpu);
    let first = filled(1.0, [1, 2, 2, 6]);
    let second = filled(500.0, [1, 2, 2, 6]);
    cache.push(0, first.clone(), first.clone());
    cache.push(1, second.clone(), second.clone());
    cache.advance(2);

    let probe = filled(0.0, [1, 2, 1, 6]);
    let (of_first, _) = cache.push(0, probe.clone(), probe.clone());
    let (of_second, _) = cache.push(1, probe.clone(), probe);
    let head = |t: Tensor<B, 4>| host(t.narrow(2, 0, 2));
    assert_eq!(head(of_first), host(first));
    assert_eq!(head(of_second), host(second));
}

#[test]
#[should_panic(expected = "outgrew its cache")]
fn the_cache_refuses_to_outgrow_its_capacity() {
    let cfg = config();
    let mut cache = Cache::<B>::new(&cfg, 4, &NdArrayDevice::Cpu);
    let three = filled(0.0, [1, 2, 3, 6]);
    cache.push(0, three.clone(), three);
    cache.advance(3);
    let two = filled(0.0, [1, 2, 2, 6]);
    cache.push(0, two.clone(), two);
}
