//! Classifier tests. The ones marked `#[ignore]` need the published artifact
//! in the model directory (`~/.trakktor/ocr/paddle/<model>/`), which one run of
//! `trakktor ocr paddle` puts there.
//!
//! The expected probabilities were read off the exported graph running under
//! PaddlePaddle itself. They sit close enough to their neighbours to catch the
//! two mistakes this model invites: reading the crop in the wrong channel
//! order, and taking its dropout for a no-op at inference.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};

use super::*;
use crate::ocr::paddle::{artifact::Artifact, net::Loader};

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle")
        .join(name)
}

/// A crop of one solid colour, `width` by `height`.
fn solid(width: usize, height: usize, colour: [u8; 3]) -> Crop {
    Crop {
        width,
        height,
        bgr: colour
            .iter()
            .cycle()
            .take(width * height * CHANNELS)
            .copied()
            .collect(),
    }
}

#[test]
fn a_crop_is_squeezed_to_the_fixed_size_and_read_red_first() {
    // A crop of no particular shape, in a colour whose three channels differ,
    // so a swapped pair cannot hide.
    let crop = solid(37, 11, [0, 128, 255]);
    let tensor = batch_tensor(&[&crop], &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, CHANNELS, HEIGHT, WIDTH]);

    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let plane = HEIGHT * WIDTH;
    let expected = |value: f64, channel: usize| {
        ((value / 255.0 - MEAN[channel]) / STD[channel]) as f32
    };
    // Red first, then green, then blue — the reverse of the order the crop
    // arrives in.
    assert!(
        (values[0] - expected(255.0, 0)).abs() < 1e-5,
        "{}",
        values[0]
    );
    assert!(
        (values[plane] - expected(128.0, 1)).abs() < 1e-5,
        "{}",
        values[plane]
    );
    assert!(
        (values[2 * plane] - expected(0.0, 2)).abs() < 1e-5,
        "{}",
        values[2 * plane]
    );
    // A solid crop stays solid however it is scaled.
    assert_eq!(values[plane - 1], values[0]);
}

#[test]
fn a_crop_without_pixels_is_never_classified() {
    let crop = Crop::empty();
    let tensor = batch_tensor(&[&crop], &Device::Cpu).unwrap();
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|value| *value == 0.0));
}

#[test]
fn turning_a_crop_around_reverses_both_axes() {
    // Three pixels across and two down, each a different grey, so every pixel
    // can be told from every other.
    let crop = Crop {
        width: 3,
        height: 2,
        bgr: vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
    };
    let turned = turn_around(&crop);
    assert_eq!(turned.width, 3);
    assert_eq!(turned.height, 2);
    // The last pixel becomes the first, with no row or column lost or blacked
    // out on the way.
    assert_eq!(
        turned.bgr,
        vec![6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    );
    assert_eq!(turn_around(&turned), crop);
    assert!(turn_around(&Crop::empty()).is_empty());
}

/// The deterministic input the goldens below were measured on.
fn synthetic(channels: usize, height: usize, width: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(channels * height * width);
    for c in 0..channels {
        for i in 0..height {
            for j in 0..width {
                let raw = ((c * 7 + i * 13 + j * 29) % 251) as f32;
                let mut value = raw / 255.0;
                value -= 0.5;
                value /= 0.5;
                data.push(value);
            }
        }
    }
    data
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-LCNet_x1_0_textline_ori"]
fn the_published_classifier_reproduces_its_reference_output() {
    let dir = model_dir("PP-LCNet_x1_0_textline_ori");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let classifier =
        Classifier::load(&Loader::new(&artifact, &device)).unwrap();

    let data = synthetic(CHANNELS, HEIGHT, WIDTH);
    let sum: f64 = data.iter().map(|v| f64::from(*v)).sum();
    assert!((sum - -769.246_582).abs() < 1e-2, "{sum}");

    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, WIDTH), &device).unwrap();
    let out = classifier.forward(&input).unwrap();
    assert_eq!(out.dims(), &[1, CLASSES]);
    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Leaving the inference-time dropout out moves these by 3.4e-2, which the
    // tolerance below does not forgive.
    assert!((values[0] - 0.356_794_36).abs() < 1e-5, "{}", values[0]);
    assert!((values[1] - 0.643_205_64).abs() < 1e-5, "{}", values[1]);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-LCNet_x1_0_textline_ori"]
fn a_mid_grey_crop_is_read_too() {
    let dir = model_dir("PP-LCNet_x1_0_textline_ori");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let classifier =
        Classifier::load(&Loader::new(&artifact, &device)).unwrap();

    let input =
        Tensor::zeros((1, CHANNELS, HEIGHT, WIDTH), DType::F32, &device)
            .unwrap();
    let values = classifier
        .forward(&input)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!((values[0] - 0.426_920_74).abs() < 1e-5, "{}", values[0]);
    assert!((values[1] - 0.573_079_2).abs() < 1e-5, "{}", values[1]);
    // Neither class reaches the threshold, so nothing is turned around on the
    // strength of a reading like this one.
    assert!(values[UPSIDE_DOWN] < THRESHOLD);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-LCNet_x1_0_textline_ori"]
fn an_undecided_crop_is_left_alone() {
    let dir = model_dir("PP-LCNet_x1_0_textline_ori");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let classifier =
        Classifier::load(&Loader::new(&artifact, &device)).unwrap();

    // Solid colour carries no direction, and a two-class model that has to
    // answer anyway should not be believed.
    let crops = vec![solid(200, 32, [90, 90, 90]), Crop::empty()];
    let scores = classifier.probabilities(&crops).unwrap();
    assert!((0.0..=1.0).contains(&scores[0]), "{}", scores[0]);
    assert_eq!(scores[1], 0.0, "a crop with no pixels is not classified");
    assert_eq!(classifier.upside_down(&crops).unwrap(), vec![false, false]);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-LCNet_x0_25_textline_ori"]
fn the_narrow_classifier_loads_from_the_same_description() {
    // The two published text-line classifiers are one network at two scales;
    // the widths come off the file, so nothing here needs a second table.
    let dir = model_dir("PP-LCNet_x0_25_textline_ori");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let classifier =
        Classifier::load(&Loader::new(&artifact, &device)).unwrap();

    let input = Tensor::from_vec(
        synthetic(CHANNELS, HEIGHT, WIDTH),
        (1, CHANNELS, HEIGHT, WIDTH),
        &device,
    )
    .unwrap();
    let values = classifier
        .forward(&input)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let total: f64 = values.iter().map(|v| f64::from(*v)).sum();
    assert!((total - 1.0).abs() < 1e-5, "{total}");
}
