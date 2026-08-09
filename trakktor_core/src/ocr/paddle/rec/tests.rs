//! Recognizer tests. The ones marked `#[ignore]` need the published artifact
//! in the model directory (`~/.trakktor/ocr/paddle/<model>/`), which one run of
//! `trakktor ocr paddle` puts there.
//!
//! The expected outputs of those are independent of this code: they were read
//! off the exported graph running under PaddlePaddle itself, so a port that
//! merely looks plausible — a stride on the wrong axis, an epsilon shared
//! between layers that do not share one — fails here rather than in someone's
//! document.

use std::path::PathBuf;

use candle_core::{DType, Device};

use super::*;
use crate::ocr::paddle::{
    artifact::Artifact, config::ModelConfig, net::Loader,
};

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle")
        .join(name)
}

fn labels(characters: &[&str]) -> Labels {
    let characters: Vec<String> =
        characters.iter().map(|c| (*c).to_string()).collect();
    Labels::new(&characters, characters.len() + 2).expect("a matching table")
}

/// One probability row per step, peaked on the named class and with the rest
/// of the mass spread evenly over the others.
fn probabilities(classes: usize, peaks: &[(usize, f32)]) -> Vec<f32> {
    let mut data = vec![0f32; peaks.len() * classes];
    for (step, (class, probability)) in peaks.iter().enumerate() {
        let rest = (1.0 - probability) / (classes - 1) as f32;
        for slot in &mut data[step * classes..(step + 1) * classes] {
            *slot = rest;
        }
        data[step * classes + class] = *probability;
    }
    data
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
fn decoding_collapses_repeats_and_drops_blanks() {
    // Classes are [blank, a, b, space]. The two `a`s in a row are one
    // character; the third, which a blank separates from them, is another.
    let labels = labels(&["a", "b"]);
    let steps = probabilities(
        labels.len(),
        &[
            (0, 0.9),
            (1, 0.8),
            (1, 0.7),
            (0, 0.6),
            (1, 0.5),
            (3, 0.95),
            (2, 0.85),
        ],
    );
    let reading = decode(&steps, labels.len(), &labels);
    assert_eq!(reading.text, "aa b");
    // The score is the mean of the steps that were kept, and of those alone.
    assert!((reading.score - 0.775).abs() < 1e-6, "{}", reading.score);
}

#[test]
fn decoding_nothing_but_blanks_reads_nothing() {
    let labels = labels(&["a", "b"]);
    let steps = probabilities(labels.len(), &[(0, 0.4), (0, 0.9), (0, 0.5)]);
    let reading = decode(&steps, labels.len(), &labels);
    assert_eq!(reading, Reading::default());
    assert_eq!(reading.score, 0.0, "an empty reading scores zero, not NaN");
}

#[test]
fn the_space_is_the_last_class() {
    let labels = labels(&["a", "b"]);
    assert_eq!(labels.text(BLANK), "");
    assert_eq!(labels.text(1), "a");
    assert_eq!(labels.text(3), " ");
    let steps = probabilities(labels.len(), &[(3, 0.6), (3, 0.6)]);
    assert_eq!(decode(&steps, labels.len(), &labels).text, " ");
}

#[test]
fn a_tie_goes_to_the_lower_class() {
    let labels = labels(&["a", "b"]);
    let steps = vec![0.1, 0.4, 0.4, 0.1];
    assert_eq!(decode(&steps, labels.len(), &labels).text, "a");
}

#[test]
fn a_dictionary_that_does_not_fit_the_class_count_is_refused() {
    let characters = vec!["a".to_string(), "b".to_string()];
    assert!(Labels::new(&characters, 4).is_ok());
    let error = Labels::new(&characters, 5).unwrap_err();
    assert!(error.to_string().contains('5'), "{error}");
}

#[test]
fn the_batch_width_follows_the_widest_crop() {
    let short = solid(48, HEIGHT, [0, 0, 0]);
    let ordinary = solid(100, HEIGHT, [0, 0, 0]);
    let long = solid(480, HEIGHT, [0, 0, 0]);
    let rule = solid(4_000, HEIGHT, [0, 0, 0]);
    // Nothing wider than the exported shape leaves the exported width.
    assert_eq!(batch_width(&[]), WIDTH);
    assert_eq!(batch_width(&[&ordinary, &short]), WIDTH);
    // A long crop takes the whole batch with it.
    assert_eq!(batch_width(&[&ordinary, &long]), 480);
    // And a crop long enough to be a printed rule is capped.
    assert_eq!(batch_width(&[&rule]), MAX_WIDTH);
}

#[test]
fn a_crop_is_resized_to_its_own_width_and_padded_with_mid_grey() {
    let white = solid(24, 48, [255, 255, 255]);
    let tensor = batch_tensor(&[&white], &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, CHANNELS, HEIGHT, WIDTH]);
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // The crop is already 48 tall, so it keeps its 24 columns; white
    // normalizes to the top of the range and the padding stays at zero, which
    // is mid-grey rather than black.
    assert_eq!(values[0], 1.0);
    assert_eq!(values[23], 1.0);
    assert_eq!(values[24], 0.0);
    assert_eq!(values[WIDTH - 1], 0.0);
}

#[test]
fn a_crop_longer_than_its_batch_is_squeezed_to_fit() {
    // The longer crop sets the width; the shorter one keeps its own and pads
    // the rest, rather than being stretched across the batch.
    let short = solid(48, 48, [0, 0, 0]);
    let long = solid(960, 48, [255, 255, 255]);
    let crops = [&short, &long];
    let width = batch_width(&crops);
    assert_eq!(width, 960);
    let tensor = batch_tensor(&crops, &Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[2, CHANNELS, HEIGHT, width]);
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(values[47], -1.0);
    assert_eq!(values[48], 0.0);
    let second = CHANNELS * HEIGHT * width;
    assert_eq!(values[second + width - 1], 1.0);
}

#[test]
fn a_crop_without_pixels_is_never_read() {
    let empty = Crop::empty();
    assert!(!readable(&empty));
    // A crop whose buffer is shorter than the extent it claims is no more
    // readable than one with no extent at all.
    let truncated = Crop {
        width: 10,
        height: 10,
        bgr: vec![0; 10],
    };
    assert!(!readable(&truncated));
    let tensor = batch_tensor(&[&empty, &truncated], &Device::Cpu).unwrap();
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|value| *value == 0.0));
}

/// The deterministic input the goldens below were measured on: no image
/// decoding, no file, just an arithmetic pattern normalized the way a crop is.
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
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn the_published_recognizer_reproduces_its_reference_output() {
    let dir = model_dir("eslav_PP-OCRv5_mobile_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();
    assert_eq!(recognizer.classes(), 519);

    let data = synthetic(CHANNELS, HEIGHT, WIDTH);
    let sum: f64 = data.iter().map(|v| f64::from(*v)).sum();
    assert!((sum - -902.015_076).abs() < 1e-2, "{sum}");

    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, WIDTH), &device).unwrap();
    let out = recognizer.forward(&input).unwrap();
    // Every eight columns of the crop are one step of the sequence.
    assert_eq!(out.dims(), &[1, 40, 519]);

    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Every row is a distribution, so the whole tensor sums to the step count.
    let total: f64 = values.iter().map(|v| f64::from(*v)).sum();
    assert!((total - 40.0).abs() < 1e-3, "{total}");
    assert!((values[0] - 0.786_371_65).abs() < 1e-5, "{}", values[0]);
    assert!((values[1] - 3.008_013e-5).abs() < 1e-6, "{}", values[1]);
    assert!(
        (values[19 * 519] - 0.940_631_5).abs() < 1e-5,
        "{}",
        values[19 * 519]
    );

    // The pattern is not text, and the model says so: the blank wins at every
    // step, which decodes to nothing at all.
    let characters = vec![String::new(); 517];
    let labels = Labels::new(&characters, recognizer.classes()).unwrap();
    assert_eq!(decode(&values, 519, &labels), Reading::default());
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn the_sequence_grows_with_the_width() {
    let dir = model_dir("eslav_PP-OCRv5_mobile_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();

    let data = synthetic(CHANNELS, HEIGHT, 160);
    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, 160), &device).unwrap();
    let out = recognizer.forward(&input).unwrap();
    assert_eq!(out.dims(), &[1, 20, 519]);
    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((values[0] - 0.760_219_93).abs() < 1e-5, "{}", values[0]);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn a_batch_reads_each_crop_as_it_would_alone() {
    let dir = model_dir("eslav_PP-OCRv5_mobile_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();

    let one = Tensor::from_vec(
        synthetic(CHANNELS, HEIGHT, WIDTH),
        (1, CHANNELS, HEIGHT, WIDTH),
        &device,
    )
    .unwrap();
    let zeros =
        Tensor::zeros((1, CHANNELS, HEIGHT, WIDTH), DType::F32, &device)
            .unwrap();
    let alone = recognizer.forward(&one).unwrap();
    let together = recognizer
        .forward(&Tensor::cat(&[&zeros, &one], 0).unwrap())
        .unwrap();

    let alone = alone.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let both = together.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Nothing couples the rows of a batch; only the width they share does.
    let offset = 40 * 519;
    for (a, b) in alone.iter().zip(&both[offset..]) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    // The all-zero crop is the mid-grey one, and it is read, not skipped.
    assert!((both[0] - 0.884_510_3).abs() < 1e-5, "{}", both[0]);
}

/// The class count of the large recognizer, and its dictionary's length.
const SERVER_CLASSES: usize = 18_385;

/// The tolerance the large recognizer's goldens are pinned to, an order of
/// magnitude looser than the small one's.
///
/// It is arithmetic, not a shortcut. This backbone is eighty-one convolutions
/// deep and reduces up to 3328 channels at a time, and candle and PaddlePaddle
/// sum those in different orders; the two disagree by at most 3.4e-4 over the
/// whole 40x18385 output, with a mean of 1e-8, and the difference lands on
/// whichever class won rather than on any step, class or region in particular.
/// The likeliest class is the same at every step. The small recognizer, with a
/// backbone a third as deep and a fifth as wide, stays inside 6e-6.
const SERVER_TOLERANCE: f32 = 5e-4;

/// The same three checks against the large recognizer, whose backbone is a
/// different network entirely — the one the large detector carries, asked to
/// read a line instead of a page. The numbers come from the same reference
/// stand as the small one's, on the same synthetic input, so the two rows are
/// comparable and neither was fitted to this code.
#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv5_server_rec"]
fn the_published_server_recognizer_reproduces_its_reference_output() {
    let dir = model_dir("PP-OCRv5_server_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();
    assert_eq!(recognizer.classes(), SERVER_CLASSES);

    let data = synthetic(CHANNELS, HEIGHT, WIDTH);
    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, WIDTH), &device).unwrap();
    let out = recognizer.forward(&input).unwrap();
    // The height is spent the same way as in the small one, and so is the
    // width: every eight columns of the crop are one step.
    assert_eq!(out.dims(), &[1, 40, SERVER_CLASSES]);

    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // Every row is a distribution, so the whole tensor sums to the step count.
    let total: f64 = values.iter().map(|v| f64::from(*v)).sum();
    assert!((total - 40.0).abs() < 1e-2, "{total}");
    assert!(
        (values[0] - 0.929_732_56).abs() < SERVER_TOLERANCE,
        "{}",
        values[0]
    );
    assert!((values[1] - 2.650_504_3e-6).abs() < 1e-7, "{}", values[1]);
    let late = values[19 * SERVER_CLASSES];
    assert!((late - 0.843_882_86).abs() < SERVER_TOLERANCE, "{late}");

    // The pattern is not text, and this model says so too.
    let characters = vec![String::new(); SERVER_CLASSES - 2];
    let labels = Labels::new(&characters, recognizer.classes()).unwrap();
    assert_eq!(decode(&values, SERVER_CLASSES, &labels), Reading::default());
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv5_server_rec"]
fn the_server_sequence_grows_with_the_width() {
    let dir = model_dir("PP-OCRv5_server_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();

    let data = synthetic(CHANNELS, HEIGHT, 160);
    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, 160), &device).unwrap();
    let out = recognizer.forward(&input).unwrap();
    assert_eq!(out.dims(), &[1, 20, SERVER_CLASSES]);
    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        (values[0] - 0.858_798_15).abs() < SERVER_TOLERANCE,
        "{}",
        values[0]
    );
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv5_server_rec"]
fn a_server_batch_reads_each_crop_as_it_would_alone() {
    let dir = model_dir("PP-OCRv5_server_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();

    let one = Tensor::from_vec(
        synthetic(CHANNELS, HEIGHT, WIDTH),
        (1, CHANNELS, HEIGHT, WIDTH),
        &device,
    )
    .unwrap();
    let zeros =
        Tensor::zeros((1, CHANNELS, HEIGHT, WIDTH), DType::F32, &device)
            .unwrap();
    let both = recognizer
        .forward(&Tensor::cat(&[&zeros, &one], 0).unwrap())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        (both[0] - 0.677_395_2).abs() < SERVER_TOLERANCE,
        "{}",
        both[0]
    );
    let offset = 40 * SERVER_CLASSES;
    let mine = both[offset];
    assert!((mine - 0.929_732_56).abs() < SERVER_TOLERANCE, "{mine}");
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn the_models_own_dictionary_fits_its_head() {
    let dir = model_dir("eslav_PP-OCRv5_mobile_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let config = ModelConfig::load(&dir).unwrap();
    let characters = config.characters().expect("a recognizer's dictionary");
    // The dictionary, the blank and the space are the whole of the head.
    let labels =
        Labels::new(characters, artifact.output_classes().unwrap()).unwrap();
    assert_eq!(labels.len(), 519);
    assert_eq!(labels.text(BLANK), "");
    assert_eq!(labels.text(labels.len() - 1), " ");
    assert_eq!(labels.text(1), characters[0]);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn reading_scatters_the_batches_back_into_page_order() {
    let dir = model_dir("eslav_PP-OCRv5_mobile_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let config = ModelConfig::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();
    let labels =
        Labels::new(config.characters().unwrap(), recognizer.classes())
            .unwrap();

    // Crops of wildly different shapes, so the sort does move them, with an
    // unreadable one in the middle to be skipped.
    let crops = vec![
        solid(600, 48, [30, 30, 30]),
        Crop::empty(),
        solid(64, 48, [200, 200, 200]),
        solid(220, 24, [10, 90, 200]),
    ];
    let readings = recognizer.read(&crops, &labels).unwrap();
    assert_eq!(readings.len(), crops.len());
    // A crop with no pixels is not read at all.
    assert_eq!(readings[1], Reading::default());
    // Solid colour is not text; whatever the others read, they score in range.
    for reading in &readings {
        assert!((0.0..=1.0).contains(&reading.score), "{reading:?}");
    }
}

/// The class count of the newest recognizer, and its dictionary's length.
const MEDIUM_CLASSES: usize = 18_710;

/// The newest recognizer, which is a different network from either of the
/// others at every level: a reparameterized backbone whose weights are taken
/// in the order the graph reads them rather than by name, under a lighter
/// sequence encoder.
///
/// Because the weights are claimed in order, a wrong load is not a missing
/// name but a *shifted* one, and the shapes that follow usually still fit.
/// This test is what catches that: the numbers come from the same reference
/// stand, on the same synthetic input, as the other two recognizers'.
#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv6_medium_rec"]
fn the_published_medium_recognizer_reproduces_its_reference_output() {
    let dir = model_dir("PP-OCRv6_medium_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let device = Device::Cpu;
    let recognizer =
        Recognizer::load(&Loader::new(&artifact, &device)).unwrap();
    assert_eq!(recognizer.classes(), MEDIUM_CLASSES);

    let data = synthetic(CHANNELS, HEIGHT, WIDTH);
    let input =
        Tensor::from_vec(data, (1, CHANNELS, HEIGHT, WIDTH), &device).unwrap();
    let out = recognizer.forward(&input).unwrap();
    // The same sequence length as the older two, which is what lets the
    // pipeline hold any of them without knowing which.
    assert_eq!(out.dims(), &[1, 40, MEDIUM_CLASSES]);

    let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let total: f64 = values.iter().map(|v| f64::from(*v)).sum();
    assert!((total - 40.0).abs() < 1e-2, "{total}");
    assert!(
        (values[0] - 0.968_619_29).abs() < SERVER_TOLERANCE,
        "{}",
        values[0]
    );
    assert!((values[1] - 3.356_355_8e-6).abs() < 1e-7, "{}", values[1]);
    let late = values[19 * MEDIUM_CLASSES];
    assert!((late - 0.825_065_85).abs() < SERVER_TOLERANCE, "{late}");

    // The pattern is not text, and this model says so too.
    let characters = vec![String::new(); MEDIUM_CLASSES - 2];
    let labels = Labels::new(&characters, recognizer.classes()).unwrap();
    assert_eq!(decode(&values, MEDIUM_CLASSES, &labels), Reading::default());
}

/// The dictionary the newest recognizer carries is read out of the only
/// description it publishes, which is YAML rather than JSON, and it is a
/// third longer than the older generation's.
#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv6_medium_rec"]
fn the_medium_dictionary_fits_its_head() {
    let dir = model_dir("PP-OCRv6_medium_rec");
    let artifact = Artifact::load(&dir).unwrap();
    let config = ModelConfig::load(&dir).unwrap();
    let characters = config.characters().expect("a recognizer's dictionary");
    let labels =
        Labels::new(characters, artifact.output_classes().unwrap()).unwrap();
    assert_eq!(labels.len(), MEDIUM_CLASSES);
    assert_eq!(labels.text(BLANK), "");
    assert_eq!(labels.text(labels.len() - 1), " ");
    // The en dash is in this dictionary and in neither of the alphabet-bound
    // ones, which is the one difference a reader of English notices.
    assert!(characters.iter().any(|c| c == "\u{2013}"));
}

/// All three recognizers read a crop into the same number of steps, so the
/// pipeline can hold any of them without knowing which.
#[test]
#[ignore = "needs every recognizer in ~/.trakktor/ocr/paddle"]
fn every_recognizer_reads_a_crop_into_the_same_number_of_steps() {
    let device = Device::Cpu;
    let mut steps = Vec::new();
    for name in [
        "eslav_PP-OCRv5_mobile_rec",
        "PP-OCRv5_server_rec",
        "PP-OCRv6_medium_rec",
    ] {
        let artifact = Artifact::load(&model_dir(name)).unwrap();
        let recognizer =
            Recognizer::load(&Loader::new(&artifact, &device)).unwrap();
        let input = Tensor::from_vec(
            synthetic(CHANNELS, HEIGHT, WIDTH),
            (1, CHANNELS, HEIGHT, WIDTH),
            &device,
        )
        .unwrap();
        steps.push(recognizer.forward(&input).unwrap().dims()[1]);
    }
    assert_eq!(steps, vec![40, 40, 40], "{steps:?}");
}
