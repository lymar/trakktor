//! Reader tests. The ones marked `#[ignore]` need the published artifacts in
//! the model directory (`~/.trakktor/ocr/paddle/<model>/`), which one run of
//! `trakktor ocr paddle` puts there.
//!
//! The expected sums are independent of this code: they were taken from the
//! same tensors loaded through PaddlePaddle itself, so a reader that pairs a
//! name with the wrong record fails here even when the shapes happen to
//! match — and they usually do, since a model of 234 tensors has only about
//! 40 distinct shapes.

use std::path::PathBuf;

use super::{Artifact, records, tensor_desc};

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle")
        .join(name)
}

fn sum(values: &[f32]) -> f64 { values.iter().map(|v| f64::from(*v)).sum() }

#[test]
fn reads_an_unpacked_tensor_descriptor() {
    // data_type = 5 (float32), dims = [16, 3, 3, 3]; the encoding published
    // models use, one tag byte per dimension.
    let bytes = [0x08, 0x05, 0x10, 0x10, 0x10, 0x03, 0x10, 0x03, 0x10, 0x03];
    let (data_type, dims) = tensor_desc(&bytes).unwrap();
    assert_eq!(data_type, 5);
    assert_eq!(dims, vec![16, 3, 3, 3]);
}

#[test]
fn reads_a_packed_tensor_descriptor() {
    // The same tensor with the repeated field packed — not what Paddle writes
    // today, but a legal encoding of the same message.
    let bytes = [0x08, 0x05, 0x12, 0x04, 0x10, 0x03, 0x03, 0x03];
    let (data_type, dims) = tensor_desc(&bytes).unwrap();
    assert_eq!(data_type, 5);
    assert_eq!(dims, vec![16, 3, 3, 3]);
}

#[test]
fn rejects_a_dimension_of_zero() {
    let bytes = [0x08, 0x05, 0x10, 0x00];
    assert!(tensor_desc(&bytes).is_err());
}

#[test]
fn rejects_an_unknown_descriptor_field() {
    let bytes = [0x08, 0x05, 0x18, 0x01];
    assert!(tensor_desc(&bytes).is_err());
}

#[test]
fn walks_a_synthetic_weights_file() {
    // Two records: a [3] and a [4, 3], written exactly as the combining
    // writer would.
    let mut bytes = Vec::new();
    let mut record = |dims: &[u8], values: &[f32]| {
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut desc = vec![0x08, 0x05];
        for dim in dims {
            desc.extend_from_slice(&[0x10, *dim]);
        }
        bytes.extend_from_slice(&(desc.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&desc);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    };
    record(&[3], &[1.0, 2.0, 3.0]);
    record(&[4, 3], &[0.5; 12]);

    let records = records(&bytes).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].dims, vec![3]);
    assert_eq!(records[1].dims, vec![4, 3]);
    assert_eq!(records[1].payload.len(), 12 * 4);
}

#[test]
fn a_truncated_weights_file_is_an_error() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&[0x08, 0x05, 0x10, 0x03]);
    bytes.extend_from_slice(&1.0f32.to_le_bytes()); // one float short of [3]
    assert!(records(&bytes).is_err());
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv5_mobile_det"]
fn loads_the_mobile_detector() {
    let artifact = Artifact::load(&model_dir("PP-OCRv5_mobile_det")).unwrap();
    assert_eq!(artifact.len(), 234);
    assert_eq!(artifact.input_dims, vec![-1, 3, -1, -1]);
    assert_eq!(artifact.output_dims, vec![-1, 1, -1, -1]);

    let first = artifact.shaped("batch_norm2d_0.b_0", &[16]).unwrap();
    assert!((sum(&first.data) - -0.005018).abs() < 1e-5);
    assert!((first.data[0] - -0.030_660_16).abs() < 1e-7);
    assert!((first.data[15] - 0.263_144_67).abs() < 1e-7);

    // Deep in the network, and a shape shared with several other tensors:
    // this is the one that catches an off-by-one in the ordering rule.
    let mid = artifact
        .shaped("conv2d_186.w_0", &[384, 384, 1, 1])
        .unwrap();
    assert!((sum(&mid.data) - -379.976_930).abs() < 1e-3);

    // The last tensor in the file: the bias of a learnable affine block.
    let last = artifact
        .shaped("learnable_affine_block_9.w_1", &[1])
        .unwrap();
    assert_eq!(last.data[0], -3.078_864_4e-5);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn loads_the_eastern_slavic_recognizer() {
    let artifact =
        Artifact::load(&model_dir("eslav_PP-OCRv5_mobile_rec")).unwrap();
    assert_eq!(artifact.len(), 234);
    assert_eq!(artifact.input_dims, vec![-1, 3, 48, -1]);
    assert_eq!(artifact.output_dims, vec![-1, -1, 519]);
    assert_eq!(artifact.output_classes(), Some(519));

    let head = artifact.shaped("linear_8.w_0", &[120, 519]).unwrap();
    assert!((sum(&head.data) - -5080.853_686).abs() < 1e-2);
    let bias = artifact.shaped("linear_8.b_0", &[519]).unwrap();
    assert!((sum(&bias.data) - -51.907_578).abs() < 1e-3);
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv5_mobile_det"]
fn reports_a_missing_weight_by_name() {
    let artifact = Artifact::load(&model_dir("PP-OCRv5_mobile_det")).unwrap();
    let error = artifact.tensor("no_such_weight").unwrap_err();
    assert!(error.to_string().contains("no_such_weight"));
}
