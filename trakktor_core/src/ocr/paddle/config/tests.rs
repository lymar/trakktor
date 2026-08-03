//! Tests for the shipped model description. The published shapes are small
//! enough to state inline; the `#[ignore]`d ones check the real files.

use std::path::PathBuf;

use super::{ModelConfig, PostProcess};

fn model_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle")
        .join(name)
}

fn parse(text: &str) -> ModelConfig {
    ModelConfig::parse(&serde_json::from_str(text).unwrap()).unwrap()
}

#[test]
fn reads_a_detector_description() {
    let config = parse(
        r#"{
          "Global": {"model_name": "PP-OCRv5_mobile_det"},
          "PreProcess": {"transform_ops": [
            {"DecodeImage": {"channel_first": false, "img_mode": "BGR"}},
            {"DetLabelEncode": null},
            {"DetResizeForTest": {"resize_long": 960}},
            {"NormalizeImage": {"mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                                "scale": "1./255.", "order": "hwc"}},
            {"ToCHWImage": null}
          ]},
          "PostProcess": {"name": "DBPostProcess", "thresh": 0.3,
                          "box_thresh": 0.6, "max_candidates": 1000,
                          "unclip_ratio": 1.5}
        }"#,
    );
    assert_eq!(config.model_name, "PP-OCRv5_mobile_det");
    let normalize = config.normalize.unwrap();
    // The scale arrives as the string `1./255.`, which upstream evaluates.
    assert!((normalize.scale - 1.0 / 255.0).abs() < 1e-9);
    assert_eq!(normalize.mean, [0.485, 0.456, 0.406]);
    let PostProcess::Db(db) = config.post else {
        panic!("not a detector")
    };
    assert_eq!(db.thresh, 0.3);
    assert_eq!(db.max_candidates, 1000);
}

#[test]
fn reads_a_recognizer_description() {
    let config = parse(
        r#"{
          "Global": {"model_name": "eslav_PP-OCRv5_mobile_rec"},
          "PreProcess": {"transform_ops": [
            {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
            {"RecResizeImg": {"image_shape": [3, 48, 320]}}
          ]},
          "PostProcess": {"name": "CTCLabelDecode",
                          "character_dict": ["!", "\"", "А", "я"]}
        }"#,
    );
    assert_eq!(config.rec_image_shape, Some([3, 48, 320]));
    assert_eq!(config.characters().unwrap().len(), 4);
    assert_eq!(config.characters().unwrap()[2], "А");
    assert!(config.normalize.is_none());
}

#[test]
fn reads_a_classifier_description() {
    // Two shapes differ here from every other model: the post-processing is a
    // map keyed by class name, and `size` is [width, height].
    let config = parse(
        r#"{
          "Global": {"model_name": "PP-LCNet_x1_0_textline_ori"},
          "PreProcess": {"transform_ops": [
            {"ResizeImage": {"size": [160, 80]}},
            {"NormalizeImage": {"channel_num": 3,
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                                "order": "", "scale": 0.00392156862745098}},
            {"ToCHWImage": null}
          ]},
          "PostProcess": {"Topk": {"topk": 1,
                                   "label_list": ["0_degree", "180_degree"]}}
        }"#,
    );
    assert_eq!(config.cls_image_size, Some([160, 80]));
    let PostProcess::Topk { labels } = config.post else {
        panic!("not a classifier")
    };
    assert_eq!(labels, vec!["0_degree", "180_degree"]);
    let normalize = config.normalize.unwrap();
    assert!((normalize.scale - 1.0 / 255.0).abs() < 1e-7);
}

#[test]
fn rejects_an_unknown_post_processing() {
    let value: serde_json::Value = serde_json::from_str(
        r#"{"Global": {"model_name": "x"},
            "PostProcess": {"name": "SASTPostProcess"}}"#,
    )
    .unwrap();
    let error = ModelConfig::parse(&value).unwrap_err();
    assert!(error.to_string().contains("SASTPostProcess"));
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/eslav_PP-OCRv5_mobile_rec"]
fn the_published_eastern_slavic_dictionary_has_517_entries() {
    let config =
        ModelConfig::load(&model_dir("eslav_PP-OCRv5_mobile_rec")).unwrap();
    let characters = config.characters().unwrap();
    assert_eq!(characters.len(), 517);
    assert!(characters.iter().any(|c| c == "ё"));
    assert!(characters.iter().any(|c| c == "–"));
}
