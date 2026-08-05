//! Reading a detection model's description.

use serde_json::json;

use super::*;

fn published() -> serde_json::Value {
    json!({
        "Global": { "model_name": "PP-DocLayout_plus-L" },
        "Preprocess": [
            { "type": "Resize", "interp": 2, "keep_ratio": false,
              "target_size": [800, 800] },
            { "type": "NormalizeImage", "norm_type": "none",
              "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0] },
            { "type": "Permute" }
        ],
        "label_list": LABEL_NAMES,
    })
}

#[test]
fn the_published_description_reads() {
    let config = LayoutConfig::parse(&published()).expect("it parses");
    assert_eq!(config.model_name, "PP-DocLayout_plus-L");
    assert_eq!(config.target_size, [800, 800]);
    assert_eq!(config.interp, 2);
    config
        .check_labels()
        .expect("the labels are the known ones");
}

#[test]
fn a_reordered_label_list_is_refused() {
    let mut value = published();
    let labels = value["label_list"].as_array_mut().expect("the labels");
    labels.swap(0, 2);
    let config = LayoutConfig::parse(&value).expect("it still parses");
    // The order *is* the class mapping; a file that moved it would otherwise
    // be read as a working model that calls a picture "text".
    let error = config.check_labels().unwrap_err().to_string();
    assert!(error.contains("class 0"), "{error}");
}

#[test]
fn a_shorter_label_list_is_refused() {
    let mut value = published();
    value["label_list"]
        .as_array_mut()
        .expect("the labels")
        .pop();
    let error = LayoutConfig::parse(&value)
        .expect("it parses")
        .check_labels()
        .unwrap_err()
        .to_string();
    assert!(error.contains("19 classes"), "{error}");
}

#[test]
fn keeping_the_aspect_ratio_is_refused() {
    let mut value = published();
    value["Preprocess"][0]["keep_ratio"] = json!(true);
    let error = LayoutConfig::parse(&value).unwrap_err().to_string();
    assert!(error.contains("aspect ratio"), "{error}");
}

#[test]
fn a_description_without_a_resize_is_refused() {
    let mut value = published();
    value["Preprocess"] = json!([{ "type": "Permute" }]);
    let error = LayoutConfig::parse(&value).unwrap_err().to_string();
    assert!(error.contains("resize"), "{error}");
}
