//! Tests for the YAML-subset reader.
//!
//! The shapes here are the ones the published descriptions actually contain —
//! taken from their structure, not invented — plus the failures a reader that
//! does not claim to parse YAML has to admit to.

use serde_json::json;

use super::*;

fn read(text: &str) -> Value { parse(text).expect("a readable document") }

#[test]
fn a_mapping_of_scalars() {
    let value = read("Global:\n  model_name: PP-OCRv6_medium_det\n");
    assert_eq!(
        value,
        json!({"Global": {"model_name": "PP-OCRv6_medium_det"}})
    );
}

#[test]
fn numbers_booleans_and_nulls_keep_their_kinds() {
    let value = read(
        "thresh: 0.2\nmax_candidates: 3000\ndebug: false\nchecknull: null\n",
    );
    assert_eq!(
        value,
        json!({
            "thresh": 0.2,
            "max_candidates": 3000,
            "debug": false,
            "checknull": null,
        })
    );
}

/// `1./255.` is what the normalization scale is written as, and it is a string:
/// upstream passes it through `eval`. Reading it as a number would be reading
/// it wrong; reading it as `1.0` would be worse.
#[test]
fn a_scale_written_as_an_expression_stays_text() {
    let value = read("scale: 1./255.\n");
    assert_eq!(value, json!({"scale": "1./255."}));
    // And a bare trailing dot is not a number either.
    assert_eq!(read("a: 1.\n"), json!({"a": "1."}));
}

/// The one shape an indentation-driven reader is most likely to get wrong: a
/// block sequence written at its own key's indentation.
#[test]
fn a_sequence_may_sit_at_its_keys_indentation() {
    let value = read("PostProcess:\n  character_dict:\n  - a\n  - b\n");
    assert_eq!(
        value,
        json!({"PostProcess": {"character_dict": ["a", "b"]}})
    );
}

#[test]
fn a_sequence_indented_under_its_key_reads_the_same() {
    let value = read("mean:\n    - 0.485\n    - 0.456\n");
    assert_eq!(value, json!({"mean": [0.485, 0.456]}));
}

/// The dictionary quotes every character YAML would otherwise read as syntax,
/// and the quote itself is written as four of them.
#[test]
fn quoted_dictionary_entries_are_one_character_each() {
    let value = read(
        "character_dict:\n- '!'\n- '\"'\n- ''''\n- '#'\n- $\n- ':'\n- '- '\n",
    );
    assert_eq!(
        value,
        json!({"character_dict": ["!", "\"", "'", "#", "$", ":", "- "]})
    );
}

/// A key is only a key when a space or the line's end follows its colon, so a
/// scalar carrying a colon stays whole.
#[test]
fn a_colon_inside_a_scalar_does_not_open_a_key() {
    assert_eq!(read("- a:b\n"), json!(["a:b"]));
    assert!(read("url: http://example.invalid/x\n").is_object());
    assert_eq!(read("url: http://x\n"), json!({"url": "http://x"}));
}

/// A mapping whose first key shares the dash's line, which is how the
/// preprocessing steps are written.
#[test]
fn a_sequence_of_mappings() {
    let value = read(
        "transform_ops:\n- DecodeImage:\n    channel_first: false\n    \
         img_mode: BGR\n- DetLabelEncode: null\n- RecResizeImg:\n    \
         image_shape:\n    - 3\n    - 48\n    - 320\n",
    );
    assert_eq!(
        value,
        json!({"transform_ops": [
            {"DecodeImage": {"channel_first": false, "img_mode": "BGR"}},
            {"DetLabelEncode": null},
            {"RecResizeImg": {"image_shape": [3, 48, 320]}},
        ]})
    );
}

/// The dynamic-shape block is a sequence of sequences written dash-on-dash.
#[test]
fn a_sequence_of_sequences_written_dash_on_dash() {
    let value = read("x:\n- - 1\n  - 3\n  - 32\n- - 1\n  - 3\n  - 736\n");
    assert_eq!(value, json!({"x": [[1, 3, 32], [1, 3, 736]]}));
}

/// The one anchor these files carry, and the alias that reuses it.
#[test]
fn an_anchor_is_reused_by_its_alias() {
    let value = read(
        "Hpi:\n  paddle:\n    shapes: &id001\n      x:\n      - 1\n  \
         tensorrt:\n    shapes: *id001\n",
    );
    let shapes = json!({"x": [1]});
    assert_eq!(value["Hpi"]["paddle"]["shapes"], shapes);
    assert_eq!(value["Hpi"]["tensorrt"]["shapes"], shapes);
}

#[test]
fn an_alias_with_no_anchor_is_an_error() {
    let error = parse("a: *nothing\n").unwrap_err().to_string();
    assert!(error.contains("nothing"), "{error}");
}

#[test]
fn comments_and_blank_lines_carry_nothing() {
    let value = read("# a note\n\nGlobal:\n\n  # another\n  a: 1\n");
    assert_eq!(value, json!({"Global": {"a": 1}}));
}

/// A shape this reader does not claim to know fails with the line in it,
/// rather than being read as something else.
#[test]
fn an_unknown_shape_is_refused_with_its_line() {
    let error = parse("a: 1\nb: {inline: map}\n").unwrap_err().to_string();
    assert!(error.contains("inline"), "{error}");
    let error = parse("a: 1\n  b: 2\n").unwrap_err().to_string();
    assert!(error.contains("line 2"), "{error}");
}

#[test]
fn an_empty_document_is_null() {
    assert_eq!(parse("").unwrap(), Value::Null);
    assert_eq!(parse("# nothing but a note\n").unwrap(), Value::Null);
}

/// The whole of a published description, in miniature: every shape the real
/// files use, in the order they use them.
#[test]
fn a_published_description_in_miniature() {
    let value = read(
        "Global:\n  model_name: PP-OCRv6_medium_det\nHpi:\n  backend_configs:\n    paddle_infer:\n      trt_dynamic_shapes: &id001\n        x:\n        - - 1\n          - 3\n          - 32\n          - 32\n    tensorrt:\n      dynamic_shapes: *id001\nPostProcess:\n  box_thresh: 0.45\n  max_candidates: 3000\n  name: DBPostProcess\n  thresh: 0.2\n  unclip_ratio: 1.4\nPreProcess:\n  transform_ops:\n  - DecodeImage:\n      channel_first: false\n      img_mode: BGR\n  - NormalizeImage:\n      mean:\n      - 0.485\n      - 0.456\n      - 0.406\n      order: hwc\n      scale: 1./255.\n      std:\n      - 0.229\n      - 0.224\n      - 0.225\n  - ToCHWImage: null\n",
    );
    assert_eq!(value["Global"]["model_name"], "PP-OCRv6_medium_det");
    assert_eq!(value["PostProcess"]["thresh"], 0.2);
    assert_eq!(value["PostProcess"]["max_candidates"], 3000);
    assert_eq!(
        value["Hpi"]["backend_configs"]["tensorrt"]["dynamic_shapes"]["x"][0],
        json!([1, 3, 32, 32])
    );
    let ops = value["PreProcess"]["transform_ops"].as_array().unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[1]["NormalizeImage"]["scale"], "1./255.");
    assert_eq!(
        ops[1]["NormalizeImage"]["mean"],
        json!([0.485, 0.456, 0.406])
    );
    assert_eq!(ops[2]["ToCHWImage"], Value::Null);
}

/// The ideographic space is an entry of the dictionary in its own right,
/// written plain. Trimming Unicode whitespace instead of the ASCII space loses
/// it and shifts every class after it by one.
#[test]
fn an_ideographic_space_is_an_entry_not_an_empty_line() {
    let value = read("character_dict:\n- a\n- \u{3000}\n- b\n");
    assert_eq!(value, json!({"character_dict": ["a", "\u{3000}", "b"]}));
}

/// The real thing: the description of a published model, read whole.
///
/// The unit tests above cover the shapes one at a time; this one is the
/// guarantee that the set of shapes is complete, because a file this reader
/// cannot read is a model trakktor cannot run.
#[test]
#[ignore = "needs ~/.trakktor/ocr/paddle/PP-OCRv6_medium_rec"]
fn a_published_description_reads_whole() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/ocr/paddle/PP-OCRv6_medium_rec/inference.yml");
    let text = std::fs::read_to_string(&path).expect("the description");
    let value = parse(&text).expect("a readable description");

    assert_eq!(value["Global"]["model_name"], "PP-OCRv6_medium_rec");
    assert_eq!(value["PostProcess"]["name"], "CTCLabelDecode");
    let dictionary = value["PostProcess"]["character_dict"].as_array().unwrap();
    assert_eq!(dictionary.len(), 18_708);
    // The quoted entries are one character each, and the plain ideographic
    // space is one of the entries rather than a blank line.
    assert!(dictionary.iter().all(|c| c.is_string()));
    let chars: Vec<&str> =
        dictionary.iter().map(|c| c.as_str().unwrap()).collect();
    assert!(chars.contains(&"'"), "the quote is a class of its own");
    assert!(chars.contains(&"\u{3000}"), "so is the ideographic space");
    assert_eq!(chars.iter().filter(|c| c.is_empty()).count(), 0);

    let ops = value["PreProcess"]["transform_ops"].as_array().unwrap();
    let shape = ops
        .iter()
        .find_map(|op| op.get("RecResizeImg"))
        .and_then(|op| op.get("image_shape"))
        .expect("the exported shape");
    assert_eq!(*shape, json!([3, 48, 320]));
}
