//! Weight-extraction checks against real model files.
//!
//! The model files live outside the repo (`tmp/vosk/models/<name>/`, fetched
//! by `scripts/asr/vosk/fetch_models.sh` or a prior `trakktor asr vosk` run),
//! so these tests are `#[ignore]` and run on demand.

use super::load_dir;

fn model_dir(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tmp/vosk/models")
        .join(name)
}

#[test]
#[ignore = "needs tmp/vosk/models/small-streaming-ru"]
fn small_streaming_ru_geometry() {
    let w = load_dir(&model_dir("small-streaming-ru")).unwrap();
    let c = &w.config;
    assert_eq!(c.stacks.len(), 6);
    assert_eq!(
        c.stacks.iter().map(|s| s.num_layers).collect::<Vec<_>>(),
        [2, 2, 2, 2, 2, 2]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.encoder_dim).collect::<Vec<_>>(),
        [192, 256, 256, 256, 256, 256]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.downsample).collect::<Vec<_>>(),
        [1, 2, 4, 8, 4, 2]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.num_heads).collect::<Vec<_>>(),
        [4, 4, 4, 8, 4, 4]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.cnn_kernel).collect::<Vec<_>>(),
        [31, 31, 15, 15, 15, 31]
    );
    assert!(c.stacks.iter().all(|s| s.query_head_dim == 32));
    assert!(c.stacks.iter().all(|s| s.value_head_dim == 12));
    assert_eq!(c.pos_dim, 48);
    assert_eq!(c.feature_dim, 80);
    assert_eq!(c.encoder_out_dim, 256);
    assert_eq!(c.joiner_dim, 512);
    assert_eq!(c.vocab_size, 500);
    assert_eq!(c.context_size, 2);
    assert_eq!(c.decoder_dim, 512);
    let streaming = c.streaming.as_ref().expect("streaming model");
    assert_eq!(streaming.window_frames, 77);
    assert_eq!(streaming.shift_frames, 64);
    assert_eq!(streaming.left_context, [128, 64, 32, 16, 32, 64]);
    // The re-assembled causal-conv edge scales exist for every conv module.
    assert!(w.contains(
        "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.\
         chunkwise_conv_scale"
    ));
}

#[test]
#[ignore = "needs tmp/vosk/models/ru"]
fn ru_geometry() {
    let w = load_dir(&model_dir("ru")).unwrap();
    let c = &w.config;
    assert_eq!(c.stacks.len(), 6);
    assert_eq!(
        c.stacks.iter().map(|s| s.num_layers).collect::<Vec<_>>(),
        [2, 2, 3, 4, 3, 2]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.encoder_dim).collect::<Vec<_>>(),
        [192, 256, 384, 512, 384, 256]
    );
    assert_eq!(
        c.stacks.iter().map(|s| s.downsample).collect::<Vec<_>>(),
        [1, 2, 4, 8, 4, 2]
    );
    assert_eq!(c.encoder_out_dim, 512);
    assert_eq!(c.vocab_size, 500);
    assert!(c.streaming.is_none());
}
