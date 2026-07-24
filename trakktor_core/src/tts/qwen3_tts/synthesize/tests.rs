//! Tests for the parts of the driver that do not need weights.

use super::*;

fn config() -> ModelConfig {
    ModelConfig::parse(include_str!("../config/testdata/custom_voice_0b6.json"))
        .expect("parses")
}

#[test]
fn the_control_tail_of_the_vocabulary_is_forbidden_except_the_stop_code() {
    let cfg = config();
    let forbidden = forbidden_codes(&cfg);

    // Audio codes live below the predictor's vocabulary and stay reachable.
    assert!(!forbidden.contains(&0));
    assert!(!forbidden.contains(&2047));
    // The control tail is closed off …
    assert!(forbidden.contains(&2048));
    assert!(forbidden.contains(&3071));
    assert!(forbidden.contains(&cfg.talker.codec_bos_id));
    assert!(forbidden.contains(&cfg.talker.codec_pad_id));
    // … except the code that ends the speech, which the model needs to stop.
    assert!(!forbidden.contains(&cfg.talker.codec_eos_token_id));
}
