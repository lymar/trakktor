//! Parsing tests over configs shaped like the published checkpoints.

use super::*;

/// A `config.json` with the geometry and tables of the 0.6B CustomVoice
/// checkpoint (trimmed to the fields the pipeline reads). Shared with the
/// prompt tests, which need the same token tables.
const CUSTOM_VOICE_0B6: &str = include_str!("testdata/custom_voice_0b6.json");

/// A `speech_tokenizer/config.json` shaped like the published 12 Hz codec.
const CODEC: &str = r#"{
  "model_type": "qwen3_tts_tokenizer_12hz",
  "input_sample_rate": 24000,
  "output_sample_rate": 24000,
  "decode_upsample_rate": 1920,
  "encode_downsample_rate": 1920,
  "decoder_config": {
    "latent_dim": 1024,
    "codebook_dim": 512,
    "codebook_size": 2048,
    "decoder_dim": 1536,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "layer_scale_initial_scale": 0.01,
    "head_dim": 64,
    "num_attention_heads": 16,
    "num_hidden_layers": 8,
    "num_key_value_heads": 16,
    "num_quantizers": 16,
    "num_semantic_quantizers": 1,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "semantic_codebook_size": 4096,
    "sliding_window": 72,
    "upsample_rates": [8, 5, 4, 3],
    "upsampling_ratios": [2, 2]
  }
}"#;

#[test]
fn parses_the_custom_voice_geometry() {
    let config = ModelConfig::parse(CUSTOM_VOICE_0B6).expect("parses");

    assert_eq!(config.model_type, ModelType::CustomVoice);
    assert_eq!(config.model_size, "0b6");
    assert_eq!(config.tts_pad_token_id, 151671);

    let talker = &config.talker;
    assert_eq!(talker.hidden_size, 1024);
    assert_eq!(talker.num_hidden_layers, 28);
    // 16 query heads over 8 key/value heads — grouped-query attention.
    assert_eq!(talker.num_key_value_groups(), 2);
    assert_eq!(talker.num_code_groups, 16);
    assert_eq!(talker.mrope_section, vec![24, 20, 20]);
    assert!(talker.mrope_interleaved);
    // The rotary sections split half the head dimension.
    assert_eq!(
        talker.mrope_section.iter().sum::<usize>(),
        talker.head_dim / 2
    );

    // The code predictor keeps its own width, independent of the backbone.
    assert_eq!(talker.code_predictor.hidden_size, 1024);
    assert_eq!(talker.code_predictor.num_hidden_layers, 5);
    assert_eq!(talker.code_predictor.vocab_size, 2048);
}

#[test]
fn resolves_speakers_case_insensitively() {
    let config = ModelConfig::parse(CUSTOM_VOICE_0B6).expect("parses");

    assert_eq!(config.talker.speaker_id("serena").unwrap(), 3066);
    assert_eq!(config.talker.speaker_id("SeReNa").unwrap(), 3066);
    assert_eq!(config.talker.speakers(), vec!["dylan", "eric", "serena"]);

    let err = config.talker.speaker_id("nobody").unwrap_err();
    assert!(matches!(err, Qwen3TtsError::UnsupportedVoice(_)));
}

#[test]
fn resolves_languages_and_leaves_auto_to_the_model() {
    let config = ModelConfig::parse(CUSTOM_VOICE_0B6).expect("parses");

    assert_eq!(config.language_id(Some("russian")).unwrap(), Some(2069));
    assert_eq!(config.language_id(Some("RUSSIAN")).unwrap(), Some(2069));
    // Both an absent language and an explicit `auto` leave the choice open.
    assert_eq!(config.language_id(None).unwrap(), None);
    assert_eq!(config.language_id(Some("auto")).unwrap(), None);

    let err = config.language_id(Some("klingon")).unwrap_err();
    assert!(matches!(err, Qwen3TtsError::UnsupportedLanguage(_)));

    // Dialects are reached through the speakers that imply them, so they are
    // not offered as target languages.
    assert!(!config.talker.languages().contains(&"beijing_dialect"));
    assert_eq!(
        config.talker.spk_is_dialect.get("eric"),
        Some(&Some("sichuan_dialect".to_owned()))
    );
    assert_eq!(config.talker.spk_is_dialect.get("serena"), Some(&None));
}

#[test]
fn a_dialect_voice_overrides_the_open_and_chinese_language_choices() {
    let config = ModelConfig::parse(CUSTOM_VOICE_0B6).expect("parses");
    let sichuan = Some(2062);

    // With the language left to the model — or set to the dialect's parent —
    // a dialect voice speaks its dialect, as the reference resolves it.
    assert_eq!(config.language_id_for_voice(None, "eric").unwrap(), sichuan);
    assert_eq!(
        config.language_id_for_voice(Some("auto"), "eric").unwrap(),
        sichuan
    );
    assert_eq!(
        config
            .language_id_for_voice(Some("chinese"), "Eric")
            .unwrap(),
        sichuan
    );
    assert_eq!(
        config.language_id_for_voice(None, "dylan").unwrap(),
        Some(2074)
    );

    // An explicit other language wins over the voice's dialect.
    assert_eq!(
        config
            .language_id_for_voice(Some("russian"), "eric")
            .unwrap(),
        Some(2069)
    );

    // A voice without a dialect changes nothing.
    assert_eq!(config.language_id_for_voice(None, "serena").unwrap(), None);
    assert_eq!(
        config
            .language_id_for_voice(Some("chinese"), "serena")
            .unwrap(),
        Some(2055)
    );

    // An unknown language still fails the same way.
    assert!(matches!(
        config.language_id_for_voice(Some("klingon"), "eric"),
        Err(Qwen3TtsError::UnsupportedLanguage(_))
    ));
}

#[test]
fn parses_the_codec_geometry() {
    let codec = CodecConfig::parse(CODEC).expect("parses");

    assert_eq!(codec.output_sample_rate, 24000);
    assert_eq!(codec.num_quantizers, 16);
    assert_eq!(codec.num_semantic_quantizers, 1);
    assert_eq!(codec.codebook_size, 2048);
    // The split quantizer runs at half the codebook width.
    assert_eq!(codec.quantizer_dim, 256);
    assert_eq!(codec.sliding_window, 72);
    // Every upsampling stage together must account for the frame rate.
    assert_eq!(codec.total_upsample(), codec.decode_upsample_rate);
    assert_eq!(codec.total_upsample(), 1920);
}

#[test]
fn generation_defaults_fall_back_to_the_reference_values() {
    let parsed = GenerationDefaults::parse(
        r#"{ "do_sample": true, "top_k": 50, "temperature": 0.9,
              "repetition_penalty": 1.05, "subtalker_dosample": true,
              "subtalker_top_k": 50, "subtalker_temperature": 0.9 }"#,
    )
    .expect("parses");
    assert!(parsed.do_sample);
    assert_eq!(parsed.top_k, 50);
    assert!((parsed.temperature - 0.9).abs() < f32::EPSILON);
    assert!((parsed.repetition_penalty - 1.05).abs() < f32::EPSILON);
    assert!(parsed.predictor_do_sample);

    // Anything the file omits keeps the reference default.
    let sparse = GenerationDefaults::parse("{}").expect("parses");
    assert_eq!(sparse.top_k, GenerationDefaults::default().top_k);
    assert_eq!(
        sparse.max_new_tokens,
        GenerationDefaults::default().max_new_tokens
    );
}

#[test]
fn rejects_a_config_missing_a_needed_field() {
    let err = ModelConfig::parse(r#"{ "tts_model_type": "custom_voice" }"#)
        .unwrap_err();
    assert!(matches!(err, Qwen3TtsError::InvalidModel(_)));
}
