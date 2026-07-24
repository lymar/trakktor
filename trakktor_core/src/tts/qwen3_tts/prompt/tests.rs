//! Layout tests. The prompt is where the two tracks are aligned, and an
//! off-by-one here would silently change what the model is asked to say, so the
//! positions are pinned explicitly.

use super::*;
use crate::tts::qwen3_tts::config::ModelConfig;

/// The trimmed 0.6B CustomVoice config the config tests also use.
fn config() -> ModelConfig {
    ModelConfig::parse(include_str!("../config/testdata/custom_voice_0b6.json"))
        .expect("parses")
}

const NEWLINE: u32 = 198;
const SERENA: u32 = 3066;
const RUSSIAN: u32 = 2069;

fn spec<'a>(text_ids: &'a [u32], language: Option<u32>) -> PromptSpec<'a> {
    PromptSpec {
        text_ids,
        newline_id: NEWLINE,
        speaker_id: Some(SERENA),
        language_id: language,
    }
}

#[test]
fn the_opening_role_is_text_only() {
    let cfg = config();
    let positions = build(&cfg, &spec(&[1, 2, 3], Some(RUSSIAN)));

    assert_eq!(
        &positions[..3],
        &[
            Position {
                text: Some(cfg.im_start_token_id),
                codec: None
            },
            Position {
                text: Some(cfg.assistant_token_id),
                codec: None
            },
            Position {
                text: Some(NEWLINE),
                codec: None
            },
        ]
    );
}

#[test]
fn a_named_language_is_announced_and_the_voice_follows_it() {
    let cfg = config();
    let talker = &cfg.talker;
    let positions = build(&cfg, &spec(&[10, 11], Some(RUSSIAN)));

    // think, think_bos, language, think_eos, speaker, pad — the last control
    // token (bos) is held back for the final position.
    let codec: Vec<Option<u32>> =
        positions[3..9].iter().map(|p| p.codec).collect();
    assert_eq!(
        codec,
        vec![
            Some(talker.codec_think_id),
            Some(talker.codec_think_bos_id),
            Some(RUSSIAN),
            Some(talker.codec_think_eos_id),
            Some(SERENA),
            Some(talker.codec_pad_id),
        ]
    );

    // The text track pads throughout the control block and announces the
    // spoken text on its last position — the one whose codec token is the pad
    // just before speech starts, not the position carrying the voice.
    let text: Vec<Option<u32>> =
        positions[3..9].iter().map(|p| p.text).collect();
    assert_eq!(
        text,
        vec![
            Some(cfg.tts_pad_token_id),
            Some(cfg.tts_pad_token_id),
            Some(cfg.tts_pad_token_id),
            Some(cfg.tts_pad_token_id),
            Some(cfg.tts_pad_token_id),
            Some(cfg.tts_bos_token_id),
        ]
    );
}

#[test]
fn without_a_language_the_control_block_is_one_shorter() {
    let cfg = config();
    let talker = &cfg.talker;
    let with = build(&cfg, &spec(&[10, 11], Some(RUSSIAN)));
    let without = build(&cfg, &spec(&[10, 11], None));

    assert_eq!(with.len(), without.len() + 1);
    assert_eq!(without[3].codec, Some(talker.codec_nothink_id));
    assert_eq!(without[4].codec, Some(talker.codec_think_bos_id));
    assert_eq!(without[5].codec, Some(talker.codec_think_eos_id));
    // The voice still follows the block.
    assert_eq!(without[6].codec, Some(SERENA));
}

#[test]
fn the_spoken_text_pads_the_codec_track_and_ends_with_the_start_token() {
    let cfg = config();
    let talker = &cfg.talker;
    let body = [111u32, 222, 333];
    let positions = build(&cfg, &spec(&body, Some(RUSSIAN)));

    // 3 role + 6 control + body + end-of-text + start.
    assert_eq!(positions.len(), 3 + 6 + body.len() + 1 + 1);

    let spoken = &positions[9..9 + body.len()];
    for (position, &id) in spoken.iter().zip(&body) {
        assert_eq!(position.text, Some(id));
        // Nothing has been uttered yet, so the codec track pads throughout.
        assert_eq!(position.codec, Some(talker.codec_pad_id));
    }

    let end_of_text = positions[9 + body.len()];
    assert_eq!(end_of_text.text, Some(cfg.tts_eos_token_id));
    assert_eq!(end_of_text.codec, Some(talker.codec_pad_id));

    // Generation continues from the final position, which starts the speech.
    let last = *positions.last().expect("non-empty");
    assert_eq!(last.text, Some(cfg.tts_pad_token_id));
    assert_eq!(last.codec, Some(talker.codec_bos_id));
}

#[test]
fn every_position_after_the_role_drives_both_tracks() {
    let cfg = config();
    let positions = build(&cfg, &spec(&[7, 8, 9], Some(RUSSIAN)));

    for (index, position) in positions.iter().enumerate().skip(3) {
        assert!(position.text.is_some(), "position {index} has no text");
        assert!(position.codec.is_some(), "position {index} has no codec");
    }
}
