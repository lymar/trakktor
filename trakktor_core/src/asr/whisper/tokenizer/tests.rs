use serde_json::Value;

use super::*;

/// Reference fixture produced by `scripts/asr/whisper/gen_tokenizer_golden.py`.
const GOLDEN: &str = include_str!("../testdata/tokenizer_golden.json");

fn ids(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as TokenId)
        .collect()
}

fn build(case: &Value) -> Tokenizer {
    let task = match case["task"].as_str() {
        None => None,
        Some("transcribe") => Some(Task::Transcribe),
        Some("translate") => Some(Task::Translate),
        Some(other) => panic!("unknown task in golden: {other}"),
    };
    Tokenizer::new(
        case["multilingual"].as_bool().unwrap(),
        case["num_languages"].as_u64().unwrap() as usize,
        case["language"].as_str(),
        task,
    )
    .expect("golden cases must construct")
}

#[test]
fn matches_reference_golden() {
    let golden: Value = serde_json::from_str(GOLDEN).unwrap();
    for case in golden["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let tok = build(case);

        let specials = &case["specials"];
        let id = |key: &str| specials[key].as_u64().unwrap() as TokenId;
        assert_eq!(tok.eot(), id("eot"), "{name}: eot");
        assert_eq!(tok.sot(), id("sot"), "{name}: sot");
        assert_eq!(tok.translate(), id("translate"), "{name}: translate");
        assert_eq!(tok.transcribe(), id("transcribe"), "{name}: transcribe");
        assert_eq!(tok.sot_lm(), id("sot_lm"), "{name}: sot_lm");
        assert_eq!(tok.sot_prev(), id("sot_prev"), "{name}: sot_prev");
        assert_eq!(tok.no_speech(), id("no_speech"), "{name}: no_speech");
        assert_eq!(
            tok.no_timestamps(),
            id("no_timestamps"),
            "{name}: no_timestamps"
        );
        assert_eq!(
            tok.timestamp_begin(),
            id("timestamp_begin"),
            "{name}: timestamp_begin"
        );
        assert_eq!(
            tok.n_vocab() as u64,
            case["n_vocab"].as_u64().unwrap(),
            "{name}: n_vocab"
        );

        assert_eq!(
            tok.language(),
            case["normalized_language"].as_str(),
            "{name}: language"
        );
        assert_eq!(
            tok.sot_sequence(),
            ids(&case["sot_sequence"]),
            "{name}: sot_sequence"
        );
        assert_eq!(
            tok.sot_sequence_including_notimestamps(),
            ids(&case["sot_sequence_including_notimestamps"]),
            "{name}: sot_sequence_including_notimestamps"
        );
        assert_eq!(
            tok.all_language_tokens(),
            ids(&case["all_language_tokens_sorted"]),
            "{name}: all_language_tokens"
        );
        assert_eq!(
            tok.non_speech_tokens(),
            ids(&case["non_speech_tokens"]),
            "{name}: non_speech_tokens"
        );

        for (text, expected) in case["encode"].as_object().unwrap() {
            assert_eq!(
                tok.encode(text),
                ids(expected),
                "{name}: encode {text:?}"
            );
        }

        let stream = ids(&case["decode_stream_ids"]);
        assert_eq!(
            tok.decode(&stream),
            case["decode_stream_plain"].as_str().unwrap(),
            "{name}: decode"
        );
        assert_eq!(
            tok.decode_with_timestamps(&stream),
            case["decode_stream_with_timestamps"].as_str().unwrap(),
            "{name}: decode_with_timestamps"
        );

        let split = &case["split"];
        let (words, word_tokens) =
            tok.split_to_word_tokens(&ids(&split["tokens"]));
        let expected_words: Vec<&str> = split["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        assert_eq!(words, expected_words, "{name}: split words");
        let expected_tokens: Vec<Vec<TokenId>> = split["word_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(ids)
            .collect();
        assert_eq!(word_tokens, expected_tokens, "{name}: split word tokens");
    }
}

#[test]
fn special_ids_match_the_published_layout() {
    // eot, sot, translate, transcribe, sot_lm, sot_prev, no_speech,
    // no_timestamps, timestamp_begin — per vocabulary and language count.
    let rows: [(bool, usize, [TokenId; 9], usize); 3] = [
        (
            false,
            99,
            [
                50256, 50257, 50357, 50358, 50359, 50360, 50361, 50362, 50363,
            ],
            51864,
        ),
        (
            true,
            99,
            [
                50257, 50258, 50358, 50359, 50360, 50361, 50362, 50363, 50364,
            ],
            51865,
        ),
        (
            true,
            100,
            [
                50257, 50258, 50359, 50360, 50361, 50362, 50363, 50364, 50365,
            ],
            51866,
        ),
    ];
    for (multilingual, num_languages, expected, n_vocab) in rows {
        let tok = Tokenizer::new(multilingual, num_languages, None, None)
            .expect("must construct");
        let actual = [
            tok.eot(),
            tok.sot(),
            tok.translate(),
            tok.transcribe(),
            tok.sot_lm(),
            tok.sot_prev(),
            tok.no_speech(),
            tok.no_timestamps(),
            tok.timestamp_begin(),
        ];
        assert_eq!(actual, expected, "ids for {multilingual}/{num_languages}");
        assert_eq!(tok.n_vocab(), n_vocab);
    }
}

#[test]
fn space_encodes_to_220_in_both_vocabularies() {
    for multilingual in [false, true] {
        let tok = Tokenizer::new(multilingual, 99, None, None).unwrap();
        assert_eq!(tok.encode(" "), vec![220], "multilingual={multilingual}");
    }
}

#[test]
fn timestamps_render_with_two_decimals() {
    let tok = Tokenizer::new(true, 99, None, None).unwrap();
    let ts = tok.timestamp_begin();
    assert_eq!(tok.decode_with_timestamps(&[ts]), "<|0.00|>");
    assert_eq!(tok.decode_with_timestamps(&[ts + 54]), "<|1.08|>");
    assert_eq!(tok.decode_with_timestamps(&[ts + 1500]), "<|30.00|>");
    // `decode` drops timestamps entirely.
    assert_eq!(tok.decode(&[ts, ts + 54]), "");
}

#[test]
fn unknown_and_out_of_set_languages_fail() {
    let err = Tokenizer::new(true, 99, Some("klingon"), None).unwrap_err();
    assert!(matches!(err, WhisperError::UnsupportedLanguage(_)));

    // Cantonese exists only in the 100-language vocabularies.
    let err = Tokenizer::new(true, 99, Some("yue"), None).unwrap_err();
    assert!(matches!(err, WhisperError::UnsupportedLanguage(_)));
    let tok = Tokenizer::new(true, 100, Some("yue"), None).unwrap();
    assert_eq!(tok.sot_sequence()[1], tok.sot() + 1 + 99);
}

#[test]
fn language_names_and_aliases_normalize() {
    let tok = Tokenizer::new(true, 99, Some("RUSSIAN"), None).unwrap();
    assert_eq!(tok.language(), Some("ru"));
    let tok = Tokenizer::new(true, 99, Some("moldavian"), None).unwrap();
    assert_eq!(tok.language(), Some("ro"));
}
