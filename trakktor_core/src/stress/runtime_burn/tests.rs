//! The same parity checks as the candle runtime, on burn. `#[ignore]` — they
//! need the converted model in `~/.trakktor/text/stress/silero-ru/` and the
//! reference dumps in `tmp/stress/golden/`.
//!
//! Two things are being asked here: that burn agrees with the reference, and
//! that it agrees with candle — the second is what makes `--runtime` a free
//! choice rather than a quality decision.

use super::*;
use crate::stress::{
    Dictionary, StressOptions, Stressor,
    accentor::bag,
    homograph,
    runtime::{
        StressRuntime,
        tests::{golden, golden_dir, model_dir, worst},
    },
    tables::Tables,
    tokenizer::WordPieces,
};

fn runtime() -> StressBurnRuntime {
    StressBurnRuntime::load_cpu(&model_dir(), Precision::F32).expect("load")
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn the_accentor_heads_match_the_reference() {
    let tables = Tables::load(&model_dir()).expect("tables");
    let runtime = runtime();
    let probes = golden("accentor.json");
    let probes = probes.as_array().expect("array");
    let bags: Vec<Vec<u32>> = probes
        .iter()
        .map(|probe| bag(probe["word"].as_str().expect("word"), &tables))
        .collect();
    let scored = runtime.accentor(&bags).expect("accentor");

    let best =
        |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::MIN), |best, (i, v)| {
                    if *v > best.1 { (i, *v) } else { best }
                })
        };
    let row = |probe: &serde_json::Value, head: &str| -> Vec<f32> {
        probe[head]
            .as_array()
            .expect(head)
            .iter()
            .map(|value| value.as_f64().expect("probability") as f32)
            .collect()
    };
    let mut deviation = 0.0f32;
    for (probe, scores) in probes.iter().zip(&scored) {
        let (stress_id, stress_prob) = best(&row(probe, "stress"));
        let (yo_id, yo_prob) = best(&row(probe, "yo"));
        assert_eq!(scores.stress_id, stress_id, "{}", probe["word"]);
        assert_eq!(scores.yo_id, yo_id, "{}", probe["word"]);
        deviation = deviation
            .max(worst(&[scores.stress_prob], &[stress_prob]))
            .max(worst(&[scores.yo_prob], &[yo_prob]));
    }
    println!("burn accentor: worst probability delta {deviation:e}");
    assert!(deviation < 1e-5, "worst {deviation:e}");
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn the_homograph_solver_matches_the_reference() {
    let tokenizer = WordPieces::load(&model_dir()).expect("tokenizer");
    let tables = Tables::load(&model_dir()).expect("tables");
    let runtime = runtime();
    let probes = golden("homographs.json");
    let mut checked = 0;
    for probe in probes.as_array().expect("array") {
        let sentence = probe["sentence"].as_str().expect("sentence");
        let word = probe["word"].as_str().expect("word");
        let found = homograph::occurrences(sentence, &tables);
        let occurrence = found
            .iter()
            .find(|occurrence| occurrence.word == word)
            .unwrap_or_else(|| panic!("{word:?} not found"));
        let context = homograph::context(sentence, occurrence, &tokenizer)
            .expect("context");
        let picks = runtime
            .homographs(std::slice::from_ref(&context), tokenizer.pad)
            .expect("homographs");
        let logit = probe["logit"].as_f64().expect("logit") as f32;
        assert_eq!(picks[0], usize::from(logit > 0.0), "{word:?}");
        checked += 1;
    }
    println!("burn homographs: {checked} contexts matched");
    assert!(checked > 0);
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn whole_texts_match_the_reference() {
    let stressor =
        Stressor::load(&model_dir(), Box::new(runtime())).expect("stressor");
    let dictionary = Dictionary::default();
    for probe in golden("texts.json").as_array().expect("array") {
        let marked = stressor
            .mark(
                probe["input"].as_str().expect("input"),
                &dictionary,
                &StressOptions::default(),
            )
            .expect("mark");
        assert_eq!(
            marked.text,
            probe["default"].as_str().expect("default"),
            "{}",
            probe["name"]
        );
    }
}

/// A page of real prose on burn, against the reference and against candle.
#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn a_page_of_prose_matches_the_reference_and_candle() {
    let source = std::fs::read_to_string(golden_dir().join("prose.txt"))
        .expect("source");
    let theirs =
        std::fs::read_to_string(golden_dir().join("prose.stressed.txt"))
            .expect("golden");

    let burn = Stressor::load(&model_dir(), Box::new(runtime()))
        .expect("stressor")
        .mark(&source, &Dictionary::default(), &StressOptions::default())
        .expect("mark");
    assert_eq!(burn.text, theirs, "burn against the reference");

    let candle = Stressor::load(
        &model_dir(),
        Box::new(
            StressRuntime::load_cpu(&model_dir(), Precision::F32)
                .expect("candle"),
        ),
    )
    .expect("stressor")
    .mark(&source, &Dictionary::default(), &StressOptions::default())
    .expect("mark");
    assert_eq!(burn.text, candle.text, "burn against candle");
    assert_eq!(burn.unstressed, candle.unstressed);
    println!("burn == candle == reference on {} words", burn.stats.words);
}

/// The burn CPU backend has no half precision, and says so rather than
/// quietly computing something else.
#[test]
fn half_precision_on_the_cpu_is_refused() {
    let error = StressBurnRuntime::load_cpu(
        std::path::Path::new("/nonexistent"),
        Precision::F16,
    );
    assert!(matches!(error, Err(StressError::InvalidOptions(_))));
}
