//! Parity against the Python reference, layer by layer. `#[ignore]` — they need
//! the converted model in `~/.trakktor/text/stress/silero-ru/` and the
//! reference dumps in `tmp/stress/golden/`.
//!
//! The dumps are taken by running the published package itself
//! (`torch.package` → `load_pickle`), so these compare the port against the
//! thing it ports, not against a re-implementation of it.

use std::path::PathBuf;

use super::*;
use crate::stress::{
    Dictionary, StressOptions, Stressor, accentor::bag, homograph,
    tables::Tables, text, tokenizer::WordPieces,
};

/// Where the reference dumps live.
pub(crate) fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tmp/stress/golden")
}

/// Where the converted model lives.
pub(crate) fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".trakktor/text/stress/silero-ru")
}

pub(crate) fn golden(name: &str) -> serde_json::Value {
    let path = golden_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("golden json")
}

/// The largest absolute difference between two rows.
pub(crate) fn worst(ours: &[f32], theirs: &[f32]) -> f32 {
    assert_eq!(ours.len(), theirs.len(), "lengths differ");
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn runtime() -> StressRuntime {
    StressRuntime::load_cpu(&model_dir(), Precision::F32).expect("load")
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn the_bags_match_the_reference() {
    // The n-gram slicing is pure string work, but it decides what the network
    // ever sees, so it is compared row for row.
    let tables = Tables::load(&model_dir()).expect("tables");
    for probe in golden("bags.json").as_array().expect("array") {
        let word = probe["word"].as_str().expect("word");
        let theirs: Vec<u32> = probe["indices"]
            .as_array()
            .expect("indices")
            .iter()
            .map(|value| value.as_u64().expect("index") as u32)
            .collect();
        assert_eq!(bag(word, &tables), theirs, "bag of {word:?}");
    }
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn the_accentor_heads_match_the_reference() {
    let tables = Tables::load(&model_dir()).expect("tables");
    let runtime = runtime();
    let probes = golden("accentor.json");
    let probes = probes.as_array().expect("array");
    let words: Vec<&str> = probes
        .iter()
        .map(|probe| probe["word"].as_str().expect("word"))
        .collect();
    let bags: Vec<Vec<u32>> =
        words.iter().map(|word| bag(word, &tables)).collect();
    let scored = runtime.accentor(&bags).expect("accentor");

    let row = |probe: &serde_json::Value, head: &str| -> Vec<f32> {
        probe[head]
            .as_array()
            .expect(head)
            .iter()
            .map(|value| value.as_f64().expect("probability") as f32)
            .collect()
    };
    let (mut deviation, mut mismatches) = (0.0f32, 0usize);
    for (probe, scores) in probes.iter().zip(&scored) {
        let stress = row(probe, "stress");
        let yo = row(probe, "yo");
        // What the rules actually consume: the argmax and its probability.
        let best = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::MIN), |best, (i, v)| {
                    if *v > best.1 { (i, *v) } else { best }
                })
        };
        let (stress_id, stress_prob) = best(&stress);
        let (yo_id, yo_prob) = best(&yo);
        if scores.stress_id != stress_id || scores.yo_id != yo_id {
            mismatches += 1;
        }
        deviation = deviation
            .max(worst(&[scores.stress_prob], &[stress_prob]))
            .max(worst(&[scores.yo_prob], &[yo_prob]));
    }
    println!(
        "accentor: {} words, worst probability delta {deviation:e}",
        probes.len()
    );
    assert_eq!(mismatches, 0, "argmax disagreements");
    assert!(deviation < 1e-5, "worst {deviation:e}");
}

#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn the_homograph_solver_matches_the_reference() {
    let tokenizer = WordPieces::load(&model_dir()).expect("tokenizer");
    let tables = Tables::load(&model_dir()).expect("tables");
    let runtime = runtime();
    let probes = golden("homographs.json");
    let probes = probes.as_array().expect("array");

    let mut checked = 0;
    for probe in probes {
        let sentence = probe["sentence"].as_str().expect("sentence");
        let word = probe["word"].as_str().expect("word");
        let found = homograph::occurrences(sentence, &tables);
        let occurrence = found
            .iter()
            .find(|occurrence| occurrence.word == word)
            .unwrap_or_else(|| panic!("{word:?} not found in {sentence:?}"));
        let context = homograph::context(sentence, occurrence, &tokenizer)
            .expect("context");

        // The tokenization has to be identical, or the encoder is reading a
        // different sentence.
        let theirs: Vec<u32> = probe["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();
        assert_eq!(context.ids, theirs, "token ids for {word:?}");
        assert_eq!(
            context.start,
            probe["homo_start"].as_u64().expect("start") as usize
        );
        assert_eq!(
            context.end,
            probe["homo_end"].as_u64().expect("end") as usize
        );

        // And so does the decision the logit encodes.
        let picks = runtime
            .homographs(std::slice::from_ref(&context), tokenizer.pad)
            .expect("homographs");
        let logit = probe["logit"].as_f64().expect("logit") as f32;
        assert_eq!(
            picks[0],
            usize::from(logit > 0.0),
            "variant for {word:?} in {sentence:?} (reference logit {logit})"
        );
        checked += 1;
    }
    println!("homographs: {checked} contexts matched");
    assert!(checked > 0);
}

/// The whole pipeline, on the texts the reference was dumped over.
#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn whole_texts_match_the_reference() {
    let stressor =
        Stressor::load(&model_dir(), Box::new(runtime())).expect("stressor");
    let dictionary = Dictionary::default();
    for probe in golden("texts.json").as_array().expect("array") {
        let name = probe["name"].as_str().expect("name");
        let input = probe["input"].as_str().expect("input");

        // The sentence split is ours, so it is checked against the dump too.
        let theirs: Vec<&str> = probe["segments"]
            .as_array()
            .expect("segments")
            .iter()
            .map(|value| value.as_str().expect("segment"))
            .collect();
        assert_eq!(text::segments(input), theirs, "segments of {name}");

        let marked = stressor
            .mark(input, &dictionary, &StressOptions::default())
            .expect("mark");
        assert_eq!(
            marked.text,
            probe["default"].as_str().expect("default"),
            "{name}"
        );
        println!("{name}: {:?}", marked.text);
    }
}

/// A page of real prose, marked end to end.
#[test]
#[ignore = "needs the converted model and tmp/stress/golden"]
fn a_page_of_prose_matches_the_reference() {
    let stressor =
        Stressor::load(&model_dir(), Box::new(runtime())).expect("stressor");
    let source = std::fs::read_to_string(golden_dir().join("prose.txt"))
        .expect("source");
    let theirs =
        std::fs::read_to_string(golden_dir().join("prose.stressed.txt"))
            .expect("golden");

    let marked = stressor
        .mark(&source, &Dictionary::default(), &StressOptions::default())
        .expect("mark");
    println!(
        "{} words, {} stressed, {} `ё` restored, {} homographs, {} left \
         unmarked",
        marked.stats.words,
        marked.stats.stressed,
        marked.stats.yo_restored,
        marked.stats.homographs,
        marked.unstressed.len()
    );
    println!("unmarked: {:?}", marked.unstressed);
    assert_eq!(marked.text, theirs);
}

/// Turning `ё` off has to leave the letters alone — that is the whole promise
/// of the flag, and the reference's own exception table would break it.
#[test]
#[ignore = "needs the converted model"]
fn without_yo_only_marks_are_added() {
    let stressor =
        Stressor::load(&model_dir(), Box::new(runtime())).expect("stressor");
    let source = std::fs::read_to_string(golden_dir().join("prose.txt"))
        .expect("source");
    let options = StressOptions {
        restore_yo: false,
        ..StressOptions::default()
    };
    let marked = stressor
        .mark(&source, &Dictionary::default(), &options)
        .expect("mark");
    let stripped: String =
        marked.text.chars().filter(|c| *c != text::STRESS).collect();
    assert_eq!(stripped, source);
    assert_eq!(marked.stats.yo_restored, 0);
}

/// The dictionary outranks the model, and the model leaves its words alone.
#[test]
#[ignore = "needs the converted model"]
fn the_dictionary_wins_over_the_model() {
    let stressor =
        Stressor::load(&model_dir(), Box::new(runtime())).expect("stressor");
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    std::io::Write::write_all(&mut file, "з+амки\nКорол+ёв\n".as_bytes())
        .expect("write");
    let dictionary =
        Dictionary::load(&[file.path().to_path_buf()]).expect("dictionary");

    let marked = stressor
        .mark(
            "Я открыл все замки. Королев тут.",
            &dictionary,
            &StressOptions::default(),
        )
        .expect("mark");
    assert!(marked.text.contains("з+амки"), "{}", marked.text);
    assert!(marked.text.contains("Корол+ёв"), "{}", marked.text);
    assert_eq!(marked.stats.from_dictionary, 2);
}
