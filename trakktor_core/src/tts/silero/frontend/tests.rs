//! Tests for the text frontend, quirks of the reference included.

use std::collections::HashMap;

use super::prepare;
use crate::tts::silero::{error::SileroError, tables::Tables};

/// The base models' alphabet, cut down to what these tests need. Its `symbols`
/// string starts at `|`, so the slice the reference takes drops `|!'` — which
/// is why `!` never reaches those models.
fn base() -> Tables {
    let symbols: Vec<String> = "_~|!'+,-.:;? \
                                абвгдеёжзийклмнопрстуфхцчшщъыьэюя—…"
        .chars()
        .map(|c| c.to_string())
        .collect();
    Tables {
        alphabet: "|!'+,-.:;? абвгдеёжзийклмнопрстуфхцчшщъыьэюя—…".to_owned(),
        letters: "абвгдеёжзийклмнопрстуфхцчшщъыьэюя".to_owned(),
        sos: "|".to_owned(),
        eos: "~".to_owned(),
        speakers: vec!["ru_one".to_owned()],
        translit: HashMap::new(),
        symbols,
    }
}

/// What the model will actually say, for a text.
fn spoken(text: &str) -> String {
    super::clean(&super::join_lines(text.trim()), &base(), None).spoken
}

#[test]
fn the_text_is_opened_and_closed_by_the_models_own_symbols() {
    let prepared =
        prepare("да", &base(), None, 1.0, 1.0, false).expect("should prepare");
    assert_eq!(prepared.ids.first(), Some(&2));
    assert_eq!(prepared.ids.last(), Some(&1));
    assert_eq!(prepared.ids.len(), "да".chars().count() + 2);
    // The per-symbol knobs are as long as the sequence, which is what the
    // model's own assertion checks.
    assert_eq!(prepared.rate.len(), prepared.ids.len());
    assert_eq!(prepared.pitch.len(), prepared.ids.len());
}

#[test]
fn stress_marks_are_ordinary_symbols() {
    let prepared = prepare("з+амок", &base(), None, 1.0, 1.0, false)
        .expect("should prepare");
    assert!(prepared.ids.contains(&5), "{:?}", prepared.ids);
    assert_eq!(spoken("з+амок"), "з+амок");
}

#[test]
fn the_text_is_lowercased_and_its_spacing_collapsed() {
    assert_eq!(spoken("  Да   ЛАДНО  "), "да ладно");
}

#[test]
fn a_line_break_ends_a_sentence_unless_a_clause_mark_already_did() {
    assert_eq!(spoken("раз\nдва"), "раз. два");
    assert_eq!(spoken("раз,\nдва"), "раз, два");
}

#[test]
fn latin_disappears_because_the_alphabet_has_none() {
    assert_eq!(spoken("да, ok, нет"), "да, , нет");
}

#[test]
fn an_em_dash_disappears_from_a_base_model() {
    // It is rewritten as an en dash, which the base alphabet does not hold —
    // upstream's own behaviour, and audible as a lost pause.
    assert_eq!(spoken("да — нет"), "да нет");
}

#[test]
fn an_exclamation_mark_disappears_from_a_base_model() {
    // `!` falls inside the three symbols the reference's slice drops.
    assert_eq!(spoken("да!"), "да");
}

#[test]
fn a_text_with_nothing_readable_is_refused() {
    let error = prepare("hello world", &base(), None, 1.0, 1.0, false)
        .expect_err("should refuse");
    assert!(matches!(error, SileroError::TextEmpty), "{error}");
}

#[test]
fn a_script_the_model_has_no_letters_for_is_transliterated() {
    let mut tables = base();
    tables.translit.insert(
        "kat".to_owned(),
        HashMap::from([
            ("ა".to_owned(), "а".to_owned()),
            ("დ".to_owned(), "д".to_owned()),
        ]),
    );
    assert_eq!(super::clean("და", &tables, Some("kat")).spoken, "да");
    // Cyrillic that is already in the alphabet is left alone.
    assert_eq!(super::clean("да", &tables, Some("kat")).spoken, "да");
}

#[test]
fn what_the_cleaning_removed_is_counted() {
    let prepared = prepare("да, ok!", &base(), None, 1.0, 1.0, false)
        .expect("should prepare");
    // Two Latin letters and the exclamation mark this alphabet cannot spell.
    assert_eq!(prepared.dropped, 3);
}

#[test]
fn the_intonation_ids_appear_only_when_asked_for() {
    let without =
        prepare("да", &base(), None, 1.0, 1.0, false).expect("should prepare");
    assert!(without.types.is_none());
    let with =
        prepare("да", &base(), None, 1.0, 1.0, true).expect("should prepare");
    assert_eq!(with.types.map(|types| types.len()), Some(with.ids.len()));
}
