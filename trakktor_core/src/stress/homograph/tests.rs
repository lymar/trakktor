//! Finding homographs and writing a reading back — the parts that need no
//! encoder.

use super::*;

fn tables() -> Tables {
    Tables::fixture(
        &[],
        &[],
        &[
            ("замки", "з+амки", "замк+и"),
            ("все", "вс+е", "вс+ё"),
            ("село", "с+ело", "сел+о"),
        ],
    )
}

#[test]
fn only_the_table_words_are_candidates() {
    let tables = tables();
    let found = occurrences("Я открыл все замки в селе.", &tables);
    let words: Vec<&str> = found
        .iter()
        .map(|occurrence| occurrence.word.as_str())
        .collect();
    assert_eq!(words, ["все", "замки"]);
}

#[test]
fn a_word_the_caller_marked_is_not_a_candidate() {
    // The mark is part of the run, so the word no longer matches the table —
    // which is exactly how the caller's own choice survives this stage.
    let tables = tables();
    assert!(occurrences("Я открыл замк+и.", &tables).is_empty());
}

#[test]
fn case_and_punctuation_do_not_hide_a_candidate() {
    let tables = tables();
    let found = occurrences("«ЗАМКИ», село!", &tables);
    let words: Vec<&str> = found
        .iter()
        .map(|occurrence| occurrence.word.as_str())
        .collect();
    assert_eq!(words, ["ЗАМКИ", "село"]);
    // The byte range is the word alone, quotes excluded.
    assert_eq!(&"«ЗАМКИ», село!"[found[0].start..found[0].end], "ЗАМКИ");
}

#[test]
fn the_chosen_reading_keeps_the_words_own_case() {
    let tables = tables();
    let occurrences = occurrences("ЗАМКИ и замки", &tables);
    let (text, _) =
        apply("ЗАМКИ и замки", &occurrences, &[1, 0], &tables, true);
    assert_eq!(text, "ЗАМК+И и з+амки");
}

#[test]
fn a_yo_reading_falls_back_to_ye_when_yo_is_off() {
    let tables = tables();
    let occurrences = occurrences("Все дома.", &tables);
    assert_eq!(
        apply("Все дома.", &occurrences, &[1], &tables, true),
        ("Вс+ё дома.".to_owned(), 1)
    );
    assert_eq!(
        apply("Все дома.", &occurrences, &[1], &tables, false),
        ("Вс+е дома.".to_owned(), 0)
    );
}

#[test]
fn the_rest_of_the_sentence_is_copied_verbatim() {
    let tables = tables();
    let text = "  Все — замки, село…  ";
    let occurrences = occurrences(text, &tables);
    let (marked, _) = apply(text, &occurrences, &[0, 0, 0], &tables, true);
    assert_eq!(marked, "  Вс+е — з+амки, с+ело…  ");
}
