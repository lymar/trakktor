//! The user dictionary, tested without a model: parsing, the key it folds a
//! word to, and how a spelling is written back into real text.

use std::io::Write;

use super::*;

/// Writes a dictionary to a temporary file and loads it.
fn load(contents: &str) -> Result<Dictionary, StressError> {
    let mut file = tempfile::NamedTempFile::new().expect("temp file");
    file.write_all(contents.as_bytes()).expect("write");
    Dictionary::load(&[file.path().to_path_buf()])
}

#[test]
fn one_entry_covers_the_word_however_its_yo_is_written() {
    let dictionary = load("Корол+ёв\n").expect("load");
    // Written with `е`, the word is rewritten with the `ё` and the mark.
    assert_eq!(dictionary.apply("Королев тут.").0, "Корол+ёв тут.");
    // Written with `ё`, the caller has already said something, so it stands.
    assert_eq!(dictionary.apply("Королёв тут.").0, "Королёв тут.");
}

#[test]
fn the_case_of_the_text_wins_over_the_case_of_the_entry() {
    let dictionary = load("ф+орзац\n").expect("load");
    assert_eq!(dictionary.apply("ФОРЗАЦ и форзац").0, "Ф+ОРЗАЦ и ф+орзац");
}

#[test]
fn punctuation_around_the_word_is_left_where_it_was() {
    let dictionary = load("з+амок\n").expect("load");
    assert_eq!(
        dictionary.apply("«замок», замок-то").0,
        "«з+амок», з+амок-то"
    );
}

#[test]
fn a_word_the_caller_already_marked_is_not_touched() {
    let dictionary = load("з+амок\n").expect("load");
    assert_eq!(dictionary.apply("зам+ок").0, "зам+ок");
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let dictionary = load("# names\n\n  ф+орзац  \n").expect("load");
    assert_eq!(dictionary.apply("форзац").0, "ф+орзац");
}

#[test]
fn a_malformed_entry_is_an_error_rather_than_a_surprise() {
    for bad in [
        "два слова\n",        // more than one word
        "hello\n",            // not Russian
        "форзац\n",           // marks nothing
        "ф+орзац\nфорз+ац\n", // the same key twice
    ] {
        assert!(
            matches!(load(bad), Err(StressError::Dictionary(_))),
            "expected a dictionary error for {bad:?}"
        );
    }
}

#[test]
fn nothing_moves_when_the_dictionary_is_empty() {
    let dictionary = Dictionary::default();
    assert!(dictionary.is_empty());
    let text = "Он запер замок и ушел.";
    assert_eq!(dictionary.apply(text), (text.to_owned(), 0));
}
