//! Tests for the sentence-type rules of the Russian model's pitch head.

use super::{Utterance, classify, classify_text, sentences, type_ids};

#[test]
fn a_plain_sentence_is_a_statement() {
    assert_eq!(classify("Сегодня тепло."), Utterance::Statement);
    assert_eq!(classify(""), Utterance::Statement);
}

#[test]
fn a_question_word_makes_a_wh_question() {
    assert_eq!(classify("Где мы сейчас?"), Utterance::WhQuestion);
    // A filler before it does not hide it.
    assert_eq!(classify("Ну и что теперь?"), Utterance::WhQuestion);
    // A stress mark does not either.
    assert_eq!(classify("Гд+е м+ы?"), Utterance::WhQuestion);
}

#[test]
fn a_question_without_one_is_a_general_question() {
    assert_eq!(classify("Ты придёшь?"), Utterance::GeneralQuestion);
}

#[test]
fn an_alternative_is_recognized_by_its_conjunction() {
    assert_eq!(classify("Чай или кофе?"), Utterance::Alternative);
    // Only as a whole word: this one merely contains the letters.
    assert_eq!(classify("Ты налил килиман?"), Utterance::GeneralQuestion);
}

#[test]
fn a_tag_needs_something_before_it() {
    assert_eq!(classify("Ты придёшь, правда?"), Utterance::Tag);
    assert_eq!(classify("Ты придёшь, не так ли?"), Utterance::Tag);
    // On its own it is just a question.
    assert_eq!(classify("Правда?"), Utterance::GeneralQuestion);
}

#[test]
fn an_exclamation_is_told_apart_from_a_question_that_ends_in_one() {
    assert_eq!(classify("Какая красота!"), Utterance::Exclamation);
    assert_eq!(classify("Ты серьёзно?!"), Utterance::GeneralQuestion);
}

#[test]
fn quotes_around_a_sentence_are_ignored() {
    assert_eq!(classify("«Ты придёшь?»"), Utterance::GeneralQuestion);
}

#[test]
fn a_text_splits_at_a_full_stop_followed_by_space() {
    assert_eq!(sentences("Раз. Два.  Три"), vec!["Раз.", "Два.", "Три"]);
    // A full stop with nothing after it does not split.
    assert_eq!(sentences("Раз.Два"), vec!["Раз.Два"]);
}

#[test]
fn every_sentence_of_a_text_is_classified() {
    let types = classify_text("Тепло. Ты придёшь? Отлично!");
    assert_eq!(
        types,
        vec![
            Utterance::Statement,
            Utterance::GeneralQuestion,
            Utterance::Exclamation,
        ]
    );
}

#[test]
fn the_ids_run_one_per_character_after_the_opening_symbol() {
    let text = "Да. Нет?";
    let types = classify_text(text);
    // Two more than the text: the opening and closing symbols.
    let ids = type_ids(text, &types, text.chars().count() + 2);
    // Position zero is the opening symbol and carries the first type.
    assert_eq!(ids[0], Utterance::Statement.id());
    // "Да." is three characters plus the space the split ate.
    assert_eq!(&ids[1..5], &[0, 0, 0, 0]);
    assert_eq!(
        &ids[5..9],
        &[
            Utterance::GeneralQuestion.id(),
            Utterance::GeneralQuestion.id(),
            Utterance::GeneralQuestion.id(),
            Utterance::GeneralQuestion.id(),
        ]
    );
    // The tail past the text falls back to the first type.
    assert_eq!(*ids.last().expect("a last id"), Utterance::Statement.id());
}
