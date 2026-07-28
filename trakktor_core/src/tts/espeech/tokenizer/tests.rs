use super::{close_reference_text, normalize};

#[test]
fn russian_text_passes_through_character_for_character() {
    let text = "Речь в этой книге пойдёт о хоббитах.";
    assert_eq!(normalize(text), text);
}

#[test]
fn stress_marks_are_ordinary_characters() {
    assert_eq!(normalize("прив+ет, з+амок"), "прив+ет, з+амок");
}

#[test]
fn substitutions_follow_the_reference() {
    assert_eq!(
        normalize("он сказал\u{201c}да\u{201d}; потом \u{2018}нет\u{2019}"),
        "он сказал\"да\", потом 'нет'"
    );
}

#[test]
fn a_latin_word_after_punctuation_gains_a_space() {
    // The reference's segmenter separates a run of Latin letters from whatever
    // is glued to its left, unless that is a space or a quote.
    assert_eq!(normalize("текст,ABC"), "текст, ABC");
    assert_eq!(normalize("текст ABC"), "текст ABC");
    assert_eq!(normalize("\"ABC"), "\"ABC");
    assert_eq!(normalize("ABC"), "ABC");
}

#[test]
fn single_latin_letters_and_symbols_do_not_gain_one() {
    assert_eq!(normalize("текст,a"), "текст,a");
    assert_eq!(normalize("текст-текст"), "текст-текст");
}

#[test]
fn a_decimal_counts_as_one_run() {
    assert_eq!(normalize("цена,1.5%"), "цена, 1.5%");
}

#[test]
fn reference_text_is_closed_with_a_period_and_two_spaces() {
    // Two, not one: the reference appends ". " and then, seeing a single-byte
    // last character, appends another space.
    assert_eq!(close_reference_text("вот так"), "вот так.  ");
    assert_eq!(close_reference_text("вот так."), "вот так.  ");
    assert_eq!(close_reference_text("вот так. "), "вот так.  ");
}
