//! The text layer is where the reference's edge cases live, so it is tested
//! on its own — no model, no tensors.

use super::*;

/// The operation promises to give back the text it was handed, so both the
/// word split and the sentence split have to be lossless.
#[test]
fn splitting_never_loses_a_character() {
    for text in [
        "Он запер замок и ушел до вечера.",
        "«Кот», (пёс) и — птица; всё: хорошо?",
        "Первая строка.\nВторая строка.\n\nТретий абзац.",
        "Кот.  Пёс.\tПтица.",
        "Он ушёл… Она осталась.",
        "   ",
        "",
        "что-то кому-то из-за чего-то",
        "В 2026 году 5 котов съели 10 кг корма.",
    ] {
        let joined: String =
            tokenize(text).iter().flat_map(|t| t.raw.clone()).collect();
        assert_eq!(joined, text, "word split of {text:?}");
        assert_eq!(segments(text).concat(), text, "sentence split of {text:?}");
    }
}

#[test]
fn sentences_end_after_the_space_that_follows_the_full_stop() {
    assert_eq!(segments("Кот. Пёс."), ["Кот. ", "Пёс."]);
    assert_eq!(segments("Кот.  Пёс."), ["Кот.  ", "Пёс."]);
    assert_eq!(segments("Он ушёл… Она тут."), ["Он ушёл… ", "Она тут."]);
    // A full stop with nothing after it does not start a new sentence.
    assert_eq!(segments("Кот.Пёс"), ["Кот.Пёс"]);
    // Every line break does.
    assert_eq!(segments("а\nб"), ["а\n", "б"]);
    assert_eq!(segments("а.\n\nб"), ["а.\n\n", "б"]);
}

#[test]
fn words_are_split_off_punctuation_and_hyphens() {
    let tokens = tokenize("что-то, кот!");
    let words: Vec<&str> = tokens
        .iter()
        .filter(|t| t.process)
        .map(|t| t.clean.as_str())
        .collect();
    // The trailing `-то` is deliberately left unmarked by the reference.
    assert_eq!(words, ["что", "кот"]);
    let raw: Vec<String> =
        tokens.iter().map(|t| t.raw.iter().collect()).collect();
    assert_eq!(raw, ["что-", "то", ", ", "кот", "!"]);
}

#[test]
fn a_word_is_cleaned_to_its_russian_letters() {
    let tokens = tokenize("«Кот2»");
    let token = tokens.iter().find(|t| t.process).unwrap();
    assert_eq!(token.clean, "кот");
    assert_eq!(token.vowels(), [2]); // the `о` of `«Кот2»`
}

#[test]
fn the_two_marker_forms_convert_into_each_other() {
    let plus = "Мен+я зов+ут.";
    let acute = render(plus, Marker::Acute);
    assert_eq!(acute, "Меня\u{301} зову\u{301}т.");
    assert_eq!(acute_to_plus(&acute), plus);
    assert_eq!(render(plus, Marker::Plus), plus);
}

#[test]
fn the_acute_form_leaves_yo_unaccented() {
    // `ё` is stressed by definition, and no Russian dictionary accents it.
    // Reading the text back restores the mark from the letter itself.
    assert_eq!(render("Л+ёва", Marker::Acute), "Лёва");
    assert_eq!(render("В+ЁДРА", Marker::Acute), "ВЁДРА");
}

#[test]
fn a_stray_mark_survives_the_round_trip() {
    // Neither form invents or drops anything it does not understand.
    assert_eq!(render("2+2", Marker::Acute), "2+2");
    assert_eq!(acute_to_plus("2\u{301}"), "2\u{301}");
}

#[test]
fn a_written_yo_and_a_written_mark_are_visible_to_the_rules() {
    let tokens = tokenize("вс+ё");
    let token = &tokens[0];
    assert!(token.has_stress());
    assert!(token.has_yo());
    assert_eq!(token.clean, "всё");
    assert_eq!(fold_yo(&token.clean), "все");
}
