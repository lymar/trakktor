use super::{Synthesizer, fit};

#[test]
fn a_supported_language_is_accepted_however_it_is_spelled() {
    for language in
        [None, Some("auto"), Some("russian"), Some("RU"), Some("rus")]
    {
        Synthesizer::check_language(language).expect("should accept");
    }
}

#[test]
fn another_language_is_refused_with_a_pointer_to_the_other_engine() {
    let error = Synthesizer::check_language(Some("english"))
        .expect_err("should refuse");
    let message = error.to_string();
    assert!(message.contains("Russian"), "{message}");
    assert!(message.contains("qwen3-tts"), "{message}");
}

#[test]
fn text_within_the_budget_is_left_whole() {
    assert_eq!(fit("Короткая фраза.", 200), vec!["Короткая фраза."]);
}

#[test]
fn a_long_sentence_is_cut_at_punctuation() {
    let text = "Речь в этой книге пойдёт главным образом о хоббитах, и её \
                страницы много расскажут читателю об их нраве.";
    // The budget is in bytes, and Cyrillic spends two of them per letter.
    let pieces = fit(text, 120);
    assert!(pieces.len() > 1, "{pieces:?}");
    assert!(
        pieces.iter().all(|piece| piece.len() <= 120),
        "{:?}",
        pieces.iter().map(String::len).collect::<Vec<_>>()
    );
    // Nothing is lost: the pieces hold the same words in the same order.
    let joined: String = pieces.join(" ");
    let normalize =
        |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert_eq!(normalize(&joined), normalize(text));
    // The first cut lands after the comma, not mid-word.
    assert!(pieces[0].ends_with(','), "{:?}", pieces[0]);
}

#[test]
fn a_run_without_punctuation_is_cut_at_a_space() {
    let text = "слово ".repeat(40);
    let pieces = fit(&text, 60);
    assert!(pieces.len() > 1);
    assert!(pieces.iter().all(|piece| piece.len() <= 60));
    assert!(pieces.iter().all(|piece| !piece.starts_with(' ')));
}

#[test]
fn a_single_word_over_the_budget_is_still_returned() {
    let word = "а".repeat(100);
    let pieces = fit(&word, 20);
    assert!(!pieces.is_empty());
    assert_eq!(pieces.concat(), word);
    // Cuts fall on character boundaries, so every piece is valid text.
    assert!(pieces.iter().all(|piece| !piece.is_empty()));
}
